"""LocalRunner — single-node DAG execution with checkpoint/resume (ADR-0003, 0010).

Run state is files-on-Storage, no database: a run manifest, per-Stage checkpoint
files, and a write-ahead log, all under ``<storage_uri>/_latence/runs/<run_id>/``.
Checkpoints are written with atomic temp-write→rename (WAL discipline), so a run
killed mid-Stage leaves either the previous checkpoint or the complete new one —
never a torn write. Re-running:

* a Stage whose checkpoint exists is skipped (resume);
* a fully-completed pipeline is a no-op.

Stages execute in topological order over their ``depends_on`` edges; for the S1
spine (Source→Parse→Export) the order is linear, but the DAG machinery is real.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, TypeVar

from latence_core.capability import (
    Chunker,
    ContentScreener,
    EntityExtractor,
    Export,
    FusedEntityRelationExtractor,
    IntakeScreener,
    Parser,
    PIIDetector,
    ProviderRegistry,
    RelationExtractor,
    ScreeningOutcome,
    Source,
)
from latence_core.contracts import (
    ChunkRecord,
    Disposition,
    DocumentRecord,
    EntityMention,
    ParserInput,
    QuarantineRecord,
    Record,
    RedactionRecord,
    RelationMention,
    ScreeningCheckpoint,
    ScreeningFinding,
)
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.quality import (
    ChunkQuality,
    CompletenessCheck,
    EntityQuality,
    ParseQuality,
    QualityReport,
    RedactionQuality,
    RelationQuality,
    ScreeningQuality,
    StageMetrics,
)
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable

# A Stage's output is one of the inter-Stage carriers the DAG threads between
# Stages: Source→ParserInput; Parse→DocumentRecord; Chunk/Content-Screening→
# ChunkRecord; Entity-Extraction→EntityMention; Relation-Extraction→RelationMention.
# A *fused* Entity+Relation Stage (ADR-0013) emits BOTH EntityMention and
# RelationMention in one output list (mixed carrier), routed to both downstream
# consumers without a separate Entity-Extraction Stage. The Runner keeps outputs in
# one per-Stage dict and lets each Stage's capability decide how to (de)serialize its
# checkpoint. Redaction→RedactionRecord (one PII-handled variant per document).
# Screening ALSO produces QuarantineRecords, but those never flow downstream — they
# are persisted to a sidecar and reported, not threaded through the DAG.
StageOutput = (
    ParserInput
    | DocumentRecord
    | ChunkRecord
    | EntityMention
    | RelationMention
    | RedactionRecord
)
_T = TypeVar("_T", bound=StageOutput)

# WAL event kinds. The log is append-only and never read for control flow in S1
# (checkpoint presence is authoritative); it exists for audit and crash forensics.
_WAL_STAGE_BEGIN = "stage_begin"
_WAL_STAGE_COMMIT = "stage_commit"


def topological_order(stages: list[Stage]) -> list[Stage]:
    """Kahn's algorithm over ``depends_on`` edges. Raises on a cycle."""
    by_name = {s.name: s for s in stages}
    indegree = {s.name: len(s.depends_on) for s in stages}
    # Deterministic: process ready nodes in declared order.
    ready = [s.name for s in stages if indegree[s.name] == 0]
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for s in stages:
            if name in s.depends_on:
                indegree[s.name] -= 1
                if indegree[s.name] == 0:
                    ready.append(s.name)
    if len(order) != len(stages):
        remaining = sorted(set(by_name) - set(order))
        msg = f"Pipeline has a dependency cycle among stages: {remaining}"
        raise ValueError(msg)
    return [by_name[n] for n in order]


class LocalRunner:
    """Executes a Pipeline on the local node with resumable file-based state."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self._registry = registry or ProviderRegistry()

    # -- state layout ------------------------------------------------------

    def _run_dir(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"

    def _checkpoint_uri(self, pipeline: Pipeline, run_id: str, stage_name: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/checkpoints/{stage_name}.jsonl"

    def _manifest_uri(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/manifest.json"

    def _wal_uri(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/wal.log"

    def _report_uri(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/quality-report.json"

    def _export_dir(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/export"

    def _quarantine_uri(self, pipeline: Pipeline, run_id: str, stage_name: str) -> str:
        # Quarantined records are retained and inspectable (CONTEXT ``Quarantine``):
        # excluded from the DAG and Export, persisted here per Screening Stage.
        return f"{self._run_dir(pipeline, run_id)}/quarantine/{stage_name}.jsonl"

    def _findings_uri(self, pipeline: Pipeline, run_id: str, stage_name: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/screening/{stage_name}.jsonl"

    # -- WAL ---------------------------------------------------------------

    def _wal_append(self, storage: Storage, uri: str, event: str, stage: str) -> None:
        line = json.dumps({"ts": time.time(), "event": event, "stage": stage}, sort_keys=True)
        existing = storage.read_text(uri) if storage.exists(uri) else ""
        storage.write_atomic(uri, (existing + line + "\n").encode("utf-8"))

    # -- checkpoint IO -----------------------------------------------------

    @classmethod
    def _output_type(cls, stage: Stage, pipeline: Pipeline) -> type[StageOutput] | type[Record]:
        """The carrier a Stage checkpoints (the records it feeds downstream).

        Source and Intake Screening thread raw ParserInputs; Parse threads parsed
        DocumentRecords; Chunk and Content Screening thread ChunkRecords. (Screening
        never threads its QuarantineRecords — those go to a sidecar, not the DAG.)
        Export is a pass-through sink: its carrier is whatever its upstream feeds —
        DocumentRecords on the S1/S2 spine, ChunkRecords once a Chunk Stage sits
        upstream — so its checkpoint re-reads with the right type on resume.
        """
        if stage.capability in (CapabilityKind.SOURCE, CapabilityKind.INTAKE_SCREENING):
            return ParserInput
        if stage.capability in (CapabilityKind.CHUNK, CapabilityKind.CONTENT_SCREENING):
            return ChunkRecord
        if stage.capability is CapabilityKind.ENTITY_EXTRACTION:
            return EntityMention
        if stage.capability is CapabilityKind.RELATION_EXTRACTION:
            return RelationMention
        if stage.capability is CapabilityKind.REDACTION:
            return RedactionRecord
        if stage.capability is CapabilityKind.FUSED_ENTITY_RELATION:
            # A fused Stage checkpoints a MIXED carrier (EntityMention +
            # RelationMention). Signalled by Record; read via _read_mixed_checkpoint.
            return Record
        if stage.capability is CapabilityKind.EXPORT:
            return cls._export_carrier(stage, pipeline)
        return DocumentRecord

    @classmethod
    def _export_carrier(cls, stage: Stage, pipeline: Pipeline) -> type[StageOutput] | type[Record]:
        """Resolve an Export's pass-through carrier from its nearest producing ancestor.

        Walks up ``depends_on`` to the closest Stage that actually *produces* the
        record kind Export re-emits: a Redaction ancestor ⇒ RedactionRecord, a
        Relation-Extraction ancestor ⇒ RelationMention, an Entity-Extraction ancestor
        ⇒ EntityMention, a Chunk / Content-Screening ancestor ⇒ ChunkRecord, otherwise
        a Parse (or Intake-Screening/Source that must feed a Parse) ancestor ⇒
        DocumentRecord. A *fused* Entity+Relation ancestor produces a MIXED carrier
        (both mentions and relations), so its checkpoint is read polymorphically
        (:meth:`_read_mixed_checkpoint`), signalled here by returning :class:`Record`.
        The frontier is walked breadth-first in ``depends_on`` order, so the nearest
        producing ancestor wins.
        """
        by_name = {s.name: s for s in pipeline.stages}
        seen: set[str] = set()
        frontier = list(stage.depends_on)
        while frontier:
            name = frontier.pop(0)
            if name in seen:
                continue
            seen.add(name)
            parent = by_name.get(name)
            if parent is None:  # pragma: no cover - validated DAG
                continue
            if parent.capability is CapabilityKind.FUSED_ENTITY_RELATION:
                return Record
            if parent.capability is CapabilityKind.REDACTION:
                return RedactionRecord
            if parent.capability is CapabilityKind.RELATION_EXTRACTION:
                return RelationMention
            if parent.capability is CapabilityKind.ENTITY_EXTRACTION:
                return EntityMention
            if parent.capability in (CapabilityKind.CHUNK, CapabilityKind.CONTENT_SCREENING):
                return ChunkRecord
            if parent.capability is CapabilityKind.PARSE:
                return DocumentRecord
            frontier.extend(parent.depends_on)
        return DocumentRecord

    def _write_checkpoint(
        self, storage: Storage, uri: str, records: list[StageOutput]
    ) -> None:
        # model_dump_json round-trips ``bytes`` content losslessly (used for
        # ParserInput's raw content); a bare dict + json.dumps would not.
        payload = "\n".join(r.model_dump_json() for r in records)
        storage.write_atomic(uri, (payload + ("\n" if records else "")).encode("utf-8"))

    def _read_checkpoint(
        self, storage: Storage, uri: str, record_type: type[StageOutput] | type[Record]
    ) -> list[StageOutput]:
        text = storage.read_text(uri)
        if record_type is Record:
            # A fused Stage's mixed carrier (or an Export re-emitting one): each line
            # is an EntityMention or a RelationMention — discriminate per line.
            return self._read_mixed_lines(text)
        records: list[StageOutput] = []
        for raw in text.splitlines():
            if raw.strip():
                records.append(record_type.model_validate_json(raw))  # type: ignore[arg-type]
        return records

    @staticmethod
    def _read_mixed_lines(text: str) -> list[StageOutput]:
        """Read a mixed EntityMention/RelationMention checkpoint (a fused Stage output).

        Each line is discriminated by the presence of the ``head_mention_id`` *key*
        in the parsed JSON object — a structural signal unique to the relation
        contract — never a substring match against the raw line. A substring test
        would misclassify an EntityMention whose ``text``/``label`` value is literally
        ``head_mention_id`` (it serialises to a line containing ``"head_mention_id"``),
        and the subsequent RelationMention validation would crash the resumed run.
        Both records are inter-Stage Records; keeping them in one checkpoint is what
        lets a single fused Stage feed both downstream consumers (S5: one pass, no
        double-extract).
        """
        records: list[StageOutput] = []
        for raw in text.splitlines():
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict) and "head_mention_id" in obj:
                records.append(RelationMention.model_validate(obj))
            else:
                records.append(EntityMention.model_validate(obj))
        return records

    # -- execution ---------------------------------------------------------

    def run(self, pipeline: Pipeline, run_id: str = "run-0001") -> QualityReport:
        storage = Storage.for_uri(pipeline.storage_uri)
        run_dir = self._run_dir(pipeline, run_id)
        storage.makedirs(run_dir)

        manifest_uri = self._manifest_uri(pipeline, run_id)
        if not storage.exists(manifest_uri):
            storage.write_atomic(
                manifest_uri,
                json.dumps(
                    {"pipeline": pipeline.model_dump(mode="json"), "run_id": run_id},
                    sort_keys=True,
                ).encode("utf-8"),
            )

        ordered = topological_order(list(pipeline.stages))
        wal_uri = self._wal_uri(pipeline, run_id)

        stage_metrics: list[StageMetrics] = []
        # Per-Stage outputs keyed by Stage name. A Stage's INPUT is the gather of
        # ALL its ``depends_on`` parents' outputs (multi-input, so a converging
        # DAG {a,b}->c sees both a's and b's records — never a single overwritten
        # ``current`` var, which corrupted data on resume). A downstream Stage
        # always reads its parents' *current* outputs from this dict, so a parent
        # that had to re-run (its checkpoint absent) feeds the child fresh data.
        outputs: dict[str, list[StageOutput]] = {}

        for stage in ordered:
            ckpt_uri = self._checkpoint_uri(pipeline, run_id, stage.name)
            incoming = self._gather_inputs(stage, outputs)

            if storage.exists(ckpt_uri):
                # Resume: this Stage already completed. Reload ITS OWN output into
                # the dict and skip re-running. A downstream Stage still reads this
                # value as one of its parents' outputs.
                own_output = self._read_checkpoint(
                    storage, ckpt_uri, self._output_type(stage, pipeline)
                )
                outputs[stage.name] = own_output
                stage_metrics.append(
                    StageMetrics(
                        name=stage.name,
                        capability=stage.capability.value,
                        provider=stage.provider,
                        records_in=len(incoming),
                        records_out=len(own_output),
                        duration_seconds=0.0,
                        from_checkpoint=True,
                    )
                )
                continue

            self._wal_append(storage, wal_uri, _WAL_STAGE_BEGIN, stage.name)
            started = time.perf_counter()
            own_output = self._execute_stage(stage, storage, pipeline, run_id, incoming)
            duration = time.perf_counter() - started

            outputs[stage.name] = own_output
            self._write_checkpoint(storage, ckpt_uri, list(own_output))
            self._wal_append(storage, wal_uri, _WAL_STAGE_COMMIT, stage.name)

            stage_metrics.append(
                StageMetrics(
                    name=stage.name,
                    capability=stage.capability.value,
                    provider=stage.provider,
                    records_in=len(incoming),
                    records_out=len(own_output),
                    duration_seconds=round(duration, 6),
                )
            )

        final_records = self._final_records(ordered, outputs)
        report = self._build_report(
            pipeline, run_id, storage, ordered, stage_metrics, outputs, final_records
        )
        storage.write_atomic(self._report_uri(pipeline, run_id), report.to_json().encode("utf-8"))
        return report

    def _execute_stage(
        self,
        stage: Stage,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        incoming: list[StageOutput],
    ) -> list[StageOutput]:
        provider_cls = self._registry.load(stage.provider)
        provider = provider_cls(stage.config)

        if stage.capability is CapabilityKind.SOURCE:
            if not isinstance(provider, Source):
                msg = f"Provider {stage.provider!r} does not satisfy the Source capability."
                raise TypeError(msg)
            # A Source is a root: it reads Storage, not upstream records.
            return list(provider.produce(storage))

        if stage.capability is CapabilityKind.INTAKE_SCREENING:
            if not isinstance(provider, IntakeScreener):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the IntakeScreener capability."
                )
                raise TypeError(msg)
            raw_inputs = self._as_type(stage, incoming, ParserInput)
            intake_outcome = provider.screen_intake(raw_inputs)
            self._persist_screening_sidecar(storage, pipeline, run_id, stage, intake_outcome)
            # Only the passed raw inputs proceed to Parse; the quarantined ones are
            # removed from the DAG (persisted for audit above), never reaching Parse.
            return list(intake_outcome.passed)

        if stage.capability is CapabilityKind.PARSE:
            if not isinstance(provider, Parser):
                msg = f"Provider {stage.provider!r} does not satisfy the Parser capability."
                raise TypeError(msg)
            parser_inputs = self._as_type(stage, incoming, ParserInput)
            return list(provider.parse(parser_inputs))

        if stage.capability is CapabilityKind.CHUNK:
            if not isinstance(provider, Chunker):
                msg = f"Provider {stage.provider!r} does not satisfy the Chunker capability."
                raise TypeError(msg)
            doc_records = self._as_type(stage, incoming, DocumentRecord)
            return list(provider.chunk(doc_records))

        if stage.capability is CapabilityKind.CONTENT_SCREENING:
            if not isinstance(provider, ContentScreener):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the ContentScreener capability."
                )
                raise TypeError(msg)
            chunk_records = self._as_type(stage, incoming, ChunkRecord)
            content_outcome = provider.screen_content(chunk_records)
            self._persist_screening_sidecar(storage, pipeline, run_id, stage, content_outcome)
            # Passed chunks (some now carrying propagated RiskMarkers) proceed to
            # Export; quarantined chunks are removed and persisted for audit.
            return list(content_outcome.passed)

        if stage.capability is CapabilityKind.ENTITY_EXTRACTION:
            if not isinstance(provider, EntityExtractor):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the EntityExtractor capability."
                )
                raise TypeError(msg)
            chunk_records = self._as_type(stage, incoming, ChunkRecord)
            return list(provider.extract(chunk_records))

        if stage.capability is CapabilityKind.RELATION_EXTRACTION:
            if not isinstance(provider, RelationExtractor):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the RelationExtractor capability."
                )
                raise TypeError(msg)
            # Relation Extraction is multi-input: it wires BOTH a Chunk (or
            # Content-Screening) parent for text/page-map context AND an
            # Entity-Extraction parent for the mentions it relates. Partition the
            # gathered inputs by carrier; a missing side is a wiring error.
            rel_chunks = [r for r in incoming if isinstance(r, ChunkRecord)]
            rel_mentions = [r for r in incoming if isinstance(r, EntityMention)]
            self._require_relation_inputs(stage, incoming, rel_chunks, rel_mentions)
            return list(provider.relate(rel_chunks, rel_mentions))

        if stage.capability is CapabilityKind.FUSED_ENTITY_RELATION:
            if not isinstance(provider, FusedEntityRelationExtractor):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the FusedEntityRelationExtractor capability."
                )
                raise TypeError(msg)
            # One pass over the chunks yields BOTH mentions and relations; the fused
            # Stage threads them together downstream (no separate Entity-Extraction
            # Stage — the S5 no-double-extract routing).
            fused_chunks = self._as_type(stage, incoming, ChunkRecord)
            fused = provider.extract_fused(fused_chunks)
            out: list[StageOutput] = [*fused.mentions, *fused.relations]
            return out

        if stage.capability is CapabilityKind.REDACTION:
            if not isinstance(provider, PIIDetector):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the PIIDetector capability."
                )
                raise TypeError(msg)
            # Redaction consumes the run's parsed DocumentRecords (the whole
            # document's assembled markdown + page map, at original-document offsets)
            # and emits one PII-handled RedactionRecord per document, keyed off
            # Classification sensitivity — threaded downstream to Export as the
            # corpus's PII variant. It depends on Parse (not Chunk): the chunk stream
            # is markup-stripped with remapped offsets, so scanning it would fabricate
            # a broken coordinate system for the spans and page resolution.
            redaction_docs = self._as_type(stage, incoming, DocumentRecord)
            return list(provider.redact(redaction_docs))

        if stage.capability is CapabilityKind.EXPORT:
            if not isinstance(provider, Export):
                msg = f"Provider {stage.provider!r} does not satisfy the Export capability."
                raise TypeError(msg)
            # Export materializes any inter-Stage Record: DocumentRecords (S1/S2
            # spine) or ChunkRecords (S3, when a Chunk Stage sits upstream). Only a
            # raw ParserInput (not a Record) is an invalid Export input.
            exportable = self._as_records(stage, incoming)
            out_dir = self._export_dir(pipeline, run_id)
            provider.export(exportable, storage, out_dir)
            # Export is a sink: it passes its input records through unchanged.
            return list(exportable)

        msg = f"Unknown capability {stage.capability!r} for stage {stage.name!r}."
        raise ValueError(msg)

    # -- Screening sidecars (quarantine + findings, off the DAG) ------------

    def _persist_screening_sidecar(
        self,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        stage: Stage,
        outcome: ScreeningOutcome[ParserInput] | ScreeningOutcome[ChunkRecord],
    ) -> None:
        """Persist a Screening Stage's quarantine records + findings for audit.

        Written atomically alongside the checkpoint so a resumed run re-reads them
        from Storage without re-running Screening (the Quality Report is rebuilt
        from these files uniformly on fresh and resumed runs). Quarantined records
        are retained and inspectable here (CONTEXT ``Quarantine``); they never enter
        the DAG or an Export.
        """
        q_uri = self._quarantine_uri(pipeline, run_id, stage.name)
        q_payload = "\n".join(r.model_dump_json() for r in outcome.quarantined)
        storage.write_atomic(
            q_uri, (q_payload + ("\n" if outcome.quarantined else "")).encode("utf-8")
        )
        f_uri = self._findings_uri(pipeline, run_id, stage.name)
        f_payload = "\n".join(f.model_dump_json() for f in outcome.findings)
        storage.write_atomic(
            f_uri, (f_payload + ("\n" if outcome.findings else "")).encode("utf-8")
        )

    def _read_quarantine(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[QuarantineRecord]:
        uri = self._quarantine_uri(pipeline, run_id, stage_name)
        if not storage.exists(uri):
            return []
        return [
            QuarantineRecord.model_validate_json(line)
            for line in storage.read_text(uri).splitlines()
            if line.strip()
        ]

    def _read_findings(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[ScreeningFinding]:
        uri = self._findings_uri(pipeline, run_id, stage_name)
        if not storage.exists(uri):
            return []
        return [
            ScreeningFinding.model_validate_json(line)
            for line in storage.read_text(uri).splitlines()
            if line.strip()
        ]

    @staticmethod
    def _gather_inputs(stage: Stage, outputs: dict[str, list[StageOutput]]) -> list[StageOutput]:
        """A Stage's input = the concatenation of ALL its parents' outputs.

        Parents are read in declared ``depends_on`` order for determinism. This
        is what makes a converging DAG ({a, b} -> c) correct and multi-input:
        c sees a's records followed by b's, from the dict — never a single
        overwritten variable, and always the parents' *current* outputs.
        """
        gathered: list[StageOutput] = []
        for parent in stage.depends_on:
            gathered.extend(outputs.get(parent, []))
        return gathered

    @staticmethod
    def _as_type(
        stage: Stage, records: list[StageOutput], expected: type[_T]
    ) -> list[_T]:
        """Narrow gathered inputs to the carrier a Stage's capability requires.

        A Parser must receive ParserInputs; an Export must receive DocumentRecords.
        A mismatch means the Pipeline wired an incompatible parent (e.g. a Source
        feeding an Export directly) — surface it as a clear typed error, not a
        downstream AttributeError.
        """
        wrong = [r for r in records if not isinstance(r, expected)]
        if wrong:
            got = sorted({type(r).__name__ for r in wrong})
            msg = (
                f"Stage {stage.name!r} ({stage.capability.value}) requires "
                f"{expected.__name__} inputs but received: {', '.join(got)}. "
                f"Check the Pipeline's depends_on wiring."
            )
            raise TypeError(msg)
        return [r for r in records if isinstance(r, expected)]

    @staticmethod
    def _require_relation_inputs(
        stage: Stage,
        incoming: list[StageOutput],
        chunks: list[ChunkRecord],
        mentions: list[EntityMention],
    ) -> None:
        """A Relation-Extraction Stage must be wired to BOTH chunks and mentions.

        It depends on an Entity-Extraction parent (the mentions it relates) and a
        Chunk / Content-Screening parent (the text + page map it needs for triggers
        and offset→page resolution). A missing side means the Pipeline forgot a
        ``depends_on`` edge — surface it as a clear typed error, not an empty result.
        An empty run (no documents ⇒ no chunks and no mentions) is legitimately empty.
        """
        if not incoming:
            return
        if not mentions:
            msg = (
                f"Stage {stage.name!r} (relation_extraction) received no EntityMentions. "
                f"A Relation-Extraction Stage must depend on an Entity-Extraction Stage "
                f"(the mentions it relates). Check the Pipeline's depends_on wiring."
            )
            raise TypeError(msg)
        if not chunks:
            msg = (
                f"Stage {stage.name!r} (relation_extraction) received no ChunkRecords. "
                f"A Relation-Extraction Stage must also depend on the Chunk (or "
                f"Content-Screening) Stage for the text + page map. Check depends_on."
            )
            raise TypeError(msg)

    @staticmethod
    def _as_records(
        stage: Stage, records: list[StageOutput]
    ) -> list[DocumentRecord | ChunkRecord | EntityMention | RelationMention | RedactionRecord]:
        """Narrow gathered inputs to the inter-Stage Records Export materializes.

        Export materializes DocumentRecords (S1/S2 spine), ChunkRecords (S3, when a
        Chunk Stage is upstream), EntityMentions (S4, an Entity-Extraction Stage
        upstream), RelationMentions (S5, a Relation-Extraction / fused Stage upstream),
        or RedactionRecords (S6, a Redaction Stage upstream — the PII-handled corpus
        variant). A fused Stage feeds a MIX of mentions and relations — both are
        Records and both export. A raw ParserInput (not a Record) means a Source was
        wired straight into Export with no Parse between. Surface it as a clear
        typed error.
        """
        exportable = (
            DocumentRecord,
            ChunkRecord,
            EntityMention,
            RelationMention,
            RedactionRecord,
        )
        wrong = [r for r in records if not isinstance(r, exportable)]
        if wrong:
            got = sorted({type(r).__name__ for r in wrong})
            msg = (
                f"Stage {stage.name!r} (export) requires Record inputs "
                f"(DocumentRecord/ChunkRecord/EntityMention/RelationMention/"
                f"RedactionRecord) but received: {', '.join(got)}. An Export needs a "
                f"Parse (and optionally a Chunk / Entity-Extraction / "
                f"Relation-Extraction / Redaction) upstream."
            )
            raise TypeError(msg)
        return [r for r in records if isinstance(r, exportable)]

    @staticmethod
    def _final_records(
        ordered: list[Stage], outputs: dict[str, list[StageOutput]]
    ) -> list[StageOutput]:
        """Records for the Quality Report: the outputs of the DAG's sink Stages.

        A sink is a Stage nothing depends on. Reporting on sinks (rather than a
        single threaded ``current``) keeps the report correct for a fan-out or
        converging DAG, where the last topological Stage is not the whole story.
        """
        depended_on = {p for s in ordered for p in s.depends_on}
        sinks = [s.name for s in ordered if s.name not in depended_on]
        final: list[StageOutput] = []
        for name in sinks:
            final.extend(outputs.get(name, []))
        return final

    def _build_report(
        self,
        pipeline: Pipeline,
        run_id: str,
        storage: Storage,
        ordered: list[Stage],
        stage_metrics: list[StageMetrics],
        outputs: dict[str, list[StageOutput]],
        final_records: Iterable[Record | StageOutput],
    ) -> QualityReport:
        records = list(final_records)
        completeness = CompletenessCheck(
            records_checked=len(records),
            # Contracts guarantee these are present; the check makes it auditable.
            records_with_provenance=sum(1 for r in records if r.provenance is not None),
            records_with_classification=sum(1 for r in records if r.classification is not None),
        )
        # Document count = distinct source documents seen across the run. Chunks
        # fan out from a document (many chunks share one document_id), so counting
        # distinct Provenance document ids is correct whether the sinks are
        # DocumentRecords or ChunkRecords.
        doc_ids = {r.provenance.document_id for r in records}
        # Parse/Chunk quality scan the Stage that produced those records (not just
        # the sinks), so a Source→Parse→Chunk→Export pipeline (whose sinks are
        # ChunkRecords) still reports Parse metrics from the Parse Stage's output.
        parsed = self._records_of_kind(pipeline, ordered, outputs, CapabilityKind.PARSE)
        chunked = self._records_of_kind(pipeline, ordered, outputs, CapabilityKind.CHUNK)
        # Content Screening attaches the risk markers, so the flagged count is read
        # from the FINAL corpus records (post-screening), not the raw Chunk output
        # (which has no markers yet). Falls back to the chunk output when there is
        # no Content-Screening Stage.
        post_content = self._records_of_kind(
            pipeline, ordered, outputs, CapabilityKind.CONTENT_SCREENING
        )
        flagged_source = post_content if post_content else chunked
        # Mentions come from an Entity-Extraction Stage OR a fused Stage (which emits
        # mentions + relations together); relations from a Relation-Extraction Stage
        # OR that same fused Stage. The quality builders isinstance-filter each list to
        # its own carrier, so a fused Stage's mixed output is passed to both.
        mentions = self._records_of_kinds(
            pipeline,
            ordered,
            outputs,
            (CapabilityKind.ENTITY_EXTRACTION, CapabilityKind.FUSED_ENTITY_RELATION),
        )
        relations = self._records_of_kinds(
            pipeline,
            ordered,
            outputs,
            (CapabilityKind.RELATION_EXTRACTION, CapabilityKind.FUSED_ENTITY_RELATION),
        )
        redactions = self._records_of_kind(pipeline, ordered, outputs, CapabilityKind.REDACTION)
        return QualityReport(
            pipeline_name=pipeline.name,
            run_id=run_id,
            document_count=len(doc_ids),
            total_duration_seconds=round(sum(m.duration_seconds for m in stage_metrics), 6),
            stages=stage_metrics,
            completeness=completeness,
            contracts_complete=completeness.passed,
            parse=self._parse_quality(pipeline, parsed),
            chunk=self._chunk_quality(pipeline, chunked, flagged_source),
            screening=self._screening_quality(pipeline, run_id, storage, ordered),
            entities=self._entity_quality(pipeline, mentions),
            relations=self._relation_quality(pipeline, relations),
            redaction=self._redaction_quality(pipeline, redactions),
        )

    @staticmethod
    def _records_of_kind(
        pipeline: Pipeline,
        ordered: list[Stage],
        outputs: dict[str, list[StageOutput]],
        kind: CapabilityKind,
    ) -> list[StageOutput]:
        """Flatten the outputs of every Stage fulfilling ``kind`` (order-stable)."""
        result: list[StageOutput] = []
        for stage in ordered:
            if stage.capability is kind:
                result.extend(outputs.get(stage.name, []))
        return result

    @staticmethod
    def _records_of_kinds(
        pipeline: Pipeline,
        ordered: list[Stage],
        outputs: dict[str, list[StageOutput]],
        kinds: tuple[CapabilityKind, ...],
    ) -> list[StageOutput]:
        """Flatten the outputs of every Stage fulfilling any of ``kinds`` (order-stable).

        Used where a record kind can be produced by more than one capability — a
        mention comes from an Entity-Extraction OR a fused Stage; a relation from a
        Relation-Extraction OR that same fused Stage — and the caller then filters the
        gathered output by carrier type.
        """
        result: list[StageOutput] = []
        for stage in ordered:
            if stage.capability in kinds:
                result.extend(outputs.get(stage.name, []))
        return result

    def _screening_quality(
        self, pipeline: Pipeline, run_id: str, storage: Storage, ordered: list[Stage]
    ) -> ScreeningQuality | None:
        """Aggregate Screening findings across both checkpoints for the report.

        Reads each Screening Stage's persisted findings + quarantine sidecars (so
        the numbers are identical on fresh and resumed runs) and rolls them into
        the Quality Report: every Quarantine disposition and reason, and every
        propagated flag, are auditable there (S3 AC). Returns ``None`` when the
        Pipeline has no Screening Stage.
        """
        screen_kinds = (CapabilityKind.INTAKE_SCREENING, CapabilityKind.CONTENT_SCREENING)
        if not any(s.capability in screen_kinds for s in pipeline.stages):
            return None
        quality = ScreeningQuality()
        for stage in ordered:
            if stage.capability not in screen_kinds:
                continue
            findings = self._read_findings(storage, pipeline, run_id, stage.name)
            quality.findings.extend(findings)
            for finding in findings:
                if finding.checkpoint is ScreeningCheckpoint.INTAKE:
                    quality.intake_screened += 1
                else:
                    quality.content_screened += 1
                if finding.verdict.value == "quarantine":
                    quality.quarantined += 1
                elif finding.verdict.value == "flag":
                    quality.flagged += 1
                else:
                    quality.passed += 1
        return quality

    @staticmethod
    def _chunk_quality(
        pipeline: Pipeline,
        records: list[StageOutput],
        flagged_source: list[StageOutput],
    ) -> ChunkQuality | None:
        """Aggregate Chunk-Stage quality: chunk counts, tokens, offset integrity.

        Only computed when the Pipeline has a Chunk Stage. ``records`` is the Chunk
        Stage's output (the canonical chunk set) from which counts, tokens, and the
        offset-preservation invariant are measured: every :class:`ChunkRecord`'s
        Provenance ``char_start``/``char_end`` must be a valid span and its
        ``page_start``/``page_end`` populated — the S3 round-trip invariant, so
        ``offset_preserving`` is True only if every chunk carried a resolvable span.
        ``flagged_source`` is the POST-Content-Screening corpus (markers are
        attached there, not at Chunk), so ``chunks_flagged`` reflects the corpus.
        """
        if not any(s.capability is CapabilityKind.CHUNK for s in pipeline.stages):
            return None
        quality = ChunkQuality()
        for record in records:
            if not isinstance(record, ChunkRecord):
                continue
            quality.chunks_total += 1
            quality.tokens_total += record.token_count
            prov = record.provenance
            has_span = (
                prov.char_start is not None
                and prov.char_end is not None
                and prov.char_end >= prov.char_start
            )
            has_pages = prov.page_start is not None and prov.page_end is not None
            if has_span and has_pages:
                quality.chunks_offset_preserving += 1
        quality.chunks_flagged = sum(
            1 for r in flagged_source if isinstance(r, ChunkRecord) and r.flagged
        )
        return quality

    @staticmethod
    def _entity_quality(
        pipeline: Pipeline, records: list[StageOutput]
    ) -> EntityQuality | None:
        """Aggregate Entity-Extraction quality: per-type counts + confidence dist.

        Only computed when the Pipeline has an Entity-Extraction Stage. ``records``
        is that Stage's output (the mention set) from which per-label counts, the
        confidence histogram, and the offset-resolution invariant are measured:
        every :class:`~latence_core.contracts.EntityMention`'s Provenance must carry
        a valid char span AND a page span — the S4 round-trip (mention → offset →
        original page) — so ``offset_resolving`` is True only when every mention
        carried a resolvable span. The confidence distribution + per-type counts are
        the S4 acceptance criterion (they "land in the Quality Report").
        """
        entity_kinds = (
            CapabilityKind.ENTITY_EXTRACTION,
            CapabilityKind.FUSED_ENTITY_RELATION,
        )
        if not any(s.capability in entity_kinds for s in pipeline.stages):
            return None
        quality = EntityQuality()
        chunk_ids_with_mentions: set[str] = set()
        for record in records:
            if not isinstance(record, EntityMention):
                continue
            quality.mentions_total += 1
            quality.per_type[record.label] = quality.per_type.get(record.label, 0) + 1
            quality.confidence.add(record.confidence)
            chunk_ids_with_mentions.add(record.chunk_record_id)
            prov = record.provenance
            has_span = (
                prov.char_start is not None
                and prov.char_end is not None
                and prov.char_end >= prov.char_start
            )
            has_pages = prov.page_start is not None and prov.page_end is not None
            if has_span and has_pages:
                quality.mentions_offset_resolving += 1
        quality.chunks_with_mentions = len(chunk_ids_with_mentions)
        # Sort per_type for a stable, deterministic report.
        quality.per_type = dict(sorted(quality.per_type.items()))
        return quality

    @staticmethod
    def _relation_quality(
        pipeline: Pipeline, records: list[StageOutput]
    ) -> RelationQuality | None:
        """Aggregate Relation-Extraction quality: per-type counts + confidence dist (S5).

        Only computed when the Pipeline has a Relation-Extraction (or fused) Stage.
        ``records`` is the relation set from which per-label counts, the confidence
        histogram, and the offset-resolution invariant are measured: every
        :class:`~latence_core.contracts.RelationMention`'s Provenance must carry a
        valid char span AND a page span — the S5 round-trip (relation → offset →
        original page) — so ``offset_resolving`` is True only when every relation
        carried a resolvable span. The per-type counts + confidence distribution are
        the S5 acceptance criterion ("relation counts + confidence land in the
        Quality Report").
        """
        relation_kinds = (
            CapabilityKind.RELATION_EXTRACTION,
            CapabilityKind.FUSED_ENTITY_RELATION,
        )
        if not any(s.capability in relation_kinds for s in pipeline.stages):
            return None
        quality = RelationQuality()
        docs_with_relations: set[str] = set()
        for record in records:
            if not isinstance(record, RelationMention):
                continue
            quality.relations_total += 1
            quality.per_type[record.label] = quality.per_type.get(record.label, 0) + 1
            quality.confidence.add(record.confidence)
            docs_with_relations.add(record.document_record_id)
            prov = record.provenance
            has_span = (
                prov.char_start is not None
                and prov.char_end is not None
                and prov.char_end >= prov.char_start
            )
            has_pages = prov.page_start is not None and prov.page_end is not None
            if has_span and has_pages:
                quality.relations_offset_resolving += 1
        quality.documents_with_relations = len(docs_with_relations)
        quality.per_type = dict(sorted(quality.per_type.items()))
        return quality

    @staticmethod
    def _redaction_quality(
        pipeline: Pipeline, records: list[StageOutput]
    ) -> RedactionQuality | None:
        """Aggregate Redaction quality: per-PII-type + per-action detection counts (S6).

        Only computed when the Pipeline has a Redaction Stage. ``records`` is the set
        of one :class:`~latence_core.contracts.RedactionRecord` per document, from
        which the per-type and per-action span counts, the documents-redacted, and the
        documents-with-PII are measured — the S6 acceptance criterion ("PII detection
        counts by type land in the Quality Report"). Only counts and type/action names
        cross into the report; **no raw PII value** ever does (a PIISpan never stores
        the surface string, so this rollup cannot leak one).
        """
        if not any(s.capability is CapabilityKind.REDACTION for s in pipeline.stages):
            return None
        quality = RedactionQuality()
        for record in records:
            if not isinstance(record, RedactionRecord):
                continue
            quality.documents_redacted += 1
            if record.pii_spans:
                quality.documents_with_pii += 1
            for span in record.pii_spans:
                quality.pii_total += 1
                quality.per_type[span.pii_type] = quality.per_type.get(span.pii_type, 0) + 1
                action = span.action.value
                quality.per_action[action] = quality.per_action.get(action, 0) + 1
        quality.per_type = dict(sorted(quality.per_type.items()))
        quality.per_action = dict(sorted(quality.per_action.items()))
        return quality

    @staticmethod
    def _parse_quality(
        pipeline: Pipeline, records: list[StageOutput]
    ) -> ParseQuality | None:
        """Aggregate Parse-Stage quality: page-map drift + parse-error counts.

        Only computed when the Pipeline actually has a Parse Stage, so a
        Source→Export pipeline reports ``parse=None`` rather than a misleading
        all-zero section. For each parsed :class:`DocumentRecord` that carries a
        page map, the Runner builds a :class:`PageOffsetIndex` and audits it —
        resolving each page's boundary offsets — so the drift diagnostics the
        ported asset produces surface in the Quality Report (S2: "drift
        diagnostics in the Quality Report"). A clean parse resolves every offset
        exactly (``page_maps_clean``). Each parsed record's assembled-markdown
        byte size is summed into ``bytes_after_parse`` — the observable rollup of
        the Provenance 'size-after-parse'.
        """
        if not any(s.capability is CapabilityKind.PARSE for s in pipeline.stages):
            return None

        quality = ParseQuality()
        for record in records:
            if not isinstance(record, DocumentRecord):
                continue
            if record.disposition is Disposition.PARSE_ERROR:
                quality.parse_errors += 1
                continue
            quality.documents_parsed += 1
            # Provenance 'size-after-parse', aggregated: the assembled-markdown
            # byte size of each parsed record (mirrors the parser's per-record
            # ``size_after_parse`` helper, summed here into the observable report).
            quality.bytes_after_parse += len(record.content.encode("utf-8"))
            page_map = record.page_map
            if page_map is None:
                continue
            quality.page_map_present += 1
            quality.pages_total += len(page_map.pages)
            index = PageOffsetIndex(page_map)
            # Audit: resolve each page's start and last-char offset. With exact
            # boundaries every lookup is exact; any drift the map carries is
            # detected here and accumulated into the diagnostics.
            for span in page_map.pages:
                index.page_for_offset(span.char_start)
                index.page_for_offset(max(span.char_end - 1, span.char_start))
            diag = index.diagnostics
            quality.page_drift_resolutions += diag.resolutions
            quality.page_drift_recovered += diag.recovered
            quality.page_drift_max_chars = max(quality.page_drift_max_chars, diag.max_drift_chars)
        return quality
