"""QualityReportBuilder — the Quality Report half of the Runner, as its own unit (H-F1).

The Quality Report is a first-class run artifact (ADR-0008): per-Stage metrics, corpus-level
KG statistics, contract-completeness checks, and optional gold-set precision/recall — the
substantiation of the "AI-ready" claim. Assembling it used to live inside
:class:`~latence_core.runner.LocalRunner` as ~830 lines of ``_build_report`` + ``_*_quality``
methods, so a second Runner (the Airflow adapter) could only reuse it by reaching **past the
interface** into six private ``LocalRunner`` methods (``# noqa: SLF001``). The audit's THEME F
verdict: *"the interface is the reuse surface — if a second impl must reach past it, the shape is
wrong."*

This module lifts that half out as a cohesive unit that depends only on what it needs, not the
whole executor:

* the stage ``outputs`` mapping + a :class:`StageRecordReader` seam that yields a Stage's records
  (from the live dict or, once released for memory bounding — #69 — its on-disk checkpoint), and
* the run identity / :class:`~latence_core.pipeline.Pipeline` /
  :class:`~latence_core.storage.Storage` it reads, plus the relation fan-out-guard tally the
  executor captured while Stages ran.

:class:`LocalRunner` now *composes* a :class:`QualityReportBuilder` (its ``_build_report`` /
``_final_records`` are thin delegators), and the Airflow adapter constructs the **public**
:class:`QualityReportBuilder` directly against the engine's public
:meth:`~latence_core.runner.LocalRunner.stage_records` reader — so neither Runner reaches past the
interface. This is a **pure structural refactor: the Quality Report is byte-identical** to the
pre-extraction output; the report-building logic below moved verbatim, only its collaborators are
now injected instead of ``self``.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Protocol

from latence_core.contracts import (
    ChunkRecord,
    DisambiguationRecord,
    DisambiguationScope,
    Disposition,
    DocumentRecord,
    EntityMention,
    FeatureRecord,
    FeatureScope,
    GraphRecord,
    GraphScope,
    Record,
    RedactionRecord,
    RelationMention,
    ScreeningCheckpoint,
    ScreeningFinding,
)
from latence_core.errors import ConfigError
from latence_core.goldset import (
    ConfusionCounts,
    GoldSet,
    score_entities,
    score_relations,
)
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import CapabilityKind
from latence_core.quality import (
    ChunkQuality,
    CompletenessCheck,
    ContextEnrichmentQuality,
    DisambiguationQuality,
    DriftDiagnostics,
    EntityQuality,
    GoldSetEvaluation,
    GraphQuality,
    LabelScore,
    ParseQuality,
    ProfilingQuality,
    QualityReport,
    RedactionQuality,
    RelationQuality,
    ScreeningQuality,
    StageMetrics,
    StageScore,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from latence_core.pipeline import Pipeline, Stage
    from latence_core.runner import StageOutput
    from latence_core.storage import Storage


# -- co-occurrence fingerprint (relocated from runner.py, re-exported there) ----------------
#
# The Quality Report is a shareable artifact; a co-occurring entity pair drawn from a
# ``confidential`` source could carry PII in its raw surface. These helpers key each pair by a
# salted sha256 fingerprint instead of the raw ``left|right`` surface. They lived in runner.py
# before H-F1; the report code that uses them moved here, so they moved too — runner.py re-exports
# them for backwards compatibility (some tests import them from there).
_CO_OCCURRENCE_FINGERPRINT_LEN = 16
_DEFAULT_FINGERPRINT_SALT = "latence"


def _co_occurrence_fingerprint(left: str, right: str, salt: str) -> str:
    """A deterministic, salted fingerprint of a co-occurrence pair.

    The Quality Report is a shareable artifact; a co-occurring entity pair drawn from a
    ``confidential`` source could carry PII in its raw surface. Rather than emit
    ``"{left}|{right}"`` (the raw surfaces) into the report, we key the co-occurrence
    entry by ``"pair:" + <sha256 hex prefix>`` over ``"{salt}:{left}|{right}"`` — the
    same salted-digest construction S6 uses in
    :func:`~latence_core.redaction_policy.placeholder_for` (``salt`` resolved from the
    pipeline's Redaction Stage config, default ``"latence"``).

    This is a **fingerprint, not encryption**: entity surfaces are low-entropy, so a
    reader who can enumerate candidate names could recover a pair by dictionary attack.
    The salt makes that attack per-deployment (the attacker must know the deployment's
    salt) and defeats precomputed rainbow tables, matching S6's threat posture — it does
    **not** make the digest irreversible. It removes the raw surface from the shareable
    report; it does not make the report safe to publish to an untrusted audience. The
    raw pair remains only in the exported, sensitivity-stamped ``CorpusFeatures`` data
    artifact. Deterministic (byte-stable across processes, not the seed-salted builtin
    ``hash``) so a seeded run renders identically, and stable per pair so the recurrence
    signal survives.
    """
    digest = hashlib.sha256(f"{salt}:{left}|{right}".encode()).hexdigest()
    return f"pair:{digest[:_CO_OCCURRENCE_FINGERPRINT_LEN]}"


def _fingerprint_salt(pipeline: Pipeline) -> str:
    """Resolve the co-occurrence fingerprint salt from the Redaction Stage config.

    Uses the identical key + default as S6's Redaction Stage
    (``config["salt"]``, default ``"latence"``) so a deployment that sets a custom salt
    gets the same salt applied to both PII surrogates and co-occurrence fingerprints. If
    the pipeline has no Redaction Stage, the default applies.
    """
    for stage in pipeline.stages:
        if stage.capability is CapabilityKind.REDACTION:
            return str(stage.config.get("salt", _DEFAULT_FINGERPRINT_SALT))
    return _DEFAULT_FINGERPRINT_SALT


class StageRecordReader(Protocol):
    """The run-state read seam the report builder needs from a Runner.

    Assembling the Quality Report reads two kinds of persisted run state, both of which are the
    Runner's file-layout concern, not the builder's:

    * :meth:`stage_records` — a Stage's output records. A Runner threads Stage outputs through a
      per-Stage ``outputs`` dict, releasing a doc-level Stage's entry once its last consumer ran
      (#69) — after which the record set is read back from the on-disk checkpoint, one batch at a
      time, to bound corpus-level memory. The builder does not care which path a Stage's records
      come from; it only needs them yielded in byte-identical order.
    * :meth:`read_findings` — a Screening Stage's persisted findings sidecar (written atomically
      beside the checkpoint), so the report's Screening rollup is identical on fresh and resumed
      runs.

    :class:`~latence_core.runner.LocalRunner` satisfies this Protocol via its public
    :meth:`~latence_core.runner.LocalRunner.stage_records` and
    :meth:`~latence_core.runner.LocalRunner.read_findings`, so the builder composes with the engine
    without naming its concrete type — and a second Runner reuses the builder the same way, through
    the public interface, never a private reach-through (H-F1).
    """

    def stage_records(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        """Yield one Stage's output — from the live dict if retained, else its checkpoint (#69)."""
        ...

    def read_findings(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[ScreeningFinding]:
        """Read a Screening Stage's persisted findings sidecar (empty if none written)."""
        ...


class QualityReportBuilder:
    """Assembles a run's :class:`~latence_core.quality.QualityReport` from its Stage outputs.

    Depends on a :class:`StageRecordReader` (to read each Stage's records back from the live
    ``outputs`` dict or its checkpoint) plus the relation fan-out-guard tally the executor
    captured — **not** the whole executor. Both Runners construct it against their engine's public
    reader and call :meth:`build`; the report is byte-identical to the pre-H-F1 in-runner build.

    The relation fan-out-guard tally is passed as a live mapping (not a snapshot): the local
    Runner mutates ``LocalRunner._fanout_by_stage`` while Stages run and reads it at report time —
    the builder reads the same mapping through :meth:`_fanout_total`. A Runner that captures no
    fan-out (the Airflow in-process path resets the engine's tally per Stage) passes an empty
    mapping, so every Stage contributes ``(0, 0, 0, 0)`` — the honest default.
    """

    def __init__(
        self,
        reader: StageRecordReader,
        *,
        fanout_by_stage: Mapping[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._reader = reader
        self._fanout_by_stage: Mapping[str, tuple[int, int, int, int]] = (
            fanout_by_stage if fanout_by_stage is not None else {}
        )

    # -- public surface ----------------------------------------------------

    def final_records(
        self,
        ordered: list[Stage],
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> list[StageOutput]:
        """Records for the Quality Report: the outputs of the DAG's sink Stages.

        A sink is a Stage nothing depends on. Reporting on sinks (rather than a
        single threaded ``current``) keeps the report correct for a fan-out or
        converging DAG, where the last topological Stage is not the whole story.
        A sink whose retained output was released (#69) is streamed back from its checkpoint.
        """
        depended_on = {p for s in ordered for p in s.depends_on}
        sinks = [s.name for s in ordered if s.name not in depended_on]
        final: list[StageOutput] = []
        for name in sinks:
            final.extend(
                self._reader.stage_records(
                    name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            )
        return final

    def build(
        self,
        pipeline: Pipeline,
        run_id: str,
        storage: Storage,
        ordered: list[Stage],
        stage_metrics: list[StageMetrics],
        outputs: Mapping[str, list[StageOutput]],
        final_records: Iterable[Record | StageOutput],
    ) -> QualityReport:
        records = list(final_records)
        completeness = self._completeness_check(records)
        # Document count = distinct source documents seen across the run. Chunks
        # fan out from a document (many chunks share one document_id), so counting
        # distinct Provenance document ids is correct whether the sinks are
        # DocumentRecords or ChunkRecords. A Profiling Stage's CORPUS-scope
        # FeatureRecord carries a synthetic corpus-scope document_id (it is not a source
        # document), so it is excluded from the source-document count.
        doc_ids = {
            r.provenance.document_id
            for r in records
            if not (isinstance(r, FeatureRecord) and r.scope is FeatureScope.CORPUS)
        }
        # Parse/Chunk quality scan the Stage that produced those records (not just
        # the sinks), so a Source→Parse→Chunk→Export pipeline (whose sinks are
        # ChunkRecords) still reports Parse metrics from the Parse Stage's output.
        parsed = self._records_of_kind(
            pipeline, ordered, outputs, CapabilityKind.PARSE, storage=storage, run_id=run_id
        )
        chunked = self._records_of_kind(
            pipeline, ordered, outputs, CapabilityKind.CHUNK, storage=storage, run_id=run_id
        )
        # Content Screening attaches the risk markers, so the flagged count is read
        # from the FINAL corpus records (post-screening), not the raw Chunk output
        # (which has no markers yet). Falls back to the chunk output when there is
        # no Content-Screening Stage.
        post_content = self._records_of_kind(
            pipeline,
            ordered,
            outputs,
            CapabilityKind.CONTENT_SCREENING,
            storage=storage,
            run_id=run_id,
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
            storage=storage,
            run_id=run_id,
        )
        relations = self._records_of_kinds(
            pipeline,
            ordered,
            outputs,
            (CapabilityKind.RELATION_EXTRACTION, CapabilityKind.FUSED_ENTITY_RELATION),
            storage=storage,
            run_id=run_id,
        )
        # A Redaction Stage that another Redaction Stage depends on (transitively) is SUPERSEDED:
        # its chunks are re-emitted downstream with its spans composed in, and only that
        # downstream version reaches the Export. Gathering the superseded Stage's output too
        # would tally every upstream span twice. Dropping it at the STAGE level (rather than
        # de-duplicating records by ``record_id``) is what keeps the rollup honest for a corpus
        # containing byte-identical documents, whose content-addressed ``record_id``s legitimately
        # repeat — see :meth:`_redaction_quality`.
        redactions = self._records_of_kind(
            pipeline,
            [s for s in ordered if s.name in self._terminal_redaction_stages(pipeline)],
            outputs,
            CapabilityKind.REDACTION,
            storage=storage,
            run_id=run_id,
        )
        features = self._records_of_kind(
            pipeline, ordered, outputs, CapabilityKind.PROFILING, storage=storage, run_id=run_id
        )
        disambiguations = self._records_of_kind(
            pipeline,
            ordered,
            outputs,
            CapabilityKind.DISAMBIGUATION,
            storage=storage,
            run_id=run_id,
        )
        graph_records = self._records_of_kind(
            pipeline,
            ordered,
            outputs,
            CapabilityKind.GRAPH_ASSEMBLY,
            storage=storage,
            run_id=run_id,
        )
        # W2-linkpred: the OPTIONAL Graph Completion Stage's predicted edges, gathered SEPARATELY
        # from the asserted graph so an inferred edge can never inflate the asserted-edge count
        # (they are a distinct capability's output, not part of graph_assembly). Empty when the
        # stack declares no Graph Completion Stage (byte-identical to today).
        inferred_records = self._records_of_kind(
            pipeline,
            ordered,
            outputs,
            CapabilityKind.GRAPH_COMPLETION,
            storage=storage,
            run_id=run_id,
        )
        # W13: the OPTIONAL Context Enrichment Stage's enriched ChunkRecords (each carrying a
        # ContextHeader), for the counts-only enrichment roll-up. Empty when the stack declares no
        # context_enrichment Stage (the section is then None — byte-identical to today).
        context_records = self._records_of_kind(
            pipeline,
            ordered,
            outputs,
            CapabilityKind.CONTEXT_ENRICHMENT,
            storage=storage,
            run_id=run_id,
        )
        # Mentions-in for the disambiguation rollup: the Entity-Extraction (or fused) output
        # the Stage consumed — the count it resolved from.
        disambig_mentions_in = self._records_of_kinds(
            pipeline,
            ordered,
            outputs,
            (CapabilityKind.ENTITY_EXTRACTION, CapabilityKind.FUSED_ENTITY_RELATION),
            storage=storage,
            run_id=run_id,
        )
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
            relations=self._relation_quality(pipeline, relations, self._fanout_total(ordered)),
            redaction=self._redaction_quality(pipeline, redactions),
            profiling=self._profiling_quality(pipeline, features),
            disambiguation=self._disambiguation_quality(
                pipeline, disambiguations, disambig_mentions_in
            ),
            graph=self._graph_quality(pipeline, graph_records, inferred_records),
            context_enrichment=self._context_enrichment_quality(
                pipeline, context_records, graph_records
            ),
            drift=self._drift_diagnostics(chunked),
            goldset=self._goldset_evaluation(pipeline, storage, mentions, relations),
        )

    # -- per-kind rollups (read records back through the reader seam) -------

    def _records_of_kind(
        self,
        pipeline: Pipeline,
        ordered: list[Stage],
        outputs: Mapping[str, list[StageOutput]],
        kind: CapabilityKind,
        *,
        storage: Storage,
        run_id: str,
    ) -> list[StageOutput]:
        """Flatten the outputs of every Stage fulfilling ``kind`` (order-stable).

        A Stage whose retained output was released (#69) is streamed back from its checkpoint,
        so the Quality Report's per-kind rollups are read one kind at a time rather than the
        whole doc-level output being held across the run.
        """
        result: list[StageOutput] = []
        for stage in ordered:
            if stage.capability is kind:
                result.extend(
                    self._reader.stage_records(
                        stage.name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                    )
                )
        return result

    def _records_of_kinds(
        self,
        pipeline: Pipeline,
        ordered: list[Stage],
        outputs: Mapping[str, list[StageOutput]],
        kinds: tuple[CapabilityKind, ...],
        *,
        storage: Storage,
        run_id: str,
    ) -> list[StageOutput]:
        """Flatten the outputs of every Stage fulfilling any of ``kinds`` (order-stable).

        Used where a record kind can be produced by more than one capability — a
        mention comes from an Entity-Extraction OR a fused Stage; a relation from a
        Relation-Extraction OR that same fused Stage — and the caller then filters the
        gathered output by carrier type. A released Stage's output is streamed from its
        checkpoint (#69).
        """
        result: list[StageOutput] = []
        for stage in ordered:
            if stage.capability in kinds:
                result.extend(
                    self._reader.stage_records(
                        stage.name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                    )
                )
        return result

    def _fanout_total(self, ordered: list[Stage]) -> tuple[int, int, int, int]:
        """Sum the per-Stage relation fan-out-guard tallies across the run (ADR-0033 / #63, #68).

        Folds every relation/fused Stage's ``(output-capped-docs, dropped-relations,
        input-capped-docs, dropped-mentions)`` tally (captured in the executor while Stages ran)
        into one corpus-wide 4-tuple for the Quality Report. A Stage restored from a checkpoint on
        resume never re-ran its Provider, so it contributes nothing here — the audited tally
        reflects Providers that actually executed this run (a resumed relation Stage's caps were
        already reflected in its committed output).
        """
        out_capped = 0
        dropped_rel = 0
        in_capped = 0
        dropped_men = 0
        for stage in ordered:
            oc, dr, ic, dm = self._fanout_by_stage.get(stage.name, (0, 0, 0, 0))
            out_capped += oc
            dropped_rel += dr
            in_capped += ic
            dropped_men += dm
        return out_capped, dropped_rel, in_capped, dropped_men

    # -- completeness / drift / gold-set ----------------------------------

    @staticmethod
    def _completeness_check(records: list[Record | StageOutput]) -> CompletenessCheck:
        """Contract-completeness over the exported records (Provenance/Classification + offsets).

        Every record must carry Provenance and Classification (contract-guaranteed, audited
        here). The S10 promotion also audits offset *alignment*: a record that carries a char
        span must carry a **well-ordered** one, and if it also carries page coordinates those
        must be well-ordered too — the "offsets aligned" half of the completeness AC. A record
        with no char span at all is vacuously aligned and not counted. Page *presence* is not
        re-required here: the sub-document carriers whose contracts require a page span
        (EntityMention/RelationMention) already guarantee it at construction, while a
        whole-document carrier (DocumentRecord/RedactionRecord/FeatureRecord) legitimately
        holds a ``[0, len)`` char span with no page span.
        """
        check = CompletenessCheck(
            records_checked=len(records),
            records_with_provenance=sum(1 for r in records if r.provenance is not None),
            records_with_classification=sum(1 for r in records if r.classification is not None),
        )
        for record in records:
            prov = record.provenance
            if prov.char_start is None and prov.char_end is None:
                # No char span to audit — vacuously aligned.
                continue
            check.offset_bearing_records += 1
            span_ok = (
                prov.char_start is not None
                and prov.char_end is not None
                and prov.char_end >= prov.char_start
            )
            pages_ok = (
                prov.page_start is None or prov.page_end is None or prov.page_end >= prov.page_start
            )
            if span_ok and pages_ok:
                check.records_with_aligned_offsets += 1
        return check

    @staticmethod
    def _drift_diagnostics(chunked: list[StageOutput]) -> DriftDiagnostics | None:
        """Provenance-integrity drift roll-up across the chunk offsets (S10).

        Audits every :class:`ChunkRecord` that carries its parent document's page map by
        resolving the chunk's ``[char_start, char_end)`` endpoints through a
        :class:`PageOffsetIndex` and accumulating the fuzzy recoveries — a finer drift
        signal than the Parse-only page-boundary audit. Only ChunkRecords are audited:
        mentions, relations, and redaction spans inherit the chunk offsets the Chunk Stage
        already validated against the page map, so they carry no independent offset→page
        drift. Returns ``None`` when there were no page-map-bearing chunk records to audit (a
        Parse-only or chunk-free run), so a pipeline with nothing to audit reports no drift
        section rather than a misleading all-zero one.
        """
        audited = 0
        resolutions = 0
        recovered = 0
        max_drift = 0
        for record in chunked:
            if not isinstance(record, ChunkRecord) or record.page_map is None:
                continue
            prov = record.provenance
            if prov.char_start is None or prov.char_end is None:
                continue
            index = PageOffsetIndex(record.page_map)
            index.page_for_offset(prov.char_start)
            index.page_for_offset(max(prov.char_end - 1, prov.char_start))
            diag = index.diagnostics
            audited += 1
            resolutions += diag.resolutions
            recovered += diag.recovered
            max_drift = max(max_drift, diag.max_drift_chars)
        if audited == 0:
            return None
        return DriftDiagnostics(
            resolutions=resolutions,
            recovered=recovered,
            max_drift_chars=max_drift,
            records_audited=audited,
        )

    def _goldset_evaluation(
        self,
        pipeline: Pipeline,
        storage: Storage,
        mentions: list[StageOutput],
        relations: list[StageOutput],
    ) -> GoldSetEvaluation | None:
        """Score the run against the adopter's gold set, when one is supplied (S10 harness).

        Returns ``None`` unless the Pipeline declares a ``gold_set_uri`` — the harness is
        opt-in (an adopter drops in 20–50 annotated documents). When present, the gold set is
        loaded from Storage (JSON or YAML), the extracted mentions/relations are mapped to
        their source file names, and per-Stage precision/recall/F1 are computed
        deterministically against the annotations. The mention side is required for both
        Stages (a relation's endpoints resolve through the mention spans), so a relations-only
        gold set with no Entity-Extraction Stage simply yields no relation matches — never an
        error.
        """
        if not pipeline.gold_set_uri:
            return None
        gold = self._load_gold_set(pipeline.gold_set_uri, storage)
        typed_mentions = [m for m in mentions if isinstance(m, EntityMention)]
        typed_relations = [r for r in relations if isinstance(r, RelationMention)]
        # Map each mention/relation's parent document to its source file name so the gold
        # annotations (keyed by file name) bind to the run's records.
        file_name_of: dict[str, str] = {}
        for m in typed_mentions:
            file_name_of.setdefault(m.document_record_id, m.provenance.file_name)
        for r in typed_relations:
            file_name_of.setdefault(r.document_record_id, r.provenance.file_name)

        gold_files = set(gold.by_file_name())
        matched_files = {
            file_name_of[doc_id] for doc_id in file_name_of if file_name_of[doc_id] in gold_files
        }
        evaluation = GoldSetEvaluation(
            documents_annotated=len(gold.documents),
            documents_matched=len(matched_files),
            match_min_overlap=gold.match_min_overlap,
        )
        entity_overall, entity_labels = score_entities(typed_mentions, gold, file_name_of)
        evaluation.stages.append(self._stage_score("entities", entity_overall, entity_labels))
        if gold.total_relations > 0:
            relation_overall, relation_labels = score_relations(
                typed_relations, typed_mentions, gold, file_name_of
            )
            evaluation.stages.append(
                self._stage_score("relations", relation_overall, relation_labels)
            )
        return evaluation

    @staticmethod
    def _load_gold_set(uri: str, storage: Storage) -> GoldSet:
        """Load + validate a gold set from Storage (JSON or YAML), by file extension."""
        text = storage.read_text(uri)
        if uri.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            msg = f"Gold set at {uri!r} must be a JSON/YAML mapping at the top level."
            # A malformed operator-supplied gold set is a config error (#73); ConfigError is a
            # ValueError subclass, so pre-taxonomy ``except ValueError`` callers keep catching it.
            raise ConfigError(msg)
        return GoldSet.from_mapping(data)

    @staticmethod
    def _stage_score(
        stage: str,
        overall: ConfusionCounts,
        per_label: dict[str, ConfusionCounts],
    ) -> StageScore:
        """Assemble a :class:`StageScore` from the confusion counts (overall + per label)."""
        return StageScore(
            stage=stage,
            true_positives=overall.true_positives,
            false_positives=overall.false_positives,
            false_negatives=overall.false_negatives,
            precision=overall.precision,
            recall=overall.recall,
            f1=overall.f1,
            per_label={
                label: LabelScore(
                    true_positives=c.true_positives,
                    false_positives=c.false_positives,
                    false_negatives=c.false_negatives,
                    precision=c.precision,
                    recall=c.recall,
                    f1=c.f1,
                )
                for label, c in per_label.items()
            },
        )

    # -- per-Stage quality rollups ----------------------------------------

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
            findings = self._reader.read_findings(storage, pipeline, run_id, stage.name)
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
    def _entity_quality(pipeline: Pipeline, records: list[StageOutput]) -> EntityQuality | None:
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
        pipeline: Pipeline,
        records: list[StageOutput],
        fanout: tuple[int, int, int, int] = (0, 0, 0, 0),
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
        Quality Report"). ``fanout`` is the run's ``(output-capped-docs, dropped-relations,
        input-capped-docs, dropped-mentions)`` fan-out-guard tally (ADR-0033 / #63 output cap;
        #68 input mention cap), surfaced as the audited "capped" marker so a bounded
        pathological corpus is never silently truncated.
        """
        relation_kinds = (
            CapabilityKind.RELATION_EXTRACTION,
            CapabilityKind.FUSED_ENTITY_RELATION,
        )
        if not any(s.capability in relation_kinds for s in pipeline.stages):
            return None
        quality = RelationQuality()
        (
            quality.fanout_capped_documents,
            quality.fanout_dropped_relations,
            quality.mention_capped_documents,
            quality.mention_dropped_mentions,
        ) = fanout
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
    def _terminal_redaction_stages(pipeline: Pipeline) -> set[str]:
        """Names of the Redaction Stages whose output actually reaches the rest of the Pipeline.

        A Redaction Stage is SUPERSEDED when some other Redaction Stage depends on it, directly or
        transitively: since W16 Redaction is a chunk→chunk transform and
        :func:`~latence_core.redaction_policy._compose_prior_spans` folds the upstream spans into
        the downstream Stage's own, the downstream Stage re-emits the same chunks carrying every
        mask the chain applied. Only that downstream version flows on to the Export, so only it may
        be tallied. Parallel Redaction branches supersede nothing and are all terminal.
        """
        by_name = {s.name: s for s in pipeline.stages}
        redaction_stages = {
            s.name for s in pipeline.stages if s.capability is CapabilityKind.REDACTION
        }
        superseded: set[str] = set()
        for name in redaction_stages:
            seen: set[str] = set()
            frontier = list(by_name[name].depends_on)
            while frontier:
                dep = frontier.pop()
                if dep in seen or dep not in by_name:
                    continue
                seen.add(dep)
                if dep in redaction_stages:
                    superseded.add(dep)
                frontier.extend(by_name[dep].depends_on)
        return redaction_stages - superseded

    @staticmethod
    def _redaction_quality(
        pipeline: Pipeline, records: list[StageOutput]
    ) -> RedactionQuality | None:
        """Aggregate Redaction quality: per-PII-type + per-action detection counts (S6/W16).

        Only computed when the Pipeline has a Redaction Stage. Since W16 (ADR-0042) Redaction is a
        chunk→chunk transform, so ``records`` is the set of redacted
        :class:`~latence_core.contracts.ChunkRecord`s (each carrying ``masked_content`` +
        ``pii_spans``). The per-type/per-action span counts are summed across chunks; the
        documents-redacted and documents-with-PII counts are rolled up to the DOCUMENT level
        (distinct
        ``document_record_id``s) so the report reads the same as the old one-record-per-document
        shape — the S6 acceptance criterion ("PII detection counts by type land in the Quality
        Report"). Only counts and type/action names cross into the report; **no raw PII value** ever
        does (a PIISpan never stores the surface string, so this rollup cannot leak one). The
        deprecated doc-level :class:`~latence_core.contracts.RedactionRecord` path is still tallied
        for backward compatibility.

        ``records`` carries the output of the TERMINAL Redaction Stages only
        (:meth:`_terminal_redaction_stages`), and every one of those records is summed with FULL
        MULTIPLICITY. Both halves matter, and they used to be got wrong in opposite directions:

        * Folding EVERY Redaction Stage's output (the flat concatenation ``_records_of_kind``
          builds) double-counts a CHAINED topology — exactly what the shipped ``gpu-learned`` /
          ``full-stack`` stacks declare since audit R5-12
          (``redact_financial.depends_on: [redact_pii]``). The same chunk appears twice: once as
          the upstream Stage emitted it, once as the downstream Stage re-emitted it after
          :func:`~latence_core.redaction_policy._compose_prior_spans` folded the upstream spans in.
          That inflates every upstream span and can name a ``pii_type`` that exists nowhere in the
          exported corpus (an upstream span replaced by an identical-extent downstream span of a
          DIFFERENT type would still be tallied under its old label). Dropping the superseded
          Stages at the source fixes this: the terminal Stage's chunks already carry every mask the
          chain applied, because composition is coverage-preserving.
        * De-duplicating the flat list by ``record_id`` instead (audit R6) fixes the chain but
          UNDER-counts the plain single-Redactor path, because ``record_id``s are
          content-addressed: two byte-identical documents share one ``document_id`` and hence one
          chunk ``record_id``, yet both are separately exported and both carry real PII. Halving
          that corpus's tally is a silent regression (audit R7), so no de-duplication by
          ``record_id`` happens here.

        The report is the operator's redaction audit trail (ADR-0034 counts-only observability), so
        it must describe the EXPORTED CORPUS — not the chain length, and not the number of
        distinct content hashes in it.
        """
        if not any(s.capability is CapabilityKind.REDACTION for s in pipeline.stages):
            return None
        quality = RedactionQuality()
        redacted_docs: set[str] = set()
        docs_with_pii: set[str] = set()
        disabled_docs: set[str] = set()
        for record in records:
            if isinstance(record, ChunkRecord):
                # Chunk-level (W16): masked_content is set on every chunk the Redaction Stage
                # produced. Roll spans up per type/action; document-level counts dedupe by parent.
                if record.masked_content is None:
                    continue
                doc_id = record.document_record_id
                redacted_docs.add(doc_id)
                if record.pii_spans:
                    docs_with_pii.add(doc_id)
                if record.redaction_disabled_for_sensitive:
                    disabled_docs.add(doc_id)
                spans = record.pii_spans
            elif isinstance(record, RedactionRecord):
                # Deprecated doc-level path: one record per document.
                redacted_docs.add(record.document_record_id)
                if record.pii_spans:
                    docs_with_pii.add(record.document_record_id)
                if record.redaction_disabled_for_sensitive:
                    disabled_docs.add(record.document_record_id)
                spans = record.pii_spans
            else:
                continue
            for span in spans:
                quality.pii_total += 1
                quality.per_type[span.pii_type] = quality.per_type.get(span.pii_type, 0) + 1
                action = span.action.value
                quality.per_action[action] = quality.per_action.get(action, 0) + 1
        quality.documents_redacted = len(redacted_docs)
        quality.documents_with_pii = len(docs_with_pii)
        # §4 no-op floor: a sensitive doc whose control ran no detectors. LOUD, first-class —
        # while this is > 0, documents_with_pii:0 is NOT a clean bill.
        quality.redaction_disabled_for_sensitive = len(disabled_docs)
        quality.per_type = dict(sorted(quality.per_type.items()))
        quality.per_action = dict(sorted(quality.per_action.items()))
        return quality

    @staticmethod
    def _profiling_quality(
        pipeline: Pipeline, records: list[StageOutput]
    ) -> ProfilingQuality | None:
        """Aggregate Profiling quality: per-document feature averages + corpus aggregates (S7).

        Only computed when the Pipeline has a Profiling Stage. ``records`` is the mix of
        DOCUMENT-scope and the single CORPUS-scope
        :class:`~latence_core.contracts.FeatureRecord`s. The document-scope records are
        averaged into a compact readability/density/diversity summary, and the
        corpus-scope record's aggregates (word/vocabulary counts, corpus Zipf α, source
        coverage, entity-type frequency, top co-occurrences) are surfaced verbatim —
        the S7 acceptance criterion ("feature summary lands in the Quality Report").
        Counts and averages only (no raw document text). All maps are already
        engine-sorted; the co-occurrence list is rendered as ``pair-fingerprint`` →
        document-count — the raw entity surfaces (possible PII from a ``confidential``
        source that would otherwise land in an ``internal``-stamped report) are replaced
        by a deterministic, **salted** sha256 fingerprint of the ``left|right`` pair (the
        same salted-digest construction S6 uses; salt from the Redaction Stage config).
        The fingerprint keeps the raw surface out of the report and defeats rainbow-table
        recovery, but it is not irreversible — a reader who can enumerate candidate
        entity names and knows the salt could still recover a pair by dictionary attack,
        so the report is *not* safe to publish to an untrusted audience. The raw surfaces
        remain only in the exported, sensitivity-stamped ``CorpusFeatures`` data artifact.
        """
        if not any(s.capability is CapabilityKind.PROFILING for s in pipeline.stages):
            return None
        quality = ProfilingQuality()
        doc_features = [
            r
            for r in records
            if isinstance(r, FeatureRecord)
            and r.scope is FeatureScope.DOCUMENT
            and r.document is not None
        ]
        n = len(doc_features)
        quality.documents_profiled = n
        if n:
            quality.avg_readability = round(
                sum(r.document.readability for r in doc_features if r.document) / n, 6
            )
            quality.avg_density = round(
                sum(r.document.density for r in doc_features if r.document) / n, 6
            )
            quality.avg_type_token_ratio = round(
                sum(r.document.type_token_ratio for r in doc_features if r.document) / n, 6
            )
            quality.avg_zipf_alpha = round(
                sum(r.document.zipf_alpha for r in doc_features if r.document) / n, 6
            )
            quality.avg_compression_ratio = round(
                sum(r.document.compression_ratio for r in doc_features if r.document) / n, 6
            )
        corpus = next(
            (
                r
                for r in records
                if isinstance(r, FeatureRecord)
                and r.scope is FeatureScope.CORPUS
                and r.corpus is not None
            ),
            None,
        )
        if corpus is not None and corpus.corpus is not None:
            cf = corpus.corpus
            quality.corpus_total_words = cf.total_words
            quality.corpus_unique_words = cf.total_unique_words
            quality.corpus_zipf_alpha = cf.corpus_zipf_alpha
            quality.language_coverage = dict(cf.language_coverage)
            quality.category_coverage = dict(cf.category_coverage)
            quality.sensitivity_coverage = dict(cf.sensitivity_coverage)
            quality.distinct_entities = len(cf.entity_frequency)
            quality.entity_type_frequency = dict(cf.entity_type_frequency)
            # Fingerprint each pair rather than emit the raw entity surfaces: a co-occurring
            # pair drawn from a ``confidential`` source must not have its (possibly PII)
            # surface rendered into the ``internal``-stamped Quality Report. The salted
            # sha256 fingerprint (same construction as S6's placeholder_for; salt resolved
            # from the Redaction Stage config) is deterministic, so equal pairs map to equal
            # keys (the recurrence signal is preserved) without printing the surface. It is a
            # fingerprint, not encryption — low-entropy surfaces remain dictionary-recoverable
            # by a reader who knows the salt, so this removes the surface from the report but
            # does not make the report publishable to an untrusted audience. The raw pair
            # remains only in the exported, sensitivity-stamped CorpusFeatures data artifact.
            salt = _fingerprint_salt(pipeline)
            quality.top_co_occurrences = {
                _co_occurrence_fingerprint(co.left, co.right, salt): co.document_count
                for co in cf.co_occurrences
            }
        return quality

    @staticmethod
    def _disambiguation_quality(
        pipeline: Pipeline,
        records: list[StageOutput],
        mentions_in: list[StageOutput],
    ) -> DisambiguationQuality | None:
        """Aggregate Disambiguation merge/link/normalisation statistics for the report (S8).

        Only computed when the Pipeline has a Disambiguation Stage. ``records`` is the mix of
        ENTITY-scope (canonical entities) and RELATION-scope (normalized relations)
        :class:`~latence_core.contracts.DisambiguationRecord`s; ``mentions_in`` the
        Entity-Extraction output the Stage consumed. Rolls up: mentions in vs canonical
        entities out; the created/merged/singleton breakdown; applied vs below-policy merges
        (the no-silent-over-merge audit) broken down by strategy; KB-linked vs unlinked
        entities; and relations normalized (with the inverted count). Counts only (no raw
        text) — the S8 acceptance criterion, safe to persist.
        """
        if not any(s.capability is CapabilityKind.DISAMBIGUATION for s in pipeline.stages):
            return None
        quality = DisambiguationQuality()
        quality.mentions_in = sum(1 for r in mentions_in if isinstance(r, EntityMention))
        entities = [
            r
            for r in records
            if isinstance(r, DisambiguationRecord)
            and r.scope is DisambiguationScope.ENTITY
            and r.entity is not None
        ]
        quality.entities_out = len(entities)
        confidence_sum = 0.0
        for record in entities:
            entity = record.entity
            if entity is None:  # pragma: no cover - filtered above
                continue
            if entity.mention_count > 1:
                quality.entities_merged += 1
            else:
                quality.entities_singleton += 1
            if entity.linked:
                quality.entities_linked += 1
            else:
                quality.entities_unlinked += 1
            confidence_sum += entity.evidence.confidence
            for decision in entity.merge_decisions:
                if decision.applied:
                    quality.merges_applied += 1
                    key = decision.strategy.value
                    quality.merges_by_strategy[key] = quality.merges_by_strategy.get(key, 0) + 1
                else:
                    quality.merges_below_policy += 1
        if entities:
            quality.avg_entity_confidence = round(confidence_sum / len(entities), 6)
        quality.merges_by_strategy = dict(sorted(quality.merges_by_strategy.items()))
        relations = [
            r
            for r in records
            if isinstance(r, DisambiguationRecord)
            and r.scope is DisambiguationScope.RELATION
            and r.relation is not None
        ]
        quality.relations_normalized = len(relations)
        quality.relations_inverted = sum(
            1 for r in relations if r.relation is not None and r.relation.inverted
        )
        return quality

    @staticmethod
    def _graph_quality(
        pipeline: Pipeline,
        records: list[StageOutput],
        inferred_records: list[StageOutput] | None = None,
    ) -> GraphQuality | None:
        """Aggregate Graph-Assembly KG statistics for the report (S9; + W2 inferred edges).

        Only computed when the Pipeline has a Graph Assembly Stage. ``records`` is the mix of
        NODE-scope (graph nodes) and EDGE-scope (graph edges)
        :class:`~latence_core.contracts.GraphRecord`s from the ASSERTED graph (the Graph Assembly
        Stage). Rolls up: node/edge counts; nodes by entity type and edges by relation label;
        KB-linked node count; and — the S9 acceptance criterion — evidence coverage, the fraction of
        edges whose Evidence names at least one source mention (1.0 = every edge is justified).

        ``inferred_records`` (W2-linkpred, ADR-0037) is the OPTIONAL Graph Completion Stage's
        PREDICTED EDGE-scope GraphRecords, counted SEPARATELY into ``inferred_edge_count`` /
        ``inferred_edges_by_scorer`` — NEVER folded into ``edges_total`` / ``edges_by_label`` /
        ``evidence_coverage``. An inferred edge must never inflate the asserted-edge statistics (the
        enterprise-honesty crux). A run with no Graph Completion Stage has ``inferred_edge_count ==
        0``. Counts only (no raw text), safe to persist. All maps are sorted for a deterministic
        report.
        """
        if not any(s.capability is CapabilityKind.GRAPH_ASSEMBLY for s in pipeline.stages):
            return None
        quality = GraphQuality()
        nodes = [
            r.node
            for r in records
            if isinstance(r, GraphRecord) and r.scope is GraphScope.NODE and r.node is not None
        ]
        edges = [
            r.edge
            for r in records
            if isinstance(r, GraphRecord) and r.scope is GraphScope.EDGE and r.edge is not None
        ]
        quality.nodes_total = len(nodes)
        quality.edges_total = len(edges)
        node_conf = 0.0
        for node in nodes:
            quality.nodes_by_type[node.label] = quality.nodes_by_type.get(node.label, 0) + 1
            if node.linked:
                quality.nodes_linked += 1
            node_conf += node.confidence
        edge_conf = 0.0
        for edge in edges:
            quality.edges_by_label[edge.label] = quality.edges_by_label.get(edge.label, 0) + 1
            if edge.evidence.mention_ids:
                quality.edges_with_evidence += 1
            edge_conf += edge.confidence
        if nodes:
            quality.avg_node_confidence = round(node_conf / len(nodes), 6)
        if edges:
            quality.avg_edge_confidence = round(edge_conf / len(edges), 6)
            quality.evidence_coverage = round(quality.edges_with_evidence / len(edges), 6)
        quality.nodes_by_type = dict(sorted(quality.nodes_by_type.items()))
        quality.edges_by_label = dict(sorted(quality.edges_by_label.items()))
        # W2-linkpred: count the OPTIONAL Graph Completion Stage's predicted edges SEPARATELY. They
        # are marked properties["inferred"]=True; we count only edges that carry that marker (a
        # defensive guard so nothing but a genuinely-inferred edge is ever tallied here).
        for r in inferred_records or []:
            if not isinstance(r, GraphRecord) or r.scope is not GraphScope.EDGE or r.edge is None:
                continue
            if r.edge.properties.get("inferred") is not True:
                continue
            quality.inferred_edge_count += 1
            scorer = str(r.edge.properties.get("scorer", "unknown"))
            quality.inferred_edges_by_scorer[scorer] = (
                quality.inferred_edges_by_scorer.get(scorer, 0) + 1
            )
        quality.inferred_edges_by_scorer = dict(sorted(quality.inferred_edges_by_scorer.items()))
        return quality

    @staticmethod
    def _context_enrichment_quality(
        pipeline: Pipeline,
        context_records: list[StageOutput],
        graph_records: list[StageOutput],
    ) -> ContextEnrichmentQuality | None:
        """Aggregate the OPTIONAL Context Enrichment Stage's coherence roll-up (W13, ADR-0039).

        Only computed when the Pipeline has a ``context_enrichment`` Stage (else None —
        byte-identical
        to today). Counts-only (no content). ``chunks_enriched`` + ``avg_neighbors_per_chunk`` are
        read off the enriched :class:`~latence_core.contracts.ChunkRecord`s directly (a chunk with a
        non-empty :class:`~latence_core.contracts.ContextHeader` counts; its ``neighbor_node_ids``
        are
        the query-time KG-expansion keys averaged). ``hub_entities_capped`` is the anti-bloat
        witness:
        the number of KG entities whose gated neighbor DEGREE exceeds the Stage's
        ``max_neighbors_per_entity`` cap — those entities contribute only their TOP-k relations
        wherever they appear (the hub-entity guard), computed from the assembled graph + the Stage's
        config so it is honest and reproducible.
        """
        stage = next(
            (s for s in pipeline.stages if s.capability is CapabilityKind.CONTEXT_ENRICHMENT),
            None,
        )
        if stage is None:
            return None
        quality = ContextEnrichmentQuality()
        headers = [
            r.context_header
            for r in context_records
            if isinstance(r, ChunkRecord) and r.context_header is not None
        ]
        quality.chunks_enriched = len(headers)
        if headers:
            total_neighbors = sum(len(h.neighbor_node_ids) for h in headers)
            quality.avg_neighbors_per_chunk = round(total_neighbors / len(headers), 6)

        # Hub-entity guard witness: a node whose gated neighbor degree exceeds the per-entity cap is
        # capped wherever it appears. Degree = distinct gated edges touching the node (each edge is
        # a neighbor from both endpoints). Mirror the Provider's edge-confidence gate + cap.
        max_neighbors = int(stage.config.get("max_neighbors_per_entity", 4))
        min_edge_confidence = float(stage.config.get("min_edge_confidence", 0.0))
        if max_neighbors > 0:
            degree: dict[str, int] = {}
            for r in graph_records:
                if (
                    not isinstance(r, GraphRecord)
                    or r.scope is not GraphScope.EDGE
                    or r.edge is None
                ):
                    continue
                edge = r.edge
                if edge.confidence < min_edge_confidence:
                    continue
                degree[edge.source_node_id] = degree.get(edge.source_node_id, 0) + 1
                degree[edge.target_node_id] = degree.get(edge.target_node_id, 0) + 1
            quality.hub_entities_capped = sum(1 for d in degree.values() if d > max_neighbors)
        return quality

    @staticmethod
    def _parse_quality(pipeline: Pipeline, records: list[StageOutput]) -> ParseQuality | None:
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
