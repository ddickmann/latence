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

from latence_core.capability import Export, Parser, ProviderRegistry, Source
from latence_core.contracts import Disposition, DocumentRecord, ParserInput, Record
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.quality import (
    CompletenessCheck,
    ParseQuality,
    QualityReport,
    StageMetrics,
)
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable

# A Stage's output is one of the two inter-Stage carriers on the S1 spine: a
# Source yields raw ParserInputs, a Parser/Export yields parsed DocumentRecords.
# The Runner keeps them in one per-Stage dict and lets each Stage's capability
# decide how to (de)serialize its checkpoint.
StageOutput = ParserInput | DocumentRecord
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

    # -- WAL ---------------------------------------------------------------

    def _wal_append(self, storage: Storage, uri: str, event: str, stage: str) -> None:
        line = json.dumps({"ts": time.time(), "event": event, "stage": stage}, sort_keys=True)
        existing = storage.read_text(uri) if storage.exists(uri) else ""
        storage.write_atomic(uri, (existing + line + "\n").encode("utf-8"))

    # -- checkpoint IO -----------------------------------------------------

    @staticmethod
    def _output_type(stage: Stage) -> type[StageOutput]:
        """The carrier a Stage checkpoints: Source→ParserInput, else DocumentRecord."""
        return ParserInput if stage.capability is CapabilityKind.SOURCE else DocumentRecord

    def _write_checkpoint(
        self, storage: Storage, uri: str, records: list[StageOutput]
    ) -> None:
        # model_dump_json round-trips ``bytes`` content losslessly (used for
        # ParserInput's raw content); a bare dict + json.dumps would not.
        payload = "\n".join(r.model_dump_json() for r in records)
        storage.write_atomic(uri, (payload + ("\n" if records else "")).encode("utf-8"))

    def _read_checkpoint(
        self, storage: Storage, uri: str, record_type: type[StageOutput]
    ) -> list[StageOutput]:
        text = storage.read_text(uri)
        records: list[StageOutput] = []
        for raw in text.splitlines():
            if raw.strip():
                records.append(record_type.model_validate_json(raw))
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
                own_output = self._read_checkpoint(storage, ckpt_uri, self._output_type(stage))
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
        report = self._build_report(pipeline, run_id, stage_metrics, final_records)
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

        if stage.capability is CapabilityKind.PARSE:
            if not isinstance(provider, Parser):
                msg = f"Provider {stage.provider!r} does not satisfy the Parser capability."
                raise TypeError(msg)
            parser_inputs = self._as_type(stage, incoming, ParserInput)
            return list(provider.parse(parser_inputs))

        if stage.capability is CapabilityKind.EXPORT:
            if not isinstance(provider, Export):
                msg = f"Provider {stage.provider!r} does not satisfy the Export capability."
                raise TypeError(msg)
            doc_records = self._as_type(stage, incoming, DocumentRecord)
            out_dir = self._export_dir(pipeline, run_id)
            provider.export(doc_records, storage, out_dir)
            # Export is a sink: it passes its input records through unchanged.
            return list(doc_records)

        msg = f"Unknown capability {stage.capability!r} for stage {stage.name!r}."
        raise ValueError(msg)

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
        stage_metrics: list[StageMetrics],
        final_records: Iterable[Record | StageOutput],
    ) -> QualityReport:
        records = list(final_records)
        completeness = CompletenessCheck(
            records_checked=len(records),
            # Contracts guarantee these are present; the check makes it auditable.
            records_with_provenance=sum(1 for r in records if r.provenance is not None),
            records_with_classification=sum(1 for r in records if r.classification is not None),
        )
        # Document count = distinct source documents seen at the Source stage.
        doc_ids = {r.provenance.document_id for r in records}
        return QualityReport(
            pipeline_name=pipeline.name,
            run_id=run_id,
            document_count=len(doc_ids),
            total_duration_seconds=round(sum(m.duration_seconds for m in stage_metrics), 6),
            stages=stage_metrics,
            completeness=completeness,
            contracts_complete=completeness.passed,
            parse=self._parse_quality(pipeline, records),
        )

    @staticmethod
    def _parse_quality(
        pipeline: Pipeline, records: list[Record | StageOutput]
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
