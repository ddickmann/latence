"""LocalRunner — single-node DAG execution with checkpoint/resume (ADR-0003, 0010).

Run state is files-on-Storage, no database: a run manifest, per-Stage checkpoint
files, and a write-ahead log, all under ``<storage_uri>/_latence/runs/<run_id>/``.
Checkpoints are written batch-granularly and streamed to a temp file that is
atomically published temp-write→rename (WAL discipline, ADR-0033), so the write
transient is O(batch) not O(all-records) and a run killed mid-Stage leaves either
the previous checkpoint or the complete new one — never a torn write. Re-running:

* a Stage whose checkpoint exists is skipped (resume);
* a fully-completed pipeline is a no-op.

Stages execute in topological order over their ``depends_on`` edges; for the S1
spine (Source→Parse→Export) the order is linear, but the DAG machinery is real.
"""

from __future__ import annotations

import json
import re
import time
from itertools import chain
from typing import TYPE_CHECKING, TypeVar, cast

from latence_core.capability import (
    Chunker,
    ContentScreener,
    ContextEnricher,
    Disambiguator,
    EntityExtractor,
    Export,
    FusedEntityRelationExtractor,
    GraphAssembler,
    GraphCompleter,
    IntakeScreener,
    LabelInducer,
    Parser,
    PIIDetector,
    Profiler,
    ProviderRegistry,
    RelationExtractor,
    ScreeningOutcome,
    Source,
    TypeConsolidator,
)
from latence_core.capability_descriptor import (
    ACCUMULATOR_KINDS,
    dag_protocol_for,
    descriptor_for,
    export_carrier_of,
)
from latence_core.contracts import (
    ChunkRecord,
    DisambiguationRecord,
    DocumentRecord,
    EntityMention,
    FeatureRecord,
    GraphRecord,
    ParserInput,
    QuarantineRecord,
    Record,
    RedactionRecord,
    RelationMention,
    ScreeningFinding,
    TypeVocabulary,
)
from latence_core.errors import ConfigError, ContractError, error_category
from latence_core.observability import (
    LineageDataset,
    OpenLineageEmitter,
    RunLineage,
    dataset_for_uri,
)
from latence_core.pipeline import CapabilityKind, ExtractOn, Pipeline, Stage
from latence_core.quality import (
    DriftDiagnostics,
    QualityReport,
    StageMetrics,
)

# H-F1: the co-occurrence fingerprint helpers now live in ``quality_report_builder`` (with the
# report code that uses them). They are re-exported here — via redundant ``as`` aliases, the
# explicit-re-export form mypy ``--strict`` and ruff both recognise — so callers that import them
# from ``latence_core.runner`` (some tests do) keep working.
from latence_core.quality_report_builder import (
    _CO_OCCURRENCE_FINGERPRINT_LEN as _CO_OCCURRENCE_FINGERPRINT_LEN,
)
from latence_core.quality_report_builder import (
    _DEFAULT_FINGERPRINT_SALT as _DEFAULT_FINGERPRINT_SALT,
)
from latence_core.quality_report_builder import QualityReportBuilder
from latence_core.quality_report_builder import (
    _co_occurrence_fingerprint as _co_occurrence_fingerprint,
)
from latence_core.quality_report_builder import (
    _fingerprint_salt as _fingerprint_salt,
)
from latence_core.stage_execution import StageContext, StageExecution
from latence_core.storage import Storage
from latence_core.tracing import RunTracer, SpanSink

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from typing import IO

# A Stage's output is one of the inter-Stage carriers the DAG threads between
# Stages: Source→ParserInput; Parse→DocumentRecord; Chunk/Content-Screening→
# ChunkRecord; Entity-Extraction→EntityMention; Relation-Extraction→RelationMention.
# A *fused* Entity+Relation Stage (ADR-0013) emits BOTH EntityMention and
# RelationMention in one output list (mixed carrier), routed to both downstream
# consumers without a separate Entity-Extraction Stage. The Runner keeps outputs in
# one per-Stage dict and lets each Stage's capability decide how to (de)serialize its
# checkpoint. Redaction→ChunkRecord (chunk→chunk transform: the SAME chunks with
# ``masked_content`` + ``pii_spans`` populated, W16/ADR-0042 — the deprecated doc-level path emits
# RedactionRecord). Profiling→FeatureRecord (per-document + one corpus-scope record). Screening ALSO
# produces QuarantineRecords, but those never flow downstream — they are persisted to a
# sidecar and reported, not threaded through the DAG.
StageOutput = (
    ParserInput
    | DocumentRecord
    | ChunkRecord
    | EntityMention
    | RelationMention
    | RedactionRecord
    | FeatureRecord
    | DisambiguationRecord
    | GraphRecord
)
_T = TypeVar("_T", bound=StageOutput)
#: The Capability Protocol a dispatch branch narrows its Provider to (see
#: :meth:`LocalRunner._capability_provider`). Unbounded: the Protocols share no base.
_P = TypeVar("_P")

# WAL event kinds. The log is append-only and never read for control flow in S1
# (checkpoint presence is authoritative); it exists for audit and crash forensics.
_WAL_STAGE_BEGIN = "stage_begin"
_WAL_STAGE_COMMIT = "stage_commit"
_WAL_STAGE_BATCH = "stage_batch"

# Checkpoint write batch size (ADR-0033 / #63): the number of records serialized and
# flushed to the temp checkpoint at once, so the write transient is O(batch) instead of the
# old O(all-records) whole-string build. Purely a memory-shaping knob — the on-disk file
# (newline-delimited records) and Stage-granular resume are identical regardless of the value.
_CHECKPOINT_BATCH_RECORDS = 1024

# Checkpoint *read-back* batch size (ADR-0033 / #69): the number of newline-delimited records
# decoded at once when an accumulator (or the Quality Report) reads a doc-level Stage's output
# back from its checkpoint. Streaming the read means the doc-level ``outputs`` dict entry can be
# released before the accumulators run — corpus-level memory then bounded by the KG state the
# accumulators hold, not by the whole doc-level output (the ADR's honest boundary). Purely a
# memory-shaping knob; the decoded records are identical regardless of the value.
_CHECKPOINT_READ_BATCH_RECORDS = 1024

# The corpus-level accumulator Capabilities (ADR-0033): they inherently consume the whole
# doc-level output, but their own output is bounded by the knowledge-graph size (canonical
# entities + nodes/edges + corpus features), not the corpus size. Everything else is a
# per-document streaming transform whose retained output list is released once its consumers have
# run (#69), so the executor no longer holds every Stage's doc-level records simultaneously.
#
# DERIVED from the Capability's own ``accumulates`` declaration (L2/C2). It used to be a
# hand-ordered tuple here, one of four capability tables in this file and one of eight across the
# codebase — and that is precisely how TYPE_CONSOLIDATION came to be listed here and in the delta
# engine but in NEITHER phase-boundary set. The tuple form (and the declaration order) is kept
# because the executor's memory-shaping decisions are order-insensitive but its logs are not.
_ACCUMULATOR_KINDS: tuple[CapabilityKind, ...] = tuple(
    kind for kind in CapabilityKind if kind in ACCUMULATOR_KINDS
)

# The doc-level Capabilities whose INPUT the Runner streams (ADR-0033, audit R2).
#
# C6 streamed the output side — the Provider's iterator goes straight into the checkpoint, so a
# Stage's records never exist as a list. Its INPUT was still materialised whole-corpus:
# ``_gather_inputs`` concatenated every parent's records into one list held for the Stage's whole
# execution, and ``_as_type`` built a second list of the same size (a third under
# ``extract_on: masked``). For the Capabilities below the Protocol is ``Iterable``-in and the
# Provider consumes it exactly once, so the Runner hands it a lazy stream read record-by-record
# from the parents' checkpoints (:meth:`LocalRunner._stream_doc_inputs`) — the input transient
# becomes O(1) records rather than O(corpus-slice).
#
# The doc-level Capabilities NOT listed here are excluded because their Protocol makes the
# materialisation the PROVIDER's, not a Runner-side buffer — streaming the input would move no
# memory and only cost a second pass:
#   * INTAKE_SCREENING / CONTENT_SCREENING return a ``ScreeningOutcome`` holding passed +
#     quarantined lists, and Content Screening's page-map re-homing needs the input set to know
#     what was dropped;
#   * RELATION_EXTRACTION and CONTEXT_ENRICHMENT partition ONE input stream into two carriers
#     (chunks + mentions / chunks + graph), so one side must be buffered whatever the Runner does;
#   * EXPORT writes a file and holds what it materialises (streamed internally in row groups).
# Their transients are the two inherent exceptions ADR-0033 already names, plus the two-carrier
# partition — see the ADR's **Honest memory boundary**.
_STREAMED_INPUT_KINDS: frozenset[CapabilityKind] = frozenset(
    {
        CapabilityKind.PARSE,
        CapabilityKind.CHUNK,
        CapabilityKind.SCHEMA_INDUCTION,
        CapabilityKind.ENTITY_EXTRACTION,
        CapabilityKind.FUSED_ENTITY_RELATION,
        CapabilityKind.REDACTION,
    }
)

# Secret-handling (issue #30): the run manifest persists the whole Pipeline
# (including every Stage's operator ``config``) verbatim. A Provider such as the S5
# relation.llm reads its ``api_key`` from ``$OPENAI_API_KEY`` (env fallback, no key
# in code/logs — the Baseline), but an operator who *inlines* ``api_key`` into the
# Pipeline YAML would otherwise write it cleartext into ``manifest.json``. Any config
# key whose (lowercased) name contains one of these tokens is redacted before the
# manifest is written. Matching is substring-based so ``api_key``, ``openai_api_key``,
# ``auth_token``, ``client_secret``, ``db_password`` are all caught.
_SENSITIVE_KEY_TOKENS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    # audit-R7: certificate-based auth was the blind spot. Entra ID app-only access to Graph
    # is *recommended* by Microsoft with a certificate, so ``private_key:``/``certificate:``
    # inlined into a Stage config is the realistic misconfiguration — and none of the tokens
    # above appears as a substring of either name, nor of ``passphrase``. Every key that
    # ``latence_source_sharepoint.auth._SECRET_CONFIG_KEYS`` refuses must be covered here:
    # that Provider guard runs when the Provider is CONSTRUCTED, which is strictly after the
    # manifest is written, so the scrubber — not the guard's ordering — is what actually keeps
    # a credential out of ``manifest.json``.
    "private_key",
    "privatekey",
    "certificate",
    "passphrase",
    "keystore",
    "keyfile",
)

# Matched against a *whole* underscore/dash/dot-delimited part of the key rather than as a
# substring, because these are short enough that substring matching would redact innocent
# config (``cert`` is inside ``certainty_threshold``). ``client_cert`` / ``tls.pem`` are caught
# here; ``certainty_threshold`` is not.
_SENSITIVE_KEY_PARTS = frozenset({"cert", "certs", "pem", "pfx", "p12"})

_KEY_PART_SPLIT = re.compile(r"[^a-z0-9]+")

_REDACTED = "***REDACTED***"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if any(token in lowered for token in _SENSITIVE_KEY_TOKENS):
        return True
    return any(part in _SENSITIVE_KEY_PARTS for part in _KEY_PART_SPLIT.split(lowered))


def scrub_sensitive(value: object) -> object:
    """Recursively redact sensitive config keys in a JSON-able manifest structure.

    Walks the ``pipeline.model_dump(mode="json")`` tree and replaces the *value* of
    any mapping key that looks like a secret (see ``_SENSITIVE_KEY_TOKENS``) with a
    fixed placeholder, keeping the key so the manifest still documents that a secret
    was configured — just not its value (issue #30). Lists and nested mappings are
    scrubbed in place-of-copy; scalars pass through. Env-var *references* the operator
    might legitimately keep (e.g. ``"$OPENAI_API_KEY"``) are redacted too — the
    manifest never needs the secret's value, referenced or inlined.
    """
    if isinstance(value, dict):
        scrubbed: dict[str, object] = {}
        for key, val in value.items():
            if isinstance(key, str) and _is_sensitive_key(key):
                scrubbed[key] = _REDACTED
            else:
                scrubbed[key] = scrub_sensitive(val)
        return scrubbed
    if isinstance(value, list):
        return [scrub_sensitive(item) for item in value]
    return value


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
        # A dependency cycle is a malformed-Pipeline config error (#73). ConfigError is a
        # ValueError subclass, so the pre-taxonomy ``except ValueError`` callers keep catching it.
        raise ConfigError(msg)
    return [by_name[n] for n in order]


class LocalRunner:
    """Executes a Pipeline on the local node with resumable file-based state."""

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        *,
        span_sink: SpanSink | None = None,
        lineage: OpenLineageEmitter | None = None,
    ) -> None:
        self._registry = registry or ProviderRegistry()
        # #185: the optional OpenLineage seam. ``None`` (the default) means OFF — no event is
        # built and no socket is opened, so a default run is byte- and behaviour-identical to a
        # pre-#185 run. An operator who points it at a lineage endpoint
        # (``build_lineage_emitter(url)``) gets per-Stage START/COMPLETE/FAIL RunEvents with the
        # Stage's input/output datasets + schema facets, in the standard vocabulary every catalog
        # (Marquez/Atlan/Collibra/Dataplex) already consumes. Counts only, never content; an
        # emission failure is logged and swallowed — lineage never fails a run.
        self._lineage = lineage
        # The run-scoped lineage handle, set for the duration of ``run()`` (mirrors the
        # ``_reuse_overlay_by_type`` pattern) so ``_run`` can emit per-Stage events without
        # threading a parameter through the whole executor. ``None`` ⇒ lineage off.
        self._lineage_run: RunLineage | None = None
        # #73: the optional tracing sink. ``None`` ⇒ each run's RunTracer defaults to the
        # silent-by-default LoggingSpanSink, so a normal run's output is unchanged; an operator
        # (or a downstream tracer, or a test) passes a CollectingSpanSink / their own SpanSink to
        # capture the per-Stage spans. Kept a thin seam — no OTel dep in core (ADR-0016).
        self._span_sink = span_sink
        # Per-Stage relation fan-out-guard tally, keyed by Stage name: (output-capped docs,
        # relations dropped [ADR-0033 / #63], input-capped docs, mentions dropped [#68]). A
        # RelationExtractor Provider exposes ``fanout_capped_documents``/
        # ``fanout_dropped_relations`` (output cap) and ``mention_capped_documents``/
        # ``mention_dropped_mentions`` (input cap) int attributes; the Runner reads them after
        # the Stage runs and folds them into the Quality Report's RelationQuality. A Provider
        # without them contributes (0, 0, 0, 0).
        self._fanout_by_stage: dict[str, tuple[int, int, int, int]] = {}
        # #69 + C6: when True (the default), a Stage's records are STREAMED from its Provider
        # straight into its checkpoint and never retained in the in-memory ``outputs`` dict at all
        # — the finalized checkpoint is the source of truth (ADR-0017) and every later reader
        # (a downstream Stage's gather, an accumulator's stream, the Quality Report) reads it back
        # batch-by-batch through :meth:`_stage_records`. That is what completes ADR-0033's bound:
        # the executor's per-Stage transient is O(checkpoint batch), not O(the Stage's records).
        # Setting it False materialises and retains every Stage's output (the pre-#69 behaviour) —
        # a parity/debug knob; the on-disk output, resume, and report are byte-identical either
        # way, only peak memory differs.
        self._release_doc_level_outputs = True
        # W17 (ADR-0043): the cross-run doc-level REUSE OVERLAY, keyed by carrier type. A delta
        # excludes an unchanged document at Source (so its doc-level Stages never run) and hands
        # the Runner that document's persisted doc-level records here; the Runner streams them into
        # the CORPUS-LEVEL accumulators (Profiling/Disambiguation/Graph) and the post-assembly
        # per-chunk transforms (Context Enrichment) / a doc-level Export ALONGSIDE the
        # freshly-extracted records, so those Stages see the whole corpus (reused ∪ fresh) exactly
        # as if every document had been re-extracted — while the doc-level Stages ran only on the
        # changed documents. The overlay is NEVER fed to a doc-level Stage (parse→…→redact) — those
        # gather via ``_gather_inputs``, which is overlay-blind — so a reused document is never
        # re-parsed/-extracted. Empty by default (``run(reuse=None)``), so a non-delta run is
        # byte-identical. Because Disambiguation folds mentions in sorted ``mention_id`` order and
        # every record id is content-addressed, a reused record is byte-identical to its fresh
        # re-extraction and the interleave order is irrelevant — the committed corpus is identical
        # to a from-scratch run of the whole corpus (the ADR-0043 equivalence property).
        self._reuse_overlay_by_type: dict[type, list[Record]] = {}
        # H-F1: the Quality Report half of the Runner, extracted into its own cohesive unit
        # (``quality_report_builder``). The engine composes it — passing itself as the
        # ``StageRecordReader`` (its public ``stage_records`` + ``read_findings``) and its live
        # ``_fanout_by_stage`` tally — so the report build is the same byte-identical logic, now a
        # relocatable module reused by both Runners through the public interface rather than via a
        # private reach-through. ``_build_report`` / ``_final_records`` below are thin delegators.
        self._report_builder = QualityReportBuilder(self, fanout_by_stage=self._fanout_by_stage)
        # C3: the Stage-execution module. Everything it takes to run ONE Stage — provider load,
        # device routing, skip-with-flag, the dispatch, the Provider tally drains (AFTER the
        # record stream is consumed), typed-error categorisation and the full StageMetrics
        # assembly — now lives behind ``execute(stage, ctx) -> StageOutcome``. This loop supplies
        # only scheduling + the files-on-Storage substrate; a second Runner supplies its own and
        # gets identical metrics for free, because it shares this object (see ``stage_execution``).
        self._stage_execution = StageExecution(
            self, self._registry, fanout_by_stage=self._fanout_by_stage
        )

    @property
    def quality_report_builder(self) -> QualityReportBuilder:
        """The engine's composed :class:`~latence_core.quality_report_builder.QualityReportBuilder`.

        Public so a second Runner reuses the *same* report seam this engine uses — bound to this
        engine's live reader (``stage_records`` / ``read_findings``) and its ``_fanout_by_stage``
        tally, so a report built through it is byte-identical to the local Runner's. This is the
        interface the Airflow adapter builds its report through, instead of reaching into the
        private ``_build_report`` / ``_final_records`` (the audit's THEME F fix, H-F1).
        """
        return self._report_builder

    @property
    def stage_execution(self) -> StageExecution:
        """The Stage-execution module every Runner runs its Stages through (C3).

        Public because it IS the Runner seam: a second Runner (Airflow, Delta, an adopter's own)
        calls ``stage_execution.execute(stage, ctx)`` and gets device routing, the skip-with-flag
        posture, the Provider tally drains and a fully-populated
        :class:`~latence_core.quality.StageMetrics` — identical to this engine's, because it is
        the same code, not a re-implementation. Bound to this engine's registry, its dispatch and
        its live fan-out tally, so a Quality Report built off either Runner carries the same
        columns (ADR-0003: the identical Stages under a pluggable Runner).
        """
        return self._stage_execution

    def dispatch_stage(
        self,
        stage: Stage,
        provider: object,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        incoming: Iterable[StageOutput],
        outputs: dict[str, list[StageOutput]],
    ) -> Iterator[StageOutput]:
        """The :class:`~latence_core.stage_execution.StageDispatch` half this engine supplies.

        An internal collaborator of :class:`~latence_core.stage_execution.StageExecution`, not part
        of its interface — a Runner adapter never calls this. Public only because the executor is a
        separate module and reaching a private name across it is exactly the reach-through this
        refactor removes.
        """
        return self._dispatch_stage(
            stage, provider, storage, pipeline, run_id, incoming, outputs
        )

    def _retaining(
        self,
        outputs: dict[str, list[StageOutput]],
        stage: Stage,
        records: Iterator[StageOutput],
    ) -> Iterator[StageOutput]:
        """Pass a Stage's record stream through, retaining it only under the parity knob (C6).

        Default (``_release_doc_level_outputs``): yields straight through, so the Stage's records
        go Provider→checkpoint without ever existing as a list — the O(batch) bound. Nothing lands
        in ``outputs``; the finalized checkpoint is the source of truth (ADR-0017) and every later
        reader goes through :meth:`_stage_records`, which falls back to it transparently.

        With the knob OFF, the stream is materialised and retained exactly as the pre-#69 executor
        did — the parity/debug path, whose only difference is peak memory.
        """
        if self._release_doc_level_outputs:
            return records
        retained = list(records)
        outputs[stage.name] = retained
        return iter(retained)

    def _retain(
        self,
        outputs: dict[str, list[StageOutput]],
        stage: Stage,
        records: Iterator[StageOutput],
    ) -> None:
        """Drain a Stage's record stream (resume path), retaining it only under the parity knob."""
        for _ in self._retaining(outputs, stage, records):
            pass

    # -- state layout ------------------------------------------------------

    def run_dir(self, pipeline: Pipeline, run_id: str) -> str:
        """The run's state/artifact directory on Storage (``<uri>/_latence/runs/<run_id>``).

        Public so a second Runner reusing this engine's report path can land the run's data
        artifacts where the shared report builder expects them, without reaching a private method
        (H-F1). The local Runner's own file-state (manifest, checkpoints, WAL) still lives here.
        """
        return f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"

    # Retained private name for the many internal call sites; a thin alias over the public method.
    def _run_dir(self, pipeline: Pipeline, run_id: str) -> str:
        return self.run_dir(pipeline, run_id)

    def _checkpoint_uri(self, pipeline: Pipeline, run_id: str, stage_name: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/checkpoints/{stage_name}.jsonl"

    def _manifest_uri(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/manifest.json"

    def _wal_uri(self, pipeline: Pipeline, run_id: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/wal.log"

    def report_uri(self, pipeline: Pipeline, run_id: str) -> str:
        """The run's Quality Report location on Storage.

        Public so a second Runner persists the report the shared
        :class:`~latence_core.quality_report_builder.QualityReportBuilder` produced to the same
        place an adopter reads it regardless of Runner — no private reach-through (H-F1).
        """
        return f"{self._run_dir(pipeline, run_id)}/quality-report.json"

    # Retained private name for the executor's own report writes; a thin alias.
    def _report_uri(self, pipeline: Pipeline, run_id: str) -> str:
        return self.report_uri(pipeline, run_id)

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

    def _wal_append_batch(
        self, storage: Storage, uri: str, stage: str, batch_index: int, records_so_far: int
    ) -> None:
        """Append a per-batch checkpoint-progress WAL event (ADR-0033 / #63).

        Logged as each checkpoint batch is durably written, so the WAL records batch-granular
        progress (``batch_index`` + cumulative ``records`` flushed) for crash forensics and
        audit. Like the other WAL events it is append-only and not read for control flow — the
        finalized checkpoint's presence remains authoritative for resume (Stage-granular skip).
        """
        line = json.dumps(
            {
                "ts": time.time(),
                "event": _WAL_STAGE_BATCH,
                "stage": stage,
                "batch": batch_index,
                "records": records_so_far,
            },
            sort_keys=True,
        )
        existing = storage.read_text(uri) if storage.exists(uri) else ""
        storage.write_atomic(uri, (existing + line + "\n").encode("utf-8"))

    # -- checkpoint IO -----------------------------------------------------

    @classmethod
    def _output_type(cls, stage: Stage, pipeline: Pipeline) -> type[StageOutput] | type[Record]:
        """The carrier a Stage checkpoints (the records it feeds downstream).

        Read straight off the Capability's own :class:`~latence_core.capability_descriptor.\
CapabilityDescriptor` (L2/C2) rather than re-decided by a fifty-line if-chain here. Every
        rationale that used to be a comment beside a branch — Source/Intake Screening thread raw
        ParserInputs, the chunk→chunk transforms (Schema Induction, Context Enrichment, Redaction)
        checkpoint the SAME ChunkRecords, a fused or type-consolidating Stage checkpoints a MIXED
        carrier signalled by :class:`~latence_core.contracts.Record` and read back via
        :meth:`_read_mixed_checkpoint` — now sits beside the ``carrier`` field that states it.

        The one thing a descriptor cannot state is Export's carrier, because it does not have one
        of its own: an Export is a pass-through sink whose carrier is whatever its nearest
        producing ancestor feeds (DocumentRecords on the S1/S2 spine, ChunkRecords once a Chunk
        Stage sits upstream), so its checkpoint re-reads with the right type on resume. A
        ``carrier`` of ``None`` is exactly that "resolve it from the DAG" signal.
        """
        descriptor = descriptor_for(stage.capability)
        if descriptor.carrier is not None:
            return cast("type[StageOutput] | type[Record]", descriptor.carrier)
        if descriptor.dag_node:
            return cls._export_carrier(stage, pipeline)
        # EMBEDDING: carrier-less AND not a DAG node (a nested Export sub-config, ADR-0017), so it
        # is never a Stage and this is unreachable through a validated Pipeline. Answer with the
        # spine's carrier rather than resolving through ``depends_on`` a non-Stage does not have.
        return DocumentRecord

    def _lineage_io(
        self,
        pipeline: Pipeline,
        run_id: str,
        stage: Stage,
        by_name: Mapping[str, Stage],
    ) -> tuple[list[LineageDataset], list[LineageDataset]]:
        """The datasets a Stage reads and writes, in OpenLineage terms (#185).

        A Stage's **output** dataset is its checkpoint on Storage — the durable artifact the run
        actually produces — carrying a ``schema`` facet derived from the Stage's carrier contract
        (field names + coarse types, never a value). Its **inputs** are its parents' checkpoints,
        so the per-Stage events chain into exactly the pipeline DAG in the catalog's graph. A
        Stage with no parents (a Source) is the graph's entry point: its input is the location it
        reads from (``path``/``uri`` in its config, else the Pipeline's Storage root) with no
        schema — the raw corpus has no carrier contract yet.
        """
        outputs = [
            dataset_for_uri(
                self._checkpoint_uri(pipeline, run_id, stage.name),
                self._output_type(stage, pipeline),
            )
        ]
        parents = [by_name[p] for p in stage.depends_on if p in by_name]
        if parents:
            inputs = [
                dataset_for_uri(
                    self._checkpoint_uri(pipeline, run_id, parent.name),
                    self._output_type(parent, pipeline),
                )
                for parent in parents
            ]
        else:
            source = stage.config.get("path") or stage.config.get("uri")
            location = source if isinstance(source, str) and source else pipeline.storage_uri
            inputs = [dataset_for_uri(location)]
        return inputs, outputs

    @staticmethod
    def _emit_stage_lineage(
        lineage: RunLineage,
        stage: Stage,
        *,
        records_in: int,
        records_out: int,
        duration_ms: float,
        inputs: Sequence[LineageDataset],
        outputs: Sequence[LineageDataset],
        from_checkpoint: bool = False,
        skipped: bool = False,
        emit_start: bool = True,
    ) -> None:
        """Emit a Stage's START (optional) + COMPLETE pair — one place, so every exit agrees.

        ``emit_start=False`` is the *executed* path, where the START was already emitted before
        the Provider ran; the resumed and device-skipped paths emit both here because they never
        reach that point. A consumer therefore always sees a well-formed START/COMPLETE pair per
        Stage regardless of which path the run took.
        """
        if emit_start:
            lineage.stage_start(
                stage.name,
                capability=stage.capability.value,
                provider=stage.provider,
                records_in=records_in,
                inputs=inputs,
                outputs=outputs,
            )
        lineage.stage_end(
            stage.name,
            capability=stage.capability.value,
            provider=stage.provider,
            records_in=records_in,
            records_out=records_out,
            duration_ms=duration_ms,
            inputs=inputs,
            outputs=outputs,
            from_checkpoint=from_checkpoint,
            skipped=skipped,
        )

    @classmethod
    def _export_carrier(cls, stage: Stage, pipeline: Pipeline) -> type[StageOutput] | type[Record]:
        """Resolve an Export's pass-through carrier from its nearest producing ancestor.

        Walks up ``depends_on`` to the closest Stage that actually *produces* the record kind
        Export re-emits, and re-emits that Stage's declared ``carrier``. WHICH ancestors count as
        producers is the Capability's own declaration (``materializes_for_export``, L2/C2), read
        here through :func:`~latence_core.capability_descriptor.export_carrier_of`: a ``None``
        answer means "not a producer, keep walking" — a Source/Intake-Screening ancestor threads
        raw ParserInputs an Export can never re-emit (so the walk continues to the Parse they must
        feed), and an Export above an Export resolves through to the real producer. A walk that
        reaches the top with no producer falls back to DocumentRecord, the spine's carrier.

        This is the second copy of the carrier table that C2 deleted: it was hand-ordered
        separately from :meth:`_output_type` and had to be kept consistent with it by memory.

        The frontier is walked breadth-first in ``depends_on`` order, so the nearest producing
        ancestor wins.
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
            carrier = export_carrier_of(parent.capability)
            if carrier is not None:
                return cast("type[StageOutput] | type[Record]", carrier)
            frontier.extend(parent.depends_on)
        return DocumentRecord

    def _write_checkpoint(
        self,
        storage: Storage,
        uri: str,
        records: Iterable[StageOutput],
        *,
        wal_uri: str | None = None,
        stage_name: str | None = None,
        tracer: RunTracer | None = None,
    ) -> int:
        """Write a Stage checkpoint batch-granularly, bounding the write transient (ADR-0033).

        The buffered path built the *entire* checkpoint as one ``"\\n".join(...)`` string before
        writing — an O(all-records) transient that, with the pre-#63 whole-corpus Export, was one
        of the peaks G1 measured. This streams instead: records are serialized in bounded batches
        (``_CHECKPOINT_BATCH_RECORDS``) and each batch is appended to a sibling temp file, so at
        most one batch's serialized text is live at once. Each completed batch is logged to the
        WAL (``stage_batch`` events) for crash forensics + audit, then the temp file is atomically
        published to the final ``uri`` (temp→rename, the WAL-atomic discipline ADR-0010 requires):
        a crash mid-Stage leaves the previous checkpoint or the complete new one, never a torn
        write — resume stays exact at Stage granularity (a finalized checkpoint ⇒ skip).

        The on-disk format is unchanged — newline-delimited ``model_dump_json`` per record, one
        record per line, trailing newline iff non-empty — so the delta engine's checkpoint reader
        and the Airflow adapter's typed decode round-trip byte-identically. ``model_dump_json``
        round-trips ``bytes`` content losslessly (ParserInput's raw content); a bare dict +
        ``json.dumps`` would not. Returns the number of records written.

        When a ``tracer`` is supplied (#73), each durably-written batch also emits a
        ``stage_batch`` trace span (batch index + cumulative records, counts only) mirroring the
        WAL batch event — so a downstream tracer sees checkpoint throughput at batch granularity.

        The temp-name minting, the parent ``makedirs`` (a streaming ``open`` does not create one)
        and the sweep of the partial write live behind
        :meth:`~latence_core.storage.Storage.atomic_writer` (L8/C9), not here. C6 made that sweep
        load-bearing rather than tidy: under streaming the *Provider* runs inside this write, so a
        Stage failure surfaces mid-file — and on a flaky store each retry mints a fresh temp name,
        so an unswept failure stranded another full-size copy of the run's records per attempt.
        """
        with storage.atomic_writer(uri) as fh:
            total, _batches = self._stream_batches(
                fh, storage, records, wal_uri=wal_uri, stage_name=stage_name, tracer=tracer
            )
        return total

    def _stream_batches(
        self,
        fh: IO[bytes],
        storage: Storage,
        records: Iterable[StageOutput],
        *,
        wal_uri: str | None,
        stage_name: str | None,
        tracer: RunTracer | None,
    ) -> tuple[int, int]:
        """Serialize ``records`` into the open handle in bounded batches; return (records, batches).

        ``storage`` is here only for the WAL append — the checkpoint bytes go to ``fh``, whose
        publication (or sweep) is :meth:`_write_checkpoint`'s ``atomic_writer`` block.
        """
        total = 0
        batch_index = 0
        batch: list[str] = []
        for record in records:
            batch.append(record.model_dump_json())
            if len(batch) >= _CHECKPOINT_BATCH_RECORDS:
                fh.write(("\n".join(batch) + "\n").encode("utf-8"))
                total += len(batch)
                batch = []
                if wal_uri is not None and stage_name is not None:
                    self._wal_append_batch(storage, wal_uri, stage_name, batch_index, total)
                # #73: mirror the WAL batch progress as a trace span (counts only), so a
                # downstream tracer sees checkpoint throughput per batch, not just per Stage.
                if tracer is not None and stage_name is not None:
                    tracer.stage_batch(stage_name, batch_index=batch_index, records_so_far=total)
                batch_index += 1
        if batch:
            fh.write(("\n".join(batch) + "\n").encode("utf-8"))
            total += len(batch)
            if wal_uri is not None and stage_name is not None:
                self._wal_append_batch(storage, wal_uri, stage_name, batch_index, total)
            if tracer is not None and stage_name is not None:
                tracer.stage_batch(stage_name, batch_index=batch_index, records_so_far=total)
            batch_index += 1
        return total, batch_index

    def _read_checkpoint(
        self, storage: Storage, uri: str, record_type: type[StageOutput] | type[Record]
    ) -> list[StageOutput]:
        return self._read_checkpoint_text(storage.read_text(uri), record_type)

    def _stream_checkpoint(
        self, storage: Storage, uri: str, record_type: type[StageOutput] | type[Record]
    ) -> Iterator[StageOutput]:
        """Yield a Stage checkpoint's records lazily, decoding at most one line at a time (#69).

        The read-back companion to :meth:`_write_checkpoint`'s batch-granular write: rather than
        materialising the whole checkpoint into a list (:meth:`_read_checkpoint`), this opens the
        file and yields one typed record per non-blank line, so an accumulator (or the Quality
        Report) that folds the doc-level output in one pass never holds it whole. This is what lets
        the executor RELEASE a doc-level Stage's ``outputs`` entry before the accumulators run —
        the corpus-level transient becomes O(batch) read-back + the accumulator's own bounded KG
        state, not O(all doc-level records) (ADR-0033's honest boundary, completed by #69).

        The typed decode is identical to :meth:`_read_checkpoint_text` (same
        ``model_validate_json`` / mixed-carrier discrimination), so a streamed read and a whole
        read produce byte-identical records in the same order — determinism and the
        Provenance/offset/Evidence contracts are untouched. The file handle is closed when the
        generator is exhausted or garbage-collected.
        """
        with storage.open(uri, "rb") as fh:
            for raw_line in fh:
                raw = raw_line.decode("utf-8")
                if not raw.strip():
                    continue
                # Decode one line through the shared typed decode (identical validation +
                # mixed-carrier discrimination as the whole-file read), so a streamed record is
                # byte-identical to the buffered one. A blank line yields nothing.
                yield from self._read_checkpoint_text(raw, record_type)

    def _read_checkpoint_text(
        self, text: str, record_type: type[StageOutput] | type[Record]
    ) -> list[StageOutput]:
        """Validate newline-delimited record JSON into typed records (the checkpoint decode).

        Split out from :meth:`_read_checkpoint` so a Runner adapter that transports Stage outputs
        over a non-file substrate (e.g. Airflow XCom, S12) reuses the *exact* same typed decode —
        the contract survives the boundary, no bespoke (de)serialization per Runner.
        """
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

    def run(
        self,
        pipeline: Pipeline,
        run_id: str = "run-0001",
        *,
        runner_name: str = "local",
        reuse: list[Record] | None = None,
    ) -> QualityReport:
        # W17 (ADR-0043): ``reuse`` carries an unchanged document's persisted doc-level records
        # (from the prior Corpus Version) that this run did NOT re-extract (the document is
        # excluded at Source). They are indexed by carrier type and streamed into the corpus-level
        # accumulators + post-assembly transforms so those Stages see the whole corpus. Set for the
        # duration of the run and cleared in ``finally`` so the instance never leaks a delta's
        # overlay into a later plain run. ``None`` ⇒ no overlay (byte-identical to a non-delta run).
        self._reuse_overlay_by_type = {}
        for record in reuse or []:
            self._reuse_overlay_by_type.setdefault(type(record), []).append(record)
        # #185: the run-level OpenLineage bracket. Off unless an emitter was wired, in which case
        # the pipeline itself is one parent job whose START/COMPLETE/FAIL nests every Stage's job
        # underneath it in the catalog. The FAIL carries the stable typed error *category* only —
        # never the exception message, which could quote a document (ADR-0034).
        lineage = (
            self._lineage.for_run(pipeline.name, run_id) if self._lineage is not None else None
        )
        self._lineage_run = lineage
        if lineage is not None:
            lineage.run_start()
        try:
            report = self._run(pipeline, run_id, runner_name=runner_name)
        except BaseException as exc:
            if lineage is not None:
                lineage.run_end(error_category=error_category(exc))
            raise
        else:
            if lineage is not None:
                lineage.run_end(documents=report.document_count)
            return report
        finally:
            self._reuse_overlay_by_type = {}
            self._lineage_run = None

    def _run(
        self, pipeline: Pipeline, run_id: str = "run-0001", *, runner_name: str = "local"
    ) -> QualityReport:
        storage = Storage.for_uri(pipeline.storage_uri)
        run_dir = self._run_dir(pipeline, run_id)
        storage.makedirs(run_dir)

        manifest_uri = self._manifest_uri(pipeline, run_id)
        if not storage.exists(manifest_uri):
            # Scrub inlined secrets (api_key/token/secret/password/…) from every Stage
            # config before persisting — an operator who inlines a key into the YAML
            # must not have it written cleartext to manifest.json (issue #30).
            manifest_pipeline = scrub_sensitive(pipeline.model_dump(mode="json"))
            storage.write_atomic(
                manifest_uri,
                json.dumps(
                    {"pipeline": manifest_pipeline, "run_id": run_id},
                    sort_keys=True,
                ).encode("utf-8"),
            )

        ordered = topological_order(list(pipeline.stages))
        wal_uri = self._wal_uri(pipeline, run_id)
        # #73: the run-scoped tracer. The correlation id is the ``run_id`` — the run *is* the
        # trace — so every span of this run is greppable together and a downstream tracer can
        # reconstruct the per-run tree. Silent by default (LoggingSpanSink at DEBUG) unless a
        # sink was supplied; carries counts + typed-error categories only, never record content.
        tracer = RunTracer(run_id, self._span_sink)

        stage_metrics: list[StageMetrics] = []
        # Per-Stage outputs keyed by Stage name. A Stage's INPUT is the gather of
        # ALL its ``depends_on`` parents' outputs (multi-input, so a converging
        # DAG {a,b}->c sees both a's and b's records — never a single overwritten
        # ``current`` var, which corrupted data on resume). A downstream Stage
        # always reads its parents' *current* outputs from this dict, so a parent
        # that had to re-run (its checkpoint absent) feeds the child fresh data.
        #
        # #69: a doc-level Stage's entry is RELEASED from this dict once every Stage that
        # depends on it has run — its checkpoint on disk is then the source of truth. Live
        # entries are bounded by the DAG fan-in, and the corpus-level accumulators
        # (Profiling/Disambiguation/Graph) read their doc-level input back from checkpoints
        # batch-by-batch (:meth:`_stage_records`) rather than holding it whole, so the
        # executor no longer retains every Stage's doc-level output simultaneously (ADR-0033).
        outputs: dict[str, list[StageOutput]] = {}
        # Remaining not-yet-run consumers per Stage (how many later Stages still depend on it),
        # used to release a doc-level Stage's retained output the moment it goes dead. The
        # Quality Report reads any released doc-level set back from its checkpoint (bounded), so
        # release never loses data the report needs.
        pending_consumers: dict[str, int] = {s.name: 0 for s in ordered}
        for s in ordered:
            for parent in s.depends_on:
                if parent in pending_consumers:
                    pending_consumers[parent] += 1

        def _release_dead_parents(just_ran: Stage) -> None:
            """Drop a parent's retained output once ``just_ran`` was its last pending consumer.

            Accumulator outputs (Profiling/Disambiguation/Graph) are bounded by the KG state and
            may still be read by the end-of-run Quality Report from the dict, so they are kept;
            a doc-level parent's output is re-readable from its checkpoint, so it is released.
            The ``_release_doc_level_outputs`` knob (default True) gates the whole mechanism.
            """
            if not self._release_doc_level_outputs:
                return
            for parent in just_ran.depends_on:
                if parent not in pending_consumers:
                    continue
                pending_consumers[parent] -= 1
                if pending_consumers[parent] > 0:
                    continue
                parent_stage = by_name.get(parent)
                if parent_stage is not None and parent_stage.capability not in _ACCUMULATOR_KINDS:
                    outputs.pop(parent, None)
            # A doc-level SINK (nothing depends on it — e.g. an Export, which passes its input
            # through unchanged) has no consumer to trigger its release, yet its retained output
            # would linger for the whole run. Release it here the moment it commits; the Quality
            # Report reads it back from its checkpoint. Accumulator sinks stay (bounded KG state).
            if (
                pending_consumers.get(just_ran.name, 0) == 0
                and just_ran.capability not in _ACCUMULATOR_KINDS
            ):
                outputs.pop(just_ran.name, None)

        by_name = {s.name: s for s in ordered}

        for stage in ordered:
            ckpt_uri = self._checkpoint_uri(pipeline, run_id, stage.name)
            # #69: a corpus-level accumulator (Profiling/Disambiguation/Graph) consumes the whole
            # doc-level output — materialising it into a list here would reintroduce the O(all
            # records) transient the ADR bounds. So its input is NOT gathered eagerly: the
            # dispatch streams each carrier from the parents' checkpoints one record at a time
            # (:meth:`_stream_inputs_of_type`). ``records_in`` for it is a streamed count only
            # (:meth:`_gathered_count`), never a retained list. Every other Stage is per-document
            # and small enough to gather eagerly (its own output is then released once dead).
            is_accumulator = stage.capability in _ACCUMULATOR_KINDS
            incoming: Iterable[StageOutput]
            if is_accumulator:
                incoming = []
                records_in = self._gathered_count(
                    stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            elif stage.capability in _STREAMED_INPUT_KINDS:
                # Audit R2: the doc-level INPUT side of the same bound. The Stage's Provider
                # consumes an ``Iterable`` exactly once, so it is handed a lazy stream over the
                # parents' checkpoints instead of a gathered list — nothing corpus-sized is ever
                # built (:data:`_STREAMED_INPUT_KINDS`). ``records_in`` is counted without
                # decoding a single record (:meth:`_streamed_input_count`), so the span/lineage
                # column stays exact and the count costs no memory and no validation pass.
                incoming = self._stream_doc_inputs(
                    stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
                records_in = self._streamed_input_count(
                    stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            else:
                incoming = self._gather_inputs(
                    stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
                records_in = len(incoming)

            # #185: this Stage's lineage datasets (its parents' checkpoints in, its own out),
            # resolved once and reused by every exit below (resumed / skipped / ran / failed) so
            # the emitted graph has the same shape whichever path the Stage takes. Built only when
            # lineage is wired — when it is off this is two empty lists and no work at all.
            lineage = self._lineage_run
            lineage_in: list[LineageDataset] = []
            lineage_out: list[LineageDataset] = []
            if lineage is not None:
                lineage_in, lineage_out = self._lineage_io(pipeline, run_id, stage, by_name)

            ctx = StageContext(
                storage=storage,
                pipeline=pipeline,
                run_id=run_id,
                incoming=incoming,
                outputs=outputs,
                runner=runner_name,
            )

            if storage.exists(ckpt_uri):
                # Resume: this Stage already completed, so replay its checkpoint through the SAME
                # executor rather than hand-building a metrics row. That is what keeps the
                # profile/device columns (including a skip flag) STICKY across a resume — the
                # executor re-routes the device from the Provider's declared profile without ever
                # executing it, so a skipped GPU Stage still reports skipped=True on resume instead
                # of quietly reappearing as an ordinary Stage.
                resumed = self._stage_execution.execute(
                    stage,
                    ctx,
                    replay=self._stream_checkpoint(
                        storage, ckpt_uri, self._output_type(stage, pipeline)
                    ),
                )
                self._retain(outputs, stage, resumed.records)
                stage_metrics.append(resumed.metrics(records_in=records_in))
                # #185: a resumed Stage still belongs in the lineage graph — its dataset edges are
                # real (the checkpoint exists and downstream reads it). Flagged ``from_checkpoint``
                # so a catalog can tell a re-executed Stage from a replayed one.
                if lineage is not None:
                    self._emit_stage_lineage(
                        lineage,
                        stage,
                        records_in=records_in,
                        records_out=resumed.records_out,
                        duration_ms=0.0,
                        inputs=lineage_in,
                        outputs=lineage_out,
                        from_checkpoint=True,
                    )
                _release_dead_parents(stage)
                continue

            self._wal_append(storage, wal_uri, _WAL_STAGE_BEGIN, stage.name)
            tracer.stage_begin(
                stage.name,
                capability=stage.capability.value,
                provider=stage.provider,
                records_in=records_in,
            )
            if lineage is not None:
                lineage.stage_start(
                    stage.name,
                    capability=stage.capability.value,
                    provider=stage.provider,
                    records_in=records_in,
                    inputs=lineage_in,
                    outputs=lineage_out,
                )
            # C3 + C6: resolve inputs → execute → persist → release. The executor owns provider
            # load, device routing, the skip-with-flag posture (ADR-0036 §2), the dispatch, the
            # Provider tally drains and the whole StageMetrics row; this loop owns scheduling and
            # the files-on-Storage substrate and nothing else. The Stage's records are STREAMED
            # from the Provider straight into the checkpoint writer — never materialised into a
            # list — and the executor's teardown fires when that stream is exhausted, which is why
            # the tallies below read the Provider's real counters rather than zeros.
            outcome = self._stage_execution.execute(stage, ctx)
            try:
                # Batch-granular streamed checkpoint write (ADR-0033 / #63 + C6): the write
                # transient is O(batch), not O(all-records); batch progress lands in the WAL; the
                # final file is published atomically (temp→rename) so resume stays exact at Stage
                # granularity. A device-skipped Stage streams zero records, so this writes the
                # empty checkpoint that makes the skip sticky on resume — same code path, no
                # special case.
                self._write_checkpoint(
                    storage,
                    ckpt_uri,
                    self._retaining(outputs, stage, outcome.records),
                    wal_uri=wal_uri,
                    stage_name=stage.name,
                    tracer=tracer,
                )
            except BaseException as exc:
                # #73: a Stage failure emits a *failed* stage_end span stamped with the typed-error
                # category (contract/config/provider/storage/screening/unknown) + message, then
                # re-raises — the Runner never swallows a Stage failure (no torn commit), it just
                # makes the failure kind observable first. Counts + category only, never content.
                failed = outcome.metrics(records_in=records_in, error=exc)
                tracer.stage_end(
                    stage.name,
                    capability=stage.capability.value,
                    provider=stage.provider,
                    records_in=records_in,
                    records_out=0,
                    duration_ms=round(failed.duration_seconds * 1000.0, 3),
                    error_category=failed.error_category,
                    error_message=str(exc),
                )
                # #185: the FAIL RunEvent. It carries the stable typed *category* and nothing else
                # about the error — the standard ``errorMessage`` facet is deliberately never
                # emitted, because ``str(exc)`` can quote a record (ADR-0034 counts-only).
                if lineage is not None:
                    lineage.stage_end(
                        stage.name,
                        capability=stage.capability.value,
                        provider=stage.provider,
                        records_in=records_in,
                        records_out=0,
                        duration_ms=round(failed.duration_seconds * 1000.0, 3),
                        inputs=lineage_in,
                        outputs=lineage_out,
                        error_category=failed.error_category,
                    )
                # #73: the taxonomy is surfaced in the report *and* the logs — so record the failed
                # Stage's typed-error metric (error_count=1 + the stable category tag, counts only,
                # never the exception message) and persist a FAIL Quality Report before re-raising.
                #
                # No-torn-write is preserved: the failed Stage's *checkpoint* is still never
                # finalized (the streamed write publishes temp→rename only after the last record,
                # and discards the temp file on failure), so a resume re-runs it. The report is not
                # a resume gate (resume keys off per-Stage checkpoints, not the report), so a
                # subsequent successful resume overwrites this FAIL report atomically. Persisting
                # must never mask the real failure: if the report build itself raises we swallow
                # *that* and re-raise the original Stage exception.
                stage_metrics.append(failed)
                try:
                    fail_report = self._build_report(
                        pipeline,
                        run_id,
                        storage,
                        ordered,
                        stage_metrics,
                        outputs,
                        self._final_records(
                            ordered, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                        ),
                    )
                    storage.write_atomic(
                        self._report_uri(pipeline, run_id),
                        fail_report.to_json().encode("utf-8"),
                    )
                except Exception:  # noqa: BLE001 - never mask the real Stage failure
                    pass
                raise
            self._wal_append(storage, wal_uri, _WAL_STAGE_COMMIT, stage.name)
            metrics = outcome.metrics(records_in=records_in)
            tracer.stage_end(
                stage.name,
                capability=stage.capability.value,
                provider=stage.provider,
                records_in=records_in,
                records_out=outcome.records_out,
                duration_ms=round(metrics.duration_seconds * 1000.0, 3),
            )
            if lineage is not None:
                self._emit_stage_lineage(
                    lineage,
                    stage,
                    records_in=records_in,
                    records_out=outcome.records_out,
                    duration_ms=round(metrics.duration_seconds * 1000.0, 3),
                    inputs=lineage_in,
                    outputs=lineage_out,
                    emit_start=False,
                    skipped=outcome.skipped,
                )
            stage_metrics.append(metrics)
            # #69: release any doc-level parent whose last consumer just ran — its checkpoint is
            # the source of truth for the end-of-run Quality Report, which reads it back
            # (bounded) via _stage_records. Under the default (streaming) knob nothing was
            # retained in the first place, so this is a no-op; it is what the parity knob needs.
            _release_dead_parents(stage)

        final_records = self._final_records(
            ordered, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        report = self._build_report(
            pipeline, run_id, storage, ordered, stage_metrics, outputs, final_records
        )
        storage.write_atomic(self._report_uri(pipeline, run_id), report.to_json().encode("utf-8"))
        return report

    @staticmethod
    def _capability_provider(stage: Stage, provider: object, protocol: Callable[..., _P]) -> _P:
        """Check ``provider`` against the Capability's DECLARED Protocol, narrowed to ``protocol``.

        The runtime check reads the Protocol off the Capability's own
        :class:`~latence_core.capability_descriptor.CapabilityDescriptor` (L2/C2) rather than from
        a per-branch ``isinstance`` in the dispatch below. That table — seventeen hand-written
        capability→Protocol pairs, each with its own message string — was one of the copies C2
        deleted, and the copy that mattered most: it is the one the run itself enforces.

        ``protocol`` is the *static* face of the same fact — what mypy narrows to, so each branch
        can call its Capability's method without a cast. It is asserted to BE the declared
        Protocol, so a branch that names the wrong one is a typed failure here rather than a
        check that quietly enforces something other than what the descriptor declares. (It is
        annotated ``Callable[..., _P]`` rather than ``type[_P]`` only because mypy reads
        ``type[SomeProtocol]`` as "a concrete subclass of"; it is always a Protocol class.)

        Raises :class:`~latence_core.errors.ContractError` naming the declared Protocol — the same
        message the seventeen inline checks produced, now derived rather than transcribed.
        """
        declared = dag_protocol_for(stage.capability)
        if declared is not protocol:
            msg = (
                f"Runner dispatch for {stage.capability} narrows to {protocol!r} but "
                f"{stage.capability} declares {declared.__name__} in "
                f"latence_core.capability_descriptor.DESCRIPTORS."
            )
            raise ContractError(msg)
        if not isinstance(provider, declared):
            msg = (
                f"Provider {stage.provider!r} does not satisfy "
                f"the {declared.__name__} capability."
            )
            raise ContractError(msg)
        return cast("_P", provider)

    def _dispatch_stage(
        self,
        stage: Stage,
        provider: object,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        incoming: Iterable[StageOutput],
        outputs: dict[str, list[StageOutput]],
    ) -> Iterator[StageOutput]:
        """Narrow a Stage's input to its capability's carrier and return the Provider's STREAM.

        C6: every Capability Protocol is already ``Iterable``-in/``Iterator``-out, so the
        Provider's iterator is handed back **uncollapsed** — the Runner writes it record-by-record
        into the checkpoint and the Stage's output never exists as a list.

        Audit R2 completes the other half: for the Capabilities in :data:`_STREAMED_INPUT_KINDS`
        the ``incoming`` this receives is itself a lazy stream over the parents' checkpoints, and
        the narrowing is lazy too (:meth:`_as_type_stream`), so the Stage's INPUT is never a list
        either. Those branches trade an eager wiring check for the bound: the same typed
        ``ContractError`` is raised at the first record of the wrong carrier instead of before the
        Provider is called (same accept/reject set — see :meth:`_as_type_stream`). Every other
        branch still receives a gathered list and validates eagerly.

        The remaining materialisations are inherent, not oversights: Export must hold the records
        it materialises (it writes a file); the screening Capabilities return an outcome object
        holding passed + quarantined; a two-carrier branch (Relation, Context Enrichment) must
        buffer one side of the partition; and the fused/consolidating Stages return two carriers
        from one call. All are bounded by what the Provider itself already held.
        """
        if stage.capability is CapabilityKind.SOURCE:
            source = self._capability_provider(stage, provider, Source)
            # A Source is a root: it reads Storage, not upstream records.
            return iter(source.produce(storage))

        # The eager branches partition their input more than once (or hand it to a Provider that
        # returns an outcome object), so they need it as a list — and they always receive one:
        # only :data:`_STREAMED_INPUT_KINDS` are handed a lazy stream. Binding it here once makes
        # which branch reads which shape explicit, and keeps a future streamed Capability from
        # silently double-reading a one-shot generator.
        gathered: list[StageOutput] = (
            []
            if stage.capability in _STREAMED_INPUT_KINDS
            else incoming
            if isinstance(incoming, list)
            else list(incoming)
        )

        if stage.capability is CapabilityKind.INTAKE_SCREENING:
            intake_screener = self._capability_provider(stage, provider, IntakeScreener)
            raw_inputs = self._as_type(stage, gathered, ParserInput)
            intake_outcome = intake_screener.screen_intake(raw_inputs)
            self._persist_screening_sidecar(storage, pipeline, run_id, stage, intake_outcome)
            # Only the passed raw inputs proceed to Parse; the quarantined ones are
            # removed from the DAG (persisted for audit above), never reaching Parse.
            return iter(intake_outcome.passed)

        if stage.capability is CapabilityKind.PARSE:
            parser = self._capability_provider(stage, provider, Parser)
            parser_inputs = self._as_type_stream(stage, incoming, ParserInput)
            return iter(parser.parse(parser_inputs))

        if stage.capability is CapabilityKind.CHUNK:
            chunker = self._capability_provider(stage, provider, Chunker)
            doc_records = self._as_type_stream(stage, incoming, DocumentRecord)
            return iter(chunker.chunk(doc_records))

        if stage.capability is CapabilityKind.CONTENT_SCREENING:
            content_screener = self._capability_provider(stage, provider, ContentScreener)
            chunk_records = self._as_type(stage, gathered, ChunkRecord)
            content_outcome = content_screener.screen_content(chunk_records)
            self._persist_screening_sidecar(storage, pipeline, run_id, stage, content_outcome)
            # Passed chunks (some now carrying propagated RiskMarkers) proceed to
            # Export; quarantined chunks are removed and persisted for audit.
            #
            # Nothing has to be re-homed when a chunk is removed. Since v19 each chunk
            # carries its OWN page spans, so dropping one takes only that chunk's page
            # resolution with it — which is the correct outcome, since the chunk is gone.
            # v18 needed a ``rehome_page_maps()`` pass here because a document's whole map
            # rode its first chunk, and quarantining that one record (a document's opening
            # chunk is a likely place for an injection marker) silently destroyed page
            # resolution for every later chunk of the document. That patch also only ever
            # covered THIS Stage — every other chunk→chunk Capability kept the hole.
            return iter(content_outcome.passed)

        if stage.capability is CapabilityKind.SCHEMA_INDUCTION:
            label_inducer = self._capability_provider(stage, provider, LabelInducer)
            # W4: a chunk→chunk transform (the Content-Screening shape). It consumes the run's
            # chunks and returns the SAME chunks with ``induced_labels`` populated (one LLM call
            # per document, broadcast to that doc's chunks). Fail-open lives in the Provider: a
            # chunk it could not annotate flows through with ``induced_labels`` still None, so a
            # downstream extractor falls back to its config labels and the run never crashes. The
            # 1:1 chunk stream then feeds Entity/Relation Extraction, which read induced labels off
            # the chunk directly. Redaction reads a per-document DocumentRecord, so when a stack
            # also wires this chunk stream into ``redact`` the Runner projects each document's
            # induced labels onto its DocumentRecord (the bridge in the REDACTION dispatch below).
            induce_chunks = self._as_type_stream(stage, incoming, ChunkRecord)
            return iter(label_inducer.induce(induce_chunks))

        if stage.capability is CapabilityKind.ENTITY_EXTRACTION:
            entity_extractor = self._capability_provider(stage, provider, EntityExtractor)
            # Audit R2: chunks in, one at a time — the corpus-sized transient this branch used to
            # hold (the gathered list + its narrowed copy + the masked ``model_copy`` of it) is
            # what made an Entity Stage over a large corpus OOM despite C6.
            entity_chunks = self._extract_input_chunks_streamed(
                pipeline, self._as_type_stream(stage, incoming, ChunkRecord)
            )
            return iter(entity_extractor.extract(entity_chunks))

        if stage.capability is CapabilityKind.RELATION_EXTRACTION:
            relation_extractor = self._capability_provider(stage, provider, RelationExtractor)
            # Relation Extraction is multi-input: it wires BOTH a Chunk (or
            # Content-Screening) parent for text/page-map context AND an
            # Entity-Extraction parent for the mentions it relates. Partition the
            # gathered inputs by carrier; a missing side is a wiring error.
            rel_chunks = [r for r in gathered if isinstance(r, ChunkRecord)]
            rel_mentions = [r for r in gathered if isinstance(r, EntityMention)]
            self._require_relation_inputs(stage, gathered, rel_chunks, rel_mentions)
            rel_chunks = self._extract_input_chunks(pipeline, rel_chunks)
            return iter(relation_extractor.relate(rel_chunks, rel_mentions))

        if stage.capability is CapabilityKind.FUSED_ENTITY_RELATION:
            fused_extractor = self._capability_provider(
                stage, provider, FusedEntityRelationExtractor
            )
            # One pass over the chunks yields BOTH mentions and relations; the fused
            # Stage threads them together downstream (no separate Entity-Extraction
            # Stage — the S5 no-double-extract routing).
            fused_chunks = self._extract_input_chunks_streamed(
                pipeline, self._as_type_stream(stage, incoming, ChunkRecord)
            )
            fused = fused_extractor.extract_fused(fused_chunks)
            return chain(fused.mentions, fused.relations)

        if stage.capability is CapabilityKind.REDACTION:
            pii_detector = self._capability_provider(stage, provider, PIIDetector)
            # W16 (ADR-0042): Redaction is a chunk→chunk transform (the fix for the document-level
            # 768-token truncation leak). It consumes the run's ChunkRecords (each ≤ the model
            # window) and yields the SAME chunks with ``masked_content`` + ``pii_spans`` populated
            # — the clean ``content`` stays UNMASKED, the Export materializes ``masked_content`` for
            # the PII-safe RAG corpus. It depends on Chunk (or Content Screening / Schema
            # Induction), NOT Parse: scanning each chunk in full masks PII anywhere in it, and a
            # chunk-local span resolves to its ORIGINAL source page through the chunk's offset_map +
            # page_slice (the ADR-0031/0042 coordinate-correctness guarantee).
            #
            # The induced PII schema (W4/ADR-0038) is read directly off each chunk's
            # ``induced_labels`` — no document bridge needed (chunks already carry it when a
            # schema_induction Stage is upstream), so a Provider UNIONs the induced pii_types onto
            # its config floor per chunk.
            #
            # The W16 wiring guidance is the streamed narrowing's own message here, rather than a
            # second check after it: ``_as_type``/``_as_type_stream`` already rejects the first
            # non-chunk record, so the separate ``no ChunkRecords`` guard could never fire and its
            # (better) advice never reached an operator.
            redaction_chunks = self._as_type_stream(
                stage,
                incoming,
                ChunkRecord,
                message=lambda got: (
                    f"Stage {stage.name!r} (redaction) received no ChunkRecords "
                    f"(got: {got}). Since W16 (ADR-0042) Redaction is a chunk→chunk "
                    f"transform and must depend on the Chunk (or Content Screening / Schema "
                    f"Induction) Stage — the chunks it redacts, not Parse. Check the "
                    f"Pipeline's depends_on wiring."
                ),
            )
            return iter(pii_detector.redact(redaction_chunks))

        if stage.capability is CapabilityKind.PROFILING:
            profiler = self._capability_provider(stage, provider, Profiler)
            # Profiling is corpus-level and multi-input: it wires a Parse parent for
            # the DocumentRecords (per-document features) and OPTIONALLY an
            # Entity-Extraction parent for the EntityMentions (cross-document entity
            # features). The mentions side is optional (a Parse-only Profiling pipeline yields
            # corpus text features with empty entity aggregates), so — unlike Relation
            # Extraction — a missing mention side is NOT an error. Any other carrier (a raw
            # ParserInput) is. #69: the doc/mention streams are read lazily from the parents'
            # checkpoints (the Profiler folds one at a time — the S7 streaming contract), so the
            # accumulator never materialises the whole doc-level output. A pre-pass validates the
            # wiring (below) before the provider consumes the streams.
            self._require_profiling_inputs_streamed(
                stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            profile_docs = self._stream_inputs_of_type(
                stage, outputs, DocumentRecord, storage=storage, pipeline=pipeline, run_id=run_id
            )
            profile_mentions = self._stream_inputs_of_type(
                stage, outputs, EntityMention, storage=storage, pipeline=pipeline, run_id=run_id
            )
            return iter(profiler.profile(profile_docs, profile_mentions))

        if stage.capability is CapabilityKind.TYPE_CONSOLIDATION:
            type_consolidator = self._capability_provider(stage, provider, TypeConsolidator)
            # T3 (ADR-0054): Type Consolidation is the corpus-level TYPE-vocabulary resolver,
            # the type-side twin of Disambiguation. It is corpus-level and multi-input like
            # Disambiguation: an Entity-Extraction (or fused) parent supplies the EntityMentions
            # whose labels it canonicalizes (REQUIRED — the labels ARE the vocabulary), and an
            # optional Relation-Extraction (or fused) parent supplies the RelationMentions whose
            # labels it canonicalizes alongside them. It re-emits BOTH carriers, relabelled, so a
            # Disambiguation Stage wired to it reads canonical types — which matters because the
            # entity resolver votes a cluster's type from its members' labels and gates its
            # low-precision rungs on type compatibility.
            #
            # NOT streamed (unlike Profiling/Disambiguation): the Stage's contract is to return the
            # SAME records relabelled, so it holds them regardless; a lazy input would only move
            # the materialisation inside the Provider. The retained set is the mention set, which
            # the extraction Stages' own checkpoints already bound.
            self._require_type_consolidation_inputs_streamed(
                stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            tc_mentions = self._stream_inputs_of_type(
                stage, outputs, EntityMention, storage=storage, pipeline=pipeline, run_id=run_id
            )
            tc_relations = self._stream_inputs_of_type(
                stage, outputs, RelationMention, storage=storage, pipeline=pipeline, run_id=run_id
            )
            consolidated = type_consolidator.consolidate(tc_mentions, tc_relations)
            # The vocabulary is not a Record and never enters the DAG — it describes the corpus's
            # SCHEMA, not a span of a document. It is persisted beside the run's checkpoints as a
            # sidecar (ADR-0017: files are the source of truth), which is what makes it survive
            # the run and lets an incremental delta run seed from it (T4).
            self._persist_type_vocabulary(storage, pipeline, run_id, stage, consolidated.vocabulary)
            return chain(consolidated.mentions, consolidated.relations)

        if stage.capability is CapabilityKind.DISAMBIGUATION:
            disambiguator = self._capability_provider(stage, provider, Disambiguator)
            # Disambiguation is corpus-level and multi-input: it wires an Entity-Extraction
            # (or fused) parent for the EntityMentions it resolves + links (REQUIRED — the
            # mentions are what it disambiguates) and OPTIONALLY a Relation-Extraction (or
            # fused) parent for the RelationMentions it normalises onto the canonical
            # entities. A fused parent supplies both from one mixed checkpoint. The mention
            # side is required (unlike Profiling), so a non-empty input with no EntityMention
            # is a wiring error. #69: mentions + relations are streamed lazily from the
            # parents' checkpoints (a fused parent's mixed checkpoint is read twice, once per
            # carrier — each pass O(batch)), so the disambiguator never materialises the whole
            # mention/relation set. A pre-pass validates the wiring before the provider runs.
            self._require_disambiguation_inputs_streamed(
                stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            disambig_mentions = self._stream_inputs_of_type(
                stage, outputs, EntityMention, storage=storage, pipeline=pipeline, run_id=run_id
            )
            disambig_relations = self._stream_inputs_of_type(
                stage, outputs, RelationMention, storage=storage, pipeline=pipeline, run_id=run_id
            )
            # W1 rework (#94): thread the run's ChunkRecords so a context-embedding Disambiguator
            # can locate each mention's parent chunk and embed surface+surrounding-context (the pod
            # proved bare-surface ER over-merges distinct short surfaces). Chunks are gathered from
            # the WHOLE pipeline — not just this Stage's ``depends_on`` — because the Chunk Stage
            # is a mention/relation ancestor, not a direct Disambiguation parent, yet the Runner
            # already holds every Stage's output (live dict or checkpoint). Same #69 streaming
            # discipline as the mention/relation gather: chunks are yielded lazily from each Stage's
            # bounded output/checkpoint, so the Runner never materialises them; the cascade ignores
            # them (byte-identical), and a run with no Chunk Stage yields an empty stream. The
            # algorithmic-path Disambiguators accept ``chunks=()`` by default, so this is additive.
            disambig_chunks = self._stream_all_records_of_type(
                outputs, ChunkRecord, storage=storage, pipeline=pipeline, run_id=run_id
            )
            return iter(
                disambiguator.disambiguate(disambig_mentions, disambig_relations, disambig_chunks)
            )

        if stage.capability is CapabilityKind.GRAPH_ASSEMBLY:
            graph_assembler = self._capability_provider(stage, provider, GraphAssembler)
            # Graph Assembly is corpus-level: it consumes the whole run's DisambiguationRecords
            # (the canonical entities + normalized relations) and builds the knowledge graph —
            # nodes, edges, and per-edge Evidence. It depends on a Disambiguation parent; a
            # non-empty input carrying no DisambiguationRecord is a wiring error. #69: the
            # DisambiguationRecords are streamed from the parent (its bounded accumulator output,
            # or its checkpoint), validated by a pre-pass that mirrors ``_as_type`` +
            # ``_require_graph_inputs`` (any wrong carrier / an all-wrong non-empty input is a
            # typed wiring error) before the assembler folds the stream.
            self._require_graph_inputs_streamed(
                stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            graph_records = self._stream_inputs_of_type(
                stage,
                outputs,
                DisambiguationRecord,
                storage=storage,
                pipeline=pipeline,
                run_id=run_id,
            )
            return iter(graph_assembler.assemble(graph_records))

        if stage.capability is CapabilityKind.GRAPH_COMPLETION:
            graph_completer = self._capability_provider(stage, provider, GraphCompleter)
            # W2-linkpred: Graph Completion is the OPTIONAL post-Assembly link-prediction Stage. It
            # consumes the whole run's assembled GraphRecords (NODE + EDGE) from its Graph-Assembly
            # parent and yields ADDITIONAL PREDICTED EDGE-scope GraphRecords (marked
            # properties.inferred=True, honest model-derived Evidence), which thread downstream to
            # Export alongside the asserted graph — distinguished by properties.inferred, never
            # merged into the asserted edges. It depends_on Graph Assembly; a non-empty input
            # carrying no GraphRecord is a wiring error. Same #69 streaming discipline as assembly:
            # the GraphRecords are streamed from the parent (bounded output/checkpoint), validated
            # by a pre-pass, before the completer folds the stream. The Stage is genuinely optional
            # — a stack without it is byte-identical to today (no predicted edges appear).
            self._require_graph_completion_inputs_streamed(
                stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            assembled = self._stream_inputs_of_type(
                stage,
                outputs,
                GraphRecord,
                storage=storage,
                pipeline=pipeline,
                run_id=run_id,
            )
            return iter(graph_completer.complete(assembled))

        if stage.capability is CapabilityKind.CONTEXT_ENRICHMENT:
            context_enricher = self._capability_provider(stage, provider, ContextEnricher)
            # W13: Context Enrichment is the OPTIONAL post-Assembly Stage. It is a chunk→chunk
            # transform (it yields the SAME ChunkRecords with ``context_header`` populated, like
            # Schema Induction) but MULTI-INPUT: it depends_on the chunk stream (Chunk /
            # Content-Screening / Schema-Induction) AND Graph Assembly, so ``incoming`` carries both
            # the ChunkRecords it enriches (and passes through 1:1) and the assembled GraphRecords
            # (NODE + EDGE) it reads the KG from. Partition gathered inputs by carrier; a missing
            # side is a wiring error. The run's EntityMentions (a chunk's canonical entities) come
            # from the WHOLE pipeline — Entity Extraction is a graph ancestor, not a direct parent,
            # so this streams every Stage's output for them, exactly as Disambiguation streams the
            # run's ChunkRecords (#94). A run with no Entity-Extraction Stage yields no mentions ⇒
            # every chunk's ``context_header`` stays None (byte-identical to no-stage). The enriched
            # chunk stream threads downstream to the corpus Export, which prepends the header to the
            # EMBEDDING input only (the stored content stays byte-unchanged — ADR-0039).
            enrich_chunks = [r for r in gathered if isinstance(r, ChunkRecord)]
            enrich_graph = [r for r in gathered if isinstance(r, GraphRecord)]
            # W17 boundary: Context Enrichment is NOT a per-document transform — a chunk's
            # ``context_header`` summarizes the mutable assembled graph, so a reused document's
            # header can go stale when a delta changes the graph, AND its enriched chunks would
            # then flow to Export, colliding with the doc-level Export overlay. Rather than hazard a
            # fragile partial reuse, the DeltaRunner DISABLES cross-run reuse for a pipeline that
            # contains a context_enrichment Stage (:meth:`DeltaRunner._reuse_supported`), so the
            # overlay is always empty here and enrichment sees the full freshly-extracted corpus —
            # correct by construction. Combining reuse with enrichment is deferred to a later wave.
            self._require_context_enrichment_inputs(
                stage, gathered, enrich_chunks, enrich_graph
            )
            enrich_mentions = self._stream_all_records_of_type(
                outputs, EntityMention, storage=storage, pipeline=pipeline, run_id=run_id
            )
            return iter(context_enricher.enrich(enrich_chunks, enrich_mentions, enrich_graph))

        if stage.capability is CapabilityKind.EXPORT:
            exporter = self._capability_provider(stage, provider, Export)
            # Export materializes any inter-Stage Record: DocumentRecords (S1/S2
            # spine) or ChunkRecords (S3, when a Chunk Stage sits upstream). Only a
            # raw ParserInput (not a Record) is an invalid Export input.
            exportable = self._as_records(stage, gathered)
            # W17: a DOC-LEVEL Export (its carrier is a per-document record — e.g. a chunk/redact
            # RAG corpus) materializes the reused documents' records too, so the exported corpus
            # is the whole corpus (fresh ∪ reused), not only the changed documents. A corpus-level
            # Export (carrier DisambiguationRecord/GraphRecord) has no overlay of that carrier, so
            # this is a no-op there — the union already arrived through the corpus-level Stages.
            export_carrier = self._export_carrier(stage, pipeline)
            overlay_flat = [
                record
                for records in self._reuse_overlay_by_type.values()
                for record in records
            ]
            extra: list[Record]
            if export_carrier is Record:
                # A fused Entity+Relation ancestor: the pass-through carrier is the mixed
                # mention/relation stream, so both reused mention carriers are materialized.
                extra = [r for r in overlay_flat if type(r) in {EntityMention, RelationMention}]
            else:
                extra = [r for r in overlay_flat if isinstance(r, export_carrier)]
            materialized: list[Record] = [*exportable, *extra]
            out_dir = self._export_dir(pipeline, run_id)
            exporter.export(materialized, storage, out_dir)
            # Export is a sink: it passes its input records through unchanged. (The overlay records
            # are exportable Records, so the cast to the StageOutput carrier is exact.)
            return iter(cast("list[StageOutput]", materialized))

        if stage.capability is CapabilityKind.EMBEDDING:
            # Embedding is an opt-in *Export augmentation* (ADR-0017), not a standalone DAG node:
            # a corpus Export names its Embedder in config['embedder'] and attaches a vector per
            # exported row. There is no threaded EmbeddingRecord, so a bare `embedding` Stage has
            # no carrier and is a wiring error — surface the fix, don't fall through to the opaque
            # unknown-capability error.
            msg = (
                f"Stage {stage.name!r} declares the 'embedding' Capability as a standalone DAG "
                f"Stage, but Embedding is an opt-in Export augmentation, not a DAG node "
                f"(ADR-0017). Remove this Stage and name the Embedder in the corpus Export's "
                f"config['embedder'] = {{'provider': {stage.provider!r}, 'config': {{...}}}}."
            )
            # Embedding-as-a-standalone-Stage is a malformed-Pipeline config error (#73);
            # ConfigError is a ValueError subclass, so the pre-taxonomy ``except ValueError``
            # callers (and the standalone-embedding-rejection test) keep catching it.
            raise ConfigError(msg)

        msg = f"Unknown capability {stage.capability!r} for stage {stage.name!r}."
        # An unknown capability is a malformed-Pipeline config error (#73), formerly a ValueError.
        raise ConfigError(msg)

    # -- Type-vocabulary sidecar (T3, off the DAG) --------------------------

    def _type_vocabulary_uri(self, pipeline: Pipeline, run_id: str, stage_name: str) -> str:
        return f"{self._run_dir(pipeline, run_id)}/type-vocabulary/{stage_name}.json"

    def _persist_type_vocabulary(
        self,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        stage: Stage,
        vocabulary: TypeVocabulary | None,
    ) -> None:
        """Persist a Type-Consolidation Stage's canonical vocabulary for audit + delta reuse (T3).

        Written even when the run consolidated nothing (an empty vocabulary is the honest answer
        to "what did this Stage decide", not a missing file), so a reader never has to
        distinguish "no file" from "no consolidation".
        """
        payload = (vocabulary if vocabulary is not None else TypeVocabulary()).model_dump_json(
            indent=2
        )
        storage.write_atomic(
            self._type_vocabulary_uri(pipeline, run_id, stage.name), payload.encode("utf-8")
        )

    def read_type_vocabulary(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> TypeVocabulary | None:
        """Read back a run's persisted canonical type vocabulary; ``None`` when none was written.

        The seam an incremental delta run (T4) reads its predecessor's vocabulary through before
        seeding the next consolidation — so a delta aliases a new type INTO an existing canonical
        cluster instead of re-electing (and thereby renaming) one.
        """
        uri = self._type_vocabulary_uri(pipeline, run_id, stage_name)
        if not storage.exists(uri):
            return None
        return TypeVocabulary.model_validate_json(storage.read_text(uri))

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

    def read_findings(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[ScreeningFinding]:
        """Read a Screening Stage's persisted findings sidecar (empty if none written).

        Public because it is one half of the
        :class:`~latence_core.quality_report_builder.StageRecordReader` seam the shared
        :class:`~latence_core.quality_report_builder.QualityReportBuilder` reads to build the
        Screening rollup — the report builder no longer reaches into a private method for it
        (H-F1). The executor's own screening path uses the same reader.
        """
        uri = self._findings_uri(pipeline, run_id, stage_name)
        if not storage.exists(uri):
            return []
        return [
            ScreeningFinding.model_validate_json(line)
            for line in storage.read_text(uri).splitlines()
            if line.strip()
        ]

    def stage_records(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        """Public reader half of the report-builder seam — delegates to :meth:`_stage_records`.

        The shared :class:`~latence_core.quality_report_builder.QualityReportBuilder` (and a second
        Runner reusing it) reads each Stage's records back through this public method instead of the
        private ``_stage_records`` — the report concern is a real, relocatable unit that composes
        with the engine through its interface, never a ``# noqa: SLF001`` reach-through (H-F1).
        """
        yield from self._stage_records(
            stage_name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )

    def _stage_records(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        """Yield one Stage's output — from the live dict if retained, else its checkpoint (#69).

        A doc-level Stage's output is released from ``outputs`` once its last consumer ran, so a
        later reader (an accumulator, the Quality Report) re-reads it lazily from the on-disk
        checkpoint here, batch-by-batch, instead of the executor holding it in memory. A retained
        entry (a live doc-level parent, or an accumulator's bounded output) is yielded directly.
        A Stage that never produced a checkpoint (not yet run, or a genuinely empty output) yields
        nothing. The streamed and the live path yield byte-identical records in identical order.
        """
        if stage_name in outputs:
            yield from outputs[stage_name]
            return
        by_name = {s.name: s for s in pipeline.stages}
        stage = by_name.get(stage_name)
        if stage is None:  # pragma: no cover - names come from the validated DAG
            return
        ckpt_uri = self._checkpoint_uri(pipeline, run_id, stage_name)
        if not storage.exists(ckpt_uri):
            return
        yield from self._stream_checkpoint(
            storage, ckpt_uri, self._output_type(stage, pipeline)
        )

    def _overlay_of_type(self, expected: type[_T]) -> list[_T]:
        """The W17 reuse overlay's records of one carrier type (empty when no delta reuse).

        The reused doc-level records for the carrier ``expected`` — appended AFTER a corpus-level
        Stage's checkpoint inputs so the Stage sees ``fresh ∪ reused``. In the overlay's stored
        (id-sorted) order; because every corpus-level accumulator that consumes them folds in
        sorted ``mention_id`` / ``record_id`` order, the append position does not perturb the output
        (ADR-0043 determinism). Empty (no reuse) ⇒ a no-op and the stream is exactly the checkpoint
        records — byte-identical to a non-delta run.
        """
        return self._reuse_overlay_by_type.get(expected, [])  # type: ignore[return-value]

    @staticmethod
    def _parent_reemits_overlay(stage: Stage, pipeline: Pipeline, expected: type[_T]) -> bool:
        """True when a parent Stage has ALREADY folded the reuse overlay into this carrier (T4).

        The overlay must reach each corpus-level accumulator exactly once. Almost every
        corpus-level Stage transforms its input into a DIFFERENT carrier (Disambiguation emits
        DisambiguationRecords, Graph Assembly GraphRecords), so appending the reused mentions to
        its stream is the only way it sees them. **Type Consolidation is the exception**: it is a
        corpus-level accumulator whose output carrier IS its input carrier — it consumes
        ``fresh ∪ overlay`` and re-emits all of it, relabelled. A downstream Stage that then
        appended the overlay AGAIN would see every reused mention twice, which surfaces as a
        ``CanonicalEntity`` whose ``member_mention_ids`` are duplicated (the resolver's
        no-double-counting contract) — a delta that crashes rather than silently mis-counts, but a
        crash all the same. So the overlay is suppressed for a carrier a parent already re-emitted.
        """
        if expected not in (EntityMention, RelationMention):
            return False
        by_name = {s.name: s for s in pipeline.stages}
        return any(
            (parent := by_name.get(name)) is not None
            and parent.capability is CapabilityKind.TYPE_CONSOLIDATION
            for name in stage.depends_on
        )

    def _stream_inputs_of_type(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        expected: type[_T],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[_T]:
        """Lazily yield a Stage's gathered inputs narrowed to one carrier type (#69).

        The streaming companion to :meth:`_gather_inputs` for the corpus-level accumulators
        (Profiling/Disambiguation/Graph): it re-reads each parent's output from the live dict or
        its checkpoint (:meth:`_stage_records`) one record at a time and yields only the
        ``expected`` carrier, so an accumulator that needs the whole doc-level stream folds it in
        without the executor materialising the stream into a retained list. A checkpoint is
        re-readable, so calling this once per carrier (Disambiguation needs both mentions and
        relations) keeps each pass O(batch), not O(all records). Parents are read in declared
        ``depends_on`` order, so the yielded order is byte-identical to :meth:`_gather_inputs`
        followed by an ``isinstance`` filter — determinism preserved.
        """
        for parent in stage.depends_on:
            for record in self._stage_records(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                if isinstance(record, expected):
                    yield record
        # W17: append the reused doc-level records of this carrier so the accumulator folds in the
        # unchanged documents' Stage outputs alongside the freshly-extracted ones (fresh ∪ reused)
        # — unless a parent already re-emitted them (:meth:`_parent_reemits_overlay`, T4).
        if not self._parent_reemits_overlay(stage, pipeline, expected):
            yield from self._overlay_of_type(expected)

    def _stream_all_records_of_type(
        self,
        outputs: dict[str, list[StageOutput]],
        expected: type[_T],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[_T]:
        """Lazily yield every Stage's output narrowed to one carrier — corpus-wide (W1 rework #94).

        A corpus-level Stage sometimes needs a carrier that is NOT one of its direct
        ``depends_on`` parents. Disambiguation depends on the Entity-Extraction (mentions) and
        optional Relation-Extraction (relations) parents, but a context-embedding Disambiguator
        also needs the run's :class:`~latence_core.contracts.ChunkRecord`s — produced by the
        Chunk Stage, a *mention ancestor*, not a Disambiguation parent. This streams every
        Stage's output in topological order (the same order :meth:`_gather_inputs` reads parents,
        so it is deterministic) from the live dict or its checkpoint (:meth:`_stage_records`), one
        record at a time, yielding only the ``expected`` carrier — so the accumulator folds the
        corpus-wide stream in without the executor materialising it (#69 memory bound preserved).
        A run with no producing Stage yields nothing (the empty-chunks fallback).
        """
        for stage in topological_order(list(pipeline.stages)):
            for record in self._stage_records(
                stage.name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                if isinstance(record, expected):
                    yield record
        # W17: the corpus-wide gather (e.g. Disambiguation's ChunkRecord stream, Context
        # Enrichment's EntityMention stream) also spans the reused documents' records.
        yield from self._overlay_of_type(expected)

    def _gathered_count(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> int:
        """Count a Stage's gathered inputs by streaming — never materialising them (#69).

        The ``records_in`` metric for a corpus-level accumulator (whose input is the whole
        doc-level output). Streams each parent's records one at a time (:meth:`_stage_records`)
        and counts, so the count is exact without holding the input as a list — the memory bound
        the ADR requires for the accumulator input side.
        """
        total = 0
        by_name = {s.name: s for s in pipeline.stages}
        for parent in stage.depends_on:
            for _ in self._stage_records(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                total += 1
            # W17: the reused documents' records of this parent's carrier are folded in too, so
            # ``records_in`` reports the whole corpus the accumulator actually consumed.
            parent_stage = by_name.get(parent)
            if parent_stage is not None:
                total += len(self._reuse_overlay_by_type.get(
                    self._output_type(parent_stage, pipeline), []
                ))
        return total

    def _gather_inputs(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> list[StageOutput]:
        """A Stage's input = the concatenation of ALL its parents' outputs.

        Parents are read in declared ``depends_on`` order for determinism. This
        is what makes a converging DAG ({a, b} -> c) correct and multi-input:
        c sees a's records followed by b's — never a single overwritten variable,
        and always the parents' *current* outputs. A parent whose retained output was
        already released (#69) is streamed back from its checkpoint (:meth:`_stage_records`),
        so gathering is release-transparent and byte-identical to reading the live dict.
        """
        gathered: list[StageOutput] = []
        for parent in stage.depends_on:
            gathered.extend(
                self._stage_records(
                    parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            )
        return gathered

    def _stream_doc_inputs(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        """The lazy form of :meth:`_gather_inputs` for a doc-level Stage (audit R2, ADR-0033).

        Same records, same order — parents read in declared ``depends_on`` order, each from the
        live dict if still retained and otherwise streamed back from its checkpoint
        (:meth:`_stage_records`) — but nothing is concatenated into a list. The Stage's Provider
        pulls one record at a time, so the whole input side of a doc-level Stage costs O(1)
        records instead of O(corpus-slice). Deterministic and byte-identical to the gathered form:
        the two differ only in what memory is held while the Stage runs.

        Overlay-blind, exactly like :meth:`_gather_inputs`: the W17 reuse overlay is folded in for
        the corpus-level accumulators only (:meth:`_stream_inputs_of_type`), never here.
        """
        for parent in stage.depends_on:
            yield from self._stage_records(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )

    def _streamed_input_count(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> int:
        """``records_in`` for a streamed doc-level Stage — counted, never materialised (audit R2).

        The metric the span, the lineage START event and the ``StageMetrics`` row all report is
        needed BEFORE the Stage executes, so it cannot be a tally taken while the stream flows.
        Counting a retained parent is a ``len``; counting a released one counts the checkpoint's
        non-blank LINES (:meth:`_stage_record_count`) — one record per line by construction of
        :meth:`_write_checkpoint`, so the count is exact without decoding (and therefore without
        re-validating) a single record. O(1) memory, and no second Pydantic pass over the corpus.
        """
        return sum(
            self._stage_record_count(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            )
            for parent in stage.depends_on
        )

    def _stage_record_count(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> int:
        """How many records one Stage produced — from the live dict, else its checkpoint's lines.

        The counting companion to :meth:`_stage_records`. A checkpoint is newline-delimited
        ``model_dump_json`` (one record per non-blank line, blank lines skipped on read), so
        counting lines equals counting decoded records — including a fused Stage's mixed-carrier
        checkpoint, where each line is still exactly one record. A Stage with no checkpoint (not
        run, or genuinely empty) counts 0.
        """
        if stage_name in outputs:
            return len(outputs[stage_name])
        ckpt_uri = self._checkpoint_uri(pipeline, run_id, stage_name)
        if not storage.exists(ckpt_uri):
            return 0
        total = 0
        with storage.open(ckpt_uri, "rb") as fh:
            for raw_line in fh:
                if raw_line.strip():
                    total += 1
        return total

    @staticmethod
    def _extract_input_chunks_streamed(
        pipeline: Pipeline, chunks: Iterable[ChunkRecord]
    ) -> Iterator[ChunkRecord]:
        """The lazy form of :meth:`_extract_input_chunks` (audit R2, W18/ADR-0044).

        Same records, one at a time. This is the transient the audit found doubling (and, in
        ``masked`` mode, tripling with fresh ``model_copy`` objects) the already corpus-sized
        input list: an Entity Stage held the gathered list, its narrowed copy AND the masked copy
        simultaneously. Yielding keeps at most one masked copy alive at a time, and the masking
        rule is byte-for-byte the eager one.
        """
        masked = pipeline.extract_on is ExtractOn.MASKED
        for chunk in chunks:
            if masked and chunk.masked_content is not None:
                yield chunk.model_copy(update={"content": chunk.masked_content})
            else:
                yield chunk

    @staticmethod
    def _extract_input_chunks(
        pipeline: Pipeline, chunks: list[ChunkRecord]
    ) -> list[ChunkRecord]:
        """The chunk-text variant the extraction Stages read (W18, ADR-0044).

        Default (``extract_on: unmasked``) returns the chunks UNCHANGED — byte-identical to pre-W18,
        the KG keeps the real entities. In ``masked`` mode each chunk's ``content`` is replaced by
        its ``masked_content`` (the redacted variant produced by the upstream redaction Stage the
        mode's validator requires), so the extractor — which reads ``content`` — never sees PII and
        it never reaches the KG. A chunk without ``masked_content`` (e.g. redaction skipped it)
        falls back to its own ``content``, so masked mode never fabricates or drops text.
        """
        if pipeline.extract_on is not ExtractOn.MASKED:
            return chunks
        return [
            c.model_copy(update={"content": c.masked_content})
            if c.masked_content is not None
            else c
            for c in chunks
        ]

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
            raise ContractError(msg)
        return [r for r in records if isinstance(r, expected)]

    @staticmethod
    def _wrong_carrier_error(stage: Stage, expected: type[_T], got: Iterable[str]) -> ContractError:
        """THE mis-wired-carrier message, shared by the gathered and the streamed narrowing."""
        return ContractError(
            f"Stage {stage.name!r} ({stage.capability.value}) requires "
            f"{expected.__name__} inputs but received: {', '.join(sorted(set(got)))}. "
            f"Check the Pipeline's depends_on wiring."
        )

    @classmethod
    def _as_type_stream(
        cls,
        stage: Stage,
        records: Iterable[StageOutput],
        expected: type[_T],
        *,
        message: Callable[[str], str] | None = None,
    ) -> Iterator[_T]:
        """The lazy form of :meth:`_as_type` — narrow a streamed input, one record at a time.

        Identical contract: every record must be the carrier the Stage's capability requires, and
        the first one that is not raises the SAME typed :class:`ContractError` naming the wrong
        carrier. Identical outcome too — :meth:`_as_type` raises when *any* record is wrong, so
        for every reachable input (all-right, all-wrong, or mixed) both forms accept or reject the
        same streams. What differs is *when*: the typed error surfaces at the first offending
        record rather than before the Provider is called. That is still strictly before the
        Stage's checkpoint is published (it is written temp→rename after the stream is drained),
        so a mis-wired Pipeline still fails loudly with no torn commit (ADR-0034) — it just does
        not pay an O(corpus) buffer for the privilege of failing a few microseconds earlier.

        ``message`` overrides the generic wiring text for a Capability that has more specific
        advice to give (Redaction's W16 chunk→chunk rule); it receives the offending carrier's
        type name.
        """
        for record in records:
            if not isinstance(record, expected):
                got = type(record).__name__
                if message is not None:
                    raise ContractError(message(got))
                raise cls._wrong_carrier_error(stage, expected, [got])
            yield record

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
            raise ContractError(msg)
        if not chunks:
            msg = (
                f"Stage {stage.name!r} (relation_extraction) received no ChunkRecords. "
                f"A Relation-Extraction Stage must also depend on the Chunk (or "
                f"Content-Screening) Stage for the text + page map. Check depends_on."
            )
            raise ContractError(msg)

    @staticmethod
    def _require_context_enrichment_inputs(
        stage: Stage,
        incoming: list[StageOutput],
        chunks: list[ChunkRecord],
        graph_records: list[GraphRecord],
    ) -> None:
        """A Context-Enrichment Stage must be wired to BOTH the chunk stream and Graph Assembly
        (W13).

        It depends on a Chunk / Content-Screening / Schema-Induction parent (the ChunkRecords it
        enriches + passes through) AND a Graph-Assembly parent (the assembled GraphRecords it reads
        the KG from). A missing side means the Pipeline forgot a ``depends_on`` edge — surface it as
        a
        clear typed error, not a silently-empty header. An empty run (no documents ⇒ no chunks and
        no
        graph) is legitimately empty. (The run's EntityMentions are gathered corpus-wide, not from a
        direct parent, so they are validated only by their own upstream Stages — a run with none
        just
        yields no headers.)
        """
        if not incoming:
            return
        if not chunks:
            got = sorted({type(r).__name__ for r in incoming})
            msg = (
                f"Stage {stage.name!r} (context_enrichment) received no ChunkRecords "
                f"(got: {', '.join(got)}). A Context-Enrichment Stage must depend on the Chunk "
                f"(or Content-Screening / Schema-Induction) Stage — the chunks it enriches. "
                f"Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)
        if not graph_records:
            got = sorted({type(r).__name__ for r in incoming})
            msg = (
                f"Stage {stage.name!r} (context_enrichment) received no GraphRecords "
                f"(got: {', '.join(got)}). A Context-Enrichment Stage must also depend on the "
                f"Graph Assembly Stage — the assembled KG it projects onto each chunk. "
                f"Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    @staticmethod
    def _require_profiling_inputs(
        stage: Stage, incoming: list[StageOutput], documents: list[DocumentRecord]
    ) -> None:
        """A Profiling Stage must be wired to a Parse parent (the documents it profiles).

        Profiling depends on a Parse parent for the DocumentRecords (per-document
        features + the corpus text aggregates). The EntityMention side is optional (a
        Parse-only Profiling pipeline is valid — it yields text features with empty
        entity aggregates), so it is not checked here. A non-empty input that carries no
        DocumentRecord means the Pipeline wired Profiling to the wrong parent (e.g. an
        Entity-Extraction Stage with no Parse) — surface it as a clear typed error. An
        empty run (no documents) is legitimately empty.
        """
        if not incoming:
            return
        if not documents:
            got = sorted({type(r).__name__ for r in incoming})
            msg = (
                f"Stage {stage.name!r} (profiling) received no DocumentRecords "
                f"(got: {', '.join(got)}). A Profiling Stage must depend on the Parse "
                f"Stage (the documents it profiles). Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    @staticmethod
    def _require_disambiguation_inputs(
        stage: Stage, incoming: list[StageOutput], mentions: list[EntityMention]
    ) -> None:
        """A Disambiguation Stage must be wired to an Entity-Extraction parent (the mentions).

        Disambiguation depends on an Entity-Extraction (or fused) parent for the
        EntityMentions it resolves and links — the mentions are REQUIRED (unlike Profiling's
        optional mentions: with no mentions there is nothing to disambiguate). The
        RelationMention side is optional (an entity-only Disambiguation pipeline is valid), so
        it is not checked here. A non-empty input that carries no EntityMention means the
        Pipeline wired Disambiguation to the wrong parent — surface it as a clear typed error.
        An empty run (no documents ⇒ no mentions) is legitimately empty.
        """
        if not incoming:
            return
        if not mentions:
            got = sorted({type(r).__name__ for r in incoming})
            msg = (
                f"Stage {stage.name!r} (disambiguation) received no EntityMentions "
                f"(got: {', '.join(got)}). A Disambiguation Stage must depend on an "
                f"Entity-Extraction (or fused) Stage (the mentions it resolves). Check "
                f"the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    @staticmethod
    def _require_graph_inputs(
        stage: Stage, incoming: list[StageOutput], records: list[DisambiguationRecord]
    ) -> None:
        """A Graph Assembly Stage must be wired to a Disambiguation parent (the entities/relations).

        Graph Assembly depends on a Disambiguation parent for the DisambiguationRecords (the
        canonical entities + normalized relations it turns into nodes + edges). A non-empty
        input that carries no DisambiguationRecord means the Pipeline wired Graph Assembly to
        the wrong parent — surface it as a clear typed error. An empty run (no documents ⇒ no
        entities) is legitimately empty. (``_as_type`` already rejects a mixed/wrong carrier;
        this guards the all-wrong case with a Stage-specific message.)
        """
        if not incoming:
            return
        if not records:  # pragma: no cover - _as_type raises first on a wrong carrier
            got = sorted({type(r).__name__ for r in incoming})
            msg = (
                f"Stage {stage.name!r} (graph_assembly) received no DisambiguationRecords "
                f"(got: {', '.join(got)}). A Graph Assembly Stage must depend on the "
                f"Disambiguation Stage (the canonical entities + relations it assembles). "
                f"Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    def _scan_input_carriers(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> tuple[set[type[StageOutput]], set[str]]:
        """Stream a Stage's gathered inputs once, returning the carrier types present (#69).

        The streaming replacement for materialising ``incoming`` just to validate an accumulator's
        wiring: it reads each parent's records one at a time (:meth:`_stage_records`) and collects
        only the set of carrier *types* (and their names) seen — O(distinct-carriers) memory, not
        O(records). The three streamed ``_require_*`` checks below use this so a mis-wired
        accumulator still raises the same typed error, without holding the input list.
        """
        present: set[type[StageOutput]] = set()
        names: set[str] = set()
        by_name = {s.name: s for s in pipeline.stages}
        for parent in stage.depends_on:
            for record in self._stage_records(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                present.add(type(record))
                names.add(type(record).__name__)
            # W17: a parent whose fresh output is empty because every one of its documents was
            # reused still "produces" that carrier via the overlay — so the wiring check sees the
            # carrier and an all-reused delta is not misread as a mis-wired empty parent.
            parent_stage = by_name.get(parent)
            if parent_stage is not None:
                parent_type = self._output_type(parent_stage, pipeline)
                if self._reuse_overlay_by_type.get(parent_type):
                    present.add(parent_type)  # type: ignore[arg-type]
                    names.add(parent_type.__name__)
        return present, names

    def _require_profiling_inputs_streamed(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> None:
        """Streaming form of :meth:`_require_profiling_inputs` (#69) — same typed error."""
        present, names = self._scan_input_carriers(
            stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        if not present:
            return
        if DocumentRecord not in present:
            got = sorted(names)
            msg = (
                f"Stage {stage.name!r} (profiling) received no DocumentRecords "
                f"(got: {', '.join(got)}). A Profiling Stage must depend on the Parse "
                f"Stage (the documents it profiles). Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    def _require_disambiguation_inputs_streamed(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> None:
        """Streaming form of :meth:`_require_disambiguation_inputs` (#69) — same typed error."""
        present, names = self._scan_input_carriers(
            stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        if not present:
            return
        if EntityMention not in present:
            got = sorted(names)
            msg = (
                f"Stage {stage.name!r} (disambiguation) received no EntityMentions "
                f"(got: {', '.join(got)}). A Disambiguation Stage must depend on an "
                f"Entity-Extraction (or fused) Stage (the mentions it resolves). Check "
                f"the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    def _require_type_consolidation_inputs_streamed(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> None:
        """A Type-Consolidation Stage must be wired to an Entity-Extraction parent (T3).

        The mentions' labels ARE the vocabulary it canonicalizes, so the mention side is REQUIRED
        (the relation side is optional — an entity-only stack is valid). A non-empty input that
        carries no EntityMention means the Pipeline wired the Stage to the wrong parent; that is a
        typed wiring error, never a silent pass-through that would leave the KG's types drifted
        while the report claimed consolidation ran (ADR-0034). An empty run is legitimately empty.
        """
        present, names = self._scan_input_carriers(
            stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        if not present:
            return
        if EntityMention not in present:
            got = sorted(names)
            msg = (
                f"Stage {stage.name!r} (type_consolidation) received no EntityMentions "
                f"(got: {', '.join(got)}). A Type-Consolidation Stage must depend on an "
                f"Entity-Extraction (or fused) Stage (the mentions whose types it "
                f"canonicalizes). Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    def _require_graph_inputs_streamed(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> None:
        """Streaming form of :meth:`_as_type` + :meth:`_require_graph_inputs` (#69).

        Graph Assembly's input must be entirely DisambiguationRecords (``_as_type`` rejected any
        other carrier); an empty run is legitimately empty. Preserves both the wrong-carrier and
        the all-wrong-non-empty typed errors while streaming — the wrong-carrier set is bounded,
        never the record list.
        """
        present, names = self._scan_input_carriers(
            stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        if not present:
            return
        wrong = sorted(t.__name__ for t in present if t is not DisambiguationRecord)
        if wrong:
            # Mirror ``_as_type``'s exact message (the check this streamed path replaces).
            msg = (
                f"Stage {stage.name!r} ({stage.capability.value}) requires "
                f"DisambiguationRecord inputs but received: {', '.join(wrong)}. "
                f"Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    def _require_graph_completion_inputs_streamed(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> None:
        """A Graph Completion Stage must be wired to a Graph-Assembly parent (W2-linkpred).

        Graph Completion consumes the assembled graph's GraphRecords (NODE + EDGE). Its input must
        be entirely GraphRecords; a non-empty input carrying any other carrier means the Pipeline
        wired Graph Completion to the wrong parent (it must depend on the Graph Assembly Stage, not
        Disambiguation or anything else). An empty run (no documents ⇒ no graph) is legitimately
        empty. Streams the carrier types once (bounded), never the record list (#69).
        """
        present, names = self._scan_input_carriers(
            stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
        if not present:
            return
        wrong = sorted(t.__name__ for t in present if t is not GraphRecord)
        if wrong:
            msg = (
                f"Stage {stage.name!r} (graph_completion) requires GraphRecord inputs "
                f"but received: {', '.join(wrong)}. A Graph Completion Stage must depend on "
                f"the Graph Assembly Stage (the assembled nodes + edges it completes). "
                f"Check the Pipeline's depends_on wiring."
            )
            raise ContractError(msg)

    @staticmethod
    def _as_records(
        stage: Stage, records: list[StageOutput]
    ) -> list[
        DocumentRecord
        | ChunkRecord
        | EntityMention
        | RelationMention
        | RedactionRecord
        | FeatureRecord
        | DisambiguationRecord
        | GraphRecord
    ]:
        """Narrow gathered inputs to the inter-Stage Records Export materializes.

        Export materializes DocumentRecords (S1/S2 spine), ChunkRecords (S3, when a
        Chunk Stage is upstream), EntityMentions (S4, an Entity-Extraction Stage
        upstream), RelationMentions (S5, a Relation-Extraction / fused Stage upstream),
        RedactionRecords (S6, a Redaction Stage upstream — the PII-handled corpus
        variant), FeatureRecords (S7, a Profiling Stage upstream — the corpus
        statistical/quality feature set), DisambiguationRecords (S8, a Disambiguation
        Stage upstream — the canonical entities + normalized relations), or GraphRecords (S9,
        a Graph Assembly Stage upstream — the knowledge-graph nodes + edges). A fused Stage
        feeds a MIX of mentions and relations — both are Records and both export. A raw
        ParserInput (not a Record) means a Source was wired straight into Export with no Parse
        between. Surface it as a clear typed error.
        """
        exportable = (
            DocumentRecord,
            ChunkRecord,
            EntityMention,
            RelationMention,
            RedactionRecord,
            FeatureRecord,
            DisambiguationRecord,
            GraphRecord,
        )
        wrong = [r for r in records if not isinstance(r, exportable)]
        if wrong:
            got = sorted({type(r).__name__ for r in wrong})
            msg = (
                f"Stage {stage.name!r} (export) requires Record inputs "
                f"(DocumentRecord/ChunkRecord/EntityMention/RelationMention/"
                f"RedactionRecord/FeatureRecord/DisambiguationRecord/GraphRecord) but received: "
                f"{', '.join(got)}. An Export needs a Parse (and optionally a Chunk / "
                f"Entity-Extraction / Relation-Extraction / Redaction / Profiling / "
                f"Disambiguation / Graph-Assembly) upstream."
            )
            raise ContractError(msg)
        return [r for r in records if isinstance(r, exportable)]

    # -- Quality Report (delegated to the extracted builder, H-F1) ----------

    def _final_records(
        self,
        ordered: list[Stage],
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        """Stream the DAG's sink-Stage records for the Quality Report (delegates to the builder).

        Kept as a thin shim over :meth:`QualityReportBuilder.final_records
        <latence_core.quality_report_builder.QualityReportBuilder.final_records>` so the executor's
        own call sites (the end-of-run build and the mid-run FAIL-report build) read the same way;
        the report logic itself now lives in the extracted builder (H-F1).

        Lazy: the sinks are streamed into the report's completeness fold one record at a time, never
        materialised, so the report phase costs ``O(1)`` in the corpus after every Stage has already
        been streamed and released (#69, ADR-0033).
        """
        return self._report_builder.final_records(
            ordered, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )

    def _build_report(
        self,
        pipeline: Pipeline,
        run_id: str,
        storage: Storage,
        ordered: list[Stage],
        stage_metrics: list[StageMetrics],
        outputs: Mapping[str, list[StageOutput]],
        final_records: Iterable[Record | StageOutput],
    ) -> QualityReport:
        """Build the run's Quality Report (delegates to the extracted builder, H-F1).

        A thin shim over :meth:`QualityReportBuilder.build
        <latence_core.quality_report_builder.QualityReportBuilder.build>` for the executor's own
        report writes. A second Runner does **not** call this private method — it constructs the
        public :class:`~latence_core.quality_report_builder.QualityReportBuilder` and calls
        ``build`` directly, so no Runner reaches past the interface (the audit's THEME F fix).
        """
        return self._report_builder.build(
            pipeline, run_id, storage, ordered, stage_metrics, outputs, final_records
        )

    @staticmethod
    def _drift_diagnostics(chunked: list[StageOutput]) -> DriftDiagnostics | None:
        """Provenance-integrity drift roll-up over chunk offsets (moved to the builder, H-F1).

        Retained as a static delegator on ``LocalRunner`` because the report's drift audit is
        exercised directly as ``LocalRunner._drift_diagnostics([...])`` by the existing
        DriftDiagnostics regression tests (issue #39.1). The logic itself now lives in
        ``QualityReportBuilder._drift_diagnostics`` (H-F1); this forwards to it verbatim so those
        tests keep passing unchanged (byte-identical behaviour).
        """
        return QualityReportBuilder._drift_diagnostics(chunked)  # noqa: SLF001

