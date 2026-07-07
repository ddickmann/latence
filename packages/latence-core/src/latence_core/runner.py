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

import hashlib
import json
import time
from typing import TYPE_CHECKING, TypeVar

from latence_core.capability import (
    Chunker,
    ContentScreener,
    Disambiguator,
    EntityExtractor,
    Export,
    FusedEntityRelationExtractor,
    GraphAssembler,
    IntakeScreener,
    Parser,
    PIIDetector,
    Profiler,
    ProviderRegistry,
    RelationExtractor,
    ScreeningOutcome,
    Source,
)
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
    ParserInput,
    QuarantineRecord,
    Record,
    RedactionRecord,
    RelationMention,
    ScreeningCheckpoint,
    ScreeningFinding,
)
from latence_core.errors import ConfigError, ContractError, error_category
from latence_core.goldset import (
    ConfusionCounts,
    GoldSet,
    score_entities,
    score_relations,
)
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.providers.device import DeviceDecision, select_device
from latence_core.providers.profile import ProviderProfile, profile_of
from latence_core.quality import (
    ChunkQuality,
    CompletenessCheck,
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
from latence_core.storage import Storage
from latence_core.tracing import RunTracer, SpanSink

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# A Stage's output is one of the inter-Stage carriers the DAG threads between
# Stages: Source→ParserInput; Parse→DocumentRecord; Chunk/Content-Screening→
# ChunkRecord; Entity-Extraction→EntityMention; Relation-Extraction→RelationMention.
# A *fused* Entity+Relation Stage (ADR-0013) emits BOTH EntityMention and
# RelationMention in one output list (mixed carrier), routed to both downstream
# consumers without a separate Entity-Extraction Stage. The Runner keeps outputs in
# one per-Stage dict and lets each Stage's capability decide how to (de)serialize its
# checkpoint. Redaction→RedactionRecord (one PII-handled variant per document).
# Profiling→FeatureRecord (per-document + one corpus-scope record). Screening ALSO
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

# The corpus-level accumulator Capabilities (ADR-0033): Profiling, Disambiguation, Graph
# Assembly. They inherently consume the whole doc-level output, but their own output is bounded
# by the knowledge-graph size (canonical entities + nodes/edges + corpus features), not the
# corpus size. Everything else is a per-document streaming transform whose retained output list
# is released once its consumers have run (#69), so the executor no longer holds every Stage's
# doc-level records simultaneously.
_ACCUMULATOR_KINDS = (
    CapabilityKind.PROFILING,
    CapabilityKind.DISAMBIGUATION,
    CapabilityKind.GRAPH_ASSEMBLY,
)

# Secret-handling (issue #30): the run manifest persists the whole Pipeline
# (including every Stage's operator ``config``) verbatim. A Provider such as the S5
# relation.llm reads its ``api_key`` from ``$OPENAI_API_KEY`` (env fallback, no key
# in code/logs — the Baseline), but an operator who *inlines* ``api_key`` into the
# Pipeline YAML would otherwise write it cleartext into ``manifest.json``. Any config
# key whose (lowercased) name contains one of these tokens is redacted before the
# manifest is written. Matching is substring-based so ``api_key``, ``openai_api_key``,
# ``auth_token``, ``client_secret``, ``db_password`` are all caught.
_SENSITIVE_KEY_TOKENS = ("api_key", "apikey", "token", "secret", "password", "passwd", "credential")
_REDACTED = "***REDACTED***"


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in _SENSITIVE_KEY_TOKENS)


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
        self, registry: ProviderRegistry | None = None, *, span_sink: SpanSink | None = None
    ) -> None:
        self._registry = registry or ProviderRegistry()
        # #73: the optional tracing sink. ``None`` ⇒ each run's RunTracer defaults to the
        # silent-by-default LoggingSpanSink, so a normal run's output is unchanged; an operator
        # (or a downstream tracer, or a test) passes a CollectingSpanSink / their own SpanSink to
        # capture the per-Stage spans. Kept a thin seam — no OTel dep in core (ADR-0016).
        self._span_sink = span_sink
        # Per-Stage endpoint-retry tally, drained into that Stage's StageMetrics (S12). An
        # endpoint-backed performance Provider exposes an ``endpoint_retries`` int attribute (how
        # many transient failures it retried through); the Runner reads it after the Stage runs and
        # resets for the next. A CPU/in-process Provider has no such attribute → 0.
        self._last_stage_retries = 0
        # Per-Stage relation fan-out-guard tally, keyed by Stage name: (output-capped docs,
        # relations dropped [ADR-0033 / #63], input-capped docs, mentions dropped [#68]). A
        # RelationExtractor Provider exposes ``fanout_capped_documents``/
        # ``fanout_dropped_relations`` (output cap) and ``mention_capped_documents``/
        # ``mention_dropped_mentions`` (input cap) int attributes; the Runner reads them after
        # the Stage runs and folds them into the Quality Report's RelationQuality. A Provider
        # without them contributes (0, 0, 0, 0).
        self._fanout_by_stage: dict[str, tuple[int, int, int, int]] = {}
        # #69: when True, a doc-level Stage's retained output is released from the in-memory
        # ``outputs`` dict once its last consumer has run (and a doc-level sink's output the moment
        # it commits) — the accumulators + Quality Report then read that output back from its
        # checkpoint. This bounds corpus-level memory to the KG state (ADR-0033's honest boundary)
        # rather than O(all doc-level records). Setting it False keeps every Stage's output
        # retained (the pre-#69 behavior) — a parity/debug knob; the on-disk output, resume, and
        # report are byte-identical either way, only peak memory differs.
        self._release_doc_level_outputs = True

    def _drain_retry_count(self) -> int:
        """Read and reset the endpoint-retry tally the just-executed Stage's Provider reported."""
        count = self._last_stage_retries
        self._last_stage_retries = 0
        return count

    @staticmethod
    def _provider_fanout(provider: object) -> tuple[int, int, int, int]:
        """The relation fan-out-guard tally a Provider exposes.

        Returns ``(output-capped-docs, dropped-relations, input-capped-docs,
        dropped-mentions)``. A RelationExtractor Provider enforcing the ADR-0033 / #63
        per-document relation OUTPUT cap tallies it in
        ``fanout_capped_documents``/``fanout_dropped_relations``; the #68 per-document
        mention INPUT cap tallies it in ``mention_capped_documents``/
        ``mention_dropped_mentions``. Any Provider without an attribute (or with a non-int)
        contributes ``0`` for it — read defensively so the seam names no base class (ADR-0004).
        """

        def _int_attr(name: str) -> int:
            value = getattr(provider, name, 0)
            return value if isinstance(value, int) else 0

        return (
            _int_attr("fanout_capped_documents"),
            _int_attr("fanout_dropped_relations"),
            _int_attr("mention_capped_documents"),
            _int_attr("mention_dropped_mentions"),
        )

    @staticmethod
    def _provider_retry_count(provider: object) -> int:
        """The endpoint-retry count a Provider exposes, or 0 if it exposes none.

        An endpoint-backed performance Provider (``latence-*-endpoint``) tallies how many transient
        endpoint failures it retried through in an ``endpoint_retries`` int attribute; an
        in-process/CPU Provider has no such attribute. Read defensively so any Provider shape is
        accepted (the seam names no base class — ADR-0004).
        """
        value = getattr(provider, "endpoint_retries", 0)
        return value if isinstance(value, int) else 0

    # -- device routing (P3-F1, ADR-0036 §2) -------------------------------

    def _route_device(
        self, stage: Stage, profile: ProviderProfile | None
    ) -> DeviceDecision:
        """Route a Stage's requested device against its Provider's declared profile.

        Reads the ``device`` config value (``auto|cpu|cuda``, default ``auto``) and calls
        the dep-light :func:`~latence_core.providers.device.select_device` seam against the
        Provider's declared :class:`~latence_core.providers.profile.ProviderProfile`
        (``None`` when it declares none). Device *placement* stays the adapter's job
        (ADR-0003/0036 §2) — the Runner only routes + records the decision. The returned
        :class:`~latence_core.providers.device.DeviceDecision` drives the G1 skip-with-flag
        posture: a GPU-only Provider on a CPU-only host is skipped (flagged), never crashed.
        """
        requested = stage.config.get("device", "auto")
        # ``select_device`` validates the requested string (a bad value is a ConfigError-y
        # ValueError); a non-string config value is coerced to its str form first so a typo
        # surfaces as an "unknown device" message rather than a TypeError.
        return select_device(str(requested), profile)

    @staticmethod
    def _profile_metric_fields(
        profile: ProviderProfile | None, requested: str, decision: DeviceDecision
    ) -> dict[str, object]:
        """The StageMetrics profile/device columns for a Stage (P3-F1).

        Surfaces the declared profile's ``compute``/``model_id``/``license`` (all ``None``
        when the Provider declares no profile — never fabricated) plus the device routing:
        the requested ``device``, the ``device_selected`` the seam resolved, and the
        ``skipped``/``skip_reason`` skip-with-flag markers.
        """
        return {
            "compute": profile.compute if profile is not None else None,
            "model_id": profile.model_id if profile is not None else None,
            "license": profile.license if profile is not None else None,
            "device": requested,
            "device_selected": decision.device,
            "skipped": decision.skipped,
            "skip_reason": decision.reason if decision.skipped else None,
        }

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
        if stage.capability is CapabilityKind.PROFILING:
            return FeatureRecord
        if stage.capability is CapabilityKind.DISAMBIGUATION:
            return DisambiguationRecord
        if stage.capability is CapabilityKind.GRAPH_ASSEMBLY:
            return GraphRecord
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
            if parent.capability is CapabilityKind.PROFILING:
                return FeatureRecord
            if parent.capability is CapabilityKind.DISAMBIGUATION:
                return DisambiguationRecord
            if parent.capability is CapabilityKind.GRAPH_ASSEMBLY:
                return GraphRecord
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
        """
        # Streaming opens bypass write_bytes' implicit parent-makedirs, so ensure the
        # checkpoint directory exists before opening the temp file.
        parent = uri.rsplit("/", 1)[0] if "/" in uri else uri
        if parent and parent != uri:
            storage.makedirs(parent)
        tmp = storage.temp_uri(uri)
        total = 0
        batch_index = 0
        with storage.open(tmp, "wb") as fh:
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
                        tracer.stage_batch(
                            stage_name, batch_index=batch_index, records_so_far=total
                        )
                    batch_index += 1
            if batch:
                fh.write(("\n".join(batch) + "\n").encode("utf-8"))
                total += len(batch)
                if wal_uri is not None and stage_name is not None:
                    self._wal_append_batch(storage, wal_uri, stage_name, batch_index, total)
                if tracer is not None and stage_name is not None:
                    tracer.stage_batch(stage_name, batch_index=batch_index, records_so_far=total)
        storage.finalize_atomic(tmp, uri)
        return total

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
            if is_accumulator:
                incoming: list[StageOutput] = []
                records_in = self._gathered_count(
                    stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            else:
                incoming = self._gather_inputs(
                    stage, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
                records_in = len(incoming)

            # P3-F1 (ADR-0036 §2) + P3-F2 (§2, the malformed-profile fix): the requested device
            # + the Provider's declared profile are re-derived on BOTH the resume and the fresh
            # path so the profile/device columns (incl. a skip flag) are STICKY across resume —
            # reading the profile is a cached, cheap class import, and re-routing never executes
            # the Provider. Three distinct outcomes, kept distinct (issue P3-F2 §2):
            #
            #   * load/import error (unknown provider, missing heavy dep) → DEFER: fall through
            #     to the normal execution path, which re-loads inside the execution ``try`` and
            #     re-raises with full failure handling (a FAIL report + a typed-error span). We do
            #     NOT map or skip it here — that would double-handle or mask the real load error.
            #   * profile declared but MALFORMED (``profile_of`` raises ``TypeError``) → FAIL
            #     LOUD: an author bug that must surface as a typed ``ConfigError`` + a FAIL report,
            #     not silently read as "no profile" and run normally (the P3-F1 deferred defect).
            #     Captured here and raised inside the execution ``try`` below so it flows through
            #     the exact same failure handling (span + FAIL report) as any Stage failure.
            #   * absent profile (declares none) → ``None``: follow the request (unchanged).
            requested_device = str(stage.config.get("device", "auto"))
            stage_profile = None
            load_ok = True
            malformed_profile_error: ConfigError | None = None
            try:
                provider_cls = self._registry.load(stage.provider)
            except Exception:  # noqa: BLE001 - a load/import error re-surfaces on the execution path
                # Load/import failure — deferred, NOT mapped (it must re-surface as the real
                # error when execution re-loads). Route the columns as "no profile", non-skip.
                load_ok = False
            else:
                try:
                    stage_profile = profile_of(provider_cls)
                except Exception as exc:  # noqa: BLE001 - any profile-declaration/-computation error is an author bug
                    # An author bug in the profile declaration — the provider class LOADED fine
                    # (that is handled above), so anything ``profile_of`` raises now is the profile
                    # itself being wrong. Two shapes, same disposition: a MALFORMED declaration
                    # (wrong return type → ``profile_of`` raises ``TypeError``), OR a ``profile()``
                    # classmethod whose BODY raises while building the descriptor (e.g. constructing
                    # an invalid ``ProviderProfile`` → pydantic ``ValidationError``). Both must fail
                    # the stage loud as a typed ConfigError (not run silently as "no profile").
                    # Capture it and raise it inside the execution ``try`` so it gets the FAIL
                    # report + typed-error span, exactly like an executed-Stage failure — never
                    # skipped-with-flag (which would mask it).
                    malformed_profile_error = ConfigError(
                        f"Provider {stage.provider!r} for stage {stage.name!r} declares a "
                        f"malformed ProviderProfile (invalid declaration or profile() error): {exc}"
                    )
                    stage_profile = None
            if load_ok and malformed_profile_error is None:
                device_decision = self._route_device(stage, stage_profile)
            else:
                device_decision = DeviceDecision(
                    device=None, skipped=False, reason="provider load deferred to execution"
                )
            profile_fields = self._profile_metric_fields(
                stage_profile, requested_device, device_decision
            )

            if storage.exists(ckpt_uri):
                # Resume: this Stage already completed. Reload ITS OWN output into
                # the dict and skip re-running. A downstream Stage still reads this
                # value as one of its parents' outputs. The profile/device columns
                # (re-derived above) are re-surfaced so a skipped GPU Stage still
                # reports skipped=True + its profile columns on resume — sticky, not
                # silently dropped to a plain non-skipped Stage.
                own_output = self._read_checkpoint(
                    storage, ckpt_uri, self._output_type(stage, pipeline)
                )
                outputs[stage.name] = own_output
                stage_metrics.append(
                    StageMetrics(
                        name=stage.name,
                        capability=stage.capability.value,
                        provider=stage.provider,
                        records_in=records_in,
                        records_out=len(own_output),
                        duration_seconds=0.0,
                        from_checkpoint=True,
                        runner=runner_name,
                        **profile_fields,  # type: ignore[arg-type]
                    )
                )
                _release_dead_parents(stage)
                continue

            # P3-F1 (ADR-0036 §2): the requested device + declared profile were routed above
            # (before the resume check) so the profile/device columns are sticky across resume.
            # A GPU-only Provider on a CPU-only host is skipped-with-flag (the G1 posture) — the
            # Stage produces an empty output, an (empty) checkpoint is written so the skip is
            # sticky on resume, and the skip is recorded in the Quality Report — never crashed,
            # never a fabricated GPU number. Device *placement* stays the adapter's job; the
            # Runner only routes + records. A Provider that declares no profile follows the
            # request; a malformed declaration / unknown provider (load_ok=False) is forced onto
            # the non-skip path below, where execution re-loads and re-raises the real error.
            if device_decision.skipped:
                # Skip-with-flag: do not instantiate/run the Provider. Emit an empty output +
                # empty checkpoint so a resume sees the Stage as already handled (the skip is
                # sticky), and record the skip in the Stage's metrics. Downstream Stages read an
                # empty parent output — the stack harness reads the skip flag from the report.
                empty_output: list[StageOutput] = []
                outputs[stage.name] = empty_output
                self._write_checkpoint(
                    storage,
                    ckpt_uri,
                    empty_output,
                    wal_uri=wal_uri,
                    stage_name=stage.name,
                    tracer=tracer,
                )
                self._wal_append(storage, wal_uri, _WAL_STAGE_COMMIT, stage.name)
                stage_metrics.append(
                    StageMetrics(
                        name=stage.name,
                        capability=stage.capability.value,
                        provider=stage.provider,
                        records_in=records_in,
                        records_out=0,
                        duration_seconds=0.0,
                        runner=runner_name,
                        **profile_fields,  # type: ignore[arg-type]
                    )
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
            started = time.perf_counter()
            try:
                # P3-F2 §2: a malformed ProviderProfile declaration is fatal — raise the captured
                # typed ConfigError HERE, inside the execution ``try``, so it flows through the
                # exact same failure handling as any Stage failure (a FAIL Quality Report + a
                # typed-error ``config`` span + re-raise), never silently read as "no profile"
                # and run normally (the P3-F1 deferred defect). It fires before the Provider is
                # instantiated — a bad declaration never gets to run.
                if malformed_profile_error is not None:
                    raise malformed_profile_error
                own_output = self._execute_stage(
                    stage, storage, pipeline, run_id, incoming, outputs
                )
            except BaseException as exc:
                # #73: a Stage failure emits a *failed* stage_end span stamped with the typed-error
                # category (contract/config/provider/storage/screening/unknown) + message, then
                # re-raises — the Runner never swallows a Stage failure (no torn commit), it just
                # makes the failure kind observable first. Counts + category only, never content.
                duration = time.perf_counter() - started
                category = error_category(exc)
                tracer.stage_end(
                    stage.name,
                    capability=stage.capability.value,
                    provider=stage.provider,
                    records_in=records_in,
                    records_out=0,
                    duration_ms=round(duration * 1000.0, 3),
                    error_category=category,
                    error_message=str(exc),
                )
                # #73: the taxonomy is surfaced in the report *and* the logs — so record the failed
                # Stage's typed-error metric (error_count=1 + the stable category tag, counts only,
                # never the exception message) and persist a FAIL Quality Report before re-raising.
                # This is the report half of #73's "surfaced in the report + logs": a persisted
                # report now carries the failed Stage's error_count, not only a transient span.
                #
                # No-torn-write is preserved: the failed Stage's *checkpoint* is still never
                # finalized (we never reached ``_write_checkpoint``), so a resume re-runs it.
                # The report is not a resume gate (resume keys off per-Stage checkpoints, not the
                # report), so a subsequent successful resume overwrites this FAIL report atomically.
                # Persisting must never mask the real failure: if the report build itself raises we
                # swallow *that* and re-raise the original Stage exception.
                stage_metrics.append(
                    StageMetrics(
                        name=stage.name,
                        capability=stage.capability.value,
                        provider=stage.provider,
                        records_in=records_in,
                        records_out=0,
                        duration_seconds=round(duration, 6),
                        runner=runner_name,
                        endpoint_retries=self._drain_retry_count(),
                        error_count=1,
                        error_category=category,
                        **profile_fields,  # type: ignore[arg-type]
                    )
                )
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
            duration = time.perf_counter() - started

            outputs[stage.name] = own_output
            # Batch-granular streamed checkpoint write (ADR-0033 / #63): the write transient is
            # O(batch), not O(all-records); batch progress lands in the WAL; the final file is
            # published atomically (temp→rename) so resume stays exact at Stage granularity.
            self._write_checkpoint(
                storage,
                ckpt_uri,
                own_output,
                wal_uri=wal_uri,
                stage_name=stage.name,
                tracer=tracer,
            )
            self._wal_append(storage, wal_uri, _WAL_STAGE_COMMIT, stage.name)
            tracer.stage_end(
                stage.name,
                capability=stage.capability.value,
                provider=stage.provider,
                records_in=records_in,
                records_out=len(own_output),
                duration_ms=round(duration * 1000.0, 3),
            )

            stage_metrics.append(
                StageMetrics(
                    name=stage.name,
                    capability=stage.capability.value,
                    provider=stage.provider,
                    records_in=records_in,
                    records_out=len(own_output),
                    duration_seconds=round(duration, 6),
                    runner=runner_name,
                    endpoint_retries=self._drain_retry_count(),
                    **profile_fields,  # type: ignore[arg-type]
                )
            )
            # #69: release any doc-level parent whose last consumer just ran — its checkpoint is
            # the source of truth for the end-of-run Quality Report, which reads it back
            # (bounded) via _stage_records. This is what drops the retained doc-level executor
            # transient the ADR flagged as remaining.
            _release_dead_parents(stage)

        final_records = self._final_records(
            ordered, outputs, storage=storage, pipeline=pipeline, run_id=run_id
        )
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
        outputs: dict[str, list[StageOutput]],
    ) -> list[StageOutput]:
        """Run one Stage and record any endpoint retries its Provider reported (S12).

        Delegates the capability dispatch to :meth:`_dispatch_stage`, then reads the endpoint-retry
        tally off the (endpoint-backed) Provider so ``run()`` can fold it into that Stage's metrics.
        Kept as the single seam a second Runner adapter reuses — identical Stage semantics + the
        same retry accounting under any substrate (ADR-0003), no fork. ``outputs`` is threaded so a
        corpus-level accumulator's dispatch can stream its input from parents' checkpoints (#69)
        rather than receiving it pre-materialised in ``incoming`` (which is empty for accumulators).
        """
        provider_cls = self._registry.load(stage.provider)
        provider = provider_cls(stage.config)
        try:
            return self._dispatch_stage(
                stage, provider, storage, pipeline, run_id, incoming, outputs
            )
        finally:
            self._last_stage_retries = self._provider_retry_count(provider)
            if stage.capability in (
                CapabilityKind.RELATION_EXTRACTION,
                CapabilityKind.FUSED_ENTITY_RELATION,
            ):
                self._fanout_by_stage[stage.name] = self._provider_fanout(provider)

    def _dispatch_stage(
        self,
        stage: Stage,
        provider: object,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        incoming: list[StageOutput],
        outputs: dict[str, list[StageOutput]],
    ) -> list[StageOutput]:
        if stage.capability is CapabilityKind.SOURCE:
            if not isinstance(provider, Source):
                msg = f"Provider {stage.provider!r} does not satisfy the Source capability."
                raise ContractError(msg)
            # A Source is a root: it reads Storage, not upstream records.
            return list(provider.produce(storage))

        if stage.capability is CapabilityKind.INTAKE_SCREENING:
            if not isinstance(provider, IntakeScreener):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the IntakeScreener capability."
                )
                raise ContractError(msg)
            raw_inputs = self._as_type(stage, incoming, ParserInput)
            intake_outcome = provider.screen_intake(raw_inputs)
            self._persist_screening_sidecar(storage, pipeline, run_id, stage, intake_outcome)
            # Only the passed raw inputs proceed to Parse; the quarantined ones are
            # removed from the DAG (persisted for audit above), never reaching Parse.
            return list(intake_outcome.passed)

        if stage.capability is CapabilityKind.PARSE:
            if not isinstance(provider, Parser):
                msg = f"Provider {stage.provider!r} does not satisfy the Parser capability."
                raise ContractError(msg)
            parser_inputs = self._as_type(stage, incoming, ParserInput)
            return list(provider.parse(parser_inputs))

        if stage.capability is CapabilityKind.CHUNK:
            if not isinstance(provider, Chunker):
                msg = f"Provider {stage.provider!r} does not satisfy the Chunker capability."
                raise ContractError(msg)
            doc_records = self._as_type(stage, incoming, DocumentRecord)
            return list(provider.chunk(doc_records))

        if stage.capability is CapabilityKind.CONTENT_SCREENING:
            if not isinstance(provider, ContentScreener):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the ContentScreener capability."
                )
                raise ContractError(msg)
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
                raise ContractError(msg)
            chunk_records = self._as_type(stage, incoming, ChunkRecord)
            return list(provider.extract(chunk_records))

        if stage.capability is CapabilityKind.RELATION_EXTRACTION:
            if not isinstance(provider, RelationExtractor):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the RelationExtractor capability."
                )
                raise ContractError(msg)
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
                raise ContractError(msg)
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
                raise ContractError(msg)
            # Redaction consumes the run's parsed DocumentRecords (the whole
            # document's assembled markdown + page map, at original-document offsets)
            # and emits one PII-handled RedactionRecord per document, keyed off
            # Classification sensitivity — threaded downstream to Export as the
            # corpus's PII variant. It depends on Parse (not Chunk): the chunk stream
            # is markup-stripped with remapped offsets, so scanning it would fabricate
            # a broken coordinate system for the spans and page resolution.
            redaction_docs = self._as_type(stage, incoming, DocumentRecord)
            return list(provider.redact(redaction_docs))

        if stage.capability is CapabilityKind.PROFILING:
            if not isinstance(provider, Profiler):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the Profiler capability."
                )
                raise ContractError(msg)
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
            return list(provider.profile(profile_docs, profile_mentions))

        if stage.capability is CapabilityKind.DISAMBIGUATION:
            if not isinstance(provider, Disambiguator):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the Disambiguator capability."
                )
                raise ContractError(msg)
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
            return list(provider.disambiguate(disambig_mentions, disambig_relations))

        if stage.capability is CapabilityKind.GRAPH_ASSEMBLY:
            if not isinstance(provider, GraphAssembler):
                msg = (
                    f"Provider {stage.provider!r} does not satisfy "
                    f"the GraphAssembler capability."
                )
                raise ContractError(msg)
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
            return list(provider.assemble(graph_records))

        if stage.capability is CapabilityKind.EXPORT:
            if not isinstance(provider, Export):
                msg = f"Provider {stage.provider!r} does not satisfy the Export capability."
                raise ContractError(msg)
            # Export materializes any inter-Stage Record: DocumentRecords (S1/S2
            # spine) or ChunkRecords (S3, when a Chunk Stage sits upstream). Only a
            # raw ParserInput (not a Record) is an invalid Export input.
            exportable = self._as_records(stage, incoming)
            out_dir = self._export_dir(pipeline, run_id)
            provider.export(exportable, storage, out_dir)
            # Export is a sink: it passes its input records through unchanged.
            return list(exportable)

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

    def _stage_records(
        self,
        stage_name: str,
        outputs: dict[str, list[StageOutput]],
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
        for parent in stage.depends_on:
            for _ in self._stage_records(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                total += 1
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
        for parent in stage.depends_on:
            for record in self._stage_records(
                parent, outputs, storage=storage, pipeline=pipeline, run_id=run_id
            ):
                present.add(type(record))
                names.add(type(record).__name__)
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

    def _final_records(
        self,
        ordered: list[Stage],
        outputs: dict[str, list[StageOutput]],
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
                self._stage_records(
                    name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                )
            )
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
        redactions = self._records_of_kind(
            pipeline, ordered, outputs, CapabilityKind.REDACTION, storage=storage, run_id=run_id
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
            graph=self._graph_quality(pipeline, graph_records),
            drift=self._drift_diagnostics(chunked),
            goldset=self._goldset_evaluation(pipeline, storage, mentions, relations),
        )

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
                prov.page_start is None
                or prov.page_end is None
                or prov.page_end >= prov.page_start
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
            file_name_of[doc_id]
            for doc_id in file_name_of
            if file_name_of[doc_id] in gold_files
        }
        evaluation = GoldSetEvaluation(
            documents_annotated=len(gold.documents),
            documents_matched=len(matched_files),
            match_min_overlap=gold.match_min_overlap,
        )
        entity_overall, entity_labels = score_entities(typed_mentions, gold, file_name_of)
        evaluation.stages.append(
            self._stage_score("entities", entity_overall, entity_labels)
        )
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

    def _records_of_kind(
        self,
        pipeline: Pipeline,
        ordered: list[Stage],
        outputs: dict[str, list[StageOutput]],
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
                    self._stage_records(
                        stage.name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                    )
                )
        return result

    def _records_of_kinds(
        self,
        pipeline: Pipeline,
        ordered: list[Stage],
        outputs: dict[str, list[StageOutput]],
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
                    self._stage_records(
                        stage.name, outputs, storage=storage, pipeline=pipeline, run_id=run_id
                    )
                )
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

    def _fanout_total(self, ordered: list[Stage]) -> tuple[int, int, int, int]:
        """Sum the per-Stage relation fan-out-guard tallies across the run (ADR-0033 / #63, #68).

        Folds every relation/fused Stage's ``(output-capped-docs, dropped-relations,
        input-capped-docs, dropped-mentions)`` tally (captured in :meth:`_execute_stage`) into
        one corpus-wide 4-tuple for the Quality Report. A Stage restored from a checkpoint on
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
        pipeline: Pipeline, records: list[StageOutput]
    ) -> GraphQuality | None:
        """Aggregate Graph-Assembly KG statistics for the report (S9).

        Only computed when the Pipeline has a Graph Assembly Stage. ``records`` is the mix of
        NODE-scope (graph nodes) and EDGE-scope (graph edges)
        :class:`~latence_core.contracts.GraphRecord`s. Rolls up: node/edge counts; nodes by
        entity type and edges by relation label; KB-linked node count; and — the S9 acceptance
        criterion — evidence coverage, the fraction of edges whose Evidence names at least one
        source mention (1.0 = every edge is justified). Counts only (no raw text), safe to
        persist. All maps are sorted for a deterministic report.
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
