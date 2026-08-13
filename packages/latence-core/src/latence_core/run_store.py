"""RunStore — the owner of the run layout on Storage and the checkpoint decode (ADR-0010).

ADR-0010 says Runner state is **files on Storage**: everything a run leaves behind lives under
``<storage_uri>/_latence/runs/<run_id>/`` — the manifest, the per-Stage checkpoints, the WAL,
the screening sidecars, the type-vocabulary sidecar, the export directory and the Quality
Report pair. That layout is total in ``(storage_uri, run_id)``, yet until this module it had
no owner: the paths were minted by hand-built string concatenation at ~12 sites across six
packages, and the typed checkpoint decode lived behind ``LocalRunner`` privates that the delta
engine and the Airflow adapter reached into with ``# noqa: SLF001`` — exactly the shape the
audit verdict quoted in :mod:`~latence_core.quality_report_builder` rejects: *"the interface
is the reuse surface — if a second impl must reach past it, the shape is wrong."*

This module is that owner. It re-homes a decision, it does not re-make one: **same paths,
same bytes, same temp→rename discipline** as the ``LocalRunner`` privates it absorbs.

* :class:`RunLayout` — the pure path half: every location one run owns, computed from
  ``(storage_uri, run_id)`` alone, so a consumer that only names paths (the console's
  read-only index, the campaign stage addressing a remote pod's run over SSH) never has to
  construct a :class:`~latence_core.storage.Storage` it will not use.
* the **decode** — :func:`decode_records` (the typed newline-delimited checkpoint decode,
  including the mixed-carrier discrimination a fused Stage's checkpoint needs),
  :func:`stream_checkpoint` / :func:`read_checkpoint` (lazy and buffered readers over it),
  :func:`count_checkpoint_records` (the exact no-decode record count), and
  :func:`stage_output_type` / :func:`export_producer` / :func:`export_carrier` (the carrier a
  Stage checkpoints, read off its CapabilityDescriptor).
* the **write discipline** — :func:`write_checkpoint` / :func:`stream_batches`: batch-granular
  serialization into an atomically-published temp file (ADR-0033 / ADR-0010's WAL-atomic
  temp→rename), with optional per-batch WAL events and trace spans.
* :class:`RunStore` — the run-scoped face composing all of the above with a live ``Storage``:
  path properties plus ``read`` / ``count`` / ``write`` over typed Stage records.

:class:`~latence_core.runner.LocalRunner` keeps its private methods as thin delegators over
this module — its internal call sites, its subclass seams (tests intercept
``_stream_batches``) and its public ``run_dir()`` / ``report_uri()`` are unchanged. Every
other consumer — the delta engine, the Airflow adapter, the CLI, the console, the demo
builder, the bake-off harness, the tuning probe, the campaign pipeline stage — goes through
this interface instead of rebuilding paths or reaching into privates.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from latence_core.capability_descriptor import descriptor_for, export_carrier_of
from latence_core.contracts import (
    DocumentRecord,
    EntityMention,
    Record,
    RelationMention,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import IO

    from latence_core.pipeline import Pipeline, Stage
    from latence_core.runner import StageOutput
    from latence_core.storage import Storage
    from latence_core.tracing import RunTracer

#: The run-state root every Runner writes under, relative to the Pipeline's ``storage_uri``.
RUNS_PREFIX = "_latence/runs"

#: The file names a run's root directory holds (one manifest, one WAL, the report pair).
MANIFEST_NAME = "manifest.json"
WAL_NAME = "wal.log"
REPORT_NAME = "quality-report.json"
REPORT_MARKDOWN_NAME = "quality-report.md"

# Checkpoint write batch size (ADR-0033 / #63): the number of records serialized and flushed
# to the temp checkpoint at once, so the write transient is O(batch) instead of the old
# O(all-records) whole-string build. Purely a memory-shaping knob — the on-disk file
# (newline-delimited records) and Stage-granular resume are identical regardless of the value.
CHECKPOINT_BATCH_RECORDS = 1024

# The WAL batch-progress event kind (ADR-0033 / #63); append-only, never read for control flow.
_WAL_STAGE_BATCH = "stage_batch"


def runs_root_for(storage_uri: str) -> str:
    """The runs root (``<storage_uri>/_latence/runs``) — every run of this Storage lives here."""
    return f"{storage_uri.rstrip('/')}/{RUNS_PREFIX}"


def run_dir_for(storage_uri: str, run_id: str) -> str:
    """One run's state/artifact directory (``<storage_uri>/_latence/runs/<run_id>``)."""
    return f"{runs_root_for(storage_uri)}/{run_id}"


@dataclass(frozen=True)
class RunLayout:
    """Every path one run owns, pure in ``(storage_uri, run_id)`` — no Storage required.

    The layout half of the RunStore: a value object a path-only consumer (the console's
    read-only run index, the campaign stage addressing a pod-side run over SSH) can use
    without holding a live :class:`~latence_core.storage.Storage`. The strings are
    byte-for-byte the ones ``LocalRunner`` has always minted — this class is where that
    decision now lives, and the runner's private path methods delegate here.
    """

    storage_uri: str
    run_id: str

    @property
    def run_dir(self) -> str:
        """The run's state/artifact directory (``<storage_uri>/_latence/runs/<run_id>``)."""
        return run_dir_for(self.storage_uri, self.run_id)

    @property
    def manifest_uri(self) -> str:
        """The run manifest (the persisted Pipeline + run id, secrets scrubbed)."""
        return f"{self.run_dir}/{MANIFEST_NAME}"

    @property
    def wal_uri(self) -> str:
        """The append-only write-ahead log (audit + crash forensics, never control flow)."""
        return f"{self.run_dir}/{WAL_NAME}"

    @property
    def report_uri(self) -> str:
        """The machine-readable Quality Report (ADR-0008; the run's last-written artifact)."""
        return f"{self.run_dir}/{REPORT_NAME}"

    @property
    def report_markdown_uri(self) -> str:
        """The human-readable Quality Report rendering the CLI persists beside the JSON (S10)."""
        return f"{self.run_dir}/{REPORT_MARKDOWN_NAME}"

    @property
    def export_dir(self) -> str:
        """The directory an Export Stage materializes the AI-ready deliverable into."""
        return f"{self.run_dir}/export"

    def checkpoint_uri(self, stage_name: str) -> str:
        """One Stage's finalized checkpoint (newline-delimited typed records; presence ⇒ resume)."""
        return f"{self.run_dir}/checkpoints/{stage_name}.jsonl"

    def quarantine_uri(self, stage_name: str) -> str:
        """A Screening Stage's retained-and-inspectable quarantined records (CONTEXT
        ``Quarantine``): excluded from the DAG and Export, persisted per Screening Stage."""
        return f"{self.run_dir}/quarantine/{stage_name}.jsonl"

    def findings_uri(self, stage_name: str) -> str:
        """A Screening Stage's findings sidecar (the Quality Report's Screening rollup source)."""
        return f"{self.run_dir}/screening/{stage_name}.jsonl"

    def type_vocabulary_uri(self, stage_name: str) -> str:
        """A Type-Consolidation Stage's canonical-vocabulary sidecar (T3, off the DAG)."""
        return f"{self.run_dir}/type-vocabulary/{stage_name}.json"


# -- carrier resolution ------------------------------------------------------


def stage_output_type(stage: Stage, pipeline: Pipeline) -> type[StageOutput] | type[Record]:
    """The carrier a Stage checkpoints (the records it feeds downstream).

    Read straight off the Capability's own :class:`~latence_core.capability_descriptor.\
CapabilityDescriptor` (L2/C2) rather than re-decided by a fifty-line if-chain. Every
    rationale that used to be a comment beside a branch — Source/Intake Screening thread raw
    ParserInputs, the chunk→chunk transforms (Schema Induction, Context Enrichment, Redaction)
    checkpoint the SAME ChunkRecords, a fused or type-consolidating Stage checkpoints a MIXED
    carrier signalled by :class:`~latence_core.contracts.Record` and decoded per line — now
    sits beside the ``carrier`` field that states it.

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
        return export_carrier(stage, pipeline)
    # EMBEDDING: carrier-less AND not a DAG node (a nested Export sub-config, ADR-0017), so it
    # is never a Stage and this is unreachable through a validated Pipeline. Answer with the
    # spine's carrier rather than resolving through ``depends_on`` a non-Stage does not have.
    return DocumentRecord


def export_carrier(stage: Stage, pipeline: Pipeline) -> type[StageOutput] | type[Record]:
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
    separately from :func:`stage_output_type` and had to be kept consistent with it by memory.

    The frontier is walked breadth-first in ``depends_on`` order, so the nearest producing
    ancestor wins.
    """
    producer = export_producer(stage, pipeline)
    if producer is not None:
        carrier = export_carrier_of(producer.capability)
        if carrier is not None:  # pragma: no branch - a producer's carrier is declared
            return cast("type[StageOutput] | type[Record]", carrier)
    return DocumentRecord


def export_producer(stage: Stage, pipeline: Pipeline) -> Stage | None:
    """The nearest ancestor Stage that PRODUCES the records an Export re-emits (ADR-0061).

    The walk :func:`export_carrier` reads its answer from, exposed as the Stage itself
    because the overlay fold rule needs the producer's LEVEL, not only its carrier: a
    corpus-level producer (Context Enrichment, Type Consolidation) re-emits the whole
    ``fresh ∪ reused`` union, so a doc-level Export below it must NOT materialize the overlay
    again, while a doc-level producer (chunk/redact/extract) emits only the fresh documents
    and the Export merges the overlay in. ``None`` when no ancestor produces an exportable
    carrier (the S1 spine fallback).
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
            return parent
        frontier.extend(parent.depends_on)
    return None


# -- checkpoint decode -------------------------------------------------------


def decode_records(text: str, record_type: type[StageOutput] | type[Record]) -> list[StageOutput]:
    """Validate newline-delimited record JSON into typed records (the checkpoint decode).

    The single typed decode every reader shares — the local Runner's resume replay, the delta
    engine's collection passes, and a Runner adapter transporting Stage outputs over a
    non-file substrate (e.g. Airflow XCom, S12) — so the contract survives every boundary
    with no bespoke (de)serialization per Runner.
    """
    if record_type is Record:
        # A fused Stage's mixed carrier (or an Export re-emitting one): each line
        # is an EntityMention or a RelationMention — discriminate per line.
        return _decode_mixed_lines(text)
    records: list[StageOutput] = []
    for raw in text.splitlines():
        if raw.strip():
            records.append(record_type.model_validate_json(raw))  # type: ignore[arg-type]
    return records


def _decode_mixed_lines(text: str) -> list[StageOutput]:
    """Decode a mixed EntityMention/RelationMention checkpoint (a fused Stage output).

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


def read_checkpoint(
    storage: Storage, uri: str, record_type: type[StageOutput] | type[Record]
) -> list[StageOutput]:
    """Read a whole checkpoint into a typed list (the buffered companion to the stream)."""
    return decode_records(storage.read_text(uri), record_type)


def stream_checkpoint(
    storage: Storage, uri: str, record_type: type[StageOutput] | type[Record]
) -> Iterator[StageOutput]:
    """Yield a Stage checkpoint's records lazily, decoding at most one line at a time (#69).

    The read-back companion to :func:`write_checkpoint`'s batch-granular write: rather than
    materialising the whole checkpoint into a list (:func:`read_checkpoint`), this opens the
    file and yields one typed record per non-blank line, so an accumulator (or the Quality
    Report) that folds the doc-level output in one pass never holds it whole. This is what lets
    the executor RELEASE a doc-level Stage's ``outputs`` entry before the accumulators run —
    the corpus-level transient becomes O(batch) read-back + the accumulator's own bounded KG
    state, not O(all doc-level records) (ADR-0033's honest boundary, completed by #69).

    The typed decode is identical to :func:`decode_records` (same ``model_validate_json`` /
    mixed-carrier discrimination), so a streamed read and a whole read produce byte-identical
    records in the same order — determinism and the Provenance/offset/Evidence contracts are
    untouched. The file handle is closed when the generator is exhausted or garbage-collected.
    """
    with storage.open(uri, "rb") as fh:
        for raw_line in fh:
            raw = raw_line.decode("utf-8")
            if not raw.strip():
                continue
            # Decode one line through the shared typed decode (identical validation +
            # mixed-carrier discrimination as the whole-file read), so a streamed record is
            # byte-identical to the buffered one. A blank line yields nothing.
            yield from decode_records(raw, record_type)


def count_checkpoint_records(storage: Storage, uri: str) -> int:
    """How many records a checkpoint holds — its non-blank LINES, exact without decoding.

    A checkpoint is newline-delimited ``model_dump_json`` (one record per non-blank line,
    blank lines skipped on read), so counting lines equals counting decoded records —
    including a fused Stage's mixed-carrier checkpoint, where each line is still exactly one
    record. A missing checkpoint (Stage not run, or genuinely empty output) counts 0.
    O(1) memory, and no Pydantic pass over the corpus.
    """
    if not storage.exists(uri):
        return 0
    total = 0
    with storage.open(uri, "rb") as fh:
        for raw_line in fh:
            if raw_line.strip():
                total += 1
    return total


# -- checkpoint write (ADR-0033 batch-granular, ADR-0010 temp→rename) --------


def wal_append_batch(
    storage: Storage, uri: str, stage: str, batch_index: int, records_so_far: int
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


def stream_batches(
    fh: IO[bytes],
    storage: Storage,
    records: Iterable[StageOutput],
    *,
    wal_uri: str | None,
    stage_name: str | None,
    tracer: RunTracer | None,
    batch_size: int = CHECKPOINT_BATCH_RECORDS,
) -> tuple[int, int]:
    """Serialize ``records`` into the open handle in bounded batches; return (records, batches).

    ``storage`` is here only for the WAL append — the checkpoint bytes go to ``fh``, whose
    publication (or sweep) is the caller's ``atomic_writer`` block
    (:func:`write_checkpoint` / :meth:`~latence_core.runner.LocalRunner._write_checkpoint`).
    When a ``tracer`` is supplied (#73), each durably-written batch also emits a
    ``stage_batch`` trace span (batch index + cumulative records, counts only) mirroring the
    WAL batch event — so a downstream tracer sees checkpoint throughput at batch granularity.
    """
    total = 0
    batch_index = 0
    batch: list[str] = []
    for record in records:
        batch.append(record.model_dump_json())
        if len(batch) >= batch_size:
            fh.write(("\n".join(batch) + "\n").encode("utf-8"))
            total += len(batch)
            batch = []
            if wal_uri is not None and stage_name is not None:
                wal_append_batch(storage, wal_uri, stage_name, batch_index, total)
            # #73: mirror the WAL batch progress as a trace span (counts only), so a
            # downstream tracer sees checkpoint throughput per batch, not just per Stage.
            if tracer is not None and stage_name is not None:
                tracer.stage_batch(stage_name, batch_index=batch_index, records_so_far=total)
            batch_index += 1
    if batch:
        fh.write(("\n".join(batch) + "\n").encode("utf-8"))
        total += len(batch)
        if wal_uri is not None and stage_name is not None:
            wal_append_batch(storage, wal_uri, stage_name, batch_index, total)
        if tracer is not None and stage_name is not None:
            tracer.stage_batch(stage_name, batch_index=batch_index, records_so_far=total)
        batch_index += 1
    return total, batch_index


def write_checkpoint(
    storage: Storage,
    uri: str,
    records: Iterable[StageOutput],
    *,
    wal_uri: str | None = None,
    stage_name: str | None = None,
    tracer: RunTracer | None = None,
) -> int:
    """Write a Stage checkpoint batch-granularly, bounding the write transient (ADR-0033).

    Records are serialized in bounded batches (:data:`CHECKPOINT_BATCH_RECORDS`) and each
    batch is appended to a sibling temp file, so at most one batch's serialized text is live
    at once. Each completed batch is logged to the WAL (``stage_batch`` events) when a
    ``wal_uri`` + ``stage_name`` are supplied, then the temp file is atomically published to
    the final ``uri`` (temp→rename, the WAL-atomic discipline ADR-0010 requires): a crash
    mid-Stage leaves the previous checkpoint or the complete new one, never a torn write —
    resume stays exact at Stage granularity (a finalized checkpoint ⇒ skip).

    The on-disk format is newline-delimited ``model_dump_json`` per record, one record per
    line, trailing newline iff non-empty — byte-identical to what ``LocalRunner`` has always
    written, so every checkpoint reader round-trips it exactly. Returns the record count.

    The temp-name minting, the parent ``makedirs`` and the sweep of a partial write live
    behind :meth:`~latence_core.storage.Storage.atomic_writer` (L8/C9), not here.
    """
    with storage.atomic_writer(uri) as fh:
        total, _batches = stream_batches(
            fh, storage, records, wal_uri=wal_uri, stage_name=stage_name, tracer=tracer
        )
    return total


class RunStore:
    """One run's files on Storage, behind one owned interface (ADR-0010's missing owner).

    Constructed from ``(storage, pipeline, run_id)`` — the tuple the whole layout is total
    in. The path properties answer *where* a run's artifacts live (delegating to
    :class:`RunLayout`, so they are byte-for-byte the runner's own paths); ``read`` /
    ``count`` / ``write`` move typed Stage records across that layout with the same decode
    and the same atomic write discipline the local Runner uses — because they are the same
    code. A second engine (the delta runner, an adapter, a harness) composes this instead of
    reaching into ``LocalRunner`` privates or rebuilding ``f"{uri}/_latence/runs/…"`` by hand.
    """

    def __init__(self, storage: Storage, pipeline: Pipeline, run_id: str) -> None:
        self._storage = storage
        self._pipeline = pipeline
        self._layout = RunLayout(pipeline.storage_uri, run_id)

    @property
    def layout(self) -> RunLayout:
        """The pure path half (also usable standalone where no Storage exists)."""
        return self._layout

    @property
    def run_dir(self) -> str:
        """The run's state/artifact directory (``<storage_uri>/_latence/runs/<run_id>``)."""
        return self._layout.run_dir

    @property
    def manifest_uri(self) -> str:
        """The run manifest (the persisted Pipeline + run id, secrets scrubbed)."""
        return self._layout.manifest_uri

    @property
    def wal_uri(self) -> str:
        """The append-only write-ahead log."""
        return self._layout.wal_uri

    @property
    def report_uri(self) -> str:
        """The machine-readable Quality Report."""
        return self._layout.report_uri

    @property
    def report_markdown_uri(self) -> str:
        """The human-readable Quality Report rendering persisted beside the JSON."""
        return self._layout.report_markdown_uri

    @property
    def export_dir(self) -> str:
        """The directory an Export Stage materializes the deliverable into."""
        return self._layout.export_dir

    def checkpoint_uri(self, stage_name: str) -> str:
        """One Stage's finalized checkpoint."""
        return self._layout.checkpoint_uri(stage_name)

    def quarantine_uri(self, stage_name: str) -> str:
        """A Screening Stage's quarantined-records sidecar."""
        return self._layout.quarantine_uri(stage_name)

    def findings_uri(self, stage_name: str) -> str:
        """A Screening Stage's findings sidecar."""
        return self._layout.findings_uri(stage_name)

    def type_vocabulary_uri(self, stage_name: str) -> str:
        """A Type-Consolidation Stage's canonical-vocabulary sidecar."""
        return self._layout.type_vocabulary_uri(stage_name)

    def output_type(self, stage: Stage) -> type[StageOutput] | type[Record]:
        """The carrier ``stage`` checkpoints (see :func:`stage_output_type`)."""
        return stage_output_type(stage, self._pipeline)

    def read(
        self, stage: Stage, record_type: type[StageOutput] | type[Record] | None = None
    ) -> Iterator[StageOutput]:
        """Lazily yield ``stage``'s checkpointed records; a missing checkpoint yields nothing.

        The typed decode streams one line at a time (:func:`stream_checkpoint`), so a reader
        folding a corpus-sized checkpoint never holds it whole (C6). ``record_type`` defaults
        to the Stage's own declared carrier (:func:`stage_output_type`); pass it explicitly to
        decode with a narrower contract (the delta engine reads a corpus-granularity Schema
        Induction Stage's checkpoint as ChunkRecords, for instance). A Stage with no finalized
        checkpoint — never run, or its run never committed — yields nothing, which is exactly
        the "skip it" answer every collection pass wants.
        """
        uri = self._layout.checkpoint_uri(stage.name)
        if not self._storage.exists(uri):
            return
        resolved = record_type if record_type is not None else self.output_type(stage)
        yield from stream_checkpoint(self._storage, uri, resolved)

    def count(self, stage_name: str) -> int:
        """How many records ``stage_name``'s checkpoint holds (0 when absent); no decode."""
        return count_checkpoint_records(self._storage, self._layout.checkpoint_uri(stage_name))

    def write(
        self,
        stage_name: str,
        records: Iterable[StageOutput],
        *,
        wal: bool = False,
        tracer: RunTracer | None = None,
    ) -> int:
        """Write ``stage_name``'s checkpoint with the run's own write discipline; return count.

        Batch-granular, atomically published (:func:`write_checkpoint`) — byte-identical to
        the file the local Runner would write for the same records. ``wal=True`` additionally
        logs per-batch progress events to this run's WAL, exactly as the local executor does.
        (``LocalRunner`` itself keeps writing through its ``_write_checkpoint`` delegator so
        its subclass seam — tests intercept ``_stream_batches`` — stays real; this method is
        the same machinery for engines that compose a RunStore instead of subclassing.)
        """
        return write_checkpoint(
            self._storage,
            self._layout.checkpoint_uri(stage_name),
            records,
            wal_uri=self._layout.wal_uri if wal else None,
            stage_name=stage_name,
            tracer=tracer,
        )
