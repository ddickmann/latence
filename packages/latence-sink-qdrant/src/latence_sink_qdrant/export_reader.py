"""Read the Export's ``records.parquet`` as a bounded stream of loadable rows.

The Export is the deliverable and the source of truth (ADR-0017); this module is the *only* place
the sink touches it, and it touches it read-only. It streams **row groups**, never the whole table,
so loading a 10-million-record corpus costs O(batch) memory exactly like the Export that wrote it
(ADR-0033) — the sink can run beside the pipeline on the same box.

What it guarantees to the loader above it:

* every yielded :class:`ExportRow` has a non-empty ``record_id`` and a vector of the corpus's one
  fixed width (the first row's width fixes it; any other width is a loud
  :class:`~latence_sink_qdrant.errors.SinkExportError`, because a vector index has exactly one
  dimension and a real store rejects the odd one out mid-upload);
* ``payload`` is every *other* exported column, verbatim as exported — the flat Parquet projection
  is already store-friendly (nested Provenance/Classification are JSON strings, the signal columns
  are native lists), so the sink neither reshapes nor reinterprets the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from latence_sink_qdrant.errors import SinkExportError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.storage import Storage

#: The Export column carrying the dense vector (``JsonlParquetExport`` writes it as a native
#: ``list<double>`` when — and only when — an Embedder is wired, ADR-0017).
EMBEDDING_COLUMN = "embedding"
#: The Export column carrying the content-addressed record identity — the sink's upsert key.
RECORD_ID_COLUMN = "record_id"
#: The default Export column-pair *stem* for the sparse retrieval signal. ``JsonlParquetExport``
#: writes ``sparse_indices`` (native ``list<int64>``) + ``sparse_values`` (native ``list<double>``)
#: when — and only when — a ``sparse_embedder`` is wired (ADR-0050/0053). A different SPLADE
#: checkpoint exported under a different stem is named with ``sparse_source``.
DEFAULT_SPARSE_SOURCE = "sparse"


def sparse_columns(source: str) -> tuple[str, str]:
    """The ``(indices, values)`` Export column names for a sparse signal named ``source``."""
    return f"{source}_indices", f"{source}_values"


@dataclass(frozen=True, slots=True)
class SparseSignal:
    """One record's sparse vector as the Export carries it: two parallel lists.

    Term ids and their weights, kept in the Export's own order (a sparse index is a set of
    ``(term, weight)`` pairs, so the order is not meaning) and validated *here* rather than at the
    wire: an Export whose two columns disagree in length is a corrupt Export, and finding that out
    as an opaque 400 from the store halfway through a load is exactly the deferred-error case
    ADR-0034 exists to prevent.
    """

    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class ExportRow:
    """One exported record, projected into the things a vector store needs.

    ``record_id`` is the corpus's own content-addressed identity (the idempotency key — see
    :func:`~latence_sink_qdrant.sink.point_id`), ``vector`` is the dense embedding (empty when the
    caller asked for a sparse-only load), ``sparse`` is the named sparse signal when one was asked
    for, and ``payload`` is every remaining exported column verbatim (including ``record_id``
    itself, so the identity is filterable and readable back out of the store without decoding a
    point id).

    The sparse columns are **removed** from ``payload`` when they are read as a vector. That is the
    whole defect this field exists to fix: they used to ride into the payload as ordinary lists —
    stored, and never queryable by the sparse ANN leg that was already asking for them.
    """

    record_id: str
    vector: list[float]
    payload: dict[str, Any]
    sparse: SparseSignal | None = None


def read_export(
    export_dir: str,
    storage: Storage,
    *,
    basename: str = "records",
    batch_size: int = 1024,
    dense: bool = True,
    sparse_source: str | None = None,
) -> Iterator[list[ExportRow]]:
    """Stream ``<export_dir>/<basename>.parquet`` as batches of at most ``batch_size`` rows.

    ``dense`` and ``sparse_source`` say which signals the caller intends to *load*, and each is
    required to exist only because it was asked for — a sparse-only (SPLADE) collection is a real
    shape, so demanding an ``embedding`` column for it would refuse a correct Export.

    Raises :class:`~latence_sink_qdrant.errors.SinkExportError` — never a bare pyarrow error — when
    the file is absent (wrong directory, or the pipeline never reached Export), when a dense load
    was asked for over a corpus exported **without** an Embedder (no ``embedding`` column: there is
    no vector to load, and a payload-only collection would answer no vector query, so the sink says
    so instead of loading a useless index), when a row carries no vector or a vector of a different
    width than the first row's, when a sparse load was asked for and the named column pair is
    absent or its two columns disagree in length, or when a ``record_id`` is missing/empty.

    Yields nothing at all for an empty (but well-formed) corpus — an empty Export is a legitimate
    state after purging every document, and the caller's reconciling sweep still runs, which is
    precisely how "purge everything" reaches the store.
    """
    uri = f"{export_dir.rstrip('/')}/{basename}.parquet"
    if not storage.exists(uri):
        msg = (
            f"No Export found at {uri!r}. Point the sink at the directory the Export Stage wrote "
            f"(it contains {basename}.parquet + {basename}.jsonl), or pass a different --basename."
        )
        raise SinkExportError(msg)

    width: int | None = None
    with storage.open(uri, "rb") as fh:
        parquet = pq.ParquetFile(fh)  # type: ignore[no-untyped-call]
        columns = set(parquet.schema_arrow.names)
        if dense and EMBEDDING_COLUMN not in columns:
            msg = (
                f"The Export at {uri!r} has no {EMBEDDING_COLUMN!r} column, so it carries no "
                "vectors to load. Wire an Embedder on the Export Stage (config "
                "`embedder: {provider: ..., config: {...}}`, ADR-0017) and re-run, then load — or "
                "load the sparse signal only, with `vector: sparse`."
            )
            raise SinkExportError(msg)
        if RECORD_ID_COLUMN not in columns:  # pragma: no cover - the Export always writes it
            msg = f"The Export at {uri!r} has no {RECORD_ID_COLUMN!r} column; it is not a corpus."
            raise SinkExportError(msg)
        if sparse_source is not None:
            _require_sparse_columns(columns, sparse_source, uri)

        ordinal = 0
        for batch in parquet.iter_batches(batch_size=batch_size):  # type: ignore[no-untyped-call]
            rows: list[ExportRow] = []
            for raw in batch.to_pylist():
                row, width = _to_row(raw, width, uri, ordinal, dense, sparse_source)
                ordinal += 1
                rows.append(row)
            if rows:
                yield rows


def _require_sparse_columns(columns: set[str], source: str, uri: str) -> None:
    """Both halves of the sparse signal must be there, and the message names both.

    Asking for a signal the Export does not carry is an operator/wiring fault, and the useful
    message is the *column names it looked for* — the stem is configurable, so "no sparse vectors"
    alone would leave the operator guessing between a missing ``sparse_embedder`` on the Export
    Stage and a mistyped ``--sparse-source``.
    """
    indices, values = sparse_columns(source)
    missing = sorted({indices, values} - columns)
    if missing:
        msg = (
            f"The Export at {uri!r} does not carry the sparse signal {source!r}: column(s) "
            f"{missing} are absent (it needs both {indices!r} and {values!r}). Wire a "
            "`sparse_embedder` on the Export Stage (ADR-0050/0053) and re-run, or name the stem "
            "the Export actually wrote with --sparse-source."
        )
        raise SinkExportError(msg)


def read_record_ids(
    export_dir: str,
    storage: Storage,
    *,
    basename: str = "records",
    batch_size: int = 1024,
) -> Iterator[list[str]]:
    """Stream just the ``record_id`` column of the Export, in batches of at most ``batch_size``.

    The reconciling sweep needs one question answered about a candidate point — *is this
    ``record_id`` still in the Export?* — and the Export is the only authority on it (ADR-0017). It
    reads a single column and one row group at a time, so asking costs O(batch) memory (ADR-0033)
    and never decodes a vector.

    Unlike :func:`read_export` this does **not** require an ``embedding`` column: it answers a
    membership question, not a load, and the caller has already established that the Export is
    loadable. A missing file is still a loud
    :class:`~latence_sink_qdrant.errors.SinkExportError` — an unreadable Export must never be read
    as "nothing is exported", which would make every stored point look retracted.
    """
    uri = f"{export_dir.rstrip('/')}/{basename}.parquet"
    if not storage.exists(uri):
        msg = (
            f"No Export found at {uri!r} while reconciling. The sink will not treat an unreadable "
            "Export as an empty corpus — that would sweep every point in the collection."
        )
        raise SinkExportError(msg)
    with storage.open(uri, "rb") as fh:
        parquet = pq.ParquetFile(fh)  # type: ignore[no-untyped-call]
        if RECORD_ID_COLUMN not in set(parquet.schema_arrow.names):  # pragma: no cover - always set
            msg = f"The Export at {uri!r} has no {RECORD_ID_COLUMN!r} column; it is not a corpus."
            raise SinkExportError(msg)
        for batch in parquet.iter_batches(  # type: ignore[no-untyped-call]
            batch_size=batch_size, columns=[RECORD_ID_COLUMN]
        ):
            ids = [rid for rid in batch.column(0).to_pylist() if isinstance(rid, str) and rid]
            if ids:
                yield ids


def _to_row(
    raw: dict[str, Any],
    width: int | None,
    uri: str,
    ordinal: int,
    dense: bool = True,
    sparse_source: str | None = None,
) -> tuple[ExportRow, int]:
    """Project one raw Parquet row into an :class:`ExportRow`, pinning/checking the vector width.

    A bad row is named by its **position** in the Export and by the *names* of the columns it
    carries — never by their values. Every error an operator reads is counts-and-schema only
    (ADR-0034): the ``content`` column is redacted chunk text, and a message that quotes it would
    walk the corpus straight into a log file and the ticket pasted from it (ADR-0044).
    """
    record_id = raw.get(RECORD_ID_COLUMN)
    if not isinstance(record_id, str) or not record_id:
        msg = (
            f"The Export at {uri!r} has a record with an empty {RECORD_ID_COLUMN!r} at row "
            f"{ordinal} (0-based). Its columns are {sorted(raw)}; the values are not shown because "
            "an Export carries corpus content. Re-run the Export, or drop the row."
        )
        raise SinkExportError(msg)

    sparse = _to_sparse(raw, record_id, uri, sparse_source) if sparse_source else None
    if not dense:
        # A sparse-only (SPLADE) load: there is no dense side to demand, and the width stays 0 so
        # the created collection declares no dense vector at all.
        payload = {k: v for k, v in raw.items() if k != EMBEDDING_COLUMN}
        _drop_sparse(payload, sparse_source)
        return ExportRow(record_id=record_id, vector=[], payload=payload, sparse=sparse), 0

    vector = raw.get(EMBEDDING_COLUMN)
    if vector is None:
        msg = (
            f"Record {record_id!r} in {uri!r} has a null {EMBEDDING_COLUMN!r}. Every record must "
            "carry a vector for a vector-store load; re-run the Export with the Embedder wired."
        )
        raise SinkExportError(msg)
    values = [float(v) for v in vector]
    if width is None:
        width = len(values)
        if width == 0:
            msg = (
                f"Record {record_id!r} in {uri!r} has a zero-width {EMBEDDING_COLUMN!r}; a vector "
                "index cannot be created with dimension 0."
            )
            raise SinkExportError(msg)
    elif len(values) != width:
        msg = (
            f"Record {record_id!r} in {uri!r} has a {len(values)}-dimensional vector but the "
            f"corpus's first record is {width}-dimensional. A collection has exactly ONE "
            "dimension, so this Export cannot be loaded; re-embed the corpus with a single "
            "Embedder."
        )
        raise SinkExportError(msg)

    payload = {k: v for k, v in raw.items() if k != EMBEDDING_COLUMN}
    _drop_sparse(payload, sparse_source)
    return ExportRow(record_id=record_id, vector=values, payload=payload, sparse=sparse), width


def _drop_sparse(payload: dict[str, Any], source: str | None) -> None:
    """Take the sparse columns *out* of the payload once they are being loaded as a vector.

    Leaving them in would duplicate every term id and weight in the point's payload — the exact
    shape the sink shipped before this (stored, never queryable) — and would also mean a re-embed
    of the sparse signal changed both the vector and the payload, doubling the write for no gain.
    """
    if source is None:
        return
    for name in sparse_columns(source):
        payload.pop(name, None)


def _to_sparse(
    raw: dict[str, Any], record_id: str, uri: str, source: str
) -> SparseSignal:
    """Project one row's sparse column pair, refusing anything a sparse index cannot hold.

    Named by ``record_id`` and by *counts* only — never by the term ids or weights themselves. The
    sparse signal is derived from redacted corpus text (ADR-0053), and a message that printed it
    would walk term-level content into a log file (ADR-0034/0044).
    """
    indices_column, values_column = sparse_columns(source)
    indices = raw.get(indices_column)
    values = raw.get(values_column)
    if indices is None or values is None:
        missing = [n for n, v in ((indices_column, indices), (values_column, values)) if v is None]
        msg = (
            f"Record {record_id!r} in {uri!r} has a null {missing} — every record of a sparse load "
            "must carry both halves of the signal. Re-run the Export with the sparse_embedder "
            "wired, or load without --sparse-source."
        )
        raise SinkExportError(msg)
    if len(indices) != len(values):
        msg = (
            f"Record {record_id!r} in {uri!r} has {len(indices)} {indices_column!r} but "
            f"{len(values)} {values_column!r}. A sparse vector is two PARALLEL columns; this "
            "Export is corrupt and a real Qdrant would reject the point with an opaque 400."
        )
        raise SinkExportError(msg)
    out_indices: list[int] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            msg = (
                f"Record {record_id!r} in {uri!r} has a {indices_column!r} entry that is not a "
                "non-negative integer term id; the value is not shown because an Export carries "
                "corpus-derived content."
            )
            raise SinkExportError(msg)
        out_indices.append(index)
    if len(set(out_indices)) != len(out_indices):
        msg = (
            f"Record {record_id!r} in {uri!r} repeats a term id in {indices_column!r} "
            f"({len(out_indices)} entries, {len(set(out_indices))} distinct). A sparse vector maps "
            "each term to ONE weight, so a real Qdrant rejects the point with an opaque 400 — "
            "caught here instead, before the collection is written to."
        )
        raise SinkExportError(msg)
    return SparseSignal(indices=out_indices, values=[float(v) for v in values])


__all__ = [
    "DEFAULT_SPARSE_SOURCE",
    "EMBEDDING_COLUMN",
    "RECORD_ID_COLUMN",
    "ExportRow",
    "SparseSignal",
    "read_export",
    "read_record_ids",
    "sparse_columns",
]
