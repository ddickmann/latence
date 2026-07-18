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


@dataclass(frozen=True, slots=True)
class ExportRow:
    """One exported record, projected into the three things a vector store needs.

    ``record_id`` is the corpus's own content-addressed identity (the idempotency key — see
    :func:`~latence_sink_qdrant.sink.point_id`), ``vector`` is the dense embedding, and ``payload``
    is every remaining exported column verbatim (including ``record_id`` itself, so the identity is
    filterable and readable back out of the store without decoding a point id).
    """

    record_id: str
    vector: list[float]
    payload: dict[str, Any]


def read_export(
    export_dir: str,
    storage: Storage,
    *,
    basename: str = "records",
    batch_size: int = 1024,
) -> Iterator[list[ExportRow]]:
    """Stream ``<export_dir>/<basename>.parquet`` as batches of at most ``batch_size`` rows.

    Raises :class:`~latence_sink_qdrant.errors.SinkExportError` — never a bare pyarrow error — when
    the file is absent (wrong directory, or the pipeline never reached Export), when the corpus was
    exported **without** an Embedder (no ``embedding`` column: there is no vector to load, and a
    payload-only collection would answer no vector query, so the sink says so instead of loading a
    useless index), when a row carries no vector or a vector of a different width than the first
    row's, or when a ``record_id`` is missing/empty.

    Yields nothing at all for an empty (but well-formed, embedding-bearing) corpus — an empty
    Export is a legitimate state after purging every document, and the caller's reconciling sweep
    still runs, which is precisely how "purge everything" reaches the store.
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
        if EMBEDDING_COLUMN not in columns:
            msg = (
                f"The Export at {uri!r} has no {EMBEDDING_COLUMN!r} column, so it carries no "
                "vectors to load. Wire an Embedder on the Export Stage (config "
                "`embedder: {provider: ..., config: {...}}`, ADR-0017) and re-run, then load."
            )
            raise SinkExportError(msg)
        if RECORD_ID_COLUMN not in columns:  # pragma: no cover - the Export always writes it
            msg = f"The Export at {uri!r} has no {RECORD_ID_COLUMN!r} column; it is not a corpus."
            raise SinkExportError(msg)

        for batch in parquet.iter_batches(batch_size=batch_size):  # type: ignore[no-untyped-call]
            rows: list[ExportRow] = []
            for raw in batch.to_pylist():
                row, width = _to_row(raw, width, uri)
                rows.append(row)
            if rows:
                yield rows


def _to_row(raw: dict[str, Any], width: int | None, uri: str) -> tuple[ExportRow, int]:
    """Project one raw Parquet row into an :class:`ExportRow`, pinning/checking the vector width."""
    record_id = raw.get(RECORD_ID_COLUMN)
    if not isinstance(record_id, str) or not record_id:
        msg = f"The Export at {uri!r} has a record with an empty {RECORD_ID_COLUMN!r}: {raw!r:.200}"
        raise SinkExportError(msg)

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
    return ExportRow(record_id=record_id, vector=values, payload=payload), width


__all__ = ["EMBEDDING_COLUMN", "RECORD_ID_COLUMN", "ExportRow", "read_export"]
