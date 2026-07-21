"""Build a SPLADE sparse index over an Export corpus (the ``latence retrieval splade-index`` core).

Tutorial §4.2 (``docs/TUTORIAL-rfc-retrieval-proof.md``) needs a *real* SPLADE-v3 sparse leg over
the exported RFC corpus: every corpus chunk encoded to a term-weight
:class:`~latence_core.capability.SparseVector`, written to a queryable ``splade/`` artifact. This
module is the driver that does it — the pure, provider-agnostic half the CLI
(``latence retrieval splade-index``) wraps.

Design (codebase-design + ADR-0016 thin core): the driver depends only on the
:class:`~latence_core.capability.SparseEmbedder` **Protocol**, never on ``latence-splade`` or any
heavy model stack. The CLI resolves the real ``sparse.splade`` Provider
(:class:`~latence_splade.provider.SpladeCompositionEmbedder`, which loads ``naver/splade-v3`` via
``transformers``) through the Provider registry and hands it in; a test hands in a faithful fake
SparseEmbedder. So the export-reading, per-chunk emission, empty-vector handling and
license-manifest logic are exercised offline without weights, while production runs the real
checkpoint through the identical path.

What it emits under ``<out>`` (files are the source of truth, ADR-0017):

* ``index.parquet`` — one row per corpus chunk **with a non-empty sparse vector**: ``record_id`` +
  the parallel ``indices`` (``list<int64>``) / ``values`` (``list<double>``) columns a store's
  sparse index ingests (ADR-0050). A chunk whose text composes to the empty vector is *counted* (in
  the manifest's ``chunks_empty``) but not written — a sparse index carries only non-zero support.
* ``manifest.json`` — the honest ADR-0012 provenance: the ``model`` checkpoint, the
  ``weights_license`` the operator accepted, ``license_verified=false`` (this tool did **not**
  verify the weights license — it recorded the operator's acceptance, it did not check the model
  card), ``operator_accepted=true``, the model ``vocab_size`` and the chunk counts.

The redaction floor (ADR-0044) rides for free: the Export's ``content`` column is the *already
PII-handled* corpus text (``export._corpus_content`` writes ``masked_content`` when present), so the
sparse vectors are built over the same redacted text the corpus ships — never re-leaking a masked
surface into a sparse term.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from latence_core.errors import LatenceError
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.capability import SparseEmbedder, SparseVector

#: The Export column carrying the content-addressed record identity (the sparse index's key).
RECORD_ID_COLUMN = "record_id"
#: The Export column carrying the redacted corpus text a chunk's sparse vector is built over.
CONTENT_COLUMN = "content"
#: Basenames written under the ``--out`` artifact directory.
INDEX_BASENAME = "index.parquet"
MANIFEST_BASENAME = "manifest.json"
#: Rows pulled from the Export (and pushed to the sparse embedder) per batch — the O(batch) bound
#: that keeps the build memory-flat regardless of corpus size, mirroring the Export that wrote it.
_DEFAULT_BATCH_ROWS = 64


class SpladeIndexError(LatenceError):
    """The Export cannot be read as a SPLADE corpus (missing file/columns) — a loud, typed refusal.

    Never a bare pyarrow error and never a silent skip (ADR-0034): a missing ``records.parquet``, a
    corpus exported without the ``content`` column, or a missing ``record_id`` each surface here
    with a message that says exactly what to fix.
    """


@dataclass(frozen=True, slots=True)
class SpladeIndexResult:
    """The outcome of a SPLADE index build — the honest counts the CLI echoes, and the manifest.

    ``chunks_indexed + chunks_empty == chunks_total`` always: every corpus chunk is either written
    (non-empty sparse vector) or counted as empty (composed to no surviving terms), never dropped
    silently.
    """

    model_id: str
    weights_license: str
    license_verified: bool
    operator_accepted: bool
    vocab_size: int
    chunks_total: int
    chunks_indexed: int
    chunks_empty: int
    index_uri: str
    manifest_uri: str


def _index_schema() -> pa.Schema:
    """The ``index.parquet`` schema (ADR-0050): ``record_id`` + parallel ``indices``/``values``."""
    return pa.schema(
        [
            pa.field(RECORD_ID_COLUMN, pa.string()),
            pa.field("indices", pa.list_(pa.int64())),
            pa.field("values", pa.list_(pa.float64())),
        ]
    )


def _read_corpus_batches(
    storage: Storage, records_uri: str, batch_size: int
) -> Iterator[list[tuple[str, str]]]:
    """Stream ``records.parquet`` as batches of ``(record_id, content)`` — O(batch) memory.

    Reads only the two columns the SPLADE build needs, row group by row group, so a huge corpus
    costs O(batch) exactly like the Export that wrote it (ADR-0033). Raises a typed
    :class:`SpladeIndexError` (never a bare pyarrow error) when the file is absent or the required
    columns are missing.
    """
    if not storage.exists(records_uri):
        msg = (
            f"No Export corpus found at {records_uri!r}. Point --export at the directory the "
            "Export Stage wrote (it contains records.parquet + records.jsonl)."
        )
        raise SpladeIndexError(msg)
    with storage.open(records_uri, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)  # type: ignore[no-untyped-call]
        columns = set(parquet_file.schema_arrow.names)
        missing = {RECORD_ID_COLUMN, CONTENT_COLUMN} - columns
        if missing:
            names = ", ".join(sorted(missing))
            msg = (
                f"Export corpus {records_uri!r} is missing required column(s): {names}. "
                "A SPLADE index needs each chunk's record_id and its (redacted) content text."
            )
            raise SpladeIndexError(msg)
        for record_batch in parquet_file.iter_batches(  # type: ignore[no-untyped-call]
            batch_size=batch_size, columns=[RECORD_ID_COLUMN, CONTENT_COLUMN]
        ):
            record_ids = record_batch.column(RECORD_ID_COLUMN).to_pylist()
            contents = record_batch.column(CONTENT_COLUMN).to_pylist()
            yield [
                (str(rid), "" if text is None else str(text))
                for rid, text in zip(record_ids, contents, strict=True)
            ]


def build_splade_index(
    *,
    export_dir: str,
    out_dir: str,
    embedder: SparseEmbedder,
    model_id: str,
    weights_license: str,
    operator_accepted: bool = True,
    storage: Storage | None = None,
    batch_size: int = _DEFAULT_BATCH_ROWS,
) -> SpladeIndexResult:
    """Encode every Export chunk to a SPLADE :class:`SparseVector`, write the ``splade/`` artifact.

    Streams ``<export_dir>/records.parquet``, composes each chunk's sparse vector through the
    injected ``embedder`` (the real ``sparse.splade`` Provider in production, a faithful fake in
    tests), writes the non-empty vectors to ``<out_dir>/index.parquet`` and the honest
    license/provenance manifest to ``<out_dir>/manifest.json``, and returns the
    :class:`SpladeIndexResult` counts.

    ``operator_accepted`` records that the caller passed the non-commercial-license acceptance gate;
    ``license_verified`` is always ``False`` (this tool records the operator's acceptance — it does
    not read the model card, ADR-0012). Raises :class:`SpladeIndexError` for an unreadable Export.
    """
    records_uri = f"{export_dir.rstrip('/')}/records.parquet"
    out_dir = out_dir.rstrip("/")
    index_uri = f"{out_dir}/{INDEX_BASENAME}"
    manifest_uri = f"{out_dir}/{MANIFEST_BASENAME}"
    store = storage or Storage.for_uri(records_uri)
    store.makedirs(out_dir)

    schema = _index_schema()
    chunks_total = 0
    chunks_indexed = 0
    with store.open(index_uri, "wb") as index_handle:
        writer = pq.ParquetWriter(index_handle, schema)  # type: ignore[no-untyped-call]
        try:
            for batch in _read_corpus_batches(store, records_uri, batch_size):
                record_ids = [rid for rid, _ in batch]
                texts = (text for _, text in batch)
                vectors: list[SparseVector] = list(embedder.encode_sparse(texts))
                if len(vectors) != len(record_ids):
                    msg = (
                        f"sparse embedder returned {len(vectors)} vectors for {len(record_ids)} "
                        "chunks; the SparseEmbedder contract yields one vector per text, in order."
                    )
                    raise SpladeIndexError(msg)
                rows: list[dict[str, Any]] = []
                for record_id, vector in zip(record_ids, vectors, strict=True):
                    chunks_total += 1
                    if not vector.indices:
                        # Empty sparse vector (no surviving terms): counted, never written — a
                        # sparse index carries only non-zero support (ADR-0050).
                        continue
                    chunks_indexed += 1
                    rows.append(
                        {
                            RECORD_ID_COLUMN: record_id,
                            "indices": list(vector.indices),
                            "values": list(vector.values),
                        }
                    )
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))  # type: ignore[no-untyped-call]
            if chunks_indexed == 0:
                # Schema-only Parquet so a reader still gets the column set for an all-empty (or
                # empty) corpus — matching the Export's empty-corpus discipline.
                writer.write_table(schema.empty_table())  # type: ignore[no-untyped-call]
        finally:
            writer.close()  # type: ignore[no-untyped-call]

    # ``vocab_size`` fixes the term-index space the sparse vectors live in; reading it resolves the
    # model (in production the real checkpoint load — needed anyway; in tests the fake's constant).
    vocab_size = int(embedder.vocab_size)
    manifest = {
        "artifact": "splade-index",
        "model": model_id,
        "weights_license": weights_license,
        "license_verified": False,
        "operator_accepted": operator_accepted,
        "license_note": (
            "weights license recorded from the operator's --accept-noncommercial-license "
            "acceptance; NOT verified by this tool (ADR-0012): no model card was read. "
            "naver/splade-v3 weights are CC-BY-NC-SA-4.0 (non-commercial) — never a default."
        ),
        "vocab_size": vocab_size,
        "chunks_total": chunks_total,
        "chunks_indexed": chunks_indexed,
        "chunks_empty": chunks_total - chunks_indexed,
        "index_file": INDEX_BASENAME,
    }
    store.write_text(manifest_uri, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return SpladeIndexResult(
        model_id=model_id,
        weights_license=weights_license,
        license_verified=False,
        operator_accepted=operator_accepted,
        vocab_size=vocab_size,
        chunks_total=chunks_total,
        chunks_indexed=chunks_indexed,
        chunks_empty=chunks_total - chunks_indexed,
        index_uri=index_uri,
        manifest_uri=manifest_uri,
    )
