"""JSONL + Parquet Export Provider (ADR-0006, S1 spine).

Materializes records in both built-in serializations from the one contract:
JSONL (streaming, human-diffable, checkpoint-friendly) and Parquet/Arrow
(lakehouse handoff — Databricks/Snowflake read Stage outputs natively). Both
are written from the same Pydantic dumps, so they never drift.

**Opt-in embeddings (ADR-0017).** Embeddings are optional: the corpus exports its
text with full Provenance/Classification whether or not an Embedder is wired, so any
vector DB can ingest it. When the Stage ``config`` names an Embedder Provider under
``embedder`` (``{"provider": ..., "config": {...}}``), this Provider resolves that
Embedder from the plugin registry and attaches a fixed-width ``embedding`` vector to
every exported row (JSONL key + Parquet column), embedding the record's **corpus
text** — the ``masked_content`` of a RedactionRecord (the PII-handled variant) or the
``content`` of a Chunk/Document. The in-core ``embedding.hashing`` reference makes the
seam exercisable end to end offline; a served model is the pluggable
``embedding.endpoint`` package, swapped by one config line (the framework names the
seam, not the model).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from latence_core.capability import Embedder, ProviderRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import Record
    from latence_core.storage import Storage

# Row-group / JSONL flush granularity for the incremental Export (ADR-0033 / #63). The
# Export no longer materialises the whole corpus as one Python list + one Arrow table
# (``pa.Table.from_pylist(flat)``) — the single biggest transient G1 measured, coincident
# with peak RSS. Instead it streams the input in bounded batches: each batch becomes one
# Parquet row group (via a ``pq.ParquetWriter``) and one JSONL flush, so the Export's peak
# memory is O(batch), not O(corpus). Byte-for-byte identical output to the buffered path
# (same per-row dumps, same sorted-key JSONL, same flat Parquet schema + values) — only the
# transient footprint changes.
_EXPORT_BATCH_ROWS = 1024


def _corpus_text(row: dict[str, Any]) -> str:
    """The corpus text an Embedder embeds for a row.

    A RedactionRecord carries the PII-handled variant in ``masked_content``; a
    Chunk/Document carries its text in ``content``. Embedding ``masked_content`` when
    present keeps the vector consistent with the exported PII-handled text (never the
    raw, un-redacted text).
    """
    if "masked_content" in row:
        return str(row.get("masked_content") or "")
    return str(row.get("content") or "")


class JsonlParquetExport:
    """Export Provider writing ``records.jsonl`` and ``records.parquet``.

    Config:
        basename: str  — output file stem (default 'records').
        embedder: dict | None  — OPTIONAL opt-in Embedder augmentation (ADR-0017). A
            ``{"provider": "<entry-point name>", "config": {...}}`` block naming an
            :class:`~latence_core.capability.Embedder` Provider (e.g. the in-core
            ``embedding.hashing`` reference, or the pluggable ``embedding.endpoint``).
            When present, every exported row gets an ``embedding`` vector of the
            Embedder's fixed ``dimension``. Absent ⇒ no embeddings (the corpus text is
            still exported with full Provenance/Classification, so any vector DB can
            ingest it and embed later).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = config or {}
        self._basename = str(config.get("basename", "records"))
        self._embedder = self._build_embedder(config.get("embedder"), registry)

    @staticmethod
    def _build_embedder(
        spec: Any,
        registry: ProviderRegistry | None,
    ) -> Embedder | None:
        """Resolve an opt-in Embedder Provider from the ``embedder`` config block.

        ``spec`` is ``None`` (no embeddings) or a ``{"provider", "config"}`` mapping.
        The named Provider is loaded from the ``latence.providers`` registry and built,
        then checked against the :class:`Embedder` seam so a mis-named Provider fails
        loudly at wiring time rather than silently exporting no vectors.
        """
        if spec is None:
            return None
        if not isinstance(spec, dict):
            msg = (
                "Export config 'embedder' must be a mapping "
                "{'provider': <name>, 'config': {...}} or absent; "
                f"got {type(spec).__name__}."
            )
            raise TypeError(msg)
        provider_name = spec.get("provider")
        if not isinstance(provider_name, str) or not provider_name:
            msg = (
                "Export config 'embedder' requires a non-empty string 'provider' "
                "entry-point name."
            )
            raise ValueError(msg)
        embedder_config = spec.get("config", {})
        if not isinstance(embedder_config, dict):
            msg = "Export config 'embedder.config' must be a mapping if present."
            raise TypeError(msg)
        registry = registry or ProviderRegistry()
        embedder_cls = registry.load(provider_name)
        embedder = embedder_cls(embedder_config)
        if not isinstance(embedder, Embedder):
            msg = (
                f"Provider {provider_name!r} named in Export config 'embedder' does not "
                f"satisfy the Embedder capability (needs a 'dimension' property + an "
                f"'embed(texts)' method)."
            )
            raise TypeError(msg)
        return embedder

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        """Materialize records to ``<basename>.jsonl`` + ``<basename>.parquet``, streaming.

        Incremental (ADR-0033 / #63): the input is consumed in bounded batches of
        ``_EXPORT_BATCH_ROWS``; each batch is dumped, (optionally) embedded, JSONL-appended,
        and written as one Parquet row group — so the Export's peak memory is O(batch), not the
        old O(corpus) ``from_pylist(whole)`` transient (peak RSS coincided with it). Both files
        are written to sibling temp paths and atomically published (:meth:`Storage.temp_uri` +
        :meth:`Storage.finalize_atomic`), so a crash mid-Export leaves the previous file or the
        complete new one, never a torn write — the same WAL guarantee as the buffered path. The
        bytes are identical to the buffered path: same per-row Pydantic JSON dumps, same
        sorted-key JSONL line order + trailing-newline rule, same flat Parquet schema + values.
        """
        embed = self._embedder is not None
        schema = self._parquet_schema(embed)

        out_dir = out_dir.rstrip("/")
        # Streaming opens bypass write_bytes' implicit parent-makedirs, so ensure the export
        # directory exists before opening the temp files (a fresh run has no export/ yet).
        storage.makedirs(out_dir)
        jsonl_uri = f"{out_dir}/{self._basename}.jsonl"
        parquet_uri = f"{out_dir}/{self._basename}.parquet"
        jsonl_tmp = storage.temp_uri(jsonl_uri)
        parquet_tmp = storage.temp_uri(parquet_uri)

        wrote_any = False
        # Two streamed writers, both to temp paths, atomically published on success. The
        # JSONL is byte-stable: one sorted-key object per line, and the trailing newline is
        # written per line (so an N-line file ends in exactly one "\n"; an empty file is empty)
        # — identical to the old ``"\n".join(lines) + ("\n" if lines else "")`` construction.
        # The Parquet writer is handed the temp file handle directly (pyarrow streams row
        # groups into any binary file-like), so it goes through the same Storage seam.
        with (
            storage.open(jsonl_tmp, "wb") as jsonl_fh,
            storage.open(parquet_tmp, "wb") as parquet_fh,
        ):
            # pyarrow ships no type stubs; the writer calls are dynamically typed.
            writer = pq.ParquetWriter(parquet_fh, schema)  # type: ignore[no-untyped-call]
            try:
                for batch_rows in self._batched(records, _EXPORT_BATCH_ROWS):
                    self._attach_embeddings(batch_rows, embed)
                    for row in batch_rows:
                        line = json.dumps(row, sort_keys=True, ensure_ascii=False)
                        jsonl_fh.write((line + "\n").encode("utf-8"))
                    record_batch = self._record_batch(batch_rows, schema, embed)
                    writer.write_table(pa.Table.from_batches([record_batch], schema))  # type: ignore[no-untyped-call]
                    wrote_any = True
                if not wrote_any:
                    # No rows: still write the (empty) schema-only Parquet so a reader gets the
                    # column set, matching the buffered path's empty ``pa.table(columns)``.
                    writer.write_table(schema.empty_table())  # type: ignore[no-untyped-call]
            finally:
                writer.close()  # type: ignore[no-untyped-call]

        storage.finalize_atomic(jsonl_tmp, jsonl_uri)
        storage.finalize_atomic(parquet_tmp, parquet_uri)
        return [jsonl_uri, parquet_uri]

    def _attach_embeddings(self, rows: list[dict[str, Any]], embed: bool) -> None:
        """Attach one fixed-width embedding vector per row in-place (opt-in, per batch).

        Embeds the row's corpus text (the PII-handled ``masked_content`` when present, else
        ``content``). Embedding per batch keeps the vector transient O(batch). Absent embedder
        ⇒ rows unchanged (embeddings are never required for Export — ADR-0017).
        """
        if not embed or self._embedder is None:
            return
        vectors = list(self._embedder.embed(_corpus_text(row) for row in rows))
        if len(vectors) != len(rows):  # pragma: no cover - defensive: Embedder contract
            msg = (
                f"Embedder returned {len(vectors)} vectors for {len(rows)} records; "
                f"the Embedder contract yields one vector per input text, in order."
            )
            raise ValueError(msg)
        for row, vector in zip(rows, vectors, strict=True):
            row["embedding"] = vector

    @staticmethod
    def _parquet_schema(embed: bool) -> pa.Schema:
        """The flat Parquet schema (ADR-0006): nested contract fields as JSON strings.

        Fixing the schema up front lets the ``ParquetWriter`` stream row-group batches into one
        file (and lets an all-empty corpus still write the column set). The optional embedding
        is a native ``list<double>`` column so a vector DB reads it directly.
        """
        fields = [
            pa.field("record_id", pa.string()),
            pa.field("schema_version", pa.int64()),
            pa.field("provenance", pa.string()),
            pa.field("classification", pa.string()),
            pa.field("content", pa.string()),
            pa.field("media_type", pa.string()),
        ]
        if embed:
            fields.append(pa.field("embedding", pa.list_(pa.float64())))
        return pa.schema(fields)

    @staticmethod
    def _record_batch(
        rows: list[dict[str, Any]], schema: pa.Schema, embed: bool
    ) -> pa.RecordBatch:
        """One Arrow RecordBatch (a Parquet row group) for a batch of dumped rows.

        Same flat projection as the buffered path: nested Provenance/Classification serialized
        as sorted-key JSON strings, content/media_type verbatim, optional embedding native.
        """
        flat = [
            {
                "record_id": row["record_id"],
                "schema_version": row["schema_version"],
                "provenance": json.dumps(row["provenance"], sort_keys=True),
                "classification": json.dumps(row["classification"], sort_keys=True),
                "content": row.get("content", ""),
                "media_type": row.get("media_type", ""),
                **({"embedding": row["embedding"]} if embed else {}),
            }
            for row in rows
        ]
        return pa.RecordBatch.from_pylist(flat, schema=schema)

    @staticmethod
    def _batched(records: Iterable[Record], size: int) -> Iterator[list[dict[str, Any]]]:
        """Yield the records dumped to JSON-able rows in bounded batches of ``size``.

        The single place the corpus is pulled through: each record is ``model_dump``ed and
        buffered only until the batch fills, so at most ``size`` rows are live at once — the
        O(batch) bound that replaces the old whole-corpus list.
        """
        batch: list[dict[str, Any]] = []
        for record in records:
            batch.append(record.model_dump(mode="json"))
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch
