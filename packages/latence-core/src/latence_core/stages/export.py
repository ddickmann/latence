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
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

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


def _context_header_text(header: dict[str, Any]) -> str:
    """Render a chunk's :class:`~latence_core.contracts.ContextHeader` into a compact embedding
    prefix.

    W13 (ADR-0039): the OPTIONAL ``context_enrichment`` Stage stamps a chunk with a compact header
    (its canonical entities + top-k KG-neighbor triples); the Export prepends THIS rendered text to
    the EMBEDDING input only, so the coherence inflates the vector, never the stored chunk. Rendered
    deterministically (the header's collections are already sorted + capped): an ``Entities:``
    clause
    (``name (type)``) and a ``Relations:`` clause (the rendered ``A —label→ B`` triples). An empty
    header (no entities and no triples) renders ``""`` so a chunk with nothing to add embeds exactly
    its own text (byte-identical to no header).
    """
    entities = header.get("entities") or []
    entity_types = header.get("entity_types") or []
    triples = header.get("neighbor_triples") or []
    parts: list[str] = []
    if entities:
        rendered = ", ".join(
            f"{name} ({etype})" if etype else str(name)
            for name, etype in zip(entities, entity_types, strict=False)
        )
        parts.append(f"Entities: {rendered}")
    if triples:
        parts.append("Relations: " + "; ".join(str(t) for t in triples))
    return " | ".join(parts)


def _corpus_content(row: dict[str, Any]) -> str:
    """The PII-safe corpus text materialized for a row (``content`` column + JSONL).

    A redacted :class:`~latence_core.contracts.ChunkRecord` (W16, ADR-0042) carries the un-redacted
    text in ``content`` and the PII-handled variant in ``masked_content``; a deprecated
    :class:`~latence_core.contracts.RedactionRecord` carries only ``masked_content``. When
    ``masked_content`` is present and non-``None``, it IS the exported corpus text — so the RAG
    corpus never materializes the raw, un-redacted ``content`` (the PII-safety guarantee: the
    stack-validate rag-scan reads every exported cell). A row with no ``masked_content`` (no
    Redaction Stage, or a non-chunk carrier) exports its own ``content`` unchanged — byte-identical
    to a run without Redaction (the additive-field guarantee).
    """
    masked = row.get("masked_content")
    if masked is not None:
        return str(masked)
    return str(row.get("content") or "")


def _corpus_text(row: dict[str, Any]) -> str:
    """The corpus text an Embedder embeds for a row (the PII-safe text + optional W13 header).

    The base is the PII-safe :func:`_corpus_content` (``masked_content`` when present, else
    ``content``), so the vector is consistent with the exported PII-handled text (never the raw,
    un-redacted text).

    W13 (ADR-0039): when the row carries a :class:`~latence_core.contracts.ContextHeader`
    (``context_enrichment`` Stage upstream), its compact rendering is PREPENDED to the embedding
    input ONLY — ``header_text + "\\n" + text`` — so the KG coherence goes into the VECTOR while the
    stored corpus text stays byte-unchanged (the LLM window never grows). A row with no header (or
    an empty one) embeds exactly its own corpus text, byte-identical to today.
    """
    base = _corpus_content(row)
    header = row.get("context_header")
    if isinstance(header, dict):
        header_text = _context_header_text(header)
        if header_text:
            return f"{header_text}\n{base}"
    return base


def _context_columns_for_row(row: dict[str, Any]) -> dict[str, Any]:
    """The two W13 Parquet context columns for a dumped row (ADR-0039).

    ``context_header`` is the chunk's :class:`~latence_core.contracts.ContextHeader` as a sorted-key
    JSON string (``""`` when the row has no header — a redaction/document row, or a chunk the
    ``context_enrichment`` Stage left unenriched), and ``neighbor_entity_ids`` its
    ``neighbor_node_ids`` as a native ``list<string>`` (``[]`` when absent) — the query-time
    KG-expansion join keys a retriever filters/reranks on (query-time expansion is the retriever's
    job — ADR-0017; W13 only emits the keys).
    """
    header = row.get("context_header")
    if isinstance(header, dict):
        return {
            "context_header": json.dumps(header, sort_keys=True),
            "neighbor_entity_ids": list(header.get("neighbor_node_ids") or []),
        }
    return {"context_header": "", "neighbor_entity_ids": []}


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
        context_columns: bool  — OPTIONAL opt-in W13 projection (ADR-0039). When True, two extra
            Parquet columns are written so a downstream retriever/vector-DB can filter/hybrid-BM25/
            rerank on the KG projection: ``context_header`` (the chunk's
            :class:`~latence_core.contracts.ContextHeader` as a JSON string, ``""`` when absent) and
            ``neighbor_entity_ids`` (the header's ``neighbor_node_ids`` as a native
            ``list<string>``,
            the query-time KG-expansion join keys). Absent/False ⇒ the columns are NOT written and
            the
            Parquet schema is byte-identical to today (the genuinely-optional guarantee — a stack
            without the ``context_enrichment`` Stage never sets it). The JSONL always carries the
            full
            ``context_header`` (or ``null``) via the record dump, exactly like every other additive
            field, so no flag is needed there.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        The JSONL/Parquet Export materialises the RAG-ready corpus (pyarrow, Apache-2.0).
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012 — the
        package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call
        cost, so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        registry: ProviderRegistry | None = None,
    ) -> None:
        config = config or {}
        self._basename = str(config.get("basename", "records"))
        self._embedder = self._build_embedder(config.get("embedder"), registry)
        # W13 (ADR-0039): opt-in Parquet projection of the ContextHeader (byte-identical schema when
        # absent — the genuinely-optional guarantee).
        self._context_columns = bool(config.get("context_columns", False))

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
        schema = self._parquet_schema(embed, self._context_columns)

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
                    record_batch = self._record_batch(
                        batch_rows, schema, embed, self._context_columns
                    )
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
    def _parquet_schema(embed: bool, context_columns: bool = False) -> pa.Schema:
        """The flat Parquet schema (ADR-0006): nested contract fields as JSON strings.

        Fixing the schema up front lets the ``ParquetWriter`` stream row-group batches into one
        file (and lets an all-empty corpus still write the column set). The optional embedding
        is a native ``list<double>`` column so a vector DB reads it directly. The W13 context
        columns (``context_header`` JSON + ``neighbor_entity_ids`` native ``list<string>``) are
        added
        ONLY when opted in, so the default schema is byte-identical to today (ADR-0039).
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
        if context_columns:
            fields.append(pa.field("context_header", pa.string()))
            fields.append(pa.field("neighbor_entity_ids", pa.list_(pa.string())))
        return pa.schema(fields)

    @staticmethod
    def _record_batch(
        rows: list[dict[str, Any]],
        schema: pa.Schema,
        embed: bool,
        context_columns: bool = False,
    ) -> pa.RecordBatch:
        """One Arrow RecordBatch (a Parquet row group) for a batch of dumped rows.

        Same flat projection as the buffered path: nested Provenance/Classification serialized
        as sorted-key JSON strings, content/media_type verbatim, optional embedding native. When
        the W13 context columns are opted in, ``context_header`` is the chunk's ContextHeader as a
        sorted-key JSON string (``""`` when absent) and ``neighbor_entity_ids`` its neighbor node
        ids
        (the query-time KG-expansion join keys) as a native ``list<string>`` (``[]`` when absent).
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
                **(_context_columns_for_row(row) if context_columns else {}),
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
            row = record.model_dump(mode="json")
            # W16 (ADR-0042): materialize the PII-safe corpus text as the exported ``content``, so a
            # redacted chunk's raw ``content`` never reaches the corpus (neither the JSONL row nor
            # Parquet ``content`` column — the rag-scan reads every exported cell). Only substitutes
            # when the Redaction Stage set ``masked_content``; a chunk with no Redaction Stage
            # (``masked_content`` None) exports its ``content`` unchanged (byte-identical to today).
            if "content" in row and row.get("masked_content") is not None:
                row["content"] = row["masked_content"]
            batch.append(row)
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch
