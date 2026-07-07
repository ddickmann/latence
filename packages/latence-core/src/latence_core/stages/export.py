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
    from collections.abc import Iterable

    from latence_core.contracts import Record
    from latence_core.storage import Storage


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
        rows: list[dict[str, Any]] = [r.model_dump(mode="json") for r in records]

        # Opt-in embeddings: attach one fixed-width vector per row (input order), embedding
        # the row's corpus text (the PII-handled masked_content when present). Absent embedder
        # ⇒ rows exported unchanged (embeddings are never required for Export — ADR-0017).
        if self._embedder is not None:
            vectors = list(self._embedder.embed(_corpus_text(row) for row in rows))
            if len(vectors) != len(rows):  # pragma: no cover - defensive: Embedder contract
                msg = (
                    f"Embedder returned {len(vectors)} vectors for {len(rows)} records; "
                    f"the Embedder contract yields one vector per input text, in order."
                )
                raise ValueError(msg)
            for row, vector in zip(rows, vectors, strict=True):
                row["embedding"] = vector

        jsonl_uri = f"{out_dir.rstrip('/')}/{self._basename}.jsonl"
        parquet_uri = f"{out_dir.rstrip('/')}/{self._basename}.parquet"

        # JSONL: one canonical (sorted-key) object per line for byte-stable output.
        lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
        blob = "\n".join(lines) + ("\n" if lines else "")
        storage.write_atomic(jsonl_uri, blob.encode("utf-8"))

        # Parquet: nested contract fields are serialized as JSON strings so the
        # schema stays flat and simple-Provider-authors friendly (ADR-0006). The
        # optional embedding is a native list<double> column so a vector DB reads it directly.
        embed = self._embedder is not None
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
        columns: dict[str, pa.Array] = {
            "record_id": pa.array([], type=pa.string()),
            "schema_version": pa.array([], type=pa.int64()),
            "provenance": pa.array([], type=pa.string()),
            "classification": pa.array([], type=pa.string()),
            "content": pa.array([], type=pa.string()),
            "media_type": pa.array([], type=pa.string()),
        }
        if embed:
            columns["embedding"] = pa.array([], type=pa.list_(pa.float64()))
        table = pa.Table.from_pylist(flat) if flat else pa.table(columns)
        with storage.open(parquet_uri, "wb") as fh:
            # pyarrow ships no type stubs; the call is dynamically typed.
            pq.write_table(table, fh)  # type: ignore[no-untyped-call]

        return [jsonl_uri, parquet_uri]
