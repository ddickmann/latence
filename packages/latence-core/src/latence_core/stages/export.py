"""JSONL + Parquet Export Provider (ADR-0006, S1 spine).

Materializes records in both built-in serializations from the one contract:
JSONL (streaming, human-diffable, checkpoint-friendly) and Parquet/Arrow
(lakehouse handoff — Databricks/Snowflake read Stage outputs natively). Both
are written from the same Pydantic dumps, so they never drift.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import Record
    from latence_core.storage import Storage


class JsonlParquetExport:
    """Export Provider writing ``records.jsonl`` and ``records.parquet``.

    Config:
        basename: str  — output file stem (default 'records').
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._basename = str(config.get("basename", "records"))

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        rows: list[dict[str, Any]] = [r.model_dump(mode="json") for r in records]

        jsonl_uri = f"{out_dir.rstrip('/')}/{self._basename}.jsonl"
        parquet_uri = f"{out_dir.rstrip('/')}/{self._basename}.parquet"

        # JSONL: one canonical (sorted-key) object per line for byte-stable output.
        lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
        blob = "\n".join(lines) + ("\n" if lines else "")
        storage.write_atomic(jsonl_uri, blob.encode("utf-8"))

        # Parquet: nested contract fields are serialized as JSON strings so the
        # schema stays flat and simple-Provider-authors friendly (ADR-0006).
        flat = [
            {
                "record_id": row["record_id"],
                "schema_version": row["schema_version"],
                "provenance": json.dumps(row["provenance"], sort_keys=True),
                "classification": json.dumps(row["classification"], sort_keys=True),
                "content": row.get("content", ""),
                "media_type": row.get("media_type", ""),
            }
            for row in rows
        ]
        table = (
            pa.Table.from_pylist(flat)
            if flat
            else pa.table(
                {
                    "record_id": pa.array([], type=pa.string()),
                    "schema_version": pa.array([], type=pa.int64()),
                    "provenance": pa.array([], type=pa.string()),
                    "classification": pa.array([], type=pa.string()),
                    "content": pa.array([], type=pa.string()),
                    "media_type": pa.array([], type=pa.string()),
                }
            )
        )
        with storage.open(parquet_uri, "wb") as fh:
            # pyarrow ships no type stubs; the call is dynamically typed.
            pq.write_table(table, fh)  # type: ignore[no-untyped-call]

        return [jsonl_uri, parquet_uri]
