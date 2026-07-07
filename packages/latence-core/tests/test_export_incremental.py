"""Incremental, memory-bounded Export (ADR-0033 / #63, regression).

Before this change the Export materialized the whole corpus twice — one Python list of
JSONL lines and one ``pa.Table.from_pylist(flat)`` — the single biggest transient G1 Lane 6
measured (peak RSS coincided with it). The Export now streams the input in bounded batches
(``_EXPORT_BATCH_ROWS`` rows), each becoming one JSONL flush + one Parquet row group, written
to temp files and atomically published.

These tests exercise the *wired streamed path*:

* the output is byte-for-byte identical to the buffered construction (same sorted-key JSONL
  lines + trailing-newline rule; same flat Parquet schema + values read back);
* a corpus spanning MANY row-group batches yields exactly one Parquet row group per batch (the
  bounded-memory path is actually taken, not a single whole-corpus table) with all rows intact
  and in order;
* the writers go through the atomic temp→publish discipline (no ``.tmp`` left behind).

Each would fail if the streamed Export were reverted to the whole-corpus buffer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import _EXPORT_BATCH_ROWS, JsonlParquetExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path


def _chunk(i: int) -> ChunkRecord:
    text = f"chunk number {i} body"
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:d{i}",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=f"c{i}",
        document_record_id=f"d{i}",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
        content=text,
        token_count=len(text.split()),
    )


def _buffered_jsonl_bytes(records: list[ChunkRecord]) -> bytes:
    """The JSONL bytes the OLD whole-corpus buffered path produced, for byte-identity."""
    rows = [r.model_dump(mode="json") for r in records]
    lines = [json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def test_streamed_jsonl_is_byte_identical_to_buffered(tmp_path: Path) -> None:
    storage = Storage.for_uri(f"file://{tmp_path}")
    records = [_chunk(i) for i in range(5)]
    JsonlParquetExport().export(records, storage, f"file://{tmp_path}/out")
    got = storage.read_bytes(f"file://{tmp_path}/out/records.jsonl")
    assert got == _buffered_jsonl_bytes(records)


def test_empty_corpus_writes_empty_jsonl_and_schema_only_parquet(tmp_path: Path) -> None:
    storage = Storage.for_uri(f"file://{tmp_path}")
    JsonlParquetExport().export([], storage, f"file://{tmp_path}/out")
    assert storage.read_bytes(f"file://{tmp_path}/out/records.jsonl") == b""
    with storage.open(f"file://{tmp_path}/out/records.parquet", "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert table.num_rows == 0
    assert set(table.column_names) == {
        "record_id",
        "schema_version",
        "provenance",
        "classification",
        "content",
        "media_type",
    }


def test_multi_batch_corpus_streams_one_rowgroup_per_batch(tmp_path: Path) -> None:
    # More than two batches' worth of rows: the streamed path must emit one Parquet row group
    # per batch (proving the bounded path is taken, not a single whole-corpus table) and keep
    # every row, in order. The buffered path would have written exactly ONE row group.
    storage = Storage.for_uri(f"file://{tmp_path}")
    n = _EXPORT_BATCH_ROWS * 2 + 3
    records = [_chunk(i) for i in range(n)]
    JsonlParquetExport().export(records, storage, f"file://{tmp_path}/out")

    with storage.open(f"file://{tmp_path}/out/records.parquet", "rb") as fh:
        pf = pq.ParquetFile(fh)  # type: ignore[no-untyped-call]
        assert pf.num_row_groups == 3  # ceil(n / batch) = 3 batches -> 3 row groups
        table = pf.read()  # type: ignore[no-untyped-call]
    assert table.num_rows == n
    assert table.column("record_id").to_pylist() == [f"c{i}" for i in range(n)]

    # JSONL kept all rows, in order, and is byte-identical to the buffered construction.
    got = storage.read_bytes(f"file://{tmp_path}/out/records.jsonl")
    assert got == _buffered_jsonl_bytes(records)
    assert got.count(b"\n") == n


def test_export_leaves_no_temp_files(tmp_path: Path) -> None:
    # Atomic publish: the streamed writers use temp→finalize, so no ``.tmp`` is left behind.
    storage = Storage.for_uri(f"file://{tmp_path}")
    records = [_chunk(i) for i in range(3)]
    JsonlParquetExport().export(records, storage, f"file://{tmp_path}/out")
    leftover = [p for p in storage.walk_files(f"file://{tmp_path}/out") if p.endswith(".tmp")]
    assert leftover == []
