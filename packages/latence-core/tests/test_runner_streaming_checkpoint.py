"""Batch-granular streamed checkpoint writes (ADR-0033 / #63, regression).

The Runner used to build a Stage's entire checkpoint as one ``"\\n".join(...)`` string before
writing — an O(all-records) transient. It now streams the checkpoint in bounded batches
(``_CHECKPOINT_BATCH_RECORDS``): each batch is appended to a sibling temp file, logged to the
WAL as a ``stage_batch`` event, and the temp file is atomically published.

These tests pin the invariants the streamed path must keep:

* the on-disk checkpoint is byte-identical to the old buffered construction (so the delta
  engine's reader + the Airflow adapter's decode still round-trip);
* a Stage whose output spans MANY batches emits one ``stage_batch`` WAL event per batch (the
  bounded path is actually taken) with monotonic cumulative record counts;
* two runs over identical input+config produce byte-identical checkpoints (determinism / the
  G1 fingerprint bar);
* resume stays exact — a second run skips every Stage from its finalized checkpoint;
* no ``.tmp`` checkpoint is left behind (atomic publish).

Each would fail if the streamed checkpoint writer were reverted or broke atomicity.
"""

from __future__ import annotations

import json
from pathlib import Path

import latence_core.runner as runner_mod
import pytest
from latence_core.contracts import ChunkRecord
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(docs_dir: Path, store: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: t
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
"""
    )


def _buffered_checkpoint_bytes(records: list[ChunkRecord]) -> bytes:
    """The bytes the OLD buffered checkpoint writer produced, for byte-identity."""
    payload = "\n".join(r.model_dump_json() for r in records)
    return (payload + ("\n" if records else "")).encode("utf-8")


def test_streamed_checkpoint_is_byte_identical_to_buffered(
    docs_dir: Path, tmp_path: Path
) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    runner = LocalRunner()
    runner.run(p, run_id="r1")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/r1"

    # Re-read the chunk checkpoint through the Runner's own decoder, then re-serialize with the
    # buffered construction and compare bytes: the streamed writer must produce the identical
    # newline-delimited file.
    ckpt_uri = f"{base}/checkpoints/chunk.jsonl"
    on_disk = s.read_bytes(ckpt_uri)
    raw = runner._read_checkpoint(s, ckpt_uri, ChunkRecord)  # noqa: SLF001
    records = [r for r in raw if isinstance(r, ChunkRecord)]
    assert len(records) == len(raw)
    assert on_disk == _buffered_checkpoint_bytes(records)
    # Non-empty corpus ⇒ file ends in exactly one newline, one record per line.
    assert on_disk.endswith(b"\n")
    assert on_disk.count(b"\n") == len(records)


def test_stage_batch_wal_events_emitted_per_batch(
    docs_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a tiny batch so the small fixture corpus spans multiple checkpoint batches, proving
    # the bounded streamed path is taken (not one whole-string write).
    monkeypatch.setattr(runner_mod, "_CHECKPOINT_BATCH_RECORDS", 1)

    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    LocalRunner().run(p, run_id="r1")
    s = Storage.for_uri(store)
    wal = s.read_text(f"{store}/_latence/runs/r1/wal.log")
    events = [json.loads(line) for line in wal.splitlines() if line.strip()]

    batch_events = [e for e in events if e["event"] == "stage_batch"]
    # At least one batch event per Stage that produced records (source/parse/chunk/export).
    assert batch_events, "expected per-batch checkpoint WAL events"
    # The chunk Stage produced >1 chunk; with batch size 1 it must emit >1 batch event.
    chunk_batches = [e for e in batch_events if e["stage"] == "chunk"]
    assert len(chunk_batches) >= 2
    # Cumulative record counts are monotonic non-decreasing within a Stage.
    counts = [e["records"] for e in chunk_batches]
    assert counts == sorted(counts)
    assert counts[-1] == len(chunk_batches)  # one record per batch at batch size 1


def test_streamed_checkpoints_are_deterministic_across_runs(
    docs_dir: Path, tmp_path: Path
) -> None:
    # Determinism / the G1 fingerprint bar: identical input+config ⇒ byte-identical checkpoints.
    store_a = f"file://{tmp_path}/a"
    store_b = f"file://{tmp_path}/b"
    LocalRunner().run(_pipeline(docs_dir, store_a), run_id="r")
    LocalRunner().run(_pipeline(docs_dir, store_b), run_id="r")
    sa = Storage.for_uri(store_a)
    sb = Storage.for_uri(store_b)
    for stage in ("source", "parse", "chunk", "export"):
        a = sa.read_bytes(f"{store_a}/_latence/runs/r/checkpoints/{stage}.jsonl")
        b = sb.read_bytes(f"{store_b}/_latence/runs/r/checkpoints/{stage}.jsonl")
        assert a == b, f"non-deterministic checkpoint for stage {stage}"


def test_resume_exact_after_streamed_write(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    LocalRunner().run(p, run_id="r1")
    report2 = LocalRunner().run(p, run_id="r1")
    assert all(m.from_checkpoint for m in report2.stages)
    assert report2.contracts_complete


def test_no_tmp_checkpoint_left_behind(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_pipeline(docs_dir, store), run_id="r1")
    s = Storage.for_uri(store)
    ckpt_dir = f"{store}/_latence/runs/r1/checkpoints"
    leftover = [p for p in s.walk_files(ckpt_dir) if p.endswith(".tmp")]
    assert leftover == []
