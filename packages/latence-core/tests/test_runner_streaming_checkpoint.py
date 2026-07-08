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
from _resume_equality import output_artifacts
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
    """Resume is *exact*: the resumed run's output is byte-identical to the first run's.

    H-E1 Part 1. The old assertion (``all(m.from_checkpoint)`` + ``contracts_complete``) is a
    boolean-not-a-property check — a resume that dropped/duplicated/reordered a record passes it
    (the P3-F1 sticky-resume bug class). This runs to completion, captures every output artifact
    (the exported ``records.{jsonl,parquet}`` + every Stage checkpoint), resumes the same
    ``run_id``, and asserts the resumed output is **byte-identical**. Byte-identity — not
    record-set identity — is the right assertion here because the dumps are deterministic
    (``model_dump_json`` line order is stable), so a correct resume reproduces the exact bytes.
    """
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    s = Storage.for_uri(store)

    report1 = LocalRunner().run(p, run_id="r1")
    first = output_artifacts(s, store, "r1")
    assert first, "the first run must have produced output artifacts"
    assert not any(m.from_checkpoint for m in report1.stages)  # fresh run ran every Stage

    report2 = LocalRunner().run(p, run_id="r1")
    second = output_artifacts(s, store, "r1")

    assert all(m.from_checkpoint for m in report2.stages)
    assert report2.contracts_complete
    # The property the old test never checked: the actual output is identical across resume.
    assert second == first, "resumed run's output artifacts diverged from the first run's"


def test_resume_equality_bites_on_a_corrupt_resume(docs_dir: Path, tmp_path: Path) -> None:
    """Prove the byte-equality assertion BITES: a corrupt resume must be caught, not passed.

    H-E1 Part 1 reviewer check. Run to completion; capture the export. Then DROP a record from
    the ``chunk`` checkpoint and delete the ``export`` checkpoint so the Export re-runs from the
    corrupted upstream — a resume that silently dropped a chunk. The re-exported corpus must now
    DIFFER from the first run's, so the byte-equality assertion (the one the upgraded resume test
    relies on) would fail — the resume test genuinely detects a dropped/duplicated/reordered
    record rather than just re-asserting a ``from_checkpoint`` flag.
    """
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/r1"

    LocalRunner().run(p, run_id="r1")
    first = output_artifacts(s, store, "r1")
    first_export = first["export/records.jsonl"]

    # Corrupt the chunk checkpoint: drop its last record. The fixture yields >1 chunk, so this
    # is a genuine record loss, not an empty-corpus no-op.
    ckpt = f"{base}/checkpoints/chunk.jsonl"
    lines = [line for line in s.read_text(ckpt).splitlines() if line.strip()]
    assert len(lines) >= 2, "fixture must produce >1 chunk for the drop to bite"
    s.write_bytes(ckpt, ("\n".join(lines[:-1]) + "\n").encode("utf-8"))
    # Delete the export checkpoint so the Export Stage re-runs from the (now corrupt) chunk
    # checkpoint instead of resuming its own finalized output.
    s._fs.rm(f"{base}/checkpoints/export.jsonl")  # noqa: SLF001

    LocalRunner().run(p, run_id="r1")
    corrupted = output_artifacts(s, store, "r1")

    # The equality assertion the resume test makes would now FAIL — the corrupt resume is caught.
    assert corrupted != first
    assert corrupted["export/records.jsonl"] != first_export
    # Specifically: a record was dropped from the exported corpus (fewer JSONL lines).
    assert corrupted["export/records.jsonl"].count(b"\n") < first_export.count(b"\n")


def test_no_tmp_checkpoint_left_behind(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_pipeline(docs_dir, store), run_id="r1")
    s = Storage.for_uri(store)
    ckpt_dir = f"{store}/_latence/runs/r1/checkpoints"
    leftover = [p for p in s.walk_files(ckpt_dir) if p.endswith(".tmp")]
    assert leftover == []
