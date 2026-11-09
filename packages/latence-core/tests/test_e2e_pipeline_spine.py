"""End-to-end S3 (the *Chunk* slice) through the local Runner on the LOCAL filesystem.

Note on the name: "S3" here is the framework's third *slice* (Chunk + Screening), NOT the
``s3://`` object store. This file runs on ``file://`` and was previously mis-named
``test_e2e_s3.py`` with an "end-to-end S3" docstring, which read as object-storage coverage it
never had (H-E1 Part 2). It is renamed to ``test_e2e_pipeline_spine.py`` and the test functions
to ``test_pipeline_spine_*``; the real ``s3://`` end-to-end lives in ``test_e2e_s3.py``,
which drives this **same** pipeline (the shared ``_spine_pipeline`` builder) over object storage.

Exercises every S3-slice acceptance criterion on a real Pipeline over a temp folder:

* Intake Screening runs BEFORE Parse and Quarantines a malicious/oversized fixture
  so it never reaches Parse.
* Chunk preserves offsets + page mapping + Provenance + Classification (round-trip:
  chunk → offset → original page).
* Content Screening runs AFTER Chunk and flags a prompt-injection fixture with a
  risk marker that survives into the exported corpus (so RAG consumers can exclude
  it).
* Quarantine dispositions + reasons and the chunk/screening metrics appear in the
  Quality Report.
* The whole run resumes idempotently from checkpoints + sidecars.
"""

from __future__ import annotations

import json
from pathlib import Path

from _resume_equality import output_artifacts
from _spine_pipeline import CORPUS, spine_pipeline
from latence_core.contracts import ChunkRecord
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _write_corpus(root: Path) -> Path:
    docs = root / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    return docs


def _pipeline(store: str, docs: Path) -> Pipeline:
    return spine_pipeline(store, f"file://{docs}")


def test_pipeline_spine_end_to_end(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s3e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s3e2e"

    # -- Intake Screening ran BEFORE Parse; the oversized file was Quarantined ---
    # It never reached Parse: no chunk / export record derives from it.
    intake_ckpt = f"{base}/checkpoints/intake.jsonl"
    passed_docs = [
        json.loads(line) for line in s.read_text(intake_ckpt).splitlines() if line.strip()
    ]
    passed_names = {d["provenance"]["file_name"] for d in passed_docs}
    assert passed_names == {"good.md", "inject.txt"}  # oversized dropped before Parse

    q_uri = f"{base}/quarantine/intake.jsonl"
    quarantined = [
        json.loads(line) for line in s.read_text(q_uri).splitlines() if line.strip()
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["provenance"]["file_name"] == "oversized.txt"
    assert quarantined[0]["category"] == "oversized"
    assert quarantined[0]["reason"]  # a human-readable reason is retained

    # -- Export holds the chunked corpus; NOTHING derives from the quarantined doc -
    jsonl_uri = f"{base}/export/records.jsonl"
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert rows, "expected exported chunks"
    src_docs = {r["provenance"]["file_name"] for r in rows}
    assert "oversized.txt" not in src_docs
    assert src_docs <= {"good.md", "inject.txt"}
    # Every exported record is a ChunkRecord that validates against the contract.
    for row in rows:
        chunk = ChunkRecord.model_validate(row)
        assert chunk.classification.sensitivity == "confidential"  # inherited losslessly
        assert chunk.provenance.char_start is not None
        assert chunk.provenance.page_start is not None

    # -- Chunk offsets round-trip: chunk → offset → original source page ---------
    # Rebuild each parent document's page map from the parse checkpoint and verify
    # every chunk's recorded page equals the page its offset resolves to.
    parse_ckpt = f"{base}/checkpoints/parse.jsonl"
    parsed = {
        json.loads(line)["record_id"]: json.loads(line)
        for line in s.read_text(parse_ckpt).splitlines()
        if line.strip()
    }
    from latence_core.contracts import PageMap

    for row in rows:
        parent = parsed[row["document_record_id"]]
        index = PageOffsetIndex(PageMap.model_validate(parent["page_map"]))
        resolved = index.page_for_offset(row["provenance"]["char_start"])
        assert row["provenance"]["page_start"] == resolved

    # -- Content Screening flagged the injection; the marker survives to the corpus
    flagged_rows = [r for r in rows if r.get("risk_markers")]
    assert flagged_rows, "prompt-injection chunk should be flagged"
    inj_flagged = [r for r in flagged_rows if r["provenance"]["file_name"] == "inject.txt"]
    assert inj_flagged
    marker = inj_flagged[0]["risk_markers"][0]
    assert marker["category"] == "prompt_injection"
    assert marker["detector"] == "screening.content_keyword"
    # The good document's chunks are NOT flagged.
    good_flagged = [r for r in flagged_rows if r["provenance"]["file_name"] == "good.md"]
    assert good_flagged == []

    # -- Quality Report: quarantine + flag dispositions and chunk metrics present -
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.screening is not None
    assert persisted.screening.quarantined == 1  # the oversized intake quarantine
    assert persisted.screening.flagged >= 1  # the propagated injection flag
    assert persisted.screening.intake_screened == 3  # all three raw docs screened
    # Every Quarantine finding carries its reason (auditable).
    q_findings = [
        f for f in persisted.screening.findings if f.verdict.value == "quarantine"
    ]
    assert q_findings and all(f.reason for f in q_findings)

    assert persisted.chunk is not None
    assert persisted.chunk.chunks_total == len(rows)
    assert persisted.chunk.offset_preserving is True  # every chunk kept its offsets
    assert persisted.chunk.chunks_flagged == len(flagged_rows)
    # Parse quality still present (chained through the S2 machinery).
    assert persisted.parse is not None
    assert persisted.parse.documents_parsed == 2

    # The returned object matches what was persisted.
    assert report.chunk is not None and report.chunk.chunks_total == persisted.chunk.chunks_total


def test_pipeline_spine_resume_is_idempotent(tmp_path: Path) -> None:
    """Resume is *exact* across the screening spine: EVERY output artifact is byte-identical.

    H-E1 Part 1. The old test compared only ``export/records.jsonl`` bytes (+ the
    ``from_checkpoint`` flags); it never checked the Parquet output or the per-Stage
    checkpoints, so a resume that corrupted a checkpoint (or drifted the Parquet) it could not
    see. It now captures the full output-artifact set (export ``records.{jsonl,parquet}`` + every
    Stage checkpoint) and asserts byte-identity across the resume.
    """
    docs = _write_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    s = Storage.for_uri(store)

    LocalRunner().run(pipeline, run_id="r")
    first = output_artifacts(s, store, "r")

    report2 = LocalRunner().run(pipeline, run_id="r")
    second = output_artifacts(s, store, "r")

    # Every Stage resumed from its checkpoint; the WHOLE output (jsonl + parquet + checkpoints)
    # is byte-identical, not just records.jsonl.
    assert all(m.from_checkpoint for m in report2.stages)
    assert second == first, "resumed spine output diverged from the first run's"
    # Screening metrics reconstruct identically from the persisted sidecars.
    assert report2.screening is not None
    assert report2.screening.quarantined == 1
    assert report2.chunk is not None
    assert report2.chunk.offset_preserving is True


def test_pipeline_spine_partial_kill_reruns_downstream_of_chunk(tmp_path: Path) -> None:
    """Delete the chunk+content+export checkpoints, resume: they re-run; upstream resumes."""
    docs = _write_corpus(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/r"

    LocalRunner().run(pipeline, run_id="r")
    for stage in ("chunk", "content", "export"):
        s._fs.rm(f"{base}/checkpoints/{stage}.jsonl")

    report = LocalRunner().run(pipeline, run_id="r")
    by_name = {m.name: m for m in report.stages}
    assert by_name["source"].from_checkpoint is True
    assert by_name["intake"].from_checkpoint is True
    assert by_name["parse"].from_checkpoint is True
    assert by_name["chunk"].from_checkpoint is False
    assert by_name["content"].from_checkpoint is False
    assert by_name["export"].from_checkpoint is False
    # The re-run still produces a valid flagged corpus.
    rows = [
        json.loads(line)
        for line in s.read_text(f"{base}/export/records.jsonl").splitlines()
        if line.strip()
    ]
    assert any(r.get("risk_markers") for r in rows)
