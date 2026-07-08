"""End-to-end S6 Redaction through the local Runner.

Source -> Parse -> Chunk -> Redaction -> Export, using the in-core deterministic
``redaction.hybrid_rule``. Asserts the exported records are RedactionRecords whose
masked_content has the PII removed and whose PII spans resolve to the correct source
page (the S6 round-trip), that the sensitivity-keyed policy drives the disposition,
that the Quality Report's RedactionQuality carries per-type + per-action counts, that
no raw PII value appears anywhere in the persisted artifacts (S6 AC), and that the run
is deterministic and resumes cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

from _resume_equality import output_artifacts
from latence_core.contracts import RedactionRecord
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

_REDACTION_YAML = """
      detectors: [email, phone]
      default_action: mask
      policies:
        internal:
          action: mask
"""


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: redaction-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact
    capability: redaction
    provider: redaction.hybrid_rule
    depends_on: [parse]
    config:{_REDACTION_YAML}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Please email alice@acme.com or call 555-123-4567 for access.\n", encoding="utf-8"
    )
    # A document with no PII: handled without error (clean masked variant).
    (docs / "clean.txt").write_text("The quarterly report is attached for review.\n", encoding="utf-8")
    # An empty document.
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_redaction_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s6e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s6e2e"

    # 1. The export holds typed RedactionRecord rows, one per source document.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [RedactionRecord.model_validate(r) for r in rows]
    assert records
    memo = next(r for r in records if "[EMAIL]" in r.masked_content)

    # 2. PII is removed from the masked variant; the raw values are gone.
    assert "alice@acme.com" not in memo.masked_content
    assert "555-123-4567" not in memo.masked_content
    assert "[EMAIL]" in memo.masked_content
    assert "[PHONE]" in memo.masked_content
    assert {s.pii_type for s in memo.pii_spans} == {"email", "phone"}
    assert memo.policy == "internal"
    assert memo.classification.sensitivity == "internal"

    # 3. The S6 round-trip: each PII span resolves to the correct source page.
    from latence_core.contracts import DocumentRecord

    parse_ckpt = f"{base}/checkpoints/parse.jsonl"
    docs_by_id = {
        d.record_id: d
        for d in (
            DocumentRecord.model_validate_json(line)
            for line in s.read_text(parse_ckpt).splitlines()
            if line.strip()
        )
    }
    for span in memo.pii_spans:
        parent = docs_by_id[memo.document_record_id]
        assert parent.page_map is not None
        idx = PageOffsetIndex(parent.page_map)
        assert idx.page_for_offset(span.char_start) == span.page_start

    # 4. A clean document yields a record with no spans and untouched text.
    clean = next(r for r in records if r.pii_count == 0 and r.masked_content.strip())
    assert "quarterly report" in clean.masked_content

    # 5. Quality Report: RedactionQuality present with per-type + per-action counts.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.redaction is not None
    rq = persisted.redaction
    assert rq.documents_redacted == len(records)
    assert rq.documents_with_pii == 1
    assert rq.pii_total == 2
    assert rq.per_type == {"email": 1, "phone": 1}
    assert rq.per_action == {"mask": 2}
    # The report is consistent with what was returned.
    assert report.redaction is not None
    assert report.redaction.pii_total == persisted.redaction.pii_total

    # 6. S6 AC: no raw PII value appears in ANY persisted run artifact.
    for uri in (jsonl_uri, f"{base}/quality-report.json"):
        blob = s.read_text(uri)
        assert "alice@acme.com" not in blob
        assert "555-123-4567" not in blob


def test_redaction_slice_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    out_a = s.read_text(f"{store}/_latence/runs/det-a/export/records.jsonl")
    out_b = s.read_text(f"{store}/_latence/runs/det-b/export/records.jsonl")
    assert out_a == out_b


def test_redaction_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    """Redaction resume is *exact*: the resumed output is byte-identical to the first run's.

    H-E1 Part 1. The old test asserted only ``all(m.from_checkpoint)`` (+ that each redact
    checkpoint record has non-null offsets) — never that the resumed **output** equals the first
    run's, so a resume that dropped/duplicated/reordered a RedactionRecord (the P3-F1 class) would
    pass. It now captures every output artifact (the exported ``records.{jsonl,parquet}`` + every
    Stage checkpoint, incl. ``redact``), resumes, and asserts byte-identity.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    s = Storage.for_uri(store)

    LocalRunner().run(pipeline, run_id="resume")
    first = output_artifacts(s, store, "resume")
    assert "checkpoints/redact.jsonl" in first  # the redaction Stage's record-set is captured

    report2 = LocalRunner().run(pipeline, run_id="resume")
    second = output_artifacts(s, store, "resume")

    # Every Stage (incl. redaction) resumes from its checkpoint.
    assert all(m.from_checkpoint for m in report2.stages)
    # The property the old test never checked: the actual output is identical across resume.
    assert second == first, "resumed redaction output diverged from the first run's"

    # The redact checkpoint still carries valid, offset-preserving RedactionRecords.
    lines = [x for x in first["checkpoints/redact.jsonl"].decode("utf-8").splitlines() if x.strip()]
    assert lines
    for line in lines:
        r = RedactionRecord.model_validate_json(line)
        assert r.provenance.char_start is not None
        assert r.provenance.char_end is not None


def test_redaction_resume_equality_bites_on_dropped_record(tmp_path: Path) -> None:
    """Prove the redaction resume equality BITES: a dropped RedactionRecord is caught.

    H-E1 Part 1 reviewer check. Drop a record from the ``redact`` checkpoint and delete the
    ``export`` checkpoint so the Export re-runs from the corrupted redaction output. The exported
    corpus must now DIFFER from the first run's — so the byte-equality assertion the resume test
    relies on genuinely detects a dropped record, not just a second ``from_checkpoint`` flag.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/resume"

    LocalRunner().run(pipeline, run_id="resume")
    first = output_artifacts(s, store, "resume")

    redact_ckpt = f"{base}/checkpoints/redact.jsonl"
    lines = [x for x in s.read_text(redact_ckpt).splitlines() if x.strip()]
    assert len(lines) >= 2, "fixture must produce >1 redaction record for the drop to bite"
    s.write_bytes(redact_ckpt, ("\n".join(lines[:-1]) + "\n").encode("utf-8"))
    s._fs.rm(f"{base}/checkpoints/export.jsonl")  # noqa: SLF001

    LocalRunner().run(pipeline, run_id="resume")
    corrupted = output_artifacts(s, store, "resume")

    assert corrupted != first
    assert corrupted["export/records.jsonl"].count(b"\n") < first["export/records.jsonl"].count(
        b"\n"
    )
