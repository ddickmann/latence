"""End-to-end S6 Redaction through the local Runner — CHUNK-LEVEL (W16, ADR-0042).

Source -> Parse -> Chunk -> Redaction -> Export, using the in-core deterministic
``redaction.hybrid_rule``. Since W16 Redaction is a **chunk→chunk transform**: it consumes
ChunkRecords and yields the SAME chunks with ``masked_content`` + ``pii_spans`` populated (the clean
``content`` stays UNMASKED). Asserts the exported records are ChunkRecords whose ``masked_content``
has the PII removed and whose chunk-local PII spans resolve to the correct source page (the round
trip); that the exported corpus text (Parquet ``content`` column + JSONL) is the MASKED variant (no
raw PII); that the redact-stage chunk's ``content`` is byte-UNCHANGED (so extraction still sees full
entities); that the Quality Report carries per-type + per-action counts; that no raw PII value
appears anywhere in the persisted artifacts (S6 AC); and that the run is deterministic and resumes
cleanly.

The HEADLINE (:func:`test_chunking_catches_pii_past_the_model_window`): content that as ONE document
exceeds the PII model window has ALL its PII masked once chunked — the fix for the document-level
768-token truncation leak, with a control proving the old doc-level path would MISS it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from _resume_equality import output_artifacts
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    DocumentRecord,
    PageMap,
    Provenance,
    RedactionAction,
    RedactionRecord,
)
from latence_core.page_index import PageIndexResolver
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.redaction_policy import (
    RedactionPolicy,
    finalize_redacted_chunk,
    plan_chunk_redaction,
)
from latence_core.runner import LocalRunner
from latence_core.stages.chunk import MarkdownChunker
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
    depends_on: [chunk]
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
    (docs / "clean.txt").write_text(
        "The quarterly report is attached for review.\n", encoding="utf-8"
    )
    # An empty document.
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_redaction_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s6e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s6e2e"

    # 1. The export holds typed ChunkRecord rows (W16: redaction is chunk→chunk).
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [ChunkRecord.model_validate(r) for r in rows]
    assert records
    memo = next(r for r in records if r.masked_content is not None and "[EMAIL]" in r.masked_content)

    # 2. PII is removed from the masked variant; the raw values are gone.
    assert memo.masked_content is not None
    assert "alice@acme.com" not in memo.masked_content
    assert "555-123-4567" not in memo.masked_content
    assert "[EMAIL]" in memo.masked_content
    assert "[PHONE]" in memo.masked_content
    assert {sp.pii_type for sp in memo.pii_spans} == {"email", "phone"}
    assert memo.classification.sensitivity == "internal"

    # 3. The exported corpus TEXT is the masked variant (the Export materializes masked_content as
    #    the `content` column + JSONL), so a RAG consumer never reads raw PII.
    assert memo.content is not None and "alice@acme.com" not in memo.content  # export-substituted

    # 4. The round-trip: each chunk-local PII span resolves to the correct source page.
    for span in memo.pii_spans:
        assert span.page_start is not None

    # 5. The redact-STAGE checkpoint keeps `content` byte-UNCHANGED (extraction sees full entities):
    #    the raw email is still in the chunk's content, while masked_content is masked.
    redact_ckpt = f"{base}/checkpoints/redact.jsonl"
    stage_chunks = [
        ChunkRecord.model_validate_json(line)
        for line in s.read_text(redact_ckpt).splitlines()
        if line.strip()
    ]
    memo_stage = next(c for c in stage_chunks if "alice@acme.com" in c.content)
    assert memo_stage.masked_content is not None and "[EMAIL]" in memo_stage.masked_content
    assert "alice@acme.com" not in memo_stage.masked_content

    # 6. A clean document yields chunks with no spans and untouched text.
    clean = next(
        r for r in records if not r.pii_spans and r.masked_content and "quarterly" in r.masked_content
    )
    assert clean.masked_content == clean.content

    # 7. Quality Report: RedactionQuality present with per-type + per-action counts.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.redaction is not None
    rq = persisted.redaction
    assert rq.documents_with_pii == 1
    assert rq.pii_total == 2
    assert rq.per_type == {"email": 1, "phone": 1}
    assert rq.per_action == {"mask": 2}
    assert report.redaction is not None
    assert report.redaction.pii_total == persisted.redaction.pii_total

    # 8. S6 AC: no raw PII value appears in ANY persisted export/report artifact.
    for uri in (jsonl_uri, f"{base}/export/records.parquet", f"{base}/quality-report.json"):
        blob = (
            s.read_text(uri)
            if uri.endswith((".jsonl", ".json"))
            else " ".join(_parquet_cells(s, uri))
        )
        assert "alice@acme.com" not in blob
        assert "555-123-4567" not in blob


def _parquet_cells(s: Storage, uri: str) -> list[str]:
    import io

    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(s.read_bytes(uri)))  # type: ignore[no-untyped-call]
    return [str(v) for col in table.columns for v in col.to_pylist()]


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


def test_additive_field_no_redact_stage_is_byte_identical_corpus(tmp_path: Path) -> None:
    """Additive-field guarantee: with NO Redaction Stage, masked_content is None and the exported
    corpus text falls back to ``content`` — byte-identical to a pre-W16 chunk export.
    """
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    # A pipeline WITHOUT a redact stage: export straight off the chunk stream.
    no_redact = Pipeline.from_yaml_text(
        f"""
name: no-redact
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', sensitivity: internal, extensions: [txt, md]}}
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
    LocalRunner().run(no_redact, run_id="nore")
    s = Storage.for_uri(store)
    rows = [
        json.loads(line)
        for line in s.read_text(f"{store}/_latence/runs/nore/export/records.jsonl").splitlines()
        if line.strip()
    ]
    # No Redaction Stage ⇒ masked_content None ⇒ the corpus text is the un-redacted content.
    memo = next(r for r in rows if "alice@acme.com" in (r.get("content") or ""))
    assert memo["masked_content"] is None
    assert memo["pii_spans"] == []


def test_redaction_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    """Redaction resume is *exact*: the resumed output is byte-identical to the first run's."""
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    s = Storage.for_uri(store)

    LocalRunner().run(pipeline, run_id="resume")
    first = output_artifacts(s, store, "resume")
    assert "checkpoints/redact.jsonl" in first  # the redaction Stage's record-set is captured

    report2 = LocalRunner().run(pipeline, run_id="resume")
    second = output_artifacts(s, store, "resume")

    assert all(m.from_checkpoint for m in report2.stages)
    assert second == first, "resumed redaction output diverged from the first run's"

    # The redact checkpoint carries valid, masked ChunkRecords.
    lines = [x for x in first["checkpoints/redact.jsonl"].decode("utf-8").splitlines() if x.strip()]
    assert lines
    for line in lines:
        c = ChunkRecord.model_validate_json(line)
        assert c.masked_content is not None  # every redacted chunk carries a PII-handled variant
        assert c.provenance.char_start is not None
        assert c.provenance.char_end is not None


def test_redaction_resume_equality_bites_on_dropped_record(tmp_path: Path) -> None:
    """Prove the redaction resume equality BITES: a dropped chunk is caught."""
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


# ---------------------------------------------------------------------------
# THE HEADLINE: chunking catches PII past the model window (the ADR-0042 fix)
# ---------------------------------------------------------------------------


class _WindowedRedactor:
    """A PIIDetector test double that simulates a fixed PII model window (like gliner's max_len).

    It only "sees" a ``SECRET-<n>`` token whose span ends within the first ``window`` characters of
    whatever text it scans — mirroring the gliner-family's SILENT truncation of any input past
    ``max_len`` (the exact bug ADR-0042 fixes). ``redact`` is the chunk→chunk seam (each chunk ≤ the
    window, so nothing is truncated); ``redact_documents`` is the DEPRECATED doc-level control (the
    whole document goes into the one window, so the tail is truncated + leaks).
    """

    _TOKEN = re.compile(r"SECRET-\d+")

    def __init__(self, window: int) -> None:
        self._window = window
        self._policy = RedactionPolicy(None, RedactionAction.MASK)

    def _accepted(self, text: str) -> list[tuple[int, int, str, str, str, float]]:
        # Truncation: a token whose end is past the window is NEVER seen (the leak).
        return [
            (m.start(), m.end(), "secret", m.group(), "windowed", 1.0)
            for m in self._TOKEN.finditer(text)
            if m.end() <= self._window
        ]

    def redact(self, chunks: Any) -> Any:
        resolver = PageIndexResolver()  # ONE per call — the map rides chunk 0 only (v18)
        for chunk in chunks:
            plan = plan_chunk_redaction(
                chunk,
                self._policy,
                detector_types={"secret"},
                sensitive_set=set(),
                min_chars=1,
                resolver=resolver,
            )
            accepted = self._accepted(chunk.content) if plan.needs_detection else []
            yield finalize_redacted_chunk(chunk, plan, accepted, "salt")

    def redact_documents(self, documents: Any) -> Any:
        from latence_core.redaction_policy import build_span

        for doc in documents:
            spans = [
                build_span(
                    pii_type=pt,
                    detector=det,
                    start=st,
                    end=en,
                    value=val,
                    action=RedactionAction.MASK,
                    salt="salt",
                    page_index=None,
                    confidence=cf,
                )
                for st, en, pt, val, det, cf in self._accepted(doc.content)
            ]
            from latence_core.redaction_policy import apply_masking

            yield RedactionRecord(
                record_id=f"{doc.record_id}#redaction",
                document_record_id=doc.record_id,
                masked_content=apply_masking(doc.content, spans),
                policy="default",
                pii_spans=spans,
                provenance=doc.provenance.model_copy(
                    update={"char_start": 0, "char_end": len(doc.content)}
                ),
                classification=doc.classification,
            )


def _long_pii_document(window: int) -> DocumentRecord:
    """A document whose FIRST PII token is inside the window and whose SECOND is far past it."""
    head = "SECRET-1 is here. " + ("filler word " * ((window // 12) + 40))
    text = head + "and finally SECRET-2 at the very end."
    assert text.index("SECRET-2") > window, "fixture must place SECRET-2 past the window"
    assembly = PageMap.from_page_texts([text])
    return DocumentRecord(
        record_id="long:doc",
        provenance=Provenance(
            source_uri="mem://long.txt",
            file_name="long.txt",
            file_type="txt",
            file_size=len(text),
            document_id="sha256:long",
            char_start=0,
            char_end=len(text),
        ),
        classification=Classification(language="en", sensitivity="internal"),
        content=assembly.text,
        page_map=assembly.page_map,
    )


def test_chunking_catches_pii_past_the_model_window() -> None:
    """HEADLINE (ADR-0042): PII past the model window is MASKED once chunked; the old doc-level path
    MISSES it.

    A single document places ``SECRET-1`` inside a 200-char PII model window and ``SECRET-2`` far
    past it. The document-level path (the deprecated control) feeds the WHOLE document into the one
    window, so ``SECRET-2`` is silently truncated and LEAKS. The chunk-level path chunks the document
    into pieces each ≤ the window, so ``SECRET-2`` is scanned in full and MASKED — nothing leaks.
    """
    window = 200
    detector = _WindowedRedactor(window)
    doc = _long_pii_document(window)

    # CONTROL — the deprecated document-level path: SECRET-2 is past the window → truncated → LEAKS.
    doc_rec = next(iter(detector.redact_documents([doc])))
    assert "[SECRET]" in doc_rec.masked_content  # SECRET-1 (inside the window) was masked
    assert "SECRET-2" in doc_rec.masked_content  # SECRET-2 LEAKED — the document-truncation bug

    # FIX — the chunk-level path: chunk the document small (each chunk ≤ the window), then redact.
    chunks = list(MarkdownChunker({"max_tokens": 24, "overlap_tokens": 0, "min_tokens": 4}).chunk([doc]))
    assert all(len(c.content) <= window for c in chunks), "each chunk must fit the window"
    redacted = list(detector.redact(chunks))

    corpus = " ".join(c.masked_content or "" for c in redacted)
    # BOTH secrets are masked once chunked — the PII past the old 768-doc-truncation point is caught.
    assert "SECRET-1" not in corpus
    assert "SECRET-2" not in corpus
    assert "[SECRET]" in corpus
    # And the clean content is byte-unchanged across the chunks (extraction still sees the tokens).
    joined_content = "".join(c.content for c in chunks)
    assert "SECRET-2" in joined_content
