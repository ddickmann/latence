"""End-to-end S2 Parse through the local Runner: Source -> document Parse -> Export.

Reads a folder holding a multi-page PDF, a text file, and a corrupt PDF; runs the
real Pipeline on the LocalRunner; and asserts the exported records carry page maps
that resolve offsets to the right source page, that the corrupt file degraded to a
PARSE_ERROR (the run did not crash), and that the Quality Report's ParseQuality
section reports the page-drift diagnostics and the parse-error count.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from latence_core.contracts import Disposition, DocumentRecord, PageMap
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: parse-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: confidential
      extensions: [txt, md, pdf]
  - name: parse
    capability: parse
    provider: parser.document
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )


def test_parse_slice_end_to_end(
    tmp_path: Path, make_text_pdf: Callable[[list[str]], bytes]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "report.pdf").write_bytes(
        make_text_pdf(
            ["The quarterly report and the numbers.", "Page two of the report follows here."]
        )
    )
    (docs / "notes.txt").write_text(
        "the framework reads the folder and exports the records\n", encoding="utf-8"
    )
    # A corrupt PDF: valid header, garbage body -> must become a PARSE_ERROR, not a crash.
    (docs / "broken.pdf").write_bytes(b"%PDF-1.4\ntotally broken pdf body\n%%EOF")

    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s2e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s2e2e"

    # 1. JSONL export exists and holds one record per source doc.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert len(rows) == 3
    by_name = {r["provenance"]["file_name"]: r for r in rows}

    # 2. The multi-page PDF has a 2-page map; offsets resolve to the right page.
    pdf_row = by_name["report.pdf"]
    assert pdf_row["disposition"] == Disposition.PARSED.value
    page_map = PageMap.model_validate(pdf_row["page_map"])
    assert len(page_map.pages) == 2
    idx = PageOffsetIndex(page_map)
    for span in page_map.pages:
        assert idx.page_for_offset(span.char_start) == span.page_number
    assert idx.diagnostics.recovered == 0
    # Classification attached at Parse; sensitivity inherited from the Source.
    assert pdf_row["classification"]["sensitivity"] == "confidential"

    # 3. The text doc parsed with language detected at Parse.
    txt_row = by_name["notes.txt"]
    assert txt_row["disposition"] == Disposition.PARSED.value
    assert txt_row["classification"]["language"] == "en"

    # 4. The corrupt PDF degraded gracefully: a PARSE_ERROR record, no crash.
    bad_row = by_name["broken.pdf"]
    assert bad_row["disposition"] == Disposition.PARSE_ERROR.value
    assert bad_row["error"]
    assert bad_row["page_map"] is None

    # 5. Quality Report: ParseQuality present with drift + error counts.
    persisted = QualityReport.model_validate_json(
        s.read_text(f"{base}/quality-report.json")
    )
    assert persisted.parse is not None
    assert persisted.parse.documents_parsed == 2
    assert persisted.parse.parse_errors == 1
    assert persisted.parse.pages_total == 3  # 2 (pdf) + 1 (txt)
    assert persisted.parse.page_maps_clean is True
    assert persisted.parse.page_drift_recovered == 0
    # bytes-after-parse: the assembled-markdown byte size summed across the two
    # parsed docs (the corrupt PDF contributes nothing — empty content).
    expected_bytes = sum(
        len(r["content"].encode("utf-8"))
        for r in rows
        if r["disposition"] == Disposition.PARSED.value
    )
    assert expected_bytes > 0
    assert persisted.parse.bytes_after_parse == expected_bytes
    # The report is internally consistent with the returned object.
    assert report.parse is not None
    assert report.parse.parse_errors == persisted.parse.parse_errors


def test_resume_skips_completed_parse_stage(
    tmp_path: Path, make_text_pdf: Callable[[list[str]], bytes]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(make_text_pdf(["Only one page here."]))
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)

    LocalRunner().run(pipeline, run_id="resume")
    # Second run resumes every Stage from its checkpoint (no re-execution).
    report2 = LocalRunner().run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)
    # The resumed export still round-trips to a valid DocumentRecord with a page map.
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/resume/checkpoints/parse.jsonl"
    line = next(x for x in s.read_text(ckpt).splitlines() if x.strip())
    rec = DocumentRecord.model_validate_json(line)
    assert rec.page_map is not None
    assert rec.disposition is Disposition.PARSED
