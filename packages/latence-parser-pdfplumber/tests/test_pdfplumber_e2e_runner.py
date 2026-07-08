"""End-to-end: the pdfplumber Parser through the local Runner + its conformance verdict (H-A1).

Two proofs the Baseline bar asks for:

1. **≥1 end-to-end test through the local Runner** — a real Source → ``parser.pdfplumber`` → Export
   pipeline on :class:`~latence_core.runner.LocalRunner` over a folder holding a multi-page PDF, a
   text file, and a corrupt PDF; asserts the exported records carry page maps that resolve offsets
   to the right source page, the corrupt file degraded to a PARSE_ERROR (the run did not crash),
   and the Quality Report's ParseQuality section reports the drift + parse-error counts. This is
   the first real Wave-1 adapter validated *through the Runner*, not just in isolation.

2. **The provider passes the P3-F3 conformance suite (C1–C6) as a registered provider** — the
   enterprise-readiness gate the first real AdapterBase adapter must clear end to end, folded into
   the same Quality Report every other metric lands in.

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest; ``LATENCE_CUDA=0``
pinned below so the device seam is reproducible).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
from latence_core.contracts import Disposition, PageMap
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: pdfplumber-parse-slice
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
    provider: parser.pdfplumber
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
    # A corrupt PDF: valid header, garbage body → must become a PARSE_ERROR, not a crash.
    (docs / "broken.pdf").write_bytes(b"%PDF-1.4\ntotally broken pdf body\n%%EOF")

    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="pdfplumber-e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/pdfplumber-e2e"

    # 1. JSONL export exists and holds one record per source doc.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert len(rows) == 3
    by_name = {r["provenance"]["file_name"]: r for r in rows}

    # 2. The multi-page PDF has a 2-page map; offsets resolve to the right page with no drift.
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
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.parse is not None
    assert persisted.parse.documents_parsed == 2
    assert persisted.parse.parse_errors == 1
    assert persisted.parse.pages_total == 3  # 2 (pdf) + 1 (txt)
    assert persisted.parse.page_maps_clean is True
    assert persisted.parse.page_drift_recovered == 0
    # bytes-after-parse: the assembled-markdown byte size summed across the two parsed docs.
    expected_bytes = sum(
        len(r["content"].encode("utf-8"))
        for r in rows
        if r["disposition"] == Disposition.PARSED.value
    )
    assert expected_bytes > 0
    assert persisted.parse.bytes_after_parse == expected_bytes
    assert report.parse is not None
    assert report.parse.parse_errors == persisted.parse.parse_errors


def test_provider_passes_conformance_c1_to_c6() -> None:
    """``parser.pdfplumber`` passes the P3-F3 conformance gate (C1–C6) as a registered provider.

    The first real AdapterBase adapter clearing the enterprise-readiness gate end to end — the
    proof the gate + seam compose. Run explicitly against the registry (the provider is discovered
    via its entry point) with the CPU-only device host pinned by the fixture above.
    """
    verdict = run_conformance(ProviderRegistry(), "parser.pdfplumber")
    assert verdict.provider == "parser.pdfplumber"
    assert verdict.capability == "parse"
    assert verdict.device_skipped is False
    assert verdict.passed, [f"{c.check}:{c.status}:{c.detail}" for c in verdict.checks]
    # Every check is PASS or a defined SKIP (never a FAIL).
    statuses = {c.check: c.status.value for c in verdict.checks}
    assert statuses["C1_valid_records"] == "pass"
    assert statuses["C2_graceful_failure"] == "pass"
    assert statuses["C3_license"] == "pass"  # permissive MIT, evidence-bearing
    assert statuses["C4_determinism"] == "pass"  # deterministic, byte-identical
    assert statuses["C5_device"] == "pass"  # CPU honored
    assert statuses["C6_no_pii_leak"] == "pass"


def test_provider_discovered_via_entry_point() -> None:
    """The provider is registered under 'latence.providers' and loads to the adapter class."""
    registry = ProviderRegistry()
    assert "parser.pdfplumber" in registry.names()
    cls = registry.load("parser.pdfplumber")
    from latence_parser_pdfplumber.provider import PdfPlumberParser

    assert cls is PdfPlumberParser
    # Its profile is surfaced by the registry (P3-F1).
    profile = registry.profile("parser.pdfplumber")
    assert profile is not None
    assert profile.license == "MIT"
