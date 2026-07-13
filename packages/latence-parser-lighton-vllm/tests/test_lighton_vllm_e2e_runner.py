"""End-to-end: the served-vLLM LightOn-OCR Parser through the local Runner + conformance + discovery.

The Baseline-bar proofs for the production-OCR slice (W10):

1. **>=1 end-to-end test through the local Runner** — a real Source -> ``parser.lighton_vllm`` ->
   Export pipeline on :class:`~latence_core.runner.LocalRunner` over a folder holding a (fixture)
   scanned PDF and a scanned image. The offline vLLM mock + rasteriser fakes (``fake_vllm``) stand in
   for the served endpoint + the heavy rasteriser deps, so the REAL rasterise -> POST -> PageMap path
   runs offline. Asserts the exported records carry page maps whose offsets resolve to the right page,
   the markdown was OCR'd (the faithful mock transcribes each page image back), and the Quality
   Report's ParseQuality section reports the drift + page counts.

2. **The provider's conformance verdict (P3-F3, C1-C6)** as a registered provider. It is infra-gated
   (compute="either", needs a vLLM server), so it is stub-scoped — its verdict passes against the
   async endpoint stub + the offline rasteriser (not device-skipped).

3. **The registry discovers the provider** via its entry point and surfaces its profile (roster +1).

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
from latence_core.contracts import Disposition, PageMap
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: lighton-vllm-parse-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: confidential
      extensions: [pdf, png]
  - name: parse
    capability: parse
    provider: parser.lighton_vllm
    depends_on: [source]
    config:
      base_url: 'http://stub-vllm/v1'
      max_concurrency: 4
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )


def test_parse_slice_end_to_end(
    fake_vllm: None,
    tmp_path: Path,
    make_pdf: Callable[[list[str]], bytes],
    make_image: Callable[[str], bytes],
) -> None:
    """A LocalRunner run with ``parser.lighton_vllm`` OCRs a scanned PDF + image, landing page maps."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "report.pdf").write_bytes(
        make_pdf(["# The quarterly report\n\nRevenue up.", "## Page two\n\nOutlook follows."])
    )
    (docs / "poster.png").write_bytes(make_image("# Poster heading\n\nA scanned image.\n"))

    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="lighton-vllm-e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/lighton-vllm-e2e"

    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert len(rows) == 2
    by_name = {r["provenance"]["file_name"]: r for r in rows}

    # The multi-page PDF has a 2-page map from OCR; offsets resolve to the right page, no drift.
    pdf_row = by_name["report.pdf"]
    assert pdf_row["disposition"] == Disposition.PARSED.value
    assert "quarterly report" in pdf_row["content"]
    page_map = PageMap.model_validate(pdf_row["page_map"])
    assert len(page_map.pages) == 2
    idx = PageOffsetIndex(page_map)
    for span in page_map.pages:
        assert idx.page_for_offset(span.char_start) == span.page_number
    assert idx.diagnostics.recovered == 0
    # Classification attached at Parse; sensitivity inherited from the Source.
    assert pdf_row["classification"]["sensitivity"] == "confidential"

    # The scanned image OCR'd to a single-page markdown record.
    img_row = by_name["poster.png"]
    assert img_row["disposition"] == Disposition.PARSED.value
    assert "Poster heading" in img_row["content"]

    # Quality Report: ParseQuality present with page + drift counts.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.parse is not None
    assert persisted.parse.documents_parsed == 2
    assert persisted.parse.parse_errors == 0
    assert persisted.parse.pages_total == 3  # 2 (pdf) + 1 (png)
    assert persisted.parse.page_maps_clean is True
    assert persisted.parse.page_drift_recovered == 0
    assert report.parse is not None
    assert report.parse.documents_parsed == persisted.parse.documents_parsed


def test_provider_conformance_stub_scoped_passes() -> None:
    """The provider's conformance verdict passes (stub-scoped, not device-skipped).

    The conformance case is self-contained — it installs the async ``AsyncOpenAI`` stub + the offline
    rasteriser fakes itself (`install_lighton_vllm_offline` + `install_openai`), so no fixture needed.
    """
    verdict = run_conformance(ProviderRegistry(), "parser.lighton_vllm")
    assert verdict.provider == "parser.lighton_vllm"
    assert verdict.capability == "parse"
    assert verdict.device_skipped is False  # compute='either' — runs, not GPU-skipped
    assert verdict.stub_scoped is True  # infra-gated: the served OCR is stubbed
    assert verdict.passed, [f"{c.check}:{c.status}:{c.detail}" for c in verdict.checks]
    statuses = {c.check: c.status.value for c in verdict.checks}
    assert statuses["C3_license"] == "pass"  # permissive Apache-2.0, evidence-bearing
    assert statuses["C1_valid_records"] == "pass"  # rasterise -> stub OCR -> PARSED with a page map


def test_provider_discovered_via_entry_point() -> None:
    """The provider is registered under 'latence.providers' and loads to the adapter class (roster+1)."""
    registry = ProviderRegistry()
    assert "parser.lighton_vllm" in registry.names()
    cls = registry.load("parser.lighton_vllm")
    from latence_parser_lighton_vllm.provider import LightOnVllmParser

    assert cls is LightOnVllmParser
    profile = registry.profile("parser.lighton_vllm")
    assert profile is not None
    assert profile.license == "Apache-2.0"
    assert profile.model_id == "lightonai/LightOnOCR-2-1B"
    assert profile.compute == "either"
