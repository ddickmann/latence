"""End-to-end: ``parser.render`` through the local Runner + its conformance verdict + registration.

Baseline-bar proofs for the universal-parser slice:

1. **≥1 end-to-end test through the local Runner** — a real Source → ``parser.render`` → Export
   pipeline on :class:`~latence_core.runner.LocalRunner` over a folder holding a (fake) multi-page
   PDF and an image. The native rasterisers are faked offline and the OCR Provider is resolved to
   the stub OCR through the REAL registry seam (``ProviderRegistry.load`` patched) — so the whole
   Source → Parse (dispatch → render → OCR delegation → PageMap) → Export spine runs, and the
   exported records carry page maps whose offsets resolve to the right source page.

2. **The provider's conformance verdict (P3-F3, C1–C6)** as a registered provider — it is
   ``compute="either"`` so it genuinely RUNS on this CPU host (not device-skipped).

3. **The registry discovers the provider** via its entry point and surfaces its profile.

Deterministic + seeded (PYTHONHASHSEED pinned by the repo-root conftest).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
from latence_core.contracts import Disposition, PageMap
from latence_core.pagemap import PageOffsetIndex
from latence_core.pipeline import Pipeline
from latence_core.providers.profile import REPO_LICENSE_CODE
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

import latence_parser_render.provider as prov


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: render-parse-slice
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
    provider: parser.render
    depends_on: [source]
    config:
      dpi: 150
      ocr:
        provider: parser.lighton
        config: {{}}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )


@pytest.fixture
def _wire_offline(
    monkeypatch: pytest.MonkeyPatch, stub_ocr_cls: type, fake_page_image_cls: type
) -> None:
    """Fake the native rasterisers and resolve the OCR Provider to the stub via the REAL registry.

    The Runner constructs ``parser.render`` from YAML config (no live objects), so the OCR Provider
    must resolve through the entry-point registry. Patching :meth:`ProviderRegistry.load` to hand the
    stub OCR for ``parser.lighton`` exercises the Provider's real registry-resolution seam offline.
    """

    def fake_pdf_to_images(self: prov.UniversalRenderParser, data: bytes) -> list[object]:
        body = data[len(b"%PDF-") :].decode("utf-8")
        return [fake_page_image_cls(seg) for seg in body.split("|||")]

    def fake_image_from_bytes(self: prov.UniversalRenderParser, data: bytes) -> object:
        return fake_page_image_cls(data[len(b"\x89PNG\r\n\x1a\n") :].decode("utf-8"))

    monkeypatch.setattr(prov.UniversalRenderParser, "_pdf_to_images", fake_pdf_to_images)
    monkeypatch.setattr(prov.UniversalRenderParser, "_image_from_bytes", fake_image_from_bytes)

    real_load = ProviderRegistry.load

    def patched_load(self: ProviderRegistry, name: str) -> type:
        if name in ("parser.lighton", "parser.glm"):
            return stub_ocr_cls
        return real_load(self, name)

    monkeypatch.setattr(ProviderRegistry, "load", patched_load)


def test_parse_slice_end_to_end(
    _wire_offline: None,
    tmp_path: Path,
    make_pdf: Callable[[list[str]], bytes],
    make_png: Callable[[str], bytes],
) -> None:
    """A LocalRunner run with ``parser.render`` renders + OCRs a PDF + image, landing page maps."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "report.pdf").write_bytes(
        make_pdf(["# Quarterly report\n\nRevenue up.", "## Page two\n\nOutlook follows."])
    )
    (docs / "poster.png").write_bytes(make_png("# Poster heading\n\nA scanned image.\n"))

    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="render-e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/render-e2e"

    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    import json

    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert len(rows) == 2
    by_name = {r["provenance"]["file_name"]: r for r in rows}

    # The multi-page PDF has a 2-page map from OCR; offsets resolve to the right page.
    pdf_row = by_name["report.pdf"]
    assert pdf_row["disposition"] == Disposition.PARSED.value
    assert "Quarterly report" in pdf_row["content"]
    page_map = PageMap.model_validate(pdf_row["page_map"])
    assert len(page_map.pages) == 2
    idx = PageOffsetIndex(page_map)
    for span in page_map.pages:
        assert idx.page_for_offset(span.char_start) == span.page_number
    # Classification attached at Parse; sensitivity inherited from the Source.
    assert pdf_row["classification"]["sensitivity"] == "confidential"

    # The image OCR'd to a single-page markdown record.
    img_row = by_name["poster.png"]
    assert img_row["disposition"] == Disposition.PARSED.value
    assert "Poster heading" in img_row["content"]

    # Quality Report: ParseQuality present with page + drift counts.
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.parse is not None
    assert persisted.parse.documents_parsed == 2
    assert persisted.parse.parse_errors == 0
    assert persisted.parse.pages_total == 3  # 2 (pdf) + 1 (png)
    assert report.parse is not None
    assert report.parse.documents_parsed == persisted.parse.documents_parsed


def test_provider_conformance_runs_on_cpu_host() -> None:
    """parser.render is compute='either' → it RUNS C1–C6 on this CPU host (not device-skipped)."""
    verdict = run_conformance(ProviderRegistry(), "parser.render")
    assert verdict.provider == "parser.render"
    assert verdict.capability == "parse"
    assert verdict.device_skipped is False
    assert verdict.passed, [f"{c.check}:{c.status}:{c.detail}" for c in verdict.checks]


def test_provider_discovered_via_entry_point() -> None:
    """The provider is registered under 'latence.providers' and loads to the adapter class."""
    registry = ProviderRegistry()
    assert "parser.render" in registry.names()
    cls = registry.load("parser.render")
    assert cls is prov.UniversalRenderParser
    profile = registry.profile("parser.render")
    assert profile is not None
    assert profile.license == REPO_LICENSE_CODE
    assert profile.compute == "either"
    assert profile.model_id is None
