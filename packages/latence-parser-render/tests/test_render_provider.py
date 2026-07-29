"""Unit tests for ``parser.render`` — dispatch, OCR delegation, security guards, graceful failure.

The heavy renderer deps are faked offline (see ``conftest.py``): a tiny in-memory renderer + a stub
OCR, faithful to the renderer→PNG-bytes→OCR round-trip. These tests drive the REAL dispatch, the
REAL security guards, the REAL OCR-delegation seam, and the REAL PageMap assembly. The reusable
builders come in as fixtures (``make_pdf``/``make_png``/``make_input``), the repo test convention.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from latence_core.contracts import Disposition, ParserInput
from latence_core.errors import ConfigError, ProviderError
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import REPO_LICENSE_CODE
from latence_parser_render.provider import UniversalRenderParser, size_after_parse


def _parse_one(provider: UniversalRenderParser, raw: ParserInput) -> Any:
    (record,) = list(provider.parse([raw]))
    return record


# --- profile + registration -------------------------------------------------


def test_profile_is_either_permissive_no_weights() -> None:
    """compute='either', model_id=None (OCR weights belong to the OCR Provider), permissive."""
    profile = UniversalRenderParser.profile()
    assert profile.compute == "either"
    assert profile.model_id is None
    assert profile.license == REPO_LICENSE_CODE
    assert profile.license_verified is True
    assert profile.batch is True
    # deterministic is honestly False (page OCR is a VLM whose float decode is not byte-identical).
    assert profile.deterministic is False


def test_provider_name_is_the_entry_point_tag() -> None:
    assert UniversalRenderParser({}).provider_name == "parser.render"


# --- happy paths: PDF / image dispatch → OCR → PageMap ----------------------


def test_pdf_multipage_ocr_and_page_map(
    render_config: dict[str, Any],
    patch_renderer: None,
    make_pdf: Callable[[list[str]], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """A multi-page PDF rasterises page-by-page, each OCR'd, into a 2-page map with exact offsets."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(
        make_pdf(["# Quarterly report\n\nRevenue up.", "## Page two\n\nOutlook."]),
        file_name="report.pdf",
        file_type="pdf",
    )
    record = _parse_one(provider, raw)

    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    assert "Quarterly report" in record.content
    assert "Outlook" in record.content
    # Offsets resolve to the right source page (exact by construction, ADR-0031).
    idx = PageOffsetIndex(record.page_map)
    for span in record.page_map.pages:
        assert idx.page_for_offset(span.char_start) == span.page_number
    # Provenance carried; sensitivity inherited from the Source's Classification.
    assert record.provenance.file_name == "report.pdf"
    assert record.classification.sensitivity == "internal"


def test_image_single_page_ocr(
    render_config: dict[str, Any],
    patch_renderer: None,
    make_png: Callable[[str], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """A bare image is a one-page document: decoded → OCR → single-page markdown record."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(
        make_png("# Poster heading\n\nA scanned image."),
        file_name="poster.png",
        file_type="png",
    )
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 1
    assert "Poster heading" in record.content
    assert size_after_parse(record) == len(record.content.encode("utf-8"))


def test_dispatch_is_case_and_dot_insensitive(
    render_config: dict[str, Any],
    patch_renderer: None,
    make_pdf: Callable[[list[str]], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """A ``.PDF`` (uppercase, leading dot in the type) dispatches to the PDF path."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(make_pdf(["hello"]), file_name="X.PDF", file_type=".PDF")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSED
    assert "hello" in record.content


# --- graceful failure: unknown type, spoof, corrupt --------------------------


def test_unknown_type_is_graceful_parse_error(
    render_config: dict[str, Any],
    patch_renderer: None,
    make_input: Callable[..., ParserInput],
) -> None:
    """An unsupported file type degrades to a PARSE_ERROR record, never a crash."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(b"\x00\x01\x02", file_name="weird.bin", file_type="bin")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert record.content == ""
    assert record.page_map is None


def test_pdf_type_spoof_is_rejected(
    render_config: dict[str, Any],
    patch_renderer: None,
    make_input: Callable[..., ParserInput],
) -> None:
    """A .pdf whose bytes are not a PDF (no %PDF- at offset 0) is a graceful PARSE_ERROR (#62)."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(b"<html>not a pdf</html>", file_name="fake.pdf", file_type="pdf")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSE_ERROR
    assert "spoof" in (record.error or "").lower()


def test_image_type_spoof_is_rejected(
    render_config: dict[str, Any],
    patch_renderer: None,
    make_input: Callable[..., ParserInput],
) -> None:
    """A declared image whose bytes match no image magic is a graceful PARSE_ERROR (#62)."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(b"not an image at all", file_name="fake.png", file_type="png")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSE_ERROR
    assert "spoof" in (record.error or "").lower()


def test_office_zip_spoof_is_rejected(
    render_config: dict[str, Any],
    make_input: Callable[..., ParserInput],
) -> None:
    """A declared .docx whose bytes are not a ZIP container is a graceful PARSE_ERROR (#62)."""
    provider = UniversalRenderParser(render_config)
    raw = make_input(b"plain text not a zip", file_name="fake.docx", file_type="docx")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSE_ERROR
    assert "zip" in (record.error or "").lower()


def test_no_ocr_configured_but_page_needs_pixels_is_graceful(
    patch_renderer: None,
    make_pdf: Callable[[list[str]], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """A PDF/image with NO ocr config → graceful PARSE_ERROR (a page needs an OCR Provider)."""
    provider = UniversalRenderParser({})  # no ocr:
    raw = make_input(make_pdf(["x"]), file_name="a.pdf", file_type="pdf")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSE_ERROR
    assert "ocr" in (record.error or "").lower()


def test_unknown_ocr_provider_is_graceful(
    patch_renderer: None,
    make_png: Callable[[str], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """An unknown OCR Provider name → graceful PARSE_ERROR (never a crash) for a pixels page."""
    provider = UniversalRenderParser({"ocr": {"provider": "parser.does_not_exist"}})
    raw = make_input(make_png("hi"), file_name="a.png", file_type="png")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSE_ERROR
    assert "could not be loaded" in (record.error or "")


# --- OCR page-error degrade --------------------------------------------------


def test_ocr_page_error_contributes_empty_text(
    render_config: dict[str, Any],
    patch_renderer: None,
    monkeypatch: pytest.MonkeyPatch,
    make_pdf: Callable[[list[str]], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """A page whose OCR record is PARSE_ERROR contributes empty text, not a crash (graceful)."""
    from latence_parser_render import provider as prov

    def bad_page_input(self: prov.UniversalRenderParser, image: Any) -> ParserInput:
        # Hand the OCR Provider bytes it cannot decode → every page OCRs to a PARSE_ERROR.
        return make_input(b"undecodable", file_name="page.png", file_type="png")

    monkeypatch.setattr(prov.UniversalRenderParser, "_page_image_input", bad_page_input)
    provider = UniversalRenderParser(render_config)
    raw = make_input(make_pdf(["one", "two"]), file_name="r.pdf", file_type="pdf")
    record = _parse_one(provider, raw)
    # Two empty pages assembled — still PARSED (the doc parsed; the pages were unreadable).
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    assert record.content.strip() == ""


# --- config validation -------------------------------------------------------


def test_bad_ocr_config_shape_fails_loud_at_construction() -> None:
    """A malformed 'ocr' config (not a mapping / missing provider) is a ConfigError (fail loud)."""
    with pytest.raises(ConfigError):
        UniversalRenderParser({"ocr": "parser.lighton"})
    with pytest.raises(ConfigError):
        UniversalRenderParser({"ocr": {"provider": ""}})
    with pytest.raises(ConfigError):
        UniversalRenderParser({"ocr": {"provider": "x", "config": "not-a-dict"}})


def test_bad_dpi_fails_loud() -> None:
    with pytest.raises(ProviderError):
        UniversalRenderParser({"dpi": 0})


def test_bad_soffice_timeout_fails_loud() -> None:
    with pytest.raises(ProviderError):
        UniversalRenderParser({"soffice_timeout_s": 0})


# --- pixel-bomb clamp (the security math, no real raster needed) -------------


def test_clamp_scale_bounds_a_giant_page() -> None:
    """A 10000x10000-pt page at scale 3 is clamped so its bitmap stays under the megapixel budget."""
    from latence_parser_render.provider import _MAX_RENDER_PIXELS

    provider = UniversalRenderParser({})
    clamped = provider._clamp_scale(10_000.0, 10_000.0, 3.0)
    pixels = 10_000.0 * clamped * 10_000.0 * clamped
    assert pixels <= _MAX_RENDER_PIXELS + 1  # lands at the budget
    assert clamped < 3.0  # genuinely reduced


def test_clamp_scale_leaves_small_pages_untouched() -> None:
    """A normal A4-ish page well under the budget keeps its requested scale."""
    provider = UniversalRenderParser({})
    assert provider._clamp_scale(595.0, 842.0, 200.0 / 72.0) == pytest.approx(200.0 / 72.0)


def test_clamp_scale_degenerate_zero_area_page() -> None:
    provider = UniversalRenderParser({})
    assert provider._clamp_scale(0.0, 0.0, 2.0) == 2.0


# --- pixel-bomb PROTECTIVE BEHAVIOR (real _pdf_to_images / _image_from_bytes bodies) ---
#
# The tests above assert the clamp *math* in isolation; the ones below run the REAL native bodies
# against a fake Pillow that HONORS ``MAX_IMAGE_PIXELS`` and a fake pypdfium2 that reports an
# oversized point size — so the guard's protective behavior fails RED if the cap assignment, the
# ``finally`` restore, or the clamp REDUCE branch is removed (the sibling parser.lighton bar).


def test_decompression_bomb_image_degrades_to_parse_error(
    render_config: dict[str, Any],
    native_render_modules: Any,
    make_bomb_png_bytes: Callable[..., bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """An over-``MAX_IMAGE_PIXELS`` image makes real Pillow raise on ``.load()`` → graceful PARSE_ERROR."""
    from latence_parser_render.provider import _MAX_RENDER_PIXELS

    provider = UniversalRenderParser(render_config)
    raw = make_input(
        make_bomb_png_bytes(pixels=_MAX_RENDER_PIXELS + 1),
        file_name="bomb.png",
        file_type="png",
    )
    record = _parse_one(provider, raw)
    # A decompression bomb never crashes and never allocates: it degrades to a typed PARSE_ERROR
    # carrying no page map and no leaked bytes/text (the native fault → guard → PARSE_ERROR path).
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.page_map is None
    assert record.content == ""


def test_in_budget_image_loads_and_restores_max_image_pixels(
    render_config: dict[str, Any],
    native_render_modules: Any,
    make_png: Callable[[str], bytes],
    make_input: Callable[..., ParserInput],
) -> None:
    """An in-budget image parses AND ``Image.MAX_IMAGE_PIXELS`` is restored to its prior value after."""
    prior_cap = native_render_modules.MAX_IMAGE_PIXELS
    provider = UniversalRenderParser(render_config)
    raw = make_input(make_png("# In budget\n\nSmall image."), file_name="ok.png", file_type="png")
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSED
    assert "In budget" in record.content
    # The Provider tightened MAX_IMAGE_PIXELS to the render budget during decode, then restored it in
    # a ``finally`` — a broken restore would leave the tighter cap and fail this assertion.
    assert prior_cap == native_render_modules.MAX_IMAGE_PIXELS


def test_oversized_pdf_page_clamped_not_crashed(
    render_config: dict[str, Any],
    native_render_modules: Any,
    make_oversized_pdf_bytes: Callable[..., bytes],
    make_input: Callable[..., ParserInput],
    last_render_document_cls: Any,
) -> None:
    """A 10000x10000-pt PDF page renders through the REAL clamp REDUCE branch → in-budget, no crash."""
    from latence_parser_render.provider import _MAX_RENDER_PIXELS

    provider = UniversalRenderParser(render_config)
    raw = make_input(
        make_oversized_pdf_bytes(["# Giant page\n\nStill renders."], width_pt=10_000.0, height_pt=10_000.0),
        file_name="giant.pdf",
        file_type="pdf",
    )
    record = _parse_one(provider, raw)
    assert record.disposition is Disposition.PARSED
    assert "Giant page" in record.content
    # The real _clamp_scale shrank the render scale so the emitted bitmap stayed under the budget.
    page = last_render_document_cls.last.pages[0]
    assert page.last_scale is not None
    clamped_pixels = 10_000.0 * page.last_scale * 10_000.0 * page.last_scale
    assert clamped_pixels <= _MAX_RENDER_PIXELS + 1
    assert page.last_scale < provider._dpi / 72.0  # genuinely reduced from the requested scale
