"""Unit tests for the LightOn-OCR Parser adapter (W1-lighton) — happy path + key failures.

These prove the slice EXERCISES ``AdapterBase`` (H-A1), the GPU device gate, and the OCR seam:
the shape-matching offline fake (the anti-false-green rule), the markdown-per-page → PageMap
(exact offsets round-trip), the Provenance carry, the batching path over documents, the declared
GPU ``ProviderProfile`` (evidence-bearing Apache-2.0), the graceful PARSE_ERROR path with NO
image/OCR-text/PII leak in the error, and — the Hardening-bar addition — the **pixel-bomb DoS cap**
(render-scale clamp + decompression-bomb image → graceful PARSE_ERROR, never a crash / huge alloc).

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest). The ``fake_lighton``
fixture pins ``LATENCE_CUDA=1`` so the ``compute="gpu"`` seam routes to cuda and the real parse
path runs against the fake (offline, no real GPU).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from latence_core.capability import Parser
from latence_core.contracts import (
    Classification,
    Disposition,
    ParserInput,
    Provenance,
)
from latence_core.errors import LatenceError, ProviderError
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import UNVERIFIED_LICENSE
from latence_parser_lighton.provider import (
    _MAX_RENDER_PIXELS,
    LightOnOcrParser,
    size_after_parse,
)


def _provenance(file_name: str, file_type: str, size: int = 100) -> Provenance:
    return Provenance(
        source_uri=f"mem://docs/{file_name}",
        file_name=file_name,
        file_type=file_type,
        file_size=size,
        document_id=f"sha256:{file_name}",
    )


def _pdf_input(content: bytes, file_name: str = "scan.pdf", sensitivity: str = "internal") -> ParserInput:
    return ParserInput(
        provenance=_provenance(file_name, "pdf", size=len(content)),
        classification=Classification(language="und", sensitivity=sensitivity),
        content=content,
    )


def _image_input(content: bytes, file_name: str = "page.png") -> ParserInput:
    return ParserInput(
        provenance=_provenance(file_name, "png", size=len(content)),
        classification=Classification(language="und", sensitivity="internal"),
        content=content,
    )


# --- the adapter genuinely uses AdapterBase (structural + seam) --------------


def test_is_adapterbase_subclass_and_parser() -> None:
    """The Provider is an AdapterBase subclass AND satisfies the Parser Capability structurally."""
    assert issubclass(LightOnOcrParser, AdapterBase)
    provider = LightOnOcrParser({})
    assert isinstance(provider, Parser)  # runtime_checkable structural check


def test_profile_is_gpu_evidence_bearing_apache() -> None:
    """The declared ProviderProfile: GPU, model-bearing, non-deterministic, evidence-bearing Apache."""
    profile = LightOnOcrParser.profile()
    assert profile.compute == "gpu"
    assert profile.model_id == "lightonai/LightOnOCR-1B-1025"
    assert profile.license == "Apache-2.0"
    assert profile.license != UNVERIFIED_LICENSE
    assert profile.license_verified is True
    assert profile.license_code == "Apache-2.0"
    assert profile.license_weights == "Apache-2.0"  # weights recorded separately (ADR-0012)
    assert "huggingface.co/lightonai/LightOnOCR-1B-1025" in (profile.license_source or "")
    assert profile.license_verified_on == "2026-07-08"
    assert profile.deterministic is False  # VLM decoding: not promised byte-identical
    assert profile.batch is True
    assert profile.cost_per_1k is None


def test_device_skips_with_flag_on_cpu_only_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compute='gpu' provider on a CPU-only host resolves to the skip-with-flag decision (G1)."""
    monkeypatch.setenv("LATENCE_CUDA", "0")
    provider = LightOnOcrParser({})
    assert provider.device.skipped is True
    assert provider.device.device is None


def test_device_routes_to_cuda_on_a_rig(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a CUDA host the device seam routes the GPU provider to cuda (placement flows through)."""
    monkeypatch.setenv("LATENCE_CUDA", "1")
    provider = LightOnOcrParser({})
    assert provider.device.skipped is False
    assert provider.device.device == "cuda"


# --- happy path: OCR markdown-per-page, offsets round-trip, Provenance carry --


def test_multipage_pdf_ocrs_to_correct_pages_and_roundtrip_offsets(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """A multi-page scanned PDF → PARSED record; 2-page map from OCR markdown; offsets round-trip."""
    pdf = make_lighton_pdf(
        ["# The quarterly report\n\nRevenue up.", "## Page two\n\nOutlook follows."]
    )
    provider = LightOnOcrParser({})
    (record,) = list(provider.parse([_pdf_input(pdf, sensitivity="confidential")]))

    assert record.disposition is Disposition.PARSED
    assert record.error is None
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    # EXACT content — the load-bearing regression guard for the decode-slice contract (the issue's #1
    # non-negotiable). If the Provider ever drops ``output_ids[0, prompt_len:]``, the fake's
    # ``<|prompt-scaffold|>`` prompt tokens leak in front of EVERY page and this exact-equality fails.
    # (Substring ``in`` assertions cannot catch a leaked prefix — they stay green on the corrupted
    # output — so this test exercises the PROVIDER path, not just the fake's faithfulness.)
    assert record.content == (
        "# The quarterly report\n\nRevenue up.\n\n## Page two\n\nOutlook follows."
    )
    assert "prompt-scaffold" not in record.content  # an un-sliced provider FAILS here
    assert record.media_type == "text/markdown"

    # Provenance carried (source lineage intact; char/page span filled to what the record occupies).
    assert record.provenance.file_name == "scan.pdf"
    assert record.provenance.char_start == 0
    assert record.provenance.char_end == len(record.content)
    assert record.provenance.page_start == 1
    assert record.provenance.page_end == 2
    # Classification carried; sensitivity inherited from the input's Classification.
    assert record.classification.sensitivity == "confidential"

    # Offsets round-trip with ZERO drift (ADR-0031, the same PageOffsetIndex the Runner audit uses).
    index = PageOffsetIndex(record.page_map)
    for span in record.page_map.pages:
        assert index.page_for_offset(span.char_start) == span.page_number
        assert index.page_for_offset(span.char_end - 1) == span.page_number
    assert index.diagnostics.max_drift_chars == 0
    assert index.diagnostics.recovered == 0


def test_pages_land_in_input_order(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """Per-page OCR markdown is assembled in page order — page 1 precedes page 2."""
    pdf = make_lighton_pdf(["AAA first page", "ZZZ second page"])
    (record,) = list(LightOnOcrParser({}).parse([_pdf_input(pdf)]))
    assert record.content.index("AAA first page") < record.content.index("ZZZ second page")


def test_single_image_input_single_page(
    fake_lighton: None, make_lighton_image: Callable[[str], bytes]
) -> None:
    """A single scanned image → a single-page markdown record with language detected at Parse."""
    img = make_lighton_image("the framework reads the pixels and extracts the text\n")
    (record,) = list(LightOnOcrParser({}).parse([_image_input(img)]))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 1
    assert record.classification.language == "en"


def test_size_after_parse_is_assembled_byte_length(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """size_after_parse is the assembled-markdown byte size (the S2 observability signal)."""
    (record,) = list(LightOnOcrParser({}).parse([_pdf_input(make_lighton_pdf(["hello world"]))]))
    assert size_after_parse(record) == len(record.content.encode("utf-8"))
    assert size_after_parse(record) > 0


def test_model_placed_on_resolved_cuda_device(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """The model is moved to the device the seam resolved (cuda), i.e. placement flows through."""
    provider = LightOnOcrParser({})
    list(provider.parse([_pdf_input(make_lighton_pdf(["x"]))]))
    assert provider._model is not None
    assert provider._model.device == "cuda"  # model.to("cuda") was called via the seam


# --- the shape-matching fake really mirrors the real output shape ------------


def test_offline_fake_is_faithful_to_generate_slice_decode_contract(fake_lighton: None) -> None:
    """The anti-false-green anchor: the fake models the REAL generate/slice/decode contract.

    A reviewer points at THIS test to confirm the offline double matches the real model — the single
    thing whose divergence made PaddleOCR-VL a false green. The real model-card API is:

        output_ids = model.generate(**inputs)                       # FULL: prompt + continuation
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]  # SLICE off the prompt tokens
        text = processor.decode(generated_ids, skip_special_tokens=True)  # -> str

    The fake is faithful to that: ``generate`` returns prompt+continuation, ``input_ids.shape[1]``
    is the prompt length, and ``decode`` of the SLICED continuation is the clean markdown — while
    decoding the FULL (un-sliced) sequence would visibly include the prompt scaffolding. So a
    Provider that forgets the slice FAILS here rather than passing green.
    """
    from transformers import (  # the injected fakes
        LightOnOcrForConditionalGeneration,
        LightOnOcrProcessor,
    )

    processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-1B-1025")
    model = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-1B-1025")

    messages = [{"role": "user", "content": [{"type": "image", "image": "FAKE_PAGE::# heading"}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    prompt_len = inputs["input_ids"].shape[1]
    assert prompt_len > 0  # the prompt (chat scaffolding) genuinely occupies tokens

    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    # The FULL sequence is longer than the prompt — generate() appended a real continuation.
    assert output_ids.shape[1] > prompt_len

    # Decoding the SLICED continuation is the clean markdown (the Provider's correct path).
    sliced = output_ids[0, prompt_len:]
    assert processor.decode(sliced, skip_special_tokens=True) == "# heading"

    # Decoding the FULL sequence (the BUG — no slice) visibly PREPENDS the prompt scaffolding, so
    # an un-sliced Provider would corrupt every page. This is the false-green trap, made to fail.
    unsliced = output_ids[0, 0:]
    corrupted = processor.decode(unsliced, skip_special_tokens=True)
    assert corrupted != "# heading"
    assert corrupted.endswith("# heading")
    assert "prompt-scaffold" in corrupted


# --- batching over multiple documents (AdapterBase.batched) ------------------


def test_batches_multiple_documents_in_input_order(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """The batching path runs over multiple documents, preserving input order (any batch_size)."""
    inputs = [
        _pdf_input(make_lighton_pdf([f"document number {i}"]), file_name=f"d{i}.pdf")
        for i in range(5)
    ]
    provider = LightOnOcrParser({"batch_size": 2})
    records = list(provider.parse(inputs))
    assert [r.provenance.file_name for r in records] == [f"d{i}.pdf" for i in range(5)]
    assert all(r.disposition is Disposition.PARSED for r in records)


def test_batch_size_invariant_output_identical(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """Batching is purely memory-shaping: batch_size=1 and batch_size=32 yield identical output."""
    inputs = [
        _pdf_input(make_lighton_pdf([f"content {i}"]), file_name=f"d{i}.pdf") for i in range(4)
    ]
    one = [r.model_dump_json() for r in LightOnOcrParser({"batch_size": 1}).parse(inputs)]
    many = [r.model_dump_json() for r in LightOnOcrParser({"batch_size": 32}).parse(inputs)]
    assert one == many


# --- the pixel-bomb DoS cap (the Hardening-bar item; untrusted input is rasterised) ----------


def test_clamp_scale_reduces_oversized_page_under_budget() -> None:
    """A giant page's render scale is clamped so the bitmap area stays under the megapixel budget."""
    provider = LightOnOcrParser({})
    # A 100 000 x 100 000-pt page at scale ~2.78 (200 DPI) would be ~7.7e10 px — far over budget.
    clamped = provider._clamp_scale(100_000.0, 100_000.0, 200.0 / 72.0)
    area_px = 100_000.0 * clamped * 100_000.0 * clamped
    assert area_px <= _MAX_RENDER_PIXELS + 1  # clamped to the budget (float slack)
    assert clamped < 200.0 / 72.0  # genuinely reduced


def test_clamp_scale_leaves_normal_page_untouched() -> None:
    """A normal (in-budget) page keeps its requested scale — the clamp only shrinks oversized pages."""
    provider = LightOnOcrParser({})
    requested = 200.0 / 72.0
    # A US-Letter page (612 x 792 pt) at 200 DPI is ~1.5 MP — well under budget, so unchanged.
    assert provider._clamp_scale(612.0, 792.0, requested) == requested


def test_oversized_pdf_page_clamped_not_crashed(
    fake_lighton: None, make_lighton_pdf: Callable[..., bytes]
) -> None:
    """An oversized PDF page rasterises with a CLAMPED scale (bitmap under budget), still PARSED."""
    pdf = make_lighton_pdf(["# still legible"], oversized=True)
    (record,) = list(LightOnOcrParser({}).parse([_pdf_input(pdf)]))
    # The page still OCRs (the clamp shrinks the raster, it does not drop the page).
    assert record.disposition is Disposition.PARSED
    assert "still legible" in record.content


def test_decompression_bomb_image_degrades_to_parse_error(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """A decompression-bomb image (pixel count over budget) → graceful PARSE_ERROR, no huge alloc."""
    bomb = make_lighton_image("evil", pixels=_MAX_RENDER_PIXELS + 1)
    (record,) = list(LightOnOcrParser({}).parse([_image_input(bomb, file_name="bomb.png")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    # The typed error names the op + native type, never the (untrusted) image content.
    assert "decode_image" in record.error
    assert "evil" not in record.error


def test_in_budget_image_loads_and_restores_max_image_pixels(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """An in-budget image OCRs fine, and the global Image.MAX_IMAGE_PIXELS is restored afterwards."""
    from PIL import Image  # the injected fake

    original = Image.MAX_IMAGE_PIXELS
    img = make_lighton_image("legible page\n", pixels=1000)
    (record,) = list(LightOnOcrParser({}).parse([_image_input(img)]))
    assert record.disposition is Disposition.PARSED
    assert original == Image.MAX_IMAGE_PIXELS  # global Pillow state restored (no side effect)


# --- graceful failure: native exception → typed → PARSE_ERROR (no PII) -------


def test_type_spoofed_pdf_rejected_before_rasteriser(fake_lighton: None) -> None:
    """A .pdf whose bytes are not a PDF is rejected as a PARSE_ERROR (offset-0 spoof guard)."""
    spoof = b"MZ\x90\x00this is actually an executable, not a pdf"
    (record,) = list(LightOnOcrParser({}).parse([_pdf_input(spoof, file_name="spoof.pdf")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "spoof" in (record.error or "").lower() or "signature" in (record.error or "").lower()


def test_type_spoofed_image_rejected_before_decode(fake_lighton: None) -> None:
    """A .png whose bytes are not a PNG is rejected as a PARSE_ERROR (image spoof guard)."""
    spoof = b"not-a-png-at-all"
    (record,) = list(LightOnOcrParser({}).parse([_image_input(spoof, file_name="spoof.png")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "signature" in (record.error or "").lower() or "spoof" in (record.error or "").lower()


def test_unsupported_file_type_is_parse_error(fake_lighton: None) -> None:
    """An unsupported file type degrades gracefully to a PARSE_ERROR, never a crash."""
    inp = ParserInput(
        provenance=_provenance("thing.xyz", "xyz"),
        classification=Classification(language="und"),
        content=b"whatever",
    )
    (record,) = list(LightOnOcrParser({}).parse([inp]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "xyz" in (record.error or "")


def test_native_model_fault_maps_to_typed_error_no_leak(
    fake_lighton: None,
    monkeypatch: pytest.MonkeyPatch,
    make_lighton_pdf: Callable[[list[str]], bytes],
) -> None:
    """A native model fault → typed error → graceful PARSE_ERROR; NO image/OCR-text/PII leak.

    Force the fake model's ``generate`` to raise a native exception whose message echoes a secret;
    the guard must map it to a typed error carrying only the provider + op + native TYPE name — the
    OCR-is-PII-bearing discipline (C2/C6).
    """
    from transformers import LightOnOcrForConditionalGeneration  # the injected fake

    class _NativeBoom(Exception):
        def __str__(self) -> str:  # a native message echoing a secret — must NOT leak
            return "boom sk-live-OCR-LEAK alice@example.com"

    def _explode(self: object, *_a: object, **_k: object) -> object:
        raise _NativeBoom

    monkeypatch.setattr(LightOnOcrForConditionalGeneration, "generate", _explode)
    provider = LightOnOcrParser({})
    (record,) = list(provider.parse([_pdf_input(make_lighton_pdf(["secret page"]))]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert "sk-live-OCR-LEAK" not in record.error
    assert "alice@example.com" not in record.error
    assert "_NativeBoom" in record.error  # only the native TYPE name
    assert "parse" in record.error  # the op label


def test_parse_error_carries_no_ocr_text_or_pii(
    fake_lighton: None,
    monkeypatch: pytest.MonkeyPatch,
    make_lighton_pdf: Callable[[list[str]], bytes],
) -> None:
    """The mapped ProviderError never carries the raw OCR text / the input content (C6)."""
    from transformers import LightOnOcrForConditionalGeneration

    def _explode(self: object, *_a: object, **_k: object) -> object:
        raise RuntimeError("cuda OOM while decoding SECRET-CONTENT-abc")

    monkeypatch.setattr(LightOnOcrForConditionalGeneration, "generate", _explode)
    with pytest.raises(ProviderError) as exc_info:
        # Drive the OCR path directly so the typed error surfaces (not yet degraded).
        LightOnOcrParser({})._ocr_pages(["FAKE_PAGE::x"])
    assert isinstance(exc_info.value, LatenceError)
    assert "SECRET-CONTENT" not in str(exc_info.value)  # native message not leaked
    assert "parse" in str(exc_info.value)


def test_bad_config_native_exception_maps_to_config_error(
    fake_lighton: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native ValueError from the loader (bad model id) → a typed ConfigError via the guard."""
    from latence_core.errors import ConfigError
    from transformers import LightOnOcrForConditionalGeneration

    def _bad_load(model_id: str, **_k: object) -> object:
        msg = "no such model checkpoint"
        raise ValueError(msg)

    monkeypatch.setattr(LightOnOcrForConditionalGeneration, "from_pretrained", _bad_load)
    with pytest.raises(ConfigError):
        LightOnOcrParser({"model": "does/not-exist"})._load_model()


# --- device gating: a CPU-only host does not fabricate a run -----------------


def test_empty_pdf_yields_empty_page_not_crash(
    fake_lighton: None, make_lighton_pdf: Callable[[list[str]], bytes]
) -> None:
    """A zero-page/empty source yields a single empty page (recorded), not a crash."""
    (record,) = list(LightOnOcrParser({}).parse([_pdf_input(make_lighton_pdf([""]))]))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert isinstance(record.content, str)


# --- W8: OCR page-batching equivalence (batched == per-page, ceil(N/batch) calls) --------


def _ocr_texts(provider: LightOnOcrParser, images: list[str]) -> list[str]:
    """Run the provider's OCR seam over a list of fake page-image markers → markdown per page."""
    return provider._ocr_pages(list(images))


# A VARIABLE-LENGTH page fixture: the pages differ in markdown length so a WRONG shared-length /
# first-item prompt-slice assumption bleeds one page's prompt/pad tokens into another's markdown and
# FAILS the record-for-record equality below. Distinct content per page also catches a mis-mapped
# batch (off-by-one / wrong order).
_VARLEN_PAGES = [
    "# short",
    "## a considerably longer second page with much more markdown text than the first",
    "third",
    "## fourth page — medium length body with a few more words here",
    "5",
]


def test_ocr_page_batching_equals_per_page_record_for_record(fake_lighton: None) -> None:
    """Batched OCR over N variable-length pages == per-page decode, record-for-record."""
    markers = [f"FAKE_PAGE::{md}" for md in _VARLEN_PAGES]

    # Per-page baseline: force ocr_batch_size=1 (one page per generate call).
    per_page = _ocr_texts(LightOnOcrParser({"ocr_batch_size": 1}), markers)
    # Batched: several pages share one generate call.
    batched = _ocr_texts(LightOnOcrParser({"ocr_batch_size": 4}), markers)

    assert per_page == _VARLEN_PAGES  # the fake decodes each marker back to its markdown
    assert batched == per_page  # record-for-record identical, variable lengths and all
    for page in batched:
        assert "prompt-scaffold" not in page  # no leaked prompt tokens (a wrong slice would leak)


def test_ocr_generate_called_ceil_n_over_batch_size_times(fake_lighton: None) -> None:
    """The model's ``generate`` is called exactly ``ceil(n_pages / ocr_batch_size)`` times."""
    markers = [f"FAKE_PAGE::{md}" for md in _VARLEN_PAGES]  # 5 pages

    provider = LightOnOcrParser({"ocr_batch_size": 2})
    provider._ocr_pages(list(markers))
    model = provider._model  # the fake, loaded on first OCR
    # 5 pages, batch 2 → ceil(5/2) = 3 generate calls (2 + 2 + 1), NOT 5.
    assert model.generate_calls == 3

    provider2 = LightOnOcrParser({"ocr_batch_size": 8})  # one batch covers all 5
    provider2._ocr_pages(list(markers))
    assert provider2._model.generate_calls == 1


def test_ocr_batch_size_config_validated() -> None:
    """``ocr_batch_size`` must be >= 1 (a batch of zero would never make progress)."""
    with pytest.raises(ProviderError, match="ocr_batch_size"):
        LightOnOcrParser({"ocr_batch_size": 0})


# --- W8: longest-side (1540px) downscale — LightOn's official preprocessing recipe --------


def test_oversized_image_downscaled_to_longest_side_1540_aspect_preserved(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """A 1200×2339 image downscales so its longest side == 1540, aspect ratio preserved (default).

    T-OCR-2 amended the width assertion: exact proportional scaling puts it at ≈790, which is 56.4
    patches — an odd patch grid, whose trailing column ``Mistral3PatchMerger``'s 2×2 ``unfold``
    silently discards. Both sides now snap onto the 28-px merge grid, so 790 → 784 (28 patches).
    The invariants themselves are property-tested in ``test_rasterize_merge_grid.py``.
    """
    img_bytes = make_lighton_image("scanned page text\n", dims=(1200, 2339))
    provider = LightOnOcrParser({})  # default max_longest_dim = 1540
    image = provider._image_from_bytes(img_bytes)
    w, h = image.size
    assert max(w, h) == 1540  # longest side capped to LightOn's target (1540 = 110 patches, even)
    # Aspect ratio preserved to the finest granularity the merge grid can express: 1200/2339 ≈
    # 0.513 → width ≈ 1540 * 0.513 ≈ 790 → the nearest grid point at or below the page is 784.
    assert abs(w - round(1200 * (1540 / 2339))) <= 28
    assert w == 784
    assert h == 1540
    assert (w % 28, h % 28) == (0, 0)


def test_image_untouched_when_cap_disabled(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """max_longest_dim=0 disables the cap — an oversized image is passed through untouched."""
    img_bytes = make_lighton_image("text\n", dims=(1200, 2339))
    image = LightOnOcrParser({"max_longest_dim": 0})._image_from_bytes(img_bytes)
    assert image.size == (1200, 2339)
    assert image.resized_to is None  # never resized


def test_image_untouched_when_already_within_cap(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """An image already within the cap AND on the merge grid is not upscaled or resampled.

    T-OCR-2 split the old single case in two. Being under the 1540 cap is no longer sufficient to
    be left alone — the sides must also be multiples of 28, or the vision tower drops the trailing
    patch row/column. 1120×1540 is under/at the cap and on the grid, so it is passed through with
    no resample at all; the "under the cap but off the grid" case is the test below.
    """
    img_bytes = make_lighton_image("text\n", dims=(1120, 1540))
    image = LightOnOcrParser({"max_longest_dim": 1540})._image_from_bytes(img_bytes)
    assert image.size == (1120, 1540)
    assert image.resized_to is None


def test_image_under_the_cap_is_snapped_down_to_the_merge_grid_never_up(
    fake_lighton: None, make_lighton_image: Callable[..., bytes]
) -> None:
    """800×1000 is under the cap but off the grid — it snaps DOWN to 784×980, never up to 1008."""
    img_bytes = make_lighton_image("text\n", dims=(800, 1000))
    image = LightOnOcrParser({"max_longest_dim": 1540})._image_from_bytes(img_bytes)
    assert image.size == (784, 980)
    assert image.size[0] <= 800 and image.size[1] <= 1000  # no upscaling, ever
