"""Unit tests for the served-vLLM LightOn-OCR Parser (W10) — happy path + the anti-false-green teeth.

These prove the drop-in Parser contract and the five reviewer checks:

1. **Preprocessing IDENTICAL to parser.lighton** — the Provider uses the SAME shared
   :class:`~latence_parser_lighton.rasterize.PageRasteriser` (asserted directly), and the 1540px /
   bomb-cap behaviour round-trips.
2. **Offsets round-trip exactly** — the per-page markdown assembles onto ``PageMap.from_page_texts``
   with zero drift (the same ``PageOffsetIndex`` the Runner audit uses).
3. **Concurrency real + bounded + fail-open** — a timing test proves ``ceil(N/max_concurrency)``,
the
   semaphore bounds ``max_in_flight``, and a per-page endpoint failure degrades that page only.
4. **The mock mirrors the REAL vLLM response shape** — an async ``chat.completions.create``
coroutine
   returning ``choices[0].message.content``, decoding the base64 image the Provider actually POSTed.
5. **max_tokens 4096 default + temp 0.2 / top_p 0.9** — asserted on the request kwargs the Provider
   sends.

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest). Offline via the
faithful async vLLM mock + the offline rasteriser fakes (``fake_vllm`` fixture); the real OCR
against
a live vLLM server is pod-validated (scripts/validate-lighton-vllm-on-gpu.md).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
from _vllm_fakes import StatusError, VllmScript
from latence_core.capability import Parser
from latence_core.contracts import (
    Classification,
    Disposition,
    ParserInput,
    Provenance,
)
from latence_core.errors import ProviderError
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.profile import UNVERIFIED_LICENSE
from latence_parser_lighton.rasterize import PageRasteriser
from latence_parser_lighton_vllm.provider import (
    _DEFAULT_MAX_TOKENS,
    LightOnVllmParser,
    size_after_parse,
)

_BASE = {"base_url": "http://stub-vllm/v1"}


def _provenance(file_name: str, file_type: str, size: int = 100) -> Provenance:
    return Provenance(
        source_uri=f"mem://docs/{file_name}",
        file_name=file_name,
        file_type=file_type,
        file_size=size,
        document_id=f"sha256:{file_name}",
    )


def _pdf_input(
    content: bytes, file_name: str = "scan.pdf", sensitivity: str = "internal"
) -> ParserInput:
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


def _provider(**extra: Any) -> LightOnVllmParser:
    return LightOnVllmParser({**_BASE, **extra})


# --- structural: drop-in Parser + declared profile --------------------------


def test_satisfies_parser_capability_structurally() -> None:
    """The Provider satisfies the Parser Capability structurally (a drop-in for parser.lighton)."""
    assert isinstance(_provider(), Parser)  # runtime_checkable structural check


def test_profile_is_endpoint_evidence_bearing_apache() -> None:
    """The declared ProviderProfile: endpoint-gated (compute='either'), model-bearing Apache-2.0."""
    profile = LightOnVllmParser.profile()
    # compute='either' — a thin HTTP client (rasterise on CPU, OCR behind the vLLM endpoint); it is
    # infra-gated like embedding.endpoint (stub-scoped in conformance), NOT device-skipped.
    assert profile.compute == "either"
    assert profile.model_id == "lightonai/LightOnOCR-2-1B"
    assert profile.license == "Apache-2.0"
    assert profile.license != UNVERIFIED_LICENSE
    assert profile.license_verified is True
    assert profile.license_code == "Apache-2.0"
    assert profile.license_weights == "Apache-2.0"  # weights recorded separately (ADR-0012)
    assert "huggingface.co/lightonai/LightOnOCR-2-1B" in (profile.license_source or "")
    assert profile.deterministic is False  # served VLM decoding: not promised byte-identical
    assert profile.batch is True
    assert profile.cost_per_1k is None


# --- #1 preprocessing IDENTICAL to parser.lighton ---------------------------


def test_uses_the_shared_rasteriser_identical_to_parser_lighton() -> None:
    """The Provider rasterises through the SAME PageRasteriser parser.lighton uses (no divergence)."""
    provider = _provider()
    assert isinstance(provider._rasteriser, PageRasteriser)
    # The rasterisation knobs default to LightOn's official recipe (200 dpi / 1540px), same as
    # parser.lighton — so the pixel preprocessing cannot silently diverge between the two back-ends.
    assert provider._dpi == 200
    assert provider._max_longest_dim == 1540


# --- happy path: OCR markdown-per-page, offsets round-trip, Provenance carry --


def test_multipage_pdf_ocrs_to_correct_pages_and_roundtrip_offsets(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """A multi-page scanned PDF → PARSED record; 2-page map from OCR markdown; offsets round-trip."""
    pdf = make_pdf(["# The quarterly report\n\nRevenue up.", "## Page two\n\nOutlook follows."])
    (record,) = list(_provider().parse([_pdf_input(pdf, sensitivity="confidential")]))

    assert record.disposition is Disposition.PARSED
    assert record.error is None
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    # EXACT content — the faithful vLLM mock transcribes each page image back to its markdown.
    assert record.content == (
        "# The quarterly report\n\nRevenue up.\n\n## Page two\n\nOutlook follows."
    )
    assert record.media_type == "text/markdown"

    # Provenance carried (source lineage intact; char/page span filled to what the record occupies).
    assert record.provenance.file_name == "scan.pdf"
    assert record.provenance.char_start == 0
    assert record.provenance.char_end == len(record.content)
    assert record.provenance.page_start == 1
    assert record.provenance.page_end == 2
    assert record.classification.sensitivity == "confidential"

    # #2: Offsets round-trip with ZERO drift (ADR-0031, the same PageOffsetIndex the Runner audit
    # uses).
    index = PageOffsetIndex(record.page_map)
    for span in record.page_map.pages:
        assert index.page_for_offset(span.char_start) == span.page_number
        assert index.page_for_offset(span.char_end - 1) == span.page_number
    assert index.diagnostics.max_drift_chars == 0
    assert index.diagnostics.recovered == 0


def test_single_image_input_single_page(
    fake_vllm: None, make_image: Callable[[str], bytes]
) -> None:
    """A single scanned image → a single-page markdown record with language detected at Parse."""
    img = make_image("the framework reads the pixels and extracts the text\n")
    (record,) = list(_provider().parse([_image_input(img)]))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 1
    assert record.classification.language == "en"


def test_size_after_parse_is_assembled_byte_length(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """size_after_parse is the assembled-markdown byte size (the S2 observability signal)."""
    (record,) = list(_provider().parse([_pdf_input(make_pdf(["hello world"]))]))
    assert size_after_parse(record) == len(record.content.encode("utf-8"))
    assert size_after_parse(record) > 0


# --- #4 the mock mirrors the REAL vLLM response shape + #5 sampling params ---


def test_request_uses_openai_multimodal_shape_and_official_sampling(
    fake_vllm: None, make_image: Callable[[str], bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POST is the REAL OpenAI multimodal shape with max_tokens 4096 / temp 0.2 / top_p 0.9.

    Captures the kwargs the Provider hands ``chat.completions.create`` and asserts: the message is
    a user turn with an ``image_url`` (a base64 PNG ``data:`` URL) + a text prompt part, and the
    sampling knobs are LightOn's official defaults — the #5 reviewer check, and #4 (the mock is the
    real client shape, not a fictional API).
    """
    captured: dict[str, Any] = {}
    import openai  # the injected fake

    original_create = openai.AsyncOpenAI().chat.completions.create.__func__

    async def _spy(self: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return await original_create(self, **kwargs)

    monkeypatch.setattr(openai.AsyncOpenAI().chat.completions.__class__, "create", _spy)

    list(_provider().parse([_image_input(make_image("# page\n"))]))

    assert captured["model"] == "lightonai/LightOnOCR-2-1B"
    assert captured["max_tokens"] == _DEFAULT_MAX_TOKENS == 4096  # content-safe, NOT 1024
    assert captured["temperature"] == 0.2  # LightOn's official
    assert captured["top_p"] == 0.9  # LightOn's official
    messages = captured["messages"]
    assert messages[0]["role"] == "user"
    parts = messages[0]["content"]
    image_part = next(p for p in parts if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert any(p["type"] == "text" for p in parts)  # the OCR instruction accompanies the image


def test_max_tokens_default_is_4096_not_1024() -> None:
    """The per-page decode ceiling defaults to 4096 (content-safe), never 1024 (truncates dense pages)."""
    assert _provider()._max_tokens == 4096
    # Explicitly overridable for known-light corpora (speed), but the DEFAULT is the content-safe
    # one.
    assert _provider(max_tokens=1024)._max_tokens == 1024


# --- #3 concurrency real + bounded (timing) + determinism -------------------


def test_wall_clock_is_ceil_n_over_c_not_n(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """N pages at concurrency C take ~ceil(N/C) x delay (concurrent), bounded by the semaphore."""
    n, concurrency, delay = 8, 4, 0.05
    VllmScript.default_delay = delay
    pdf = make_pdf([f"page {i}" for i in range(n)])
    start = time.monotonic()
    (record,) = list(_provider(max_concurrency=concurrency).parse([_pdf_input(pdf)]))
    elapsed = time.monotonic() - start
    assert record.page_map is not None
    assert len(record.page_map.pages) == n
    # ceil(8/4) = 2 waves x 0.05 = ~0.10s, NOT 8 x 0.05 = 0.40s (sequential).
    assert elapsed < 0.30, f"expected ~0.10s (concurrent), got {elapsed:.3f}s (looks sequential)"
    assert VllmScript.max_in_flight <= concurrency  # bounded by the semaphore
    assert VllmScript.max_in_flight > 1  # but genuinely concurrent


def test_max_concurrency_one_serializes(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """max_concurrency=1 is sequential — only ever one request in flight (byte-identical output)."""
    n, delay = 4, 0.02
    VllmScript.default_delay = delay
    pdf = make_pdf([f"page {i}" for i in range(n)])
    (record,) = list(_provider(max_concurrency=1).parse([_pdf_input(pdf)]))
    assert VllmScript.max_in_flight == 1  # never more than one in flight
    assert record.page_map is not None
    assert len(record.page_map.pages) == n


def test_batched_pages_equal_per_page_record_for_record(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """Concurrent (batched) page OCR == per-page (max_concurrency=1), record-for-record with offsets.

    The equivalence crux: fanning the pages out concurrently must be byte-identical to running them
    one at a time — same markdown, same page count, same offsets, same order. A mis-mapped/out-of-
    order concurrent assembly FAILS here.
    """
    # VARIABLE-LENGTH pages so a wrong shared-length / out-of-order assembly would surface.
    pages = [
        "# short",
        "## a considerably longer second page with much more markdown text than the first",
        "third",
        "## fourth page — medium length body with a few more words here",
        "5",
    ]
    pdf = make_pdf(pages)

    (per_page,) = list(_provider(max_concurrency=1).parse([_pdf_input(pdf)]))
    VllmScript.reset()
    (concurrent,) = list(_provider(max_concurrency=8).parse([_pdf_input(pdf)]))

    assert per_page.model_dump_json() == concurrent.model_dump_json()  # record-for-record identical
    # And each page landed in order with the right markdown.
    assert concurrent.content.split("\n\n") == pages


def test_shuffled_completion_order_is_page_ordered(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """Later pages complete FIRST (descending delays) → the page map is still in input page order.

    Determinism under concurrency: the assembly is index-keyed, so a shuffled completion order never
    perturbs the page order or the offsets — the same discipline relation.llm carries.
    """
    n = 4
    pages = [f"page_{i} content" for i in range(n)]
    # page i sleeps (n-i)*delay so page (n-1) completes first, page 0 last.
    for i, md in enumerate(pages):
        VllmScript.delay_for[md] = (n - i) * 0.02
    pdf = make_pdf(pages)
    (record,) = list(_provider(max_concurrency=n).parse([_pdf_input(pdf)]))
    # The page markdown lands in INPUT order despite the reversed completion order.
    assert record.content.split("\n\n") == pages
    assert record.page_map is not None
    index = PageOffsetIndex(record.page_map)
    for span in record.page_map.pages:
        assert index.page_for_offset(span.char_start) == span.page_number


def test_two_runs_byte_identical(fake_vllm: None, make_pdf: Callable[[list[str]], bytes]) -> None:
    """Two concurrent runs over the same input are byte-identical (seeded, order-independent)."""
    pages = [f"page {i}" for i in range(6)]
    pdf = make_pdf(pages)
    dumps = []
    for _ in range(2):
        VllmScript.reset()
        for i, md in enumerate(pages):
            VllmScript.delay_for[md] = (6 - i) * 0.005
        (record,) = list(_provider(max_concurrency=3).parse([_pdf_input(pdf)]))
        dumps.append(record.model_dump_json())
    assert dumps[0] == dumps[1]


# --- #3 per-page fail-open (one bad page never aborts the document) ----------


def test_one_page_endpoint_failure_fails_open_document_keeps_good_pages(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """A terminal 4xx on ONE page degrades that page to empty; the document + other pages survive."""
    pages = ["good page one", "BAD PAGE", "good page three"]
    VllmScript.overrides["BAD PAGE"] = [("raise", StatusError("bad request", status_code=400))]
    pdf = make_pdf(pages)
    (record,) = list(_provider(max_concurrency=3).parse([_pdf_input(pdf)]))
    # The document is still PARSED (fail-open), the two good pages OCR'd, the bad page is empty.
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 3  # every page still has a span (the bad one is empty)
    assert "good page one" in record.content
    assert "good page three" in record.content
    # The failed page contributed no markdown, but did NOT abort the document.
    per_page = record.content.split("\n\n")
    assert per_page == ["good page one", "", "good page three"]


def test_all_pages_failing_degrades_document_to_parse_error(
    fake_vllm: None, make_image: Callable[[str], bytes]
) -> None:
    """A single-page doc whose only page fails → a per-document PARSE_ERROR (nothing decoded)."""
    # A terminal 4xx (not retryable) fails the page on the first attempt → the page fails open →
    # the single-page document decoded nothing → a per-document PARSE_ERROR.
    VllmScript.overrides["only page"] = [("raise", StatusError("bad request", status_code=400))]
    (record,) = list(_provider(max_concurrency=1).parse([_image_input(make_image("only page"))]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert "page OCR failed" in record.error  # the typed per-page fail-open reason surfaced


def test_retryable_then_success_tallies_endpoint_retries(
    fake_vllm: None, make_image: Callable[[str], bytes]
) -> None:
    """A 503-then-200 page is retried (backoff) and succeeds; the retry is tallied for the Report."""
    VllmScript.overrides["flaky page"] = [
        ("raise", StatusError("overloaded", status_code=503)),
        ("reply", "flaky page"),
    ]
    provider = _provider(max_concurrency=1)
    (record,) = list(provider.parse([_image_input(make_image("flaky page"))]))
    assert record.disposition is Disposition.PARSED
    assert "flaky page" in record.content
    assert provider.endpoint_retries == 1  # one transient failure absorbed (S12 observability)


# --- guards intact: spoof / bomb / unsupported → graceful PARSE_ERROR --------


def test_type_spoofed_pdf_rejected_before_endpoint(fake_vllm: None) -> None:
    """A .pdf whose bytes are not a PDF → PARSE_ERROR (spoof guard), no endpoint call."""
    spoof = b"MZ\x90\x00this is actually an executable, not a pdf"
    (record,) = list(_provider().parse([_pdf_input(spoof, file_name="spoof.pdf")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "signature" in (record.error or "").lower() or "spoof" in (record.error or "").lower()
    assert VllmScript.calls == 0  # rejected before any POST


def test_type_spoofed_image_rejected_before_endpoint(fake_vllm: None) -> None:
    """A .png whose bytes are not a PNG → PARSE_ERROR (image spoof guard), no endpoint call."""
    spoof = b"not-a-png-at-all"
    (record,) = list(_provider().parse([_image_input(spoof, file_name="spoof.png")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "signature" in (record.error or "").lower() or "spoof" in (record.error or "").lower()
    assert VllmScript.calls == 0


def test_decompression_bomb_image_degrades_to_parse_error(
    fake_vllm: None, make_image: Callable[..., bytes]
) -> None:
    """A decompression-bomb image (pixel count over budget) → graceful PARSE_ERROR, no huge alloc."""
    from latence_parser_lighton.rasterize import MAX_RENDER_PIXELS

    bomb = make_image("evil", pixels=MAX_RENDER_PIXELS + 1)
    (record,) = list(_provider().parse([_image_input(bomb, file_name="bomb.png")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert "decode_image" in record.error
    assert "evil" not in record.error  # the typed error never carries the untrusted content
    assert VllmScript.calls == 0


def test_oversized_pdf_page_clamped_not_crashed(
    fake_vllm: None, make_pdf: Callable[..., bytes]
) -> None:
    """An oversized PDF page rasterises with a CLAMPED scale (bitmap under budget), still PARSED."""
    pdf = make_pdf(["# still legible"], oversized=True)
    (record,) = list(_provider().parse([_pdf_input(pdf)]))
    assert record.disposition is Disposition.PARSED
    assert "still legible" in record.content


def test_oversized_image_downscaled_to_longest_side_1540(
    fake_vllm: None, make_image: Callable[..., bytes]
) -> None:
    """A 1200×2339 image downscales so its longest side == 1540, aspect preserved (shared recipe)."""
    img_bytes = make_image("scanned page text\n", dims=(1200, 2339))
    image = _provider()._rasteriser._image_from_bytes(img_bytes)
    w, h = image.size
    assert max(w, h) == 1540  # LightOn's official longest-side target, identical to parser.lighton
    assert h == 1540
    assert w == round(1200 * (1540 / 2339))


def test_unsupported_file_type_is_parse_error(fake_vllm: None) -> None:
    """An unsupported file type degrades gracefully to a PARSE_ERROR, never a crash."""
    inp = ParserInput(
        provenance=_provenance("thing.xyz", "xyz"),
        classification=Classification(language="und"),
        content=b"whatever",
    )
    (record,) = list(_provider().parse([inp]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "xyz" in (record.error or "")


def test_empty_pdf_yields_empty_page_not_crash(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """A zero-page/empty source yields a single empty page (recorded), not a crash."""
    (record,) = list(_provider().parse([_pdf_input(make_pdf([""]))]))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert isinstance(record.content, str)


# --- config validation ------------------------------------------------------


def test_max_tokens_config_validated() -> None:
    """``max_tokens`` must be >= 1."""
    with pytest.raises(ProviderError, match="max_tokens"):
        _provider(max_tokens=0)


def test_dpi_config_validated() -> None:
    """``dpi`` must be >= 1 (shared with parser.lighton)."""
    with pytest.raises(ProviderError, match="dpi"):
        _provider(dpi=0)


def test_max_longest_dim_config_validated() -> None:
    """``max_longest_dim`` must be >= 0 (0 disables)."""
    with pytest.raises(ProviderError, match="max_longest_dim"):
        _provider(max_longest_dim=-1)


# --- multi-document run: one bad doc never aborts the run --------------------


def test_batches_multiple_documents_in_input_order(
    fake_vllm: None, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """Multiple documents parse in input order; a spoofed one is a PARSE_ERROR, others PARSED."""
    inputs = [
        _pdf_input(make_pdf([f"document number {i}"]), file_name=f"d{i}.pdf") for i in range(3)
    ]
    inputs.insert(1, _pdf_input(b"not a pdf", file_name="bad.pdf"))  # a spoof in the middle
    records = list(_provider().parse(inputs))
    assert [r.provenance.file_name for r in records] == ["d0.pdf", "bad.pdf", "d1.pdf", "d2.pdf"]
    assert records[1].disposition is Disposition.PARSE_ERROR  # the spoof, degraded
    assert all(records[i].disposition is Disposition.PARSED for i in (0, 2, 3))


# --- the fault guard maps native rasteriser faults to typed, content-free errors ---


def test_native_rasteriser_fault_maps_to_typed_error_no_leak(
    fake_vllm: None, monkeypatch: pytest.MonkeyPatch, make_pdf: Callable[[list[str]], bytes]
) -> None:
    """A native rasteriser fault → typed ProviderError → PARSE_ERROR; NO content/secret leak (C2/C6)."""
    import pypdfium2  # the injected fake

    class _NativeBoom(Exception):
        def __str__(self) -> str:
            return "boom sk-live-OCR-LEAK alice@example.com"

    def _explode(self: object, *_a: object, **_k: object) -> object:
        raise _NativeBoom

    monkeypatch.setattr(pypdfium2.PdfDocument, "__iter__", _explode)
    (record,) = list(_provider().parse([_pdf_input(make_pdf(["secret page"]))]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert "sk-live-OCR-LEAK" not in record.error
    assert "alice@example.com" not in record.error
    assert "_NativeBoom" in record.error  # only the native TYPE name
    assert "rasterise" in record.error  # the op label
