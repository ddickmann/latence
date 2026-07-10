"""``parser.lighton`` — a real GPU OCR Parser over LightOn-OCR (W1-lighton, #91, Apache-2.0).

The first **GPU-class, image→markdown** Parser Provider (CONTEXT ``Parser``), built THROUGH the
:class:`~latence_core.providers.adapter.AdapterBase` template (P3-F2/ADR-0036 §1) exactly like
``parser.pdfplumber`` (H-A1). It unlocks what the CPU text-layer parsers cannot: a **scanned /
image-only** page (no text layer). ``parser.pdfplumber`` / ``parser.document`` extract an existing
text layer; :class:`LightOnOcrParser` **reads pixels** with a compact 1B vision-language model
(``lightonai/LightOnOCR-1B-1025``) that emits **markdown text per page**, and takes that markdown
directly onto the exact-offset :func:`~latence_core.contracts.PageMap.from_page_texts` seam
(ADR-0031) — the same page/offset path the pdfplumber Parser uses, never a reinvented weaker mapper.

Why LightOn-OCR (the 2026-07 pivot, see the repo pivot commit): it is a torch/transformers VLM
(**Apache-2.0** weights AND code — license-verified below), so it runs on the transformers stack
already on the validation pod, on a single 16 GB GPU, with **no** paddle stack (contrast the dropped
PaddleOCR-VL) and no GPL/NC-licensed weights (contrast the dropped Surya). It emits page markdown
end to end, so there is **no** bbox/line model to port — the markdown-per-page → PageMap path is the
whole integration.

Real model API (verified 2026-07 from the HF model card, https://huggingface.co/lightonai/LightOnOCR-1B-1025):
    ``LightOnOcrForConditionalGeneration`` + ``LightOnOcrProcessor`` (transformers). Per page:
    build a chat message with the page image, ``processor.apply_chat_template(...,
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")``, run
    ``output_ids = model.generate(...)``, **slice the prompt tokens off** the generated sequence
    (``generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]``), then decode just the
    continuation (``processor.decode(generated_ids, skip_special_tokens=True)`` → the page's
    markdown ``str``). The slice is **load-bearing**: HF ``generate`` returns the full prompt +
    continuation sequence, so decoding it WITHOUT slicing off the prompt tokens prepends the
    chat-template / user turn / image-placeholder scaffolding to the OCR markdown on every page —
    which would corrupt content, PageMap offsets, size, and language. Recommended decoding:
    ``max_new_tokens=4096, temperature=0.2, top_p=0.9`` (the card's sampling default). This Provider
    decodes **greedily** (``do_sample=False``) by default so the output is as reproducible as the
    model allows on fixed input — but VLM float decoding is not promised byte-identical across
    hardware/library versions, so the profile declares ``deterministic=False`` honestly (verify on
    the pod; do not overclaim).

**The anti-false-green rule (baked in from the PaddleOCR-VL false green).** The offline test double
MUST be **faithful to the real generate/decode contract**: its ``generate`` returns a full
prompt-tokens + continuation sequence (a 2-D tensor-like), so the Provider's prompt-token slice
(``output_ids[0, inputs["input_ids"].shape[1]:]``) is genuinely exercised — an un-sliced decode
would visibly include the prompt scaffolding and the offset round-trip would break. ``decode``
yields the page's markdown ``str`` from the sliced continuation only — see :mod:`tests`'
``_FakeProcessor`` / ``_FakeModel``. A conformance/e2e reviewer can point at the fake and see it
matches the model card's ``generate`` → slice → ``decode`` contract, prompt tokens included. The
single thing whose divergence made PaddleOCR-VL a false green was a fabricated output shape; here
the fake mirrors the real sequence shape *including the prompt tokens the Provider must slice*.

**Pixel-bomb DoS cap (the Hardening-bar item — this is the first framework Parser that rasterises
untrusted input).** The sibling CPU parsers only extract an existing text layer; this one **reads
pixels**, so a maliciously oversized page or a decompression-bomb image could drive a multi-GB
allocation — and the intake ``max_bytes`` cap does **not** bound it (a tiny PDF can declare huge
page dimensions; an image's declared pixel count is independent of its byte size). Before any
``.to_pil()`` / ``.load()`` the Provider bounds the raster:

* it clamps the PDF render scale by a **max-megapixel budget** (:data:`_MAX_RENDER_MEGAPIXELS`):
  it probes each page's point size via ``pypdfium2`` (:meth:`_clamp_scale`) and reduces the
  DPI-derived scale so the rasterised bitmap stays under the budget — a 10 000×10 000-pt page can
  never rasterise to a gigapixel bitmap;
* it sets a conservative ``Image.MAX_IMAGE_PIXELS`` (restored afterwards, no global side effect) so
  Pillow **raises** ``DecompressionBombError`` on an over-budget image rather than allocating it —
  that native fault maps through :meth:`guard` to a typed ProviderError and degrades to a graceful
  ``PARSE_ERROR`` record, never a crash or a multi-GB allocation.

A regression fixture asserts an oversized page/image degrades to a PARSE_ERROR (never a huge alloc).

What it EXERCISES on ``AdapterBase`` (H-A1 — use the seam, do not bypass):

* **device gating** (:meth:`AdapterBase.device`): ``compute="gpu"`` — on a CPU-only host the device
  seam returns the **skip-with-flag** decision (the G1 posture), so the Runner/conformance skip it
  rather than fabricating a CPU run; on the pod (``LATENCE_CUDA=1`` / real CUDA) it routes to
  ``cuda`` and the model is placed there. Device *placement* stays the adapter's job (ADR-0003): the
  resolved ``self.device.device`` flows into ``model.to(...)``.
* **native fault → typed taxonomy** (:meth:`AdapterBase.guard`): a model/CUDA/transformers failure
  becomes a typed :class:`~latence_core.errors.ProviderError` carrying only the provider + op +
  native *type name* — **never image bytes or OCR text** (C2/C6: OCR output is untrusted,
  PII-bearing). That typed error is caught per-document and degraded to a graceful ``PARSE_ERROR``
  :class:`~latence_core.contracts.DocumentRecord` (ADR-0020), so one bad page never aborts the run.
* **batching + streaming** (:meth:`AdapterBase.batched`): batches over input *documents*
  (input-order-preserving, one batch at a time — ADR-0033), and within a document runs the VLM over
  the page images.
* **Provenance + exact offsets** (:meth:`AdapterBase.carry_provenance` +
  :func:`~latence_core.contracts.PageMap.from_page_texts`): the per-page markdown is assembled with
  the core PageMap seam (boundaries exact by construction), and Provenance is carried from the
  ``ParserInput`` via the base helper — offsets round-trip, pages correct.

Heavy deps (``transformers``, ``torch``, ``pillow``, ``pypdfium2`` for PDF→image rasterisation)
live ONLY in this package (ADR-0016); ``latence-core`` never imports them. The import is deferred to
first use so Provider discovery stays torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
)
from latence_core.errors import LatenceError, ProviderError
from latence_core.language import detect_language
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.perf import load_with_fa2_fallback
from latence_core.providers.profile import ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ParserInput

# The license-verified checkpoint (Apache-2.0 weights AND code — see :meth:`profile`).
_DEFAULT_MODEL = "lightonai/LightOnOCR-1B-1025"

# The per-page decode ceiling — the CONTENT-SAFE default (W8, pod-corrected). 4096 is the
# natural-EOS budget: on dense real German banking pages max_new_tokens=1024 TRUNCATED ~25% of the
# OCR text (19905→14951 chars — the pages genuinely emit >1024 tokens), silently dropping document
# content, which is worse than slow for enterprise. LightOn's 1024 example is fine for typical pages
# but truncates dense multi-column documents — tune it DOWN for known-light corpora (speed), never
# below your pages' real token length. Fully configurable; bounded so a pathological page cannot
# generate unboundedly.
_DEFAULT_MAX_NEW_TOKENS = 4096

# The target LONGEST image dimension (px) a rasterised page / decoded image is downscaled to before
# OCR, aspect ratio preserved (W8). This is LightOn's OFFICIAL preprocessing recipe: "Render PDFs at
# 200 DPI to a target longest dimension of 1540px". An A4 at 200 DPI is ~2339px tall — fed at full
# size that is ~2.3× the vision tokens (compute) vs LightOn's target, the single biggest throughput
# lever. NOT a byte-identical no-op (OCR output changes with resolution), so it is a config knob
# defaulting to LightOn's 1540; ``0`` (or ``None``) disables the cap.
_DEFAULT_MAX_LONGEST_DIM = 1540

# How many page images share one ``generate()`` call (W8 OCR page-batching). Today's one-page-per-
# generate is the throughput bottleneck; batching N pages left-pads them into one forward pass. A
# small default (memory-shaping only — batched OCR is record-for-record identical to per-page under
# greedy decoding + correct left-padding + per-item prompt-slice). Overridable via ``batch_size``.
_DEFAULT_OCR_BATCH_SIZE = 4

# --- Pixel-bomb DoS cap (the Hardening-bar item; this Parser rasterises untrusted input) ------
# The maximum megapixels a rasterised page / decoded image may occupy. A PDF page's render scale is
# clamped so its bitmap stays under this budget (a giant page cannot force a gigapixel bitmap), and
# Pillow's ``Image.MAX_IMAGE_PIXELS`` is set to this budget so a decompression-bomb image RAISES
# rather than allocating multi-GB. ~25 MP is generous for a legible document scan (≈ an A4 page at
# 600 DPI) while bounding a single page's raster to well under a gigabyte. The intake max_bytes cap
# does NOT bound this — page dimensions / declared pixel counts are independent of file byte size.
_MAX_RENDER_MEGAPIXELS = 25.0
_MAX_RENDER_PIXELS = int(_MAX_RENDER_MEGAPIXELS * 1_000_000)

# File types this Provider rasterises to page images then OCRs. A PDF is rendered page-by-page
# (pypdfium2) so the page map is real (one OCR pass per page); a bare image is a single page.
_PDF_EXTENSIONS = frozenset({"pdf"})
_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"})

# %PDF- magic at offset 0 (a bounded leading BOM/whitespace window only) — the offset-0 type-spoof
# guard mirroring ``parser.pdfplumber`` / the intake Screener (#62): a ``.pdf`` whose bytes are not
# a PDF is rejected as a parse failure before the rasteriser, defense-in-depth.
_PDF_MAGIC = b"%PDF-"
_PDF_LEADING_SLACK = 8
_PDF_LEADING_SKIP = b"\x00\r\n\t \xef\xbb\xbf\xff\xfe"

# Magic-byte prefixes for the image formats this Provider accepts — the offset-0 spoof guard for a
# declared image whose bytes are not that image (defense-in-depth, mirroring the PDF guard). A
# format not in this table is not spoof-checked (accepted and handed to Pillow, which fails
# gracefully through the guard) — the table covers the common raster formats.
_IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"RIFF",  # WEBP (RIFF container; "WEBP" follows at offset 8)
    b"II*\x00",  # TIFF little-endian
    b"MM\x00*",  # TIFF big-endian
    b"BM",  # BMP
)


class LightOnOcrParser(AdapterBase):
    """GPU OCR Parser Provider over LightOn-OCR: page image(s) → markdown records + page map.

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass (H-A1) satisfying the
    :class:`~latence_core.capability.Parser` Protocol structurally (it implements ``parse``). It
    declares a ``compute="gpu"`` :class:`~latence_core.providers.profile.ProviderProfile`, resolves
    its device through the base (skip-with-flag on a CPU-only host), batches over input docs via
    :meth:`AdapterBase.batched`, maps the native transformers/CUDA fault through
    :meth:`AdapterBase.guard`, and carries Provenance via :meth:`AdapterBase.carry_provenance` onto
    the exact-offset :func:`~latence_core.contracts.PageMap.from_page_texts` seam.

    Config:
        model: str — the LightOn-OCR checkpoint id (default ``lightonai/LightOnOCR-1B-1025``,
            Apache-2.0). A restricted-license checkpoint would be a deliberate opt-in, not default.
        category: str | None — content category stamped onto Classification (else inherited).
        sensitivity: str | None — sensitivity stamped onto Classification (else inherited).
        page_separator: str — markdown joined between pages (default ``"\\n\\n"``).
        max_new_tokens: int — the per-page decode ceiling (default 4096, content-safe on dense
            pages; LightOn's 1024 example truncates dense multi-column docs — tune to your corpus).
        ocr_batch_size: int — how many page images share one ``generate()`` call (default 4; W8
            page-batching, memory-shaping — batched OCR is record-for-record identical to per-page).
        max_longest_dim: int — the target longest image side in px, aspect preserved (default 1540,
           LightOn's official recipe; ``0`` disables). Downscale only, LANCZOS. NOT byte-identical —
            it trades resolution for throughput (the single biggest lever).
        dpi: int — the PDF→image rasterisation DPI (default 200; higher = sharper, slower).
        batch_size: int — how many input documents the batching path groups per call (default from
            AdapterBase). Purely memory-shaping — outputs are input-order identical (ADR-0033).
        device: str — ``auto|cpu|cuda`` (resolved by the base; a GPU-only provider skips-with-flag
            on a CPU-only host).

    ``config_exceptions``: a native ``ValueError``/``KeyError``/``OSError`` from transformers means
    an operator handed a bad config (a bad ``model`` id, a missing weight file) → a
    :class:`~latence_core.errors.ConfigError` via :meth:`guard`; everything else (a CUDA/runtime
    OCR fault) is a runtime :class:`~latence_core.errors.ProviderError`.
    """

    #: A native bad-config exception from transformers → ConfigError; else → ProviderError (guard).
    config_exceptions: ClassVar[tuple[type[Exception], ...]] = (ValueError, KeyError, OSError)

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): GPU, model-bearing, evidence-bearing Apache-2.0.

        ``compute="gpu"``: LightOn-OCR is a 1B vision-language model — it needs a CUDA GPU to be
        practical, so the device seam **skips it with-flag** on a CPU-only host (the G1 posture,
        never a fabricated CPU run) and routes it to ``cuda`` on a rig. (CPU inference is
        technically possible but far too slow to run as a real Parser, so ``either`` is not
        declared — the honest compute requirement is GPU.)

        License is **verified** per ADR-0012, recorded evidence-bearingly (H-B1 — the claim is
        inseparable from its proof): weights and code are recorded **separately**
        (``license_code="Apache-2.0"`` for the transformers loader path + the LightOn-OCR code;
        ``license_weights="Apache-2.0"`` for the ``lightonai/LightOnOCR-1B-1025`` checkpoint), with
        a ``license_source`` citing the HF model card where a human checked, and
        ``license_verified_on="2026-07-08"``. The headline ``license`` is the **weights** SPDX
        (``"Apache-2.0"``) — the effective license of what the markdown embodies.

        ``deterministic=False``: a VLM's float decoding is not guaranteed byte-identical across
        hardware / library / BLAS versions (unlike a pure-code text parser), so the profile declares
        it non-deterministic rather than overclaim reproducibility — even with greedy decoding, the
        cross-environment byte-identity C4 asserts is not promised (verify on the pod; declared
        False, not assumed). ``batch=True`` (it batches over documents). ``memory_mb`` is the
        approximate resident VRAM footprint of the 1B checkpoint in bf16 + the transformers stack
        (~3.5 GB, declared not measured — P3-F1 is descriptor-only). ``cost_per_1k=None``
        (in-process, no per-call vendor cost).
        """
        return ProviderProfile(
            compute="gpu",
            memory_mb=3584,
            model_id=_DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            # Where a human actually checked BOTH weights and code (ADR-0012 / research-diligence):
            license_source=(
                "https://huggingface.co/lightonai/LightOnOCR-1B-1025 "
                "(model card: weights + code both Apache-2.0, verified 2026-07-08)"
            ),
            license_verified_on="2026-07-08",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            # W8: honest declared load-time perf posture (descriptor-only, P3-F1; not measured):
            # bf16 on GPU / fp32 on CPU + sdpa (FA2 when flash_attn present), OCR page-batched.
            perf_note="bf16-GPU/fp32-CPU, sdpa (FA2 when flash_attn present), OCR page-batched",
        )

    @property
    def provider_name(self) -> str:
        """The registered entry-point name — the stable tag a mapped error carries (no content)."""
        return "parser.lighton"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Resolve the device against the declared GPU profile via the base (skip-with-flag on a
        # CPU-only host; cuda on a rig). Placement flows through the seam (H-A1).
        super().__init__(config)
        cfg = self.config
        self._model_id = str(cfg.get("model", _DEFAULT_MODEL))
        self._category = cfg.get("category")
        self._sensitivity = cfg.get("sensitivity")
        self._page_separator = str(cfg.get("page_separator", "\n\n"))
        self._max_new_tokens = int(cfg.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))
        if self._max_new_tokens < 1:
            msg = f"{self.provider_name}: max_new_tokens must be >= 1, got {self._max_new_tokens}."
            raise ProviderError(msg)
        # W8 OCR page-batching: how many page images share one generate() call (>=1). Distinct from
        # the document-level ``batch_size`` (AdapterBase, over input docs) — this batches PAGES
        # a document. Default 4; a positive int (a batch of zero would never make progress).
        self._ocr_batch_size = int(cfg.get("ocr_batch_size", _DEFAULT_OCR_BATCH_SIZE))
        if self._ocr_batch_size < 1:
            msg = f"{self.provider_name}: ocr_batch_size must be >= 1, got {self._ocr_batch_size}."
            raise ProviderError(msg)
        self._dpi = int(cfg.get("dpi", 200))
        if self._dpi < 1:
            msg = f"{self.provider_name}: dpi must be >= 1, got {self._dpi}."
            raise ProviderError(msg)
        # W8: LightOn's official longest-side downscale target (px), aspect preserved. Default 1540
        # (LightOn's recipe); 0 or None disables. A negative value is an operator config error.
        raw_longest = cfg.get("max_longest_dim", _DEFAULT_MAX_LONGEST_DIM)
        self._max_longest_dim = int(raw_longest) if raw_longest else 0
        if self._max_longest_dim < 0:
            msg = (
                f"{self.provider_name}: max_longest_dim must be >= 0 (0 disables), "
                f"got {self._max_longest_dim}."
            )
            raise ProviderError(msg)
        # Lazily loaded on first parse (cached); constructing the Provider never imports torch.
        self._model: Any = None
        self._processor: Any = None

    # -- Parser Capability method (structural; the base adds no protocol) ---

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        """Consume raw :class:`ParserInput`s and yield parsed markdown records, in input order.

        Runs the base's batching path (:meth:`AdapterBase.batched`) over input documents so the
        seam's streaming-friendly, input-order-preserving batching is exercised; each document is
        OCR'd one at a time. Ordering is preserved by construction (ADR-0033).
        """
        yield from self.batched(list(inputs), self._parse_batch)

    def _parse_batch(self, batch: list[ParserInput]) -> list[DocumentRecord]:
        """OCR one batch of raw inputs → one :class:`DocumentRecord` each, in order."""
        return [self._parse_one(raw) for raw in batch]

    # -- per-document parse -------------------------------------------------

    def _parse_one(self, raw: ParserInput) -> DocumentRecord:
        """OCR a single raw input into a PARSED (or, on failure, PARSE_ERROR) record.

        The native transformers/CUDA fault a model run raises is mapped by :meth:`AdapterBase.guard`
        to a typed :class:`~latence_core.errors.ProviderError` (no image/OCR-text/PII), then caught
        here and degraded to a graceful ``PARSE_ERROR`` record (ADR-0020) — so one bad document
        never aborts the run. A framework :class:`~latence_core.errors.LatenceError` (the guarded
        fault, or an explicit parse failure) is the only thing caught; a genuinely unexpected bug
        still propagates untyped.
        """
        file_type = raw.provenance.file_type.lower()
        try:
            images = self._rasterise(raw, file_type)
            page_texts = self._ocr_pages(images)
        except LatenceError as exc:
            # Both the guarded native fault (ProviderError) and an explicit _ParseFailure land here
            # → a graceful PARSE_ERROR record. The message is the framework-typed one (provider +
            # op + native type name), carrying no image bytes / OCR text / PII.
            return self._error_record(raw, str(exc))

        assembly = PageMap.from_page_texts(page_texts, separator=self._page_separator)
        text = assembly.text
        n_pages = len(assembly.page_map.pages)

        # Carry Provenance via the base helper (source lineage intact; only the char/page span
        # narrowed) — never a hand-rolled mapper (H-A1). A ParserInput is a ProvenanceCarrier.
        provenance, _ = self.carry_provenance(
            raw,
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=n_pages,
        )
        classification = self._classify(raw, text)
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=provenance,
            classification=classification,
            content=text,
            media_type="text/markdown",
            page_map=assembly.page_map,
            disposition=Disposition.PARSED,
        )

    def _error_record(self, raw: ParserInput, reason: str) -> DocumentRecord:
        """A ``PARSE_ERROR`` record: Provenance kept, no content, typed reason (ADR-0020)."""
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=raw.provenance,
            classification=self._classify(raw, ""),
            content="",
            media_type="text/markdown",
            page_map=None,
            disposition=Disposition.PARSE_ERROR,
            error=reason,
        )

    # -- rasterise raw bytes → page images (spoof-guarded, pixel-capped) ----

    def _rasterise(self, raw: ParserInput, file_type: str) -> list[Any]:
        """Turn the raw source bytes into one Pillow image per page (spoof-guarded, pixel-capped).

        A PDF is rendered page-by-page via ``pypdfium2`` (the per-page split is what makes the page
        map real — one OCR pass per page); a bare image is a single page. An unsupported type or a
        type-spoofed file (bytes not matching the declared type) is a graceful parse failure
        (``_ParseFailure`` → PARSE_ERROR), not a crash — mirroring ``parser.pdfplumber`` / #62. The
        rasteriser clamps the render scale / decode by the max-megapixel budget so an oversized page
        or a decompression-bomb image degrades gracefully rather than allocating multi-GB.
        """
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        if file_type in _PDF_EXTENSIONS:
            self._guard_pdf_signature(data)
            return self._pdf_to_images(data)
        if file_type in _IMAGE_EXTENSIONS:
            self._guard_image_signature(data)
            return [self._image_from_bytes(data)]
        msg = f"Unsupported file type {file_type!r} for the LightOn-OCR Parser."
        raise _ParseFailure(msg)

    def _guard_pdf_signature(self, data: bytes) -> None:
        """Reject a ``.pdf`` whose bytes are not a PDF (offset-0 spoof guard, like pdfplumber)."""
        offset = 0
        limit = min(len(data), _PDF_LEADING_SLACK)
        while offset < limit and data[offset] in _PDF_LEADING_SKIP:
            offset += 1
        if data[offset : offset + len(_PDF_MAGIC)] != _PDF_MAGIC:
            msg = (
                "File declared as PDF but the %PDF- signature is not at the start of the file "
                "(type spoof?)."
            )
            raise _ParseFailure(msg)

    def _guard_image_signature(self, data: bytes) -> None:
        """Reject a declared image whose bytes match no known image magic (offset-0 spoof guard)."""
        if not any(data.startswith(magic) for magic in _IMAGE_MAGIC):
            msg = (
                "File declared as an image but its bytes match no known image signature "
                "(type spoof?)."
            )
            raise _ParseFailure(msg)

    def _clamp_scale(self, width_pt: float, height_pt: float, scale: float) -> float:
        """Reduce a PDF render scale so the rasterised bitmap stays under the megapixel budget.

        pypdfium2 renders a page of ``width_pt × height_pt`` points at ``scale`` px/pt, so the
        bitmap is ``(width_pt*scale) × (height_pt*scale)`` pixels. If that exceeds
        :data:`_MAX_RENDER_PIXELS` (the pixel-bomb budget) the scale is reduced by ``sqrt(budget /
        pixels)`` so the area lands exactly at the budget — a 10 000×10 000-pt page can never
        rasterise to a gigapixel bitmap (the Hardening-bar finding). A degenerate (zero-area) page
        keeps the requested scale (nothing to clamp).
        """
        area_pt = max(width_pt, 0.0) * max(height_pt, 0.0)
        if area_pt <= 0.0 or scale <= 0.0:
            return scale
        pixels = area_pt * scale * scale
        if pixels <= _MAX_RENDER_PIXELS:
            return scale
        return float(scale * (_MAX_RENDER_PIXELS / pixels) ** 0.5)

    def _pdf_to_images(self, data: bytes) -> list[Any]:
        """Rasterise every PDF page to a Pillow image via ``pypdfium2`` (ordered, pixel-capped).

        Imported inside the method so a missing/malformed dependency never breaks Provider
        discovery, and so ``latence-core`` stays free of the heavy dep (ADR-0016). A missing dep is
        a graceful parse failure. Each page's render scale is clamped by :meth:`_clamp_scale` so an
        oversized page cannot force a gigapixel bitmap (the pixel-bomb cap). The native pypdfium2
        fault a corrupt PDF raises is mapped by :meth:`AdapterBase.guard` to a typed ProviderError
        (no content), then degraded to a PARSE_ERROR record by :meth:`_parse_one`.
        """
        try:
            import io

            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pypdfium2 is required to rasterise PDFs for OCR."
            raise _ParseFailure(msg) from exc

        requested_scale = self._dpi / 72.0  # pypdfium2 renders at 72 DPI * scale
        images: list[Any] = []
        with self.guard("rasterise"):
            pdf = pdfium.PdfDocument(io.BytesIO(data))
            try:
                for page in pdf:
                    width_pt, height_pt = page.get_size()
                    scale = self._clamp_scale(width_pt, height_pt, requested_scale)
                    bitmap = page.render(scale=scale)
                    images.append(self._downscale_longest(bitmap.to_pil()))
            finally:
                pdf.close()
        return images or []

    def _image_from_bytes(self, data: bytes) -> Any:
        """Decode a single raster image from bytes via Pillow (dep-isolated; guarded; bomb-capped).

        ``Image.MAX_IMAGE_PIXELS`` is tightened to the megapixel budget so a decompression-bomb
        image (declared pixel count above the budget) makes Pillow **raise**
        ``DecompressionBombError`` during ``.load()`` rather than allocating multi-GB — that native
        fault maps through :meth:`guard` to a typed ProviderError and degrades to a PARSE_ERROR
        record. The prior value is restored in a ``finally`` so we never leave global Pillow state
        mutated for other users of the process.
        """
        try:
            import io

            from PIL import Image
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pillow is required to decode images for OCR."
            raise _ParseFailure(msg) from exc
        previous_cap = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = _MAX_RENDER_PIXELS
        try:
            with self.guard("decode_image"):
                image = Image.open(io.BytesIO(data))
                image.load()  # a decompression-bomb image raises here (bounded, no huge alloc)
                return self._downscale_longest(image.convert("RGB"))
        finally:
            Image.MAX_IMAGE_PIXELS = previous_cap

    def _downscale_longest(self, image: Any) -> Any:
        """Downscale a page image so its LONGEST side == ``max_longest_dim`` px (aspect preserved).

        LightOn's OFFICIAL preprocessing (W8): "Render PDFs at 200 DPI to a target longest dimension
        of 1540px, maintaining aspect ratio." Applied to BOTH rasterised PDF pages and bare-image
        inputs, AFTER the megapixel bomb-clamp (so a bomb still degrades gracefully first). When the
        cap is disabled (``max_longest_dim == 0``) or the image is already within it, the image is
        returned untouched — no upscaling, no resample cost. A downscale uses LANCZOS (Pillow's
        high-quality resampling) so text stays legible. This is NOT byte-identical to the un-capped
        path (OCR output changes with resolution) — it is the throughput knob, defaulting to
        LightOn's 1540.
        """
        cap = self._max_longest_dim
        if cap <= 0:
            return image
        width, height = image.size
        longest = max(width, height)
        if longest <= cap:
            return image
        from PIL import (
            Image,  # noqa: PLC0415 - deferred heavy dep (ADR-0016); already imported above
        )

        ratio = cap / float(longest)
        new_size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    # -- LightOn-OCR inference: page image(s) → markdown per page -----------

    def _ocr_pages(self, images: list[Any]) -> list[str]:
        """Run LightOn-OCR over the page images → one markdown string per page (input order).

        The whole VLM integration: load the model + processor lazily (cached), and for each page
        image build a chat message, ``apply_chat_template``, ``generate``, **slice the prompt tokens
        off the generated sequence**, then ``decode`` — exactly the real model-card API
        (https://huggingface.co/lightonai/LightOnOCR-1B-1025):

            output_ids = model.generate(**inputs, max_new_tokens=...)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]   # SLICE off the prompt
            output_text = processor.decode(generated_ids, skip_special_tokens=True)

        The slice is load-bearing: HF ``generate`` returns the **full** sequence (prompt +
        continuation), so decoding it WITHOUT slicing off the ``inputs["input_ids"].shape[1]``
        prompt tokens would prepend the chat-template / user turn / image-placeholder scaffolding to
        the OCR markdown on every page — corrupting content, PageMap offsets, size, and language.
        The native call is under :meth:`guard` so a transformers/CUDA fault maps to a typed
        ProviderError carrying no image/OCR text.
        """
        if not images:
            # A zero-page source (e.g. an empty PDF) yields a single empty page — recorded, never
            # crashed (mirrors the pdfplumber empty-page discipline).
            return [""]
        model, processor = self._load_model()
        page_texts: list[str] = []
        with self.guard("parse"):
            for start in range(0, len(images), self._ocr_batch_size):
                chunk = images[start : start + self._ocr_batch_size]
                page_texts.extend(self._ocr_one_batch(model, processor, chunk))
        return page_texts

    def _ocr_one_batch(self, model: Any, processor: Any, chunk: list[Any]) -> list[str]:
        """OCR one batch of page images in a SINGLE ``generate()`` call → markdown per page (W8).

        The whole W8 page-batching contract, built to be **record-for-record identical to the
        per-page path** under greedy decoding:

        * one chat message per page image (:meth:`_build_messages`) → a *list* of message-lists;
        * ``apply_chat_template(..., padding=True)`` LEFT-pads them into one batch with a shared
          ``input_ids`` width ``L`` and a per-item ``attention_mask`` (1 on the item's real tokens,
          0 on the left pad) — left-padding is mandatory for a decoder-only ``generate`` so every
          item's continuation begins at the SAME column ``L``;
        * ONE ``model.generate(**inputs, do_sample=False)`` (greedy) over the whole batch;
        * per item, the continuation is ``output_ids[i, L:]`` — sliced by the shared padded prompt
          width ``L`` (``input_ids.shape[1]``), which is exactly where each item's generation starts
          under left-padding. The per-item real prompt length (``attention_mask[i].sum()``) is
          honoured by the mask so batching a variable-length page next to a short one does NOT bleed
          one page's prompt/pad tokens into another's markdown (the variable-length equivalence the
          test asserts). Decoding without the ``L`` slice would prepend the chat/prompt scaffolding
          to every page (the false-green trap the single-item path already guards).

        Greedy (``do_sample=False``) + correct left-padding/attention-mask ⇒ the batched markdown is
       byte-identical to running each page alone — validated offline by the faithful fake VLM and on
        the pod against the real model.
        """
        messages_batch = [self._build_messages(image) for image in chunk]
        inputs = processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,  # LEFT-pad the batch (decoder-only generate needs aligned continuations)
        )
        device = self.device.device
        if device is not None:
            inputs = inputs.to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        # generate() returns the FULL sequence (prompt + continuation) for each item. Under
        # left-padding every item's continuation starts at the shared padded prompt width L =
        # input_ids.shape[1]; slice each item there so the prompt/pad tokens are stripped and only
        # the page's markdown continuation is decoded. Per-item, not a single shared decode.
        prompt_width = inputs["input_ids"].shape[1]
        texts: list[str] = []
        for i in range(len(chunk)):
            generated_ids = output_ids[i, prompt_width:]
            texts.append(processor.decode(generated_ids, skip_special_tokens=True))
        return texts

    @staticmethod
    def _build_messages(image: Any) -> list[dict[str, Any]]:
        """The single-image OCR chat message LightOn-OCR expects (image content, no user text)."""
        return [{"role": "user", "content": [{"type": "image", "image": image}]}]

    def _load_model(self) -> tuple[Any, Any]:
        """Import transformers + load the LightOn-OCR model/processor on first use (cached).

        Deferred so Provider discovery never needs torch, and a missing optional dependency
        surfaces as a clear install hint only when the Provider is actually run. The device the
        seam resolved (``self.device.device``) is where the model is placed — placement stays the
        adapter's job (ADR-0003). The native transformers load fault (a bad ``model`` id → a
        ConfigError via :attr:`config_exceptions`; anything else → ProviderError) is mapped by the
        guard, no content in the message.
        """
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            from transformers import (
                LightOnOcrForConditionalGeneration,
                LightOnOcrProcessor,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'transformers' package (with LightOn-OCR support) is required to run "
                "LightOnOcrParser. Install it with: pip install latence-parser-lighton"
            )
            raise ImportError(msg) from exc
        device = self.device.device
        # W8 perf seam: bf16-on-GPU + sdpa/FA2 (graceful FA2→sdpa) into the VLM load, instead of the
        # fp32/eager default (the ~2× miss). The processor takes no perf kwargs. After placement the
        # model is optionally compiled (guarded, off by default).
        perf = self.load_perf_kwargs()
        with self.guard("load_model"):
            model = load_with_fa2_fallback(
                LightOnOcrForConditionalGeneration.from_pretrained, self._model_id, perf
            )
            if device is not None:
                model = model.to(device)
            processor = LightOnOcrProcessor.from_pretrained(self._model_id)
        model = self.maybe_compile(model)
        self._model = model
        self._processor = processor
        return model, processor

    # -- classification -----------------------------------------------------

    def _classify(self, raw: ParserInput, text: str) -> Classification:
        """Attach Classification at Parse: language from the OCR'd text (shared core heuristic).

        Language is auto-detected from the OCR'd markdown via the shared core heuristic (Parse sees
        the real content). Category and sensitivity are the config override, else the value
        inherited from the ParserInput's Classification, so the Source's stamp is not dropped —
        matching ``parser.pdfplumber`` exactly.
        """
        inherited = raw.classification
        language = detect_language(text) if text else inherited.language
        category = self._category if self._category is not None else inherited.category
        sensitivity = (
            str(self._sensitivity) if self._sensitivity is not None else inherited.sensitivity
        )
        return Classification(language=language, category=category, sensitivity=sensitivity)


def size_after_parse(record: DocumentRecord) -> int:
    """Assembled-markdown size in bytes — the Provenance 'size-after-parse' (S2 observability).

    The per-record view the Runner sums across parsed records into the Quality Report's
    ``ParseQuality.bytes_after_parse``. Mirrors the in-core / pdfplumber Parser helper so a Pipeline
    that swaps this Provider in gets the same observability signal.
    """
    return len(record.content.encode("utf-8"))


class _ParseFailure(LatenceError):
    """A recoverable, framework-typed parse failure → a PARSE_ERROR record (ADR-0020).

    A :class:`~latence_core.errors.LatenceError` subclass so it flows through the same typed-error
    boundary the guarded native fault does: :meth:`LightOnOcrParser._parse_one` catches
    ``LatenceError`` and turns either into a graceful PARSE_ERROR record. It carries only a static
    reason (unsupported type, spoofed signature, missing dep) — never input content / OCR text.
    """

    category = "provider"
