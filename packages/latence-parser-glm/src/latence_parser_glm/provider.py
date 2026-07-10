"""``parser.glm`` — a real GPU OCR Parser over GLM-OCR (W1-glm, #91, MIT).

A **GPU-class, image→markdown** Parser Provider (CONTEXT ``Parser``) over **GLM-OCR**
(``zai-org/GLM-OCR``, Zhipu/Z.ai — **MIT** weights AND code; the layout component PP-DocLayoutV3 is
Apache-2.0 — license-verified 2026-07 from the HF model card), built THROUGH the
:class:`~latence_core.providers.adapter.AdapterBase` template (P3-F2/ADR-0036 §1) exactly like the
pod-validated ``parser.lighton`` (#125) — its **near-twin**. GLM-OCR is a compact **0.9B**
vision-language OCR model, transformers-native (``GlmOcrForConditionalGeneration``, contributed to
HF transformers), so it validates cleanly on the torch GPU pod. Like ``parser.lighton`` it unlocks
what the CPU text-layer parsers cannot: a **scanned / image-only** page (no text layer) — it
**reads pixels** and emits **markdown per page**, taken directly onto the exact-offset
:func:`~latence_core.contracts.PageMap.from_page_texts` seam (ADR-0031) — never a reinvented mapper.

Why GLM-OCR alongside LightOn-OCR (the second permissive OCR provider): it is a torch/transformers
VLM (**MIT** weights AND code — even more permissive than LightOn's Apache-2.0), a *smaller* 0.9B
checkpoint (~3–4 GB VRAM in bf16), so it runs on a single ≥16 GB GPU with no paddle stack and no
GPL/NC weights. End-to-end image→markdown, so — as with LightOn — there is **no** bbox/line model to
port: the markdown-per-page → PageMap path is the whole integration.

Real model API (verified 2026-07 from the HF model card, https://huggingface.co/zai-org/GLM-OCR):
    ``GlmOcrForConditionalGeneration`` + ``AutoProcessor`` (transformers). Per page, build a chat
    message with the page image **and a text recognition instruction**, then
    ``processor.apply_chat_template(..., add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt")``, run
    ``generated_ids = model.generate(...)``, **slice the prompt tokens off** the generated sequence
    (``generated_ids[0][inputs["input_ids"].shape[1]:]``), then decode just the continuation
    (``processor.decode(..., skip_special_tokens=True)`` → the page's markdown ``str``). The slice
    is **load-bearing** exactly as for LightOn: HF ``generate`` returns the full prompt +
    continuation sequence, so decoding it WITHOUT slicing off the prompt tokens prepends the
    chat-template / user turn / image-placeholder / instruction scaffolding to the OCR markdown on
    every page — which would corrupt content, PageMap offsets, size, and language. This Provider
    decodes **greedily** (``do_sample=False``) by default so the output is as reproducible as the
    model allows on fixed input — but VLM float decoding is not promised byte-identical across
    hardware/library versions, so the profile declares ``deterministic=False`` honestly (verify on
    the pod; do not overclaim). The card's per-document ceiling is ``max_new_tokens=8192``.

**The anti-false-green rule (carried from the PaddleOCR-VL false green + the LightOn discipline).**
The offline test double MUST be **faithful to the real generate/decode contract**: its ``generate``
returns a full prompt-tokens + continuation sequence, so the Provider's prompt-token slice
(``generated_ids[0][inputs["input_ids"].shape[1]:]``) is genuinely exercised — an un-sliced
decode would visibly include the prompt scaffolding and the offset round-trip would break. See
:mod:`tests`'
``_FakeProcessor`` / ``_FakeModel``: a reviewer points at
``test_offline_fake_is_faithful_to_generate_slice_decode_contract`` to confirm the double matches
the model card's ``generate`` → slice → ``decode`` contract, prompt tokens included. The single
thing whose divergence made PaddleOCR-VL a false green was a fabricated output shape; here the fake
mirrors the real sequence shape *including the prompt tokens the Provider must slice*.

**Pixel-bomb DoS cap (built in from the start — the LightOn security finding GLM must not defer).**
This Parser rasterises *untrusted* input to bitmaps, so a maliciously oversized page/image could
force a multi-GB allocation. Before any ``.load()`` / ``.to_pil()`` the Provider:

* clamps the PDF render scale by a **max-megapixel budget** (``_MAX_RENDER_MEGAPIXELS``): it probes
  each page's point size via ``pypdfium2`` and reduces the DPI-derived scale so the rasterised
  bitmap stays under the budget — a 10 000×10 000-pt page can never rasterise to a gigapixel bitmap;
* sets a conservative ``Image.MAX_IMAGE_PIXELS`` so Pillow **raises** on a decompression-bomb image
  (an image whose declared pixel count exceeds the budget) rather than allocating it — the native
  ``Image.DecompressionBombError`` maps through :meth:`guard`/``_ParseFailure`` to a graceful
  PARSE_ERROR, not a crash or a multi-GB allocation.

A fixture asserts an oversized page/image degrades to a PARSE_ERROR (never a crash / huge alloc).

What it EXERCISES on ``AdapterBase`` (H-A1 — use the seam, do not bypass): device gating
(``compute="gpu"`` — skip-with-flag on a CPU-only host, cuda on a rig), native fault → typed
:class:`~latence_core.errors.ProviderError` (no image/OCR text — C2/C6) degraded to a graceful
``PARSE_ERROR`` :class:`~latence_core.contracts.DocumentRecord` (ADR-0020), batching over input
documents (ADR-0033), and Provenance + exact offsets via :meth:`AdapterBase.carry_provenance` onto
:func:`~latence_core.contracts.PageMap.from_page_texts`.

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

# The license-verified checkpoint (MIT weights AND code — see :meth:`profile`).
_DEFAULT_MODEL = "zai-org/GLM-OCR"

# The per-page decode ceiling — the CONTENT-SAFE default (W8, pod-corrected). 4096 is the
# natural-EOS budget: on dense real German banking pages max_new_tokens=1024 TRUNCATED ~25% of the
# OCR text (silently dropping document content — worse than slow for enterprise). LightOn's 1024
# example is fine for typical pages but truncates dense multi-column docs — tune it DOWN for
# known-light corpora (speed), never below your pages' real token length. Fully configurable;
# bounded so a pathological page cannot generate unboundedly.
_DEFAULT_MAX_NEW_TOKENS = 4096

# The target LONGEST image dimension (px) a rasterised page / decoded image is downscaled to before
# OCR, aspect ratio preserved (W8). Mirrors the LightOn sibling's official-recipe 1540px cap:
# cap: an A4 at 200 DPI is ~2339px tall — fed full size that is ~2.3× the vision tokens vs a 1540px
# target. NOT a byte-identical no-op (OCR output changes with resolution); a config knob defaulting
# to 1540, ``0`` (or ``None``) disables the cap.
_DEFAULT_MAX_LONGEST_DIM = 1540

# How many page images share one ``generate()`` call (W8 OCR page-batching). Today's one-page-per-
# generate is the throughput bottleneck; batching N pages left-pads them into one forward pass. A
# small default (memory-shaping only — batched OCR is record-for-record identical to per-page under
# greedy decoding + correct left-padding + per-item prompt-slice). Overridable via ocr_batch_size.
_DEFAULT_OCR_BATCH_SIZE = 4

# The text-recognition instruction GLM-OCR expects alongside the page image (unlike LightOn-OCR,
# which is image-only, GLM-OCR is prompted with an OCR instruction). From the model card.
_OCR_INSTRUCTION = "Text Recognition:"

# --- Pixel-bomb DoS cap (built in from the start; the LightOn security finding) --------------
# The maximum megapixels a rasterised page/decoded image may occupy. A PDF page's render scale is
# clamped so its bitmap stays under this budget (a giant page cannot force a gigapixel bitmap), and
# Pillow's ``Image.MAX_IMAGE_PIXELS`` is set to this budget so a decompression-bomb image RAISES
# rather than allocating multi-GB. ~25 MP is generous for a legible document scan (≈ an A4 page at
# 600 DPI) while bounding a single page's raster to well under a gigabyte.
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
# declared image whose bytes are not that image (defense-in-depth, mirroring the PDF guard).
_IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"RIFF",  # WEBP (RIFF container; "WEBP" follows at offset 8)
    b"II*\x00",  # TIFF little-endian
    b"MM\x00*",  # TIFF big-endian
    b"BM",  # BMP
)


class GlmOcrParser(AdapterBase):
    """GPU OCR Parser Provider over GLM-OCR: page image(s) → markdown records + page map.

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass (H-A1) satisfying the
    :class:`~latence_core.capability.Parser` Protocol structurally (it implements ``parse``). The
    near-twin of :class:`~latence_parser_lighton.provider.LightOnOcrParser`: same AdapterBase shape,
    same image→markdown model, same generate→**slice-prompt-tokens**→decode contract, same
    PageMap-from-page-texts offset path, same type-spoof + guard + skip-with-flag + evidence-license
    discipline. What changes is the model + loader + license (MIT) — and the **pixel-bomb cap**
    built in from the start (LightOn deferred it; GLM does not).

    Config:
        model: str — the GLM-OCR checkpoint id (default ``zai-org/GLM-OCR``, MIT). A
            restricted-license checkpoint would be a deliberate opt-in, not default.
        category: str | None — content category stamped onto Classification (else inherited).
        sensitivity: str | None — sensitivity stamped onto Classification (else inherited).
        page_separator: str — markdown joined between pages (default ``"\\n\\n"``).
        max_new_tokens: int — the per-page decode ceiling (default 4096, content-safe on dense
            pages; tune down for known-light corpora, never below a page's real token length).
        ocr_batch_size: int — how many page images share one ``generate()`` call (default 4; W8
            page-batching, memory-shaping — batched OCR is record-for-record identical to per-page).
        max_longest_dim: int — the target longest image side in px, aspect preserved (default 1540;
           ``0`` disables). Downscale only, LANCZOS. NOT byte-identical — resolution for throughput.
        prompt: str — the OCR instruction paired with the page image (default
            ``"Text Recognition:"``).
        dpi: int — the PDF→image rasterisation DPI (default 200; higher = sharper, slower). The
            effective scale is clamped by the max-megapixel budget so a giant page can never force
            an oversized bitmap.
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
        """The declared descriptor (ADR-0036 §2): GPU, model-bearing, evidence-bearing MIT.

        ``compute="gpu"``: GLM-OCR is a 0.9B vision-language model — it needs a CUDA GPU to be
        practical, so the device seam **skips it with-flag** on a CPU-only host (the G1 posture,
        never a fabricated CPU run) and routes it to ``cuda`` on a rig. (CPU inference is
        technically possible but far too slow to run as a real Parser, so ``either`` is not declared
        — the honest compute requirement is GPU.)

        License is **verified** per ADR-0012, recorded evidence-bearingly (H-B1 — the claim is
        inseparable from its proof): weights and code are recorded **separately**
        (``license_code="MIT"`` for the transformers loader path + the GLM-OCR code;
        ``license_weights="MIT"`` for the ``zai-org/GLM-OCR`` checkpoint), with a ``license_source``
        citing the HF model card where a human checked (and noting that the pipeline's layout
        component PP-DocLayoutV3 is Apache-2.0 — a permissive component license, recorded honestly),
        and ``license_verified_on="2026-07-08"``. The headline ``license`` is the **weights** SPDX
        (``"MIT"``) — the effective license of what the markdown embodies. Both MIT and Apache-2.0
        are permissive (ADR-0012), so this passes conformance C3.

        ``deterministic=False``: a VLM's float decoding is not guaranteed byte-identical across
        hardware / library / BLAS versions (unlike a pure-code text parser), so the profile declares
        it non-deterministic rather than overclaim reproducibility — even with greedy decoding the
        cross-environment byte-identity C4 asserts is not promised (verify on the pod; declared
        False, not assumed). ``batch=True`` (it batches over documents). ``memory_mb`` is the
        approximate resident VRAM footprint of the 0.9B checkpoint in bf16 + the transformers stack
        (~3.5 GB, declared not measured — P3-F1 is descriptor-only). ``cost_per_1k=None``
        (in-process, no per-call vendor cost).
        """
        return ProviderProfile(
            compute="gpu",
            memory_mb=3584,
            model_id=_DEFAULT_MODEL,
            license="MIT",
            license_verified=True,
            license_code="MIT",
            license_weights="MIT",
            # Where a human actually checked BOTH weights and code (ADR-0012 / research-diligence):
            license_source=(
                "https://huggingface.co/zai-org/GLM-OCR "
                "(model card: weights + code both MIT; the pipeline's PP-DocLayoutV3 layout "
                "component is Apache-2.0 — both permissive; verified 2026-07-08)"
            ),
            license_verified_on="2026-07-08",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            # W8: honest declared load-time perf posture (descriptor-only, P3-F1; not measured).
            perf_note="bf16-GPU/fp32-CPU, sdpa (FA2 when flash_attn present), OCR page-batched",
        )

    @property
    def provider_name(self) -> str:
        """The registered entry-point name — the stable tag a mapped error carries (no content)."""
        return "parser.glm"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Resolve the device against the declared GPU profile via the base (skip-with-flag on a
        # CPU-only host; cuda on a rig). Placement flows through the seam (H-A1).
        super().__init__(config)
        cfg = self.config
        self._model_id = str(cfg.get("model", _DEFAULT_MODEL))
        self._category = cfg.get("category")
        self._sensitivity = cfg.get("sensitivity")
        self._page_separator = str(cfg.get("page_separator", "\n\n"))
        self._prompt = str(cfg.get("prompt", _OCR_INSTRUCTION))
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
        # W8: longest-side downscale target (px), aspect preserved. Default 1540; 0/None disables.
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
        fault, an oversized-input pixel-bomb rejection, or an explicit parse failure) is the only
        thing caught; a genuinely unexpected bug still propagates untyped.
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
        msg = f"Unsupported file type {file_type!r} for the GLM-OCR Parser."
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
        rasterise to a gigapixel bitmap (the LightOn security finding, built in from the start). A
        degenerate (zero-area) page keeps the requested scale (nothing to clamp).
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

        ``Image.MAX_IMAGE_PIXELS`` is set to the megapixel budget so a decompression-bomb image
        (declared pixel count above the budget) makes Pillow **raise** ``DecompressionBombError``
        during ``.load()`` rather than allocating multi-GB — that native fault maps through
        :meth:`guard` to a typed ProviderError and degrades to a PARSE_ERROR record. (The prior
        value is restored so we do not mutate global Pillow state for other users of the process.)
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

        The W8 longest-side cap (mirrors the LightOn sibling's official-recipe 1540px). Applied to
        BOTH rasterised PDF pages and bare-image inputs, AFTER the megapixel bomb-clamp. Disabled
        (``max_longest_dim == 0``) or an already-in-budget image is returned untouched — no
        upscaling, no resample cost. A downscale uses LANCZOS so text stays legible. NOT byte-
        identical to the un-capped path (OCR output changes with resolution) — it is the throughput
        knob, defaulting to 1540.
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

    # -- GLM-OCR inference: page image(s) → markdown per page ---------------

    def _ocr_pages(self, images: list[Any]) -> list[str]:
        """Run GLM-OCR over the page images → one markdown string per page (input order).

        The whole VLM integration: load the model + processor lazily (cached), and for each page
        image build a chat message (image + the OCR instruction), ``apply_chat_template``,
        ``generate``, **slice the prompt tokens off the generated sequence**, then ``decode`` —
        exactly the real model-card API (https://huggingface.co/zai-org/GLM-OCR):

            generated_ids = model.generate(**inputs, max_new_tokens=...)
            continuation = generated_ids[0][inputs["input_ids"].shape[1]:]   # SLICE off the prompt
            output_text = processor.decode(continuation, skip_special_tokens=True)

        The slice is load-bearing: HF ``generate`` returns the **full** sequence (prompt +
        continuation), so decoding it WITHOUT slicing off the ``inputs["input_ids"].shape[1]``
        prompt tokens would prepend the chat-template / user turn / image-placeholder / instruction
        scaffolding to the OCR markdown on every page — corrupting content, PageMap offsets, size,
        and language. The native call is under :meth:`guard` so a transformers/CUDA fault maps to a
        typed ProviderError carrying no image/OCR text.
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

        * one chat message (image + the OCR instruction) per page → a *list* of message-lists;
        * ``apply_chat_template(..., padding=True)`` LEFT-pads them into one batch with a shared
         ``input_ids`` width ``L`` and a per-item ``attention_mask`` — left-padding is mandatory for
          a decoder-only ``generate`` so every item's continuation begins at the SAME column ``L``;
        * ONE ``model.generate(**inputs, do_sample=False)`` (greedy) over the whole batch;
       * per item, the continuation is ``generated_ids[i][L:]`` — sliced by the shared padded prompt
          width ``L`` (``input_ids.shape[1]``), which is exactly where each item's generation starts
          under left-padding. The per-item real prompt length (``attention_mask[i].sum()``) is
          honoured by the mask so batching a variable-length page next to a short one does NOT bleed
          one page's prompt/pad tokens into another's markdown. Decoding without the ``L`` slice
          prepend the chat/prompt/instruction scaffolding to every page (the false-green trap).

        Greedy (``do_sample=False``) + correct left-padding/attention-mask ⇒ the batched markdown is
        byte-identical to running each page alone (validated offline by the fake VLM + on the pod).
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=False,
        )
        # generate() returns the FULL sequence (prompt + continuation) for each item. Under
        # left-padding every item's continuation starts at the shared padded prompt width L =
        # input_ids.shape[1]; slice each item there (generated_ids[i][L:]) so only the page's
        # markdown continuation is decoded. Per-item, not a single shared decode.
        prompt_width = inputs["input_ids"].shape[1]
        texts: list[str] = []
        for i in range(len(chunk)):
            continuation = generated_ids[i][prompt_width:]
            texts.append(processor.decode(continuation, skip_special_tokens=True))
        return texts

    def _build_messages(self, image: Any) -> list[dict[str, Any]]:
        """The single-image OCR chat message GLM-OCR expects (image + the OCR instruction text).

        Unlike LightOn-OCR (image-only), GLM-OCR is prompted with a text recognition instruction
        alongside the page image — per the model card. The instruction is configurable (``prompt``).
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

    def _load_model(self) -> tuple[Any, Any]:
        """Import transformers + load the GLM-OCR model/processor on first use (cached).

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
                AutoProcessor,
                GlmOcrForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'transformers' package (with GLM-OCR support) is required to run "
                "GlmOcrParser. Install it with: pip install latence-parser-glm"
            )
            raise ImportError(msg) from exc
        device = self.device.device
        # W8 perf seam: bf16-on-GPU + sdpa/FA2 (graceful FA2→sdpa) into the VLM load, instead of the
        # fp32/eager default (the ~2× miss). The processor takes no perf kwargs. After placement the
        # model is optionally compiled (guarded, off by default).
        perf = self.load_perf_kwargs()
        with self.guard("load_model"):
            model = load_with_fa2_fallback(
                GlmOcrForConditionalGeneration.from_pretrained, self._model_id, perf
            )
            if device is not None:
                model = model.to(device)
            processor = AutoProcessor.from_pretrained(self._model_id)
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
        matching ``parser.pdfplumber`` / ``parser.lighton`` exactly.
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
    ``ParseQuality.bytes_after_parse``. Mirrors the in-core / pdfplumber / lighton Parser helper so
    a Pipeline that swaps this Provider in gets the same observability signal.
    """
    return len(record.content.encode("utf-8"))


class _ParseFailure(LatenceError):
    """A recoverable, framework-typed parse failure → a PARSE_ERROR record (ADR-0020).

    A :class:`~latence_core.errors.LatenceError` subclass so it flows through the same typed-error
    boundary the guarded native fault does: :meth:`GlmOcrParser._parse_one` catches
    ``LatenceError`` and turns either into a graceful PARSE_ERROR record. It carries only a static
    reason (unsupported
    type, spoofed signature, missing dep) — never input content / OCR text.
    """

    category = "provider"
