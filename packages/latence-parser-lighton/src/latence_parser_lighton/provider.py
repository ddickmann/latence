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
from latence_core.providers.perf import load_transformers_model
from latence_core.providers.profile import ProviderProfile

from latence_parser_lighton.rasterize import (
    DEFAULT_DPI,
    DEFAULT_MAX_LONGEST_DIM,
    MAX_RENDER_MEGAPIXELS,
    MAX_RENDER_PIXELS,
    PageRasteriser,
    RasterFailure,
)

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
# defaulting to LightOn's 1540; ``0`` (or ``None``) disables the cap. Sourced from the SHARED
# rasteriser so ``parser.lighton`` and ``parser.lighton_vllm`` cannot diverge (W10).
_DEFAULT_MAX_LONGEST_DIM = DEFAULT_MAX_LONGEST_DIM

# How many page images share one ``generate()`` call (W8 OCR page-batching). Today's one-page-per-
# generate is the throughput bottleneck; batching N pages left-pads them into one forward pass. A
# small default (memory-shaping only — batched OCR is record-for-record identical to per-page under
# greedy decoding + correct left-padding + per-item prompt-slice). Overridable via ``batch_size``.
_DEFAULT_OCR_BATCH_SIZE = 4

# --- Pixel-bomb DoS cap (the Hardening-bar item; this Parser rasterises untrusted input) ------
# The megapixel budget + pixel count live in the SHARED rasteriser (W10) so both OCR back-ends bound
# the raster identically. Re-exported here under the module-private names existing tests reference.
_MAX_RENDER_MEGAPIXELS = MAX_RENDER_MEGAPIXELS
_MAX_RENDER_PIXELS = MAX_RENDER_PIXELS


def _import_ocr_backbone() -> tuple[Any, Any]:
    """Return the (model class, processor class) pair — the import factory the perf seam calls.

    The model half is handed to :func:`~latence_core.providers.perf.load_transformers_model` rather
    than imported inline, so the whole load choreography (JIT kill → thread cap → import → perf
    kwargs → guarded load → **verified** placement → compile) runs in one place and in one order.
    The processor takes no perf kwargs and is loaded by the Provider itself. The native dependency
    and its install hint stay in this package (ADR-0036 §2); only the instant of the import moved.
    """
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
    return LightOnOcrForConditionalGeneration, LightOnOcrProcessor


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
        self._dpi = int(cfg.get("dpi", DEFAULT_DPI))
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
        # The SHARED rasteriser (W10): the SAME preprocessing (dpi / megapixel bomb-cap / 1540px
        # LANCZOS longest-side / spoof guards) both this in-process Parser and
        # ``parser.lighton_vllm`` run — so the two OCR back-ends cannot silently diverge at the
        # pixel boundary. The Provider's ``guard`` maps a native rasteriser fault to a typed error.
        self._rasteriser = PageRasteriser(
            dpi=self._dpi, max_longest_dim=self._max_longest_dim, guard=self.guard
        )
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

    # -- rasterise raw bytes → page images (delegated to the SHARED rasteriser, W10) --------------

    def _rasterise(self, raw: ParserInput, file_type: str) -> list[Any]:
        """Turn the raw source bytes into one Pillow image per page via the SHARED rasteriser.

        Delegates to :class:`~latence_parser_lighton.rasterize.PageRasteriser` (the single source of
        truth for the preprocessing recipe — dpi / megapixel bomb-cap / 1540px LANCZOS / spoof
        guards) so this in-process Parser and ``parser.lighton_vllm`` cannot silently diverge. The
        rasteriser raises a :class:`~latence_parser_lighton.rasterize.RasterFailure` (a
        ``LatenceError``) on an unsupported/spoofed/undecodable input, caught by :meth:`_parse_one`.
        """
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        return self._rasteriser.rasterise(data, file_type)

    def _clamp_scale(self, width_pt: float, height_pt: float, scale: float) -> float:
        """Compat passthrough to the shared rasteriser's render-scale clamp (existing W8 tests)."""
        return self._rasteriser._clamp_scale(width_pt, height_pt, scale)

    def _image_from_bytes(self, data: bytes) -> Any:
        """Compat passthrough to the shared rasteriser's single-image decode (existing W8 tests)."""
        return self._rasteriser._image_from_bytes(data)

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
        # C4: one call owns the whole ordered sequence — skip-decision → JIT kill → intra-op thread
        # cap → backbone import → bf16-on-GPU + sdpa/FA2 perf kwargs (graceful FA2→sdpa) → placement
        # on the resolved device WITH read-back verification → guarded compile. The placement is the
        # fix that was still open here: this Provider used a bare ``model = model.to(device)``, so
        # #191's "prove it landed or raise" covered 3 of the 5 model-placing Providers and a model
        # that silently reported a different device than the one requested ran anyway.
        model = load_transformers_model(
            lambda: _import_ocr_backbone()[0],
            self._model_id,
            self._config,
            self.device,
            provider_name=self.provider_name,
            guard=self.guard,
        )
        # The processor takes no perf kwargs and is not placed — a plain load under the guard.
        with self.guard("load_model"):
            processor = _import_ocr_backbone()[1].from_pretrained(self._model_id)
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

    A one-line delegation to :attr:`~latence_core.contracts.DocumentRecord.size_after_parse`,
    which is where the rule lives: the size is a property of the record, not of the Parser that
    produced it. Kept as a module-level function because this package publishes the name in its
    ``__all__``; every Parser package delegates to the same implementation, so the signal the
    Runner sums into ``ParseQuality.bytes_after_parse`` cannot drift between Providers.
    """
    return record.size_after_parse


# The recoverable rasterisation failure now lives in the shared rasteriser (:class:`RasterFailure`).
# Kept as a module-level alias so any external reference to the historical name still resolves; it
# is a :class:`~latence_core.errors.LatenceError`, so :meth:`LightOnOcrParser._parse_one` still
# catches it and degrades to a graceful PARSE_ERROR record (ADR-0020).
_ParseFailure = RasterFailure
