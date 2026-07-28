"""The shared PDF/image rasteriser for the LightOn-OCR Parsers (in-process AND served-vLLM).

The single source of truth for **page-image preprocessing** — the exact preprocessing recipe both
``parser.lighton`` (in-process transformers, W1/W8) and ``parser.lighton_vllm`` (served vLLM, W10)
run, so the two paths are byte-for-byte identical at the pixel-preparation boundary and cannot
silently diverge (the W10 reviewer's #1 check: "preprocessing IDENTICAL to parser.lighton, no
silent divergence"). Extracted verbatim from ``parser.lighton``'s W8 rasteriser; the in-process
Provider now delegates to this class, and the served Provider reuses the SAME class — one
rasteriser, two OCR back-ends.

What it does per document (LightOn's OFFICIAL preprocessing, verified from the model cards):

* **Rasterise** a PDF page-by-page via ``pypdfium2`` at ``dpi`` (default 200) so the page map is
  real (one image per page); a bare image is a single page.
* **Pixel-bomb DoS cap** — the first framework Parser that rasterises *untrusted* input, so before
  any ``.to_pil()`` / ``.load()`` it bounds the raster: the PDF render scale is clamped by a
  **max-megapixel budget** (:data:`MAX_RENDER_MEGAPIXELS`) so a giant page cannot force a gigapixel
  bitmap, and Pillow's ``Image.MAX_IMAGE_PIXELS`` is tightened so a decompression-bomb image
  **raises** rather than allocating multi-GB (restored afterward, no global side effect).
* **Longest-side downscale, onto the 28-px patch-merge grid** — after the bomb-clamp, a
  page/image whose longest side exceeds ``max_longest_dim`` (default **1540**, LightOn's recipe:
  "render PDFs at 200 DPI to a target longest dimension of 1540px, aspect preserved") is
  downscaled with **LANCZOS**, and BOTH sides are snapped down to a multiple of **28** px
  (:func:`merge_grid_target`) because ``Mistral3PatchMerger`` merges 14-px patches 2x2 with
  ``unfold(kernel=2, stride=2)`` and silently drops an odd trailing patch row/column. Never an
  upscale; ``0``/``None`` disables both.
* **Offset-0 type-spoof guards** — a ``.pdf`` whose bytes are not a PDF, or an image whose bytes
  match no known image magic, is rejected as a graceful :class:`RasterFailure` (defense-in-depth,
  #62).

The heavy deps (``pypdfium2``, ``pillow``) are imported **inside** the methods so importing this
module (for Provider discovery) never needs them (ADR-0016), and a missing dep is a graceful
:class:`RasterFailure`, not an import crash. Native rasteriser faults are surfaced through an
injected ``guard`` context manager (the Provider's :meth:`AdapterBase.guard`) so they map to the
Provider's typed error taxonomy carrying no image bytes / content — the caller owns the taxonomy,
this module owns the pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.errors import LatenceError

if TYPE_CHECKING:
    from collections.abc import Iterator

# --- LightOn's OFFICIAL preprocessing defaults (verified from the model cards) ----------------
# The PDF->image rasterisation DPI. LightOn renders at 200 DPI; higher = sharper but slower.
DEFAULT_DPI = 200
# The target LONGEST image dimension (px) a rasterised page / decoded image is downscaled to, aspect
# preserved. LightOn's recipe: "render PDFs at 200 DPI to a target longest dimension of 1540px". An
# A4 at 200 DPI is ~2339px tall — fed full-size that is ~2.3x the vision tokens vs the target. NOT
# byte-identical (OCR output changes with resolution), so a config knob; 0/None disables.
DEFAULT_MAX_LONGEST_DIM = 1540

# --- The patch-merge grid BOTH sides must land on (T-OCR-2) -----------------------------------
# LightOnOCR is a ``mistral3`` model over a ``pixtral`` vision tower: the page is cut into 14-px
# patches, and ``Mistral3PatchMerger.forward`` then merges them 2x2 with
# ``torch.nn.functional.unfold(kernel_size=2, stride=2)``. ``unfold`` emits ONLY COMPLETE WINDOWS,
# so an odd patch row or column has no 2x2 partner and is **silently discarded** together with the
# ~14 px of page it covered. 14 x 2 = 28: a side that is a multiple of 28 always has an even patch
# count, so nothing is dropped. 1540 is already safe (1540/14 = 110, even) — but the SHORTER side
# used to land wherever the aspect ratio put it, and a US-Letter page at 200 DPI (1700x2200) scales
# to width 1190 = 85 patches = ODD, losing the right-hand 14 px of every such page.
#
# Honest scope (issues/e2e-extended/DECISIONS.md § D1a): the mechanism is verified in the
# vLLM/transformers source; the visible-loss consequence is inferred, not LightOn-confirmed. It is
# cheap to satisfy and cheap to test, so it is satisfied.
PATCH_SIZE = 14
PATCH_MERGE = 2
MERGE_GRID = PATCH_SIZE * PATCH_MERGE


def merge_grid_target(width: int, height: int, cap: int) -> tuple[int, int] | None:
    """The page size to resize ``width x height`` to so BOTH sides sit on the 28-px merge grid.

    Returns ``None`` when the image must be left exactly as it is — either it is already aligned
    and within ``cap`` (no needless resample), or a side is narrower than one 28-px merge window,
    where aligning it would mean **enlarging** the page. Never upscaling is a hard rule: LightOn's
    recipe caps resolution, it does not manufacture it, and an upscaled page costs vision tokens
    without adding information.

    ``cap <= 0`` disables the longest-side knob entirely (``max_longest_dim = 0``, "hand me the raw
    raster") and therefore disables the grid rounding with it — a deliberate, documented bypass:
    an operator who turns the preprocessing off owns the patch grid.

    Guarantees, all property-tested in ``test_rasterize_merge_grid.py``:

    * both returned sides are ``0 mod 28`` — no patch row/column is dropped by the 2x2 merge;
    * ``max(w, h) <= cap`` — LightOn's longest-dimension recipe still holds;
    * neither side exceeds its input — no upscaling, ever;
    * each side is within ONE 28-px step of where exact proportional scaling would put it, so the
      aspect ratio is preserved to the finest granularity the grid can express.
    """
    if cap <= 0 or width <= 0 or height <= 0:
        return None
    longest = max(width, height)
    # Downscale-only: an image already inside the cap keeps its scale and is merely snapped.
    scale = 1.0 if longest <= cap else cap / float(longest)

    sides: list[int] = []
    for side in (width, height):
        # The most this side may become: its own size (no upscaling) and never above the cap.
        limit = min(side, cap)
        if limit < MERGE_GRID:
            return None  # a strip thinner than one merge window cannot be aligned downward
        snapped = int(round(side * scale / MERGE_GRID)) * MERGE_GRID
        if snapped > limit:  # nearest grid point overshoots -> step down to the one below
            snapped = (limit // MERGE_GRID) * MERGE_GRID
        snapped = max(snapped, MERGE_GRID)  # an extreme ratio can round a side to zero
        sides.append(snapped)

    target = (sides[0], sides[1])
    return None if target == (width, height) else target

# --- Pixel-bomb DoS cap (the Hardening-bar item; this Parser rasterises untrusted input) ------
# The maximum megapixels a rasterised page / decoded image may occupy. A PDF page's render scale is
# clamped so its bitmap stays under this budget (a giant page cannot force a gigapixel bitmap), and
# Pillow's ``Image.MAX_IMAGE_PIXELS`` is set to this budget so a decompression-bomb image RAISES
# rather than allocating multi-GB. ~25 MP is generous for a legible document scan (~ an A4 at 600
# DPI) while bounding a single page's raster to well under a gigabyte. The intake max_bytes cap does
# NOT bound this — page dimensions / declared pixel counts are independent of file byte size.
MAX_RENDER_MEGAPIXELS = 25.0
MAX_RENDER_PIXELS = int(MAX_RENDER_MEGAPIXELS * 1_000_000)

# File types the rasteriser turns into page images. A PDF is rendered page-by-page (pypdfium2) so
# the page map is real (one OCR pass per page); a bare image is a single page.
PDF_EXTENSIONS = frozenset({"pdf"})
IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"})

# %PDF- magic at offset 0 (a bounded leading BOM/whitespace window only) — the offset-0 type-spoof
# guard mirroring ``parser.pdfplumber`` / the intake Screener (#62): a ``.pdf`` whose bytes are not
# a PDF is rejected as a parse failure before the rasteriser, defense-in-depth.
PDF_MAGIC = b"%PDF-"
_PDF_LEADING_SLACK = 8
_PDF_LEADING_SKIP = b"\x00\r\n\t \xef\xbb\xbf\xff\xfe"

# Magic-byte prefixes for the image formats this rasteriser accepts — the offset-0 spoof guard for a
# declared image whose bytes are not that image (defense-in-depth, mirroring the PDF guard). A
# format not in this table is not spoof-checked (handed to Pillow, which fails gracefully).
IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"RIFF",  # WEBP (RIFF container; "WEBP" follows at offset 8)
    b"II*\x00",  # TIFF little-endian
    b"MM\x00*",  # TIFF big-endian
    b"BM",  # BMP
)


class RasterFailure(LatenceError):
    """A recoverable, framework-typed rasterisation failure → the caller's PARSE_ERROR (ADR-0020).

    A :class:`~latence_core.errors.LatenceError` subclass so it flows through the same typed-error
    boundary a Parser Provider's guarded native fault does: the Provider catches ``LatenceError``
    and turns either into a graceful PARSE_ERROR record. It carries only a static reason
    (unsupported type, spoofed signature, missing dep) — never input content / OCR text.
    """

    category = "provider"


class PageRasteriser:
    """Turn raw PDF/image bytes into Pillow page images (spoof-guarded, pixel-capped, downscaled).

    Constructed with the resolved ``dpi`` + ``max_longest_dim`` (already validated by the Provider)
    and a ``guard`` factory — the Provider's ``AdapterBase.guard`` — so a native pypdfium2/Pillow
    fault maps to the Provider's typed taxonomy (no content). Both ``parser.lighton`` and
    ``parser.lighton_vllm`` construct one and call :meth:`rasterise`, so the preprocessing is
    provably identical.
    """

    def __init__(
        self,
        *,
        dpi: int,
        max_longest_dim: int,
        guard: Any,
    ) -> None:
        self._dpi = dpi
        self._max_longest_dim = max_longest_dim
        # ``guard`` is a callable(op_label) -> context manager (the Provider's AdapterBase.guard),
        # so a native rasteriser fault becomes the Provider's typed ProviderError (no image bytes).
        self._guard: Any = guard

    # -- rasterise raw bytes → page images (spoof-guarded, pixel-capped) ----

    def rasterise(self, data: bytes, file_type: str) -> list[Any]:
        """Turn the raw source bytes into one Pillow image per page (spoof-guarded, pixel-capped).

        A PDF is rendered page-by-page via ``pypdfium2`` (the per-page split is what makes the page
        map real — one OCR pass per page); a bare image is a single page. An unsupported type or a
        type-spoofed file (bytes not matching the declared type) is a graceful
        :class:`RasterFailure` (→ the caller's PARSE_ERROR), not a crash — mirroring
        ``parser.pdfplumber`` / #62.
        """
        ftype = file_type.lower()
        if ftype in PDF_EXTENSIONS:
            self._guard_pdf_signature(data)
            return self._pdf_to_images(data)
        if ftype in IMAGE_EXTENSIONS:
            self._guard_image_signature(data)
            return [self._image_from_bytes(data)]
        msg = f"Unsupported file type {ftype!r} for the LightOn-OCR Parser."
        raise RasterFailure(msg)

    def _guard_pdf_signature(self, data: bytes) -> None:
        """Reject a ``.pdf`` whose bytes are not a PDF (offset-0 spoof guard, like pdfplumber)."""
        offset = 0
        limit = min(len(data), _PDF_LEADING_SLACK)
        while offset < limit and data[offset] in _PDF_LEADING_SKIP:
            offset += 1
        if data[offset : offset + len(PDF_MAGIC)] != PDF_MAGIC:
            msg = (
                "File declared as PDF but the %PDF- signature is not at the start of the file "
                "(type spoof?)."
            )
            raise RasterFailure(msg)

    def _guard_image_signature(self, data: bytes) -> None:
        """Reject a declared image whose bytes match no known image magic (offset-0 spoof guard)."""
        if not any(data.startswith(magic) for magic in IMAGE_MAGIC):
            msg = (
                "File declared as an image but its bytes match no known image signature "
                "(type spoof?)."
            )
            raise RasterFailure(msg)

    def _clamp_scale(self, width_pt: float, height_pt: float, scale: float) -> float:
        """Reduce a PDF render scale so the rasterised bitmap stays under the megapixel budget.

        pypdfium2 renders a page of ``width_pt x height_pt`` points at ``scale`` px/pt, so the
        bitmap is ``(width_pt*scale) x (height_pt*scale)`` pixels. If that exceeds
        :data:`MAX_RENDER_PIXELS` (the pixel-bomb budget) the scale is reduced by
        ``sqrt(budget / pixels)`` so the area lands exactly at the budget — a 10 000x10 000-pt page
        can never rasterise to a gigapixel bitmap. A degenerate (zero-area) page keeps the requested
        scale (nothing to clamp).
        """
        area_pt = max(width_pt, 0.0) * max(height_pt, 0.0)
        if area_pt <= 0.0 or scale <= 0.0:
            return scale
        pixels = area_pt * scale * scale
        if pixels <= MAX_RENDER_PIXELS:
            return scale
        return float(scale * (MAX_RENDER_PIXELS / pixels) ** 0.5)

    def _pdf_to_images(self, data: bytes) -> list[Any]:
        """Rasterise every PDF page to a Pillow image via ``pypdfium2`` (ordered, pixel-capped).

        Imported inside the method so a missing/malformed dependency never breaks Provider
        discovery, and so ``latence-core`` stays free of the heavy dep (ADR-0016). A missing dep is
        a graceful :class:`RasterFailure`. Each page's render scale is clamped by
        :meth:`_clamp_scale` so an oversized page cannot force a gigapixel bitmap (the pixel-bomb
        cap). The native pypdfium2 fault a corrupt PDF raises is mapped by the injected ``guard`` to
        the Provider's typed error.
        """
        try:
            import io

            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pypdfium2 is required to rasterise PDFs for OCR."
            raise RasterFailure(msg) from exc

        requested_scale = self._dpi / 72.0  # pypdfium2 renders at 72 DPI * scale
        images: list[Any] = []
        with self._guarded("rasterise"):
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
        fault maps through the injected ``guard`` to the Provider's typed error and degrades to a
        PARSE_ERROR record. The prior value is restored in a ``finally`` so we never leave global
        Pillow state mutated for other users of the process.
        """
        try:
            import io

            from PIL import Image
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pillow is required to decode images for OCR."
            raise RasterFailure(msg) from exc
        previous_cap = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = MAX_RENDER_PIXELS
        try:
            with self._guarded("decode_image"):
                image = Image.open(io.BytesIO(data))
                image.load()  # a decompression-bomb image raises here (bounded, no huge alloc)
                return self._downscale_longest(image.convert("RGB"))
        finally:
            Image.MAX_IMAGE_PIXELS = previous_cap

    def _downscale_longest(self, image: Any) -> Any:
        """Downscale a page image onto the merge grid, longest side <= ``max_longest_dim`` px.

        LightOn's OFFICIAL preprocessing: "render PDFs at 200 DPI to a target longest dimension of
        1540px, maintaining aspect ratio." Applied to BOTH rasterised PDF pages and bare-image
        inputs, AFTER the megapixel bomb-clamp (so a bomb still degrades gracefully first).

        On top of that cap, BOTH sides are snapped down onto the 28-px patch-merge grid — see
        :func:`merge_grid_target` for why (an odd patch row is silently dropped by
        ``Mistral3PatchMerger``'s ``unfold(kernel=2, stride=2)``). That makes this a downscale even
        for a page already inside the cap, which is the point: 800x1000 becomes 784x980 rather than
        losing its bottom and right edges inside the vision tower. It is still never an UPSCALE.

        When the cap is disabled (``max_longest_dim == 0``), or the image is already aligned and in
        budget, the image is returned untouched — no resample cost. A resize uses LANCZOS (Pillow's
        high-quality resampling) so text stays legible. NOT byte-identical to the un-capped path
        (OCR output changes with resolution) — the throughput knob, defaulting to LightOn's 1540.
        """
        width, height = image.size
        target = merge_grid_target(width, height, self._max_longest_dim)
        if target is None:
            return image
        from PIL import (
            Image,  # noqa: PLC0415 - deferred heavy dep (ADR-0016); already imported above
        )

        return image.resize(target, Image.Resampling.LANCZOS)

    def _guarded(self, op: str) -> Any:
        """Open the Provider's guard for ``op`` (native fault → typed ProviderError, no content).

        Returns the Provider's guard context manager (typed ``Any`` because the injected ``guard`` —
        the Provider's :meth:`AdapterBase.guard` — is supplied opaquely, so this module carries no
        dependency on the Provider's concrete guard type).
        """
        return self._guard(op)


def image_to_data_url(image: Any, *, jpeg_quality: int = 95) -> str:
    """Encode a Pillow page image as a base64 PNG ``data:`` URL for an OpenAI multimodal message.

    The served-vLLM path posts each rasterised page as an ``image_url`` whose ``url`` is a
    ``data:image/png;base64,...`` URL — the OpenAI multimodal image contract vLLM's
    chat-completions endpoint serves. PNG (lossless) keeps the OCR input identical to the pixels the
    rasteriser produced (no JPEG artefacts on text). Pillow is imported lazily (ADR-0016).
    ``jpeg_quality`` is accepted for API symmetry but unused for PNG (lossless) — kept so a future
    JPEG mode is a one-line change without a signature break.
    """
    import base64
    import io

    buffer = io.BytesIO()
    # RGB so a palette/GA image encodes cleanly; the rasteriser already converts, this is defensive.
    to_save = image if getattr(image, "mode", "RGB") == "RGB" else image.convert("RGB")
    to_save.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def iter_page_batches(images: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield ``images`` in contiguous chunks of ``size`` (>=1), preserving order (shared helper)."""
    step = max(1, size)
    for start in range(0, len(images), step):
        yield images[start : start + step]
