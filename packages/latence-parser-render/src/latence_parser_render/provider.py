"""``parser.render`` — the universal any-file → markdown Parser (W5-universal-parser).

A **license-clean port of the colsearch batch converter** ("any file → batches of images"): one
Parser Provider (CONTEXT ``Parser``) that turns **ANY** common file type — PDF, image, or office
document (docx/pptx/xlsx) — into a :class:`~latence_core.contracts.DocumentRecord` +
:class:`~latence_core.contracts.PageMap`, dispatching by type. It is the *front-end* an OCR pipeline
needs so a real corpus (scanned PDFs, photos, office files) all flows through one Parse Stage.

**License discipline — the whole reason for a clean port (ADR-0012, research-diligence).** colsearch
renders PDF with PyMuPDF (``fitz``) — **AGPL-3.0**, copyleft. This Provider NEVER imports or depends
on ``fitz``/PyMuPDF (a test greps the package to enforce it). It uses only PERMISSIVE deps, the same
the framework already ships:

* **PDF → images:** ``pypdfium2`` (Apache-2.0 / BSD-3, wraps Google PDFium) — the SAME renderer the
  :class:`~latence_parser_lighton.provider.LightOnOcrParser` uses. NOT ``fitz``.
* **Office (docx/pptx/xlsx):** prefer **LibreOffice headless** (``soffice --convert-to pdf``,
  MPL-2.0, invoked as a **subprocess** — a faithful page render) → ``pypdfium2`` images, WHEN
  ``soffice`` is on ``PATH``; else fall back to **python-docx / openpyxl / python-pptx** (MIT) text
  extraction → markdown (no OCR). ``soffice`` is **OPTIONAL** — a host without it degrades to text
  extraction, never a crash. LibreOffice is external (not linked/vendored), so its copyleft never
  reaches this package.
* **Images:** ``Pillow`` (HPND) — a bare image is a one-page document.

Heavy deps live ONLY in this package (ADR-0016); imports are deferred to first use so Provider
discovery stays light.

**OCR delegation (reuse the pod-validated OCR models, do not reinvent).** This Provider is the
*renderer* — it turns any file into page images; the actual pixels→markdown OCR is delegated to a
**configured OCR Provider** (``ocr: {provider: parser.lighton | parser.glm, config: {...}}``). It
resolves that Provider from the framework registry (ADR-0004) and hands it each page image as a
one-page ``png`` :class:`~latence_core.contracts.ParserInput` **through the same ``Parser.parse``
seam** — so the OCR Provider is used exactly as it is elsewhere, no private method reached into, no
seam forked. The office text-fallback path needs **no** OCR (it extracts a text layer directly), so
``ocr`` is only required when a page must be read from pixels (PDF / image / soffice-office).

**Dispatch + graceful failure (like every other Parser, ADR-0020).** Unknown/unsupported types,
type-spoofed files, corrupt inputs, a soffice timeout, or a missing OCR Provider all degrade to a
graceful ``PARSE_ERROR`` :class:`~latence_core.contracts.DocumentRecord` — never a crash that aborts
the run.

**Security (carry the parser lessons — this rasterises untrusted input).**

* **Pixel-bomb DoS cap** — the ``parser.lighton`` discipline: PDF render scale clamped by a
  max-megapixel budget (:data:`_MAX_RENDER_MEGAPIXELS`) so a 10 000×10 000-pt page can never
  rasterise to a gigapixel bitmap; ``Image.MAX_IMAGE_PIXELS`` tightened so a decompression-bomb
  image **raises** rather than allocating multi-GB (restored afterwards, no global side effect).
* **Offset-0 type-spoof guard (#62)** — a ``.pdf`` / declared image whose bytes are not that type is
  rejected before the rasteriser (defense-in-depth, mirroring ``parser.pdfplumber`` / the intake
  Screener).
* **soffice subprocess** — run with an explicit argument list (**no shell**, no injection), a hard
  **timeout**, and in an isolated temp dir; a timeout / non-zero exit / missing binary is a graceful
  parse failure, never a hang or a crash.
* **native fault → typed error, no leak** — a native library fault maps through
  :meth:`~latence_core.providers.adapter.AdapterBase.guard` to a typed
  :class:`~latence_core.errors.ProviderError` carrying only provider + op + native type name —
  **never** input bytes / extracted text / OCR content (C2/C6).

**Exact offsets/pages (ADR-0031).** The per-page markdown (OCR'd or text-extracted) is assembled via
:func:`~latence_core.contracts.PageMap.from_page_texts` — boundaries exact by construction — and
Provenance is carried via :meth:`~latence_core.providers.adapter.AdapterBase.carry_provenance`.

``compute="either"``: the rendering/text-extraction is CPU; OCR routes to the configured OCR
Provider's own device (a GPU OCR Provider skips-with-flag on a CPU-only host through ITS device
seam). So on a CPU host this Provider still runs its renderer and, with an offline/stub OCR
Provider, end-to-end — it is not itself GPU-gated.
"""

from __future__ import annotations

import io
import shutil
import subprocess  # noqa: S404 - used ONLY with an explicit arg list (no shell), a timeout, no injection
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from latence_core.capability import ProviderRegistry
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    Provenance,
)
from latence_core.errors import ConfigError, LatenceError, ProviderError
from latence_core.language import detect_language
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.capability import Parser
    from latence_core.contracts import ParserInput

# --- Pixel-bomb DoS cap (the Hardening-bar item; this Parser rasterises untrusted input) ------
# The maximum megapixels a rasterised page / decoded image may occupy — mirrors parser.lighton. A
# PDF page's render scale is clamped so its bitmap stays under the budget (a giant page cannot force
# a gigapixel bitmap), and Pillow's Image.MAX_IMAGE_PIXELS is set to this budget so a
# decompression-bomb image RAISES rather than allocating multi-GB. ~25 MP is generous for a legible
# document scan while bounding a single page's raster to well under a gigabyte. The intake max_bytes
# cap does NOT bound this — page dimensions / declared pixel counts are independent of byte size.
_MAX_RENDER_MEGAPIXELS = 25.0
_MAX_RENDER_PIXELS = int(_MAX_RENDER_MEGAPIXELS * 1_000_000)

# The default soffice subprocess timeout (seconds). A malicious / pathological office file must not
# be able to hang the run; a timeout is a graceful parse failure (the file falls through to the
# text-extraction fallback is NOT attempted after a soffice failure — a soffice failure means the
# faithful render failed, so we degrade to the text path explicitly, see _office_to_page_texts).
_DEFAULT_SOFFICE_TIMEOUT_S = 120

# File-type dispatch tables. A PDF is rendered page-by-page (pypdfium2); a bare image is one page;
# an office file goes soffice→pdf→pypdfium2 when soffice is present, else a MIT text-reader.
_PDF_EXTENSIONS = frozenset({"pdf"})
_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"})
_DOCX_EXTENSIONS = frozenset({"docx"})
_PPTX_EXTENSIONS = frozenset({"pptx"})
_XLSX_EXTENSIONS = frozenset({"xlsx"})
_OFFICE_EXTENSIONS = _DOCX_EXTENSIONS | _PPTX_EXTENSIONS | _XLSX_EXTENSIONS

# %PDF- magic at offset 0 (a bounded leading BOM/whitespace window only) — the offset-0 type-spoof
# guard mirroring parser.pdfplumber / parser.lighton / the intake Screener (#62).
_PDF_MAGIC = b"%PDF-"
_PDF_LEADING_SLACK = 8
_PDF_LEADING_SKIP = b"\x00\r\n\t \xef\xbb\xbf\xff\xfe"

# Magic-byte prefixes for the image formats — the offset-0 spoof guard for a declared image whose
# bytes are not that image (defense-in-depth, mirroring parser.lighton).
_IMAGE_MAGIC: tuple[bytes, ...] = (
    b"\x89PNG\r\n\x1a\n",  # PNG
    b"\xff\xd8\xff",  # JPEG
    b"RIFF",  # WEBP (RIFF container; "WEBP" follows at offset 8)
    b"II*\x00",  # TIFF little-endian
    b"MM\x00*",  # TIFF big-endian
    b"BM",  # BMP
)

# Office Open XML files are ZIP containers — offset-0 spoof guard: an office file must start with a
# ZIP local-file-header ("PK\x03\x04") or the rarer empty/spanned markers. A declared .docx that is
# not a ZIP is a spoof (defense-in-depth; the real reader would fail, but we reject early).
_ZIP_MAGIC: tuple[bytes, ...] = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


class UniversalRenderParser(AdapterBase):
    """Universal any-file Parser Provider: PDF/image/office → markdown records + page map.

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass (H-A1) satisfying the
    :class:`~latence_core.capability.Parser` Protocol structurally (it implements ``parse``). It
    declares a ``compute="either"`` :class:`~latence_core.providers.profile.ProviderProfile`,
    batches over input docs via :meth:`AdapterBase.batched`, maps native faults through
    :meth:`AdapterBase.guard`, carries Provenance via :meth:`AdapterBase.carry_provenance`, and
    assembles per-page text onto :func:`~latence_core.contracts.PageMap.from_page_texts`.

    Config:
        ocr: dict — ``{"provider": "parser.lighton" | "parser.glm", "config": {...}}``. The OCR
            Provider each rasterised page image is delegated to (through its ``Parser.parse`` seam).
            REQUIRED to OCR a PDF / image / soffice-rendered office file; the office
            **text-fallback** path needs no OCR. A missing/unknown OCR Provider is a graceful
            PARSE_ERROR for a page that needs pixels-OCR (never a crash).
        category: str | None — content category stamped onto Classification (else inherited).
        sensitivity: str | None — sensitivity stamped onto Classification (else inherited).
        page_separator: str — markdown joined between pages (default ``"\\n\\n"``).
        dpi: int — the PDF→image rasterisation DPI (default 200; higher = sharper, slower).
        soffice_timeout_s: int — the soffice subprocess hard timeout (default 120s).
        prefer_soffice: bool — when False, always use the MIT text-extraction path for office files
            even if soffice is present (default True: faithful render when soffice is available).
        batch_size / device — resolved by :class:`AdapterBase` (device only shapes the *renderer*;
            OCR device is the OCR Provider's own concern).

    ``config_exceptions``: a native ``ValueError``/``KeyError``/``OSError`` from a reader means an
    operator handed a bad config / a genuinely malformed file → a
    :class:`~latence_core.errors.ConfigError` via :meth:`guard`; everything else is a runtime
    :class:`~latence_core.errors.ProviderError`. Both degrade to a graceful PARSE_ERROR.
    """

    #: A native bad-config/malformed-input exception → ConfigError; else → ProviderError (guard).
    config_exceptions: ClassVar[tuple[type[Exception], ...]] = (ValueError, KeyError, OSError)

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): ``compute="either"``, permissive, AGPL-free.

        ``compute="either"``: the rendering (pypdfium2/Pillow) and office text-extraction are pure
        CPU code; page OCR is delegated to a configured OCR Provider that routes to its OWN device.
        So this Provider is not itself GPU-gated — on a CPU-only host it still runs its renderer
        and, with an offline/stub OCR Provider, end-to-end.

        License is **verified** per ADR-0012, recorded evidence-bearingly (H-B1). This Provider
        ships no model weights of its own (``model_id=None``): OCR weights belong to the delegated
        OCR Provider (whose own profile records its weights license). The Provider's code and its
        dependencies are ALL permissive — pypdfium2 (Apache-2.0/BSD-3), Pillow (HPND), python-docx /
        openpyxl / python-pptx (MIT); LibreOffice, when used, is an EXTERNAL MPL-2.0 subprocess tool
        (not linked/vendored). NO AGPL anywhere (``fitz``/PyMuPDF is never imported — a test
        enforces it). The headline/code license is this repo's PolyForm-Noncommercial-1.0.0 (the
        Provider is code).

        ``deterministic=False``: page OCR is delegated to a VLM OCR Provider whose float decoding is
        not promised byte-identical across hardware/library versions, so this composite Provider
        declares non-determinism honestly rather than overclaim (the text-only office path IS
        deterministic, but the profile is one honest declaration for the Provider as a whole).
        ``batch=True`` (it batches over documents). ``memory_mb`` is the approximate renderer
        footprint only (the OCR Provider's VRAM is its own declaration). ``cost_per_1k=None``.
        """
        return ProviderProfile(
            compute="either",
            memory_mb=512,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_weights=None,
            # Where a human checked the permissive deps + the AGPL-free discipline (ADR-0012):
            license_source=(
                f"{REPO_LICENSE_SOURCE} (framework code {REPO_LICENSE_CODE}); "
                "deps all permissive — "
                "pypdfium2 Apache-2.0/BSD-3, Pillow HPND, python-docx/openpyxl/python-pptx MIT; "
                "LibreOffice MPL-2.0 external subprocess (not vendored); NO fitz/PyMuPDF/AGPL. "
                "Verified 2026-07-09."
            ),
            license_verified_on="2026-07-09",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    @property
    def provider_name(self) -> str:
        """The registered entry-point name — the stable tag a mapped error carries (no content)."""
        return "parser.render"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self._category = cfg.get("category")
        self._sensitivity = cfg.get("sensitivity")
        self._page_separator = str(cfg.get("page_separator", "\n\n"))
        self._dpi = int(cfg.get("dpi", 200))
        if self._dpi < 1:
            msg = f"{self.provider_name}: dpi must be >= 1, got {self._dpi}."
            raise ProviderError(msg)
        self._soffice_timeout_s = int(cfg.get("soffice_timeout_s", _DEFAULT_SOFFICE_TIMEOUT_S))
        if self._soffice_timeout_s < 1:
            msg = (
                f"{self.provider_name}: soffice_timeout_s must be >= 1, "
                f"got {self._soffice_timeout_s}."
            )
            raise ProviderError(msg)
        self._prefer_soffice = bool(cfg.get("prefer_soffice", True))
        # The configured OCR sub-provider spec: {"provider": name, "config": {...}}. Validated
        # eagerly for shape (an operator config error should fail loud at construction), but the
        # actual OCR Provider is resolved+instantiated lazily on first page that needs pixels-OCR
        # (so an office-text-only run never touches an OCR Provider, and discovery stays light).
        self._ocr_spec = self._validate_ocr_spec(cfg.get("ocr"))
        self._ocr_provider: Parser | None = None
        # A registry override seam so a test can inject a stub-OCR registry without a real install;
        # None ⇒ the default entry-point registry (the production path).
        self._registry: ProviderRegistry = self._ocr_registry()

    # -- config validation --------------------------------------------------

    def _validate_ocr_spec(self, raw: Any) -> dict[str, Any] | None:
        """Validate the ``ocr`` config shape at construction (fail loud on an operator typo).

        ``None`` (no OCR configured) is allowed — an office-text-only run needs no OCR; a page that
        DOES need pixels-OCR then degrades to a graceful PARSE_ERROR with a clear reason. A present
        ``ocr`` must be a mapping with a string ``provider`` and an optional dict ``config``.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            msg = f"{self.provider_name}: 'ocr' config must be a mapping, got {type(raw).__name__}."
            raise ConfigError(msg)
        provider = raw.get("provider")
        if not isinstance(provider, str) or not provider:
            msg = f"{self.provider_name}: 'ocr.provider' must be a non-empty provider name."
            raise ConfigError(msg)
        sub_config = raw.get("config", {})
        if not isinstance(sub_config, dict):
            msg = f"{self.provider_name}: 'ocr.config' must be a mapping."
            raise ConfigError(msg)
        return {"provider": provider, "config": dict(sub_config)}

    def _ocr_registry(self) -> ProviderRegistry:
        """The registry used to resolve the OCR sub-provider (overridable via config for tests).

        Production uses the default entry-point registry (ADR-0004). A test may pass
        ``config["ocr_registry"]`` — a :class:`ProviderRegistry` (or any object with a compatible
        ``load``) — to resolve a stub OCR Provider without installing a real package. The override
        is a test seam only; a non-registry value falls back to the default registry.
        """
        override = self._config.get("ocr_registry")
        if isinstance(override, ProviderRegistry):
            return override
        return ProviderRegistry()

    # -- Parser Capability method (structural; the base adds no protocol) ---

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        """Consume raw :class:`ParserInput`s and yield parsed markdown records, in input order.

        Runs the base's batching path (:meth:`AdapterBase.batched`) over input documents so the
        seam's streaming-friendly, input-order-preserving batching is exercised; each document is
        rendered + OCR'd (or text-extracted) one at a time. Ordering is preserved (ADR-0033).
        """
        yield from self.batched(list(inputs), self._parse_batch)

    def _parse_batch(self, batch: list[ParserInput]) -> list[DocumentRecord]:
        """Render+parse one batch of raw inputs → one :class:`DocumentRecord` each, in order."""
        return [self._parse_one(raw) for raw in batch]

    # -- per-document parse -------------------------------------------------

    def _parse_one(self, raw: ParserInput) -> DocumentRecord:
        """Parse a single raw input into a PARSED (or, on failure, PARSE_ERROR) record.

        Dispatch by file type → per-page markdown, then assemble via PageMap.from_page_texts. A
        native fault mapped by :meth:`guard` to a typed ProviderError, or an explicit
        :class:`_ParseFailure`, is caught here and degraded to a graceful PARSE_ERROR record
        (ADR-0020) — one bad document never aborts the run. A genuinely unexpected bug still
        propagates untyped.
        """
        file_type = raw.provenance.file_type.lower().lstrip(".")
        try:
            page_texts = self._render_to_page_texts(raw, file_type)
        except LatenceError as exc:
            # Both the guarded native fault (ProviderError) and an explicit _ParseFailure land here
            # → a graceful PARSE_ERROR. The message is the framework-typed one (provider + op +
            # native type name), carrying no input bytes / extracted text / OCR content.
            return self._error_record(raw, str(exc))

        assembly = PageMap.from_page_texts(page_texts, separator=self._page_separator)
        text = assembly.text
        n_pages = len(assembly.page_map.pages)
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

    def _render_to_page_texts(self, raw: ParserInput, file_type: str) -> list[str]:
        """Dispatch a raw input to its renderer/extractor and return one markdown string per page.

        The any-file dispatch: PDF/image → rasterise → OCR; office → soffice→pdf→OCR when soffice is
        present else MIT text extraction; unknown → graceful :class:`_ParseFailure` (→ PARSE_ERROR).
        """
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        if file_type in _PDF_EXTENSIONS:
            self._guard_pdf_signature(data)
            images = self._pdf_to_images(data)
            return self._ocr_images(images)
        if file_type in _IMAGE_EXTENSIONS:
            self._guard_image_signature(data)
            return self._ocr_images([self._image_from_bytes(data)])
        if file_type in _OFFICE_EXTENSIONS:
            self._guard_zip_signature(data, file_type)
            return self._office_to_page_texts(data, file_type)
        msg = f"Unsupported file type {file_type!r} for the universal Parser."
        raise _ParseFailure(msg)

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

    # -- offset-0 type-spoof guards (#62) -----------------------------------

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

    def _guard_zip_signature(self, data: bytes, file_type: str) -> None:
        """Reject a declared office file whose bytes are not a ZIP container (offset-0 spoof guard).

        Office Open XML (.docx/.pptx/.xlsx) files are ZIP archives, so a valid one begins with a ZIP
        local-file-header. A declared office file that is not a ZIP is a spoof rejected early,
        before soffice or a reader touches it (defense-in-depth, mirroring the PDF/image guards).
        """
        if not any(data.startswith(magic) for magic in _ZIP_MAGIC):
            msg = (
                f"File declared as {file_type} (Office Open XML) but its bytes are not a ZIP "
                "container (type spoof?)."
            )
            raise _ParseFailure(msg)

    # -- PDF / image rasterisation (pixel-capped) ---------------------------

    def _clamp_scale(self, width_pt: float, height_pt: float, scale: float) -> float:
        """Reduce a PDF render scale so the rasterised bitmap stays under the megapixel budget.

        Mirrors :meth:`LightOnOcrParser._clamp_scale`: pypdfium2 renders ``width_pt × height_pt`` at
        ``scale`` px/pt, so the bitmap is ``(width_pt*scale) × (height_pt*scale)`` pixels; if that
        exceeds :data:`_MAX_RENDER_PIXELS` the scale is reduced by ``sqrt(budget/pixels)`` so the
        area lands at the budget — a giant page can never rasterise to a gigapixel bitmap. A
        degenerate (zero-area) page keeps the requested scale.
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

        Imported inside the method so a missing/malformed dependency never breaks Provider discovery
        and the heavy dep stays out of the core (ADR-0016). A missing dep is a graceful parse
        failure. Each page's render scale is clamped by :meth:`_clamp_scale`. The native pypdfium2
        fault a corrupt PDF raises is mapped by :meth:`guard` to a typed ProviderError (no content),
        then degraded to a PARSE_ERROR by :meth:`_parse_one`. NOT ``fitz`` — pypdfium2 (Apache/BSD).
        """
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pypdfium2 is required to rasterise PDFs."
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
                    images.append(bitmap.to_pil())
            finally:
                pdf.close()
        return images

    def _image_from_bytes(self, data: bytes) -> Any:
        """Decode a single raster image from bytes via Pillow (dep-isolated; guarded; bomb-capped).

        ``Image.MAX_IMAGE_PIXELS`` is tightened to the megapixel budget so a decompression-bomb
        image makes Pillow **raise** during ``.load()`` rather than allocating multi-GB — the
        native fault maps through :meth:`guard` to a typed ProviderError and degrades to a
        PARSE_ERROR. The prior value is restored in a ``finally`` (no side effect). Mirrors lighton.
        """
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pillow is required to decode images."
            raise _ParseFailure(msg) from exc
        previous_cap = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = _MAX_RENDER_PIXELS
        try:
            with self.guard("decode_image"):
                image = Image.open(io.BytesIO(data))
                image.load()  # a decompression-bomb image raises here (bounded, no huge alloc)
                return image.convert("RGB")
        finally:
            Image.MAX_IMAGE_PIXELS = previous_cap

    # -- office dispatch: soffice→pdf→OCR when present, else MIT text extraction ---

    def _office_to_page_texts(self, data: bytes, file_type: str) -> list[str]:
        """Turn an office file into per-page markdown: faithful render via soffice, else text.

        WHEN ``soffice`` is on PATH (and not disabled via ``prefer_soffice=False``): convert
        office→PDF via a subprocess (no shell, timeout, temp dir), then rasterise+OCR the PDF — a
        faithful page render. A soffice failure (timeout / non-zero exit) degrades to the MIT
        text-extraction path so a flaky/absent LibreOffice never fails the document. When soffice is
        absent: python-docx / openpyxl / python-pptx (MIT) text extraction directly (no OCR).
        """
        if self._prefer_soffice and self._soffice_path() is not None:
            try:
                pdf_bytes = self._soffice_convert_to_pdf(data, file_type)
            except _ParseFailure:
                # A soffice failure is not fatal: fall through to the text path (graceful degrade).
                pdf_bytes = None
            if pdf_bytes is not None:
                self._guard_pdf_signature(pdf_bytes)
                return self._ocr_images(self._pdf_to_images(pdf_bytes))
        return self._office_text_page_texts(data, file_type)

    @staticmethod
    def _soffice_path() -> str | None:
        """The absolute path to ``soffice`` on PATH, or ``None`` (soffice is OPTIONAL)."""
        return shutil.which("soffice") or shutil.which("libreoffice")

    def _soffice_convert_to_pdf(self, data: bytes, file_type: str) -> bytes:
        """Convert office bytes → PDF bytes via a hardened LibreOffice-headless subprocess.

        Run with an explicit **argument list** (``shell=False`` — no shell, no injection), a hard
        **timeout**, and an isolated temp dir (soffice writes the output PDF beside the input). A
        timeout, a non-zero exit, or a missing output file is a graceful :class:`_ParseFailure`
        (caught by :meth:`_office_to_page_texts`, which then degrades to the text path). LibreOffice
        is an EXTERNAL MPL-2.0 tool invoked as a subprocess — not linked or vendored.
        """
        soffice = self._soffice_path()
        if soffice is None:  # pragma: no cover - guarded by the caller's presence check
            msg = "soffice is not on PATH."
            raise _ParseFailure(msg)
        with tempfile.TemporaryDirectory(prefix="latence-render-") as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / f"input.{file_type}"
            src.write_bytes(data)
            profile_dir = tmp_path / "profile"
            # -env:UserInstallation isolates the LibreOffice profile to the temp dir so concurrent
            # runs don't collide on the shared user profile (a known soffice-headless footgun).
            argv = [
                soffice,
                "--headless",
                "--nolockcheck",
                "--nodefault",
                "--norestore",
                f"-env:UserInstallation=file://{profile_dir}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(tmp_path),
                str(src),
            ]
            try:
                completed = subprocess.run(  # noqa: S603 - explicit argv, no shell, hardcoded flags
                    argv,
                    capture_output=True,
                    timeout=self._soffice_timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                msg = f"soffice conversion timed out after {self._soffice_timeout_s}s."
                raise _ParseFailure(msg) from exc
            except OSError as exc:
                # A native exec fault (e.g. the binary vanished mid-run) — no stderr/content leaked.
                msg = f"soffice conversion failed to start (native {type(exc).__name__})."
                raise _ParseFailure(msg) from exc
            if completed.returncode != 0:
                # Deliberately do NOT interpolate stderr — it can echo the input path/content.
                msg = f"soffice conversion exited {completed.returncode}."
                raise _ParseFailure(msg)
            out_pdf = tmp_path / "input.pdf"
            if not out_pdf.is_file():
                msg = "soffice produced no PDF output."
                raise _ParseFailure(msg)
            return out_pdf.read_bytes()

    def _office_text_page_texts(self, data: bytes, file_type: str) -> list[str]:
        """MIT text extraction (no soffice, no OCR): docx paragraphs / xlsx sheets / pptx slides.

        The graceful fallback when soffice is absent. Each reader is imported lazily (dep-isolated,
        ADR-0016) and its native fault is guarded to a typed ProviderError. The page split is honest
        per format: docx → one page (the paragraph flow has no page layout without a renderer); xlsx
        → one page per worksheet; pptx → one page per slide. A missing reader dep is a graceful
        parse failure.
        """
        if file_type in _DOCX_EXTENSIONS:
            return [self._docx_text(data)]
        if file_type in _XLSX_EXTENSIONS:
            return self._xlsx_page_texts(data)
        if file_type in _PPTX_EXTENSIONS:
            return self._pptx_page_texts(data)
        # Unreachable (the dispatch already filtered to office types), but explicit is safer.
        msg = f"No text extractor for {file_type!r}."
        raise _ParseFailure(msg)

    def _docx_text(self, data: bytes) -> str:
        """Extract docx paragraph text as markdown via python-docx (MIT). One page (no layout)."""
        try:
            import docx
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "python-docx is required for docx text extraction."
            raise _ParseFailure(msg) from exc
        with self.guard("docx_text"):
            document = docx.Document(io.BytesIO(data))
            paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _xlsx_page_texts(self, data: bytes) -> list[str]:
        """Extract xlsx cell text as one markdown page per worksheet via openpyxl (MIT)."""
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "openpyxl is required for xlsx text extraction."
            raise _ParseFailure(msg) from exc
        with self.guard("xlsx_text"):
            workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            pages: list[str] = []
            for sheet in workbook.worksheets:
                lines: list[str] = []
                for row in sheet.iter_rows(values_only=True):
                    cells = ["" if c is None else str(c) for c in row]
                    if any(cell.strip() for cell in cells):
                        lines.append("\t".join(cells).rstrip())
                header = f"# {sheet.title}"
                body = "\n".join(lines)
                pages.append(f"{header}\n\n{body}" if body else header)
            workbook.close()
        return pages or [""]

    def _pptx_page_texts(self, data: bytes) -> list[str]:
        """Extract pptx text as one markdown page per slide via python-pptx (MIT)."""
        try:
            from pptx import Presentation
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "python-pptx is required for pptx text extraction."
            raise _ParseFailure(msg) from exc
        with self.guard("pptx_text"):
            presentation = Presentation(io.BytesIO(data))
            pages: list[str] = []
            for slide in presentation.slides:
                fragments: list[str] = []
                for shape in slide.shapes:
                    text = getattr(shape, "text", "")
                    if isinstance(text, str) and text.strip():
                        fragments.append(text)
                pages.append("\n\n".join(fragments))
        return pages or [""]

    # -- OCR delegation to the configured OCR Provider ----------------------

    def _ocr_images(self, images: list[Any]) -> list[str]:
        """Delegate each page image to the configured OCR Provider → one markdown string per page.

        Reuses the OCR Provider through its ``Parser.parse`` seam (no private method reached into,
        no seam forked): each page image is encoded to PNG bytes and wrapped as a one-page ``png``
        :class:`~latence_core.contracts.ParserInput`; the OCR Provider parses it and returns a
        :class:`~latence_core.contracts.DocumentRecord` whose ``content`` is that page's markdown. A
        page whose OCR record is a PARSE_ERROR contributes empty text (recorded, never a crash).

        A zero-page source (e.g. an empty PDF) yields a single empty page (mirrors the pdfplumber /
        lighton empty-page discipline). No OCR configured → a graceful :class:`_ParseFailure` (a
        page that needs pixels-OCR cannot be read without an OCR Provider).
        """
        if not images:
            return [""]
        ocr = self._load_ocr_provider()
        page_texts: list[str] = []
        for image in images:
            page_input = self._page_image_input(image)
            records = list(ocr.parse([page_input]))
            page_texts.append(self._page_text_from_records(records))
        return page_texts

    def _page_text_from_records(self, records: list[DocumentRecord]) -> str:
        """The markdown a page's OCR produced: the PARSED record's content, else empty."""
        for record in records:
            if record.disposition is Disposition.PARSED:
                return record.content
        return ""

    def _page_image_input(self, image: Any) -> ParserInput:
        """Encode a Pillow page image to PNG bytes and wrap it as a one-page ``png`` ParserInput.

        The neutral, lossless hand-off contract to the OCR Provider: a page image is a one-page
        ``png`` raw document. Provenance/Classification are synthetic (a per-page carrier), never
        surfaced downstream — only the OCR'd ``content`` is used, then reassembled under the REAL
        document's Provenance by :meth:`_parse_one`.
        """
        from latence_core.contracts import ParserInput

        buffer = io.BytesIO()
        with self.guard("encode_page"):
            image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        provenance = Provenance(
            source_uri="memory://render-page",
            document_id="render-page",
            file_name="page.png",
            file_type="png",
            file_size=len(png_bytes),
        )
        classification = Classification(language="und")
        return ParserInput(
            provenance=provenance,
            classification=classification,
            content=png_bytes,
            config={},
        )

    def _load_ocr_provider(self) -> Parser:
        """Resolve + instantiate the configured OCR Provider from the registry (cached, lazy).

        Deferred so an office-text-only run never resolves an OCR Provider and Provider discovery
        stays light. A missing ``ocr`` config, an unknown Provider name, a Provider whose import
        fails (a missing heavy dep, ADR-0016), or a resolved object that is not a Parser is a
        graceful :class:`_ParseFailure` (→ a PARSE_ERROR for the page needing pixels-OCR), never
        a crash. The instantiated Provider is cached so a multi-page document loads it once.
        """
        if self._ocr_provider is not None:
            return self._ocr_provider
        if self._ocr_spec is None:
            msg = (
                "No OCR Provider configured ('ocr: {provider: parser.lighton|parser.glm}') but a "
                "page requires pixels-OCR (PDF/image/soffice-rendered office)."
            )
            raise _ParseFailure(msg)
        name = self._ocr_spec["provider"]
        try:
            provider_cls = self._registry.load(name)
        except (KeyError, ImportError, TypeError) as exc:
            # Unknown name / import failure (missing heavy dep) / not-a-class — a config-level
            # failure surfaced as a graceful parse failure, no content leaked.
            msg = f"OCR Provider {name!r} could not be loaded (native {type(exc).__name__})."
            raise _ParseFailure(msg) from exc
        with self.guard("load_ocr"):
            instance = provider_cls(self._ocr_spec["config"])
        if not _is_parser(instance):
            msg = f"OCR Provider {name!r} does not satisfy the Parser capability."
            raise _ParseFailure(msg)
        provider = cast("Parser", instance)
        self._ocr_provider = provider
        return provider

    # -- classification -----------------------------------------------------

    def _classify(self, raw: ParserInput, text: str) -> Classification:
        """Attach Classification at Parse: language from the parsed text (shared core heuristic).

        Language is auto-detected from the parsed markdown via the shared core heuristic. Category
        and sensitivity are the config override, else the value inherited from the ParserInput's
        Classification (the Source's stamp is not dropped) — matching parser.pdfplumber / lighton.
        """
        inherited = raw.classification
        language = detect_language(text) if text else inherited.language
        category = self._category if self._category is not None else inherited.category
        sensitivity = (
            str(self._sensitivity) if self._sensitivity is not None else inherited.sensitivity
        )
        return Classification(language=language, category=category, sensitivity=sensitivity)


def _is_parser(obj: object) -> bool:
    """Structural Parser check (has a callable ``parse``) — the OCR sub-provider must be a Parser.

    Imported lazily and checked structurally rather than via ``isinstance(obj, Parser)`` so this
    stays a pure duck-typed check with no import-time coupling; a real OCR Provider (lighton/glm)
    and a stub both satisfy it.
    """
    return callable(getattr(obj, "parse", None))


def size_after_parse(record: DocumentRecord) -> int:
    """Assembled-markdown size in bytes — the Provenance 'size-after-parse' (S2 observability).

    Mirrors the pdfplumber / lighton Parser helper so a Pipeline that swaps this Provider in gets
    the same observability signal the Runner sums into the Quality Report's ``bytes_after_parse``.
    """
    return len(record.content.encode("utf-8"))


class _ParseFailure(LatenceError):
    """A recoverable, framework-typed parse failure → a PARSE_ERROR record (ADR-0020).

    A :class:`~latence_core.errors.LatenceError` subclass so it flows through the same typed-error
    boundary the guarded native fault does: :meth:`UniversalRenderParser._parse_one` catches
    ``LatenceError`` and turns either into a graceful PARSE_ERROR record. It carries only a static
    reason (unsupported type, spoofed signature, soffice failure, missing/unknown OCR Provider) —
    never input content / extracted text / OCR content.
    """

    category = "provider"
