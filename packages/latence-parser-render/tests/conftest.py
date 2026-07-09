"""Faithful offline doubles for the universal Parser: a tiny in-memory renderer + a stub OCR.

The heavy render/office deps (``pypdfium2``/``Pillow``/``python-docx``/``openpyxl``/``python-pptx``)
are NOT installed in the CPU-first dev/CI env (ADR-0016). These doubles stand in for them WITHOUT
weakening the contract under test:

* :class:`FakePageImage` mirrors the ONLY thing the Provider does with a rendered page image — it
  ``.save(buffer, format="PNG")`` — writing deterministic bytes that encode the page's intended
  text, so the round-trip renderer→PNG-bytes→OCR is genuinely exercised (not stubbed away).
* :class:`StubOcrParser` is a REAL Parser Provider (satisfies the Capability structurally) that
  decodes those deterministic PNG bytes back to the page markdown — the faithful stand-in for a
  pod-validated OCR model, resolved through the Provider's registry seam exactly as
  ``parser.lighton``/``parser.glm`` would be.

So a test drives the real dispatch (PDF/image/office), the real security guards, the real OCR
delegation seam, and the real PageMap assembly — only the native rasteriser bytes and the OCR model
are faithfully faked. This is the anti-false-green discipline the sibling OCR parsers carry.

The reusable builders/doubles are exposed as **fixtures** (the repo test convention — see
``latence-parser-lighton``'s conftest), not importable module functions, so cross-test sharing works
without a ``tests`` package on ``sys.path``.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    ParserInput,
    Provenance,
)
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

# A tiny magic + length-prefixed framing so the stub OCR can round-trip a page's text through the
# "PNG bytes" the Provider hands it — deterministic, no real Pillow needed.
_FAKE_PNG_MAGIC = b"FAKEPNG\x00"


def encode_fake_png(text: str) -> bytes:
    """Encode page text into deterministic 'PNG' bytes (magic + len-prefixed UTF-8)."""
    payload = text.encode("utf-8")
    return _FAKE_PNG_MAGIC + struct.pack(">I", len(payload)) + payload


def decode_fake_png(data: bytes) -> str:
    """Decode the page text a :func:`encode_fake_png` produced (the stub OCR's 'model')."""
    if not data.startswith(_FAKE_PNG_MAGIC):
        msg = "not a fake-PNG page image"
        raise ValueError(msg)
    (length,) = struct.unpack(">I", data[len(_FAKE_PNG_MAGIC) : len(_FAKE_PNG_MAGIC) + 4])
    start = len(_FAKE_PNG_MAGIC) + 4
    return data[start : start + length].decode("utf-8")


class FakePageImage:
    """A rendered page image stand-in exposing the one method the Provider uses: ``.save``.

    The Provider encodes each rendered page via ``image.save(buffer, format="PNG")`` before handing
    it to the OCR Provider. This fake writes deterministic bytes carrying the page's intended text,
    so the renderer→bytes→OCR round-trip is genuinely exercised (an un-wired OCR would fail).
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def save(self, buffer: Any, format: str = "PNG") -> None:  # noqa: A002 - Pillow's kw name
        assert format == "PNG"
        buffer.write(encode_fake_png(self._text))


class StubOcrParser:
    """A faithful offline OCR Provider: decode a fake-PNG page image back to its markdown.

    A REAL Parser Provider (structural Capability) so the Provider's OCR-delegation seam is exercised
    exactly as it would be against ``parser.lighton``/``parser.glm``: it receives a one-page ``png``
    :class:`ParserInput`, 'OCRs' the pixels (here: decodes the fake-PNG bytes), and returns a PARSED
    :class:`DocumentRecord` whose ``content`` is that page's markdown. Undecodable bytes → a graceful
    PARSE_ERROR (mirroring a real OCR degrade), so the Provider's empty-on-error path is covered too.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="either",
            memory_mb=16,
            model_id="stub-ocr",
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-09",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        for raw in inputs:
            data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
            try:
                text = decode_fake_png(data)
                disposition = Disposition.PARSED
                error = None
            except (ValueError, struct.error):
                text = ""
                disposition = Disposition.PARSE_ERROR
                error = "stub OCR could not decode the page image"
            assembly = PageMap.from_page_texts([text])
            yield DocumentRecord(
                record_id=raw.provenance.document_id,
                provenance=raw.provenance.model_copy(
                    update={"char_start": 0, "char_end": assembly.page_map.total_chars}
                ),
                classification=raw.classification,
                content=assembly.text if disposition is Disposition.PARSED else "",
                media_type="text/markdown",
                page_map=assembly.page_map if disposition is Disposition.PARSED else None,
                disposition=disposition,
                error=error,
            )


class StubRegistry(ProviderRegistry):
    """A registry whose ``load`` resolves a fixed name to the stub OCR (no real install needed)."""

    def __init__(self, mapping: dict[str, type]) -> None:
        super().__init__()
        self._mapping = mapping

    def load(self, name: str) -> type:
        if name in self._mapping:
            return self._mapping[name]
        return super().load(name)


def make_fake_pdf(pages: list[str]) -> bytes:
    """Build a spoof-guard-passing fake PDF whose payload the patched renderer splits into pages."""
    return b"%PDF-" + "|||".join(pages).encode("utf-8")


def make_fake_png(text: str) -> bytes:
    """Build a spoof-guard-passing fake PNG whose payload the patched renderer decodes to text."""
    return b"\x89PNG\r\n\x1a\n" + text.encode("utf-8")


def parser_input(content: bytes, *, file_name: str, file_type: str) -> ParserInput:
    """A canonical ParserInput for a raw source document (Provenance + Classification stamped)."""
    return ParserInput(
        provenance=Provenance(
            source_uri=f"file://corpus/{file_name}",
            file_name=file_name,
            file_type=file_type,
            file_size=len(content),
            document_id=f"sha256:{file_name}",
        ),
        classification=Classification(language="und", sensitivity="internal"),
        content=content,
    )


# --- fixtures (the repo convention: reusable builders/doubles handed to tests) ---


@pytest.fixture
def stub_ocr_registry() -> ProviderRegistry:
    """A registry resolving ``parser.lighton``/``parser.glm``/``ocr.stub`` to the stub OCR."""
    return StubRegistry(
        {"parser.lighton": StubOcrParser, "parser.glm": StubOcrParser, "ocr.stub": StubOcrParser}
    )


@pytest.fixture
def render_config(stub_ocr_registry: ProviderRegistry) -> dict[str, Any]:
    """The base ``parser.render`` config wired to the stub OCR via the registry override seam."""
    return {
        "ocr": {"provider": "parser.lighton", "config": {}},
        "ocr_registry": stub_ocr_registry,
    }


@pytest.fixture
def patch_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the native rasterisers so a PDF/image renders to :class:`FakePageImage`s offline.

    ``_pdf_to_images`` returns one fake page image per ``|||``-separated segment of the (fake) PDF
    bytes' payload; ``_image_from_bytes`` returns a single fake page image carrying the decoded
    payload. The security guards, dispatch, OCR delegation, and PageMap assembly all run for real;
    only the native pypdfium2/Pillow byte-crunching is replaced.
    """
    from latence_parser_render import provider as prov

    def fake_pdf_to_images(self: prov.UniversalRenderParser, data: bytes) -> list[Any]:
        body = data[len(b"%PDF-") :].decode("utf-8")
        return [FakePageImage(seg) for seg in body.split("|||")]

    def fake_image_from_bytes(self: prov.UniversalRenderParser, data: bytes) -> Any:
        for magic in (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM"):
            if data.startswith(magic):
                return FakePageImage(data[len(magic) :].decode("utf-8"))
        return FakePageImage(data.decode("utf-8", errors="replace"))

    monkeypatch.setattr(prov.UniversalRenderParser, "_pdf_to_images", fake_pdf_to_images)
    monkeypatch.setattr(prov.UniversalRenderParser, "_image_from_bytes", fake_image_from_bytes)


# --- native-path doubles: run the REAL _pdf_to_images / _image_from_bytes bodies ---
#
# ``patch_renderer`` above monkeypatches the Provider's ``_pdf_to_images`` / ``_image_from_bytes``
# methods, which is enough to exercise dispatch + OCR delegation but DELIBERATELY skips the native
# rasteriser bodies — so the pixel-bomb DoS cap (``_clamp_scale`` + ``Image.MAX_IMAGE_PIXELS``) is
# never actually run. The fixtures below instead inject faithful fake ``pypdfium2`` / ``PIL.Image``
# modules into ``sys.modules`` (the sibling ``parser.lighton`` discipline), so the REAL bodies run:
# the real clamp math drives a real ``page.render(scale=...)``, and the real ``MAX_IMAGE_PIXELS``
# set-then-restore-in-finally wraps a fake Pillow that HONORS the cap — an over-budget image raises
# a ``DecompressionBombError`` on ``.load()`` exactly like real Pillow. This makes the security
# guard's protective behavior (bomb → PARSE_ERROR; clamp → shrink; cap restored) fail red if broken.


class _FakeDecompressionBombError(Exception):
    """Stand-in for ``PIL.Image.DecompressionBombError`` — raised when an image is over budget."""


class _FakeNativeImage:
    """A fake Pillow image: ``.load()`` honors ``MAX_IMAGE_PIXELS``; ``.convert``/``.save`` round-trip.

    ``load()`` reads the module-level ``MAX_IMAGE_PIXELS`` cap the Provider tightened and raises a
    :class:`_FakeDecompressionBombError` when the declared pixel count exceeds it — exactly the
    decompression-bomb degradation path. ``.save(buffer, "PNG")`` emits the same deterministic bytes
    :class:`FakePageImage` does, so the render→bytes→OCR round-trip still works for in-budget images.
    """

    def __init__(self, text: str, pixels: int, image_module: Any) -> None:
        self._text = text
        self._pixels = pixels
        self._image_module = image_module

    def load(self) -> None:
        cap = self._image_module.MAX_IMAGE_PIXELS
        if cap is not None and self._pixels > cap:
            msg = f"Image size ({self._pixels} pixels) exceeds limit of {cap} pixels"
            raise self._image_module.DecompressionBombError(msg)

    def convert(self, mode: str) -> _FakeNativeImage:  # noqa: ARG002 - Pillow's mode arg
        return self

    def save(self, buffer: Any, format: str = "PNG") -> None:  # noqa: A002 - Pillow's kw name
        assert format == "PNG"
        buffer.write(encode_fake_png(self._text))


class _FakeNativeBitmap:
    """A pypdfium2 bitmap stand-in: ``.to_pil()`` yields the in-budget page image (post-clamp)."""

    def __init__(self, text: str, pixels: int, image_module: Any) -> None:
        self._text = text
        self._pixels = pixels
        self._image_module = image_module

    def to_pil(self) -> _FakeNativeImage:
        return _FakeNativeImage(self._text, self._pixels, self._image_module)


class _FakeNativePage:
    """A pypdfium2 page stand-in: reports a point size so the Provider's ``_clamp_scale`` runs.

    ``render(scale=...)`` records the clamped scale the Provider computed and produces a bitmap whose
    declared pixel count is ``(width_pt*scale) × (height_pt*scale)`` — so an oversized page forces a
    genuine clamp (the REDUCE branch) and the resulting bitmap lands UNDER the budget, never crashing.
    """

    def __init__(self, text: str, width_pt: float, height_pt: float, image_module: Any) -> None:
        self._text = text
        self._width_pt = width_pt
        self._height_pt = height_pt
        self._image_module = image_module
        self.last_scale: float | None = None

    def get_size(self) -> tuple[float, float]:
        return (self._width_pt, self._height_pt)

    def render(self, scale: float = 1.0, **_kwargs: Any) -> _FakeNativeBitmap:
        self.last_scale = scale
        pixels = int(self._width_pt * scale * self._height_pt * scale)
        return _FakeNativeBitmap(self._text, pixels, self._image_module)


class _FakeNativePdfDocument:
    """A pypdfium2 ``PdfDocument`` stand-in over the fake PDF framing (``%PDF-`` + optional SIZE)."""

    #: The most recently constructed document, so a test can read back each page's clamped scale.
    last: _FakeNativePdfDocument | None = None

    def __init__(self, stream: Any, image_module: Any) -> None:
        data = stream.read() if hasattr(stream, "read") else stream
        body = bytes(data)[len(b"%PDF-") :].decode("utf-8")
        width_pt, height_pt = 595.0, 842.0  # A4 points, unless a SIZE header overrides
        if body.startswith("SIZE:"):
            header, _, body = body.partition("\n")
            dims = header[len("SIZE:") :]
            w_str, _, h_str = dims.partition("x")
            width_pt, height_pt = float(w_str), float(h_str)
        self.pages = [
            _FakeNativePage(seg, width_pt, height_pt, image_module) for seg in body.split("|||")
        ]
        _FakeNativePdfDocument.last = self

    def __iter__(self) -> Iterator[_FakeNativePage]:
        return iter(self.pages)

    def close(self) -> None:
        return None


def make_oversized_pdf(pages: list[str], *, width_pt: float, height_pt: float) -> bytes:
    """A spoof-guard-passing fake PDF whose pages declare an oversized point size (pixel-bomb)."""
    header = f"SIZE:{width_pt}x{height_pt}\n"
    return b"%PDF-" + (header + "|||".join(pages)).encode("utf-8")


def make_bomb_png(*, pixels: int) -> bytes:
    """A spoof-guard-passing fake PNG that declares ``pixels`` — an over-budget decompression bomb."""
    return b"\x89PNG\r\n\x1a\n" + f"BOMB:{pixels}".encode()


@pytest.fixture
def native_render_modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject faithful fake ``pypdfium2`` + ``PIL.Image`` into ``sys.modules`` (real bodies run).

    Unlike :func:`patch_renderer`, this does NOT replace the Provider's methods — it fakes the heavy
    native modules the REAL ``_pdf_to_images`` / ``_image_from_bytes`` import, so the pixel-bomb cap
    (``_clamp_scale`` + the ``MAX_IMAGE_PIXELS`` set/restore) genuinely executes. The fake ``PIL``
    honors the cap (raises on an over-budget image); the fake ``pypdfium2`` reports the page point
    size so the clamp runs. Returns the fake ``PIL.Image`` module so a test can assert cap restore.
    """
    import sys
    import types

    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.MAX_IMAGE_PIXELS = 89_478_485  # type: ignore[attr-defined]  # Pillow's default
    fake_pil_image.DecompressionBombError = _FakeDecompressionBombError  # type: ignore[attr-defined]

    def _open(stream: Any) -> _FakeNativeImage:
        data = stream.read() if hasattr(stream, "read") else stream
        raw = bytes(data)
        for magic in (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM"):
            if raw.startswith(magic):
                body = raw[len(magic) :].decode("utf-8")
                if body.startswith("BOMB:"):
                    return _FakeNativeImage("", int(body[len("BOMB:") :]), fake_pil_image)
                return _FakeNativeImage(body, 1, fake_pil_image)
        return _FakeNativeImage(raw.decode("utf-8", errors="replace"), 1, fake_pil_image)

    fake_pil_image.open = _open  # type: ignore[attr-defined]
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = fake_pil_image  # type: ignore[attr-defined]

    fake_pdfium = types.ModuleType("pypdfium2")
    fake_pdfium.PdfDocument = lambda stream: _FakeNativePdfDocument(stream, fake_pil_image)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_pil_image)
    monkeypatch.setitem(sys.modules, "pypdfium2", fake_pdfium)
    _FakeNativePdfDocument.last = None
    return fake_pil_image


@pytest.fixture
def make_oversized_pdf_bytes() -> Callable[..., bytes]:
    return make_oversized_pdf


@pytest.fixture
def make_bomb_png_bytes() -> Callable[..., bytes]:
    return make_bomb_png


@pytest.fixture
def last_render_document_cls() -> type:
    return _FakeNativePdfDocument


@pytest.fixture
def make_pdf() -> Callable[[list[str]], bytes]:
    return make_fake_pdf


@pytest.fixture
def make_png() -> Callable[[str], bytes]:
    return make_fake_png


@pytest.fixture
def make_input() -> Callable[..., ParserInput]:
    return parser_input


@pytest.fixture
def stub_ocr_cls() -> type:
    return StubOcrParser


@pytest.fixture
def fake_page_image_cls() -> type:
    return FakePageImage
