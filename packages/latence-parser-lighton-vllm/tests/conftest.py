"""Offline test doubles for the served-vLLM LightOn-OCR Parser — the FAITHFUL vLLM mock + rasteriser.

Two anti-false-green rules this slice honours:

1. **The async vLLM mock mirrors the REAL vLLM OpenAI chat-completions response shape** — NOT a
   fictional API. ``AsyncOpenAI().chat.completions.create(...)`` is a **coroutine** returning an
   object with ``.choices[0].message.content`` (the page's markdown), exactly the shape the real vLLM
   server returns (the same shape ``relation.llm`` / ``embedding.endpoint`` mock against). The mock
   decodes the page image the Provider POSTed back to that page's intended markdown (it reads the
   base64 ``data:`` URL the Provider built from the rasterised image), so the Provider's REAL
   rasterise → data-URL → POST → decode path is exercised end to end — a Provider that mangles the
   image or drops a page FAILS. The mock records ``max_in_flight`` (the semaphore bound) and honours a
   per-call ``asyncio.sleep`` delay so a timing test can prove the concurrency is real.

2. **The rasteriser is the SHARED one, run for real offline** — the fake ``pypdfium2`` / ``PIL``
   modules make the REAL :class:`~latence_parser_lighton.rasterize.PageRasteriser` code path run
   (render-scale clamp, bomb cap, 1540px downscale, spoof guards) without the heavy native deps
   (ADR-0016). A page image carries its intended markdown as a marker so the round-trip is
   deterministic and inspectable; an over-budget fixture image raises a decompression-bomb error on
   ``load()`` exactly like real Pillow, so the graceful-degradation path is real, not stubbed green.

These fakes are injected into ``sys.modules`` (``openai``, ``pypdfium2``, ``PIL``) so the Provider's
deferred imports pick them up in-process. The real deps are not installed in the CPU-first env; the
real OCR is validated against a live vLLM server on a CUDA pod (scripts/validate-lighton-vllm-on-gpu.md).
"""

from __future__ import annotations

import asyncio
import base64
import sys
import types
from typing import Any

import pytest
from _vllm_fakes import FAKE_PAGE_MARKER, StatusError, VllmScript

__all__ = ["FAKE_PAGE_MARKER", "StatusError", "VllmScript"]

# The default fake page/image PIXEL size — deliberately UNDER the 1540 longest-side cap so the plain
# fixtures are NOT downscaled; the downscale test drives an explicitly-oversized fake image.
_FAKE_IMG_W = 800
_FAKE_IMG_H = 1000


# ---------------------------------------------------------------------------
# The faithful async vLLM OpenAI mock (choices[0].message.content = markdown).
# ---------------------------------------------------------------------------


def _decode_posted_markdown(messages: list[dict[str, Any]]) -> str:
    """Recover the page markdown from the OpenAI multimodal message the Provider POSTed.

    Faithful to the REAL request shape: ``content`` is a list with an ``image_url`` part whose
    ``url`` is a ``data:image/png;base64,<...>`` URL. The fake rasteriser encoded the page marker
    (``FAKE_PAGE::<md>``) as the "PNG" payload, so decoding the base64 recovers it — the Provider's
    real rasterise → data-URL → POST path is genuinely exercised.
    """
    content = messages[0]["content"]
    image_part = next(p for p in content if p.get("type") == "image_url")
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,"), f"not a PNG data URL: {url[:40]}"
    b64 = url.split(",", 1)[1]
    raw = base64.b64decode(b64).decode("utf-8")
    assert raw.startswith(FAKE_PAGE_MARKER), f"payload missing marker: {raw[:40]}"
    return raw[len(FAKE_PAGE_MARKER) :]


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Faithful async ``chat.completions``: a coroutine returning the real vLLM response shape."""

    async def create(self, **kwargs: Any) -> _FakeResponse:
        VllmScript.calls += 1
        VllmScript.in_flight += 1
        VllmScript.max_in_flight = max(VllmScript.max_in_flight, VllmScript.in_flight)
        try:
            markdown = _decode_posted_markdown(kwargs["messages"])
            await asyncio.sleep(VllmScript.delay_for.get(markdown, VllmScript.default_delay))
            steps = VllmScript.overrides.get(markdown)
            if steps:
                kind, payload = steps.pop(0)
                if kind == "raise":
                    assert isinstance(payload, BaseException)
                    raise payload
                return _FakeResponse(str(payload))
            # The faithful default: the endpoint transcribes the page image back to its markdown.
            return _FakeResponse(markdown)
        finally:
            VllmScript.in_flight -= 1


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        self.chat = _FakeChat()


# ---------------------------------------------------------------------------
# The faithful offline rasteriser fakes (pypdfium2 + PIL) — real PageRasteriser path.
# ---------------------------------------------------------------------------


class _FakeImage:
    """A minimal fake Pillow image: carries the OCR marker + a ``.size``; supports ``.resize``/save.

    ``.save(buffer, format="PNG")`` writes the marker bytes so the Provider's ``image_to_data_url``
    (base64 of the saved PNG) round-trips the page markdown to the fake vLLM — the real
    rasterise→data-URL→POST path. ``.size``/``.resize`` let the shared 1540px downscale run for real.
    """

    def __init__(self, marker: str, size: tuple[int, int] = (_FAKE_IMG_W, _FAKE_IMG_H)) -> None:
        self._marker = marker
        self.size = size
        self.mode = "RGB"
        self.resized_to: tuple[int, int] | None = None

    def resize(self, size: tuple[int, int], _resample: Any = None) -> _FakeImage:
        out = _FakeImage(self._marker, size=size)
        out.resized_to = size
        return out

    def convert(self, _mode: str) -> _FakeImage:
        return self

    def save(self, buffer: Any, format: str = "PNG") -> None:  # noqa: A002 - Pillow's kw name
        buffer.write(self._marker.encode("utf-8"))


class _FakePage:
    """A fake pypdfium2 page: ``get_size()`` reports its point size; ``render().to_pil()``."""

    def __init__(self, marker: str, width_pt: float = 612.0, height_pt: float = 792.0) -> None:
        self._marker = marker
        self._width_pt = width_pt
        self._height_pt = height_pt

    def get_size(self) -> tuple[float, float]:
        return (self._width_pt, self._height_pt)

    def render(self, scale: float = 1.0, **_kwargs: Any) -> _FakePage:
        return self

    def to_pil(self) -> _FakeImage:
        return _FakeImage(self._marker)


class _FakePdfDocument:
    """A fake pypdfium2 PdfDocument: iterates fake pages parsed from the fixture bytes."""

    def __init__(self, buffer: Any) -> None:
        data = buffer.read() if hasattr(buffer, "read") else buffer
        text = bytes(data).decode("utf-8", errors="replace")
        self._width_pt = 612.0
        self._height_pt = 792.0
        if text.startswith("SIZE:"):
            header, _, text = text.partition("\n")
            dims = header[len("SIZE:") :]
            w_str, _, h_str = dims.partition("x")
            self._width_pt = float(w_str)
            self._height_pt = float(h_str)
        _, _, body = text.partition("PAGES:\n")
        self._pages = body.split("\n---PAGE---\n") if body else [""]

    def __iter__(self) -> Any:
        for md in self._pages:
            yield _FakePage(
                FAKE_PAGE_MARKER + md, width_pt=self._width_pt, height_pt=self._height_pt
            )

    def close(self) -> None:
        pass


class _FakeDecompressionBombError(Exception):
    """Stand-in for ``PIL.Image.DecompressionBombError`` (raised when a fixture image is over budget)."""


class _FakePilImage:
    """A minimal fake PIL.Image.Image: open().load()/convert() round-trip the marker string."""

    def __init__(
        self,
        marker: str,
        pixels: int,
        image_module: Any,
        size: tuple[int, int] = (_FAKE_IMG_W, _FAKE_IMG_H),
    ) -> None:
        self._marker = marker
        self._pixels = pixels
        self._image_module = image_module
        self._size = size

    def load(self) -> None:
        cap = self._image_module.MAX_IMAGE_PIXELS
        if cap is not None and self._pixels > cap:
            msg = f"Image size ({self._pixels} pixels) exceeds limit of {cap} pixels"
            raise _FakeDecompressionBombError(msg)

    def convert(self, _mode: str) -> _FakeImage:
        return _FakeImage(self._marker, size=self._size)


class _FakeResampling:
    LANCZOS = "LANCZOS"


def _install_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")
    module.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _install_pdfium(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("pypdfium2")
    module.PdfDocument = _FakePdfDocument  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", module)


def _install_pil(monkeypatch: pytest.MonkeyPatch) -> None:
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.MAX_IMAGE_PIXELS = 89478485  # type: ignore[attr-defined]  # Pillow's real default
    image_mod.DecompressionBombError = _FakeDecompressionBombError  # type: ignore[attr-defined]
    image_mod.Resampling = _FakeResampling  # type: ignore[attr-defined]

    def _open(buffer: Any) -> _FakePilImage:
        data = buffer.read() if hasattr(buffer, "read") else buffer
        text = bytes(data).decode("utf-8", errors="replace")
        pixels = 1024
        size = (_FAKE_IMG_W, _FAKE_IMG_H)
        if "PIXELS:" in text:
            _, _, after = text.partition("PIXELS:")
            count_str, _, text = after.partition("|")
            pixels = int(count_str)
        if "DIMS:" in text:
            _, _, after = text.partition("DIMS:")
            dims, _, text = after.partition("|")
            w_str, _, h_str = dims.partition("x")
            size = (int(w_str), int(h_str))
        _, _, md = text.partition("IMG:")
        return _FakePilImage(
            FAKE_PAGE_MARKER + md, pixels=pixels, image_module=image_mod, size=size
        )

    image_mod.open = _open  # type: ignore[attr-defined]
    pil.Image = image_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


@pytest.fixture()
def fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the faithful async vLLM mock + the offline rasteriser fakes (openai/pypdfium2/PIL)."""
    _install_openai(monkeypatch)
    _install_pdfium(monkeypatch)
    _install_pil(monkeypatch)
    VllmScript.reset()


@pytest.fixture()
def make_pdf() -> Any:
    """A builder emitting fixture PDF bytes the fake rasteriser splits into pages (valid %PDF- magic)."""

    def _build(pages: list[str], *, oversized: bool = False) -> bytes:
        body = "\n---PAGE---\n".join(pages)
        prefix = "%PDF-1.4\n"
        if oversized:
            prefix += "SIZE:100000x100000\n"
        return (prefix + "PAGES:\n" + body).encode("utf-8")

    return _build


@pytest.fixture()
def make_image() -> Any:
    """A builder emitting fixture PNG bytes (valid PNG magic) carrying one page's markdown."""

    def _build(
        markdown: str,
        *,
        pixels: int | None = None,
        dims: tuple[int, int] | None = None,
    ) -> bytes:
        head = b"\x89PNG\r\n\x1a\n"
        prefix = f"PIXELS:{pixels}|".encode() if pixels is not None else b""
        if dims is not None:
            prefix += f"DIMS:{dims[0]}x{dims[1]}|".encode()
        return head + prefix + b"IMG:" + markdown.encode("utf-8")

    return _build
