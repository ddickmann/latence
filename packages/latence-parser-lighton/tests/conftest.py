"""Offline test doubles for the LightOn-OCR Parser — the FAITHFUL generate/decode fakes.

The single rule this slice is built to honor (baked in from the PaddleOCR-VL false green): the
offline model double MUST be **faithful to the real generate/decode contract**, prompt tokens
included. The real API is ``LightOnOcrForConditionalGeneration`` / ``LightOnOcrProcessor`` from
transformers, and per page (verified 2026-07 from https://huggingface.co/lightonai/LightOnOCR-1B-1025):

    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt")
    inputs = inputs.to(device)
    output_ids = model.generate(**inputs, max_new_tokens=..., do_sample=False)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]   # SLICE off the prompt tokens
    output_text = processor.decode(generated_ids, skip_special_tokens=True)  # -> str

The load-bearing fact: HF ``generate`` returns the **full** sequence (prompt + continuation). The
Provider MUST slice the ``inputs["input_ids"].shape[1]`` prompt tokens off before decoding, or the
chat/prompt scaffolding is prepended to the OCR markdown (the exact false-green trap). So these
fakes are faithful to *that*: :class:`_FakeModel.generate` returns a 2-D token tensor whose first
``input_ids.shape[1]`` columns are PROMPT tokens (encoding a visible ``<|prompt-scaffold|>`` marker)
and whose remaining columns are the page markdown — an un-sliced decode would visibly include the
scaffolding and break the offset round-trip. :class:`_FakeProcessor.decode` decodes token ids ->
``str`` (the real ``decode`` shape). ``inputs["input_ids"]`` exposes ``.shape`` so the Provider's
slice is exercised for real. If the Provider ever drops the slice, the fake makes it FAIL — never
pass green.

The pixel-bomb DoS cap is exercised offline too: :class:`_FakePage` reports a per-page point size so
the Provider's render-scale clamp (``_clamp_scale``) genuinely runs, and the fake ``PIL.Image``
honors ``MAX_IMAGE_PIXELS`` — an over-budget fixture image raises a ``DecompressionBombError`` on
``load()`` exactly like real Pillow, so the graceful-degradation path is real, not stubbed green.

These fakes are injected into ``sys.modules`` (``transformers``, ``pypdfium2``, ``PIL``) so the
Provider's deferred imports pick them up in-process — the established gliner/st/endpoint offline
discipline. The real deps are NOT installed in the CPU-first env (ADR-0016); the real OCR is
validated on a CUDA pod (scripts/validate-lighton-on-gpu.md).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

# The page-image sentinel our fake rasteriser emits and the fake OCR "reads". A str carrying the
# page's intended markdown, so the fake decode is deterministic + inspectable (never a real bitmap).
FAKE_PAGE_MARKER = "FAKE_PAGE::"

# The PROMPT-scaffold text the fake tokeniser prepends as "prompt tokens" (the chat template / user
# turn / image-placeholder stand-in). If the Provider forgot to slice it off before decoding, THIS
# text would show up prepended to the OCR markdown — the exact false-green symptom the slice prevents.
FAKE_PROMPT_SCAFFOLD = "<|prompt-scaffold|>"


def _tokenise(text: str) -> list[int]:
    """Encode text -> token ids (one Unicode codepoint per token). Inverse of :func:`_detokenise`."""
    return [ord(ch) for ch in text]


def _detokenise(tokens: list[int]) -> str:
    """Decode token ids -> text (the inverse of :func:`_tokenise`)."""
    return "".join(chr(t) for t in tokens)


# The left-pad token id the fake tokeniser uses to align a batch's prompts to a shared width. A
# codepoint outside the FAKE_PROMPT_SCAFFOLD / markdown ranges so a pad token that ever leaked into a
# decode would be visible. ``decode(skip_special_tokens=True)`` drops it (like real Pillow/HF pads).
_PAD_ID = 0


class _FakeTokenRow:
    """A 1-D slice of token ids (the sliced continuation) that ``decode`` turns back into text."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens


class _FakeTokenRows:
    """A minimal 2-D token tensor stand-in: rows of int token ids, indexable like a torch tensor.

    Supports the operations the Provider + the fake use on real transformers tensors:
      * ``.shape`` -> ``(rows, cols)`` (so ``inputs["input_ids"].shape[1]`` = the padded prompt width),
      * ``output_ids[0, prompt_len:]`` AND ``output_ids[i, prompt_width:]`` 2-D indexing -> a
        :class:`_FakeTokenRow` continuation slice (the single-item and batched Provider paths).

    Each prompt row stashes its OWN ``continuations[r]`` tokens (the markdown the model will append)
    AND its ``true_len`` (the real, unpadded prompt length behind any left-pad), so the fake
    ``generate`` reconstructs each item's full left-padded prompt + continuation independently — no
    global state, and a variable-length batch is faithful (a short page next to a long one).
    """

    def __init__(
        self,
        rows: list[list[int]],
        continuations: list[list[int]] | None = None,
        true_lens: list[int] | None = None,
    ) -> None:
        self._rows = rows
        self.continuations = continuations or [[] for _ in rows]
        self.true_lens = true_lens or [len(r) for r in rows]

    @property
    def shape(self) -> tuple[int, int]:
        cols = len(self._rows[0]) if self._rows else 0
        return (len(self._rows), cols)

    def __getitem__(self, key: Any) -> _FakeTokenRow:
        # The Provider does ``output_ids[i, prompt_width:]`` -> (row index, column slice).
        row_key, col_key = key
        row = self._rows[row_key]
        return _FakeTokenRow(row[col_key])


class _FakeAttentionMask:
    """A 2-D 0/1 attention mask stand-in: 0 on left-pad columns, 1 on an item's real prompt tokens."""

    def __init__(self, mask: list[list[int]]) -> None:
        self._mask = mask

    @property
    def shape(self) -> tuple[int, int]:
        cols = len(self._mask[0]) if self._mask else 0
        return (len(self._mask), cols)

    def __getitem__(self, row: int) -> list[int]:
        return self._mask[row]


class _FakeBatchEncoding:
    """The dict-like handle ``apply_chat_template(return_dict=True)`` returns; ``.to(device)`` no-op.

    Faithful to the real batched contract: it holds ONE OR MORE prompt rows, LEFT-padded to a shared
    width, plus the matching ``attention_mask`` (0 on pad, 1 on real tokens). ``input_ids.shape[1]``
    is the padded width — the exact slice offset every item's continuation begins at under
    left-padding. Each row stashes its own markdown continuation so ``generate`` echoes back each
    item's prompt+continuation independently (variable-length faithful). ``**inputs`` unpacking works
    because it is a mapping over ``input_ids`` + ``attention_mask``; ``.to(device)`` returns self.
    """

    def __init__(self, payloads: list[str]) -> None:
        prompt_scaffold = _tokenise(FAKE_PROMPT_SCAFFOLD)
        # Give each item a DIFFERENT true prompt length so a variable-length batch is exercised: item
        # r's real prompt is the scaffold repeated (r+1) times. A wrong (shared unpadded length or
        # first-item length) slice would then bleed pad/prompt tokens into another item's markdown.
        raw_prompts: list[list[int]] = []
        continuations: list[list[int]] = []
        for r, payload in enumerate(payloads):
            markdown = payload[len(FAKE_PAGE_MARKER) :] if payload.startswith(FAKE_PAGE_MARKER) else payload
            raw_prompts.append(list(prompt_scaffold) * (r + 1))
            continuations.append(_tokenise(markdown))
        width = max((len(p) for p in raw_prompts), default=0)
        # LEFT-pad each prompt to the shared width; the attention mask is 0 on the pad, 1 on the real
        # (trailing) tokens — the real left-padding shape a decoder-only generate needs.
        padded: list[list[int]] = []
        mask: list[list[int]] = []
        true_lens: list[int] = []
        for prompt in raw_prompts:
            pad = width - len(prompt)
            padded.append([_PAD_ID] * pad + prompt)
            mask.append([0] * pad + [1] * len(prompt))
            true_lens.append(len(prompt))
        self.input_ids = _FakeTokenRows(padded, continuations=continuations, true_lens=true_lens)
        self.attention_mask = _FakeAttentionMask(mask)
        self.moved_to: str | None = None

    def to(self, device: str) -> _FakeBatchEncoding:
        self.moved_to = device
        return self

    def keys(self) -> list[str]:  # so ``**inputs`` sees both keys
        return ["input_ids", "attention_mask"]

    def __getitem__(self, key: str) -> Any:
        if key == "input_ids":
            return self.input_ids
        if key == "attention_mask":
            return self.attention_mask
        raise KeyError(key)


def _extract_image(conversation: list[dict[str, Any]]) -> str:
    """Pull the fake page-image marker (a str) out of a single conversation's user message."""
    content = conversation[0]["content"]
    return str(content[0]["image"])


class _FakeProcessor:
    """Faithful fake for ``LightOnOcrProcessor`` (apply_chat_template + decode).

    ``apply_chat_template`` accepts EITHER a single conversation (a list of role-dicts — the
    single-item path) OR a list of conversations (a list of lists — the W8 batched path) and returns
    a LEFT-padded :class:`_FakeBatchEncoding` carrying prompt + continuation tokens + attention mask;
    ``decode`` turns a :class:`_FakeTokenRow` (the SLICED continuation) back into the page's markdown
    ``str`` — the exact real ``processor.decode`` shape. If the caller forgot to slice and passed the
    full 2-D sequence, decode would render the ``<|prompt-scaffold|>`` prefix too, breaking the
    round-trip; if a batched caller sliced at the WRONG (shared unpadded / first-item) offset, a
    variable-length item's pad/prompt tokens would surface in the markdown — either way, FAIL.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _FakeProcessor:
        return cls(model_id)

    def apply_chat_template(self, messages: Any, **_kwargs: Any) -> _FakeBatchEncoding:
        # A batch is a list whose first element is itself a conversation (a list); a single
        # conversation is a list whose first element is a role-dict. Detect and normalise to a list
        # of conversations so both the single-item and batched Provider paths are faithful.
        is_batch = bool(messages) and isinstance(messages[0], list)
        conversations = messages if is_batch else [messages]
        payloads = [_extract_image(conv) for conv in conversations]
        return _FakeBatchEncoding(payloads)

    def decode(self, token_row: Any, **_kwargs: Any) -> str:
        # REAL SHAPE: decode(generated_ids) -> str. generated_ids is the SLICED continuation the
        # Provider produced via output_ids[i, prompt_width:]; render its tokens back to markdown.
        # skip_special_tokens=True drops any leaked pad id (0) — but a correct slice leaks none.
        if isinstance(token_row, _FakeTokenRow):
            tokens = [t for t in token_row.tokens if t != _PAD_ID]
            return _detokenise(tokens)
        # Tolerate a bare list of token ids (defensive; not the Provider's path).
        return _detokenise([t for t in token_row if t != _PAD_ID])


class _FakeModel:
    """Faithful fake for ``LightOnOcrForConditionalGeneration`` (from_pretrained/to/generate).

    ``generate(**inputs, ...)`` returns the **full** 2-D token sequence (prompt + continuation) for
    EACH item — exactly like HF ``generate`` under left-padding, where every item's continuation
    begins at the shared padded prompt width. So the Provider MUST slice
    ``output_ids[i, input_ids.shape[1]:]`` before decoding. The continuation is that item's page
    markdown; the leading columns are its left-pad + ``<|prompt-scaffold|>`` prompt tokens. Records
    ``generate_calls`` so a test can assert ``ceil(n_pages / batch_size)`` calls.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device: str | None = None
        self.generate_calls = 0

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _FakeModel:
        return cls(model_id)

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def generate(self, input_ids: _FakeTokenRows, **_kwargs: Any) -> _FakeTokenRows:
        # ``**inputs`` unpacks input_ids (the left-padded prompt rows) into this call; the
        # attention_mask arrives as a kwarg (accepted, real HF signature). Echo each padded prompt
        # row, then append THAT row's continuation — the real HF prompt+continuation shape per item.
        self.generate_calls += 1
        full = [list(row) + list(cont) for row, cont in zip(input_ids._rows, input_ids.continuations, strict=True)]
        return _FakeTokenRows(full)


# The default fake page/image PIXEL size — deliberately UNDER the 1540 longest-side cap so the
# existing single/multi-page fixtures are NOT downscaled (their OCR text stays byte-identical); the
# W8 downscale test drives an explicitly-oversized fake image to exercise the resize path.
_FAKE_IMG_W = 800
_FAKE_IMG_H = 1000


class _FakeImage:
    """A minimal fake Pillow image: carries the OCR marker + a ``.size``, supports ``.resize``.

    ``str(image)`` yields the marker so ``apply_chat_template`` reads the intended page markdown off
    it exactly as before (the OCR is marker-driven). ``.size`` + ``.resize`` let the Provider's W8
    longest-side downscale (``_downscale_longest``) run for real offline: an over-cap fake image is
    genuinely resized (aspect preserved), and the marker is carried through the resize unchanged so
    the OCR text is stable across the downscale (resolution changes tokens on the real model, not on
    this deterministic fake — the fake proves the resize MATH + that batching holds at fixed size).
    """

    def __init__(self, marker: str, size: tuple[int, int] = (_FAKE_IMG_W, _FAKE_IMG_H)) -> None:
        self._marker = marker
        self.size = size
        self.resized_to: tuple[int, int] | None = None

    def resize(self, size: tuple[int, int], _resample: Any = None) -> _FakeImage:
        out = _FakeImage(self._marker, size=size)
        out.resized_to = size
        return out

    def __str__(self) -> str:
        return self._marker


class _FakePage:
    """A fake pypdfium2 page: ``get_size()`` reports its point size; ``render(scale=...).to_pil()``.

    The point size is what the Provider's render-scale clamp reads (``_clamp_scale``) — an oversized
    page (huge point size) forces the clamp to shrink the scale so the bitmap stays under budget.
    """

    def __init__(self, marker: str, width_pt: float = 612.0, height_pt: float = 792.0) -> None:
        self._marker = marker
        self._width_pt = width_pt
        self._height_pt = height_pt
        self.rendered_scale: float | None = None

    def get_size(self) -> tuple[float, float]:
        return (self._width_pt, self._height_pt)

    def render(self, scale: float = 1.0, **_kwargs: Any) -> _FakePage:
        self.rendered_scale = scale
        return self

    def to_pil(self) -> _FakeImage:
        return _FakeImage(self._marker)


class _FakePdfDocument:
    """A fake pypdfium2 PdfDocument: iterates fake pages parsed from the fixture bytes.

    The fixture PDF bytes are ``%PDF-1.4`` + newline-joined page markdowns after a marker line, so
    the fake rasteriser yields one :class:`_FakePage` per intended page carrying
    ``FAKE_PAGE::<md>`` — the deterministic offline stand-in for a rendered bitmap. An optional
    ``SIZE:<w>x<h>`` header line sets an oversized point size for every page — the pixel-bomb
    fixture, so the Provider's render-scale clamp is genuinely exercised offline.
    """

    def __init__(self, buffer: Any) -> None:
        data = buffer.read() if hasattr(buffer, "read") else buffer
        text = bytes(data).decode("utf-8", errors="replace")
        # Optional oversized point-size header (the pixel-bomb fixture): "SIZE:WxH\n".
        self._width_pt = 612.0
        self._height_pt = 792.0
        if text.startswith("SIZE:"):
            header, _, text = text.partition("\n")
            dims = header[len("SIZE:") :]
            w_str, _, h_str = dims.partition("x")
            self._width_pt = float(w_str)
            self._height_pt = float(h_str)
        # Pages are the lines after the "PAGES:" marker (see make_lighton_pdf below).
        _, _, body = text.partition("PAGES:\n")
        self._pages = body.split("\n---PAGE---\n") if body else [""]

    def __iter__(self) -> Any:
        for md in self._pages:
            yield _FakePage(FAKE_PAGE_MARKER + md, width_pt=self._width_pt, height_pt=self._height_pt)

    def close(self) -> None:
        pass


def _install_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")
    module.LightOnOcrForConditionalGeneration = _FakeModel  # type: ignore[attr-defined]
    module.LightOnOcrProcessor = _FakeProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def _install_pdfium(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("pypdfium2")
    module.PdfDocument = _FakePdfDocument  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", module)


class _FakeDecompressionBombError(Exception):
    """Stand-in for ``PIL.Image.DecompressionBombError`` — raised when a fixture image is over budget."""


class _FakePilImage:
    """A minimal fake PIL.Image.Image: open().load()/convert() round-trip the marker string.

    ``load()`` honors the module-level ``MAX_IMAGE_PIXELS`` cap: if the fixture image declares a pixel
    count above the cap, it raises :class:`_FakeDecompressionBombError` (like real Pillow) rather than
    "allocating" — so the Provider's decompression-bomb guard is genuinely exercised offline.
    """

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
        # Return a _FakeImage carrying the marker + the declared pixel size so the Provider's W8
        # longest-side downscale runs against a real ``.size``/``.resize`` (aspect-preserving resize).
        return _FakeImage(self._marker, size=self._size)


class _FakeResampling:
    """Stand-in for ``PIL.Image.Resampling`` — only ``LANCZOS`` is referenced by the Provider."""

    LANCZOS = "LANCZOS"


def _install_pil(monkeypatch: pytest.MonkeyPatch) -> None:
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.MAX_IMAGE_PIXELS = 89478485  # type: ignore[attr-defined]  # Pillow's real default
    image_mod.DecompressionBombError = _FakeDecompressionBombError  # type: ignore[attr-defined]
    image_mod.Resampling = _FakeResampling  # type: ignore[attr-defined]

    def _open(buffer: Any) -> _FakePilImage:
        data = buffer.read() if hasattr(buffer, "read") else buffer
        text = bytes(data).decode("utf-8", errors="replace")
        # A single-image fixture carries an optional "PIXELS:<n>|" segment (the bomb fixture) and an
        # optional "DIMS:<w>x<h>|" segment (the W8 downscale fixture) before "IMG:<md>", after the PNG
        # magic. Default to a small in-budget count/size. ``partition`` (not ``startswith``) so the
        # leading magic bytes do not defeat the parse.
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
        return _FakePilImage(FAKE_PAGE_MARKER + md, pixels=pixels, image_module=image_mod, size=size)

    image_mod.open = _open  # type: ignore[attr-defined]
    pil.Image = image_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


@pytest.fixture()
def fake_lighton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install ALL offline doubles (transformers + pypdfium2 + PIL) and pin a CUDA host.

    ``LATENCE_CUDA=1`` so the ``compute="gpu"`` device seam routes to cuda and the model actually
    runs against the fake (offline, no real GPU) — this is what lets the real parse path be
    exercised on this CPU host without fabricating a CPU run of a GPU model.
    """
    _install_transformers(monkeypatch)
    _install_pdfium(monkeypatch)
    _install_pil(monkeypatch)
    monkeypatch.setenv("LATENCE_CUDA", "1")


@pytest.fixture()
def make_lighton_pdf() -> Any:
    """A builder emitting fixture PDF bytes the fake rasteriser splits into pages.

    Valid ``%PDF-`` magic (passes the offset-0 spoof guard) + a ``PAGES:`` marker + the intended
    per-page markdown joined by ``---PAGE---`` — the deterministic offline stand-in for a real PDF.
    An optional ``oversized`` flag prepends a huge ``SIZE:WxH`` header so every page reports an
    over-budget point size — the pixel-bomb fixture that forces the Provider's render-scale clamp to
    shrink the scale.
    """

    def _build(pages: list[str], *, oversized: bool = False) -> bytes:
        body = "\n---PAGE---\n".join(pages)
        prefix = "%PDF-1.4\n"
        if oversized:
            # 100 000 x 100 000 pt at 200 DPI (scale ~2.78) would be ~7.7e10 px — far over budget.
            prefix += "SIZE:100000x100000\n"
        return (prefix + "PAGES:\n" + body).encode("utf-8")

    return _build


@pytest.fixture()
def make_lighton_image() -> Any:
    """A builder emitting fixture PNG bytes (valid PNG magic) carrying one page's markdown.

    An optional ``pixels`` count declares the image's total pixel size; when it exceeds the Provider's
    tightened ``MAX_IMAGE_PIXELS`` budget the fake Pillow raises a decompression-bomb error on load —
    the image pixel-bomb fixture. An optional ``dims`` ``(w, h)`` declares the decoded image's
    width×height so the Provider's W8 longest-side downscale runs against a real ``.size`` — the
    downscale fixture.
    """

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
