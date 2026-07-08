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


class _FakeTokenRow:
    """A 1-D slice of token ids (the sliced continuation) that ``decode`` turns back into text."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens


class _FakeTokenRows:
    """A minimal 2-D token tensor stand-in: rows of int token ids, indexable like a torch tensor.

    Supports the two operations the Provider + the fake use on real transformers tensors:
      * ``.shape`` -> ``(rows, cols)`` (so ``inputs["input_ids"].shape[1]`` = the prompt length), and
      * ``output_ids[0, prompt_len:]`` 2-D indexing -> a :class:`_FakeTokenRow` continuation slice.

    ``input_ids`` (the prompt) additionally stashes the ``continuation`` tokens the model will append,
    so the fake ``generate`` can reconstruct the full prompt+continuation sequence from ``**inputs``
    alone (no global state). Tokens are Unicode codepoints so ``decode`` is the inverse ``chr`` map —
    deterministic, inspectable, and NEVER a real tensor (offline).
    """

    def __init__(self, rows: list[list[int]], continuation: list[int] | None = None) -> None:
        self._rows = rows
        self.continuation = continuation or []

    @property
    def shape(self) -> tuple[int, int]:
        cols = len(self._rows[0]) if self._rows else 0
        return (len(self._rows), cols)

    def __getitem__(self, key: Any) -> _FakeTokenRow:
        # The Provider does ``output_ids[0, prompt_len:]`` -> (row index, column slice).
        row_key, col_key = key
        row = self._rows[row_key]
        return _FakeTokenRow(row[col_key])


class _FakeBatchEncoding:
    """The dict-like handle ``apply_chat_template(return_dict=True)`` returns; ``.to(device)`` no-op.

    Exposes ``input_ids`` as a :class:`_FakeTokenRows` whose single row is the PROMPT tokens (the
    ``<|prompt-scaffold|>`` stand-in) — so ``inputs["input_ids"].shape[1]`` is the true prompt length,
    the exact slice offset the Provider must use. It also stashes the page-markdown continuation on
    that row so ``model.generate(**inputs)`` can echo back prompt+continuation (the real HF shape).
    ``**inputs`` unpacking works because it is a mapping; ``.to(device)`` returns self.
    """

    def __init__(self, payload: str) -> None:
        # The page markdown the fake "OCR" produced from the marker image.
        markdown = payload[len(FAKE_PAGE_MARKER) :] if payload.startswith(FAKE_PAGE_MARKER) else payload
        prompt_tokens = _tokenise(FAKE_PROMPT_SCAFFOLD)
        continuation_tokens = _tokenise(markdown)
        # input_ids is JUST the prompt (what the model is conditioned on); generate() appends the
        # continuation. shape[1] = prompt length — the slice offset the Provider must use.
        self.input_ids = _FakeTokenRows([list(prompt_tokens)], continuation=continuation_tokens)
        self.moved_to: str | None = None

    def to(self, device: str) -> _FakeBatchEncoding:
        self.moved_to = device
        return self

    def keys(self) -> list[str]:  # so ``**inputs`` sees the input_ids key
        return ["input_ids"]

    def __getitem__(self, key: str) -> _FakeTokenRows:
        if key == "input_ids":
            return self.input_ids
        raise KeyError(key)


class _FakeProcessor:
    """Faithful fake for ``LightOnOcrProcessor`` (apply_chat_template + decode).

    ``apply_chat_template`` reads the image marker off the chat message and returns a
    :class:`_FakeBatchEncoding` carrying prompt + continuation tokens; ``decode`` turns a
    :class:`_FakeTokenRow` (the SLICED continuation) back into the page's markdown ``str`` — the exact
    real ``processor.decode`` shape. If the caller forgot to slice and passed the full 2-D sequence,
    decode would render the ``<|prompt-scaffold|>`` prefix too, breaking the round-trip.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _FakeProcessor:
        return cls(model_id)

    def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> _FakeBatchEncoding:
        # Extract the fake page image (a str marker) from the user message's image content.
        content = messages[0]["content"]
        image = content[0]["image"]
        return _FakeBatchEncoding(str(image))

    def decode(self, token_row: Any, **_kwargs: Any) -> str:
        # REAL SHAPE: decode(generated_ids) -> str. generated_ids is the SLICED continuation the
        # Provider produced via output_ids[0, prompt_len:]; render its tokens back to markdown.
        if isinstance(token_row, _FakeTokenRow):
            return _detokenise(token_row.tokens)
        # Tolerate a bare list of token ids (defensive; not the Provider's path).
        return _detokenise(list(token_row))


class _FakeModel:
    """Faithful fake for ``LightOnOcrForConditionalGeneration`` (from_pretrained/to/generate).

    ``generate(**inputs, ...)`` returns the **full** 2-D token sequence (prompt + continuation) —
    exactly like HF ``generate`` — so the Provider MUST slice ``output_ids[0, input_ids.shape[1]:]``
    before decoding. The continuation is the page markdown; the prompt columns are the
    ``<|prompt-scaffold|>`` tokens the real chat template contributes.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.device: str | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _FakeModel:
        return cls(model_id)

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def generate(self, input_ids: _FakeTokenRows, **_kwargs: Any) -> _FakeTokenRows:
        # ``**inputs`` unpacks input_ids (the prompt tokens) into this call. The real HF generate()
        # returns prompt + continuation: echo the prompt row, then append the continuation tokens the
        # encoding stashed. The Provider must slice the prompt columns back off before decoding.
        prompt_row = list(input_ids._rows[0])
        full = [prompt_row + list(input_ids.continuation)]
        return _FakeTokenRows(full)


class _FakePage:
    """A fake pypdfium2 page: ``render(scale=...).to_pil()`` yields the page's marker image."""

    def __init__(self, marker: str) -> None:
        self._marker = marker

    def render(self, **_kwargs: Any) -> _FakePage:
        return self

    def to_pil(self) -> str:
        return self._marker


class _FakePdfDocument:
    """A fake pypdfium2 PdfDocument: iterates fake pages parsed from the fixture bytes.

    The fixture PDF bytes are ``%PDF-1.4`` + newline-joined page markdowns after a marker line, so
    the fake rasteriser yields one :class:`_FakePage` per intended page carrying
    ``FAKE_PAGE::<md>`` — the deterministic offline stand-in for a rendered bitmap.
    """

    def __init__(self, buffer: Any) -> None:
        data = buffer.read() if hasattr(buffer, "read") else buffer
        text = bytes(data).decode("utf-8", errors="replace")
        # Pages are the lines after the "PAGES:" marker (see make_lighton_pdf below).
        _, _, body = text.partition("PAGES:\n")
        self._pages = body.split("\n---PAGE---\n") if body else [""]

    def __iter__(self) -> Any:
        for md in self._pages:
            yield _FakePage(FAKE_PAGE_MARKER + md)

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


class _FakePilImage:
    """A minimal fake PIL.Image.Image: open().load()/convert() round-trip the marker string."""

    def __init__(self, marker: str) -> None:
        self._marker = marker

    def load(self) -> None:
        pass

    def convert(self, _mode: str) -> str:
        return self._marker


def _install_pil(monkeypatch: pytest.MonkeyPatch) -> None:
    pil = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")

    def _open(buffer: Any) -> _FakePilImage:
        data = buffer.read() if hasattr(buffer, "read") else buffer
        text = bytes(data).decode("utf-8", errors="replace")
        # A single-image fixture carries "IMG:<md>" after its magic bytes; strip to the markdown.
        _, _, md = text.partition("IMG:")
        return _FakePilImage(FAKE_PAGE_MARKER + md)

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
    """

    def _build(pages: list[str]) -> bytes:
        body = "\n---PAGE---\n".join(pages)
        return b"%PDF-1.4\nPAGES:\n" + body.encode("utf-8")

    return _build


@pytest.fixture()
def make_lighton_image() -> Any:
    """A builder emitting fixture PNG bytes (valid PNG magic) carrying one page's markdown."""

    def _build(markdown: str) -> bytes:
        return b"\x89PNG\r\n\x1a\nIMG:" + markdown.encode("utf-8")

    return _build
