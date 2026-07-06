"""EndpointDocumentParser: the Parse performance path over an OpenAI-compatible endpoint (S12).

The real ``openai`` client is intentionally NOT installed in the dev/test env (ADR-0007/0016:
offline, deterministic). These tests inject a tiny fake ``openai`` module so the Provider's markdown
assembly, page-map + Provenance contract, Classification, retry accounting, and graceful
PARSE_ERROR degradation are exercised offline — the model is the endpoint's concern, not this seam's.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    ParserInput,
    Provenance,
)
from latence_parser_endpoint import EndpointDocumentParser


def _parser_input(content: bytes | str = "Hello world.\n") -> ParserInput:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
    )
    return ParserInput(
        provenance=prov,
        classification=Classification(language="und", sensitivity="internal"),
        content=content,
    )


class _FakeCompletions:
    reply: str = "Hello world."
    error: Exception | None = None
    calls: int = 0

    def create(self, **kwargs: Any) -> Any:
        type(self).calls += 1
        if type(self).error is not None and type(self).calls <= _FakeCompletions._fail_times:
            raise type(self).error
        msg = types.SimpleNamespace(content=type(self).reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    _fail_times: int = 0


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.chat = _FakeChat()


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOpenAI]:
    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _FakeCompletions.reply = "Hello world."
    _FakeCompletions.error = None
    _FakeCompletions.calls = 0
    _FakeCompletions._fail_times = 0
    return _FakeOpenAI


def test_transcribes_to_document_record(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = "# Title\n\nBody text."
    out = list(EndpointDocumentParser({"base_url": "http://vllm/v1"}).parse([_parser_input()]))
    assert len(out) == 1
    rec = out[0]
    assert isinstance(rec, DocumentRecord)
    assert rec.disposition is Disposition.PARSED
    assert rec.content == "# Title\n\nBody text."
    assert rec.media_type == "text/markdown"
    assert rec.page_map is not None
    assert rec.provenance.char_start == 0
    assert rec.provenance.char_end == len(rec.content)
    assert rec.provenance.page_start == 1
    assert rec.provenance.page_end == 1
    # Language re-detected at Parse from the decoded text; sensitivity inherited.
    assert rec.classification.sensitivity == "internal"


def test_multi_page_transcription_maps_pages(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = "Page one text.\n\n---PAGE---\n\nPage two text."
    out = list(EndpointDocumentParser({"base_url": "http://vllm/v1"}).parse([_parser_input()]))
    rec = out[0]
    assert rec.provenance.page_end == 2
    assert rec.page_map is not None


def test_base_url_and_key_passed_to_client(fake_openai: type[_FakeOpenAI]) -> None:
    provider = EndpointDocumentParser({"base_url": "http://localhost:8000/v1", "api_key": "k"})
    list(provider.parse([_parser_input()]))
    assert fake_openai.last_kwargs["base_url"] == "http://localhost:8000/v1"
    assert fake_openai.last_kwargs["api_key"] == "k"


def test_transient_failure_is_retried_and_counted(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.error = _StatusError("overloaded", status_code=503)
    _FakeCompletions._fail_times = 2  # first two calls 503, third succeeds
    _FakeCompletions.reply = "recovered"
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "retry_attempts": 5})
    out = list(provider.parse([_parser_input()]))
    assert out[0].content == "recovered"
    assert out[0].disposition is Disposition.PARSED
    # Two retries absorbed, surfaced for the Quality Report.
    assert provider.endpoint_retries == 2


def test_terminal_failure_yields_parse_error(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.error = _StatusError("bad request", status_code=400)
    _FakeCompletions._fail_times = 99  # always fails; 400 is terminal (no retry)
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "retry_attempts": 5})
    out = list(provider.parse([_parser_input()]))
    # A terminal endpoint failure degrades gracefully to PARSE_ERROR, never a crash.
    assert out[0].disposition is Disposition.PARSE_ERROR
    assert out[0].error is not None and "bad request" in out[0].error
    assert provider.endpoint_retries == 0  # terminal, not retried


def test_retries_exhausted_yields_parse_error(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.error = _StatusError("still down", status_code=503)
    _FakeCompletions._fail_times = 99
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "retry_attempts": 3})
    out = list(provider.parse([_parser_input()]))
    assert out[0].disposition is Disposition.PARSE_ERROR
    assert "endpoint parse failed" in (out[0].error or "")


def test_binary_input_sent_as_data_url(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = "transcribed pdf"
    out = list(
        EndpointDocumentParser({"base_url": "http://vllm/v1"}).parse(
            [_parser_input(content=b"%PDF-1.4 fake")]
        )
    )
    assert out[0].content == "transcribed pdf"


def test_missing_openai_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1"})
    with pytest.raises(ImportError, match="pip install latence-parser-endpoint"):
        list(provider.parse([_parser_input()]))


def test_oversized_binary_input_is_rejected_before_encoding(
    fake_openai: type[_FakeOpenAI],
) -> None:
    """Adversarial (issue #44.1): an over-cap binary input must NOT be base64-encoded or sent.

    Simulates a zip-bomb-expanded / oversized input reaching the parser (e.g. Screening bypassed):
    a payload just over a small configured cap must degrade to PARSE_ERROR with the size reason and
    make ZERO endpoint calls — proving the guard fires before the base64/data-URL inflation path,
    not after paying its memory/request cost.
    """
    _FakeCompletions.calls = 0
    _FakeCompletions.reply = "should never be produced"
    payload = b"\x00" * 2048  # 2 KiB, over the 1 KiB cap below
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "max_input_bytes": 1024})
    out = list(provider.parse([_parser_input(content=payload)]))
    assert len(out) == 1
    assert out[0].disposition is Disposition.PARSE_ERROR
    assert "over the 1024-byte parser cap" in (out[0].error or "")
    assert out[0].content == ""
    # The endpoint was never called — the input was rejected before encoding, not transcribed.
    assert _FakeCompletions.calls == 0
    # Provenance survives the rejection (auditable).
    assert out[0].provenance.file_name == "a.txt"


def test_oversized_text_input_is_rejected_before_encoding(
    fake_openai: type[_FakeOpenAI],
) -> None:
    """A huge text input inflates the prompt the same way; the cap covers it (UTF-8 byte length)."""
    _FakeCompletions.calls = 0
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "max_input_bytes": 1024})
    out = list(provider.parse([_parser_input(content="A" * 2048)]))
    assert out[0].disposition is Disposition.PARSE_ERROR
    assert _FakeCompletions.calls == 0


def test_at_cap_input_still_parses(fake_openai: type[_FakeOpenAI]) -> None:
    """An input exactly at the cap is NOT rejected — the guard is strictly ``>`` cap."""
    _FakeCompletions.calls = 0
    _FakeCompletions.reply = "ok"
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "max_input_bytes": 16})
    out = list(provider.parse([_parser_input(content=b"\x00" * 16)]))
    assert out[0].disposition is Disposition.PARSED
    assert _FakeCompletions.calls == 1


def test_cap_disabled_allows_large_input(fake_openai: type[_FakeOpenAI]) -> None:
    """``max_input_bytes <= 0`` disables the guard for an adopter who caps upstream."""
    _FakeCompletions.calls = 0
    _FakeCompletions.reply = "ok"
    provider = EndpointDocumentParser({"base_url": "http://vllm/v1", "max_input_bytes": 0})
    out = list(provider.parse([_parser_input(content=b"\x00" * 4096)]))
    assert out[0].disposition is Disposition.PARSED
    assert _FakeCompletions.calls == 1
