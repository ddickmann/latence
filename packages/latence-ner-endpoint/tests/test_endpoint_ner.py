"""EndpointEntityExtractor: the NER performance path over an OpenAI-compatible endpoint (S12).

The real ``openai`` client is intentionally NOT installed (ADR-0007/0016: offline, deterministic).
A fake ``openai`` module returns canned JSON so the Provider's span validation, offset+page
Provenance (the S4 round-trip), zero-shot config, retry accounting, lease acquisition, and
hallucination-dropping are exercised offline.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest
from latence_core.contracts import ChunkRecord, Classification, EntityMention, PageMap, Provenance
from latence_ner_endpoint import EndpointEntityExtractor
from latence_ner_endpoint.provider import _POOL


def _chunk(text: str, *, char_start: int = 0, page_map: PageMap | None = None) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:d1",
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id="d1#chunk-0",
        document_record_id="d1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
        page_map=page_map,
    )


class _FakeCompletions:
    reply: str = "{}"
    error: Exception | None = None
    fail_times: int = 0
    calls: int = 0

    def create(self, **kwargs: Any) -> Any:
        type(self).calls += 1
        if type(self).error is not None and type(self).calls <= type(self).fail_times:
            raise type(self).error
        msg = types.SimpleNamespace(content=type(self).reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.chat = _FakeChat()


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _FakeCompletions.reply = "{}"
    _FakeCompletions.error = None
    _FakeCompletions.fail_times = 0
    _FakeCompletions.calls = 0
    # A fresh pool each test so lease ref-counts don't leak across tests.
    _POOL._leases.clear()  # noqa: SLF001


def test_requires_labels_config() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        EndpointEntityExtractor({})
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        EndpointEntityExtractor({"labels": []})


def test_extracts_and_resolves_offsets(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps(
        {"entities": [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]}
    )
    chunk = _chunk("Alice works at Acme.", char_start=10)
    out = list(EndpointEntityExtractor({"labels": ["person"]}).extract([chunk]))
    assert len(out) == 1
    m = out[0]
    assert isinstance(m, EntityMention)
    assert m.text == "Alice"
    assert m.label == "person"
    # Chunk-local [0,5) shifted by the chunk's char_start=10 into document coordinates.
    assert m.provenance.char_start == 10
    assert m.provenance.char_end == 15
    assert m.provenance.page_start == 1
    assert m.classification.sensitivity == "internal"


def test_hallucinated_spans_dropped(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps(
        {
            "entities": [
                {"start": 0, "end": 999, "label": "person"},  # out of range
                {"start": 5, "end": 2, "label": "person"},  # inverted
                {"start": 0, "end": 5, "label": "bogus"},  # unknown label
                {"start": 0, "end": 5, "label": "person", "confidence": 0.1},  # below threshold
                {"start": 0, "end": 5, "label": "person", "confidence": 0.9},  # valid
            ]
        }
    )
    chunk = _chunk("Alice works at Acme.")
    out = list(EndpointEntityExtractor({"labels": ["person"], "threshold": 0.5}).extract([chunk]))
    assert [m.confidence for m in out] == [0.9]
    assert out[0].index == 0  # re-indexed after drops


def test_empty_chunk_skips_model(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps({"entities": [{"start": 0, "end": 5, "label": "person"}]})
    chunk = _chunk("   ")
    out = list(EndpointEntityExtractor({"labels": ["person"], "min_chars": 2}).extract([chunk]))
    assert out == []
    assert _FakeCompletions.calls == 0


def test_transient_failure_retried_and_counted(fake_openai: None) -> None:
    _FakeCompletions.error = _StatusError("429", status_code=429)
    _FakeCompletions.fail_times = 1
    _FakeCompletions.reply = json.dumps(
        {"entities": [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]}
    )
    provider = EndpointEntityExtractor({"labels": ["person"], "retry_attempts": 3})
    out = list(provider.extract([_chunk("Alice works at Acme.")]))
    assert len(out) == 1
    assert provider.endpoint_retries == 1


def test_terminal_failure_surfaces(fake_openai: None) -> None:
    from latence_core.endpoint import EndpointError

    _FakeCompletions.error = _StatusError("bad request", status_code=400)
    _FakeCompletions.fail_times = 99
    provider = EndpointEntityExtractor({"labels": ["person"], "retry_attempts": 3})
    with pytest.raises(EndpointError, match="bad request"):
        list(provider.extract([_chunk("Alice works at Acme.")]))


def test_lease_released_after_stage(fake_openai: None) -> None:
    _FakeCompletions.reply = json.dumps({"entities": []})
    provider = EndpointEntityExtractor({"labels": ["person"], "base_url": "http://vllm/v1"})
    list(provider.extract([_chunk("Alice works at Acme.")]))
    # The ref-counted lease is back to 0 (activated once, released once) after the Stage's stream.
    assert provider._lease.refcount == 0  # noqa: SLF001
    assert provider._lease.live is False  # noqa: SLF001


def test_missing_openai_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    _POOL._leases.clear()  # noqa: SLF001
    provider = EndpointEntityExtractor({"labels": ["person"]})
    with pytest.raises(ImportError, match="pip install latence-ner-endpoint"):
        list(provider.extract([_chunk("Alice works at Acme.")]))
