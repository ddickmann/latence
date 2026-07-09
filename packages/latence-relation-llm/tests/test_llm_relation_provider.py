"""LLMRelationExtractor: LLM Relation Extraction over an OpenAI-compatible endpoint (S5).

The real ``openai`` client is intentionally NOT installed in the dev/test env
(ADR-0007/0016: offline, deterministic). These tests inject a tiny fake ``openai``
module returning a canned JSON reply so the Provider's mention-index resolution,
contract stamping, offset math, and hallucination-dropping are exercised offline —
the model itself is the endpoint's concern, not this seam's. One test covers the
missing-dependency error path and the zero-shot config validation.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    InducedLabels,
    PageMap,
    Provenance,
    RelationMention,
)
from latence_core.endpoint import EndpointError
from latence_relation_llm import LLMRelationExtractor


def _chunk(
    text: str,
    *,
    char_start: int = 0,
    page_map: PageMap | None = None,
) -> ChunkRecord:
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


def _mention(idx: int, label: str, text: str, start: int, end: int) -> EntityMention:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:d1",
        char_start=start,
        char_end=end,
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=f"d1#chunk-0#entity-{idx}",
        document_record_id="d1",
        chunk_record_id="d1#chunk-0",
        index=idx,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
    )


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeCompletions:
    reply: str = "{}"
    error: Exception | None = None
    _fail_times: int = 0
    calls: int = 0
    last_messages: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        type(self).calls += 1
        type(self).last_messages = list(kwargs.get("messages", []))
        err = type(self).error
        if err is not None and type(self).calls <= type(self)._fail_times:
            raise err
        msg = types.SimpleNamespace(content=type(self).reply)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs
        self.chat = _FakeChat()


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[_FakeOpenAI]:
    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _FakeCompletions.reply = "{}"
    _FakeCompletions.error = None
    _FakeCompletions._fail_times = 0
    _FakeCompletions.calls = 0
    _FakeCompletions.last_messages = []
    return _FakeOpenAI


def _induced_chunk(text: str, induced: InducedLabels) -> ChunkRecord:
    return _chunk(text).model_copy(update={"induced_labels": induced})


def test_requires_labels_config() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        LLMRelationExtractor({})
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        LLMRelationExtractor({"labels": []})


def test_extracts_relation_and_resolves_indices(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = json.dumps(
        {"relations": [{"head": 0, "tail": 1, "label": "works_for", "confidence": 0.8}]}
    )
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [
        _mention(0, "PERSON", "Alice", 0, 5),
        _mention(1, "ORG", "Acme Corp", 16, 25),
    ]
    out = list(LLMRelationExtractor({"labels": ["works_for"]}).relate([chunk], mentions))
    assert len(out) == 1
    r = out[0]
    assert isinstance(r, RelationMention)
    assert r.label == "works_for"
    assert r.head_mention_id == "d1#chunk-0#entity-0"
    assert r.tail_mention_id == "d1#chunk-0#entity-1"
    assert r.head_text == "Alice"
    assert r.tail_text == "Acme Corp"
    # Covering span spans both endpoints; page inherited from the endpoints.
    assert r.provenance.char_start == 0
    assert r.provenance.char_end == 25
    assert r.provenance.page_start == 1
    assert r.classification.sensitivity == "internal"


def _relation_labels_from_prompt(messages: list[dict[str, Any]]) -> list[str]:
    """Pull the 'Relation labels: a, b, c' line the provider embeds in the user prompt."""
    for message in messages:
        content = str(message.get("content", ""))
        for line in content.splitlines():
            if line.startswith("Relation labels:"):
                return [x.strip() for x in line.removeprefix("Relation labels:").split(",") if x.strip()]
    raise AssertionError("no 'Relation labels:' line found in the prompt")


def test_induced_relation_types_union_onto_relation_slot(fake_openai: type[_FakeOpenAI]) -> None:
    # W4 seam (TESTS lens): the LLM relation extractor asks the model for the UNION of the config
    # relation labels + the induced relation_types (the RELATION slot only — NOT entity_types), read
    # off the chunk's induced_labels. A regression that mis-slotted or dropped this FAILS here.
    _FakeCompletions.reply = json.dumps({"relations": []})
    induced = InducedLabels(
        model_id="test-model",
        entity_types=["location"],  # must NOT leak into the relation-label prompt
        relation_types=["located_in"],  # NOT in the config relation floor
        pii_types=[],
    )
    chunk = _induced_chunk("Alice works for Acme Corp in Berlin.", induced)
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    list(LLMRelationExtractor({"labels": ["works_for"]}).relate([chunk], mentions))
    asked = _relation_labels_from_prompt(_FakeCompletions.last_messages)
    # config floor first, then the induced relation type — and the induced entity type is absent.
    assert asked == ["works_for", "located_in"]
    assert "location" not in asked


def test_no_induced_labels_asks_for_config_relation_floor(fake_openai: type[_FakeOpenAI]) -> None:
    # Byte-identical backward-compat: no induced labels ⇒ exactly the config relation labels.
    _FakeCompletions.reply = json.dumps({"relations": []})
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    list(LLMRelationExtractor({"labels": ["works_for"]}).relate([chunk], mentions))
    assert _relation_labels_from_prompt(_FakeCompletions.last_messages) == ["works_for"]


def test_base_url_and_key_passed_to_client(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = json.dumps({"relations": []})
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    provider = LLMRelationExtractor(
        {"labels": ["works_for"], "base_url": "http://localhost:11434/v1", "api_key": "ollama"}
    )
    list(provider.relate([chunk], mentions))
    assert fake_openai.last_kwargs["base_url"] == "http://localhost:11434/v1"
    assert fake_openai.last_kwargs["api_key"] == "ollama"


def test_out_of_range_and_self_and_unknown_dropped(fake_openai: type[_FakeOpenAI]) -> None:
    # 99 is out of range; head==tail is a self-relation; 'bogus' is an unknown label.
    _FakeCompletions.reply = json.dumps(
        {
            "relations": [
                {"head": 0, "tail": 99, "label": "works_for"},
                {"head": 0, "tail": 0, "label": "works_for"},
                {"head": 0, "tail": 1, "label": "bogus"},
                {"head": 0, "tail": 1, "label": "works_for", "confidence": 0.7},
            ]
        }
    )
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    out = list(LLMRelationExtractor({"labels": ["works_for"]}).relate([chunk], mentions))
    # Only the last, valid relation survives.
    assert [r.confidence for r in out] == [0.7]


def test_non_json_reply_yields_no_relations(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = "sorry, I can't do that"
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    assert list(LLMRelationExtractor({"labels": ["works_for"]}).relate([chunk], mentions)) == []


def test_fewer_than_two_mentions_skips_model(fake_openai: type[_FakeOpenAI]) -> None:
    _FakeCompletions.reply = json.dumps(
        {"relations": [{"head": 0, "tail": 1, "label": "works_for"}]}
    )
    chunk = _chunk("Alice waved.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5)]
    assert list(LLMRelationExtractor({"labels": ["works_for"]}).relate([chunk], mentions)) == []


def test_missing_openai_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    provider = LLMRelationExtractor({"labels": ["works_for"]})
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    with pytest.raises(ImportError, match="pip install latence-relation-llm"):
        list(provider.relate([chunk], mentions))


def test_transient_failure_is_retried_and_counted(fake_openai: type[_FakeOpenAI]) -> None:
    """Regression (issue #44.3): a 5xx blip is retried through the ported retry AND surfaces.

    Before the fix the RE Provider called the endpoint directly (no retry, no ``endpoint_retries``),
    so a transient 503 crashed the Stage and RE retries never appeared in the Quality Report like
    Parse/NER/Embedder. Now two 503s are absorbed, the third try succeeds, and the retry count is
    tallied on ``endpoint_retries`` — the attribute the Runner folds into the report.
    """
    _FakeCompletions.error = _StatusError("overloaded", status_code=503)
    _FakeCompletions._fail_times = 2
    _FakeCompletions.reply = json.dumps(
        {"relations": [{"head": 0, "tail": 1, "label": "works_for", "confidence": 0.8}]}
    )
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    provider = LLMRelationExtractor({"labels": ["works_for"]})
    out = list(provider.relate([chunk], mentions))
    assert len(out) == 1  # succeeded after the retries
    assert _FakeCompletions.calls == 3  # two failures + one success
    assert provider.endpoint_retries == 2  # surfaced for the Quality Report


def test_terminal_failure_propagates_without_retry(fake_openai: type[_FakeOpenAI]) -> None:
    """A terminal 4xx fails fast: one call, EndpointError raised, no retry (issue #44.3)."""
    _FakeCompletions.error = _StatusError("bad request", status_code=400)
    _FakeCompletions._fail_times = 99
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    provider = LLMRelationExtractor({"labels": ["works_for"]})
    with pytest.raises(EndpointError, match="bad request"):
        list(provider.relate([chunk], mentions))
    assert _FakeCompletions.calls == 1
    assert provider.endpoint_retries == 0


def test_retries_exhausted_surface_the_error(fake_openai: type[_FakeOpenAI]) -> None:
    """When every attempt is a retryable failure, the final EndpointError surfaces (issue #44.3)."""
    _FakeCompletions.error = _StatusError("still down", status_code=503)
    _FakeCompletions._fail_times = 99
    chunk = _chunk("Alice works for Acme Corp.")
    mentions = [_mention(0, "PERSON", "Alice", 0, 5), _mention(1, "ORG", "Acme Corp", 16, 25)]
    provider = LLMRelationExtractor(
        {"labels": ["works_for"], "retry_attempts": 3}
    )
    with pytest.raises(EndpointError, match="still down"):
        list(provider.relate([chunk], mentions))
    assert _FakeCompletions.calls == 3  # attempts exhausted
    assert provider.endpoint_retries == 2  # two retries taken before the final raise
