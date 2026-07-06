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
    PageMap,
    Provenance,
    RelationMention,
)
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


class _FakeCompletions:
    reply: str = "{}"

    def create(self, **kwargs: Any) -> Any:
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
    return _FakeOpenAI


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
