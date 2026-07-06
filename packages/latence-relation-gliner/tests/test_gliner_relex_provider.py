"""GlinerRelexExtractor: fused zero-shot Entity+Relation Extraction (S5, ADR-0013).

The real ``gliner`` stack (torch/transformers) is intentionally NOT installed in the
dev/test env (ADR-0007/0016). These tests inject a tiny fake ``gliner`` module whose
model returns fixed chunk-local entity + relation spans, so the Provider's fused
output, contract stamping, offset→page math, and relation→mention resolution are
exercised deterministically and offline. One test covers the missing-dependency error
path and the zero-shot config validation (both entity AND relation labels required).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.capability import FusedExtraction
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
    RelationMention,
)
from latence_relation_gliner import GlinerRelexExtractor


def _chunk(
    content: str,
    *,
    char_start: int = 100,
    char_end: int | None = None,
    page_map: PageMap | None = None,
    offset_map: OffsetMap | None = None,
) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=char_start,
        char_end=char_start + len(content) if char_end is None else char_end,
        page_start=2,
        page_end=2,
    )
    return ChunkRecord(
        record_id="doc1#chunk-0",
        document_record_id="doc1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        token_count=len(content.split()),
        page_map=page_map,
        offset_map=offset_map,
    )


class _FakeGLiNER:
    """Stands in for gliner.GLiNER with a predict_with_relations method."""

    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    model_id: str = ""

    @classmethod
    def from_pretrained(cls, model_id: str) -> _FakeGLiNER:
        inst = cls()
        inst.model_id = model_id
        return inst

    def predict_with_relations(
        self,
        text: str,
        labels: list[str],
        relation_labels: list[str],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        ents = [e for e in type(self).entities if e["label"] in labels and e["score"] >= threshold]
        rels = [
            r
            for r in type(self).relations
            if r.get("relation", r.get("label")) in relation_labels
            and r.get("score", 1.0) >= threshold
        ]
        return {"entities": ents, "relations": rels}


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _FakeGLiNER.entities = []
    _FakeGLiNER.relations = []
    return _FakeGLiNER


def _config() -> dict[str, Any]:
    return {"labels": ["person", "organization"], "relation_labels": ["works_for"]}


def test_requires_entity_and_relation_labels() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        GlinerRelexExtractor({"relation_labels": ["works_for"]})
    with pytest.raises(ValueError, match="requires a non-empty 'relation_labels' list"):
        GlinerRelexExtractor({"labels": ["person"]})


def test_default_model_is_permissive_apache() -> None:
    extractor = GlinerRelexExtractor(_config())
    assert extractor._model_id == "knowledgator/gliner-relex-multi-v1.0"


def test_fused_emits_mentions_and_relations(fake_gliner: type[_FakeGLiNER]) -> None:
    # "Alice works for Acme Corp." — Alice [0,5), Acme Corp [16,25) in chunk-local text.
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.95},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 16, "end": 25},
            "relation": "works_for",
            "score": 0.88,
        }
    ]
    chunk = _chunk("Alice works for Acme Corp.", char_start=100)
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    assert isinstance(result, FusedExtraction)
    assert [m.text for m in result.mentions] == ["Alice", "Acme Corp"]
    assert all(isinstance(m, EntityMention) for m in result.mentions)
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert isinstance(rel, RelationMention)
    assert rel.label == "works_for"
    # The relation's endpoints resolve to the mentions this pass produced.
    ids = {m.text: m.record_id for m in result.mentions}
    assert rel.head_mention_id == ids["Alice"]
    assert rel.tail_mention_id == ids["Acme Corp"]
    assert rel.confidence == 0.88
    # Chunk-local offsets shifted into the parent-document coordinate system.
    assert result.mentions[0].provenance.char_start == 100
    assert rel.provenance.char_start == 100
    assert rel.provenance.char_end == 125
    # Page inherited from the chunk (no multi-page map here).
    assert rel.provenance.page_start == 2


def test_relation_endpoint_offsets_shift_to_document(fake_gliner: type[_FakeGLiNER]) -> None:
    # Multi-page chunk: head on page 1, tail on page 2 (via the covering span).
    assembly = PageMap.from_page_texts(["Alice is here.", "She joined Globex."])
    # page 1 [0,16), page 2 [16,...); "Alice" at 0, "Globex" at 27 in assembled text.
    text = assembly.text
    g_start = text.find("Globex")
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": g_start, "end": g_start + 6, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": g_start, "end": g_start + 6},
            "relation": "works_for",
            "score": 0.8,
        }
    ]
    chunk = _chunk(text, char_start=0, page_map=assembly.page_map)
    cfg = {"labels": ["person", "organization"], "relation_labels": ["works_for"]}
    result = GlinerRelexExtractor(cfg).extract_fused([chunk])
    rel = result.relations[0]
    # Covering span straddles the page boundary: records both pages.
    assert rel.provenance.page_start == 1
    assert rel.provenance.page_end == 2


def test_offsets_resolve_through_offset_map_past_stripped_markup(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    # Original "**Note:** Alice joined Globex." strips to "Note: Alice joined Globex.":
    # the stripped ``**`` before the prose shifts every later mention right in the
    # original, so char_start+local is a lower bound. The carried offset_map corrects it.
    original = "**Note:** Alice joined Globex."
    content = "Note: Alice joined Globex."
    offset_map = OffsetMap(
        segments=[
            OffsetSegment(local_start=0, doc_start=2),
            OffsetSegment(local_start=5, doc_start=9),
        ],
        length=len(content),
    )
    fake_gliner.entities = [
        {"start": 6, "end": 11, "label": "person", "score": 0.9},
        {"start": 19, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 6, "end": 11},
            "tail": {"start": 19, "end": 25},
            "relation": "works_for",
            "score": 0.8,
        }
    ]
    chunk = _chunk(content, char_start=2, char_end=len(original), offset_map=offset_map)
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    by_text = {m.text: m for m in result.mentions}
    assert set(by_text) == {"Alice", "Globex"}
    for m in result.mentions:
        # TRUE original span (not the char_start+local lower bound).
        assert original[m.provenance.char_start : m.provenance.char_end] == m.text
    assert by_text["Alice"].provenance.char_start == 10  # not 2 + 6 = 8
    assert by_text["Globex"].provenance.char_start == 23  # not 2 + 19 = 21
    # The relation's covering span is the min/max of the corrected endpoints.
    rel = result.relations[0]
    assert (rel.provenance.char_start, rel.provenance.char_end) == (10, 29)
    assert original[rel.provenance.char_start : rel.provenance.char_end] == "Alice joined Globex"


def test_empty_chunk_skipped_without_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    result = GlinerRelexExtractor(_config()).extract_fused([_chunk("")])
    assert result.mentions == []
    assert result.relations == []


def test_relation_with_unresolvable_endpoint_dropped(fake_gliner: type[_FakeGLiNER]) -> None:
    # A relation referencing an entity span the model did not emit is dropped, so no
    # un-resolvable record leaks out.
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 999, "end": 1004},  # no such entity
            "relation": "works_for",
            "score": 0.8,
        }
    ]
    chunk = _chunk("Alice works for Acme Corp.")
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    assert len(result.mentions) == 2
    assert result.relations == []


def test_extract_half_returns_only_mentions(fake_gliner: type[_FakeGLiNER]) -> None:
    # The Provider also fits the plain EntityExtractor seam via .extract().
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    fake_gliner.relations = []
    chunk = _chunk("Alice waved.")
    mentions = GlinerRelexExtractor(_config()).extract([chunk])
    assert [m.text for m in mentions] == ["Alice"]


def test_missing_gliner_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner", None)
    extractor = GlinerRelexExtractor(_config())
    with pytest.raises(ImportError, match="pip install latence-relation-gliner"):
        extractor.extract_fused([_chunk("Alice works for Acme Corp.")])
