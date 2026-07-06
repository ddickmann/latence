"""GlinerEntityExtractor: learned zero-shot Entity Extraction over GLiNER (S4).

The real ``gliner`` stack (torch/transformers) is intentionally NOT installed in
the dev/test environment (ADR-0007/0016: CPU-first, offline, deterministic). These
tests inject a tiny fake ``gliner`` module so the Provider's contract-stamping and
offset math are exercised deterministically and offline — the model itself is
GLiNER's concern, not this seam's. One test also covers the missing-dependency
error path and the zero-shot config validation.
"""

from __future__ import annotations

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
)
from latence_ner_gliner import GlinerEntityExtractor


def _chunk(
    content: str,
    *,
    char_start: int = 100,
    page_map: PageMap | None = None,
    record_id: str = "doc1#chunk-0",
) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=char_start,
        char_end=char_start + len(content),
        page_start=2,
        page_end=2,
    )
    return ChunkRecord(
        record_id=record_id,
        document_record_id="doc1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        token_count=len(content.split()),
        page_map=page_map,
    )


class _FakeGLiNER:
    """Stands in for gliner.GLiNER: returns fixed chunk-local spans + scores."""

    #: chunk-local entity dicts the fake "model" returns for any predict call.
    returns: list[dict[str, Any]] = []
    model_id: str = ""

    @classmethod
    def from_pretrained(cls, model_id: str) -> _FakeGLiNER:
        inst = cls()
        inst.model_id = model_id
        return inst

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        # Echo only entities whose label is in the requested set (zero-shot), above
        # the threshold — mirroring the real API's contract.
        return [
            e for e in type(self).returns if e["label"] in labels and e["score"] >= threshold
        ]


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    return _FakeGLiNER


def test_requires_labels_config() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        GlinerEntityExtractor({})
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        GlinerEntityExtractor({"labels": []})


def test_default_model_is_permissive_apache() -> None:
    extractor = GlinerEntityExtractor({"labels": ["person"]})
    # The default checkpoint is the Apache-2.0 v2.1 (never a CC-BY-NC one).
    assert extractor._model_id == "urchade/gliner_multi-v2.1"


def test_extracts_and_offsets_to_document_span(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.returns = [
        {"start": 0, "end": 5, "label": "person", "score": 0.91},
        {"start": 10, "end": 13, "label": "person", "score": 0.77},
    ]
    chunk = _chunk("Alice and Bob spoke.", char_start=100)
    mentions = list(GlinerEntityExtractor({"labels": ["person"]}).extract([chunk]))
    assert [m.text for m in mentions] == ["Alice", "Bob"]
    assert all(isinstance(m, EntityMention) for m in mentions)
    # Chunk-local offsets shifted into the parent-document span.
    assert mentions[0].provenance.char_start == 100
    assert mentions[0].provenance.char_end == 105
    assert mentions[1].provenance.char_start == 110
    # Page range inherited from the parent chunk (S4 round-trip).
    assert mentions[0].provenance.page_start == 2
    # GLiNER's score becomes the mention confidence.
    assert mentions[0].confidence == 0.91
    # Links + classification stamped from the chunk.
    assert mentions[0].chunk_record_id == "doc1#chunk-0"
    assert mentions[0].classification.sensitivity == "internal"


def test_mention_page_resolves_to_its_own_page_in_multipage_chunk(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    # A chunk spanning two pages (carried page map): a mention on page 2 must cite
    # page 2, not inherit the chunk's first page (S4 "correct source page").
    assembly = PageMap.from_page_texts(["Alice is here.", "Bob is over here."])
    # page 1 [0,16), page 2 [16,33); "Alice" at 0, "Bob" at 16.
    fake_gliner.returns = [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": 16, "end": 19, "label": "person", "score": 0.9},
    ]
    chunk = _chunk(assembly.text, char_start=0, page_map=assembly.page_map)
    mentions = list(GlinerEntityExtractor({"labels": ["person"]}).extract([chunk]))
    by_text = {m.text: m for m in mentions}
    assert (by_text["Alice"].provenance.page_start, by_text["Alice"].provenance.page_end) == (1, 1)
    assert (by_text["Bob"].provenance.page_start, by_text["Bob"].provenance.page_end) == (2, 2)


def test_threshold_filters_low_scores(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.returns = [
        {"start": 0, "end": 5, "label": "person", "score": 0.4},
        {"start": 10, "end": 13, "label": "person", "score": 0.9},
    ]
    chunk = _chunk("Alice and Bob spoke.")
    out = list(GlinerEntityExtractor({"labels": ["person"], "threshold": 0.5}).extract([chunk]))
    assert [m.text for m in out] == ["Bob"]


def test_empty_chunk_skipped_without_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.returns = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    out = list(GlinerEntityExtractor({"labels": ["person"]}).extract([_chunk("")]))
    assert out == []


def test_missing_gliner_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the dep not being installed: importing gliner fails.
    monkeypatch.setitem(sys.modules, "gliner", None)
    extractor = GlinerEntityExtractor({"labels": ["person"]})
    with pytest.raises(ImportError, match="pip install latence-ner-gliner"):
        list(extractor.extract([_chunk("Alice spoke.")]))
