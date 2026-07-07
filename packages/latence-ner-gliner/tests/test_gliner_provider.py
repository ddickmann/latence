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
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
)
from latence_ner_gliner import GlinerEntityExtractor


def _chunk(
    content: str,
    *,
    char_start: int = 100,
    char_end: int | None = None,
    page_map: PageMap | None = None,
    offset_map: OffsetMap | None = None,
    record_id: str = "doc1#chunk-0",
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
        record_id=record_id,
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


def test_mention_offset_resolves_through_offset_map_past_stripped_markup(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    # Original "**Note:** Alice joined Globex." strips to "Note: Alice joined Globex.":
    # the stripped ``**`` before the prose shifts every later mention right in the
    # original, so char_start+local is a lower bound. The carried offset_map corrects it
    # — the exact learned-NER blind spot issue #48 names (the gazetteer fix must reach
    # here too, not just the in-core Provider).
    original = "**Note:** Alice joined Globex."
    content = "Note: Alice joined Globex."
    offset_map = OffsetMap(
        segments=[
            OffsetSegment(local_start=0, doc_start=2),
            OffsetSegment(local_start=5, doc_start=9),
        ],
        length=len(content),
    )
    fake_gliner.returns = [
        {"start": 6, "end": 11, "label": "person", "score": 0.9},
        {"start": 19, "end": 25, "label": "organization", "score": 0.9},
    ]
    chunk = _chunk(
        content, char_start=2, char_end=len(original), offset_map=offset_map
    )
    mentions = list(
        GlinerEntityExtractor({"labels": ["person", "organization"]}).extract([chunk])
    )
    by_text = {m.text: m for m in mentions}
    assert set(by_text) == {"Alice", "Globex"}
    for m in mentions:
        # TRUE original span (not the char_start+local lower bound) — would fail on the
        # bare shift: original[8:13] == 'Alice' is false ("ce jo" or ":** A" etc.).
        assert original[m.provenance.char_start : m.provenance.char_end] == m.text
    assert by_text["Alice"].provenance.char_start == 10  # not 2 + 6 = 8
    assert by_text["Globex"].provenance.char_start == 23  # not 2 + 19 = 21


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


def test_declares_profile_either_verified_apache() -> None:
    # P3-F1 (ADR-0036 §2): the GLiNER Provider declares a profile the registry surfaces.
    # compute="either" (CPU or GPU), verified Apache-2.0 (weights AND code, ADR-0012),
    # the default checkpoint as model_id, and deterministic=False (learned float inference).
    from latence_core.providers.profile import ProviderProfile, profile_of

    prof = profile_of(GlinerEntityExtractor)
    assert isinstance(prof, ProviderProfile)
    assert prof.compute == "either"
    assert prof.model_id == "urchade/gliner_multi-v2.1"
    assert prof.license == "Apache-2.0"
    assert prof.license_verified is True
    assert prof.deterministic is False
    assert prof.batch is True
    assert prof.cost_per_1k is None
