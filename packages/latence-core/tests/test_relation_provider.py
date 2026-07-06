"""PatternRelationExtractor: zero-shot trigger-phrase Relation Extraction (S5).

Unit tests for the in-core reference Provider: happy-path typed relations with
head/tail mention refs + a covering document span that resolves to the correct
source page (the S5 round-trip), zero-shot relation-label configurability,
deterministic ordering, the window gate, symmetric relations, config validation,
and the no-relatable-pair / fewer-than-two-mention degradations.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    PageMap,
    Provenance,
)
from latence_core.stages.entity import GazetteerEntityExtractor
from latence_core.stages.relation import PatternRelationExtractor

_ENTITY_CONFIG = {
    "labels": {
        "PERSON": {"terms": ["Alice", "Bob"]},
        "ORG": {"terms": ["Acme Corp", "Globex"]},
        "LOC": {"terms": ["Berlin", "Paris"]},
    }
}


def _doc_chunk(
    text: str,
    *,
    char_start: int = 0,
    page_map: PageMap | None = None,
    record_id: str = "d1#chunk-0",
    document_record_id: str = "d1",
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
        record_id=record_id,
        document_record_id=document_record_id,
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=text,
        token_count=len(text.split()),
        page_map=page_map,
    )


def _mentions(chunk: ChunkRecord) -> list[EntityMention]:
    return list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract([chunk]))


def test_extracts_typed_relation_with_trigger() -> None:
    chunk = _doc_chunk("Alice works for Acme Corp.")
    mentions = _mentions(chunk)
    cfg = {
        "relations": {
            "works_for": {
                "head": "PERSON",
                "tail": "ORG",
                "triggers": ["works for"],
                "confidence": 0.9,
            }
        }
    }
    relations = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    assert len(relations) == 1
    r = relations[0]
    assert r.label == "works_for"
    assert r.head_text == "Alice"
    assert r.tail_text == "Acme Corp"
    assert r.confidence == 0.9
    # Head/tail refs point at the real mention record_ids.
    ids = {m.text: m.record_id for m in mentions}
    assert r.head_mention_id == ids["Alice"]
    assert r.tail_mention_id == ids["Acme Corp"]
    # Covering span spans both endpoints; resolves within the assembled text.
    assert r.provenance.char_start == 0
    assert chunk.content[r.provenance.char_start : r.provenance.char_end].startswith("Alice")
    # Classification inherited from the head mention (via the chunk), unchanged.
    assert r.classification.sensitivity == "internal"


def test_trigger_gates_the_relation() -> None:
    # Same two entity types, but the required trigger phrase is absent between them.
    chunk = _doc_chunk("Alice and Acme Corp both attended.")
    mentions = _mentions(chunk)
    cfg = {"relations": {"works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]}}}
    assert list(PatternRelationExtractor(cfg).relate([chunk], mentions)) == []


def test_zero_shot_relation_labels_configurable_not_hardcoded() -> None:
    # A different domain: gene/drug relations from the SAME Provider.
    chunk = _doc_chunk("BRCA1 is targeted by aspirin.", record_id="d1#chunk-0")
    ent_cfg = {"labels": {"GENE": {"terms": ["BRCA1"]}, "DRUG": {"terms": ["aspirin"]}}}
    mentions = list(GazetteerEntityExtractor(ent_cfg).extract([chunk]))
    cfg = {
        "relations": {
            "targeted_by": {"head": "GENE", "tail": "DRUG", "triggers": ["targeted by"]}
        }
    }
    relations = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    assert [r.label for r in relations] == ["targeted_by"]


def test_relation_page_resolves_via_covering_span_in_multipage_chunk() -> None:
    # Two pages; head on page 1, tail on page 2 -> the covering span straddles the
    # boundary, so the relation records both pages (S5 "correct source page").
    assembly = PageMap.from_page_texts(["Alice is here.", "She works for Globex now."])
    # page 1 [0,16), page 2 [16,...); "Alice" on p1, "Globex" on p2.
    chunk = _doc_chunk(assembly.text, char_start=0, page_map=assembly.page_map)
    mentions = _mentions(chunk)
    cfg = {
        "relations": {
            "works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]}
        }
    }
    relations = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    assert len(relations) == 1
    r = relations[0]
    assert r.provenance.page_start == 1
    assert r.provenance.page_end == 2


def test_window_gate_drops_distant_pairs() -> None:
    # A trigger elsewhere in the doc must not relate two far-apart mentions.
    far = "Alice. " + "x" * 300 + " Acme Corp works for someone."
    chunk = _doc_chunk(far)
    mentions = _mentions(chunk)
    cfg = {
        "relations": {
            "works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]}
        },
        "window_chars": 50,
    }
    # Alice..Acme Corp are >50 chars apart, so no relation despite a trigger later.
    assert list(PatternRelationExtractor(cfg).relate([chunk], mentions)) == []


def test_symmetric_relation_matches_reverse_order() -> None:
    # Text order is LOC ... ORG (Berlin before Acme Corp), with the trigger BETWEEN
    # them. The rule declares ORG->LOC; symmetric lets the reverse text order match,
    # and the emitted head is still the ORG (the rule's head type).
    chunk = _doc_chunk("Berlin is home to Acme Corp.")
    mentions = _mentions(chunk)
    cfg = {
        "relations": {
            "located_in": {
                "head": "ORG",
                "tail": "LOC",
                "triggers": ["is home to"],
                "symmetric": True,
            }
        }
    }
    relations = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    assert len(relations) == 1
    assert relations[0].head_text == "Acme Corp"
    assert relations[0].tail_text == "Berlin"


def test_pure_cooccurrence_rule_when_no_triggers() -> None:
    # A rule with no triggers/patterns is a pure in-window co-occurrence rule.
    chunk = _doc_chunk("Alice, Acme Corp.")
    mentions = _mentions(chunk)
    cfg = {"relations": {"associated_with": {"head": "PERSON", "tail": "ORG"}}}
    relations = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    assert [r.label for r in relations] == ["associated_with"]


def test_deterministic_ordering_and_indices() -> None:
    chunk = _doc_chunk("Alice works for Acme Corp in Berlin.")
    mentions = _mentions(chunk)
    cfg = {
        "relations": {
            "works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]},
            "located_in": {"head": "ORG", "tail": "LOC", "triggers": ["in"]},
        }
    }
    a = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    b = list(PatternRelationExtractor(cfg).relate([chunk], mentions))
    # Byte-identical across runs (hash-stable, seeded Baseline bar).
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    # Ordered by covering-span start; indices 0..n in that order.
    starts = [int(r.provenance.char_start or 0) for r in a]
    assert starts == sorted(starts)
    assert [r.index for r in a] == list(range(len(a)))


def test_fewer_than_two_mentions_yields_no_relations() -> None:
    chunk = _doc_chunk("Alice waved.")
    mentions = _mentions(chunk)
    assert len(mentions) == 1
    cfg = {"relations": {"works_for": {"head": "PERSON", "tail": "ORG"}}}
    assert list(PatternRelationExtractor(cfg).relate([chunk], mentions)) == []


def test_no_matching_pair_yields_no_relations() -> None:
    # Two PERSON mentions, but the rule needs PERSON->ORG.
    chunk = _doc_chunk("Alice met Bob.")
    mentions = _mentions(chunk)
    cfg = {"relations": {"works_for": {"head": "PERSON", "tail": "ORG"}}}
    assert list(PatternRelationExtractor(cfg).relate([chunk], mentions)) == []


def test_no_relations_configured_extracts_nothing() -> None:
    chunk = _doc_chunk("Alice works for Acme Corp.")
    mentions = _mentions(chunk)
    assert list(PatternRelationExtractor({}).relate([chunk], mentions)) == []
    assert list(PatternRelationExtractor({"relations": ["works_for"]}).relate([chunk], mentions)) == []


def test_relation_requires_head_and_tail_labels() -> None:
    with pytest.raises(ValueError, match="must declare 'head' and 'tail'"):
        PatternRelationExtractor({"relations": {"bad": {"triggers": ["x"]}}})


# -- issue #30 (hardening): _document_texts memory amplification fuzz --------


def test_document_texts_reconstructs_and_stays_bounded_under_fuzz() -> None:
    """Seeded fuzz over adversarial chunk layouts: correct placement, bounded buffer.

    ``_document_texts`` allocates ``buf = [" "] * end`` where ``end`` is the max
    ``char_start + len(content)`` — bounded by Intake Screening's size cap upstream,
    but exercised here with sparse/large offsets, overlaps, and empty chunks to prove
    it never mis-places content nor allocates beyond the true document span (issue #30).
    Deterministic: a fixed seed keeps the run hash-stable (Baseline bar).
    """
    import random

    rng = random.Random(1730)
    for _ in range(200):
        n_chunks = rng.randint(1, 8)
        chunks: list[ChunkRecord] = []
        for i in range(n_chunks):
            start = rng.randint(0, 5_000)
            length = rng.randint(0, 40)
            content = "".join(rng.choice("abcXYZ ") for _ in range(length))
            chunks.append(
                _doc_chunk(content, char_start=start, record_id=f"d1#chunk-{i}")
            )
        texts = PatternRelationExtractor._document_texts(chunks)
        assert set(texts) == {"d1"}
        reconstructed = texts["d1"]
        expected_end = max((c.provenance.char_start or 0) + len(c.content) for c in chunks)
        # The buffer is exactly the document span — no amplification past the last char.
        assert len(reconstructed) == expected_end
        # Every chunk's content sits at its own offset (last writer wins on overlap,
        # matching the in-place slice assignment).
        placed = [" "] * expected_end
        for c in chunks:
            s = c.provenance.char_start or 0
            placed[s : s + len(c.content)] = c.content
        assert reconstructed == "".join(placed)


def test_relations_scoped_per_document() -> None:
    # Two documents' mentions in one pass; a relation never crosses documents.
    c0 = _doc_chunk("Alice works for Acme Corp.", record_id="d1#chunk-0", document_record_id="d1")
    c1 = _doc_chunk(
        "Bob works for Globex.",
        char_start=0,
        record_id="d2#chunk-0",
        document_record_id="d2",
    )
    # d2's provenance document_id differs only logically; give it its own doc id.
    c1 = c1.model_copy(
        update={"provenance": c1.provenance.model_copy(update={"document_id": "sha256:d2"})}
    )
    m0 = list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract([c0]))
    m1 = list(GazetteerEntityExtractor(_ENTITY_CONFIG).extract([c1]))
    cfg = {"relations": {"works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]}}}
    relations = list(PatternRelationExtractor(cfg).relate([c0, c1], m0 + m1))
    by_doc = {r.document_record_id for r in relations}
    assert by_doc == {"d1", "d2"}
    # Each relation's endpoints belong to its own document.
    for r in relations:
        assert r.head_mention_id.startswith(r.document_record_id)
        assert r.tail_mention_id.startswith(r.document_record_id)
