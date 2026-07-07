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
from latence_core.stages.relation import PatternRelationExtractor, _Candidate

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


def _entity_storm_chunk(count: int) -> tuple[ChunkRecord, list[EntityMention]]:
    """A pathological low-cardinality document: ``count`` same-typed mentions packed
    tightly together, the adversarial input the ADR-0033 / #63 quadratic-fan-out guard
    exists to bound. Every ORG mention sits within a small window of every other, so a
    symmetric no-trigger co-occurrence rule matches O(count²) pairs in one document.
    """
    # Build a doc text of adjacent "Xn " tokens so every pair falls inside the window.
    surfaces = [f"E{i:03d}" for i in range(count)]
    text = " ".join(surfaces)
    chunk = _doc_chunk(text, record_id="storm#chunk-0", document_record_id="storm")
    mentions: list[EntityMention] = []
    offset = 0
    for i, surface in enumerate(surfaces):
        start = text.index(surface, offset)
        end = start + len(surface)
        offset = end
        prov = chunk.provenance.model_copy(
            update={"char_start": start, "char_end": end, "page_start": 1, "page_end": 1}
        )
        mentions.append(
            EntityMention(
                record_id=f"storm#mention-{i}",
                document_record_id="storm",
                chunk_record_id="storm#chunk-0",
                index=i,
                label="ORG",
                text=surface,
                confidence=0.9,
                provenance=prov,
                classification=chunk.classification,
            )
        )
    return chunk, mentions


def test_quadratic_fanout_guard_caps_per_document_relations() -> None:
    # #63 / ADR-0033 regression: a low-cardinality mention storm with a symmetric
    # co-occurrence rule (no trigger, wide window) would otherwise emit O(n²) relations
    # from ONE document — the ~0.084·n² blowup G1 Lane 6 measured (5k docs -> ~12 GB,
    # killed). Reverting the cap makes this assertion fail (it would emit far more than
    # the cap), and the audited capped tallies would be 0.
    count = 60  # 60 mentions => up to 60*59/2 = 1770 symmetric pairs, uncapped.
    chunk, mentions = _entity_storm_chunk(count)
    cap = 100
    cfg = {
        "relations": {
            # Symmetric, no trigger => every ORG-ORG pair in-window is a candidate.
            "co_occurs": {"head": "ORG", "tail": "ORG", "symmetric": True},
        },
        "window_chars": 100_000,  # force every pair into one window
        "max_relations_per_document": cap,
    }
    provider = PatternRelationExtractor(cfg)
    relations = list(provider.relate([chunk], mentions))
    # The emitted relations are bounded by the cap — not the quadratic pair count.
    assert len(relations) == cap
    # The guard is AUDITED, not silent: exactly one document was capped and the dropped
    # count is the difference from the true (quadratic) candidate total.
    assert provider.fanout_capped_documents == 1
    total_pairs = count * (count - 1) // 2
    assert provider.fanout_dropped_relations == total_pairs - cap
    # Determinism: the retained relations are the lowest-ordered stable prefix, so a
    # second run yields byte-identical output.
    provider2 = PatternRelationExtractor(cfg)
    relations2 = list(provider2.relate([chunk], mentions))
    assert [r.record_id for r in relations] == [r.record_id for r in relations2]
    assert [r.model_dump_json() for r in relations] == [r.model_dump_json() for r in relations2]


def test_quadratic_fanout_guard_disabled_below_cap_is_noop() -> None:
    # A normal document under the cap is untouched and reports no capping.
    chunk = _doc_chunk("Alice works for Acme Corp.")
    mentions = _mentions(chunk)
    cfg = {
        "relations": {"works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]}},
        "max_relations_per_document": 10_000,
    }
    provider = PatternRelationExtractor(cfg)
    relations = list(provider.relate([chunk], mentions))
    assert len(relations) == 1
    assert provider.fanout_capped_documents == 0
    assert provider.fanout_dropped_relations == 0


def test_input_mention_cap_bounds_candidate_transient() -> None:
    # #68 regression: the #63 output cap trims the EMITTED relations, but the O(mentions²)
    # candidate list is still materialised in full BEFORE the sort trims it — unbounded
    # when a wide window_chars defeats the inner-loop break. The input-side mention cap
    # (max_mentions_per_document) bounds the mentions that reach the scan, so the transient
    # candidate list is O(cap²), not O(count²). Reverting the input cap makes _find_candidates
    # receive all `count` mentions and build the full quadratic transient — this assertion
    # (the scan sees at most `mention_cap` mentions) then fails, and the audited input tallies
    # would be 0.
    count = 400  # 400 same-typed mentions => 400*399/2 = 79_800 pairs, uncapped.
    chunk, mentions = _entity_storm_chunk(count)
    mention_cap = 50
    cfg = {
        "relations": {
            "co_occurs": {"head": "ORG", "tail": "ORG", "symmetric": True},
        },
        "window_chars": 100_000,  # wide window: the inner-loop break never fires.
        # Output cap generous so it does NOT do the trimming — the input cap must.
        "max_relations_per_document": 10_000,
        "max_mentions_per_document": mention_cap,
    }
    provider = PatternRelationExtractor(cfg)

    # Wrap _find_candidates to record how many mentions actually enter the O(mentions²) scan.
    seen_input_sizes: list[int] = []
    original = provider._find_candidates

    def _spy(ordered: list[EntityMention], text: str) -> list[_Candidate]:
        seen_input_sizes.append(len(ordered))
        return original(ordered, text)

    provider._find_candidates = _spy  # type: ignore[method-assign]
    relations = list(provider.relate([chunk], mentions))

    # The TRANSIENT is bounded on the input side: the scan never saw more than the cap.
    assert seen_input_sizes == [mention_cap]
    # So the emitted relations are O(cap²), not O(count²) — proving the output cap didn't
    # do the trimming (it was set generously above the resulting count).
    assert len(relations) == mention_cap * (mention_cap - 1) // 2
    assert provider.fanout_capped_documents == 0  # output cap not hit
    assert provider.fanout_dropped_relations == 0
    # The input cap IS audited: one document trimmed, the excess mentions dropped.
    assert provider.mention_capped_documents == 1
    assert provider.mention_dropped_mentions == count - mention_cap


def test_input_mention_cap_keeps_lowest_ordered_prefix_and_preserves_offsets() -> None:
    # #68: the retained mentions are the lowest-ordered document-order prefix (by char
    # offset, then record id), so the covering spans / provenance offsets of the retained
    # relations are exactly those of the first `cap` mentions — unchanged, not a random
    # subset. Determinism: a second run is byte-identical.
    count = 200
    chunk, mentions = _entity_storm_chunk(count)
    mention_cap = 20
    cfg = {
        "relations": {"co_occurs": {"head": "ORG", "tail": "ORG", "symmetric": True}},
        "window_chars": 100_000,
        "max_mentions_per_document": mention_cap,
    }
    provider = PatternRelationExtractor(cfg)
    relations = list(provider.relate([chunk], mentions))

    # Every relation's endpoints are among the first `cap` mentions in document order.
    kept_ids = {m.record_id for m in mentions[:mention_cap]}
    for r in relations:
        assert r.head_mention_id in kept_ids
        assert r.tail_mention_id in kept_ids
    # The covering span of every retained relation stays within the first cap mentions'
    # char extent — offsets are preserved, never shifted by the cap.
    max_kept_end = max(m.provenance.char_end or 0 for m in mentions[:mention_cap])
    for r in relations:
        assert r.provenance.char_start is not None
        assert r.provenance.char_end is not None
        assert r.provenance.char_end <= max_kept_end

    # Deterministic: a fresh Provider yields byte-identical output.
    provider2 = PatternRelationExtractor(cfg)
    relations2 = list(provider2.relate([chunk], mentions))
    assert [r.model_dump_json() for r in relations] == [r.model_dump_json() for r in relations2]


def test_input_mention_cap_disabled_below_cap_is_noop() -> None:
    # A normal document under the input cap is untouched and reports no input capping.
    chunk = _doc_chunk("Alice works for Acme Corp.")
    mentions = _mentions(chunk)
    cfg = {
        "relations": {"works_for": {"head": "PERSON", "tail": "ORG", "triggers": ["works for"]}},
        "max_mentions_per_document": 5_000,
    }
    provider = PatternRelationExtractor(cfg)
    relations = list(provider.relate([chunk], mentions))
    assert len(relations) == 1
    assert provider.mention_capped_documents == 0
    assert provider.mention_dropped_mentions == 0
