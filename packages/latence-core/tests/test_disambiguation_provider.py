"""CascadeDisambiguator (S8, in-core reference): mentions (+ relations) → canonical entities.

Unit tests for the reference Provider: it emits one ENTITY-scope DisambiguationRecord per
resolved cluster (canonical-text order) then RELATION-scope records; a canonical entity
inherits the page-accurate Provenance of its representative mention (S8 page-map integration),
carries Evidence + the audited merge decisions (including a below-policy one, no silent
over-merge), links to a configured KB with a graceful fallback for the unlinked, and remaps
relation endpoints onto canonical entities (dropping unresolved ones). Deterministic.
"""

from __future__ import annotations

from latence_core.contracts import (
    Classification,
    DisambiguationScope,
    EntityMention,
    MergeStrategy,
    Provenance,
    RelationMention,
)
from latence_core.stages.disambiguation import CascadeDisambiguator


def _mention(
    mid: str, doc: str, text: str, label: str, start: int, *, page: int = 1, conf: float = 0.9
) -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///c/{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc}",
        char_start=start,
        char_end=start + len(text),
        page_start=page,
        page_end=page,
    )
    return EntityMention(
        record_id=mid,
        document_record_id=doc,
        chunk_record_id=f"{doc}#chunk-0",
        index=0,
        label=label,
        text=text,
        confidence=conf,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
    )


def _relation(rid: str, doc: str, label: str, head: EntityMention, tail: EntityMention) -> RelationMention:
    lo = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
    hi = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
    prov = head.provenance.model_copy(
        update={"char_start": lo, "char_end": hi, "page_start": 1, "page_end": 1}
    )
    return RelationMention(
        record_id=rid,
        document_record_id=doc,
        index=0,
        label=label,
        head_mention_id=head.record_id,
        tail_mention_id=tail.record_id,
        head_text=head.text,
        tail_text=tail.text,
        confidence=0.8,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
    )


# -- entity resolution --------------------------------------------------------


def test_cross_document_mentions_merge_into_one_canonical_entity() -> None:
    mentions = [
        _mention("m1", "d1", "Acme", "ORG", 0),
        _mention("m2", "d2", "acme", "ORG", 5),
        _mention("m3", "d1", "Alice", "PERSON", 10),
    ]
    records = list(CascadeDisambiguator().disambiguate(mentions, []))
    entities = [r for r in records if r.scope is DisambiguationScope.ENTITY]
    assert len(entities) == 2  # {Acme, acme} + {Alice}
    acme = next(r for r in entities if r.entity and r.entity.canonical_text.lower() == "acme")
    assert acme.entity is not None
    assert sorted(acme.entity.member_mention_ids) == ["m1", "m2"]
    assert sorted(acme.entity.source_document_ids) == ["sha256:d1", "sha256:d2"]


def test_canonical_entity_inherits_page_accurate_provenance() -> None:
    m = _mention("m1", "d1", "Acme", "ORG", 40, page=7)
    [record] = list(CascadeDisambiguator().disambiguate([m], []))
    assert record.entity is not None
    # The canonical entity cites the representative mention's exact page (S8 page-map sub-task).
    assert record.provenance.page_start == 7
    assert record.provenance.char_start == 40


def test_below_policy_merge_is_recorded_not_applied() -> None:
    # "Acme" within "Acme Corporation" scores substring ratio 0.5, below the 0.82 policy:
    # the two stay separate AND the decision is logged with applied=False on the emitted
    # target entity — the audit half of "no silent over-merge" (S8 AC), delivered at the
    # Provider layer (not just in the engine), so the Quality Report can count it.
    mentions = [_mention("m1", "d1", "Acme", "ORG", 0), _mention("m2", "d2", "Acme Corporation", "ORG", 0)]
    records = list(CascadeDisambiguator().disambiguate(mentions, []))
    entities = [r for r in records if r.scope is DisambiguationScope.ENTITY]
    assert len(entities) == 2  # not merged

    # The below-policy decision must survive onto exactly one emitted canonical entity, with
    # applied=False and the SUBSTRING strategy. Without the audit being attached to a cluster
    # it would be silently dropped and merges_below_policy would be a permanent 0.
    below = [
        d
        for r in entities
        if r.entity is not None
        for d in r.entity.merge_decisions
        if not d.applied
    ]
    assert len(below) == 1
    assert below[0].strategy is MergeStrategy.SUBSTRING
    assert "below policy" in below[0].reason.lower()
    # And it is not double-counted as an applied merge anywhere.
    assert not any(
        d.applied
        for r in entities
        if r.entity is not None
        for d in r.entity.merge_decisions
    )


def test_applied_merge_carries_audited_decision_and_evidence() -> None:
    mentions = [
        _mention("m1", "d1", "International Business Machines", "ORG", 0),
        _mention("m2", "d2", "IBM", "ORG", 0),
    ]
    records = list(CascadeDisambiguator().disambiguate(mentions, []))
    entity = next(r for r in records if r.scope is DisambiguationScope.ENTITY).entity
    assert entity is not None
    assert entity.mention_count == 2
    applied = [d for d in entity.merge_decisions if d.applied]
    assert applied and applied[0].strategy is MergeStrategy.ACRONYM
    # Evidence links back to the source mentions.
    assert set(entity.evidence.mention_ids) == {"m1", "m2"}


# -- KB linking ---------------------------------------------------------------


def test_kb_link_and_graceful_fallback() -> None:
    config = {
        "kb": [{"kb_id": "Q1", "name": "Acme Corporation", "entity_type": "ORG", "aliases": ["Acme"]}],
        "kb_threshold": 0.5,
    }
    mentions = [_mention("m1", "d1", "Acme", "ORG", 0), _mention("m2", "d2", "Globex", "ORG", 0)]
    records = list(CascadeDisambiguator(config).disambiguate(mentions, []))
    entities = {r.entity.canonical_text: r.entity for r in records if r.entity}
    assert entities["Acme"].kb_id == "Q1"          # linked
    assert entities["Globex"].kb_id is None         # graceful fallback (unlinked)


# -- relation normalisation ---------------------------------------------------


def test_relations_remap_onto_canonical_entities() -> None:
    alice = _mention("m1", "d1", "Alice", "PERSON", 0)
    acme = _mention("m2", "d1", "Acme", "ORG", 20)
    rel = _relation("r1", "d1", "employed_by", alice, acme)
    config = {"label_aliases": {"employed_by": "works_for"}}
    records = list(CascadeDisambiguator(config).disambiguate([alice, acme], [rel]))
    relations = [r for r in records if r.scope is DisambiguationScope.RELATION]
    assert len(relations) == 1
    nr = relations[0].relation
    assert nr is not None
    assert nr.label == "works_for"
    assert nr.head_entity_id == "entity:alice"
    assert nr.tail_entity_id == "entity:acme"
    assert nr.head_text == "Alice" and nr.tail_text == "Acme"


def test_relation_inverse_detection_swaps_direction_on_canonical_entities() -> None:
    # The Provider must carry the engine's inverse detection all the way onto the emitted
    # NormalizedRelation: `employs` is configured as the inverse of `works_for`, so
    # "Acme employs Alice" is rewritten to "Alice works_for Acme" with inverted=True and the
    # canonical endpoints swapped (Alice as head, Acme as tail). Only the pure engine had a
    # test before; this pins the delivered contract-level behaviour.
    alice = _mention("m1", "d1", "Alice", "PERSON", 0)
    acme = _mention("m2", "d1", "Acme", "ORG", 20)
    rel = _relation("r1", "d1", "employs", acme, alice)  # head=Acme, tail=Alice
    config = {"inverses": {"employs": "works_for"}}
    records = list(CascadeDisambiguator(config).disambiguate([alice, acme], [rel]))
    relations = [r.relation for r in records if r.scope is DisambiguationScope.RELATION]
    assert len(relations) == 1
    nr = relations[0]
    assert nr is not None
    assert nr.label == "works_for"
    assert nr.inverted is True
    # Direction swapped: the person is now the head, the org the tail.
    assert nr.head_entity_id == "entity:alice"
    assert nr.tail_entity_id == "entity:acme"
    assert nr.head_text == "Alice" and nr.tail_text == "Acme"


def test_relation_with_unresolved_endpoint_is_dropped() -> None:
    alice = _mention("m1", "d1", "Alice", "PERSON", 0)
    acme = _mention("m2", "d1", "Acme", "ORG", 20)
    # A relation referencing a mention id that was not in the mention set: endpoint unresolved.
    ghost = _mention("m9", "d1", "Ghost", "ORG", 40)
    rel = _relation("r1", "d1", "works_for", alice, ghost)
    records = list(CascadeDisambiguator().disambiguate([alice, acme], [rel]))
    assert [r for r in records if r.scope is DisambiguationScope.RELATION] == []


# -- ordering / determinism ---------------------------------------------------


def test_entities_before_relations_and_deterministic() -> None:
    alice = _mention("m1", "d1", "Alice", "PERSON", 0)
    acme = _mention("m2", "d1", "Acme", "ORG", 20)
    rel = _relation("r1", "d1", "works_for", alice, acme)
    a = [r.model_dump_json() for r in CascadeDisambiguator().disambiguate([alice, acme], [rel])]
    b = [r.model_dump_json() for r in CascadeDisambiguator().disambiguate([acme, alice], [rel])]
    assert a == b
    scopes = [r.scope for r in CascadeDisambiguator().disambiguate([alice, acme], [rel])]
    # All ENTITY records come before any RELATION record.
    assert scopes == sorted(scopes, key=lambda s: 0 if s is DisambiguationScope.ENTITY else 1)


def test_empty_mentions_yield_no_records() -> None:
    assert list(CascadeDisambiguator().disambiguate([], [])) == []
