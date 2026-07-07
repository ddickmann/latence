"""Second reference Disambiguator Provider (#77): ExactSurfaceDisambiguator parity + benchmark.

The 'providers' hardening bar requires ≥2 reference providers per Capability with parity
tests (same typed contracts) + a benchmark vs the existing one.
:class:`ExactSurfaceDisambiguator` is the second CPU Disambiguator behind the same seam as
:class:`CascadeDisambiguator`.

* **Parity**: both satisfy the ``Disambiguator`` protocol and emit the same typed contract —
  ENTITY records (canonical entities with Evidence, page-accurate Provenance, KB link,
  audited merges) then normalized RELATION records; deterministic; no silent over-merge.
* **Benchmark / difference**: the exact-surface Provider merges ONLY identical normalized
  surfaces + configured aliases (high precision, lower recall), where the cascade also fuzzy-
  merges via acronym/substring. The "does not fuzzy-merge" assertion is the reverting
  regression guard — re-enabling the fuzzy rungs would FAIL it.
"""

from __future__ import annotations

from latence_core.capability import Disambiguator
from latence_core.contracts import (
    Classification,
    DisambiguationScope,
    EntityMention,
    Provenance,
    RelationMention,
)
from latence_core.stages.disambiguation import (
    CascadeDisambiguator,
    ExactSurfaceDisambiguator,
)


def _mention(mid: str, doc: str, text: str, label: str, start: int = 0) -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc}",
        char_start=start,
        char_end=start + len(text),
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=mid,
        document_record_id=doc,
        chunk_record_id=f"{doc}#chunk-0",
        index=0,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
    )


def _relation(rid: str, doc: str, head: EntityMention, tail: EntityMention) -> RelationMention:
    lo = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
    hi = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
    prov = head.provenance.model_copy(
        update={"char_start": lo, "char_end": hi, "page_start": 1, "page_end": 1}
    )
    return RelationMention(
        record_id=rid,
        document_record_id=doc,
        index=0,
        label="works_for",
        head_mention_id=head.record_id,
        tail_mention_id=tail.record_id,
        head_text=head.text,
        tail_text=tail.text,
        confidence=0.8,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
    )


def test_second_disambiguator_satisfies_the_same_seam() -> None:
    assert isinstance(ExactSurfaceDisambiguator(), Disambiguator)


def test_second_disambiguator_merges_exact_normalized_surfaces() -> None:
    """Parity: identical normalized surfaces still merge cross-document (case-insensitive)."""
    mentions = [
        _mention("m1", "d1", "Acme", "ORG", 0),
        _mention("m2", "d2", "acme", "ORG", 5),  # same surface, different case + document
        _mention("m3", "d1", "Alice", "PERSON", 10),
    ]
    entities = [
        r for r in ExactSurfaceDisambiguator().disambiguate(mentions, [])
        if r.scope is DisambiguationScope.ENTITY
    ]
    assert len(entities) == 2  # {Acme, acme} + {Alice}
    acme = next(r.entity for r in entities if r.entity and r.entity.canonical_text.lower() == "acme")
    assert acme is not None
    assert sorted(acme.member_mention_ids) == ["m1", "m2"]
    assert set(acme.source_document_ids) == {"sha256:d1", "sha256:d2"}
    assert acme.evidence.mention_ids  # every entity carries Evidence (parity)


def test_benchmark_does_not_fuzzy_merge_where_cascade_does() -> None:
    """Benchmark: cascade acronym-merges IBM↔International Business Machines; exact does NOT."""
    mentions = [
        _mention("m1", "d1", "IBM", "ORG", 0),
        _mention("m2", "d2", "International Business Machines", "ORG", 10),
        _mention("m3", "d1", "IBM", "ORG", 40),  # exact duplicate
    ]
    cascade = [
        r for r in CascadeDisambiguator().disambiguate(mentions, [])
        if r.scope is DisambiguationScope.ENTITY
    ]
    exact = [
        r for r in ExactSurfaceDisambiguator().disambiguate(mentions, [])
        if r.scope is DisambiguationScope.ENTITY
    ]
    # Cascade folds all three into one entity via the acronym rung.
    assert len(cascade) == 1
    # Exact keeps IBM (the two exact mentions) separate from the full name — higher precision.
    assert len(exact) == 2
    names = sorted(r.entity.canonical_text for r in exact if r.entity is not None)
    assert names == ["IBM", "International Business Machines"]
    # And it produced NO fuzzy below-policy audit (exact_only stops before the fuzzy rungs).
    for r in exact:
        assert r.entity is not None
        for decision in r.entity.merge_decisions:
            assert decision.strategy.value in {"exact", "alias"}


def test_second_disambiguator_merges_via_configured_alias() -> None:
    """An explicit alias is the sanctioned way to unify two distinct surfaces (precision)."""
    mentions = [
        _mention("m1", "d1", "IBM", "ORG", 0),
        _mention("m2", "d2", "International Business Machines", "ORG", 10),
    ]
    provider = ExactSurfaceDisambiguator(
        {"aliases": {"International Business Machines": "IBM"}}
    )
    entities = [
        r for r in provider.disambiguate(mentions, [])
        if r.scope is DisambiguationScope.ENTITY
    ]
    assert len(entities) == 1  # the alias unifies them
    entity = entities[0].entity
    assert entity is not None
    assert sorted(entity.member_mention_ids) == ["m1", "m2"]


def test_second_disambiguator_normalizes_relations_onto_canonical_entities() -> None:
    """Relation parity: endpoints remap onto canonical entities; unresolved are dropped."""
    acme1 = _mention("m1", "d1", "Acme", "ORG", 0)
    alice = _mention("m2", "d1", "Alice", "PERSON", 10)
    mentions = [acme1, alice]
    relations = [_relation("r1", "d1", alice, acme1)]
    records = list(ExactSurfaceDisambiguator().disambiguate(mentions, relations))
    rel_records = [r for r in records if r.scope is DisambiguationScope.RELATION]
    assert len(rel_records) == 1
    normalized = rel_records[0].relation
    assert normalized is not None
    assert normalized.head_entity_id != normalized.tail_entity_id  # remapped, not self-loop
    assert normalized.evidence.mention_ids  # relation Evidence carried


def test_second_disambiguator_is_deterministic() -> None:
    mentions = [
        _mention("m1", "d1", "Acme", "ORG", 0),
        _mention("m2", "d2", "Acme", "ORG", 5),
        _mention("m3", "d1", "Beta", "ORG", 10),
    ]
    provider = ExactSurfaceDisambiguator()
    first = [
        (r.record_id, r.scope) for r in provider.disambiguate(mentions, [])
    ]
    second = [
        (r.record_id, r.scope) for r in provider.disambiguate(mentions, [])
    ]
    assert first == second
