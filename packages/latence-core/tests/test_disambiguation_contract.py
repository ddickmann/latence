"""S8 Disambiguation contract validation.

The scoped DisambiguationRecord enforces Provenance + Classification (every Record does),
scope/payload consistency (an ENTITY record must carry an entity and no relation; a RELATION
record the inverse), CanonicalEntity member uniqueness, NormalizedRelation distinct
endpoints, and Evidence/MergeDecision well-formedness — so a Provider cannot emit a mis-scoped
or unjustified record.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    CanonicalEntity,
    Classification,
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    MergeDecision,
    MergeStrategy,
    NormalizedRelation,
    Provenance,
)
from pydantic import ValidationError


def _prov(doc_id: str = "d1") -> Provenance:
    return Provenance(
        source_uri=f"file:///c/{doc_id}.txt",
        file_name=f"{doc_id}.txt",
        file_type="txt",
        file_size=10,
        document_id=f"sha256:{doc_id}",
        char_start=0,
        char_end=5,
        page_start=1,
        page_end=1,
    )


def _classification() -> Classification:
    return Classification(language="en", sensitivity="public")


def _evidence(conf: float = 0.9) -> Evidence:
    return Evidence(mention_ids=["m1"], document_ids=["sha256:d1"], snippet="x", confidence=conf)


def _entity(**kw: object) -> CanonicalEntity:
    defaults: dict[str, object] = {
        "record_id": "entity:acme",
        "canonical_text": "Acme",
        "entity_type": "ORG",
        "member_mention_ids": ["m1", "m2"],
        "source_document_ids": ["sha256:d1"],
        "evidence": _evidence(),
        "provenance": _prov(),
        "classification": _classification(),
    }
    defaults.update(kw)
    return CanonicalEntity(**defaults)  # type: ignore[arg-type]


# -- Provenance + Classification enforced -------------------------------------


def test_canonical_entity_requires_provenance_and_classification() -> None:
    with pytest.raises(ValidationError):
        CanonicalEntity(  # type: ignore[call-arg]
            record_id="e1",
            canonical_text="Acme",
            entity_type="ORG",
            member_mention_ids=["m1"],
            evidence=_evidence(),
        )


def test_disambiguation_record_stamps_schema_version() -> None:
    record = DisambiguationRecord(
        record_id="e1#canonical",
        scope=DisambiguationScope.ENTITY,
        entity=_entity(),
        provenance=_prov(),
        classification=_classification(),
    )
    assert record.schema_version == SCHEMA_VERSION


# -- scope / payload consistency ---------------------------------------------


def test_entity_scope_must_carry_entity_not_relation() -> None:
    with pytest.raises(ValidationError):
        DisambiguationRecord(
            record_id="e1",
            scope=DisambiguationScope.ENTITY,
            entity=None,
            provenance=_prov(),
            classification=_classification(),
        )


def test_entity_scope_rejects_a_relation_payload() -> None:
    rel = NormalizedRelation(
        record_id="r1#normalized",
        source_relation_id="r1",
        label="works_for",
        head_entity_id="entity:alice",
        tail_entity_id="entity:acme",
        head_text="Alice",
        tail_text="Acme",
        confidence=0.8,
        evidence=_evidence(),
        provenance=_prov(),
        classification=_classification(),
    )
    with pytest.raises(ValidationError):
        DisambiguationRecord(
            record_id="e1",
            scope=DisambiguationScope.ENTITY,
            entity=_entity(),
            relation=rel,
            provenance=_prov(),
            classification=_classification(),
        )


def test_relation_scope_must_carry_relation() -> None:
    with pytest.raises(ValidationError):
        DisambiguationRecord(
            record_id="r1",
            scope=DisambiguationScope.RELATION,
            relation=None,
            provenance=_prov(),
            classification=_classification(),
        )


# -- CanonicalEntity invariants ----------------------------------------------


def test_canonical_entity_member_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        _entity(member_mention_ids=["m1", "m1"])


def test_canonical_entity_requires_at_least_one_member() -> None:
    with pytest.raises(ValidationError):
        _entity(member_mention_ids=[])


def test_canonical_entity_linked_and_mention_count() -> None:
    linked = _entity(kb_id="Q1")
    assert linked.linked is True
    assert linked.mention_count == 2
    unlinked = _entity(kb_id=None, member_mention_ids=["m1"])
    assert unlinked.linked is False
    assert unlinked.mention_count == 1


# -- NormalizedRelation invariants -------------------------------------------


def test_normalized_relation_endpoints_must_differ() -> None:
    with pytest.raises(ValidationError):
        NormalizedRelation(
            record_id="r1#normalized",
            source_relation_id="r1",
            label="works_for",
            head_entity_id="entity:same",
            tail_entity_id="entity:same",
            head_text="X",
            tail_text="X",
            confidence=0.8,
            evidence=_evidence(),
            provenance=_prov(),
            classification=_classification(),
        )


# -- MergeDecision / Evidence -------------------------------------------------


def test_merge_decision_records_below_policy_as_not_applied() -> None:
    decision = MergeDecision(
        source_text="Acme Corp",
        target_text="Acme",
        strategy=MergeStrategy.SUBSTRING,
        confidence=0.5,
        applied=False,
        reason="substring below policy",
        evidence=_evidence(0.5),
    )
    assert decision.applied is False
    assert decision.strategy is MergeStrategy.SUBSTRING


def test_evidence_confidence_bounded() -> None:
    with pytest.raises(ValidationError):
        Evidence(mention_ids=["m1"], confidence=1.5)
