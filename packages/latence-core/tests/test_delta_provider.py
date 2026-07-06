"""S11 DeltaProcessor reference-Provider tests: registration + retract/splice + reconciliation."""

from __future__ import annotations

from latence_core.capability import DeltaOutcome, DeltaProcessor, ProviderRegistry
from latence_core.contracts import (
    CanonicalEntity,
    Classification,
    Delta,
    DeltaEntry,
    DeltaKind,
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    Provenance,
)


def _entity(record_id: str, canonical: str, members: int, docs: list[str]) -> DisambiguationRecord:
    prov = Provenance(
        source_uri="mem://x",
        file_name="x.txt",
        file_type="txt",
        file_size=1,
        document_id=docs[0],
        char_start=0,
        char_end=1,
    )
    cls = Classification(language="en")
    entity = CanonicalEntity(
        record_id=record_id,
        canonical_text=canonical,
        entity_type="ORG",
        member_mention_ids=[f"{record_id}:m{i}" for i in range(members)],
        source_document_ids=docs,
        evidence=Evidence(mention_ids=[f"{record_id}:m0"], document_ids=docs, confidence=0.9),
        provenance=prov,
        classification=cls,
    )
    return DisambiguationRecord(
        record_id=f"{record_id}#r",
        scope=DisambiguationScope.ENTITY,
        entity=entity,
        provenance=prov,
        classification=cls,
    )


def _load() -> DeltaProcessor:
    provider_cls = ProviderRegistry().load("delta.affected_set")
    provider = provider_cls({})
    assert isinstance(provider, DeltaProcessor)
    return provider


def test_delta_provider_is_registered_via_entry_point() -> None:
    assert "delta.affected_set" in ProviderRegistry().names()


def test_delta_provider_add_merges_affected_region() -> None:
    processor = _load()
    parent = [_entity("ibm", "IBM", 1, ["doc-a"])]
    current = [_entity("ibm", "IBM", 2, ["doc-a", "doc-b"])]  # merged after adding doc-b
    delta = Delta(
        parent_version=0,
        entries=[DeltaEntry(kind=DeltaKind.ADDED, document_id="doc-b", content_hash="sha256:b")],
        drift=0.5,
        triggered_reconciliation=False,
    )
    outcome = processor.apply_delta(parent, current, delta)
    assert isinstance(outcome, DeltaOutcome)
    # The merged entity (2 members) replaces the parent's stale 1-member entity.
    live = {r.entity.canonical_text: r.entity.mention_count for r in outcome.records if r.entity}
    assert live["IBM"] == 2
    assert outcome.entities_merged >= 1


def test_delta_provider_reconciliation_replaces_whole_corpus() -> None:
    processor = _load()
    parent = [_entity("old", "Old", 1, ["doc-a"])]
    current = [_entity("new", "New", 1, ["doc-a"])]
    delta = Delta(
        parent_version=0,
        entries=[DeltaEntry(kind=DeltaKind.UPDATED, document_id="doc-a", content_hash="sha256:x")],
        drift=1.0,
        triggered_reconciliation=True,
    )
    outcome = processor.apply_delta(parent, current, delta)
    # On a reconciliation every parent record is retracted and the current set is the live set.
    assert {r.record_id for r in outcome.records} == {"new#r"}
    assert {r.record_id for r in outcome.retracted} == {"old#r"}


def test_delta_provider_add_keeps_untouched_record_verbatim() -> None:
    # The defining property of the slice (AC#1 / ADR-0018): an add touches ONLY the affected set.
    # A parent entity from an UNRELATED, untouched document must be carried forward as the SAME
    # object (identity), not re-derived — proving the surgical splice, not a full recompute. The
    # affected entity IS superseded by the current one, so the two paths are distinguishable.
    processor = _load()
    untouched = _entity("globex", "Globex", 1, ["doc-z"])  # unrelated to the add on doc-b
    affected_parent = _entity("ibm", "IBM", 1, ["doc-a"])
    parent = [affected_parent, untouched]
    current = [_entity("ibm", "IBM", 2, ["doc-a", "doc-b"])]  # only the IBM region re-derived
    delta = Delta(
        parent_version=0,
        entries=[DeltaEntry(kind=DeltaKind.ADDED, document_id="doc-b", content_hash="sha256:b")],
        drift=0.25,
        triggered_reconciliation=False,  # below threshold — the surgical affected-set path
    )
    outcome = processor.apply_delta(parent, current, delta)
    by_id = {r.record_id: r for r in outcome.records}
    # The untouched Globex record is the very same object carried forward (identity), never touched.
    assert by_id["globex#r"] is untouched
    # The affected IBM record is the fresh current derivation (2 members), not the stale parent.
    assert by_id["ibm#r"].entity is not None
    assert by_id["ibm#r"].entity.mention_count == 2
    assert affected_parent not in outcome.records  # the stale parent was superseded


def test_delta_provider_delete_splits() -> None:
    processor = _load()
    # Parent: a 2-member IBM merged across doc-a + doc-b. Current (doc-b deleted): 1-member IBM.
    parent = [_entity("ibm", "IBM", 2, ["doc-a", "doc-b"])]
    current = [_entity("ibm", "IBM", 1, ["doc-a"])]
    delta = Delta(
        parent_version=1,
        entries=[DeltaEntry(kind=DeltaKind.DELETED, document_id="doc-b")],
        drift=0.5,
        triggered_reconciliation=False,
    )
    outcome = processor.apply_delta(parent, current, delta)
    live = {r.entity.canonical_text: r.entity.mention_count for r in outcome.records if r.entity}
    assert live["IBM"] == 1  # split back down
    assert outcome.entities_split >= 1
