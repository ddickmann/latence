"""Unit tests for the induced-label seam (W4): union, backward-compat, order-stability, contract."""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    DocumentRecord,
    InducedLabels,
    Provenance,
)
from latence_core.induced import (
    LabelFloor,
    LabelSlot,
    effective_labels,
    induced_for_slot,
    induced_labels_by_document,
    project_induced_onto_documents,
    resolve_label_floor,
)
from latence_core.redaction_policy import GUARD_SAFETY_FLOOR, guard_category


def _induced(
    *,
    entity_types: list[str] | None = None,
    relation_types: list[str] | None = None,
    pii_types: list[str] | None = None,
) -> InducedLabels:
    return InducedLabels(
        model_id="test-model",
        entity_types=entity_types or [],
        relation_types=relation_types or [],
        pii_types=pii_types or [],
    )


def test_no_induced_returns_config_unchanged() -> None:
    # The byte-identical backward-compat proof: None induced ⇒ config labels back, same order.
    config = ["person", "org", "place"]
    assert effective_labels(config, None, LabelSlot.ENTITY) == config


def test_empty_slot_returns_config_unchanged() -> None:
    config = ["person", "org"]
    induced = _induced(entity_types=[])
    assert effective_labels(config, induced, LabelSlot.ENTITY) == config


def test_union_appends_new_induced_after_config_floor() -> None:
    config = ["person", "org"]
    induced = _induced(entity_types=["location", "product"])
    assert effective_labels(config, induced, LabelSlot.ENTITY) == [
        "person",
        "org",
        "location",
        "product",
    ]


def test_union_dedupes_config_and_induced_order_stable() -> None:
    config = ["person", "org"]
    induced = _induced(entity_types=["org", "person", "gene"])  # overlaps + one new
    # Config floor first (its order), then only NEW induced labels (their order).
    assert effective_labels(config, induced, LabelSlot.ENTITY) == ["person", "org", "gene"]


def test_union_dedupes_case_insensitively_config_surface_form_wins() -> None:
    # W12: a config 'Bank' and an induced 'bank' collapse to ONE entry — the config surface form is
    # kept (config comes first), the induced case-variant is dropped, defense-in-depth against any
    # residual case drift between the config floor and the canonicalized induced labels.
    config = ["Bank"]
    induced = _induced(entity_types=["bank", "account"])
    assert effective_labels(config, induced, LabelSlot.ENTITY) == ["Bank", "account"]


def test_slots_are_independent() -> None:
    induced = _induced(
        entity_types=["person"], relation_types=["works_for"], pii_types=["email"]
    )
    assert effective_labels([], induced, LabelSlot.ENTITY) == ["person"]
    assert effective_labels([], induced, LabelSlot.RELATION) == ["works_for"]
    assert effective_labels([], induced, LabelSlot.PII) == ["email"]


def test_empty_config_with_induced_is_fully_unsupervised() -> None:
    induced = _induced(entity_types=["a", "b"])
    assert effective_labels([], induced, LabelSlot.ENTITY) == ["a", "b"]


def test_config_dedupes_itself() -> None:
    # A config list with its own dupes is deduped order-stable even with no induced labels.
    assert effective_labels(["x", "x", "y"], None, LabelSlot.ENTITY) == ["x", "y"]


def test_induced_for_slot_none_is_empty() -> None:
    assert induced_for_slot(None, LabelSlot.PII) == []


def test_induced_labels_contract_requires_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        InducedLabels(entity_types=["x"])  # type: ignore[call-arg]


def test_induced_labels_defaults_are_empty_lists() -> None:
    labels = InducedLabels(model_id="m")
    assert labels.entity_types == []
    assert labels.relation_types == []
    assert labels.pii_types == []
    assert labels.guard_types == []
    assert labels.confidences == {}


# -- T2: the floor POLICY (relevance labels per chunk, safety floors universal) ---------------


def test_per_chunk_floor_returns_only_the_records_own_induced_labels() -> None:
    """The core of the redesign: an inducing record is prompted with ITS OWN types, nothing else."""
    induced = _induced(entity_types=["vessel", "port"])
    assert effective_labels(
        ["person", "organization"], induced, LabelSlot.ENTITY, floor=LabelFloor.PER_CHUNK
    ) == ["vessel", "port"]


def test_universal_floor_is_the_default_and_still_unions() -> None:
    """Every pre-existing caller passes no ``floor`` and must keep the union — byte-identical."""
    induced = _induced(entity_types=["vessel"])
    assert effective_labels(["person"], induced, LabelSlot.ENTITY) == ["person", "vessel"]
    assert effective_labels(
        ["person"], induced, LabelSlot.ENTITY, floor=LabelFloor.UNIVERSAL
    ) == ["person", "vessel"]


def test_per_chunk_floor_falls_back_to_config_when_the_slot_is_empty() -> None:
    """An empty induced slot must never yield an EMPTY prompt — the config floor is the net."""
    induced = _induced(entity_types=["vessel"])  # entity induced, relation slot empty
    assert effective_labels(
        ["works for"], induced, LabelSlot.RELATION, floor=LabelFloor.PER_CHUNK
    ) == ["works for"]
    assert effective_labels(["person"], None, LabelSlot.ENTITY, floor=LabelFloor.PER_CHUNK) == [
        "person"
    ]


def test_per_chunk_floor_dedupes_the_induced_set_case_insensitively() -> None:
    induced = _induced(entity_types=["Vessel", "vessel", "port"])
    assert effective_labels(
        ["person"], induced, LabelSlot.ENTITY, floor=LabelFloor.PER_CHUNK
    ) == ["Vessel", "port"]


def test_guard_slot_is_read_like_any_other() -> None:
    induced = InducedLabels(model_id="m", guard_types=["toxicity"])
    assert induced_for_slot(induced, LabelSlot.GUARD) == ["toxicity"]
    assert effective_labels(list(GUARD_SAFETY_FLOOR), induced, LabelSlot.GUARD) == [
        "prompt injection",
        "harmful content",
        "toxicity",
    ]


def test_resolve_label_floor_defaults_and_rejects_typos() -> None:
    """A typo'd policy must be loud: a silently ignored key would leave the other policy running."""
    assert resolve_label_floor({}, default=LabelFloor.PER_CHUNK) is LabelFloor.PER_CHUNK
    assert (
        resolve_label_floor({"label_floor": "universal"}, default=LabelFloor.PER_CHUNK)
        is LabelFloor.UNIVERSAL
    )
    with pytest.raises(ValueError, match="label_floor must be one of"):
        resolve_label_floor({"label_floor": "per-chunk"}, default=LabelFloor.PER_CHUNK)


def test_guard_category_maps_the_floor_to_the_screening_vocabulary() -> None:
    """The guard floor reports under the SAME categories screening.content_keyword emits."""
    assert guard_category("prompt injection") == "prompt_injection"
    assert guard_category("Harmful  Content") == "harmful"
    assert guard_category("Malware Instructions") == "malware_instructions"


# -- chunk→document bridge (W4 redaction seam) --------------------------------


def _prov(doc_id: str) -> Provenance:
    return Provenance(
        source_uri=f"file:///{doc_id}.txt",
        file_name=f"{doc_id}.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc_id}",
        char_start=0,
        char_end=10,
        page_start=1,
        page_end=1,
    )


def _chunk(doc_id: str, index: int, induced: InducedLabels | None) -> ChunkRecord:
    return ChunkRecord(
        record_id=f"{doc_id}#{index}",
        document_record_id=doc_id,
        index=index,
        content="text",
        token_count=1,
        provenance=_prov(doc_id),
        classification=Classification(language="en", sensitivity="internal"),
        induced_labels=induced,
    )


def _document(doc_id: str) -> DocumentRecord:
    return DocumentRecord(
        record_id=doc_id,
        provenance=_prov(doc_id),
        classification=Classification(language="en", sensitivity="internal"),
        content="text",
    )


def test_induced_labels_by_document_first_populated_chunk_wins() -> None:
    a = _induced(pii_types=["iban"])
    b = _induced(pii_types=["passport_number"])
    chunks = [
        _chunk("d1", 0, None),  # fail-open chunk contributes nothing
        _chunk("d1", 1, a),  # first populated chunk of d1 wins
        _chunk("d1", 2, b),
        _chunk("d2", 0, b),
    ]
    by_doc = induced_labels_by_document(chunks)
    assert by_doc["d1"] is a
    assert by_doc["d2"] is b


def test_induced_labels_by_document_absent_when_all_chunks_failed_open() -> None:
    chunks = [_chunk("d1", 0, None), _chunk("d1", 1, None)]
    assert induced_labels_by_document(chunks) == {}


def test_project_stamps_induced_pii_onto_document() -> None:
    # The core of the W4 redaction fix: a document's chunk-carried induced pii_types land on the
    # DocumentRecord that Redaction reads, so effective_labels(config, doc.induced, PII) unions them.
    induced = _induced(pii_types=["iban", "passport_number"])
    docs = [_document("d1")]
    chunks = [_chunk("d1", 0, induced)]
    stamped = project_induced_onto_documents(docs, chunks)
    assert stamped[0].induced_labels is induced
    # And Redaction's effective PII labels are now config ∪ induced.
    assert effective_labels(
        ["email"], stamped[0].induced_labels, LabelSlot.PII
    ) == ["email", "iban", "passport_number"]


def test_project_leaves_document_unchanged_when_no_induced_chunk() -> None:
    # Byte-identical backward-compat: no induced chunk ⇒ induced_labels stays None (config floor).
    docs = [_document("d1")]
    chunks = [_chunk("d1", 0, None)]
    stamped = project_induced_onto_documents(docs, chunks)
    assert stamped[0].induced_labels is None
    assert stamped[0] is docs[0]  # untouched, not a needless copy


def test_project_no_chunks_is_identity() -> None:
    docs = [_document("d1"), _document("d2")]
    stamped = project_induced_onto_documents(docs, [])
    assert [d.record_id for d in stamped] == ["d1", "d2"]
    assert all(d.induced_labels is None for d in stamped)


def test_project_preserves_document_order_and_is_per_document() -> None:
    a = _induced(pii_types=["iban"])
    docs = [_document("d1"), _document("d2"), _document("d3")]
    chunks = [_chunk("d2", 0, a)]  # only d2 has an induced schema
    stamped = project_induced_onto_documents(docs, chunks)
    assert [d.record_id for d in stamped] == ["d1", "d2", "d3"]
    assert stamped[0].induced_labels is None
    assert stamped[1].induced_labels is a
    assert stamped[2].induced_labels is None


def test_project_does_not_overwrite_preexisting_document_induced_labels() -> None:
    preexisting = _induced(pii_types=["ssn"])
    docs = [_document("d1").model_copy(update={"induced_labels": preexisting})]
    chunks = [_chunk("d1", 0, _induced(pii_types=["iban"]))]
    stamped = project_induced_onto_documents(docs, chunks)
    assert stamped[0].induced_labels is preexisting  # doc-level stamp wins, not overwritten
