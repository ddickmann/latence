"""The gliner2 record-shape contract, pinned against a MEASURED A/B across the major boundary.

ADR-0064 rests on one asymmetry: gliner2 2.0 changed the LOADER and the CHECKPOINTS but not the
INFERENCE contract. That was not read off a docstring — it was measured. On the pod (RTX 2000 Ada
16 GB, driver 580.159.04, 2026-08-25), gliner2 **1.3.2** + ``fastino/gliner2-multi-v1`` (span head)
and gliner2 **2.0.0** + ``fastino/gliner2.5-multi-v1`` (boundary head) were run in TWO SEPARATE
VENVS — the disjoint pin made real — on the same text with the same labels under identical
transformers 4.57.6 / torch 2.4.1+cu124, and the structural signature of the output was diffed.
Every structural field came back SAME; the only DIFFs were ``head_class`` (GLiNER2 →
BoundaryExtractor) and ``gliner2_version``.

This module is what stops that finding from rotting. Four fakes in this workspace claim to mirror
the shape — the two stubs in :mod:`latence_core.conformance.stub`, plus the test-local fakes in
``latence-extract-gliner2``, ``latence-pii-gliner2`` and ``latence-gliner25``. All four are checked
against ONE definition (``stub.GLINER2_SHAPE_CONTRACT``) rather than against each other, so a
future gliner2 release that moves a key is one edit and several red tests — never a fake that
silently drifts away from the API it stands in for, which would turn every test built on it into a
decoration.

The two checked HERE are the two that live in core and can therefore be driven side by side in one
assertion; the other two are checked in their own packages against the same imported constants,
because a test-local fake in another distribution cannot be imported across the package boundary
(pytest's ``prepend`` import mode gives every ``tests/`` its own top-level namespace, which is the
same collision ``scripts/typecheck.sh`` exists to work around, ADR-0032).
"""

from __future__ import annotations

from typing import Any

import pytest
from latence_core.conformance import stub
from latence_core.conformance.stub import (
    GLINER2_ENTITIES_ENVELOPE,
    GLINER2_ENTITY_ROW_KEYS,
    GLINER2_RELATION_ENDPOINT_KEYS,
    GLINER2_RELATION_ROW_KEYS,
    GLINER2_RELATIONS_ENVELOPE,
    GLINER2_SHAPE_CONTRACT,
    assert_matches_gliner2_shape_contract,
    gliner2_shape_signature,
)

_TEXT = "Alice works for Acme Corp in Berlin. Contact alice@example.com about Acme Corp."
_ENTITY_LABELS = ["person", "org"]
_RELATION_LABELS = ["works_for"]


def _drive(model: Any) -> dict[str, Any]:
    """One fake → its structural signature, through the REAL batched entry points."""
    entities = model.batch_extract_entities(
        [_TEXT], _ENTITY_LABELS, threshold=0.5, include_spans=True, include_confidence=True
    )
    relations = model.batch_extract_relations(
        [_TEXT], _RELATION_LABELS, threshold=0.5, include_spans=True, include_confidence=True
    )
    return gliner2_shape_signature(entities, relations)


def test_the_span_and_boundary_fakes_produce_the_IDENTICAL_signature() -> None:
    """The measured finding, as one assertion: the record shape survives the major bump.

    Driven side by side rather than compared field by field, so this says what the pod A/B said —
    "diff the two signatures and get nothing" — instead of restating a key list twice.
    """
    with stub.install_gliner2(
        entities=stub.gliner2_entities(), relations=stub.gliner2_relations()
    ):
        span = _drive(stub._StubGLiNER2())  # noqa: SLF001 - the stub IS the subject
    with stub.install_gliner25(
        entities=stub.gliner2_entities(), relations=stub.gliner2_relations()
    ):
        boundary = _drive(stub._StubBoundaryExtractor())  # noqa: SLF001 - the stub IS the subject

    assert span == boundary
    # And the thing that makes the equality meaningful: it is not two empty dicts agreeing.
    assert span["entity_row_keys"] == sorted(GLINER2_ENTITY_ROW_KEYS)


@pytest.mark.parametrize("architecture", ["span", "boundary"])
def test_each_core_fake_matches_the_pod_measured_contract(architecture: str) -> None:
    """Each fake separately, against the measurement — so "identical" cannot mean "identically wrong".

    Two fakes that drifted the same way would pass the equality test above and fail this one.
    """
    installer = stub.install_gliner2 if architecture == "span" else stub.install_gliner25
    factory = stub._StubGLiNER2 if architecture == "span" else stub._StubBoundaryExtractor  # noqa: SLF001
    with installer(entities=stub.gliner2_entities(), relations=stub.gliner2_relations()):
        signature = _drive(factory())
    assert_matches_gliner2_shape_contract(signature)


def test_the_contract_names_the_keys_the_pod_diff_recorded() -> None:
    """The constants themselves, spelled out once — the row the pod printed, transcribed."""
    assert GLINER2_SHAPE_CONTRACT["top_level_keys_entities"] == ["entities"]
    assert GLINER2_SHAPE_CONTRACT["top_level_keys_relations"] == ["relation_extraction"]
    assert GLINER2_SHAPE_CONTRACT["entity_row_keys"] == ["confidence", "end", "start", "text"]
    assert GLINER2_SHAPE_CONTRACT["relation_row_keys"] == ["head", "tail"]
    assert GLINER2_SHAPE_CONTRACT["relation_endpoint_keys"] == [
        "confidence",
        "end",
        "start",
        "text",
    ]
    assert GLINER2_SHAPE_CONTRACT["entity_offsets_are_ints"] is True
    assert GLINER2_SHAPE_CONTRACT["relation_offsets_are_ints"] is True
    # The envelope constants and the key tuples are one source of truth, not two.
    assert GLINER2_ENTITIES_ENVELOPE == "entities"
    assert GLINER2_RELATIONS_ENVELOPE == "relation_extraction"
    assert sorted(GLINER2_RELATION_ENDPOINT_KEYS) == sorted(GLINER2_ENTITY_ROW_KEYS)
    assert GLINER2_RELATION_ROW_KEYS == ("head", "tail")


def test_the_assertion_actually_bites() -> None:
    """A checker that cannot fail is a decoration. Prove each half of it rejects a drift."""
    good = dict(GLINER2_SHAPE_CONTRACT)
    assert_matches_gliner2_shape_contract(good)

    # A renamed key (the shape of a real upstream refactor).
    with pytest.raises(AssertionError, match="entity_row_keys"):
        assert_matches_gliner2_shape_contract(
            {**good, "entity_row_keys": ["confidence", "end", "score", "start", "text"]}
        )
    # A relation-level score appearing (the shape of a real upstream addition).
    with pytest.raises(AssertionError, match="relation_row_keys"):
        assert_matches_gliner2_shape_contract(
            {**good, "relation_row_keys": ["confidence", "head", "tail"]}
        )
    # Offsets arriving as strings — the failure that would silently un-ground every span.
    with pytest.raises(AssertionError, match="entity_offsets_are_ints"):
        assert_matches_gliner2_shape_contract({**good, "entity_offsets_are_ints": False})


def test_a_signature_over_entities_only_is_checked_on_the_entity_half() -> None:
    """The PII fakes drive entities only; they must not fail for a half they never claimed."""
    with stub.install_gliner2(entities=stub.gliner2_pii_entities()):
        entities = stub._StubGLiNER2().batch_extract_entities(  # noqa: SLF001 - the stub is the subject
            [_TEXT], ["email"], threshold=0.5, include_spans=True, include_confidence=True
        )
    signature = gliner2_shape_signature(entities)
    assert set(signature) == {
        "top_level_keys_entities",
        "entity_row_keys",
        "entity_offsets_are_ints",
    }
    assert_matches_gliner2_shape_contract(signature)
