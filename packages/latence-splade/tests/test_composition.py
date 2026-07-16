"""The ADR-0053 composition algebra, tested through its interface with plain numbers (no model).

:func:`~latence_splade.composition.compose` is pure term-weight algebra — the whole point is that
the ADR-0053 policy (confidence gate, boosts, model-weight, summed union) is verifiable without a
model, a tokenizer, or torch. These tests assert the *observable* composed ``{vocab_id: weight}``
map for hand-built legs, never the internals.
"""

from __future__ import annotations

from latence_splade.composition import compose


def _compose(**overrides: object) -> dict[int, float]:
    kwargs: dict[str, object] = {
        "model_terms": {},
        "entity_terms": [],
        "metadata_terms": [],
        "confidence_threshold": 0.5,
        "entity_boost": 1.0,
        "metadata_boost": 1.0,
        "model_weight": 1.0,
    }
    kwargs.update(overrides)
    return compose(**kwargs)  # type: ignore[arg-type]


def test_model_leg_alone_passes_through_at_unit_weight() -> None:
    assert _compose(model_terms={1: 2.0, 7: 0.5}) == {1: 2.0, 7: 0.5}


def test_entity_injection_adds_boost_per_vocab_id_and_sums_on_collision() -> None:
    # Entity ids [1, 3] with confidence above the gate: id 1 collides with the model leg (summed),
    # id 3 is new (the boost alone).
    out = _compose(model_terms={1: 2.0}, entity_terms=[([1, 3], 0.9)], entity_boost=1.0)
    assert out == {1: 3.0, 3: 1.0}


def test_confidence_gate_drops_entities_below_threshold() -> None:
    # confidence 0.3 < threshold 0.5 → the entity contributes nothing (weak NER never pollutes).
    out = _compose(
        model_terms={1: 2.0}, entity_terms=[([9], 0.3)], confidence_threshold=0.5
    )
    assert out == {1: 2.0}
    # At the boundary (== threshold) the entity IS admitted (>= gate).
    admitted = _compose(model_terms={1: 2.0}, entity_terms=[([9], 0.5)], confidence_threshold=0.5)
    assert admitted == {1: 2.0, 9: 1.0}


def test_metadata_injection_adds_its_own_boost() -> None:
    out = _compose(model_terms={1: 2.0}, metadata_terms=[[5]], metadata_boost=2.0)
    assert out == {1: 2.0, 5: 2.0}


def test_weighted_mode_scales_the_model_leg() -> None:
    # The only difference between inject (model_weight=1.0) and weighted is the model-leg multiplier.
    out = _compose(model_terms={1: 2.0, 4: 1.0}, model_weight=0.5)
    assert out == {1: 1.0, 4: 0.5}


def test_all_three_legs_sum_on_a_shared_id() -> None:
    out = _compose(
        model_terms={1: 2.0},
        entity_terms=[([1], 0.9)],
        metadata_terms=[[1]],
        entity_boost=1.0,
        metadata_boost=3.0,
    )
    assert out == {1: 2.0 + 1.0 + 3.0}


def test_custom_boosts_are_honoured() -> None:
    out = _compose(entity_terms=[([2], 1.0)], entity_boost=4.0)
    assert out == {2: 4.0}
