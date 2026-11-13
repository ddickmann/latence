"""The score against a hand-checked list: counted right on an export with known content."""

from __future__ import annotations

from typing import Any

import pytest

from estate.score import score


def _export() -> dict[str, Any]:
    concept = lambda name: {"id": f"concept:{name}"}  # noqa: E731
    return {
        "concepts": [concept(n) for n in ("s.dealers", "s.claims", "s.lines", "s.parts", "s.bulletins", "m.profile", "value:town", "value:region", "value:unit")],
        "classes": {
            "classes": [{"id": "class:warranty_claim", "parent": "class:event"}],
            "subclass_of": [
                {"concept": "concept:s.dealers", "parent": "class:organisation"},      # right
                {"concept": "concept:s.claims", "parent": "class:warranty_claim"},    # right, through a class of the model's
                {"concept": "concept:s.parts", "parent": "class:document"},           # wrong: a product
                {"concept": "concept:m.profile", "parent": "class:event"},            # wrong: the list says none
                {"concept": "concept:value:unit", "parent": "class:place"},           # nobody judged it
            ]},
        "structure": {
            "roll_ups": [{"from": "concept:value:town", "to": "concept:value:region"}, {"from": "concept:value:unit", "to": "concept:value:region"}],
            "part_of": [{"from": "concept:s.lines", "to": "concept:s.claims", "kind": "part_of"}, {"from": "concept:s.lines", "to": "concept:s.parts", "kind": "links"}],
        },
    }


GOLD: dict[str, Any] = {
    "checked_by": "a reviewer, 2026-11-11",
    "classes": {"s.dealers": "organisation", "s.claims": "event", "s.parts": "product", "s.bulletins": "document", "m.profile": "none", "value:town": "place"},
    "steps": [["value:town", "value:region"], ["s.dealers", "value:town"]],
    "part_of": [["s.lines", "s.claims"]],
    "links": [],
    "unsure": {"classes": ["value:unit"], "steps": [["value:unit", "value:region"]]},
}


def test_precision_and_recall_per_kind_with_every_miss_and_every_wrong_one_by_name() -> None:
    out = score(_export(), GOLD)
    assert out["counts_as_measured"] is True and out["checked_by"] == "a reviewer, 2026-11-11"
    c = out["classes"]
    assert (c["proposed"], c["right"], c["in_the_list"]) == (4, 2, 5) and c["precision"] == 0.5 and c["recall"] == 0.4
    assert c["wrong"] == ["m.profile: proposed event, the list says none", "s.parts: proposed document, the list says product"]
    assert c["missed"] == ["s.bulletins: the list says document, not placed", "value:town: the list says place, not placed"]
    assert out["steps"] == {"proposed": 1, "in_the_list": 2, "right": 1, "precision": 1.0, "recall": 0.5, "wrong": [], "missed": ["s.dealers -> value:town"]}
    assert out["part_of"]["precision"] == 1.0 and out["part_of"]["recall"] == 1.0
    # A link was proposed and the list holds none: precision is zero, and recall is not a number.
    assert out["links"]["precision"] == 0.0 and out["links"]["recall"] is None and out["links"]["wrong"] == ["s.lines -> s.parts"]


def test_a_list_nobody_checked_gives_numbers_that_do_not_count() -> None:
    out = score(_export(), {**GOLD, "checked_by": None})
    assert out["counts_as_measured"] is False and out["classes"]["precision"] == 0.5


def test_nothing_proposed_is_not_a_precision_of_zero() -> None:
    export = _export()
    export["structure"]["roll_ups"] = []
    out = score(export, GOLD)
    assert out["steps"]["precision"] is None and out["steps"]["recall"] == 0.0


def test_a_name_the_export_does_not_hold_is_an_error_and_so_is_a_class_that_is_no_top_class() -> None:
    with pytest.raises(ValueError, match="s.dealer'"):
        score(_export(), {**GOLD, "classes": {"s.dealer": "organisation"}})
    with pytest.raises(ValueError, match="workshop"):
        score(_export(), {**GOLD, "classes": {"s.dealers": "workshop"}})
