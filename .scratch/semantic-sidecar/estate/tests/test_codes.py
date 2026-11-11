"""The contract of ``graph-export/3``: the engine writes codes, not sentences, and every
proposal a person can decide carries the claim it makes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from estate.cli import build
from estate.codes import CODES, check, render, say, table
from estate.ledger import claim_of

from .test_ledger_export import project  # noqa: F401  (the toy estate as a build folder)

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
# Keys that hold a name, an identifier or a number, never a sentence.
TEXT_KEYS = {"text"}


def _statements(node: Any, found: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if "code" in node and "params" in node:
            found.append(node)
        for v in node.values():
            _statements(v, found)
    elif isinstance(node, list):
        for v in node:
            _statements(v, found)


def _sentences(node: Any, path: str, found: list[str]) -> None:
    """Strings with three or more words outside a text object: a sentence somebody wrote."""
    if isinstance(node, dict):
        if set(node) >= {"text", "lang", "origin"}:
            return
        for k, v in node.items():
            _sentences(v, f"{path}.{k}", found)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _sentences(v, f"{path}[{i}]", found)
    elif isinstance(node, str) and len(node.split()) >= 3 and not path.endswith((".expr", ".on", ".label", ".title", ".id", ".a", ".b", ".from", ".to", ".word", ".parent", ".ref", ".key")):
        found.append(f"{path}: {node}")


def test_the_export_holds_no_sentence_the_engine_wrote(project: dict[str, Path]) -> None:  # noqa: F811
    build(**project)
    export = json.loads((project["out"] / "ontology.json").read_text())
    found: list[str] = []
    _sentences({k: v for k, v in export.items() if k not in ("nodes",)}, "export", found)
    assert found == []
    assert '"detail"' not in json.dumps(export)


def test_every_statement_in_the_export_is_in_the_table_with_its_parameters(project: dict[str, Path]) -> None:  # noqa: F811
    build(**project)
    found: list[dict[str, Any]] = []
    _statements(json.loads((project["out"] / "ontology.json").read_text()), found)
    assert len(found) > 20
    for item in found:
        check(item)
        assert render(item)  # the English can be written from the parameters it carries


def test_a_statement_with_a_wrong_parameter_is_a_defect_where_it_is_made() -> None:
    with pytest.raises(ValueError, match="takes"):
        say("declared_key", cols=["id"])
    with pytest.raises(KeyError):
        say("no_such_code")
    assert render(say("declared_key", columns=["a", "b"])) == "primary key (a, b)"


def test_the_committed_table_of_codes_is_the_one_the_code_holds() -> None:
    assert (SCHEMAS / "codes.json").read_text(encoding="utf-8") == table()
    for code, (params, english) in CODES.items():
        assert set(params) == {p.split("}")[0] for p in english.split("{")[1:]}, code  # the English uses exactly the parameters


def test_a_claim_follows_what_is_stated_and_not_what_was_counted() -> None:
    step = {"id": "roll_up:value:a->value:b", "from": "concept:value:a", "to": "concept:value:b", "kind": "roll_up", "support": {"values_checked": 10}}
    assert claim_of("roll_up", step) == claim_of("roll_up", {**step, "support": {"values_checked": 99}, "signals": ["x"], "label": "other"})
    assert claim_of("roll_up", step) != claim_of("roll_up", {**step, "to": "concept:value:c"})
    placed = {"id": "subclass:s.dealers", "concept": "concept:s.dealers", "parent": "class:organisation"}
    assert claim_of("subclass", placed) != claim_of("subclass", {**placed, "parent": "class:place"})  # the same id, another claim
    assert claim_of("equivalent", {"concepts": ["x", "y"]}) == claim_of("equivalent", {"concepts": ["y", "x"]})
    assert claim_of("possibly_same", {"a": "n1", "b": "n2"}) == claim_of("possibly_same", {"a": "n2", "b": "n1"})


def test_every_decidable_element_of_the_export_carries_its_claim(project: dict[str, Path]) -> None:  # noqa: F811
    build(**project)
    export = json.loads((project["out"] / "ontology.json").read_text())
    groups = [export["concepts"], export["relations"], export["measures"], export["review"], export["classes"]["classes"], export["classes"]["subclass_of"],
              *export["structure"].values()]
    assert all(len(el["claim"]) == 16 for group in groups for el in group) and sum(map(len, groups)) > 10
