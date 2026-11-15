"""An outside vocabulary as the third voice on the class tree (ticket 55), on the toy estate.

The vocabulary here is written for the test (its licence CC0-1.0 is the test's own statement).
On the toy estate the pattern voice places the dealers (organisation), the owners (person), the
parts (product) and the markets (place), and leaves the claims and the visits unplaced."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml
from _ontology_toy import Project
from latence_core.errors import ConfigError
from latence_ontology import SCHEMAS
from latence_ontology.cli import build
from latence_ontology.vocabulary import read_vocabulary

SCHEMA = json.loads((SCHEMAS / "graph-export-4.schema.json").read_text(encoding="utf-8"))
VOCABULARY: dict[str, Any] = {
    "name": "test-vocabulary",
    "version": "1",
    "licence": "CC0-1.0",
    "licence_source": "written for this test",
    "classes": [
        {"id": "T1", "labels": ["Claim", "warranty claim"], "top": "event"},
        {"id": "T2", "labels": ["Dealer", "dealership"], "top": "organisation"},
        {"id": "T3", "labels": ["Owner"], "top": "organisation"},  # the pattern says person
        {"id": "T4", "labels": ["Model"], "top": "product"},
        {"id": "T5", "labels": ["Model"], "top": "document"},  # one label, two top classes
    ],
}


def _with(project: Project, vocabulary: dict[str, Any], **said: Any) -> Project:
    path = project["config"].parent / "vocabulary.json"
    path.write_text(json.dumps(vocabulary))
    cfg = yaml.safe_load(project["config"].read_text())
    cfg["classes"] = {**(cfg.get("classes") or {}),
                      "vocabulary": {"file": "vocabulary.json", **said}}
    project["config"].write_text(yaml.safe_dump(cfg))
    return project


def _placed(project: Project) -> dict[str, dict[str, Any]]:
    export = json.loads((project["out"] / "ontology.json").read_text())
    return {s["concept"]: s for s in export["classes"]["subclass_of"]}


def test_a_match_proposes_and_a_disagreement_is_contested(project: Project) -> None:
    build(**_with(project, VOCABULARY))
    export = json.loads((project["out"] / "ontology.json").read_text())
    jsonschema.validate(export, SCHEMA)
    placed = _placed(project)
    # A match where no other voice spoke: it proposes, uncontested.
    claims = placed["concept:shop.claims"]
    assert claims["parent"] == "class:event" and claims["contested"] is False
    assert [s["code"] for s in claims["signals"]] == ["vocabulary.matched"]
    assert claims["signals"][0]["params"] == {"vocabulary": "test-vocabulary", "label": "Claim"}
    # A match that agrees with the pattern voice: both are heard, nothing is contested.
    dealers = placed["concept:shop.dealers"]
    assert dealers["parent"] == "class:organisation" and dealers["contested"] is False
    assert [s["code"] for s in dealers["signals"]] == [
        "attribute_pattern.organisation", "vocabulary.matched"]
    # A disagreement: the placement stays the older voice's, and it is contested.
    owners = placed["concept:shop.owners"]
    assert owners["parent"] == "class:person" and owners["contested"] is True
    assert owners["other_parent"] == "class:organisation"
    # No match, or a label two top classes share: nothing is proposed.
    assert "concept:shop.visits" not in placed and "concept:value:model" not in placed
    assert export["classes"]["voices"]["vocabulary"]["licence"] == "CC0-1.0"
    assert export["classes"]["voices"]["vocabulary"]["licence_source"] == "written for this test"
    assert len(export["classes"]["voices"]["vocabulary"]["digest"]) == 16


def test_without_a_vocabulary_nothing_changes(project: Project) -> None:
    build(**project)
    export = json.loads((project["out"] / "ontology.json").read_text())
    assert "vocabulary" not in export["classes"]["voices"]
    assert "concept:shop.claims" not in _placed(project)


def test_a_vocabulary_whose_licence_is_not_allowed_or_not_stated_is_refused(
    project: Project, tmp_path: Path
) -> None:
    share_alike = {**VOCABULARY, "licence": "CC-BY-SA-3.0"}
    with pytest.raises(ConfigError, match="licensed CC-BY-SA-3.0"):
        build(**_with(project, share_alike))
    # The estate file may allow it, and then it is recorded as such.
    build(**_with(project, share_alike, licences_allowed=["CC-BY-SA-3.0"]))
    export = json.loads((project["out"] / "ontology.json").read_text())
    assert export["classes"]["voices"]["vocabulary"]["licence"] == "CC-BY-SA-3.0"
    unstated = tmp_path / "unstated.json"
    unstated.write_text(json.dumps({k: v for k, v in VOCABULARY.items() if k != "licence"}))
    with pytest.raises(ConfigError, match="states its licence"):
        read_vocabulary(unstated)
    off_tree = tmp_path / "off-tree.json"
    off_tree.write_text(json.dumps({**VOCABULARY, "classes": [
        {"id": "X", "labels": ["Claim"], "top": "thing"}]}))
    with pytest.raises(ConfigError, match="no top class"):
        read_vocabulary(off_tree)
