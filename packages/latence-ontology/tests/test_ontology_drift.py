"""Drift becomes a new proposal and the old decision is marked stale (ticket 53; the owner,
14 Nov): a renamed column, a retyped column, a split table and a dropped table, each under an
element a person approved, on a copy of the toy estate changed between two builds."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb
import jsonschema
import pytest
from _ontology_toy import Project
from latence_ontology.cli import build
from test_ontology_ledger_export import SCHEMA, _export


@pytest.fixture()
def changing(project: Project, tmp_path: Path) -> Project:
    copy = tmp_path / "changing.duckdb"
    shutil.copyfile(project["duckdb_path"], copy)
    changed: Project = {**project, "duckdb_path": copy}
    return changed


def _approve(project: Project, *ids: str) -> None:
    export = _export(project)
    elements = {el["id"]: el for part in (export["concepts"], export["relations"]) for el in part}
    with open(project["out"] / "decisions.jsonl", "a", encoding="utf-8") as fh:
        for i in ids:
            fh.write(json.dumps({"subject": i, "verdict": "approved", "by": "reviewer-1", "claim": elements[i]["claim"]}) + "\n")


def _change(project: Project, *sql: str) -> None:
    con = duckdb.connect(str(project["duckdb_path"]))
    for s in sql:
        con.execute(s)
    con.close()


def _drift(project: Project) -> list[dict[str, Any]]:
    summary = build(**project)
    export = _export(project)
    jsonschema.validate(export, SCHEMA)
    assert summary.get("drift", []) == export.get("drift", [])
    return list(export.get("drift", []))


def test_nothing_drifts_when_nothing_changed(changing: Project) -> None:
    build(**changing)
    _approve(changing, "concept:shop.parts", "concept:shop.claims")
    assert _drift(changing) == []
    assert {c["id"]: c["state"] for c in _export(changing)["concepts"] if c["id"] in ("concept:shop.parts", "concept:shop.claims")} == {
        "concept:shop.parts": "approved", "concept:shop.claims": "approved"}


def test_a_renamed_column_and_a_retyped_one(changing: Project) -> None:
    build(**changing)
    _approve(changing, "concept:shop.dealers", "concept:shop.claims")
    _change(changing, "ALTER TABLE shop.dealers RENAME COLUMN name TO dealer_name", "ALTER TABLE shop.claims ALTER paid TYPE VARCHAR")
    drift = {d["subject"]: d for d in _drift(changing)}
    assert drift["concept:shop.dealers"]["drift"] == "renamed" and drift["concept:shop.dealers"]["columns"] == {"name": "dealer_name"}
    assert drift["concept:shop.claims"]["drift"] == "retyped"
    assert drift["concept:shop.claims"]["columns"] == {"paid": {"was": "DECIMAL(21,1)", "now": "VARCHAR"}}
    states = {c["id"]: c["state"] for c in _export(changing)["concepts"]}
    assert states["concept:shop.dealers"] == "proposed" and states["concept:shop.claims"] == "proposed"  # back to review
    lines = (changing["out"] / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["subject"] == "concept:shop.claims" for line in lines)  # the old decision stays on the record
    # A later build says the same: the drift is measured against what was decided, not against the build before.
    assert {d["subject"]: d["drift"] for d in _drift(changing)} == {"concept:shop.dealers": "renamed", "concept:shop.claims": "retyped"}


def test_a_split_table_and_a_dropped_one(changing: Project) -> None:
    build(**changing)
    _approve(changing, "concept:shop.parts", "concept:shop.visits")
    _change(changing, "CREATE TABLE shop.part_systems AS SELECT part_id, system FROM shop.parts",
            "CREATE TABLE shop.part_models AS SELECT part_id, make, model FROM shop.parts",
            "DROP TABLE shop.parts", "DROP TABLE shop.visits")
    drift = {d["subject"]: d for d in _drift(changing)}
    assert drift["concept:shop.parts"]["drift"] == "split"
    assert drift["concept:shop.parts"]["now"] == ["concept:shop.part_models", "concept:shop.part_systems"]
    assert drift["concept:shop.visits"] == {"subject": "concept:shop.visits", "kind": "concept", "decided_on": drift["concept:shop.visits"]["decided_on"],
                                            "verdict": "approved", "by": "reviewer-1", "drift": "dropped", "now": []}
    concepts = {c["id"]: c for c in _export(changing)["concepts"]}
    assert concepts["concept:shop.part_models"]["state"] == "proposed" and "concept:shop.parts" not in concepts
