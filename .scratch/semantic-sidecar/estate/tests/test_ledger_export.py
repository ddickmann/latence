from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from estate.cli import build
from estate.ledger import read_decisions

from .conftest import OPTIONS, SENSITIVE, VIEWS

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "schemas" / "graph-export-3.schema.json").read_text())


@pytest.fixture()
def project(tmp_path: Path, toy_path: Path) -> dict[str, Path]:
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({"name": "toy", "title": "Toy estate", "schema_prefix": "acme_", "schemas": ["shop", "marts"],
                                   "personal_objects": ["shop.owners", "shop.vehicles"], "resolve": ["make"],
                                   "sensitive": SENSITIVE, "ignore_columns": list(OPTIONS["ignore_columns"]),
                                   "derived_schemas": list(OPTIONS["derived_schemas"])}))
    views = tmp_path / "views.txt"
    views.write_text(VIEWS)
    verified = tmp_path / "catalog.yaml"
    verified.write_text("verified:\n  - shop_claims.shop_dealers\n  - shop_claims.marts_claims_by_dealer\n  - shop_claims.shop_vehicles\n")
    return {"config": cfg, "duckdb_path": toy_path, "views": views, "verified": verified, "out": tmp_path / "out"}


def _export(project: dict[str, Path]) -> dict:
    return json.loads((project["out"] / "ontology.json").read_text())


def test_the_export_matches_its_schema_and_says_where_it_came_from(project: dict[str, Path]) -> None:
    summary = build(**project)
    export = _export(project)
    jsonschema.validate(export, SCHEMA)
    assert export["meta"]["schema_origin"] == "induced" and export["meta"]["instances"] == "available"
    assert export["meta"]["instance_counts"]["certified"] is False and export["nodes"] and export["edges"]
    assert summary["concepts"] == len(export["concepts"]) and summary["new_assertions"] > 0


def test_every_element_is_an_assertion_with_an_actor_and_its_signals(project: dict[str, Path]) -> None:
    build(**project)
    export = _export(project)
    rows = [json.loads(x) for x in (project["out"] / "assertions.jsonl").read_text().splitlines()]
    subjects = {r["subject"] for r in rows}
    assert {e["id"] for k in ("concepts", "relations", "measures") for e in export[k]} <= subjects
    assert all(r["actor"].startswith("engine:") and r["signals"] for r in rows if r["kind"] != "concept")


def test_a_rebuild_appends_nothing_new_and_gives_the_same_file(project: dict[str, Path]) -> None:
    build(**project)
    first = (project["out"] / "ontology.json").read_bytes()
    assert build(**project)["new_assertions"] == 0
    assert (project["out"] / "ontology.json").read_bytes() == first


def test_a_decision_survives_a_rebuild_and_a_rejected_element_leaves_the_layer(project: dict[str, Path]) -> None:
    build(**project)
    rejected = "relation:shop.claims.part_id->shop.parts"
    approved = "concept:shop.dealers"
    assert any(r["id"] == rejected for r in _export(project)["relations"])
    with open(project["out"] / "decisions.jsonl", "a") as fh:
        fh.write(json.dumps({"subject": rejected, "verdict": "rejected", "by": "reviewer-1", "reason": "three percent of the part numbers are not parts"}) + "\n")
        fh.write(json.dumps({"subject": approved, "verdict": "approved", "by": "reviewer-1"}) + "\n")
    for _ in range(2):  # the second run is a rebuild on top of the decisions
        summary = build(**project)
        export = _export(project)
        assert not any(r["id"] == rejected for r in export["relations"])
        dealers = next(c for c in export["concepts"] if c["id"] == approved)
        assert dealers["decision"] == {"verdict": "approved", "by": "reviewer-1"}
        assert summary["decisions"] == {"approved": 1, "rejected": 1, "unknown_subjects": []}
    assert rejected in {json.loads(x)["subject"] for x in (project["out"] / "assertions.jsonl").read_text().splitlines()}


def test_a_decision_without_an_author_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "decisions.jsonl"
    f.write_text(json.dumps({"subject": "concept:x", "verdict": "approved"}) + "\n")
    with pytest.raises(ValueError, match="who made it"):
        read_decisions(f)


def test_no_value_of_a_personal_column_reaches_the_export(project: dict[str, Path]) -> None:
    build(**project)
    text = (project["out"] / "ontology.json").read_text() + (project["out"] / "assertions.jsonl").read_text()
    assert "@example.org" not in text and "Person 7" not in text and '"V7"' not in text
    export = _export(project)
    assert next(c for c in export["concepts"] if c["id"] == "concept:shop.owners")["instances"] == "withheld_personal"


def test_the_hosts_verified_joins_are_compared_one_by_one(project: dict[str, Path]) -> None:
    build(**project)
    states = {(j["from"], j["to"]): j["state"] for j in _export(project)["meta"]["host_verified_joins"]}
    assert states == {("shop.claims", "shop.dealers"): "found", ("shop.claims", "marts.claims_by_dealer"): "left_out_derived",
                      ("shop.claims", "shop.vehicles"): "not_checked_personal_key"}
