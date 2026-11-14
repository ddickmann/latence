"""The five states of an element (ticket 51): draft, proposed, approved, certified, retired. A
named reviewer approves, the data owner certifies what is approved, and the engine approves
nothing unless a recorded calibration proves one kind safe, and then that kind only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from estate.cli import build

from .test_ledger_export import SCHEMA, _export, project  # noqa: F401  (the fixture)

DEALERS, RELATION = "concept:shop.dealers", "relation:shop.claims.part_id->shop.parts"


def _write(path: Path, *lines: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(line) + "\n" for line in lines)


def _states(export: dict[str, Any]) -> dict[str, str]:
    groups = [export["concepts"], export["relations"], export["measures"], export["review"],
              *(v for k, v in export["structure"].items() if k != "not_proposed"), export["classes"]["classes"], export["classes"]["subclass_of"]]
    return {el["id"]: el["state"] for part in groups for el in part}


def _estate(project: dict[str, Path], **lifecycle: Any) -> None:
    cfg = yaml.safe_load(project["config"].read_text(encoding="utf-8"))
    project["config"].write_text(yaml.safe_dump({**cfg, "lifecycle": {"data_owner": "owner-of-the-data", **lifecycle}}), encoding="utf-8")


def test_a_reviewer_approves_and_only_the_data_owner_certifies_what_is_approved(project: dict[str, Path]) -> None:
    _estate(project)
    build(**project)
    export = _export(project)
    states = _states(export)
    assert set(states.values()) == {"proposed"}  # nobody decided anything; the engine approved nothing
    dealers = next(c for c in export["concepts"] if c["id"] == DEALERS)
    relation = next(r for r in export["relations"] if r["id"] == RELATION)
    measure = export["measures"][0]
    _write(project["out"] / "decisions.jsonl",
           {"subject": DEALERS, "verdict": "approved", "by": "reviewer-1", "claim": dealers["claim"]},
           {"subject": RELATION, "verdict": "rejected", "by": "reviewer-1", "claim": relation["claim"]},
           {"subject": measure["id"], "verdict": "approved", "by": "reviewer-1", "claim": measure["claim"]})
    _write(project["out"] / "certifications.jsonl",
           {"subject": DEALERS, "by": "owner-of-the-data", "claim": dealers["claim"], "at": "2026-11-14"},        # certified
           {"subject": measure["id"], "by": "reviewer-1", "claim": measure["claim"], "at": "2026-11-14"},       # not the data owner
           {"subject": RELATION, "by": "owner-of-the-data", "claim": relation["claim"], "at": "2026-11-14"},      # not approved
           {"subject": "concept:shop.parts", "by": "owner-of-the-data", "claim": "0" * 16, "at": "2026-11-14"})  # on another statement
    summary = build(**project)
    export = _export(project)
    jsonschema.validate(export, SCHEMA)
    states = _states(export)
    assert states[DEALERS] == "certified" and states[measure["id"]] == "approved" and RELATION not in states  # a rejected element leaves the layer
    assert next(c for c in export["concepts"] if c["id"] == DEALERS)["certification"] == {"by": "owner-of-the-data", "at": "2026-11-14"}
    assert summary["lifecycle"]["refused"] == [
        {"subject": "concept:shop.parts", "why": "stale"}, {"subject": measure["id"], "why": "not_the_data_owner"}, {"subject": RELATION, "why": "not_approved"}]
    assert summary["lifecycle"]["retired"] == [RELATION]  # withdrawn, kept on the record
    assert summary["lifecycle"]["states"] == {"approved": 1, "certified": 1, "draft": len(export["dropped"]),
                                              "proposed": len(states) - 2, "retired": 1}


def test_the_engine_approves_one_kind_only_where_a_recorded_calibration_proves_it_safe(project: dict[str, Path], tmp_path: Path) -> None:
    calibration = tmp_path / "relations-calibrated.json"
    calibration.write_text(json.dumps({"kind": "relation", "precision_lower_bound": 0.97, "target": 0.95, "labels": 400,
                                       "by": "owner-of-the-data", "at": "2026-11-14"}), encoding="utf-8")
    _estate(project, auto_approve=[{"kind": "relation", "calibration": str(calibration)}])
    summary = build(**project)
    export = _export(project)
    states = _states(export)
    relations = {r["id"] for r in export["relations"] if r["status"] == "proposed"}
    assert relations and all(states[r] == "approved" for r in relations)
    assert {c["id"]: states[c["id"]] for c in export["concepts"]} == {c["id"]: "proposed" for c in export["concepts"]}  # that kind only
    made = next(r for r in export["relations"] if r["id"] in relations)["decision"]
    assert made == {"verdict": "approved", "by": "engine", "calibration": str(calibration)}
    assert summary["lifecycle"]["approved_by_the_engine"] == {"relation": len(relations)}
    # A person's decision outranks the engine's.
    first = sorted(relations)[0]
    claim = next(r for r in export["relations"] if r["id"] == first)["claim"]
    _write(project["out"] / "decisions.jsonl", {"subject": first, "verdict": "rejected", "by": "reviewer-1", "claim": claim})
    build(**project)
    assert first not in _states(_export(project))
    # A calibration below its target proves nothing, and none is used.
    calibration.write_text(json.dumps({"kind": "relation", "precision_lower_bound": 0.93, "target": 0.95, "labels": 400,
                                       "by": "owner-of-the-data", "at": "2026-11-14"}), encoding="utf-8")
    summary = build(**project)
    assert set(_states(_export(project)).values()) == {"proposed"} and summary["lifecycle"]["approved_by_the_engine"] == {}
    assert summary["lifecycle"]["calibrations_refused"] == [{"kind": "relation", "why": "below_its_target"}]


def test_the_command_line_writes_a_certification_on_the_statement_the_build_makes_now(project: dict[str, Path]) -> None:
    from estate.cli import certify

    _estate(project)
    build(**project)
    dealers = next(c for c in _export(project)["concepts"] if c["id"] == DEALERS)
    _write(project["out"] / "decisions.jsonl", {"subject": DEALERS, "verdict": "approved", "by": "reviewer-1", "claim": dealers["claim"]})
    assert certify(project["out"], DEALERS, "owner-of-the-data", "2026-11-14") == {
        "subject": DEALERS, "by": "owner-of-the-data", "claim": dealers["claim"], "at": "2026-11-14"}
    build(**project)
    assert _states(_export(project))[DEALERS] == "certified"
