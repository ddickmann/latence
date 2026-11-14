"""A column that turns sensitive is erased from every file of the build folder and the ledger,
recorded as an erasure (ticket 54; the owner, 14 Nov)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from estate.cli import build

from .test_ledger_export import _export, project  # noqa: F401  (the fixture)

MARKETS = ["shop.dealers.market", "shop.claims.market", "shop.visits.market", "shop.owners.market", "shop.vehicles.market"]


def _everything(out: Path) -> str:
    return "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(out.iterdir()) if p.is_file() and p.name != "erasures.jsonl")


def test_a_column_that_turns_sensitive_leaves_no_value_in_the_folder_and_the_erasure_is_recorded_once(project: dict[str, Path]) -> None:
    build(**project)
    export = _export(project)
    assert '"north"' in json.dumps(export) and "value:market:north" in json.dumps(export)  # the values are read as ordinary
    with open(project["out"] / "decisions.jsonl", "a", encoding="utf-8") as fh:  # a person's decision on two of its values
        fh.write(json.dumps({"subject": "same:market:north~south", "verdict": "rejected", "by": "reviewer-1"}) + "\n")
        fh.write(json.dumps({"subject": "concept:shop.dealers", "verdict": "approved", "by": "reviewer-1"}) + "\n")
    (project["out"] / "proposal.yaml").write_text('value: "north"\n', encoding="utf-8")  # a file written from them
    cfg = yaml.safe_load(project["config"].read_text(encoding="utf-8"))
    project["config"].write_text(yaml.safe_dump({**cfg, "sensitive": [*cfg["sensitive"], *MARKETS]}), encoding="utf-8")
    summary = build(**project)
    left = _everything(project["out"])
    assert '"north"' not in left and '"south"' not in left and "market:north" not in left
    erasures = [json.loads(line) for line in (project["out"] / "erasures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert erasures == summary["erased"] == [{"column": "market", "objects": erasures[0]["objects"],
                                             "build": 2, "values": 2, "lines_removed": 1, "files_removed": ["proposal.yaml"]}]
    assert set(erasures[0]["objects"]) <= {m.rsplit(".", 1)[0] for m in MARKETS} and "shop.dealers" in erasures[0]["objects"]
    assert "north" not in (project["out"] / "erasures.jsonl").read_text(encoding="utf-8")  # an erasure names no value
    decisions = (project["out"] / "decisions.jsonl").read_text(encoding="utf-8")
    assert "concept:shop.dealers" in decisions  # what held no value stays
    assert next(c for c in _export(project)["concepts"] if c["id"] == "concept:shop.dealers")["decision"]["verdict"] == "approved"
    # A second build finds nothing more to erase, and records nothing.
    summary = build(**project)
    assert "erased" not in summary and len((project["out"] / "erasures.jsonl").read_text(encoding="utf-8").splitlines()) == 1
