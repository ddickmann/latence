"""What a roll-up's ``share`` is counted over (R5, decision D-2 (a)). Each ``broader`` edge
records the member's rows within the step's own objects (``member_rows``, graph-export/4), and
the proposal file's ``support.share`` is the edge's rows over that, so it is at most 1 by
construction. The member node's own ``rows`` is counted over other objects (personal objects are
not counted there), which is how the toy estate once wrote a share of 1.0769. The proposal file
is also checked against its shipped schema before it is written, so a share above 1 is refused
rather than written."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest
from _ontology_toy import Project
from latence_core.errors import ContractError
from latence_ontology.cli import build
from latence_ontology.proposal import checked, proposal, to_yaml

AT = "2026-11-15"


def _approved_export(project: Project) -> dict[str, Any]:
    """The project built with every step between two value sets approved."""
    build(**project)
    export: dict[str, Any] = json.loads((project["out"] / "ontology.json").read_text())
    steps = [r for r in export["structure"]["roll_ups"]
             if r["from"].startswith("concept:value:") and r["to"].startswith("concept:value:")]
    assert steps, "the toy estate has a step between two value sets"
    (project["out"] / "decisions.jsonl").write_text("".join(
        json.dumps({"subject": s["id"], "verdict": "approved", "by": "reviewer-1",
                    "claim": s["claim"], "at": AT}) + "\n" for s in steps))
    build(**project)
    built: dict[str, Any] = json.loads((project["out"] / "ontology.json").read_text())
    return built


def _broader(export: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in export["edges"] if e["kind"] == "broader"]


def _shares(project: Project) -> list[float]:
    export = json.loads((project["out"] / "ontology.json").read_text())
    decisions = {}
    for line in (project["out"] / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        decisions[d["subject"]] = d
    return [e["broader"]["support"]["share"]
            for e in _document(export, decisions)["entries"] if "broader" in e]


def _document(export: dict[str, Any], decisions: dict[str, Any]) -> dict[str, Any]:
    document, _report = proposal(export, decisions, decided_on=AT)
    to_yaml(document, "toy")  # refused if it breaks its schema
    return document


def test_every_broader_edge_counts_its_rows_within_the_members_rows(project: Project) -> None:
    edges = _broader(_approved_export(project))
    assert edges
    assert all(1 <= e["rows"] <= e["member_rows"] for e in edges)


def test_personal_objects_carrying_both_columns_keep_the_share_at_most_1_and_the_edges(
    project: Project, tmp_path: Path
) -> None:
    before = _broader(_approved_export(project))
    # The same estate, with the personal object ``shop.vehicles`` carrying the finer column as
    # well: its rows now count on the edges, and still not on the member nodes.
    toy = tmp_path / "personal.duckdb"
    shutil.copy(project["duckdb_path"], toy)
    con = duckdb.connect(str(toy))
    con.execute("ALTER TABLE shop.vehicles ADD COLUMN model VARCHAR")
    con.execute("UPDATE shop.vehicles SET model = 'M' || (source_line % 8)")
    con.close()
    personal = Project(config=project["config"], duckdb_path=toy, views=project["views"],
                       verified=project["verified"], out=tmp_path / "personal-out")
    after_export = _approved_export(personal)
    after = _broader(after_export)
    assert {(e["from"], e["to"], e["relation"]) for e in after} == {
        (e["from"], e["to"], e["relation"]) for e in before}
    assert all(1 <= e["rows"] <= e["member_rows"] for e in after)
    nodes = {n["id"]: n.get("rows") or 0 for n in after_export["nodes"]}
    # The case the share was once wrong on: an edge carrying more rows than its member's node.
    assert any(e["rows"] > nodes.get(e["from"], 0) for e in after)
    shares = _shares(personal)
    assert shares and all(0 < s <= 1 for s in shares)


def test_a_proposal_file_that_breaks_its_schema_is_not_written() -> None:
    support: dict[str, Any] = {"rows": 280, "share": 1.0769}
    entry = {"broader": {"object": "shop.claims", "column": "model", "value": "M1",
                         "broader": ["BETA"], "support": support},
             "origin": {"element": "roll_up:value:model->value:make", "claim": "0" * 16,
                        "decision": {"verdict": "approved", "by": "reviewer-1", "at": AT}}}
    document = {"format": "layer-proposal/3", "source": "ontology", "entries": [entry]}
    with pytest.raises(ContractError, match="breaks its schema"):
        to_yaml(document, "toy")
    with pytest.raises(ContractError, match="no shipped schema"):
        checked({**document, "format": "layer-proposal/9"})
    support["share"] = 1.0
    assert to_yaml(document, "toy").startswith("# ")
