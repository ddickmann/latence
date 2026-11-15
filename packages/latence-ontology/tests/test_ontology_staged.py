"""Approved changes reach the catalog as staged proposals its owner applies, never as direct
writes (ticket 52; the owner, 14 Nov): a file of statements, one per change, each with the
decision it rests on, which the engine never runs."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import latence_ontology.staged
import yaml
from _ontology_toy import Project
from latence_ontology.cli import build, stage
from latence_ontology.staged import staged_statements
from test_ontology_ledger_export import _export

RELATION = "relation:shop.claims.part_id->shop.parts"
DEALERS = "concept:shop.dealers"


def _decide(project: Project, *lines: dict[str, Any]) -> None:
    with open(project["out"] / "decisions.jsonl", "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(line) + "\n" for line in lines)


def _catalog(project: Project) -> None:
    cfg = yaml.safe_load(project["config"].read_text(encoding="utf-8"))
    project["config"].write_text(yaml.safe_dump({**cfg, "catalog": "demo"}), encoding="utf-8")


def test_an_approved_key_and_relation_become_statements_with_their_origin_and_nothing_else_does(project: Project) -> None:
    _catalog(project)
    build(**project)
    export = _export(project)
    relation = next(r for r in export["relations"] if r["id"] == RELATION)
    dealers = next(c for c in export["concepts"] if c["id"] == DEALERS)
    parts = next(c for c in export["concepts"] if c["id"] == "concept:shop.parts")
    other = next(r for r in export["relations"] if r["status"] == "proposed" and r["id"] != RELATION and not r["from"].startswith("concept:value:"))
    _decide(project, {"subject": RELATION, "verdict": "approved", "by": "reviewer-1", "claim": relation["claim"], "at": "2026-11-14"},
            {"subject": DEALERS, "verdict": "approved", "by": "reviewer-1", "claim": dealers["claim"], "at": "2026-11-14"},
            {"subject": parts["id"], "verdict": "approved", "by": "reviewer-1", "claim": "0" * 16, "at": "2026-11-14"},  # another statement
            {"subject": other["id"], "verdict": "rejected", "by": "reviewer-1", "claim": other["claim"], "at": "2026-11-14"})
    build(**project)
    report = stage(project["config"], project["out"])
    text = (project["out"] / "staged-catalog.sql").read_text(encoding="utf-8")
    assert report["statements"] == 2 and len([line for line in text.splitlines() if line.startswith("ALTER TABLE")]) == 2
    assert "ALTER TABLE `demo`.`acme_shop`.`dealers` ADD CONSTRAINT `dealers_pk` PRIMARY KEY (`dealer_id`) NOT ENFORCED;" in text
    assert ("ALTER TABLE `demo`.`acme_shop`.`claims` ADD CONSTRAINT `claims_part_id_fk` FOREIGN KEY (`part_id`) "
            "REFERENCES `demo`.`acme_shop`.`parts` (`part_id`) NOT ENFORCED;") in text
    assert f"-- {RELATION}, claim {relation['claim']}, approved by reviewer-1 on 2026-11-14" in text
    assert "parts_pk" not in text and other["id"] not in text
    assert report["not_staged"] == [{"subject": parts["id"], "why": "stale"}]
    assert text.startswith("-- Staged for the catalog's owner to review and apply.")


def test_what_the_catalog_already_declares_and_what_only_the_engine_approved_is_not_staged() -> None:
    export = {"concepts": [{"id": "concept:s.t", "object_name": "s.t", "key": ["id"], "claim": "a" * 16,
                            "signals": [{"kind": "catalog", "code": "declared_key", "params": {"columns": ["id"]}, "ref": ""}]},
                           {"id": "concept:s.u", "object_name": "s.u", "key": ["code"], "claim": "b" * 16, "signals": []}],
              "relations": []}
    decided = {"concept:s.t": {"verdict": "approved", "by": "reviewer-1", "claim": "a" * 16, "at": "2026-11-14"},
               "concept:s.u": {"verdict": "approved", "by": "engine", "claim": "b" * 16, "calibration": "c.json"}}
    statements, report = staged_statements(export, decided, catalog="demo", prefix="")
    assert statements == [] and report["not_staged"] == [{"subject": "concept:s.t", "why": "declared_in_the_catalog"},
                                                         {"subject": "concept:s.u", "why": "not_a_person"}]


def test_a_name_cannot_break_a_statement_and_nothing_here_writes_to_the_catalog() -> None:
    export = {"concepts": [{"id": "concept:s.odd`table", "object_name": "s.odd`table", "key": ["a`b"], "claim": "a" * 16, "signals": []}],
              "relations": []}
    decided = {"concept:s.odd`table": {"verdict": "approved", "by": "reviewer-1", "claim": "a" * 16, "at": "2026-11-14"}}
    statements, _ = staged_statements(export, decided, catalog="demo", prefix="")
    assert statements[0]["sql"] == "ALTER TABLE `demo`.`s`.`odd``table` ADD CONSTRAINT `odd_table_pk` PRIMARY KEY (`a``b`) NOT ENFORCED"
    source = (Path(latence_ontology.staged.__file__)).read_text(encoding="utf-8")
    imported = {n.module for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {".reader_databricks", "reader_databricks", "urllib.request", ".reader"}  # it writes a file, it calls nothing
