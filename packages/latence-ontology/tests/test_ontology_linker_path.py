"""The build with the record linker, through the real call (ticket 44): the build writes the
records file, runs the linker in its own environment by the command line the linker reads,
takes its answer into the export, and the proposal file written from that build is the one a
build without the linker writes, because no link between records is ever in it.

The linker's environment (Splink and its stack, `spike/requirements-link.txt`) is not here, so a
stand-in runs in its place. It is held to the real linker's command line: its argument parser is
the one `spike/link_records.py` declares, read from that file, so an argument the build passes
and the linker does not take stops the stand-in as it stops the linker. It reads the records
file the build wrote and checks the columns the linker reads, and answers in `resolution/3`
from the truth column, as the linker's labels do where the seed's truth stands in.
"""

from __future__ import annotations

import ast
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import test_ontology_proposal as tp  # the yard estate and its helpers
import yaml
from latence_ontology.cli import build, propose
from latence_ontology.resolution import COLUMNS

LINKER = Path(__file__).resolve().parents[3] / ".scratch" / "semantic-sidecar" / "spike" / "link_records.py"
TRUTH = {**{f"PA{i}": f"other:PA{i}" for i in range(1, 46)}, "PA1": "D1", "PA2": "D1", "PA3": "D2"}

STAND_IN = '''
import argparse, json
from pathlib import Path
import duckdb

ap = argparse.ArgumentParser()
{arguments}
a = ap.parse_args()
names = [r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{{a.records}}')").fetchall()]
assert names == {columns!r}, names
rows = duckdb.sql(f"SELECT unique_id, source_system, cluster FROM read_parquet('{{a.records}}') ORDER BY unique_id").fetchall()
master = a.duplicate_free[0]
masters = {{c: u for u, s, c in rows if s == master}}
said = {{frozenset((d["a"], d["b"])): d for d in json.loads(a.labels.read_text())}} if a.labels else {{}}
ev = [{{"kind": "name", "code": "name.same_core", "params": {{}}, "ref": ""}},
      {{"kind": "source", "code": "source.two", "params": {{"a": master, "b": "partner portal (seeded)"}}, "ref": ""}}]
clusters, accepted, review = {{}}, [], []
for u, s, c in rows:
    clusters[u] = masters.get(c, u)
    if s != master and c in masters:
        accepted.append({{"a": masters[c], "b": u, "p": 0.9, "evidence": ev}})
    elif s != master and u.endswith(":PA4"):
        review.append({{"a": "shop.dealers:D3", "b": u, "p": 0.55, "evidence": ev}})
counted = {{"reviewer": 0, "simulated": len(accepted), "open": 0}}
answer = {{
    "format": "resolution/3", "state": "certified", "records": len(rows),
    "sources": {{s: sum(r[1] == s for r in rows) for s in sorted({{r[1] for r in rows}})}}, "candidate_pairs": 40,
    "labels": {{"training": counted, "accept_test": counted, "reject_test": counted, "total": 3 * len(accepted)}},
    "truth_stands_in": not a.no_truth, "certified_on": "simulated", "requests": {{"training": [], "accept_test": [], "reject_test": []}},
    "lines": {{"accept_at": 0.6, "reject_below": 0.2, "target": 0.95, "delta": 0.05}}, "certified": True,
    "duplicate_free": {{"sources": list(a.duplicate_free), "pairs_ruled_out": 0}},
    "counts": {{"accept": len(accepted), "review": len(review), "reject": 0}},
    "clusters": clusters, "accepted": accepted, "review": review, "review_total": len(review),
    "given": {{"n_cal": a.n_cal, "n_ltt": a.n_ltt, "answers": len(said)}},
}}
a.out.write_text(json.dumps(answer, indent=1, sort_keys=True) + "\\n", encoding="utf-8")
'''


def _arguments_of_the_linker() -> str:
    """The `ap.add_argument(...)` lines of the real linker's `main`, as source."""
    tree = ast.parse(LINKER.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [s for s in main.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
             and isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "add_argument"]
    assert len(calls) >= 4, "the linker's command line is no longer where this test reads it"
    return "\n".join(ast.unparse(c) for c in calls)


def _stand_in(folder: Path) -> Path:
    """A folder laid out as the build expects the linker's: `link_records.py` and its own
    `.venv/bin/python` (here a wrapper around this interpreter)."""
    (folder / ".venv" / "bin").mkdir(parents=True)
    python = folder / ".venv" / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    (folder / "link_records.py").write_text(STAND_IN.format(arguments=_arguments_of_the_linker(), columns=list(COLUMNS)),
                                            encoding="utf-8")
    return folder


def _yard_with_truth(yard: dict[str, Any]) -> dict[str, Any]:
    yard["out"].mkdir(parents=True, exist_ok=True)
    (yard["out"] / "seed_truth.json").write_text(json.dumps({"seed": 7, "of": "shop.dealers", "truth": TRUTH}), encoding="utf-8")
    return yard


def _decide(out: Path, *lines: dict[str, Any]) -> None:
    with open(out / "decisions.jsonl", "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(line) + "\n" for line in lines)


def test_a_build_with_the_linker_runs_it_by_its_own_command_line_and_its_proposal_file_is_the_one_without_it(
        yard: dict[str, Any], tmp_path: Path) -> None:
    assert os.access(LINKER, os.R_OK)
    linker = _stand_in(tmp_path / "linker")
    plain = {**yard, "out": tmp_path / "plain"}
    _yard_with_truth(yard)
    summary = build(**yard, resolver=linker)
    export = tp._export(yard)
    links = sorted((e["from"], e["to"]) for e in export["edges"] if e["kind"] == "same_as")
    assert links == [("partner.accounts:PA1", "shop.dealers:D1"), ("partner.accounts:PA2", "shop.dealers:D1"),
                     ("partner.accounts:PA3", "shop.dealers:D2")]
    assert summary["resolution"]["linked"] == 3 and export["meta"]["resolution"]["certified_on"] == "simulated"
    given = json.loads((yard["out"] / "resolution.json").read_text(encoding="utf-8"))["given"]
    assert given == {"n_cal": tp.ESTATE["link"]["labels"]["fit"], "n_ltt": tp.ESTATE["link"]["labels"]["test"], "answers": 0}
    # The same decisions on both builds: two maker spellings, the dealers, and a person's link
    # of a dealer to a seeded account.
    left = next(r for r in export["review"] if r["id"] == "same:shop.dealers:D3~partner.accounts:PA4")
    lines = [tp._line(tp._pair(export, "Alfa")), tp._line(tp._pair(export, "Beta")),
             tp._line(tp._concept(export, "concept:shop.dealers")), tp._line(left)]
    _decide(yard["out"], *lines)
    build(**yard, resolver=linker)  # the person's link is passed to the linker and decided by them
    after = tp._export(yard)
    assert json.loads((yard["out"] / "resolution.json").read_text(encoding="utf-8"))["given"]["answers"] == 1
    assert ("partner.accounts:PA4", "shop.dealers:D3", "reviewer-1") in {
        (e["from"], e["to"], e.get("decided_by")) for e in after["edges"] if e["kind"] == "same_as"}
    with_linker = propose(yard["config"], yard["out"])
    build(**plain)
    _decide(plain["out"], *lines)
    without = propose(plain["config"], plain["out"])
    assert (yard["out"] / "proposal.yaml").read_bytes() == (plain["out"] / "proposal.yaml").read_bytes()
    assert with_linker["entries"] == without["entries"] and len(with_linker["entries"]) == 3
    assert {"subject": left["id"], "why": "seeded"} in with_linker["not_written"]
    assert yaml.safe_load((yard["out"] / "proposal.yaml").read_text(encoding="utf-8"))["source"] == "ontology"


def test_the_stand_in_refuses_what_the_linker_refuses(tmp_path: Path) -> None:
    """The stand-in's command line is the linker's: an argument the linker does not take stops it."""
    import subprocess

    linker = _stand_in(tmp_path / "linker")
    done = subprocess.run([str(linker / ".venv" / "bin" / "python"), "link_records.py", "--records", "r.parquet", "--out", "o.json",
                           "--n-calibration", "4"], cwd=linker, capture_output=True, text=True, check=False)
    assert done.returncode == 2 and "unrecognized arguments: --n-calibration" in done.stderr
