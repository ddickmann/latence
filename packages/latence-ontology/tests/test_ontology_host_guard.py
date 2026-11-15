"""The export holds only what every signed-in person of a host may read (ticket 46, the host's
ticket 238). The host refuses to serve an export that holds a row of an object about persons, a
row keyed by a personal column, a value set over a personal column, a row that carries more than
its id, concept, label, key and count, a reference to a row or concept it does not hold, or
anything it cannot check (another format, a concept that does not say what its rows may hold,
an attribute that does not say whether it is personal).

The rules are checked here as the host states them, by code of this side's own: no code of the
host is in this repository. With the host's correction of 14 Nov (its ticket 238, option A of
this side's ticket 46): a row of a value set may carry ``aliases`` (its other spellings), and a
pair's end may name a row the pair's ``merged_into`` says it became, where the export holds it."""

from __future__ import annotations

import json
from typing import Any

from _ontology_toy import SENSITIVE, Project
from latence_ontology.cli import build

ROW_FIELDS = {"id", "concept", "label", "key", "rows"}
VALUE_ROW_FIELDS = ROW_FIELDS | {"aliases"}
ROWS_MAY_HOLD = {"available", "withheld_personal", "without_personal_columns"}


def what_a_host_refuses(export: dict[str, Any]) -> list[str]:
    """Why a host would not serve the export to every signed-in person; empty when it would."""
    if export.get("schema") != "graph-export/3":
        return ["another format"]
    why: list[str] = []
    concepts = {c["id"]: c for c in export["concepts"]}
    personal: dict[str, set[str]] = {}
    for c in export["concepts"]:
        if c.get("instances") not in ROWS_MAY_HOLD:
            why.append(f"{c['id']}: does not say what its rows may hold")
        for a in c.get("attributes") or []:
            if not isinstance(a.get("sensitive"), bool):
                why.append(f"{c['id']}.{a.get('column')}: does not say whether it is personal")
            elif a["sensitive"]:
                personal.setdefault(str(c.get("object_name")), set()).add(str(a["column"]))
    rows = {}
    for n in export["nodes"]:
        rows[n["id"]] = n
        c = concepts.get(n["concept"])
        allowed = VALUE_ROW_FIELDS if c is not None and c.get("kind") == "reference" else ROW_FIELDS
        if set(n) - allowed:
            why.append(f"a row of {n['concept']} carries {sorted(set(n) - allowed)}")
        if c is None:
            why.append(f"a row of {n['concept']}, which is no concept of the export")
        elif c.get("instances") == "withheld_personal":
            why.append(f"a row of {c['id']}, an object about persons")
        elif set(c.get("key") or []) & personal.get(str(c.get("object_name")), set()):
            why.append(f"a row of {c['id']}, keyed by a personal column")
        elif c.get("kind") == "reference" and any(c.get("column") in personal.get(o, set()) for o in c.get("objects") or []):
            why.append(f"a value of {c['id']}, a personal column")
    named = [(f"edge {e['kind']}", e[end]) for e in export["edges"] for end in ("from", "to", "via") if isinstance(e.get(end), str)]
    pairs = [*export["review"], *(p for sample in (export.get("label_requests") or {}).values() for p in sample)]
    named += [(f"pair {p['id']}", p[end]) for p in pairs for end in ("a", "b") if not (p[end] not in rows and p.get("merged_into") in rows)]
    for where, name in named:
        if name.startswith("concept:") and name not in concepts or not name.startswith("concept:") and name not in rows:
            why.append(f"{where} names what the export does not hold")
    return why


def test_every_personal_column_says_so_and_every_concept_says_what_its_rows_may_hold(project: Project) -> None:
    build(**project)
    export = json.loads((project["out"] / "ontology.json").read_text(encoding="utf-8"))
    said = {(c["object_name"], a["column"]): a["sensitive"] for c in export["concepts"] if c.get("object_name") for a in c.get("attributes") or []}
    assert all(isinstance(v, bool) for v in said.values()) and all(c.get("instances") in ROWS_MAY_HOLD for c in export["concepts"])
    personal = {tuple(s.rsplit(".", 1)) for s in SENSITIVE}
    shown = {k for k in said if k in personal}
    assert shown and all(said[k] is True for k in shown)  # every personal column the export names says it is personal
    assert not [k for k, v in said.items() if v and k not in personal]  # and no other does
    keyed = {(c["object_name"], k) for c in export["concepts"] if c.get("object_name") for k in c.get("key") or []}
    assert not keyed & personal or all(
        next(c for c in export["concepts"] if c.get("object_name") == o)["instances"] != "available" for o, _ in keyed & personal)


def test_a_build_passes_what_the_host_checks_before_it_serves_an_export(project: Project) -> None:
    build(**project)
    export = json.loads((project["out"] / "ontology.json").read_text(encoding="utf-8"))
    assert {c["instances"] for c in export["concepts"]} >= {"withheld_personal", "available"}  # the rules have something to bite on
    assert what_a_host_refuses(export) == []
    # The check can disagree: a row of an object about persons, or a field a row may not carry, is refused.
    owners = next(c for c in export["concepts"] if c["id"] == "concept:shop.owners")
    bad = {**export, "nodes": [*export["nodes"], {"id": "shop.owners:O7", "concept": owners["id"], "label": "x", "key": "O7"}]}
    assert what_a_host_refuses(bad) == ["a row of concept:shop.owners, an object about persons"]
    first = export["nodes"][0]
    assert what_a_host_refuses({**export, "nodes": [{**first, "email": "x"}, *export["nodes"][1:]]}) == [f"a row of {first['concept']} carries ['email']"]


def test_a_build_that_applied_a_persons_same_value_passes_too(yard: dict[str, Any]) -> None:
    """The case that broke the first guard: two spellings a person called one value, applied by
    the next build. The kept row carries the other spelling, and the decided pair names a row
    that became the kept one."""
    build(**yard)
    export = json.loads((yard["out"] / "ontology.json").read_text(encoding="utf-8"))
    alfa = next(r for r in export["review"] if r["a"].startswith("value:") and "Alfa" in r["id"])
    (yard["out"] / "decisions.jsonl").write_text(json.dumps({"subject": alfa["id"], "verdict": "approved", "by": "reviewer-1", "claim": alfa["claim"]}) + "\n",
                                                 encoding="utf-8")
    build(**yard)
    after = json.loads((yard["out"] / "ontology.json").read_text(encoding="utf-8"))
    assert any("aliases" in n for n in after["nodes"]) and any("merged_into" in r for r in after["review"])  # what the first guard refused
    assert what_a_host_refuses(after) == []
    # Still refused: aliases on a row of a concept that is no value set, and a pair whose end became no row it holds.
    dealer = next(n for n in after["nodes"] if n["concept"] == "concept:shop.dealers")
    assert what_a_host_refuses({**after, "nodes": [*after["nodes"], {**dealer, "id": "shop.dealers:X", "aliases": ["y"]}]}) == [
        "a row of concept:shop.dealers carries ['aliases']"]
    gone = next(r for r in after["review"] if "merged_into" in r)
    assert what_a_host_refuses({**after, "review": [{**r, "merged_into": "value:maker:nowhere"} if r is gone else r for r in after["review"]]}) == [
        f"pair {gone['id']} names what the export does not hold"]
