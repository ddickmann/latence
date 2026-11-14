"""Drift: the catalog changed under an element a person decided (ticket 53; the owner, 14 Nov).

A decision is made on what an element stated (its claim). When the catalog changes under a
decided element, the next build says what drifted, and the element goes back to review:

- **renamed**: a column the decided statement names is gone from its object, and the object
  has a column it did not have (``columns``: old name to new, where the types say which);
- **retyped**: a column of the decided concept has another type now;
- **split**: the decided concept's object is gone, and two or more new objects hold its columns
  between them (``now``: their concepts);
- **dropped**: the decided element's object is gone, and nothing holds its columns.

The old decision stays on the record and is listed as stale with the kind of drift; it is no
longer applied, so the element as it is now is a proposal again. What was decided is read from
the ledger: the first assertion of the element whose claim is the claim the decision was made
on, with what the ledger knew of its object up to then. Neither a decision nor an assertion
carries a time today, so a change made after that first statement and before the decision is
reported as drift too: the cautious side, since it sends the element back to review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ledger import claim_of

KINDS = ("concept", "relation")


def prior_statements(path: Path, builds: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Every statement the ledger holds per element, oldest first, each with the last line of
    the build that wrote it (``_through``; ``builds`` says where each build's lines begin; a
    ledger without it is read as one build)."""
    out: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    starts = sorted(int(json.loads(x)["first_line"]) for x in builds.read_text(encoding="utf-8").splitlines() if x.strip()) \
        if builds is not None and builds.exists() else []
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for n, line in enumerate(lines):
        a = json.loads(line)
        if a.get("kind") in KINDS:
            through = min([s for s in starts if s > n], default=len(lines)) - 1
            out.setdefault(a["subject"], []).append({**a, "_line": n, "_through": through})
    return out


def _columns(payloads: list[dict[str, Any]], obj: str) -> dict[str, str]:
    """An object's columns as statements about it name them, with a type where one is stated."""
    cols: dict[str, str] = {}
    for kind, p in payloads:
        if kind == "concept" and p.get("object_name") == obj:
            cols.update({str(a["column"]): str(a.get("type") or "") for a in p.get("attributes") or []})
            cols.update({str(k): cols.get(str(k), "") for k in p.get("key") or []})
        elif kind == "relation" and str(p.get("from", "")).removeprefix("concept:") == obj:
            cols.update({str(c): cols.get(str(c), "") for c in p.get("columns") or []})
    return cols


def find_drift(decisions: dict[str, dict[str, Any]], model: dict[str, Any], prior: dict[str, list[dict[str, Any]]],
               objects: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """What drifted under each decided concept or relation. ``objects`` are the catalog's
    objects as read now (full name -> column -> type)."""
    now = {el["id"]: (kind, el) for kind, part in (("concept", model["concepts"]), ("relation", model["relations"])) for el in part}
    ordered = sorted(((a["_line"], a["kind"], a["payload"]) for statements in prior.values() for a in statements), key=lambda x: x[0])
    out = []
    for subject, d in sorted(decisions.items()):
        statements = prior.get(subject) or []
        if not statements or statements[0]["kind"] not in KINDS:
            continue
        kind = statements[0]["kind"]
        # The first statement on the claim the decision was made on: a claim does not cover every
        # column, so later statements under the same claim may already hold the drift.
        decided = next((a for a in statements if d.get("claim") and claim_of(kind, a["payload"]) == d["claim"]), statements[-1])
        then = decided["payload"]
        # What the ledger knew when that statement was made: the object's columns then.
        latest = [(k, p) for n, k, p in ordered if n <= decided["_through"]]
        known_before = {str(p.get("object_name")) for k, p in latest if k == "concept"}
        obj = str(then.get("object_name") if kind == "concept" else str(then.get("from", "")).removeprefix("concept:"))
        if obj.startswith("value:"):
            continue  # a value set is no object of the catalog
        was = _columns(latest, obj)
        named = set(then.get("columns") or []) if kind == "relation" else set(was)
        entry: dict[str, Any] | None = None
        if obj not in objects:
            new = sorted(o for o in objects if o not in known_before and set(objects[o]) & set(was))
            covered = set().union(*(set(objects[o]) for o in new)) if new else set()
            if len(new) >= 2 and set(was) <= covered:
                entry = {"drift": "split", "now": [f"concept:{o}" for o in new]}
            else:
                entry = {"drift": "dropped", "now": []}
        else:
            gone = sorted(c for c in named if c not in objects[obj])
            added = sorted(c for c in objects[obj] if c not in was)
            if gone and added:
                renames = {g: a for g in gone for a in added
                           if (was.get(g) and was.get(g) == objects[obj][a]) or (len(gone) == 1 and len(added) == 1)}
                entry = {"drift": "renamed", "columns": renames, "now": [i for i, (k, el) in now.items() if k == kind and i != subject
                                                                          and (el.get("object_name") == obj or str(el.get("from", "")).removeprefix("concept:") == obj)
                                                                          and set(renames.values()) & set(el.get("columns") or el.get("key") or [])]}
            elif gone:
                entry = {"drift": "dropped", "now": []}
            elif kind == "concept":
                retyped = sorted(c for c, t in was.items() if t and c in objects[obj] and objects[obj][c] and objects[obj][c] != t)
                if retyped:
                    entry = {"drift": "retyped", "columns": {c: {"was": was[c], "now": objects[obj][c]} for c in retyped}, "now": [subject]}
        if entry is not None:
            out.append({"subject": subject, "kind": kind, "decided_on": str(d.get("claim") or ""), "verdict": d["verdict"], "by": d["by"], **entry})
    return out
