"""The instance level: entities, the values they refer to, and facts as counted edges.

What becomes a node:

- every value of a value set (a make, a market), with how many rows carry it;
- every row of an entity object that has a single-column key, is not personal and is small
  enough to be master data.

What never becomes a node: a row of a fact object (a claim, a claim line) and anything from an
object the estate marks as personal. A fact is shown as a count on an edge: "this dealer has
312 claims", "this dealer and this part meet in 14 claims", each with the objects and columns
the count was taken over, so it can be re-run.

Names that may stand for the same thing (two spellings of a manufacturer) are listed as
review items with the reason. Nothing is merged here: without labels there is no calibrated
band, so the export says ``certified: false`` and a person decides.

No value of a sensitive column is read: such a column is never selected.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from .induce import _q, label_of
from .reader import CatalogObject, Reader

MAX_ENTITY_ROWS = 2_000
MAX_VALUES = 400
MAX_PAIR_EDGES = 1_500
NAME_COLUMNS = ("name", "name_en", "title", "label", "subject", "description")
LEGAL = {"inc", "llc", "ltd", "corp", "corporation", "co", "company", "gmbh", "ag", "kg", "plc", "lp", "sa", "the", "of", "and"}
# Words many organisations carry: sharing one is no reason to think two names are the same.
GENERIC = {"motor", "motors", "group", "holdings", "international", "global", "services", "systems", "america", "american",
           "usa", "us", "north", "south", "east", "west", "europe", "operations", "manufacturing", "engineering", "distributors"}


def _label_column(obj: CatalogObject, key: str) -> str:
    for want in NAME_COLUMNS:
        col = obj.column(want)
        if col is not None and not col.sensitive:
            return want
    return key


def _core(name: str) -> tuple[list[str], list[str]]:
    """The content words of a name, and those of a bracketed second name
    ("Chrysler (FCA US, LLC)" -> ["chrysler"], ["fca", "us"])."""
    inner = " ".join(re.findall(r"\(([^()]*)\)", name))
    outer = re.sub(r"\([^()]*\)", " ", name)

    def words(s: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in LEGAL]

    return words(outer), words(inner)


def possibly_same(values: list[tuple[str, int]]) -> list[dict[str, Any]]:
    """Pairs of names that may stand for the same thing, with the reason. A shared first
    content word is the strong reason; a shared rare word anywhere is the weak one."""
    cores = {v: _core(v) for v, _ in values}
    seen_in: dict[str, int] = {}
    for outer, inner in cores.values():
        for w in set(outer + inner):
            seen_in[w] = seen_in.get(w, 0) + 1
    out = []
    for (a, na), (b, nb) in itertools.combinations(sorted(values), 2):
        (oa, ia), (ob, ib) = cores[a], cores[b]
        if not oa or not ob:
            continue
        shared = [w for w in dict.fromkeys(oa + ia) if w in set(ob + ib) and len(w) >= 3 and w not in GENERIC]
        rare_at = max(2, min(3, len(values) // 4))  # a word in more names than this is not a reason
        if oa[0] == ob[0]:
            reason, strength = f"both names begin with “{oa[0]}”", "strong"
        elif shared and any(seen_in[w] <= rare_at for w in shared):
            rare = next(w for w in shared if seen_in[w] <= rare_at)
            reason, strength = f"both names contain “{rare}”", "weak"
        else:
            continue
        out.append({"a": a, "b": b, "rows_a": na, "rows_b": nb, "reason": reason, "strength": strength,
                    "shared_words": shared})
    return out


def build_instances(reader: Reader, model: dict[str, Any], personal_objects: tuple[str, ...] = (),
                    resolve: tuple[str, ...] = ()) -> dict[str, Any]:
    """Nodes, edges and review items for an induced model. ``personal_objects`` never give
    nodes. ``resolve`` names the value sets whose values are names to compare."""
    objects = {o.full_name: o for o in reader.objects()}
    concepts = {c["id"]: c for c in model["concepts"]}
    personal = set(personal_objects)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    queries: list[str] = []

    def run(sql: str) -> list[tuple[Any, ...]]:
        queries.append(sql)
        return reader.query(sql)

    # ---- value nodes ------------------------------------------------------------------------
    value_cols = {c["column"]: c for c in model["concepts"] if c["kind"] == "reference"}
    for col, c in sorted(value_cols.items()):
        parts = [f"SELECT CAST({_q(col)} AS VARCHAR) AS v FROM {reader.quote(objects[o])} WHERE {_q(col)} IS NOT NULL"
                 for o in c["objects"] if o in objects and o not in personal]
        if not parts:
            continue
        rows = run(f"SELECT v, count(*) FROM ({' UNION ALL '.join(parts)}) t GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT {MAX_VALUES}")
        for v, n in rows:
            nodes.append({"id": f"value:{col}:{v}", "concept": c["id"], "label": str(v), "rows": int(n)})
        if col in resolve:
            for pair in possibly_same([(str(v), int(n)) for v, n in rows]):
                review.append({"id": f"same:{col}:{pair['a']}~{pair['b']}", "kind": "possibly_same", "concept": c["id"],
                               "a": f"value:{col}:{pair['a']}", "b": f"value:{col}:{pair['b']}",
                               **{k: pair[k] for k in ("rows_a", "rows_b", "reason", "strength", "shared_words")}})

    # ---- entity nodes -----------------------------------------------------------------------
    referred_to = {r["to"] for r in model["relations"] if not r["from"].startswith("concept:value:")}
    node_objects: dict[str, tuple[CatalogObject, str]] = {}
    for c in model["concepts"]:
        name = c.get("object_name")
        if c["kind"] not in ("entity",) or name is None or name in personal or name not in objects:
            continue
        obj, key = objects[name], c.get("key") or []
        if len(key) != 1 or not obj.rows or obj.rows > MAX_ENTITY_ROWS or c["id"] not in referred_to:
            continue  # master data is what other objects refer to; a table nothing points at is a list of events
        label_col = _label_column(obj, key[0])
        refs = [col for col in value_cols if obj.column(col) is not None and not obj.column(col).sensitive]  # type: ignore[union-attr]
        select = ", ".join(f"CAST({_q(x)} AS VARCHAR)" for x in [key[0], label_col, *refs])
        for row in run(f"SELECT {select} FROM {reader.quote(obj)} ORDER BY 1"):
            nid = f"{name}:{row[0]}"
            nodes.append({"id": nid, "concept": c["id"], "label": str(row[1]) if row[1] is not None else str(row[0]), "key": str(row[0])})
            for col, v in zip(refs, row[2:]):
                if v is not None:
                    edges.append({"from": nid, "to": f"value:{col}:{v}", "relation": f"relation:{name}.{col}->value:{col}", "kind": "refers_to"})
        node_objects[name] = (obj, key[0])

    # ---- facts as counts on edges -----------------------------------------------------------
    by_child: dict[str, list[dict[str, Any]]] = {}
    for r in model["relations"]:
        child, parent = r["from"].removeprefix("concept:"), r["to"].removeprefix("concept:")
        if parent in node_objects and child in objects and child not in node_objects and len(r["columns"]) == 1:
            if objects[child].column(r["columns"][0]) is not None and not objects[child].column(r["columns"][0]).sensitive:  # type: ignore[union-attr]
                by_child.setdefault(child, []).append(r)
    for child, rels in sorted(by_child.items()):
        obj = objects[child]
        for r in rels:
            parent = r["to"].removeprefix("concept:")
            col = r["columns"][0]
            sql = f"SELECT CAST({_q(col)} AS VARCHAR), count(*) FROM {reader.quote(obj)} WHERE {_q(col)} IS NOT NULL GROUP BY 1 ORDER BY 1"
            for v, n in run(sql):
                edges.append({"from": f"{parent}:{v}", "to": r["from"], "relation": r["id"], "kind": "fact_count", "rows": int(n),
                              "counted": {"object": child, "group_by": [col]}})
        for a, b in itertools.combinations(rels, 2):
            pa, pb = a["to"].removeprefix("concept:"), b["to"].removeprefix("concept:")
            ca, cb = a["columns"][0], b["columns"][0]
            if pa == pb:
                continue
            sql = (f"SELECT CAST({_q(ca)} AS VARCHAR), CAST({_q(cb)} AS VARCHAR), count(*) FROM {reader.quote(obj)} "
                   f"WHERE {_q(ca)} IS NOT NULL AND {_q(cb)} IS NOT NULL GROUP BY 1, 2 ORDER BY 3 DESC, 1, 2 LIMIT {MAX_PAIR_EDGES}")
            for va, vb, n in run(sql):
                edges.append({"from": f"{pa}:{va}", "to": f"{pb}:{vb}", "relation": None, "kind": "meet_in", "rows": int(n),
                              "via": f"concept:{child}", "counted": {"object": child, "group_by": [ca, cb]}})

    known = {n["id"] for n in nodes} | set(concepts)
    edges = sorted((e for e in edges if e["from"] in known and e["to"] in known),
                   key=lambda e: (e["kind"], e["from"], e["to"], str(e.get("via")), str(e.get("relation"))))
    per_concept: dict[str, int] = {}
    for n in nodes:
        per_concept[n["concept"]] = per_concept.get(n["concept"], 0) + 1
    return {
        "nodes": nodes,
        "edges": edges,
        "review": review,
        "meta": {"nodes": len(nodes), "edges": len(edges), "review_items": len(review), "certified": False,
                 "nodes_per_concept": dict(sorted(per_concept.items())), "queries": len(queries),
                 "labels": {c["id"]: label_of(c["id"].split(".")[-1].split(":")[-1]) for c in model["concepts"]}},
    }
