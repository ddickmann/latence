"""Structure between the concepts that the data itself proves.

The induced model is flat: concepts and the relations between them. This module adds what
makes it hierarchical, from counts alone, with no model and no guess:

- **roll-ups**: one value set groups another (every model belongs to one make, every component
  to one system). Checked in every object that carries both; a roll-up that breaks for more
  than the support floor allows is not proposed, and one that breaks for a few values says how
  many. Master data hangs below the value set it refers to (a dealer below its dealer group).
- **equivalent value sets**: two columns that cut the rows the same way (market and distance
  unit). They are one level, not two.
- **part-of**: a concept whose key contains another concept's key is a part of it (a claim
  line of a claim); one whose key contains two is a link between them.
- **the same thing at another layer**: a derived object keyed like an entity describes that
  entity.

For every roll-up step the instance level gets ``broader`` edges: which value sits under
which. Where a child has more than one parent in the data the most frequent one is taken and
the edge says it is contested.

Everything here is a proposal with its support on the data. What it cannot give is meaning:
that a dealer is an organisation and a claim an event. That needs names and a person.
"""

from __future__ import annotations

import itertools
from typing import Any

from .codes import say
from .ledger import Decisions
from .induce import SUPPORT_FLOOR
from .reader import CatalogObject, Reader

MIN_OBJECTS = 2  # a roll-up seen in one object only may be an accident of that object
MIN_ROWS_FOR_A_NAME = 10  # fewer rows say nothing about whether a column groups them


def _value_sets(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["column"]: c for c in model["concepts"] if c["kind"] == "reference"}


def _carried(reader: Reader, objects: dict[str, CatalogObject], vs: dict[str, dict[str, Any]], a: str, b: str,
             personal: set[str]) -> list[CatalogObject]:
    """Objects that carry both columns, neither of them sensitive. A personal object may be
    counted over (an aggregate of two non-personal columns), like everywhere else in induction."""
    out = []
    for name in vs[a]["objects"]:
        o = objects.get(name)
        if o is None or name not in vs[b]["objects"]:
            continue
        ca, cb = o.column(a), o.column(b)
        if ca is not None and cb is not None and not ca.sensitive and not cb.sensitive:
            out.append(o)
    return out


def _breaks(reader: Reader, o: CatalogObject, a: str, b: str) -> tuple[int, int]:
    """(values of a seen with a value of b, how many of them go with more than one b)."""
    n, bad = reader.query(
        f"SELECT count(*), coalesce(sum(CASE WHEN k > 1 THEN 1 ELSE 0 END), 0) FROM "
        f"(SELECT {reader.ident(a)}, count(DISTINCT {reader.ident(b)}) k FROM {reader.quote(o)} WHERE {reader.ident(a)} IS NOT NULL AND {reader.ident(b)} IS NOT NULL GROUP BY 1) t")[0]
    return int(n), int(bad)


MAX_BREAKING_NAMED = 20


def _breaking(reader: Reader, objects: list[CatalogObject], a: str, b: str) -> list[dict[str, Any]]:
    """The values of a that go with more than one value of b, each with the values it goes
    with, most frequent first."""
    parts = [f"SELECT {reader.as_text(reader.ident(a))} x, {reader.as_text(reader.ident(b))} y FROM {reader.quote(o)} WHERE {reader.ident(a)} IS NOT NULL AND {reader.ident(b)} IS NOT NULL" for o in objects]
    rows = reader.query(f"SELECT x, y, count(*) FROM ({' UNION ALL '.join(parts)}) t GROUP BY 1, 2 ORDER BY 1, 3 DESC, 2")
    parents: dict[str, list[str]] = {}
    for x, y, _n in rows:
        parents.setdefault(x, []).append(y)
    return [{"value": x, "parents": ys} for x, ys in sorted(parents.items()) if len(ys) > 1][:MAX_BREAKING_NAMED]


def entity_names(reader: Reader, model: dict[str, Any], personal_objects: tuple[str, ...] = ()) -> dict[str, list[dict[str, Any]]]:
    """Value sets that are the name of an entity and not a way to group it.

    The induction takes a column for a value set when several objects share it and its values
    repeat somewhere. A title does that too: it repeats in every table that reports on the
    thing it names. In the table of the thing itself it does not: there, most rows have a
    value of their own. So a value set with more than half as many values as rows in the table
    of an entity with a single key is that entity's name. Nothing rolls up from a name."""
    objects = {o.full_name: o for o in reader.objects()}
    out: dict[str, list[dict[str, Any]]] = {}
    for col, v in sorted(_value_sets(model).items()):
        for c in model["concepts"]:
            name = str(c.get("object_name"))
            o = objects.get(name)
            if o is None or c["kind"] not in ("entity", "fact") or len(c.get("key") or []) != 1 or name in personal_objects or name not in v["objects"]:
                continue
            column = o.column(col)
            if column is None or column.sensitive:
                continue
            rows, values = reader.query(f"SELECT count({reader.ident(col)}), count(DISTINCT {reader.ident(col)}) FROM {reader.quote(o)}")[0]
            if rows >= MIN_ROWS_FOR_A_NAME and 2 * values > rows:
                out.setdefault(col, []).append({"entity": c["id"], "object": name, "keys": int(rows), "values": int(values)})
    return out


def build_structure(reader: Reader, model: dict[str, Any], personal_objects: tuple[str, ...] = (),
                    decisions: Decisions | None = None) -> dict[str, Any]:
    """Everything here is a proposal. A person's decision changes it where it is built: a
    rejected step is gone before the steps are reduced to the direct ones, so the step it hid
    comes back; a rejected equivalence leaves two levels; an approved element says so."""
    asked = decisions or Decisions()
    objects = {o.full_name: o for o in reader.objects()}
    vs = _value_sets(model)
    personal = set(personal_objects)
    concepts = {c["id"]: c for c in model["concepts"]}
    names = entity_names(reader, model, personal_objects)
    not_proposed = [{"id": f"not_proposed:value:{col}", "concept": vs[col]["id"], "entity": e["entity"],
                     "signals": [say("names_an_entity", e["entity"], value_set=vs[col]["label"], entity=concepts[e["entity"]]["label"], keys=e["keys"], values=e["values"])]}
                    for col, found in sorted(names.items()) for e in found[:1]]

    # ---- which value set determines which, and how cleanly ----------------------------------
    checked: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in itertools.permutations(sorted(vs), 2):
        if a in names or b in names:
            continue  # a name is no level
        both = _carried(reader, objects, vs, a, b, personal)
        if len(both) < MIN_OBJECTS:
            continue
        n = bad = 0
        for o in both:
            k, v = _breaks(reader, o, a, b)
            n, bad = n + k, bad + v
        if n and 1 - bad / n >= SUPPORT_FLOOR:
            checked[(a, b)] = {"objects": [o.full_name for o in both], "values_checked": n, "values_breaking": bad, "share": round(1 - bad / n, 4)}

    equivalent = []
    for a, b in sorted({tuple(sorted(k)) for k in checked if (k[1], k[0]) in checked}):
        if checked[(a, b)]["values_breaking"] == 0 and checked[(b, a)]["values_breaking"] == 0:
            keep, other = (a, b) if len(vs[a]["objects"]) >= len(vs[b]["objects"]) else (b, a)
            el = {"id": f"equivalent:{keep}={other}", "concepts": [vs[keep]["id"], vs[other]["id"]], "status": "proposed",
                  "signals": [say("same_grouping", objects=len(checked[(a, b)]["objects"]), a=keep, b=other)]}
            if not asked.rejected("equivalent", el):
                equivalent.append(asked.marked("equivalent", el))
    lesser = {e["concepts"][1].removeprefix("concept:value:") for e in equivalent}
    same = {frozenset(c.removeprefix("concept:value:") for c in e["concepts"]) for e in equivalent}

    steps = {(a, b): s for (a, b), s in checked.items()
             if frozenset((a, b)) not in same and a not in lesser and b not in lesser and vs[a]["values"] > vs[b]["values"]}
    steps = {(a, b): s for (a, b), s in steps.items()
             if not asked.rejected("roll_up", {"id": f"roll_up:value:{a}->value:{b}", "from": vs[a]["id"], "to": vs[b]["id"], "kind": "roll_up"})}
    for a, c in sorted(steps):  # what follows from two steps says nothing new
        if any((a, b) in steps and (b, c) in steps for b in vs if b not in (a, c)):
            steps.pop((a, c))

    roll_ups = []
    for (a, b), s in sorted(steps.items()):
        clean = s["values_breaking"] == 0
        signal = (say("functional_dependency.all", objects=len(s["objects"]), a=a, b=b) if clean else
                  say("functional_dependency.most", objects=len(s["objects"]), holding=s["values_checked"] - s["values_breaking"], checked=s["values_checked"], a=a, b=b))
        roll_ups.append(asked.marked("roll_up", {
            "id": f"roll_up:value:{a}->value:{b}", "from": vs[a]["id"], "to": vs[b]["id"], "kind": "roll_up", "status": "proposed",
            "label": f"{vs[a]['label']} → {vs[b]['label']}", "support": s,
            **({} if clean else {"breaking": _breaking(reader, [objects[n] for n in s["objects"]], a, b)}),
            "signals": [signal]}))
    # Master data below the value set it refers to: the last level of a hierarchy.
    refers = [r for r in model["relations"] if r["to"].startswith("concept:value:") and not r["from"].startswith("concept:value:")]
    below: dict[str, list[str]] = {}
    for r in refers:
        below.setdefault(r["from"], []).append(r["to"].removeprefix("concept:value:"))
    for cid, cols in sorted(below.items()):
        c = concepts.get(cid)
        if c is None or c["kind"] != "entity" or len(c.get("key") or []) != 1 or c.get("object_name") in personal:
            continue
        levels = {x for step in steps for x in step}
        finest = [a for a in cols if a in levels and a not in lesser and a not in names and not any((x, a) in steps for x in cols)]  # no finer level on this entity
        o = objects.get(str(c.get("object_name")))
        for a in sorted(finest):
            support = None
            signal = say("refers_to_a_level.all", concept=c["label"], level=a)
            if o is not None and o.column(a) is not None and not o.column(a).sensitive:  # type: ignore[union-attr]
                rows, filled = reader.query(f"SELECT count(*), count({reader.ident(a)}) FROM {reader.quote(o)}")[0]
                support = {"rows": int(rows), "rows_with_the_level": int(filled), "share": round(filled / rows, 4) if rows else None}
                if filled < rows:  # a level only some rows carry: the others are in no tree
                    signal = say("refers_to_a_level.some", filled=int(filled), rows=int(rows), concept=c["label"], level=a)
            el = {"id": f"roll_up:{c['object_name']}->value:{a}", "from": cid, "to": vs[a]["id"], "kind": "belongs_to", "status": "proposed",
                  "label": f"{c['label']} → {vs[a]['label']}", "support": support, "signals": [signal]}
            if not asked.rejected("roll_up", el):
                roll_ups.append(asked.marked("roll_up", el))

    # ---- part-of, from keys ------------------------------------------------------------------
    part_of = []
    for c in model["concepts"]:
        key = set(c.get("key") or [])
        if len(key) < 2 or c["kind"] == "derived":
            continue
        # A reference to one column of a whole's key counts too: it is a second thing the rows
        # are about, so the concept links two things and is a part of neither.
        wholes = [r for r in model["relations"] if r["from"] == c["id"] and set(r["columns"]) <= key
                  and concepts.get(r["to"], {}).get("kind") in ("entity", "fact")
                  and (r.get("partial_key") or set(concepts[r["to"]].get("key") or []) == set(r.get("to_columns") or r["columns"]))]
        for r in sorted(wholes, key=lambda r: r["to"]):
            code = "key_contains_part_of_key" if r.get("partial_key") else "key_contains_key"
            el = {"id": f"part_of:{c['id'].removeprefix('concept:')}->{r['to'].removeprefix('concept:')}", "from": c["id"], "to": r["to"],
                  "kind": "part_of" if len(wholes) == 1 else "links", "status": "proposed", "relation": r["id"],
                  "signals": [say(code, concept=c["label"], key=list(c["key"]), whole=concepts[r["to"]]["label"], columns=list(r["columns"]))]}
            if not asked.rejected("part_of", el):
                part_of.append(asked.marked("part_of", el))

    # ---- the same thing at another layer -----------------------------------------------------
    by_key: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for c in model["concepts"]:
        if c.get("key") and c["kind"] != "reference":
            by_key.setdefault(tuple(sorted(c["key"])), []).append(c)
    describes = []
    for key, group in sorted(by_key.items()):
        base = [c for c in group if c["kind"] != "derived"]
        if len(base) != 1:
            continue
        for d in sorted((c for c in group if c["kind"] == "derived"), key=lambda c: c["id"]):
            el = {"id": f"describes:{d['id'].removeprefix('concept:')}->{base[0]['id'].removeprefix('concept:')}", "from": d["id"], "to": base[0]["id"],
                  "kind": "describes", "status": "proposed", "signals": [say("same_key", base=base[0]["label"], key=list(key))]}
            if not asked.rejected("describes", el):
                describes.append(asked.marked("describes", el))

    return {"roll_ups": roll_ups, "equivalent": equivalent, "part_of": part_of, "describes": describes, "not_proposed": not_proposed,
            "_steps": {f"{a}->{b}": s["objects"] for (a, b), s in steps.items()}}


def broader_edges(reader: Reader, structure: dict[str, Any], known: set[str]) -> list[dict[str, Any]]:
    """For every roll-up between two value sets: which value sits under which, counted over the
    objects that carry both. A value the data puts under two values gets no edge: the step names
    it among its breaking values, and a person says where it belongs."""
    objects = {o.full_name: o for o in reader.objects()}
    out = []
    for step, names in sorted(structure["_steps"].items()):
        a, b = step.split("->")
        parts = [f"SELECT {reader.as_text(reader.ident(a))} x, {reader.as_text(reader.ident(b))} y FROM {reader.quote(objects[n])} WHERE {reader.ident(a)} IS NOT NULL AND {reader.ident(b)} IS NOT NULL"
                 for n in names]
        rows = reader.query(f"SELECT x, y, count(*) FROM ({' UNION ALL '.join(parts)}) t GROUP BY 1, 2 ORDER BY 1, 3 DESC, 2")
        seen: dict[str, int] = {}
        for x, _y, _n in rows:
            seen[x] = seen.get(x, 0) + 1
        done: set[str] = set()
        for x, y, n in rows:
            if x in done or seen[x] > 1:
                continue
            done.add(x)
            edge = {"from": f"value:{a}:{x}", "to": f"value:{b}:{y}", "kind": "broader", "relation": f"roll_up:value:{a}->value:{b}", "rows": int(n)}
            if edge["from"] in known and edge["to"] in known:
                out.append(edge)
    return out


def explain_review(review: list[dict[str, Any]], edges: list[dict[str, Any]], nodes: list[dict[str, Any]], concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Two names that may be the same, seen against the hierarchy: if both sit under the same
    value one level up (two manufacturer names under one make), that is a reason more; if they
    sit under different ones, it is a reason against. Returns new items; nothing is decided."""
    parent = {e["from"]: e["to"] for e in edges if e["kind"] == "broader"}
    label = {n["id"]: n["label"] for n in nodes}
    of = {n["id"]: n["concept"] for n in nodes}
    concept_label = {c["id"]: c["label"] for c in concepts}
    out = []
    for item in review:
        pa, pb = parent.get(item["a"]), parent.get(item["b"])
        if pa is None or pb is None:
            out.append(item)
            continue
        level = concept_label.get(of.get(pa, ""), "")
        if pa == pb:
            out.append({**item, "reasons": [*item["reasons"], say("same_parent", pa, level=level, parent=label.get(pa, pa))], "strength": "strong", "same_parent": pa})
        else:
            out.append({**item, "reasons": [*item["reasons"], say("different_parents", level=level, a=label.get(pa, pa), b=label.get(pb, pb))], "strength": "weak"})
    return out
