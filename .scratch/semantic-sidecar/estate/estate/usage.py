"""What people ask, as a signal on the elements they use.

A host writes a log of the questions it answered (``usage/1``, one JSON object per line): for
each question the measures, the dimensions and the joins of the query that answered it, and
the terms the record states a meaning for. No value of a filter is in it, and a record that
carries one is refused.

Here the log is read and matched to elements the model already holds, by name: a measure of a
metric view, a value set, a relation. A match adds a count to the element and a signal that
says so. What matches nothing is counted and listed, so that a reader sees what people ask for
that the model does not hold. Nothing is created from use and no status changes: that a join
was used says that someone found it useful, not that it is right.

A term becomes a proposed synonym only where the record states what it means (a question that
was asked back and answered, a synonym the host's own layer holds). Guessing a meaning from
the words of a question would be this engine's invention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .codes import say

FORMAT = "usage/1"
FIELDS = {"format", "source", "synthetic", "id", "language", "answered", "measures", "dimensions", "joins", "terms"}


def read_usage(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("format") != FORMAT:
                raise ValueError(f"{path}:{n}: this build reads {FORMAT}, not {r.get('format')!r}")
            if set(r) - FIELDS or not r.get("id") or not r.get("source"):
                raise ValueError(f"{path}:{n}: a record holds {sorted(FIELDS)}, with an id and a source; got {sorted(set(r) - FIELDS)} beside them")
            for d in r.get("dimensions", []):
                if "values" in d or "value" in d:
                    raise ValueError(f"{path}:{n}: a dimension carries values; a usage record names columns only")
            out.append(r)
    return out


def _bump(counts: dict[str, dict[str, Any]], key: str, record: dict[str, Any]) -> None:
    c = counts.setdefault(key, {"answers": 0, "sources": {}, "synthetic_only": True})
    c["answers"] += 1
    c["sources"][record["source"]] = c["sources"].get(record["source"], 0) + 1
    c["synthetic_only"] = c["synthetic_only"] and bool(record.get("synthetic"))


def attach(model: dict[str, Any], records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The model with what was used counted on its elements, and a summary. Only answered
    questions count. Returns a new model."""
    measures = {(m["concept"].removeprefix("concept:"), m["name"]): m["id"] for m in model["measures"]}
    by_name: dict[str, list[str]] = {}
    for m in model["measures"]:
        by_name.setdefault(m["name"], []).append(m["id"])
    value_sets = {c["column"]: c["id"] for c in model["concepts"] if c["kind"] == "reference"}
    relations = {(r["from"].removeprefix("concept:"), tuple(r["columns"]), r["to"].removeprefix("concept:")): r["id"] for r in model["relations"]}
    concepts = {c["id"]: c for c in model["concepts"]}
    keys = {(c["object_name"], c["key"][0]): c["id"] for c in model["concepts"] if c.get("object_name") and len(c.get("key") or []) == 1}
    carried = {(c["object_name"], a["column"]) for c in model["concepts"] if c.get("object_name") for a in c.get("attributes") or []}

    used: dict[str, dict[str, Any]] = {}
    missed: dict[tuple[str, str], dict[str, Any]] = {}
    terms: dict[str, dict[str, dict[str, Any]]] = {}
    matched = {"measures": 0, "dimensions": 0, "joins": 0, "terms": 0}
    sources: dict[str, int] = {}
    answered = 0

    def miss(kind: str, name: str, **more: Any) -> None:
        entry = missed.setdefault((kind, name), {"kind": kind, "name": name, **more, "answers": 0})
        entry["answers"] += 1

    for r in records:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
        if not r.get("answered"):
            continue
        answered += 1
        for m in r.get("measures", []):
            same_name = by_name.get(m["name"], [])
            mid = measures.get((m.get("object", ""), m["name"])) or (same_name[0] if len(same_name) == 1 else None)
            if mid is None:
                miss("measure", f"{m.get('object', '?')}.{m['name']}")
            else:
                matched["measures"] += 1
                _bump(used, mid, r)
        for d in r.get("dimensions", []):
            where = [tuple(c.rsplit(".", 1)) for c in d.get("columns", [])]  # (object, column)
            hit = sorted({col for _obj, col in where} & set(value_sets))
            keyed = sorted(keys[pair] for pair in where if pair in keys)
            objects = sorted({obj for obj, _col in where})
            # A value set by its column; else the entity whose key the dimension is; else, where
            # it is a column of one object and the concept holds that column, that concept.
            target = value_sets[hit[0]] if hit else keyed[0] if keyed else f"concept:{objects[0]}" if len(objects) == 1 and all(pair in carried for pair in where) else None
            if target is None:
                miss("dimension", d["name"], columns=sorted(d.get("columns", [])))
            else:
                matched["dimensions"] += 1
                _bump(used, target, r)
        for j in r.get("joins", []):
            rid = relations.get((j["from"], tuple(j["columns"]), j["to"]))
            if rid is None:
                miss("join", f"{j['from']}({', '.join(j['columns'])}) -> {j['to']}")
            else:
                matched["joins"] += 1
                _bump(used, rid, r)
        for t in r.get("terms", []):
            if t["element"] not in concepts:
                miss("term", f"{t['text']} -> {t['element']}")
                continue
            matched["terms"] += 1
            entry = terms.setdefault(t["element"], {}).setdefault(t["text"].strip().lower(), {"text": t["text"].strip(), "answers": 0, "how": set()})
            entry["answers"] += 1
            entry["how"].add(t["how"])

    def with_usage(el: dict[str, Any]) -> dict[str, Any]:
        out = dict(el)
        if el["id"] in used:
            u = used[el["id"]]
            out["usage"] = {"answers": u["answers"], "sources": dict(sorted(u["sources"].items())), "synthetic_only": u["synthetic_only"]}
            out["signals"] = [*el.get("signals", []), say("used_in_answers", answers=u["answers"], sources=sorted(u["sources"]))]
        proposed = [{"text": t["text"], "answers": t["answers"], "how": sorted(t["how"])}
                    for key, t in sorted(terms.get(el["id"], {}).items()) if key not in {s.lower() for s in el.get("synonyms") or []} and key != el["label"].lower()]
        if proposed:
            out["proposed_synonyms"] = proposed
        return out

    out = {**model, **{kind: [with_usage(el) for el in model[kind]] for kind in ("concepts", "relations", "measures")}}
    meta = {"records": len(records), "answered": answered, "sources": dict(sorted(sources.items())), "matched": matched,
            "unmatched": [missed[k] for k in sorted(missed)]}
    return out, meta
