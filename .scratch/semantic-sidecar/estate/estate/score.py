"""How right the proposals are, against a list a person checked.

The list (``estates/<name>.gold.yaml``) says, for one estate: the top class each concept is a
kind of (or ``none``), the hierarchy steps that are right, and what is a part of what or links
what. Entries under ``unsure`` are judged by nobody and count on neither side.

A list nobody corrected is the build's own opinion read back to it. The score says who checked
the list (``checked_by``); without a name the numbers are printed and marked as not counting.
"""

from __future__ import annotations

from typing import Any

TOPS = ("organisation", "person", "product", "event", "place", "document")


def _short(cid: str) -> str:
    return cid.removeprefix("concept:")


def _rate(right: int, of: int) -> float | None:
    return round(right / of, 4) if of else None


def _pairs(proposed: set[tuple[str, ...]], gold: set[tuple[str, ...]], unsure: set[tuple[str, ...]]) -> dict[str, Any]:
    proposed, gold = proposed - unsure, gold - unsure
    right = proposed & gold
    return {"proposed": len(proposed), "in_the_list": len(gold), "right": len(right),
            "precision": _rate(len(right), len(proposed)), "recall": _rate(len(right), len(gold)),
            "wrong": sorted(" -> ".join(p) for p in proposed - gold), "missed": sorted(" -> ".join(p) for p in gold - proposed)}


def score(export: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Precision and recall per kind, with every wrong proposal and every miss by name.
    ``precision`` is ``None`` where nothing was proposed and ``recall`` where the list holds
    nothing: neither is a zero."""
    known = {_short(c["id"]) for c in export["concepts"]}
    unsure = gold.get("unsure") or {}
    named = set(gold.get("classes") or {}) | {x for key in ("steps", "part_of", "links") for pair in (gold.get(key) or []) + (unsure.get(key) or []) for x in pair}
    unknown = sorted(named - known)
    if unknown:
        raise ValueError(f"the list names concepts the export does not hold: {unknown}")
    bad = sorted({str(v) for v in (gold.get("classes") or {}).values()} - set(TOPS) - {"none"})
    if bad:
        raise ValueError(f"the list names classes that are not top classes: {bad}")

    classes = export.get("classes") or {}
    parent = {c["id"]: c["parent"] for c in classes.get("classes", [])}

    def top_of(cid: str) -> str | None:
        seen: set[str] = set()
        while cid in parent and cid not in seen:
            seen.add(cid)
            cid = parent[cid]
        name = cid.removeprefix("class:")
        return name if name in TOPS else None

    placed = {_short(s["concept"]): top_of(s["parent"]) for s in classes.get("subclass_of", [])}
    skip = set(unsure.get("classes") or [])
    judged = {c: str(v) for c, v in (gold.get("classes") or {}).items() if c not in skip}
    proposed = {c: t for c, t in placed.items() if c in judged}
    right = {c for c, t in proposed.items() if t == judged[c]}
    wanted = {c for c, v in judged.items() if v != "none"}
    out: dict[str, Any] = {
        "checked_by": gold.get("checked_by") or None,
        "counts_as_measured": bool(gold.get("checked_by")),
        "classes": {
            "proposed": len(proposed), "in_the_list": len(wanted), "right": len(right),
            "precision": _rate(len(right), len(proposed)), "recall": _rate(len(right & wanted), len(wanted)),
            "wrong": sorted(f"{c}: proposed {proposed[c]}, the list says {judged[c]}" for c in set(proposed) - right),
            "missed": sorted(f"{c}: the list says {judged[c]}, not placed" for c in wanted - set(proposed)),
            "not_in_the_list": sorted(set(placed) - set(judged) - skip),
        },
    }
    st = export.get("structure") or {}

    def as_pairs(items: list[Any]) -> set[tuple[str, ...]]:
        return {(str(a), str(b)) for a, b in items}

    steps = {(_short(r["from"]), _short(r["to"])) for r in st.get("roll_ups", [])}
    out["steps"] = _pairs(steps, as_pairs(gold.get("steps") or []), as_pairs(unsure.get("steps") or []))
    for kind in ("part_of", "links"):
        found = {(_short(p["from"]), _short(p["to"])) for p in st.get("part_of", []) if p["kind"] == kind}
        out[kind] = _pairs(found, as_pairs(gold.get(kind) or []), as_pairs(unsure.get(kind) or []))
    return out
