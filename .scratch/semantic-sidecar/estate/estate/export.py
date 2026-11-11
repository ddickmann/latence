"""The export a screen reads: ``graph-export/3``.

The schema level (concepts with their signals and attributes, relations with their support on
the data, measures bound to concepts) and, when instances are built, the instance level: nodes
for values and master rows, edges that refer or count, records of a second source linked to
their master record (``same_as``), and review items. No value of a sensitive column is
written, and no row of a personal object.

Two rules of the contract (version 3):

- the engine writes no sentence. What it says about an element is a code with parameters
  (``codes.py``); text that came from the catalog or from a model is carried as text, with its
  language and where it came from;
- every element a person can decide carries ``claim``, a digest of what it states. A decision
  is made on a claim, so it cannot carry over to a proposal that now says something else.
"""

from __future__ import annotations

from typing import Any

from .codes import text
from .ledger import claim_of

SCHEMA = "graph-export/3"


def _claimed(kind: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**el, "claim": claim_of(kind, el)} for el in items]


def build_export(layer: dict[str, Any], *, source: str, title: str, verified: list[tuple[str, str]] | None = None,
                 instances: dict[str, Any] | None = None, structure: dict[str, Any] | None = None,
                 classes: dict[str, Any] | None = None, language: str = "en") -> dict[str, Any]:
    """``language`` is the language the catalog's comments are written in."""
    concepts = [{**c, "claim": claim_of("concept", c), "definition": text(c.get("definition"), "catalog", language),
                 **({"attributes": [{**a, "definition": text(a.get("definition"), "catalog", language)} for a in c["attributes"]]} if "attributes" in c else {})}
                for c in layer["concepts"]]
    relations = _claimed("relation", layer["relations"])
    measures = [{**m, "claim": claim_of("measure", m), "definition": text(m.get("definition"), "catalog", language)} for m in layer["measures"]]
    structure = structure or {}
    classes = classes or {"top": [], "classes": [], "subclass_of": [], "voices": {"attribute_pattern": False, "language_model": None}, "unplaced": []}
    found = {(r["from"].removeprefix("concept:"), r["to"].removeprefix("concept:")) for r in relations}
    names = {c["id"].removeprefix("concept:"): c for c in concepts}
    compared = []
    for child, parent in verified or []:
        if (child, parent) in found:
            state = "found"
        elif names.get(parent, {}).get("kind") == "derived" or names.get(child, {}).get("kind") == "derived":
            state = "left_out_derived"
        elif names.get(parent, {}).get("instances") == "withheld_personal" and not names.get(parent, {}).get("key"):
            state = "not_checked_personal_key"
        else:
            state = "not_found"
        compared.append({"from": child, "to": parent, "state": state})
    return {
        "schema": SCHEMA,
        "meta": {
            "source": source, "title": title, "schema_origin": "induced", "synthetic": True,
            "objects": layer["meta"]["objects"], "metric_views": layer["meta"]["metric_views"],
            "support_floor": layer["meta"]["support_floor"],
            "counts": {
                "concepts": len(concepts), "relations": len(relations), "measures": len(measures),
                "declared": sum(e["status"] == "declared" for e in concepts + relations),
                "proposed": sum(e["status"] == "proposed" for e in concepts + relations),
                "dropped_below_floor": len(layer["meta"]["dropped_below_floor"]),
            },
            "usage": layer["meta"].get("usage"),
            "decisions": layer["meta"].get("decisions", {"approved": 0, "rejected": 0, "unknown_subjects": []}),
            "host_verified_joins": compared,
            "instances": "not_in_this_cut" if instances is None else "available",
            "instance_counts": None if instances is None else {
                "nodes": instances["meta"]["nodes"], "edges": instances["meta"]["edges"],
                "review_items": instances["meta"]["review_items"], "certified": instances["meta"]["certified"],
                "nodes_per_concept": instances["meta"]["nodes_per_concept"]},
            "resolution": None if instances is None else instances["meta"].get("resolution"),
        },
        "concepts": concepts,
        "relations": relations,
        "measures": measures,
        "dropped": layer["meta"]["dropped_below_floor"],
        "structure": {"roll_ups": _claimed("roll_up", structure.get("roll_ups", [])), "equivalent": _claimed("equivalent", structure.get("equivalent", [])),
                      "part_of": _claimed("part_of", structure.get("part_of", [])), "describes": _claimed("describes", structure.get("describes", [])),
                      "not_proposed": structure.get("not_proposed", [])},
        "classes": {**classes, "classes": _claimed("class", classes["classes"]), "subclass_of": _claimed("subclass", classes["subclass_of"])},
        "nodes": [] if instances is None else instances["nodes"],
        "edges": [] if instances is None else instances["edges"],
        "review": [] if instances is None else _claimed("possibly_same", instances["review"]),
        "label_requests": {name: _claimed("possibly_same", sample) for name, sample in ((instances or {}).get("label_requests") or {}).items()},
    }
