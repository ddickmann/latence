"""The export a screen reads: ``graph-export/2``.

The schema level (concepts with their signals and attributes, relations with their support on
the data, measures bound to concepts) and, when instances are built, the instance level: nodes
for values and master rows, edges that refer or count, records of a second source linked to
their master record (``same_as``), and review items. No value of a sensitive column is
written, and no row of a personal object.
"""

from __future__ import annotations

from typing import Any

SCHEMA = "graph-export/2"


def build_export(layer: dict[str, Any], *, source: str, title: str, verified: list[tuple[str, str]] | None = None,
                 instances: dict[str, Any] | None = None) -> dict[str, Any]:
    concepts, relations = layer["concepts"], layer["relations"]
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
                "concepts": len(concepts), "relations": len(relations), "measures": len(layer["measures"]),
                "declared": sum(e["status"] == "declared" for e in concepts + relations),
                "proposed": sum(e["status"] == "proposed" for e in concepts + relations),
                "dropped_below_floor": len(layer["meta"]["dropped_below_floor"]),
            },
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
        "measures": layer["measures"],
        "dropped": layer["meta"]["dropped_below_floor"],
        "nodes": [] if instances is None else instances["nodes"],
        "edges": [] if instances is None else instances["edges"],
        "review": [] if instances is None else instances["review"],
    }
