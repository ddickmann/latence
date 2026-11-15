"""A small synthetic estate's export (``graph-export/4``), as a build writes it, for the tests.

A dealer master with names a careless writer would let a reader take for something else, a
value set of makers in which three pairs of names may be the same (the first name of each pair
carries more rows), a value set of markets that a seeded second source shares, a seeded list of
portal accounts, a relation, a measure, and the machine's links between dealers and accounts,
certified on simulated labels. Only the fields the writer reads are filled in.
"""

from __future__ import annotations

from typing import Any

ODD_NAMES = {7: 'Müller & Söhne: "Autohaus" #7', 8: "0123", 9: "Yes"}
DEALERS = {f"D{k}": ODD_NAMES.get(k, f"Dealer {k} GmbH") for k in range(1, 31)}
ACCOUNTS = {f"PA{k}": f"Portal account {k}" for k in range(1, 6)}
# (the pair's two names, the rows each carries)
MAKER_PAIRS = {
    "Alfa": (("Alfa Works, Inc.", 133), ("Alfa Works North", 44)),
    "Beta": (("Beta Motor Corp.", 300), ("Beta Motor of America", 66)),
    "Gama": (("Gama (GMA US LLC)", 120), ("Gama (GMA US, LLC)", 50)),
}


def claim(n: int) -> str:
    """A claim: what an element states, as the build hashes it (16 hex digits)."""
    return f"{n:016x}"


def _pair(column: str, a: tuple[str, int], b: tuple[str, int], n: int) -> dict[str, Any]:
    return {"id": f"same:{column}:{a[0]}~{b[0]}", "kind": "possibly_same",
            "concept": f"concept:value:{column}", "a": f"value:{column}:{a[0]}",
            "b": f"value:{column}:{b[0]}", "rows_a": a[1], "rows_b": b[1], "claim": claim(n)}


def _holds(owner: str, column: str, values: int, n: int) -> dict[str, Any]:
    return {"id": f"relation:{owner}~{column}", "from": f"concept:{owner}",
            "to": f"concept:value:{column}", "columns": [column], "status": "proposed",
            "claim": claim(n),
            "signals": [{"kind": "data", "code": "shared_column.holds",
                         "params": {"values": values}, "ref": ""}]}


def export() -> dict[str, Any]:
    """The export of the estate before anyone decided anything."""
    concepts = [
        {"id": "concept:shop.dealers", "kind": "entity", "object_name": "shop.dealers",
         "key": ["dealer_id"], "claim": claim(1), "status": "proposed"},
        {"id": "concept:partner.accounts", "kind": "entity", "object_name": "partner.accounts",
         "key": ["account_id"], "seeded": True, "claim": claim(2), "status": "proposed"},
        {"id": "concept:shop.recalls", "kind": "event", "object_name": "shop.recalls",
         "key": ["recall_ref"], "claim": claim(3), "status": "proposed"},
        {"id": "concept:value:maker", "kind": "reference", "column": "maker",
         "objects": ["shop.bulletins", "shop.claims", "shop.recalls"], "claim": claim(4),
         "status": "proposed"},
        {"id": "concept:value:market", "kind": "reference", "column": "market",
         "objects": ["partner.accounts", "shop.claims", "shop.dealers"], "claim": claim(5),
         "status": "proposed"},
    ]
    relations = [
        _holds("shop.recalls", "maker", 7, 10), _holds("shop.bulletins", "maker", 6, 11),
        _holds("shop.claims", "maker", 3, 12), _holds("shop.dealers", "market", 2, 13),
        {"id": "relation:shop.claims~shop.dealers", "from": "concept:shop.claims",
         "to": "concept:shop.dealers", "columns": ["dealer_id"], "status": "proposed",
         "claim": claim(14), "signals": []},
    ]
    nodes: list[dict[str, Any]] = [
        {"id": f"shop.dealers:{key}", "concept": "concept:shop.dealers", "key": key, "label": name}
        for key, name in DEALERS.items()]
    nodes += [{"id": f"partner.accounts:{key}", "concept": "concept:partner.accounts", "key": key,
               "label": name} for key, name in ACCOUNTS.items()]
    review: list[dict[str, Any]] = []
    for n, (a, b) in enumerate(MAKER_PAIRS.values(), start=20):
        for name, rows in (a, b):
            nodes.append({"id": f"value:maker:{name}", "concept": "concept:value:maker",
                          "label": name, "rows": rows})
        review.append(_pair("maker", a, b, n))
    for name, rows in (("North Region", 400), ("North Area", 30), ("South", 200)):
        nodes.append({"id": f"value:market:{name}", "concept": "concept:value:market",
                      "label": name, "rows": rows})
    review.append(_pair("market", ("North Region", 400), ("North Area", 30), 30))
    review.append({"id": "same:shop.dealers:D3~partner.accounts:PA4", "kind": "possibly_same",
                   "a": "shop.dealers:D3", "b": "partner.accounts:PA4", "p": 0.55,
                   "claim": claim(31)})
    review.append({"id": "same:shop.dealers:D1~shop.dealers:D2", "kind": "possibly_same",
                   "a": "shop.dealers:D1", "b": "shop.dealers:D2", "p": 0.61, "claim": claim(32)})
    links = [{"kind": "same_as", "from": f"shop.dealers:D{d}", "to": f"partner.accounts:PA{a}",
              "p": 0.9, "certified_on": "simulated"} for d, a in ((1, 1), (1, 2), (2, 3))]
    return {
        "format": "graph-export/4", "concepts": concepts, "relations": relations,
        "measures": [{"id": "measure:shop.recalls:count", "concept": "concept:shop.recalls",
                      "expr": "count(*)", "claim": claim(40)}],
        "structure": {"roll_ups": [{"id": "roll_up:market", "claim": claim(41)}]},
        "classes": {"classes": [], "subclass_of": [{"id": "subclass:dealer~organisation",
                                                    "claim": claim(42)}]},
        "nodes": nodes, "edges": links, "review": review, "label_requests": {},
        "meta": {"resolution": {"certified_on": "simulated", "seeded": True}},
    }


def element(doc: dict[str, Any], element_id: str) -> dict[str, Any]:
    """An element of the export by its id, wherever it is listed."""
    lists = [doc["concepts"], doc["relations"], doc["measures"], doc["review"],
             *doc["structure"].values(), *doc["classes"].values()]
    found: dict[str, Any] = next(el for part in lists for el in part if el["id"] == element_id)
    return found


def pair(doc: dict[str, Any], word: str) -> dict[str, Any]:
    """The review pair whose id holds ``word``."""
    found: dict[str, Any] = next(r for r in doc["review"] if word in r["id"])
    return found


def line(el: dict[str, Any], verdict: str = "approved", by: str = "reviewer-1",
         at: str | None = "2026-11-12", stated: str | None = "") -> dict[str, Any]:
    """A decision on an element, as a person's decisions file holds it; ``stated`` is the claim
    the decision was made on (the element's own unless given, none when ``None``)."""
    said = el["claim"] if stated == "" else stated
    return {"subject": el["id"], "verdict": verdict, "by": by,
            **({"claim": said} if said else {}), **({"at": at} if at else {})}


def in_force(*lines: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The decisions in force: the last line on each element counts."""
    return {d["subject"]: d for d in lines}
