"""Export the resolved estate as a small graph the workbench can draw (ticket 22).

Two levels, as in the domain model (ADR-0069):

- the schema level: the Concepts and the relations between them, each with a count;
- the instance level: resolved organisations (one per machine cluster) with the records that
  support them, the values those records give and how many records give each, the reference
  values they point to (city, country, industry), and the organisations the review band says
  they might be the same as.

The schema here is declared for the synthetic estate: it is read off the estate's columns, not
induced. Schema induction (ADR-0073) is designed and not built. Synthetic data only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = "graph-export/1"
RECORD_FIELDS = ["unique_id", "source_system", "name", "city", "postcode", "country", "industry", "reg_id"]
VALUE_FIELDS = ["name", "city", "postcode", "country", "industry", "reg_id"]
CONCEPTS = [
    ("organisation", "Organisation", "entity",
     "A legal entity the estate does business with, resolved from the records that describe it."),
    ("city", "City", "reference", "The place an organisation's records give as its location."),
    ("country", "Country", "reference", "The country a city lies in."),
    ("industry", "Industry", "reference", "The line of business the records assign to an organisation."),
    ("source_system", "Source system", "provenance", "A system of record that holds records about organisations."),
]


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v) != ""


def _display_name(names: list[str]) -> str:
    """The most frequent spelling, ignoring case. Ties go to a spelling in mixed case (a record
    typed in capitals or lower case says less about how the name is written), then to the longer."""
    if not names:
        return "(unnamed)"
    by_key: dict[str, list[str]] = defaultdict(list)
    for n in names:
        by_key[n.casefold()].append(n)

    def flat(n: str) -> bool:
        return n.isupper() or n.islower()

    key = sorted(by_key, key=lambda k: (-len(by_key[k]), all(flat(n) for n in by_key[k]), -len(k), k))[0]
    return sorted(by_key[key], key=lambda n: (flat(n), n))[0]


def _counted(values: list[Any]) -> list[dict[str, Any]]:
    c = Counter(str(v) for v in values if _present(v))
    return [{"value": v, "records": n} for v, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]


def build_graph_export(
    *,
    raw: pd.DataFrame,
    clusters: dict[str, str],
    cand: pd.DataFrame,
    p: np.ndarray,
    band: np.ndarray,
    focus_records: set[str],
    review_item_ids: set[str],
    industry_names: dict[str, str],
    max_entities: int,
    source: str,
) -> dict[str, Any]:
    fields = [f for f in RECORD_FIELDS if f in raw.columns]
    rows = {r["unique_id"]: {f: (r.get(f) if _present(r.get(f)) else None) for f in fields} for r in raw.to_dict("records")}
    members: dict[str, list[str]] = defaultdict(list)
    for rid, c in clusters.items():
        members[c].append(rid)

    def majority(c: str, field: str) -> str | None:
        vals = _counted([rows[r].get(field) for r in members[c]])
        return vals[0]["value"] if vals else None

    # Whole-estate counts for the schema level and the reference nodes.
    city_of = {c: majority(c, "city") for c in members}
    industry_of = {c: majority(c, "industry") for c in members}
    country_of = {c: majority(c, "country") for c in members}
    city_country: dict[str, Counter[str]] = defaultdict(Counter)
    for c in members:
        if city_of[c] and country_of[c]:
            city_country[city_of[c]][country_of[c]] += 1  # type: ignore[index]
    country_of_city = {city: cc.most_common(1)[0][0] for city, cc in city_country.items()}
    n_city = Counter(v for v in city_of.values() if v)
    n_industry = Counter(v for v in industry_of.values() if v)
    n_country = Counter(v for v in country_of.values() if v)
    n_system = Counter(str(r["source_system"]) for r in rows.values() if r.get("source_system"))

    # Which organisations to carry in full: the ones the review queue touches first, then the
    # ones with the most records, so the sample has something to explore around every review item.
    focus = {clusters[r] for r in focus_records if r in clusters}
    rest = sorted((c for c in members if c not in focus), key=lambda c: (-len(members[c]), c))
    chosen = sorted(focus) + rest[: max(0, max_entities - len(focus))]
    chosen_set = set(chosen)

    # Review-band pairs between two different organisations: "might be the same as".
    maybe: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    ls, rs = cand.l.to_numpy(), cand.r.to_numpy()
    for i in np.where(band == "review")[0]:
        a, b = clusters[ls[i]], clusters[rs[i]]
        if a == b or a not in chosen_set or b not in chosen_set:
            continue
        item = f"{ls[i]}~{rs[i]}"
        for x, y in ((a, b), (b, a)):
            e = maybe[x].setdefault(y, {"entity": f"org:{y}", "pairs": 0, "probability": 0.0, "items": []})
            e["pairs"] += 1
            e["probability"] = max(e["probability"], round(float(p[i]), 4))
            if item in review_item_ids:
                e["items"].append(item)

    entities = []
    for c in chosen:
        recs = sorted(members[c])
        entities.append({
            "id": f"org:{c}",
            "label": _display_name([str(rows[r]["name"]) for r in recs if rows[r].get("name")]),
            "records": [rows[r] for r in recs],
            "values": {f: _counted([rows[r].get(f) for r in recs]) for f in VALUE_FIELDS if f in fields},
            "city": city_of[c],
            "country": country_of[c],
            "industry": industry_of[c],
            "systems": sorted({str(rows[r]["source_system"]) for r in recs if rows[r].get("source_system")}),
            "possibly_same": sorted(maybe[c].values(), key=lambda e: (-e["probability"], e["entity"])),
        })

    def in_sample(field_of: dict[str, str | None], value: str) -> int:
        return sum(1 for c in chosen if field_of[c] == value)

    nodes = (
        [{"id": f"city:{v}", "type": "city", "label": v, "organisations": n, "in_sample": in_sample(city_of, v),
          "country": country_of_city.get(v)} for v, n in sorted(n_city.items())]
        + [{"id": f"country:{v}", "type": "country", "label": v, "organisations": n, "in_sample": in_sample(country_of, v),
            "cities": sorted(c for c, k in country_of_city.items() if k == v)} for v, n in sorted(n_country.items())]
        + [{"id": f"industry:{v}", "type": "industry", "label": industry_names.get(v, v), "code": v, "organisations": n,
            "in_sample": in_sample(industry_of, v)} for v, n in sorted(n_industry.items())]
        + [{"id": f"system:{v}", "type": "source_system", "label": v, "records": n,
            "in_sample": sum(1 for c in chosen if any(rows[r].get("source_system") == v for r in members[c]))}
           for v, n in sorted(n_system.items())]
    )
    counts = {"organisation": len(members), "city": len(n_city), "country": len(n_country),
              "industry": len(n_industry), "source_system": len(n_system)}
    total = len(members)
    return {
        "schema": SCHEMA,
        "meta": {
            "source": source,
            "synthetic": True,
            "schema_origin": "declared",
            "records": int(len(raw)),
            "organisations": total,
            "organisations_in_sample": len(entities),
            "multi_record_organisations": sum(1 for m in members.values() if len(m) > 1),
        },
        "concepts": [
            {"id": cid, "label": label, "kind": kind, "count": counts[cid], "definition": definition}
            for cid, label, kind, definition in CONCEPTS
        ],
        "relations": [
            {"id": "located_in", "from": "organisation", "to": "city", "label": "located in",
             "count": sum(1 for v in city_of.values() if v)},
            {"id": "operates_in", "from": "organisation", "to": "industry", "label": "operates in",
             "count": sum(1 for v in industry_of.values() if v)},
            {"id": "recorded_in", "from": "organisation", "to": "source_system", "label": "recorded in",
             "count": int(len(raw))},
            {"id": "in_country", "from": "city", "to": "country", "label": "lies in", "count": len(country_of_city)},
        ],
        "attributes": [
            {"concept": "organisation", "id": f, "label": label,
             "filled": round(sum(1 for c in members if majority(c, f)) / total, 4) if total else 0.0}
            for f, label in (("name", "Name"), ("postcode", "Postcode"), ("reg_id", "Registration id"))
        ],
        "nodes": nodes,
        "entities": entities,
    }
