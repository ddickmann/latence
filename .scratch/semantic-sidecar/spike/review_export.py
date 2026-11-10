"""Export the review band as items the expert workbench can show (tickets 20 and 22).

One item is one candidate pair the machine could not decide, with what a reviewer needs to
decide it: both records as the sources hold them, the matcher's evidence field by field (the
level each comparison fell into and the match weight it contributed), the calibrated probability,
and why the pair is in the review band. Items are grouped: a group is a connected set of review
pairs, which the workbench shows as one cluster so that one decision can settle several pairs.

Synthetic data only. ``gold_same`` is the generator's truth, kept so the prototype can show a
simulated audit; a real export has no such field.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from er import plain_closure

SCHEMA = "review-export/1"
RECORD_FIELDS = {
    "org": ["unique_id", "source_system", "name", "city", "postcode", "country", "industry", "reg_id"],
    "person": ["unique_id", "first_name", "surname", "dob", "birth_place", "postcode_fake", "occupation", "gender"],
}
# Which raw field a normalised comparison column was derived from, for the reviewer's eyes.
FIELD_OF = {
    "name_core": "name", "legal_canon": "name", "num_tokens": "name", "city_n": "city",
    "pc_out": "postcode", "reg_norm": "reg_id", "industry": "industry",
}
LABEL_OF = {
    "name_core": "Name", "legal_canon": "Legal form", "num_tokens": "Number in name", "city_n": "City",
    "pc_out": "Postcode area", "reg_norm": "Registration id", "industry": "Industry",
}


def _level_labels(model: dict[str, Any]) -> dict[str, dict[int, str]]:
    """comparison column -> {gamma value -> the level's label}. Gamma is -1 for the null level
    and counts the other levels from the weakest (0) up to the strongest."""
    out: dict[str, dict[int, str]] = {}
    for comp in model["comparisons"]:
        levels = comp["comparison_levels"]
        non_null = [lv for lv in levels if not lv.get("is_null_level")]
        labels = {len(non_null) - 1 - i: str(lv["label_for_charts"]) for i, lv in enumerate(non_null)}
        labels[-1] = "missing on at least one side"
        out[str(comp["output_column_name"])] = labels
    return out


def _clean(v: Any) -> Any:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _why_unsure(evidence: list[dict[str, Any]], p: float) -> str:
    """One sentence a reviewer can check against the evidence rows: what pulls each way."""
    pro = sorted((e for e in evidence if e["weight"] > 0.5), key=lambda e: -e["weight"])
    con = sorted((e for e in evidence if e["weight"] < -0.5), key=lambda e: e["weight"])
    gone = [e for e in evidence if e["level"] == -1]
    parts = []
    if pro:
        parts.append("for a match: " + ", ".join(e["label"].lower() for e in pro[:3]))
    if con:
        parts.append("against: " + ", ".join(e["label"].lower() for e in con[:3]))
    if gone:
        parts.append("no evidence from: " + ", ".join(e["label"].lower() for e in gone[:3]))
    return f"Calibrated probability {p:.2f}. " + "; ".join(parts) + "."


def build_review_export(
    *,
    raw: pd.DataFrame,
    cand: pd.DataFrame,
    p: np.ndarray,
    p_raw: np.ndarray,
    band: np.ndarray,
    y: np.ndarray,
    veto: np.ndarray,
    detail: dict[str, Any],
    accept_at: float | None,
    reject_below: float | None,
    kind: str,
    max_groups: int,
    seed: int,
    source: str,
    audit_n: int = 60,
) -> dict[str, Any]:
    fields = [f for f in RECORD_FIELDS[kind] if f in raw.columns]
    rec = {r["unique_id"]: {f: _clean(r.get(f)) for f in fields} for r in raw.to_dict("records")}
    labels = _level_labels(detail["model"])
    comps = list(labels)
    pred = detail["pred"].set_index(["unique_id_l", "unique_id_r"])
    prior = float(detail["model"]["probability_two_random_records_match"])
    prior_weight = float(np.log2(prior / (1 - prior)))

    def evidence(left: str, right: str) -> list[dict[str, Any]]:
        if (left, right) not in pred.index:
            return []
        row = pred.loc[(left, right)]
        out = []
        for c in comps:
            gamma = int(row[f"gamma_{c}"])
            weight = float(row[f"mw_{c}"]) + float(row.get(f"mw_tf_adj_{c}", 0.0) or 0.0)
            out.append({
                "comparison": c,
                "label": LABEL_OF.get(c, c),
                "field": FIELD_OF.get(c, c),
                "level": gamma,
                "level_label": labels[c].get(gamma, str(gamma)),
                "weight": round(weight, 3),
                "left": _clean(row.get(f"{c}_l")),
                "right": _clean(row.get(f"{c}_r")),
            })
        return out

    ls, rs = cand.l.to_numpy(), cand.r.to_numpy()

    def item(i: int, group: str | None) -> dict[str, Any]:
        ev = evidence(ls[i], rs[i])
        return {
            "id": f"{ls[i]}~{rs[i]}",
            "group": group,
            "band": str(band[i]),
            "probability": round(float(p[i]), 4),
            "match_weight": round(prior_weight + sum(e["weight"] for e in ev), 3),
            "prior_weight": round(prior_weight, 3),
            "left": rec[ls[i]],
            "right": rec[rs[i]],
            "evidence": ev,
            "why_unsure": _why_unsure(ev, float(p[i])),
            "vetoed": bool(veto[i]),
            "gold_same": bool(y[i]),
        }

    rev = np.where(band == "review")[0]
    group_of = plain_closure(set(ls[rev]) | set(rs[rev]), zip(ls[rev], rs[rev])) if len(rev) else {}
    by_group: dict[str, list[int]] = defaultdict(list)
    for i in rev:
        by_group[group_of[ls[i]]].append(int(i))

    # Priority: how many records one decision touches, times how undecided the machine is.
    def priority(idx: list[int]) -> float:
        members = {ls[i] for i in idx} | {rs[i] for i in idx}
        uncertainty = float(np.mean([1 - abs(2 * p[i] - 1) for i in idx]))
        return len(members) * uncertainty

    ranked = sorted(by_group.items(), key=lambda kv: (-priority(kv[1]), kv[0]))[:max_groups]
    in_export = {m for _g, idx in ranked for i in idx for m in (ls[i], rs[i])}
    # Edges the workbench draws for a group: the review pairs, plus what the machine already
    # accepted or rejected between the same records (the context a split or merge needs).
    context = [
        int(i) for i in np.where(band != "review")[0] if ls[i] in in_export and rs[i] in in_export
    ]
    groups = []
    for rank, (g, idx) in enumerate(ranked):
        members = sorted({ls[i] for i in idx} | {rs[i] for i in idx})
        mset = set(members)
        edges = [
            {"left": ls[i], "right": rs[i], "band": str(band[i]), "probability": round(float(p[i]), 4)}
            for i in idx + [c for c in context if ls[c] in mset and rs[c] in mset]
        ]
        groups.append({
            "id": g, "rank": rank, "priority": round(priority(idx), 3), "records": members,
            "labels": {m: rec[m].get("name") or " ".join(str(rec[m].get(k) or "") for k in ("first_name", "surname")).strip() for m in members},
            "edges": edges, "items": [f"{ls[i]}~{rs[i]}" for i in idx],
        })
    items = [item(i, g) for g, idx in ranked for i in idx]

    rng = np.random.default_rng(seed)
    acc = np.where(band == "accept")[0]
    sample = rng.choice(acc, size=min(audit_n, len(acc)), replace=False) if len(acc) else []
    audit = [item(int(i), None) for i in sorted(int(x) for x in sample)]

    return {
        "schema": SCHEMA,
        "meta": {
            "source": source,
            "kind": kind,
            "synthetic": True,
            "records": int(len(raw)),
            "candidate_pairs": int(len(cand)),
            "bands": {"accept_at": accept_at, "reject_below": reject_below},
            "counts": {b: int((band == b).sum()) for b in ("accept", "review", "reject")},
            "review_groups_total": len(by_group),
            "review_groups_exported": len(groups),
            "record_fields": fields,
            "comparisons": [{"comparison": c, "label": LABEL_OF.get(c, c), "field": FIELD_OF.get(c, c)} for c in comps],
        },
        "groups": groups,
        "items": items,
        "audit": audit,
    }
