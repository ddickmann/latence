"""Link the records of two or more sources and write what a host needs to show the result.

Input: one parquet file in the estate layout of ``run_a.py`` (``unique_id``, ``source_system``,
``entity_type``, ``name``, ``city``, ``postcode``, ``country``, ``industry``, ``reg_id``,
``cluster``). ``cluster`` is the truth: it stands in for a reviewer's labels and is what the
realised figures are counted against. It is never evidence.

Output: one JSON file (``resolution/1``): the certified lines and how many labels they cost,
every accepted pair and every pair left for review with what it rests on, the cluster of
every record, and how right the machine was.

Usage: python link_records.py --records records.parquet --out resolution.json [--n-cal 400 --n-ltt 900]
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pairfeat import org_features
from run_a import LINK_PASSES, normalise_org, record_numbers, run


def evidence_of(f: dict[str, float], sources: tuple[str, str]) -> list[dict[str, str]]:
    """What a pair rests on, in words, from its evidence row."""
    if f["name_same"] == 1:
        name = "the same name"
    elif f["core_same"] == 1:
        name = "the same name apart from the legal form"
    elif f["name_jw"] >= 0.92:
        name = "nearly the same name"
    else:
        name = f"the names share {round(100 * f['token_share'])}% of their distinctive words"
    out = [{"kind": "name", "detail": name}]
    town = {1.0: "the same town", -1.0: "different towns", 0.0: "one record has no town"}[f["city"]]
    out.append({"kind": "town", "detail": town})
    if f["postcode"] != 0:
        out.append({"kind": "postcode", "detail": "the same postcode" if f["postcode"] == 1 else "different postcodes"})
    if f["legal"] == -1:
        out.append({"kind": "legal form", "detail": "different legal forms"})
    out.append({"kind": "source", "detail": f"both from {sources[0]}" if sources[0] == sources[1] else f"{sources[0]} and {sources[1]}"})
    return out


def link(records: Path, n_cal: int, n_ltt: int, seed: int = 7, max_review: int = 500, duplicate_free: tuple[str, ...] = ()) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "pairs.npz"
        res = run("estate", records, n_cal, n_ltt, seed, baseline_max=0, learned=True, dump=dump,
                  duplicate_free=duplicate_free, passes=LINK_PASSES, link_only=bool(duplicate_free))
        z = np.load(dump, allow_pickle=False)
        l_ids, r_ids, p, band = z["l"].astype(str), z["r"].astype(str), z["p"], z["band"].astype(str)
        ids, machine = z["ids"].astype(str), z["machine"].astype(str)
    raw = pd.read_parquet(records)
    rec = normalise_org(raw)
    rid, _ = record_numbers(rec.unique_id.to_numpy())
    rec = rec.iloc[np.argsort(rid, kind="stable")].reset_index(drop=True)  # row k is record k, in id order
    number = {u: k for k, u in enumerate(rec.unique_id.tolist())}
    source = dict(zip(rec.unique_id.tolist(), rec.source_system.tolist()))

    keep = np.where((band == "accept") | (band == "review"))[0]
    left = np.array([number[u] for u in l_ids[keep]], dtype=np.int64)
    right = np.array([number[u] for u in r_ids[keep]], dtype=np.int64)
    X, names = org_features(rec, left, right) if len(keep) else (np.zeros((0, 0)), [])
    pairs: dict[str, list[dict[str, Any]]] = {"accept": [], "review": []}
    for row, i in enumerate(keep.tolist()):
        a, b = str(l_ids[i]), str(r_ids[i])
        pairs[str(band[i])].append({"a": a, "b": b, "p": round(float(p[i]), 4),
                                    "evidence": evidence_of(dict(zip(names, X[row].tolist())), (source[a], source[b]))})
    review = sorted(pairs["review"], key=lambda d: (-d["p"], d["a"], d["b"]))
    b_, mo, learned = res["bands"], res["clustering"]["machine_only"], res["bands"]["learned"]
    return {
        "format": "resolution/1",
        "records": int(res["records"]),
        "sources": {str(k): int(v) for k, v in raw.source_system.value_counts().sort_index().items()},
        "candidate_pairs": int(res["blocking"].get("link_only", {}).get("pairs_between_sources", res["blocking"]["_union"]["pairs"])),
        "labels": {**learned["labels_used"], "total": int(sum(learned["labels_used"].values())), "from": "the truth column of the input"},
        "lines": {"accept_at": b_["accept_at"], "reject_below": b_["reject_below"], "target": b_["target"], "delta": b_["delta"]},
        "certified": b_["accept_at"] is not None,
        "duplicate_free": b_["duplicate_free"],
        "first_stage": res["first_stage"],
        "counts": b_["counts"],
        "realised": {"accept_precision": b_["auto_accept_precision_realised"], "accept_precision_lower_bound": b_["auto_accept_precision_wilson_lower"],
                     "pairwise_precision": mo["pairwise_precision"], "pairwise_recall": mo["pairwise_recall"],
                     "candidates_hold_share_of_true_pairs": res["blocking"]["_union"]["pair_completeness"]},
        "clusters": dict(sorted(zip(ids.tolist(), machine.tolist()))),
        "accepted": sorted(pairs["accept"], key=lambda d: (d["a"], d["b"])),
        "review": review[:max_review],
        "review_total": len(review),
        "evidence_names": names,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--n-ltt", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duplicate-free", action="append", default=[], help="a source that holds no duplicates (repeat for more)")
    a = ap.parse_args()
    out = link(a.records, a.n_cal, a.n_ltt, a.seed, duplicate_free=tuple(a.duplicate_free))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("records", "candidate_pairs", "labels", "lines", "certified", "duplicate_free", "counts", "realised", "review_total")}, indent=1, default=float))


if __name__ == "__main__":
    main()
