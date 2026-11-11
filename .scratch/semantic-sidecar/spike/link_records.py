"""Link the records of two or more sources and write what a host needs to show the result.

Input: one parquet file in the estate layout of ``run_a.py`` (``unique_id``, ``source_system``,
``entity_type``, ``name``, ``city``, ``postcode``, ``country``, ``industry``, ``reg_id``,
``cluster``). ``cluster`` is the truth: it stands in for a reviewer's labels and is what the
realised figures are counted against. It is never evidence.

Output: one JSON file (``resolution/3``): the certified lines and how many labels they cost,
every accepted pair and every pair left for review with what it rests on, the cluster of
every record, and how right the machine was.

A reviewer's answers come in with ``--labels``; with ``--no-truth`` the truth column does
not stand in for the ones that are missing, and the output says which pairs are asked for.

Usage: python link_records.py --records records.parquet --out resolution.json [--n-cal 400 --n-ltt 900] [--labels answers.json] [--no-truth]
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


def _say(code: str, **params: object) -> dict[str, object]:
    return {"kind": code.split(".", 1)[0], "code": code, "params": params, "ref": ""}


def evidence_of(f: dict[str, float], sources: tuple[str, str]) -> list[dict[str, object]]:
    """What a pair rests on, from its evidence row, as codes a host renders in its own words
    (the table of codes is the estate build's ``codes.py``, which checks every one it reads)."""
    if f["name_same"] == 1:
        name = _say("name.same")
    elif f["core_same"] == 1:
        name = _say("name.same_core")
    elif f["name_jw"] >= 0.92:
        name = _say("name.near")
    else:
        name = _say("name.token_share", percent=round(100 * f["token_share"]))
    out = [name, _say({1.0: "town.same", -1.0: "town.different", 0.0: "town.missing"}[f["city"]])]
    if f["postcode"] != 0:
        out.append(_say("postcode.same" if f["postcode"] == 1 else "postcode.different"))
    if f["legal"] == -1:
        out.append(_say("legal_form.different"))
    out.append(_say("source.same", source=sources[0]) if sources[0] == sources[1] else _say("source.two", a=sources[0], b=sources[1]))
    return out


SAMPLES = {1: "training", 2: "accept_test", 3: "reject_test"}
BY = {0: None, 1: "reviewer", 2: "simulated"}


def read_labels(path: Path) -> dict[tuple[str, str], int]:
    """A reviewer's answers: a JSON list of ``{"a": id, "b": id, "same": true|false}``."""
    return {(str(d["a"]), str(d["b"])): int(bool(d["same"])) for d in json.loads(path.read_text(encoding="utf-8"))}


def link(records: Path, n_cal: int, n_ltt: int, seed: int = 7, max_review: int = 500, duplicate_free: tuple[str, ...] = (),
         labels: dict[tuple[str, str], int] | None = None, truth_stands_in: bool = True) -> dict[str, Any]:
    """``labels`` are a reviewer's answers; they win over the truth column. Where the truth may
    not stand in (``truth_stands_in=False``), the answer says what is still asked for
    (``state``): first the pairs to learn from, then, once those are answered, the two test
    samples. A line is certified only on test samples that are answered in full."""
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "pairs.npz"
        res = run("estate", records, n_cal, n_ltt, seed, baseline_max=0, learned=True, dump=dump,
                  duplicate_free=duplicate_free, passes=LINK_PASSES, link_only=bool(duplicate_free),
                  labels=labels, truth_stands_in=truth_stands_in)
        z = np.load(dump, allow_pickle=False)
        l_ids, r_ids, p, band = z["l"].astype(str), z["r"].astype(str), z["p"], z["band"].astype(str)
        ids, machine = z["ids"].astype(str), z["machine"].astype(str)
        sample, label, label_from = z["sample"], z["label"], z["label_from"]
    raw = pd.read_parquet(records)
    rec = normalise_org(raw)
    rid, _ = record_numbers(rec.unique_id.to_numpy())
    rec = rec.iloc[np.argsort(rid, kind="stable")].reset_index(drop=True)  # row k is record k, in id order
    number = {u: k for k, u in enumerate(rec.unique_id.tolist())}
    source = dict(zip(rec.unique_id.tolist(), rec.source_system.tolist()))
    master = set(duplicate_free)

    # What each pair rests on is worked out for the pairs that are shown: accepted, left for
    # review, or asked of a reviewer.
    keep = np.where((band == "accept") | (band == "review") | (sample > 0))[0]
    left = np.array([number[u] for u in l_ids[keep]], dtype=np.int64)
    right = np.array([number[u] for u in r_ids[keep]], dtype=np.int64)
    X, names = org_features(rec, left, right) if len(keep) else (np.zeros((0, 0)), [])
    pairs: dict[str, list[dict[str, Any]]] = {"accept": [], "review": []}
    requests: dict[str, list[dict[str, Any]]] = {name: [] for name in SAMPLES.values()}
    for row, i in enumerate(keep.tolist()):
        a, b = str(l_ids[i]), str(r_ids[i])
        evidence = evidence_of(dict(zip(names, X[row].tolist())), (source[a], source[b]))
        if band[i] in pairs:
            pairs[str(band[i])].append({"a": a, "b": b, "p": round(float(p[i]), 4), "evidence": evidence})
        if sample[i] > 0:
            # The record of the source without duplicates first, as a host names a pair. No
            # score goes with a pair that is asked for: the reviewer is not to be led by it.
            first, second = (b, a) if source[b] in master and source[a] not in master else (a, b)
            requests[SAMPLES[int(sample[i])]].append({
                "id": f"same:{first}~{second}", "a": first, "b": second, "evidence": evidence,
                "label": None if label[i] < 0 else ("same" if label[i] == 1 else "different"), "by": BY[int(label_from[i])] if label[i] >= 0 else None})
    for name in requests:
        requests[name].sort(key=lambda d: d["id"])
    review = sorted(pairs["review"], key=lambda d: (-d["p"], d["a"], d["b"]))
    b_, mo, learned = res["bands"], res["clustering"]["machine_only"], res["bands"]["learned"]
    counted = {name: {"reviewer": sum(d["by"] == "reviewer" for d in asked), "simulated": sum(d["by"] == "simulated" for d in asked),
                      "open": sum(d["label"] is None for d in asked)} for name, asked in requests.items()}
    people = sum(c["reviewer"] for c in counted.values())
    stood_in = sum(c["simulated"] for c in counted.values())
    certified = b_["accept_at"] is not None
    return {
        "format": "resolution/3",
        "state": learned["state"],
        "records": int(res["records"]),
        "sources": {str(k): int(v) for k, v in raw.source_system.value_counts().sort_index().items()},
        "candidate_pairs": int(res["blocking"].get("link_only", {}).get("pairs_between_sources", res["blocking"]["_union"]["pairs"])),
        "labels": {**counted, "total": int(sum(len(asked) for asked in requests.values()))},
        "truth_stands_in": bool(truth_stands_in),
        "lines": {"accept_at": b_["accept_at"], "reject_below": b_["reject_below"], "target": b_["target"], "delta": b_["delta"]},
        "certified": certified,
        # Whose labels a certified line rests on: a reviewer's alone, the truth column's alone, or both.
        "certified_on": None if not certified else "reviewer" if not stood_in else "simulated" if not people else "mixed",
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
        "requests": requests,
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
    ap.add_argument("--labels", type=Path, help="a reviewer's answers: a JSON list of {a, b, same}")
    ap.add_argument("--no-truth", action="store_true", help="the truth column may not stand in for a reviewer: a pair without an answer is open")
    a = ap.parse_args()
    out = link(a.records, a.n_cal, a.n_ltt, a.seed, duplicate_free=tuple(a.duplicate_free),
               labels=read_labels(a.labels) if a.labels else None, truth_stands_in=not a.no_truth)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("state", "records", "candidate_pairs", "labels", "lines", "certified", "certified_on", "duplicate_free", "counts", "realised", "review_total")}, indent=1, default=float))


if __name__ == "__main__":
    main()
