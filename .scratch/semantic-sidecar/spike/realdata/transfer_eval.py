"""Does a scorer fitted on one estate work on another? Fit on the labels of estate A, score
estate B without one training label from B, and certify B's lines on B's own test labels.

Usage: python realdata/transfer_eval.py out/pairs_small out/pairs_100k out/transfer.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from learned import certify, draw_training, fit_scorer  # noqa: E402
from pairfeat import org_features, pattern_strata  # noqa: E402


def load(d: Path) -> dict:
    pairs, rec = pd.read_parquet(d / "pairs.parquet"), pd.read_parquet(d / "recs.parquet")
    left, right = pairs.l.to_numpy(), pairs.r.to_numpy()
    X, names = org_features(rec, left, right, pairs.p.to_numpy())
    gold = int(rec.groupby("cluster").size().map(lambda k: k * (k - 1) // 2).sum())
    return {"name": d.name, "X": X, "names": names, "y": pairs.y.to_numpy().astype(int), "veto": pairs.veto.to_numpy(), "gold": gold,
            "strata": pattern_strata(X, names)}


def judge(e: dict, s: np.ndarray, used: np.ndarray, n_accept: int, n_reject: int, seed: int) -> dict:
    s = s.copy()
    s[e["veto"]] = 0.0
    c = certify(s, e["y"], used, n_accept, n_reject, np.random.default_rng(seed))
    acc = s >= c.accept_at if c.accept_at is not None else np.zeros(len(s), bool)
    rej = s < c.reject_below if c.reject_below is not None else np.zeros(len(s), bool)
    y = e["y"]
    return {"accepted": int(acc.sum()), "accept_precision": float(y[acc].mean()) if acc.any() else None, "pair_recall": float(y[acc].sum() / e["gold"]),
            "review": int((~acc & ~rej).sum()), "rejected_true": int(y[rej].sum()), "labels_in_target": int(len(used) + len(c.accept_test) + len(c.reject_test))}


A, B = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
rows = []
for src, dst in ((A, B), (B, A)):
    for cols_name in ("all evidence", "without the first stage's score", "without source and first stage"):
        drop = {"all evidence": set(), "without the first stage's score": {"first_stage_logit"},
                "without source and first stage": {"first_stage_logit", "same_source"}}[cols_name]
        keep = [k for k, n in enumerate(src["names"]) if n not in drop]
        for seed in (7, 8, 9):
            rng = np.random.default_rng(seed)
            idx, w = draw_training(src["strata"], 2000, rng)
            scorer = fit_scorer(src["X"][:, keep], src["y"], idx, w)
            none = np.array([], dtype=int)
            for n_accept, n_reject in ((300, 300), (600, 400), (2000, 1000)):
                r = judge(dst, scorer(dst["X"][:, keep]), none, n_accept, n_reject, seed)
                rows.append({"fitted_on": src["name"], "applied_to": dst["name"], "evidence": cols_name, "seed": seed, "how": "carried over", **r})
                print(json.dumps(rows[-1]), flush=True)
            # the same target budget spent on the target alone: a third to fit, the rest to test
            for total in (600, 1000, 3000):
                n_fit = total // 3
                idx_b, w_b = draw_training(dst["strata"], n_fit, np.random.default_rng(seed))
                own = fit_scorer(dst["X"][:, keep], dst["y"], idx_b, w_b)
                rest = total - n_fit
                r = judge(dst, own(dst["X"][:, keep]), idx_b, rest * 3 // 5, rest - rest * 3 // 5, seed)
                rows.append({"fitted_on": dst["name"], "applied_to": dst["name"], "evidence": cols_name, "seed": seed, "how": "own labels", **r})
                print(json.dumps(rows[-1]), flush=True)
            Path(sys.argv[3]).write_text(json.dumps(rows, indent=1))
