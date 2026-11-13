"""Try the label-fitted scorer on dumped candidate pairs (``pairs_dump.py``).

Usage: python realdata/learn_eval.py out/pairs_small [seeds]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from learned import certify, draw_training, fit_score  # noqa: E402
from pairfeat import org_features, pattern_strata, rank_features  # noqa: E402

d = Path(sys.argv[1])
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
pairs = pd.read_parquet(d / "pairs.parquet")
rec = pd.read_parquet(d / "recs.parquet")
left, right, y, veto = pairs.l.to_numpy(), pairs.r.to_numpy(), pairs.y.to_numpy(), pairs.veto.to_numpy()
gold = int((rec.groupby("cluster").size().map(lambda k: k * (k - 1) // 2)).sum())
t = time.time()
X, names = org_features(rec, left, right, pairs.p.to_numpy())
print(f"features {X.shape} in {time.time() - t:.0f}s; gold pairs {gold}; candidate true {int(y.sum())}; vetoed true {int(y[veto].sum())}", flush=True)
strata = pattern_strata(X, names)
print("strata", int(strata.max()) + 1, flush=True)
rule = (X[:, names.index("name_same")] == 1) & (X[:, names.index("same_source")] == 0)
print(f"reference rule (identical name across sources): {int(rule.sum())} pairs, precision {y[rule].mean():.4f}, recall {y[rule].sum() / gold:.4f}")
no_first = [i for i, n in enumerate(names) if n != "first_stage_logit"]


def one(sampling: str, model: str, n_train: int, ranks: bool, first: bool, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    Xs = X if first else X[:, no_first]
    if sampling == "uniform":
        idx, w = rng.permutation(len(X))[:n_train], None
    else:
        idx, w = draw_training(strata, n_train, rng)
    s = fit_score(Xs, y, idx, w, model)
    if ranks:
        R, _ = rank_features(len(rec), left, right, s)
        s = fit_score(np.column_stack([Xs, R]), y, idx, w, model)
    c = certify(s, y, idx, 1000, 500, rng)
    acc = s >= c.accept_at if c.accept_at is not None else np.zeros(len(s), bool)
    rej = s < c.reject_below if c.reject_below is not None else np.zeros(len(s), bool)
    rev = ~acc & ~rej
    return {"accepted": acc.sum(), "precision": y[acc].mean() if acc.any() else np.nan, "recall": y[acc].sum() / gold,
            "review": rev.sum(), "review_true": y[rev].sum(), "rejected_true": y[rej].sum(), "abstain": float(c.accept_at is None),
            "labels": len(idx) + len(c.accept_test) + len(c.reject_test), "train_pos": y[idx].sum()}


rows = []
for sampling in ("uniform", "strata"):
    for model in ("logit", "gbt"):
        for n_train in (300, 1000, 2500):
            for ranks in (False, True):
                for first in (True, False):
                    if not first and (ranks or n_train != 1000):
                        continue
                    t = time.time()
                    rs = [one(sampling, model, n_train, ranks, first, 100 + k) for k in range(seeds)]
                    m = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
                    worst = min(r["precision"] for r in rs if not np.isnan(r["precision"])) if any(not np.isnan(r["precision"]) for r in rs) else np.nan
                    rows.append({"sampling": sampling, "model": model, "n_train": n_train, "ranks": ranks, "first_stage": first, **m, "worst_precision": worst})
                    print(f"{sampling:8s} {model:5s} train {n_train:5d} ranks {int(ranks)} first {int(first)} | accepted {m['accepted']:9.0f} P {m['precision']:.4f} (worst {worst:.4f}) "
                          f"R {m['recall']:.4f} | review {m['review']:9.0f} (true {m['review_true']:6.0f}) rejected true {m['rejected_true']:6.0f} | abstain {m['abstain']:.1f} "
                          f"| labels {m['labels']:.0f} train pos {m['train_pos']:.0f} | {time.time() - t:.0f}s", flush=True)
pd.DataFrame(rows).to_json(d / "learn_eval.json", orient="records", indent=1)
