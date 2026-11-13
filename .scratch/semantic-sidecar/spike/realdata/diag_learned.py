"""How pure the fitted scorer's ranking is above even odds, and what the wrong pairs there look like.

Usage: python realdata/diag_learned.py data/estate_1m.parquet [n_train]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from learned import draw_training, fit_score  # noqa: E402
from pairfeat import ORG_PATTERN, org_features, pattern_strata, rank_features  # noqa: E402
from run_a import ORG_PASSES, _codes, block, normalise_org, record_numbers, score, vetoes  # noqa: E402

t0 = time.time()
df = normalise_org(pd.read_parquet(sys.argv[1]))
n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rid, _ = record_numbers(df.unique_id.to_numpy())
df["rid"] = rid
rec = df.iloc[np.argsort(rid, kind="stable")].reset_index(drop=True)
left, right, _ = block(df, ORG_PASSES)
p_raw = score(df, left, right, "org")
veto = vetoes(rec, left, right, "org")
g = _codes(rec.cluster.astype(str).to_numpy())
y = (g[left] == g[right]).astype(int)
X, names = org_features(rec, left, right, p_raw)
print("features", X.shape, round(time.time() - t0), flush=True)
strata = pattern_strata(X, names, ORG_PATTERN)
rng = np.random.default_rng(7)
idx, w = draw_training(strata, n_train, rng)
first = fit_score(X, y, idx, w)
R, rn = rank_features(len(rec), left, right, first)
s = fit_score(np.column_stack([X, R]), y, idx, w)
s[veto] = 0.0
print("fitted", round(time.time() - t0), flush=True)


def table(title: str, sc: np.ndarray, mask: np.ndarray) -> None:
    print(f"\n== {title}: {int(mask.sum())} pairs, precision {y[mask].mean():.4f}")
    v, yy = sc[mask], y[mask]
    order = np.argsort(-v, kind="stable")
    for k, part in enumerate(np.array_split(order, 20)):
        cum = order[: sum(len(q) for q in np.array_split(order, 20)[: k + 1])]
        print(f"   top {5 * (k + 1):3d}%  score >= {v[part[-1]]:.4f}  slice precision {yy[part].mean():.4f}  cumulative {yy[cum].mean():.4f}")


above = s >= 0.5
table("fitted scorer, pairs at or above even odds", s, above)
table("the same pairs ranked by the first stage alone", p_raw, above)
table("first-fit scorer (no rank features), its own pairs above even odds", first, first >= 0.5)
print("\n== true matches: above", int(y[above].sum()), "below", int(y[~above].sum()), "| first stage p >= 0.72:", int((p_raw >= 0.72).sum()), "precision", round(float(y[p_raw >= 0.72].mean()), 4))
F = np.column_stack([X, R])
fn = names + rn
wrong, right_ = above & (y == 0), above & (y == 1)
print("\n== mean of each feature among wrong and right pairs above even odds")
for k, f in enumerate(fn):
    print(f"   {f:18s} wrong {F[wrong, k].mean():8.3f}   right {F[right_, k].mean():8.3f}")
show = rec[["name", "city", "postcode", "industry", "reg_id", "source_system"]]
pick = np.where(wrong)[0][np.random.default_rng(1).permutation(int(wrong.sum()))[:14]]
print("\n== wrong pairs above even odds: sample")
for i in pick:
    a, b = show.iloc[int(left[i])].tolist(), show.iloc[int(right[i])].tolist()
    print("   ", round(float(s[i]), 3), a, "|", b)
