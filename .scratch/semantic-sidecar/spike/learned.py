"""A scorer fitted on reviewer labels over the evidence of ``pairfeat``, and its certified bands.

Three label samples, all drawn before any of their labels is read:

- training: spread over the patterns of evidence (same source or not, how alike the names are,
  the postcode, the legal form), so that rare but decisive patterns are labelled too. A uniform
  sample of candidate pairs is mostly obvious non-matches;
- accept test: uniform over the pairs the fitted scorer puts at or above ``floor``;
- reject test: uniform over the pairs below it.

Learn-then-Test on the accept test certifies an accept line at or above the floor; on the
reject test, a reject line at or below it. A sample that is uniform over a region is uniform
over every part of the region cut off by a line on the score, so each test is as valid as on a
uniform sample of everything, and spends its labels where the line will be.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from er import _ltt_threshold


def draw_training(strata: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``n`` pairs spread evenly over the strata (a small stratum is taken whole and its share
    passes to the others). Returns the pairs and a weight per pair: how many pairs of its
    stratum it stands for."""
    sizes = np.bincount(strata)
    take = np.zeros(len(sizes), dtype=np.int64)
    left, open_ = n, np.where(sizes > 0)[0]
    while left > 0 and len(open_):
        share = max(1, left // len(open_))
        add = np.minimum(share, sizes[open_] - take[open_])
        add[np.cumsum(add) > left] = 0
        if add.sum() == 0:
            break
        take[open_] += add
        left -= int(add.sum())
        open_ = open_[take[open_] < sizes[open_]]
    order = rng.permutation(len(strata))
    by = order[np.argsort(strata[order], kind="stable")]  # shuffled within each stratum
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    idx = np.concatenate([by[s : s + t] for s, t in zip(starts, take) if t > 0])
    return idx, (sizes[strata[idx]] / take[strata[idx]]).astype(float)


def fit_scorer(X: np.ndarray, y: np.ndarray, idx: np.ndarray, weights: np.ndarray | None, model: str = "gbt") -> Callable[[np.ndarray], np.ndarray]:
    """Fit on the labelled pairs ``idx`` (``y`` is read only there). Returns the scorer: evidence
    in, probability out. It can score pairs of another estate that lays out the same evidence."""
    yt = y[idx].astype(int)
    if len(set(yt.tolist())) < 2:
        base = float(yt.mean()) if len(yt) else 0.0
        return lambda Z: np.full(len(Z), base)
    if model == "logit":
        mu, sd = X[idx].mean(axis=0), X[idx].std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(max_iter=3000).fit((X[idx] - mu) / sd, yt, sample_weight=weights)
        return lambda Z: clf.predict_proba((Z - mu) / sd)[:, 1]
    gbt = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=8,
                                         l2_regularization=1.0, random_state=0)
    if min(np.bincount(yt)) >= 10:
        # Five fits, each calibrated on the fifth of the labels it did not see, averaged. Trees
        # fitted once on all labels rate what they have memorised as certain; measured
        # (2026-11-11) the calibrated average found more of the true links on the synthetic
        # estate (recall 0.904 against 0.886) and was more precise on the real registers
        # (0.999 against 0.983).
        folds = CalibratedClassifierCV(gbt, method="sigmoid", cv=5).fit(X[idx], yt, sample_weight=weights)
        return lambda Z: folds.predict_proba(Z)[:, 1]
    gbt.fit(X[idx], yt, sample_weight=weights)
    return lambda Z: gbt.predict_proba(Z)[:, 1]


def fit_score(X: np.ndarray, y: np.ndarray, idx: np.ndarray, weights: np.ndarray | None, model: str = "gbt") -> np.ndarray:
    """Fit on the labelled pairs ``idx`` and score every pair."""
    return fit_scorer(X, y, idx, weights, model)(X)


@dataclass
class Certified:
    accept_at: float | None
    reject_below: float | None
    accept_test: np.ndarray
    reject_test: np.ndarray
    floor: float
    trace: dict[str, list[dict[str, float]]] = field(default_factory=dict)


def _quantile_grid(values: np.ndarray, steps: int = 200) -> list[float]:
    """Lines to try, from the most cautious down: quantiles of the scores, so that every step
    adds about the same number of pairs wherever the scores happen to crowd."""
    if len(values) == 0:
        return []
    q = np.quantile(values, np.linspace(1.0, 0.0, steps + 1), method="lower")
    return [float(t) for t in dict.fromkeys(q.tolist())]


def certify(score: np.ndarray, y: np.ndarray, used: np.ndarray, n_accept: int, n_reject: int, rng: np.random.Generator, *,
            floor: float = 0.5, target: float = 0.95, delta: float = 0.05, reject_target: float = 0.99) -> Certified:
    """Accept and reject lines on ``score`` from two fresh labelled samples (``y`` is read only
    there). ``used`` are pairs already labelled; they are never drawn again. The accept line
    lies at or above the floor and the reject line at or below it. The floor is even odds: a
    pair the scorer itself rates below that is never accepted, however much room the accepted
    set as a whole would leave, and one it rates above is never rejected."""
    free = np.ones(len(score), dtype=bool)
    free[used] = False
    above, below = np.where(free & (score >= floor))[0], np.where(free & (score < floor))[0]
    ta = rng.permutation(above)[:n_accept]
    tr = rng.permutation(below)[:n_reject]
    # The first line tried has a fifth of the test labels behind it. On 100 labelled pairs three
    # wrong ones fail the step, and in a fixed sequence a failed first step ends the test.
    start = max(100, len(ta) // 5)
    accept_at, a_trace = _ltt_threshold(score[ta], 1 - y[ta].astype(int), target, delta, _quantile_grid(score[above]), min_n=start)
    # A line with so few labelled pairs behind it that even zero errors could not pass is
    # skipped, not failed: that reads the scores only, so the sequence is still fixed in advance.
    need = max(100, math.ceil(math.log(delta) / math.log(reject_target)))
    at, r_trace = _ltt_threshold(-score[tr], y[tr].astype(int), reject_target, delta, _quantile_grid(-score[below]), min_n=need)
    reject_below = None if at is None else float(np.nextafter(-at, np.inf))  # every pair scored -at or less
    return Certified(accept_at, reject_below, ta, tr, floor, {"accept": a_trace, "reject": r_trace})
