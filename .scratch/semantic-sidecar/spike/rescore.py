"""Second-stage scoring of the pairs the first stage leaves uncertain (ticket 20, decision 5).

Splink scores every candidate pair. The pairs that fall between its reject and accept bands are
written out as text and scored by neural pair scorers (a reranker, a decision model). Those
scores join the Splink probability as features of a logistic calibrator, and Learn-then-Test
sets a second pair of bands on a labelled sample drawn from the uncertain region only.

Two stages each make their own accept claim, so each runs at delta / 2: by the union bound the
combined accepted set keeps precision >= target with probability >= 1 - delta.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from er import Bands, learn_then_test_bands

ORG_TEXT_COLS = ("name", "city", "postcode", "country", "industry", "reg_id")
PERSON_TEXT_COLS = ("first_name", "surname", "dob", "birth_place", "postcode_fake", "occupation")


def record_text(row: Mapping[str, Any], cols: Sequence[str]) -> str:
    """One record as a line of text for a pair scorer. Missing fields are left out, not written
    as "None": an absent value is no evidence, and a literal "None" would look like agreement."""
    parts = [f"{c}: {row[c]}" for c in cols if c in row and pd.notna(row[c]) and str(row[c]).strip()]
    return " | ".join(parts)


def uncertain_region(p: np.ndarray, bands: Bands, veto: np.ndarray) -> np.ndarray:
    """Pairs the first stage neither accepts nor rejects. With no accept band (or no reject
    band) the region extends to that end: an abstaining first stage leaves everything open."""
    lo = bands.reject_below if bands.reject_below is not None else 0.0
    hi = bands.accept_at if bands.accept_at is not None else np.inf
    return (p >= lo) & (p < hi) & ~veto


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def feature_matrix(p1: np.ndarray, scores: Mapping[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """[logit(first-stage p), one column per scorer]. A scorer that returned NaN for a pair (it
    could not answer) gets the column median plus its own "missing" indicator column."""
    cols, names = [_logit(p1)], ["splink_logit"]
    for name in sorted(scores):
        s = np.asarray(scores[name], dtype=float)
        if s.shape != p1.shape:
            raise ValueError(f"scorer {name!r} returned {s.shape}, expected {p1.shape}")
        miss = ~np.isfinite(s)
        if miss.all():
            raise ValueError(f"scorer {name!r} returned no finite score")
        cols.append(np.where(miss, np.nanmedian(np.where(miss, np.nan, s)), s))
        names.append(name)
        if miss.any():
            cols.append(miss.astype(float))
            names.append(f"{name}_missing")
    return np.column_stack(cols), names


@dataclass
class StageTwo:
    p2: np.ndarray  # calibrated probability for every uncertain pair
    bands: Bands
    feature_names: list[str]
    coefficients: list[float]
    n_cal: int
    n_ltt: int


def stage_two(
    p1: np.ndarray,
    scores: Mapping[str, np.ndarray],
    y: np.ndarray,
    cal_idx: np.ndarray,
    ltt_idx: np.ndarray,
    *,
    target: float = 0.95,
    delta: float = 0.025,
    reject_target: float = 0.99,
) -> StageTwo:
    """Calibrate and band the uncertain pairs. All arrays are indexed over the uncertain region;
    ``cal_idx`` and ``ltt_idx`` are disjoint labelled samples from it (``y`` is read only there)."""
    if set(cal_idx.tolist()) & set(ltt_idx.tolist()):
        raise ValueError("calibration and Learn-then-Test samples must be disjoint")
    X, names = feature_matrix(p1, scores)
    mu, sd = X[cal_idx].mean(axis=0), X[cal_idx].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    if len(set(y[cal_idx].tolist())) < 2:
        # One class only in the labelled sample: nothing to learn, keep the first-stage order.
        p2, coef = p1.copy(), [1.0] + [0.0] * (len(names) - 1)
    else:
        lr = LogisticRegression(max_iter=2000).fit(Z[cal_idx], y[cal_idx])
        p2, coef = lr.predict_proba(Z)[:, 1], [float(c) for c in lr.coef_[0]]
    bands = learn_then_test_bands(p2[ltt_idx], y[ltt_idx], target=target, delta=delta, reject_target=reject_target)
    return StageTwo(p2, bands, names, coef, int(len(cal_idx)), int(len(ltt_idx)))
