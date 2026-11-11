"""Evidence per candidate pair, as numbers a model fitted on reviewer labels can weigh.

The first stage (Fellegi-Sunter by expectation-maximisation) learns without labels what
agreement on a field is worth. On two real company registers it learned the wrong thing: so
many companies share a registered office that "same address" became its picture of a match,
and two records with the identical name got a probability near zero (2026-11-11). What a field
is worth in an estate is something a few hundred reviewer labels say directly. This module
lays out the evidence; ``learned.py`` fits the weights.

Every feature is a comparison of two records or a count over the estate. None reads the truth.
A three-way comparison is +1 (both present and equal), -1 (both present and different) and 0
(one side missing): a missing value is never a difference.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import jellyfish
import numpy as np
import pandas as pd

ORG_PATTERN = ("same_source", "name_level", "postcode", "legal")  # the strata labels are drawn from
PERSON_PATTERN = ("first_name", "surname", "dob")
PERSON_FIELDS = ("first_name", "surname", "dob", "birth_place", "postcode_fake", "occupation", "gender")


def _codes(values: Sequence[object]) -> np.ndarray:
    codes, _ = pd.factorize(pd.Series(list(values), dtype=object), use_na_sentinel=True)
    return codes.astype(np.int64)


def three_way(values: Sequence[object], left: np.ndarray, right: np.ndarray) -> np.ndarray:
    c = _codes(values)
    cl, cr = c[left], c[right]
    both = (cl >= 0) & (cr >= 0)
    return np.where(both, np.where(cl == cr, 1.0, -1.0), 0.0)


def _clean(v: object) -> str | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def squeezed(v: object) -> str | None:
    """A value with case, spaces and punctuation taken out: "Shedden Farms Ltd." -> "SHEDDENFARMSLTD"."""
    s = _clean(v)
    return re.sub(r"[^A-Za-z0-9]", "", s).upper() or None if s else None


def _counts_of(values: Sequence[object]) -> np.ndarray:
    """For every record, how many records carry its value (0 where it has none)."""
    c = _codes(values)
    if not (c >= 0).any():  # a field no record carries
        return np.zeros(len(c), dtype=np.float64)
    n = np.bincount(c[c >= 0])
    return np.where(c >= 0, n[np.clip(c, 0, None)], 0).astype(np.float64)


def candidate_counts(n: int, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """How many candidate pairs each record is in."""
    return (np.bincount(left, minlength=n) + np.bincount(right, minlength=n)).astype(np.float64)


def rank_features(n: int, left: np.ndarray, right: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Where a pair stands among the other candidates of its two records: whether it is each
    record's best pair, and how far ahead of (or behind) that record's best other pair it is.
    One real thing has one best partner; a pair that is clearly ahead on both sides is stronger
    evidence than its own score says, and a pair that is second best is weaker.

    Not used by the pipeline (2026-11-11): between two registers without duplicates a second
    fit with these changed nothing, and on an estate with several records per organisation
    it cost precision, because a record there has several right partners. It is kept for a
    link between sources that are each known to be free of duplicates."""
    rec = np.concatenate([left, right])
    sc = np.concatenate([score, score])
    order = np.lexsort((-sc, rec))  # by record, best pair first
    rec_s, sc_s = rec[order], sc[order]
    first = np.r_[True, rec_s[1:] != rec_s[:-1]]
    nxt = np.r_[False, first[:-1]] & ~first  # the second pair of a record that has two
    best = np.full(n, -np.inf)
    second = np.full(n, -np.inf)
    best[rec_s[first]] = sc_s[first]
    second[rec_s[nxt]] = sc_s[nxt]
    cols = []
    for side in (left, right):
        is_best = score >= best[side]
        other = np.where(is_best, second[side], best[side])  # the best pair of this record that is not this one
        margin = np.where(np.isfinite(other), score - other, 1.0)
        cols += [is_best.astype(float), np.clip(margin, -1.0, 1.0)]
    out = np.column_stack(cols)
    # A pair has no left and right: order the two sides so the feature does not depend on which is which.
    lo = np.minimum(out[:, [0, 1]], out[:, [2, 3]])
    hi = np.maximum(out[:, [0, 1]], out[:, [2, 3]])
    return np.column_stack([lo, hi]), ["best_for_both", "margin_min", "best_for_one", "margin_max"]


def master_features(left: np.ndarray, right: np.ndarray, score: np.ndarray, source: np.ndarray, duplicate_free: set[int]) -> tuple[np.ndarray, list[str]]:
    """Where a pair stands among a record's candidates in a source that holds no duplicates.

    Such a source (a master list with a key) has at most one record for any real thing. So for
    a record of another source, at most one of its candidates there is right: a candidate that
    is clearly ahead of the others is stronger than its own score says, and two candidates
    level with each other (the same name in two towns, and no town on the record) are a tie
    nobody should break by guessing. ``source`` is a code per record, ``duplicate_free`` the
    codes of the sources without duplicates. For a pair that touches no such source every
    column is 0.

    ``master_pair`` 1 if it does; ``master_best`` 1 if no other candidate of that source scores
    higher; ``master_margin`` its score minus the best other candidate's (1 if there is none);
    ``master_rivals`` how many candidates the record has there.
    """
    n_src = int(source.max()) + 1 if len(source) else 1
    sl, sr = source[left], source[right]
    out = np.zeros((len(left), 4))
    best_all = np.ones(len(left))
    margin_all = np.ones(len(left))
    for x, m_src, mask in ((left, sr, np.isin(sr, list(duplicate_free)) & (sl != sr)), (right, sl, np.isin(sl, list(duplicate_free)) & (sl != sr))):
        i = np.where(mask)[0]
        if not len(i):
            continue
        key = x[i].astype(np.int64) * n_src + m_src[i]
        order = np.lexsort((-score[i], key))
        k_s, s_s = key[order], score[i][order]
        first = np.r_[True, k_s[1:] != k_s[:-1]]
        group = np.cumsum(first) - 1
        best = s_s[first][group]
        size = np.bincount(group)[group]
        second_at = np.where(first)[0] + 1  # the second entry of each group, if it has one
        second = np.full(int(group.max()) + 1, -np.inf)
        has = second_at < len(s_s)
        ok = np.zeros(len(second_at), dtype=bool)
        ok[has] = ~first[second_at[has]]
        second[ok] = s_s[second_at[ok]]
        is_best = s_s >= best
        other = np.where(is_best, second[group], best)
        margin = np.where(np.isfinite(other), s_s - other, 1.0)
        rows = i[order]
        out[rows, 0] = 1.0
        best_all[rows] = np.minimum(best_all[rows], is_best.astype(float))
        margin_all[rows] = np.minimum(margin_all[rows], np.clip(margin, -1.0, 1.0))
        out[rows, 3] = np.maximum(out[rows, 3], np.minimum(size, 50))
    touched = out[:, 0] == 1
    out[touched, 1], out[touched, 2] = best_all[touched], margin_all[touched]
    return out, ["master_pair", "master_best", "master_margin", "master_rivals"]


def org_features(rec: pd.DataFrame, left: np.ndarray, right: np.ndarray, p1: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """The evidence for every candidate pair of organisation records. ``rec`` is the normalised
    frame in record-number order (row ``k`` is record ``k``)."""
    n = len(rec)
    names = [squeezed(v) for v in rec["name"]]
    cores = [_clean(v) or "" for v in rec["name_core"]]
    tokens = [frozenset(c.split()) for c in cores]
    df = Counter(t for ts in tokens for t in ts)
    idf = {t: math.log((n + 1) / (k + 0.5)) for t, k in df.items()}

    li, ri = left.tolist(), right.tolist()
    jw = np.fromiter((jellyfish.jaro_winkler_similarity(cores[a], cores[b]) if cores[a] and cores[b] else 0.0 for a, b in zip(li, ri)),
                     dtype=np.float64, count=len(li))
    share = np.empty(len(li))
    apart = np.empty(len(li))
    for k, (a, b) in enumerate(zip(li, ri)):
        ta, tb = tokens[a], tokens[b]
        both = sum(idf[t] for t in ta & tb)
        diff = sum(idf[t] for t in ta ^ tb)
        share[k] = both / (both + diff) if both + diff > 0 else 0.0
        apart[k] = diff
    name_same = three_way(names, left, right) == 1
    core_same = three_way([c or None for c in cores], left, right) == 1
    postcode = three_way([squeezed(v) for v in rec["postcode"]], left, right)
    legal = three_way(rec["legal_canon"].tolist(), left, right)
    same_source = three_way(rec["source_system"].tolist(), left, right) == 1
    name_freq = _counts_of([c or None for c in cores])
    addr_freq = _counts_of([squeezed(v) for v in rec["postcode"]])
    cand = candidate_counts(n, left, right)
    level = np.where(name_same, 3.0, np.where(core_same, 2.0, np.where(jw >= 0.92, 1.0, 0.0)))
    cols: dict[str, np.ndarray] = {
        "same_source": same_source.astype(float),
        "name_level": level,
        "name_same": name_same.astype(float),
        "core_same": core_same.astype(float),
        "name_jw": jw,
        "token_share": share,
        "token_apart": np.minimum(apart, 40.0),
        "legal": legal,
        "postcode": postcode,
        "pc_out": three_way(rec["pc_out"].tolist(), left, right),
        "city": three_way(rec["city_n"].tolist(), left, right),
        "industry": three_way(rec["industry"].tolist(), left, right),
        "numbers": three_way(rec["num_tokens"].tolist(), left, right),
        "reg": three_way(rec["reg_norm"].tolist(), left, right),
        "name_freq": np.log1p(np.maximum(name_freq[left], name_freq[right])),
        "addr_freq": np.log1p(np.maximum(addr_freq[left], addr_freq[right])),
        "candidates_min": np.log1p(np.minimum(cand[left], cand[right])),
        "candidates_max": np.log1p(np.maximum(cand[left], cand[right])),
    }
    if p1 is not None:
        q = np.clip(p1, 1e-6, 1 - 1e-6)
        cols["first_stage_logit"] = np.log(q / (1 - q))
    return np.column_stack(list(cols.values())), list(cols)


def pattern_strata(X: np.ndarray, names: list[str], by: Sequence[str] = ORG_PATTERN) -> np.ndarray:
    """A stratum number per pair: pairs with the same values of the pattern features."""
    key = np.zeros(len(X), dtype=np.int64)
    for f in by:
        col = X[:, names.index(f)]
        codes = np.unique(col, return_inverse=True)[1]
        key = key * (int(codes.max()) + 1) + codes
    return np.unique(key, return_inverse=True)[1]


def person_features(rec: pd.DataFrame, left: np.ndarray, right: np.ndarray, p1: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """The evidence for every candidate pair of person records: each field three ways, how
    alike the two names are, and how many candidates each record has."""
    cols: dict[str, np.ndarray] = {f: three_way(rec[f].tolist(), left, right) for f in PERSON_FIELDS}
    li, ri = left.tolist(), right.tolist()
    for f in ("first_name", "surname"):
        v = [_clean(x) or "" for x in rec[f]]
        cols[f"{f}_jw"] = np.fromiter((jellyfish.jaro_winkler_similarity(v[a], v[b]) if v[a] and v[b] else 0.0 for a, b in zip(li, ri)),
                                      dtype=np.float64, count=len(li))
    cand = candidate_counts(len(rec), left, right)
    cols["candidates_min"] = np.log1p(np.minimum(cand[left], cand[right]))
    cols["candidates_max"] = np.log1p(np.maximum(cand[left], cand[right]))
    if p1 is not None:
        q = np.clip(p1, 1e-6, 1 - 1e-6)
        cols["first_stage_logit"] = np.log(q / (1 - q))
    return np.column_stack(list(cols.values())), list(cols)
