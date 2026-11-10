"""Feasibility-spike library: metrics, Learn-then-Test bands, calibration, constrained clustering.

Throwaway prototype code for ticket 20. Nothing here is shipped; it exists to measure whether the
candidate design in ../design/resolution-at-scale.md is worth building.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom
from sklearn.isotonic import IsotonicRegression

# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def pairwise_and_bcubed(pred: dict[str, str], gold: dict[str, str]) -> dict[str, float]:
    """Pairwise P/R/F1 and B-cubed P/R/F1 of a predicted clustering against gold.

    ``pred`` and ``gold`` map every record id to a cluster id; both must cover the same ids.
    Computed from the cluster contingency table, so it is O(records), not O(pairs).
    """
    assert pred.keys() == gold.keys(), "pred and gold must cover the same records"
    cell: dict[tuple[str, str], int] = defaultdict(int)
    psize: dict[str, int] = defaultdict(int)
    gsize: dict[str, int] = defaultdict(int)
    for rid, p in pred.items():
        g = gold[rid]
        cell[(p, g)] += 1
        psize[p] += 1
        gsize[g] += 1
    tp = sum(_pairs(n) for n in cell.values())
    pp = sum(_pairs(n) for n in psize.values())
    gp = sum(_pairs(n) for n in gsize.values())
    prec = tp / pp if pp else 1.0
    rec = tp / gp if gp else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # B-cubed: per record, |pred ∩ gold| / |pred| and / |gold|, averaged.
    n = len(pred)
    bp = sum(c * c / psize[p] for (p, _g), c in cell.items()) / n
    br = sum(c * c / gsize[g] for (_p, g), c in cell.items()) / n
    bf = 2 * bp * br / (bp + br) if bp + br else 0.0
    return {
        "pairwise_precision": prec,
        "pairwise_recall": rec,
        "pairwise_f1": f1,
        "bcubed_precision": bp,
        "bcubed_recall": br,
        "bcubed_f1": bf,
        "predicted_clusters": len(psize),
        "gold_clusters": len(gsize),
    }


# --------------------------------------------------------------------------------------------
# Calibration and Learn-then-Test bands
# --------------------------------------------------------------------------------------------


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(scores, labels.astype(float))
    return iso


def _ltt_threshold(
    scores: np.ndarray,
    errors: np.ndarray,
    target: float,
    delta: float,
    grid: Sequence[float],
    min_n: int = 100,
) -> tuple[float | None, list[dict[str, float]]]:
    """Fixed-sequence Learn-then-Test (Angelopoulos et al.) for a one-sided band.

    ``grid`` is ordered from the most conservative threshold to the most permissive. For each
    threshold the null hypothesis is "the error rate of the selected set exceeds 1 - target";
    it is rejected when the binomial tail P(Bin(n, 1-target) <= k) <= delta. Fixed-sequence
    testing walks the grid in order and stops at the first threshold whose null cannot be
    rejected, which controls the family-wise error at delta without a Bonferroni correction.
    Returns the last threshold that passed (None if even the most conservative fails).
    """
    alpha = 1.0 - target
    passed: float | None = None
    trace: list[dict[str, float]] = []
    for t in grid:
        sel = scores >= t
        n = int(sel.sum())
        k = int(errors[sel].sum())
        p_value = float(binom.cdf(k, n, alpha)) if n else 1.0
        trace.append({"threshold": float(t), "n": n, "errors": k, "p_value": p_value})
        if n < min_n:
            # Too few pairs to ever reject the null: skipping uses only the scores, never the
            # labels, so the fixed sequence is still set before the labels are looked at.
            continue
        if p_value <= delta:
            passed = float(t)
        else:
            break
    return passed, trace


@dataclass
class Bands:
    accept_at: float | None  # calibrated prob >= accept_at  -> auto-accept
    reject_below: float | None  # calibrated prob < reject_below -> auto-reject
    target: float
    delta: float
    n_labelled: int
    accept_trace: list[dict[str, float]] = field(default_factory=list)
    reject_trace: list[dict[str, float]] = field(default_factory=list)

    def band(self, p: float) -> str:
        if self.accept_at is not None and p >= self.accept_at:
            return "accept"
        if self.reject_below is not None and p < self.reject_below:
            return "reject"
        return "review"


def learn_then_test_bands(
    cal_probs: np.ndarray,
    labels: np.ndarray,
    *,
    target: float = 0.95,
    delta: float = 0.05,
    reject_target: float = 0.99,
) -> Bands:
    """Accept and reject thresholds with a finite-sample guarantee on a held-out labelled sample.

    Accept: P(precision of the accepted set >= target) >= 1 - delta (the owner's bar, 0.95).
    Reject: P(fraction of true matches among rejected pairs <= 1 - reject_target) >= 1 - delta.
    The reject side needs its own, stricter target: rejected pairs are the large majority, so
    letting 5% of them be true matches would silently drop many matches.
    """
    grid_hi = list(np.round(np.arange(1.0, 0.0, -0.005), 3))
    accept_at, a_trace = _ltt_threshold(cal_probs, 1 - labels, target, delta, grid_hi)
    # Reject band: mirror the problem. Score = 1 - p, "error" = a true match that gets rejected.
    neg_scores = 1.0 - cal_probs
    reject_score_at, r_trace = _ltt_threshold(neg_scores, labels, reject_target, delta, grid_hi)
    reject_below = None if reject_score_at is None else 1.0 - reject_score_at + 1e-12
    if accept_at is not None and reject_below is not None and reject_below > accept_at:
        reject_below = accept_at  # bands never overlap
    return Bands(accept_at, reject_below, target, delta, int(len(labels)), a_trace, r_trace)


# --------------------------------------------------------------------------------------------
# Constrained clustering
# --------------------------------------------------------------------------------------------


class ConstrainedClusterer:
    """Union-Find that refuses merges a cannot-link or a strong negative would cross.

    Each root keeps the set of record ids it must never join. Merging two clusters unions their
    member sets and their cannot sets (small into large), so a check is O(min cluster size).
    """

    def __init__(self, ids: Iterable[str]) -> None:
        self.parent: dict[str, str] = {i: i for i in ids}
        self.members: dict[str, set[str]] = {i: {i} for i in self.parent}
        self.cannot: dict[str, set[str]] = defaultdict(set)
        self.blocked_merges = 0
        self.blocked_by_linkage = 0
        self.merges = 0

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def add_cannot_link(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            raise ValueError(f"cannot-link {a}~{b} contradicts an existing must-link")
        self.cannot[ra].add(b)
        self.cannot[rb].add(a)

    def _crosses(self, ra: str, rb: str) -> bool:
        small, large = (ra, rb) if len(self.members[ra]) <= len(self.members[rb]) else (rb, ra)
        lm = self.members[large]
        if any(x in lm for x in self.cannot.get(small, ())):
            return True
        sm = self.members[small]
        return any(x in sm for x in self.cannot.get(large, ()))

    def try_union(self, a: str, b: str, *, force: bool = False) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if not force and self._crosses(ra, rb):
            self.blocked_merges += 1
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra] |= self.members.pop(rb)
        if rb in self.cannot:
            self.cannot[ra] |= self.cannot.pop(rb)
        self.merges += 1
        return True

    def linkage_ok(
        self,
        a: str,
        b: str,
        adj: dict[str, dict[str, float]],
        *,
        min_mean: float = 0.5,
        strong_negative: float = 0.05,
    ) -> bool:
        """Cluster-level check: may the clusters of ``a`` and ``b`` merge?

        Looks at every SCORED pair crossing the two clusters (unscored pairs are no evidence,
        not negative evidence). Refuses when any crossing pair is a strong negative, or when the
        crossing pairs are on average not confident matches. This is what stops a chain of
        individually good edges (A~B, B~C) from gluing together records the scorer itself says
        do not match (A !~ C).
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        small, large = (ra, rb) if len(self.members[ra]) <= len(self.members[rb]) else (rb, ra)
        lm = self.members[large]
        total, n = 0.0, 0
        for x in self.members[small]:
            for y, p in adj.get(x, {}).items():
                if y in lm:
                    if p < strong_negative:
                        return False
                    total += p
                    n += 1
        return n == 0 or total / n >= min_mean

    def try_union_checked(self, a: str, b: str, adj: dict[str, dict[str, float]], **kw: float) -> bool:
        if not self.linkage_ok(a, b, adj, **kw):
            self.blocked_by_linkage += 1
            return False
        return self.try_union(a, b)

    def clusters(self) -> dict[str, str]:
        return {i: self.find(i) for i in self.parent}


class NeighbourTable:
    """Every record's scored neighbours, as sorted arrays.

    At 10^6 records the candidate set is about 3 x 10^7 pairs. A dictionary of dictionaries of
    them, and a Python set of every strong negative, do not fit in memory; three flat arrays do.
    Records are the integers 0..n-1. ``start[i]:start[i+1]`` is record ``i``'s slice of ``dst``
    (the neighbour), ``p`` (the pair's probability) and each array in ``flags`` (per-pair marks,
    for example "a merge must never cross this pair"). Every pair appears under both its ends.
    """

    def __init__(self, n: int, left: np.ndarray, right: np.ndarray, p: np.ndarray, flags: dict[str, np.ndarray]) -> None:
        src = np.concatenate([left, right])
        order = np.argsort(src, kind="stable")
        self.dst = np.concatenate([right, left])[order].astype(np.int32)
        self.p = np.concatenate([p, p])[order].astype(np.float64)
        self.flags = {name: np.concatenate([f, f])[order] for name, f in flags.items()}
        self.start = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=self.start[1:])


class ArrayClusterer:
    """The decisions of :class:`ConstrainedClusterer`, over integer records and a NeighbourTable.

    Two things differ in how it is held, nothing in what it decides:

    - A cannot-link is a flagged pair in the table (``hard``), not an entry in a per-cluster set.
      ``ConstrainedClusterer`` refuses a merge when a cannot-link has one end in each cluster;
      this refuses it when a flagged pair has one end in each cluster. That is the same test.
    - ``label[i]`` is record ``i``'s cluster, kept flat: a merge relabels the smaller cluster.

    ``test_array_clusterer.py`` runs both on the same random inputs and requires the same
    partition and the same counters.
    """

    def __init__(self, n: int, table: NeighbourTable, hard: str) -> None:
        self.label = np.arange(n, dtype=np.int32)
        self._members: dict[int, list[int]] = {}  # clusters of two or more; a singleton is implicit
        self.table = table
        self.hard = table.flags[hard]
        self.blocked_merges = 0
        self.blocked_by_linkage = 0
        self.merges = 0

    def members(self, cluster: int) -> list[int]:
        return self._members.get(cluster) or [cluster]

    def _crossing(self, ca: int, cb: int) -> tuple[np.ndarray, np.ndarray]:
        """Probability and hard flag of every scored pair with one end in each cluster."""
        ma, mb = self.members(ca), self.members(cb)
        small, other = (ma, cb) if len(ma) <= len(mb) else (mb, ca)
        t = self.table
        ps: list[np.ndarray] = []
        hs: list[np.ndarray] = []
        for x in small:
            s, e = t.start[x], t.start[x + 1]
            sel = self.label[t.dst[s:e]] == other
            ps.append(t.p[s:e][sel])
            hs.append(self.hard[s:e][sel])
        if len(ps) == 1:
            return ps[0], hs[0]
        return np.concatenate(ps), np.concatenate(hs)

    def _merge(self, ca: int, cb: int) -> None:
        ma, mb = self.members(ca), self.members(cb)
        if len(ma) < len(mb):
            ca, cb, ma, mb = cb, ca, mb, ma
        self.label[mb] = ca
        self._members[ca] = ma + mb
        self._members.pop(cb, None)
        self.merges += 1

    def try_union(self, a: int, b: int) -> bool:
        ca, cb = int(self.label[a]), int(self.label[b])
        if ca == cb:
            return True
        if self._crossing(ca, cb)[1].any():
            self.blocked_merges += 1
            return False
        self._merge(ca, cb)
        return True

    def try_union_checked(self, a: int, b: int, *, min_mean: float = 0.5, strong_negative: float = 0.05) -> bool:
        """The cluster-level check of ``ConstrainedClusterer.linkage_ok``, then the merge."""
        ca, cb = int(self.label[a]), int(self.label[b])
        if ca == cb:
            return True
        ps, hard = self._crossing(ca, cb)
        if ps.size and (ps.min() < strong_negative or math.fsum(ps.tolist()) / ps.size < min_mean):
            self.blocked_by_linkage += 1
            return False
        if hard.any():
            self.blocked_merges += 1
            return False
        self._merge(ca, cb)
        return True


def plain_closure(ids: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Unconstrained transitive closure (what connected components at a threshold does)."""
    cc = ConstrainedClusterer(ids)
    for a, b in edges:
        cc.try_union(a, b, force=True)
    return cc.clusters()


def stable_ids(new: dict[str, str], previous: dict[str, str] | None) -> dict[str, str]:
    """Map each new cluster to the previous cluster id it overlaps most (deterministic ties)."""
    if not previous:
        return new
    overlap: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rid, c in new.items():
        if rid in previous:
            overlap[c][previous[rid]] += 1
    taken: set[str] = set()
    rename: dict[str, str] = {}
    for c in sorted(overlap, key=lambda c: (-max(overlap[c].values()), c)):
        for old, _n in sorted(overlap[c].items(), key=lambda kv: (-kv[1], kv[0])):
            if old not in taken:
                rename[c] = old
                taken.add(old)
                break
    return {rid: rename.get(c, c) for rid, c in new.items()}


def human_minutes(n_items: int, seconds_per_item: float = 20.0) -> float:
    """Estimated expert effort: reported, gates nothing during development (owner, 2026-11-08)."""
    return n_items * seconds_per_item / 60.0


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d
