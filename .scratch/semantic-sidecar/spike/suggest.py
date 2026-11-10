"""A suggestion for each review pair, learned from decisions already made (ticket 22).

The review band is where the matcher's probability is near even, so the probability itself is
no suggestion. What a reviewer's earlier decisions add is which *combinations* of field
agreement turned out to be the same entity and which did not: the matcher weighs each field on
its own and cannot see that. A shallow decision tree over the matcher's own agreement levels
learns those combinations and stays readable: a pair falls into one leaf, the leaf is a short
list of conditions, and the leaf's record is "n of m labelled pairs like this were the same".

Each pair's suggestion comes from a tree that never saw that pair's label (two folds), so the
agreement reported with the truth is out of sample. On synthetic data the labels are the
generator's truth, standing in for recorded decisions. Nothing here decides anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH = 3
MIN_LEAF = 15


def _describe(comparison: str, label: str, level_labels: dict[int, str], lo: float, hi: float) -> dict[str, Any]:
    """One condition of a leaf, as the range of agreement levels it allows for one field.

    Levels are -1 (a value is missing), then 0 (the weakest agreement) upwards. A tree splits on
    "level <= t", so a leaf allows a contiguous range.
    """
    levels = sorted(level for level in level_labels if lo < level <= hi)
    return {"comparison": comparison, "label": label, "levels": levels, "text": wording(levels, max(level_labels))}


def wording(levels: list[int], top: int) -> str:
    """Plain words for a range of agreement levels: -1 is "missing", 0 "differs", ``top`` "agrees",
    anything between is "close"."""
    if not levels:
        return "any"
    lo, hi = levels[0], levels[-1]
    if lo == -1:
        if hi == -1:
            return "missing on one side"
        if hi == 0:
            return "differs or is missing"
        return "any" if hi == top else "not the same, or missing"
    if lo == top:
        return "agrees"
    if hi == top:
        return "present on both sides" if lo == 0 else "close or the same"
    if hi == 0:
        return "differs"
    return "not the same" if lo == 0 else "close, not the same"


def _leaves(tree: DecisionTreeClassifier, comparisons: list[str], labels: dict[str, str], level_labels: dict[str, dict[int, str]]) -> dict[int, list[dict[str, Any]]]:
    """leaf id -> its conditions, one per field the path tests, in the order they are tested."""
    t = tree.tree_
    out: dict[int, list[dict[str, Any]]] = {}

    def walk(node: int, bounds: dict[int, tuple[float, float]], order: list[int]) -> None:
        if t.children_left[node] == -1:
            out[node] = [
                d
                for f in order
                if (d := _describe(comparisons[f], labels.get(comparisons[f], comparisons[f]), level_labels[comparisons[f]], *bounds[f]))["text"] != "any"
            ]
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        lo, hi = bounds.get(f, (-np.inf, np.inf))
        seen = order if f in order else [*order, f]
        walk(int(t.children_left[node]), {**bounds, f: (lo, min(hi, thr))}, seen)
        walk(int(t.children_right[node]), {**bounds, f: (max(lo, thr), hi)}, seen)

    walk(0, {}, [])
    return out


def suggest(
    gammas: np.ndarray,
    same: np.ndarray,
    comparisons: list[str],
    labels: dict[str, str],
    level_labels: dict[str, dict[int, str]],
    seed: int,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    """A suggestion per pair, and how often the suggestions match the labels out of sample.

    ``gammas`` is pairs x comparisons (the matcher's agreement level per field), ``same`` the
    label per pair. Returns one suggestion per pair (None when there are too few labelled pairs
    to learn from) and a summary for the export's ``meta``.
    """
    n = len(same)
    if n < 4 * MIN_LEAF or len(set(same.tolist())) < 2:
        return [None] * n, {"available": False, "reason": f"{n} labelled review pairs are too few to learn from"}
    fold = np.random.default_rng(seed).permutation(n) % 2
    out: list[dict[str, Any] | None] = [None] * n
    for k in (0, 1):
        train, test = fold != k, fold == k
        tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=0)
        tree.fit(gammas[train], same[train])
        conditions = _leaves(tree, comparisons, labels, level_labels)
        train_leaf = tree.apply(gammas[train])
        record = {leaf: (int(same[train][train_leaf == leaf].sum()), int((train_leaf == leaf).sum())) for leaf in conditions}
        for i, leaf in zip(np.where(test)[0].tolist(), tree.apply(gammas[test]).tolist()):
            n_same, n_all = record[leaf]
            verdict = "same" if 2 * n_same >= n_all else "different"
            out[i] = {
                "verdict": verdict,
                "like_this": n_all,
                "like_this_agree": n_same if verdict == "same" else n_all - n_same,
                "conditions": conditions[leaf],
            }
    right = sum(1 for s, y in zip(out, same.tolist()) if s is not None and (s["verdict"] == "same") == bool(y))
    return out, {
        "available": True,
        "how": f"a decision tree of depth {MAX_DEPTH} over the matcher's agreement level per field, fitted on labelled review pairs",
        "labelled_pairs": int(n),
        "out_of_sample_right": int(right),
        "out_of_sample_share": round(right / n, 4),
        "always_same_share": round(float(same.mean()), 4),
        "labels": "the generator's truth, standing in for recorded decisions",
    }
