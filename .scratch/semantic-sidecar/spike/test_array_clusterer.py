"""The array clusterer decides what the dictionary clusterer decides."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from er import ArrayClusterer, ConstrainedClusterer, NeighbourTable


def _partition(labels: dict[int, object]) -> set[frozenset[int]]:
    groups: dict[object, set[int]] = defaultdict(set)
    for record, cluster in labels.items():
        groups[cluster].add(record)
    return {frozenset(g) for g in groups.values()}


def _random_case(seed: int, n: int = 60, pairs: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, int]] = set()
    while len(seen) < pairs:
        a, b = sorted(int(x) for x in rng.integers(0, n, size=2))
        if a != b:
            seen.add((a, b))
    left = np.array([a for a, _ in sorted(seen)], dtype=np.int32)
    right = np.array([b for _, b in sorted(seen)], dtype=np.int32)
    # A few probability levels, as an isotonic calibrator gives, including strong negatives.
    p = rng.choice([0.0, 0.01, 0.04, 0.3, 0.5, 0.58, 0.75, 0.9, 0.97], size=pairs)
    hard = rng.random(pairs) < 0.08  # vetoes, wherever the probability sits
    expert_same = rng.random(pairs) < 0.1
    return left, right, p, hard, expert_same


@pytest.mark.parametrize("seed", range(40))
def test_same_partition_and_counters_as_the_dictionary_clusterer(seed: int) -> None:
    n = 60
    left, right, p, hard, expert_same = _random_case(seed, n)
    expert_same &= ~hard
    accepts = np.where(p >= 0.58)[0]
    accepts = accepts[np.argsort(-p[accepts], kind="stable")]

    old = ConstrainedClusterer(range(n))  # type: ignore[arg-type]
    adj: dict[int, dict[int, float]] = defaultdict(dict)
    for a, b, pv in zip(left.tolist(), right.tolist(), p.tolist()):
        adj[a][b] = pv
        adj[b][a] = pv
    for i in np.where(hard)[0]:
        old.add_cannot_link(int(left[i]), int(right[i]))  # type: ignore[arg-type]
    for i in np.where(expert_same)[0]:
        old.try_union(int(left[i]), int(right[i]))  # type: ignore[arg-type]
    for i in accepts:
        old.try_union_checked(int(left[i]), int(right[i]), adj)  # type: ignore[arg-type]

    new = ArrayClusterer(n, NeighbourTable(n, left, right, p, {"hard": hard}), "hard")
    for i in np.where(expert_same)[0]:
        new.try_union(int(left[i]), int(right[i]))
    for i in accepts:
        new.try_union_checked(int(left[i]), int(right[i]))

    assert _partition(dict(enumerate(new.label.tolist()))) == _partition(old.clusters())  # type: ignore[arg-type]
    assert (new.merges, new.blocked_merges, new.blocked_by_linkage) == (old.merges, old.blocked_merges, old.blocked_by_linkage)


def test_the_table_lists_every_pair_under_both_ends() -> None:
    left, right = np.array([0, 0, 2], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32)
    t = NeighbourTable(4, left, right, np.array([0.9, 0.1, 0.5]), {"hard": np.array([False, True, False])})
    assert t.start.tolist() == [0, 2, 3, 5, 6]
    assert sorted(zip(t.dst[0:2].tolist(), t.p[0:2].tolist(), t.flags["hard"][0:2].tolist())) == [(1, 0.9, False), (2, 0.1, True)]
    assert t.dst[5:6].tolist() == [2]


def test_a_flagged_pair_blocks_a_merge_it_would_cross() -> None:
    # 0~1 and 1~2 are confident; 0~2 is flagged without being a strong negative.
    left, right = np.array([0, 1, 0], dtype=np.int32), np.array([1, 2, 2], dtype=np.int32)
    p = np.array([0.9, 0.9, 0.6])
    c = ArrayClusterer(3, NeighbourTable(3, left, right, p, {"hard": np.array([False, False, True])}), "hard")
    assert c.try_union_checked(0, 1)
    assert not c.try_union_checked(1, 2)
    assert (c.merges, c.blocked_merges, c.blocked_by_linkage) == (1, 1, 0)
    assert c.label.tolist()[0] == c.label.tolist()[1] != c.label.tolist()[2]
