"""Compare two dumps of run_a.py (--dump): do two implementations give the same result?

Same candidate pairs, the same probability and band for each, and the same two partitions of
the records into clusters. Cluster ids are arbitrary names, so partitions are compared after
renaming every cluster to its smallest member. Exit code 0 when everything matches.

Usage: python equivalence.py out/ref/12k.npz out/new/12k.npz
"""

from __future__ import annotations

import sys

import numpy as np


def canonical(ids: np.ndarray, labels: np.ndarray) -> dict[str, str]:
    """record id -> the smallest record id of its cluster."""
    smallest: dict[str, str] = {}
    for i, c in zip(ids.tolist(), labels.tolist()):
        if c not in smallest or i < smallest[c]:
            smallest[c] = i
    return {i: smallest[c] for i, c in zip(ids.tolist(), labels.tolist())}


def compare(a_path: str, b_path: str, p_tolerance: float = 0.0) -> list[str]:
    a, b = np.load(a_path), np.load(b_path)
    problems: list[str] = []
    ka = np.char.add(np.char.add(a["l"], "~"), a["r"])
    kb = np.char.add(np.char.add(b["l"], "~"), b["r"])
    oa, ob = np.argsort(ka, kind="stable"), np.argsort(kb, kind="stable")
    if len(ka) != len(kb) or not np.array_equal(ka[oa], kb[ob]):
        return [f"candidate pairs differ: {len(ka)} against {len(kb)}"]
    for name in ("p_raw", "p"):
        diff = np.abs(a[name][oa] - b[name][ob])
        if diff.max(initial=0.0) > p_tolerance:
            problems.append(f"{name}: {int((diff > p_tolerance).sum())} pairs differ, largest difference {diff.max():.3g}")
    for name in ("band", "veto"):
        if not np.array_equal(a[name][oa], b[name][ob]):
            problems.append(f"{name}: {int((a[name][oa] != b[name][ob]).sum())} pairs differ")
    for name in ("accept_at", "reject_below"):
        if not np.array_equal(a[name], b[name], equal_nan=True):
            problems.append(f"{name}: {a[name][0]} against {b[name][0]}")
    for name in ("machine", "expert"):
        ca, cb = canonical(a["ids"], a[name]), canonical(b["ids"], b[name])
        if ca.keys() != cb.keys():
            problems.append(f"{name}: the two runs cover different records")
        else:
            moved = sum(1 for i in ca if ca[i] != cb[i])
            if moved:
                problems.append(f"{name} clustering: {moved} records are in a different cluster")
    return problems


if __name__ == "__main__":
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    found = compare(sys.argv[1], sys.argv[2], tol)
    print("\n".join(found) if found else "identical: pairs, probabilities, bands and both clusterings")
    sys.exit(1 if found else 0)
