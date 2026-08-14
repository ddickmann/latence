"""Prove the optimization work did not move a single quality number.

The S5 matrix was measured twice: once on the code that predates the ticket-01/02/03
latency work and the architecture pass (``results-preopt/``), once on current head
(``results/``). The optimizations were meant to be *pure speed* — every recall, nDCG,
chain and gate number must reproduce bit-for-bit; only ``latency_ms_*`` may move.

This script is that oracle, and it is deliberately narrow about what it is allowed to
claim. Two things make a cell pair incomparable, and neither is a regression:

* **Different query populations.** Twenty pre-optimization cells were measured under
  ``--limit 1000`` on the three large corpora; the re-run measured them whole (hotpotqa
  3,703 / uda 8,583 / wiki2multihop 12,576). A recall computed over a different query
  set is a different quantity, so those pairs are reported as NOT COMPARABLE rather than
  silently counted as either a pass or a failure.
* **Metrics that did not exist yet.** ``doc_recall@5`` / ``doc_recall@10`` were added
  after the pre-optimization sweep. A key going absent → present is an addition; only a
  key present in *both* whose value moved is evidence of a regression.

Everything else must match exactly — with one measured caveat. The pre-optimization sweep ran
four contended processes with an unpinned BLAS thread count; the re-run ran one process at
``OMP_NUM_THREADS=1``. Thread count changes float summation order in the dense leg's ``gemv``
and in the query embedder, which changes *rankings*: 33 of 1,209 musique queries get a different
dense pool at 4 threads versus 1, and one gets a different gold-hit count. That is the harness's
reproducibility floor across the two sweeps — about ±1 query in 1,209 — and it is why
``run_matrix`` now refuses to run unless the thread count is pinned. Cells produced under the
pinned contract are comparable without this caveat; these two matrices straddle it.

Run:

    python benchmark/sota-campaign/stages/s5-retrieval/check_preopt_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

STAGE = Path(__file__).resolve().parent
PREOPT = STAGE / "results-preopt"
CURRENT = STAGE / "results"

#: Latency is the thing the optimizations were allowed to change.
VOLATILE = ("latency_ms_median", "latency_ms_p95")

#: Run provenance, not measurement: which slice of the query set the run was asked for.
#: It records how the cell was invoked, so it differs whenever the two sweeps were
#: invoked differently — that is what ``comparable()`` is for, and repeating it here
#: as a "difference" would only add noise.
PROVENANCE = ("limit",)


def strip_volatile(value: Any) -> Any:
    """Return *value* with every latency and run-provenance key removed, recursively."""
    drop = VOLATILE + PROVENANCE
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if k not in drop}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def query_count(cell: dict[str, Any]) -> int | None:
    """How many queries this cell's metrics were computed over."""
    count = cell.get("metrics", {}).get("queries")
    return count if isinstance(count, int) else None


def diff_paths(before: Any, after: Any, path: str = "") -> list[str]:
    """Return one human-readable line per structural difference.

    A key that is absent in *before* and present in *after* is an added metric, not a
    changed one — the pre-optimization sweep predates ``doc_recall@*``. Only a key
    present on both sides can testify that a number moved.
    """
    if type(before) is not type(after):
        return [f"{path}: type {type(before).__name__} -> {type(after).__name__}"]
    if isinstance(before, dict):
        diffs: list[str] = []
        for key in sorted(set(before) & set(after)):
            here = f"{path}.{key}" if path else key
            diffs.extend(diff_paths(before[key], after[key], here))
        for key in sorted(set(before) - set(after)):
            diffs.append(f"{path}.{key}: {before[key]!r} -> absent" if path else f"{key}: dropped")
        return diffs
    if isinstance(before, list):
        if len(before) != len(after):
            return [f"{path}: length {len(before)} -> {len(after)}"]
        pairs = enumerate(zip(before, after, strict=True))
        return [d for i, (b, a) in pairs for d in diff_paths(b, a, f"{path}[{i}]")]
    return [] if before == after else [f"{path}: {before!r} -> {after!r}"]


def speedup(before: dict[str, Any], after: dict[str, Any]) -> tuple[float, float] | None:
    """Return (before_ms, after_ms) medians for the headline metrics block."""
    b = before.get("metrics", {}).get("latency_ms_median")
    a = after.get("metrics", {}).get("latency_ms_median")
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        return float(b), float(a)
    return None


def main() -> int:
    if not PREOPT.is_dir():
        print(f"missing {PREOPT}", file=sys.stderr)
        return 2
    if not CURRENT.is_dir():
        print(f"missing {CURRENT}", file=sys.stderr)
        return 2

    compared = 0
    failures: list[str] = []
    missing: list[str] = []
    incomparable: list[tuple[str, int | None, int | None]] = []
    timings: list[tuple[str, float, float]] = []

    for old in sorted(PREOPT.glob("*.json")):
        new = CURRENT / old.name
        if not new.is_file():
            missing.append(old.name)
            continue
        before = json.loads(old.read_text())
        after = json.loads(new.read_text())
        pair = speedup(before, after)
        if pair is not None:
            timings.append((old.stem, pair[0], pair[1]))

        qb, qa = query_count(before), query_count(after)
        if qb != qa:
            incomparable.append((old.stem, qb, qa))
            continue

        diffs = diff_paths(strip_volatile(before), strip_volatile(after))
        compared += 1
        if diffs:
            failures.append(f"{old.name}:\n  " + "\n  ".join(diffs[:20]))

    print(f"compared {compared} cell(s) measured over the same query set")
    if missing:
        print(f"not re-measured ({len(missing)}): {', '.join(missing)}")
    if incomparable:
        print(f"\nnot comparable ({len(incomparable)}) — the query population changed, so a")
        print("recall difference here is a different quantity, not a regression:")
        for stem, qb, qa in incomparable:
            print(f"  {stem:<40} {qb} -> {qa} queries")

    if timings:
        print("\nlatency (headline leg, median ms)")
        print(f"  {'cell':<40} {'before':>10} {'after':>10} {'speedup':>9}")
        for stem, b, a in sorted(timings, key=lambda t: -(t[1] / t[2]) if t[2] else 0):
            factor = f"{b / a:.1f}x" if a else "n/a"
            print(f"  {stem:<40} {b:>10.1f} {a:>10.1f} {factor:>9}")

    if failures:
        print(f"\nQUALITY PARITY FAILED — {len(failures)} cell(s) moved:", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"\nQUALITY PARITY HELD — {compared} cell(s), zero non-latency differences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
