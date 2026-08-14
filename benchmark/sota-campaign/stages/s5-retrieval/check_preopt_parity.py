"""Prove the optimization work did not move a single quality number.

The S5 matrix was measured twice: once on the code that predates the ticket-01/02/03
latency work and the architecture pass (``results-preopt/``), once on current head
(``results/``). The optimizations were meant to be *pure speed* — every recall, nDCG,
chain and gate number must reproduce bit-for-bit; only ``latency_ms_*`` may move.

This script is that oracle. It walks every cell present in both directories, strips the
latency keys, and compares the rest structurally. Any surviving difference is a
regression in disguise and fails the run.

Run: python benchmark/sota-campaign/stages/s5-retrieval/check_preopt_parity.py
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


def strip_volatile(value: Any) -> Any:
    """Return *value* with every latency key removed, recursively."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def diff_paths(before: Any, after: Any, path: str = "") -> list[str]:
    """Return one human-readable line per structural difference."""
    if type(before) is not type(after):
        return [f"{path}: type {type(before).__name__} -> {type(after).__name__}"]
    if isinstance(before, dict):
        diffs: list[str] = []
        for key in sorted(set(before) | set(after)):
            here = f"{path}.{key}" if path else key
            if key not in before:
                diffs.append(f"{here}: absent -> {after[key]!r}")
            elif key not in after:
                diffs.append(f"{here}: {before[key]!r} -> absent")
            else:
                diffs.extend(diff_paths(before[key], after[key], here))
        return diffs
    if isinstance(before, list):
        if len(before) != len(after):
            return [f"{path}: length {len(before)} -> {len(after)}"]
        return [d for i, (b, a) in enumerate(zip(before, after, strict=True)) for d in diff_paths(b, a, f"{path}[{i}]")]
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
        diffs = diff_paths(strip_volatile(before), strip_volatile(after))
        compared += 1
        if diffs:
            failures.append(f"{old.name}:\n  " + "\n  ".join(diffs[:20]))

    print(f"compared {compared} cell(s) present in both matrices")
    if missing:
        print(f"not re-measured ({len(missing)}): {', '.join(missing)}")

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
