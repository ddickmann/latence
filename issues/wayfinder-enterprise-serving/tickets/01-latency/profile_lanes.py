# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Per-lane wall-time profile over real exports: medians + p95, JSON out.

Usage::

    PYTHONHASHSEED=0 .venv/bin/python issues/wayfinder-enterprise-serving/tickets/01-latency/\
profile_lanes.py <label> <dataset> [<dataset> ...] [--limit N]

``label`` names the output file (``results/<label>-<dataset>.json``) — ``before``/``after``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import (  # noqa: E402
    lane_cases,
    load_runner,
    percentile,
    run_hyper_ladder,
    run_hyper_rescue,
    run_kg,
    timed,
)


def profile_dataset(label: str, dataset: str, limit: int | None) -> dict[str, object]:
    """Time the three lanes over the (sampled) query set of ``dataset``.

    ``label == "before"`` opens the store uncached (the pre-ticket scan-per-call path,
    preserved verbatim in the package); any other label runs the shipped default (the
    cached index). The first hypergraph call of an ``after`` run pays the one-time index
    build; it is timed separately and excluded from the per-query distribution.
    """
    cache = label != "before"
    started = time.perf_counter()
    runner = load_runner(dataset)
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    cases = lane_cases(runner, dataset, limit=limit)
    case_s = time.perf_counter() - started
    print(f"[{dataset}] runner {load_s:.1f}s, {len(cases)} cases ({case_s:.1f}s)", flush=True)

    index_build_s = 0.0
    if cache:
        started = time.perf_counter()
        run_hyper_ladder(runner, cases[0], cache=True)
        index_build_s = time.perf_counter() - started

    timings: dict[str, list[float]] = {"kg": [], "hyper_ladder": [], "hyper_rescue": []}
    for i, case in enumerate(cases):
        _, ms = timed(run_kg, runner, case)
        timings["kg"].append(ms)
        _, ms = timed(run_hyper_ladder, runner, case, cache=cache)
        timings["hyper_ladder"].append(ms)
        _, ms = timed(run_hyper_rescue, runner, case, cache=cache)
        timings["hyper_rescue"].append(ms)
        if (i + 1) % 25 == 0:
            print(f"[{dataset}] {i + 1}/{len(cases)}", flush=True)

    out: dict[str, object] = {
        "label": label,
        "dataset": dataset,
        "queries": len(cases),
        "load_seconds": round(load_s, 2),
        "index_build_seconds": round(index_build_s, 2),
    }
    for lane, values in timings.items():
        out[lane] = {
            "median_ms": round(statistics.median(values), 2),
            "mean_ms": round(statistics.mean(values), 2),
            "p95_ms": round(percentile(values, 95), 2),
            "max_ms": round(max(values), 2),
        }
        print(f"[{dataset}] {lane}: {out[lane]}", flush=True)
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label, datasets = args[0], args[1:]
    limit: int | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--limit"):
            limit = int(arg.split("=", 1)[1])
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    for dataset in datasets:
        result = profile_dataset(label, dataset, limit)
        path = results_dir / f"{label}-{dataset}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
