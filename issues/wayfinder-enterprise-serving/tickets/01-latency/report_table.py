# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Render the BEFORE/AFTER latency table (+ speedups) from ``results/*.json``."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
DATASETS = ("multihop_rag", "vidoseek", "ohr_bench", "wiki2multihop", "uda")
LANES = ("kg", "hyper_ladder", "hyper_rescue")


def main() -> None:
    print(
        "| dataset | lane | before median | before p95 | after median | after p95 "
        "| median speedup | p95 speedup |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for dataset in DATASETS:
        before_path = RESULTS / f"before-{dataset}.json"
        after_path = RESULTS / f"after-{dataset}.json"
        if not before_path.is_file() or not after_path.is_file():
            continue
        before = json.loads(before_path.read_text())
        after = json.loads(after_path.read_text())
        for lane in LANES:
            b, a = before[lane], after[lane]
            print(
                f"| {dataset} | {lane} | {b['median_ms']:.1f} ms | {b['p95_ms']:.1f} ms "
                f"| {a['median_ms']:.2f} ms | {a['p95_ms']:.2f} ms "
                f"| **{b['median_ms'] / a['median_ms']:.0f}x** "
                f"| {b['p95_ms'] / a['p95_ms']:.0f}x |"
            )
        print(
            f"| {dataset} | (index build, once/process) | — | — | "
            f"{after['index_build_seconds']:.1f} s | — | — | — |"
        )


if __name__ == "__main__":
    main()
