# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Byte-identical parity over FULL real query sets: the ticket-01 gate.

For every usable gold query of a dataset, each lane is executed through every code path and the
complete results are compared with ``==`` — ranked chunk ``(id, score)`` lists AND the full
:class:`~latence_retrieval.hypergraph.HypergraphSignals` (ρ ids/order/floats, fields, anchors,
vertex confidences, stage counters). Zero tolerance: any difference is a counted, printed diff.

Paths compared per query:

* ``kg``       — ``GraphRetriever.search`` (tiered, the shipped default) vs the preserved
  exhaustive resolution (``_exhaustive_hits`` + sort/truncate — the pre-ticket path, verbatim).
* ``hyper_*``  — the per-call scan path (``cache=False``, the pre-ticket path, verbatim) vs the
  shared accelerated index (``cache=True``) vs the pure-Python index kernels.

Usage::

    PYTHONHASHSEED=0 .venv/bin/python .../parity.py <dataset> [--limit N]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.s5.legs import POOL_K, Runner  # noqa: E402
from harness import (  # noqa: E402
    COE_BUDGET,
    LaneCase,
    lane_cases,
    load_runner,
    run_hyper_ladder,
    run_hyper_rescue,
    run_kg,
)
from latence_retrieval.coe import select as coe_select  # noqa: E402
from latence_retrieval.contracts import Capability, RetrievalQuery  # noqa: E402
from latence_retrieval.hypergraph import (  # noqa: E402
    HyperedgeKV,
    HypergraphIndex,
    HypergraphSignals,
    hypergraph_signals,
)


def signals_tuple(signals: HypergraphSignals) -> tuple[object, ...]:
    return (
        signals.rho,
        signals.fields,
        signals.anchors,
        signals.vertex_confidence,
        signals.expanded_candidates,
        signals.pruned_candidates,
    )


def kg_exhaustive(runner: Runner, case: LaneCase) -> list[tuple[str, float]]:
    """The pre-ticket kg ranking through the preserved exhaustive path."""
    if runner.graph is None or runner.seeder is None:
        return []
    seeds = runner.seeder.seeds(case.question)
    if not seeds:
        return []
    graph = runner.graph
    query = RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=seeds, top_k=POOL_K)
    distances = dict(graph._resolve_graph().neighbors(list(dict.fromkeys(seeds)), graph._max_hops))
    for seed in seeds:
        distances.setdefault(seed, 0)
    reachable = {node: hop for node, hop in distances.items() if hop <= graph._max_hops}
    if not graph._include_seed_documents:
        reachable = {node: hop for node, hop in reachable.items() if hop > 0}
    if not reachable:
        return []
    hits = graph._exhaustive_hits(query, reachable)
    hits.sort(key=lambda c: (-c.score, c.id))
    return [(c.id, c.score) for c in hits[: query.top_k]]


def main() -> None:
    dataset = sys.argv[1]
    limit: int | None = None
    for arg in sys.argv[2:]:
        if arg.startswith("--limit"):
            limit = int(arg.split("=", 1)[1])
    runner = load_runner(dataset)
    cases = lane_cases(runner, dataset, limit=limit)
    assert runner.kv_path is not None
    with HyperedgeKV.open(str(runner.kv_path), cache=False) as kv:
        fallback_index = HypergraphIndex(kv, accelerated=False)

    def open_fallback() -> HyperedgeKV:
        """An uncached reader carrying the pure-Python-kernel index."""
        assert runner.kv_path is not None
        reader = HyperedgeKV.open(str(runner.kv_path), cache=False)
        reader._index = fallback_index
        return reader

    def fallback_ladder(case: LaneCase) -> tuple[object, ...]:
        """`run_hyper_ladder`'s body over the fallback index (same statements, same order)."""
        with open_fallback() as kv:
            signals = hypergraph_signals(kv, alias_surfaces=case.ngrams)
            hyper = runner.chunks_from_hyperedges(
                sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
            )
            coe: list[str] = []
            if signals.rho:
                coe = [
                    c.id
                    for c in runner.chunks_from_hyperedges(
                        coe_select(signals, budget=COE_BUDGET).selected
                    )
                ]
        return ([(c.id, c.score) for c in hyper], coe, signals)

    def fallback_rescue(case: LaneCase) -> tuple[object, ...]:
        with open_fallback() as kv:
            signals = hypergraph_signals(
                kv, alias_surfaces=case.ngrams, ann_node_ids=list(case.rescue_seeds)
            )
            ordered = sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
            hyper = runner.chunks_from_hyperedges(ordered)
        return ([(c.id, c.score) for c in hyper], signals)

    fallback_runners = {"hyper_ladder": fallback_ladder, "hyper_rescue": fallback_rescue}

    counts = {"queries": len(cases), "kg": 0, "hyper_ladder": 0, "hyper_rescue": 0}
    diffs = 0
    started = time.perf_counter()
    for i, case in enumerate(cases):
        # kg: tiered (shipped) vs exhaustive (pre-ticket)
        tiered = run_kg(runner, case)
        exhaustive = kg_exhaustive(runner, case)
        if tiered != exhaustive:
            diffs += 1
            print(f"KG DIFF {case.query_id}: {tiered[:3]} != {exhaustive[:3]}", flush=True)
        else:
            counts["kg"] += 1

        # hyper lanes: scan (pre-ticket) vs indexed (shipped) vs pure-python kernels
        for lane, fn in (("hyper_ladder", run_hyper_ladder), ("hyper_rescue", run_hyper_rescue)):
            scan = fn(runner, case, cache=False)
            indexed = fn(runner, case, cache=True)
            fallback = fallback_runners[lane](case)
            scan_cmp = tuple(
                signals_tuple(part) if isinstance(part, HypergraphSignals) else part
                for part in scan
            )
            for name, other in (("indexed", indexed), ("fallback", fallback)):
                other_cmp = tuple(
                    signals_tuple(part) if isinstance(part, HypergraphSignals) else part
                    for part in other  # type: ignore[union-attr]
                )
                if scan_cmp != other_cmp:
                    diffs += 1
                    print(f"{lane.upper()} {name} DIFF {case.query_id}", flush=True)
                    break
            else:
                counts[lane] += 1
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"[{dataset}] {i + 1}/{len(cases)} parity-checked ({elapsed:.0f}s)", flush=True)

    out = {"dataset": dataset, "diffs": diffs, **counts}
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"parity-{dataset}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)
    if diffs:
        sys.exit(1)


if __name__ == "__main__":
    main()
