# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Ticket-01 harness: per-lane latency profiling + byte-identical parity over real S5 exports.

Read-only consumer of :mod:`benchmark.s5` (never modifies benchmark behavior): it loads the same
exports, builds the same :class:`~benchmark.s5.legs.Runner`, and drives the SHIPPED
``latence_retrieval`` graph/hypergraph lanes exactly the way ``legs.py``/``rescue.py`` do.

Three lanes, matching how the S5 matrix pays for the graph:

* ``kg`` — alias seeding + :class:`GraphRetriever` traversal (the ladder's leg-3 list and the
  rescue family's traversal lane).
* ``hyper_ladder`` — ``HyperedgeKV.open`` + ``hypergraph_signals(alias_surfaces=...)`` + chunk
  resolution + CoE selection (the ladder's leg-4/5 path, per-query open included).
* ``hyper_rescue`` — the rescue family's hypergraph entry: alias surfaces AND tier-0
  ``ann_node_ids`` seeded from the query's aliases plus the top BM25 documents' entities
  (dense is ticket 02's lane and stays out of this profile; the seed set is computed once per
  query OUTSIDE the timed region and shared byte-identically by every code path under test).

Sampling is deterministic (fixed stride over the loaded query order); parity runs take the full
query set.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:  # pragma: no cover - script wiring
    sys.path.insert(0, str(REPO))

from benchmark.s5.corpus import load_export  # noqa: E402
from benchmark.s5.datasets import DATASETS  # noqa: E402
from benchmark.s5.legs import POOL_K, Runner, build_runner, query_ngrams  # noqa: E402
from benchmark.s5.rescue import SEED_DOCS_K, SEED_NODE_CAP  # noqa: E402
from latence_retrieval.coe import select as coe_select  # noqa: E402
from latence_retrieval.contracts import Capability, RetrievalQuery  # noqa: E402
from latence_retrieval.hypergraph import (  # noqa: E402
    HyperedgeKV,
    HypergraphSignals,
    hypergraph_signals,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.s5.goldmap import GoldQuery

COE_BUDGET = 4000
LANES = ("kg", "hyper_ladder", "hyper_rescue")


@dataclass
class LaneCase:
    """One query's precomputed lane inputs — identical for every code path under test."""

    query_id: str
    question: str
    ngrams: list[str]
    seeds: tuple[str, ...]
    rescue_seeds: tuple[str, ...]


def load_runner(dataset: str) -> Runner:
    """The S5 runner for ``dataset`` with every graph-lane component (no dense matrix)."""
    spec = DATASETS[dataset]
    export = load_export(spec.export_dir, dataset, with_embeddings=False)
    return build_runner(export, ["kg", "hyperedges"])


def lane_cases(runner: Runner, dataset: str, limit: int | None = None) -> list[LaneCase]:
    """Deterministic query cases: full set, or an evenly-strided sample of ``limit``."""
    spec = DATASETS[dataset]
    queries: list[GoldQuery] = [q for q in spec.queries() if q.exclusion is None]
    if limit is not None and len(queries) > limit:
        stride = len(queries) // limit
        queries = [queries[i * stride] for i in range(limit)]
    cases = []
    for q in queries:
        seeds = tuple(runner.seeder.seeds(q.question)) if runner.seeder else ()
        cases.append(
            LaneCase(
                query_id=q.query_id,
                question=q.question,
                ngrams=query_ngrams(q.question),
                seeds=seeds,
                rescue_seeds=_rescue_seeds(runner, q.question, seeds),
            )
        )
    return cases


def _rescue_seeds(runner: Runner, question: str, query_seeds: tuple[str, ...]) -> tuple[str, ...]:
    """The rescue lane's seed set: query aliases first, then top-BM25 documents' entities.

    Mirrors ``GraphRescueRetrieval._graph_seeds`` (the shipped pipeline the S5b rescue family
    runs; :data:`~benchmark.s5.rescue.SEED_DOCS_K` / :data:`~benchmark.s5.rescue.SEED_NODE_CAP`
    are its defaults, pinned there) with the BM25 ranking as the first stage — dense is
    deliberately absent here, see the module docstring.
    """
    seeds: list[str] = list(dict.fromkeys(query_seeds))
    seen = set(seeds)
    if runner.bm25 is not None:
        base = runner.bm25.search(
            RetrievalQuery(mode=Capability.BM25, text=question, top_k=POOL_K)
        )
        rows = runner.export.rows
        for candidate in base[:SEED_DOCS_K]:
            position = runner.export.index_of(candidate.id)
            if position is None:
                continue
            for node in rows[position].node_ids:
                if node not in seen:
                    seen.add(node)
                    seeds.append(node)
    return tuple(seeds[:SEED_NODE_CAP])


# -- the timed lane bodies (exactly what legs.py / rescue.py pay per query) ---------------------


def open_kv(path: str, *, cache: bool = True) -> HyperedgeKV:
    """Open the store, passing ``cache`` through when the reader supports it (AFTER code).

    The BEFORE measurement runs against the pre-ticket reader, whose ``open`` takes no
    ``cache`` keyword — falling back keeps one harness usable on both sides of the change.
    """
    try:
        return HyperedgeKV.open(path, cache=cache)
    except TypeError:
        return HyperedgeKV.open(path)


def run_kg(runner: Runner, case: LaneCase) -> list[tuple[str, float]]:
    """The traversal lane: seeding is part of the lane cost, exactly as in the ladder."""
    if runner.graph is None or runner.seeder is None:
        return []
    seeds = runner.seeder.seeds(case.question)
    if not seeds:
        return []
    hits = runner.graph.search(
        RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=seeds, top_k=POOL_K)
    )
    return [(c.id, c.score) for c in hits]


def run_hyper_ladder(
    runner: Runner, case: LaneCase, *, cache: bool = True
) -> tuple[list[tuple[str, float]], list[str], HypergraphSignals]:
    """The ladder's hyperedges+coe legs for one query (per-query store open included)."""
    assert runner.kv_path is not None
    with open_kv(str(runner.kv_path), cache=cache) as kv:
        signals = hypergraph_signals(kv, alias_surfaces=case.ngrams)
        hyper = runner.chunks_from_hyperedges(
            sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
        )
        coe: list[str] = []
        if signals.rho:
            coe = [c.id for c in runner.chunks_from_hyperedges(
                coe_select(signals, budget=COE_BUDGET).selected
            )]
    return ([(c.id, c.score) for c in hyper], coe, signals)


def run_hyper_rescue(
    runner: Runner, case: LaneCase, *, cache: bool = True
) -> tuple[list[tuple[str, float]], HypergraphSignals]:
    """The rescue family's hypergraph lane for one query (alias + tier-0 entry)."""
    assert runner.kv_path is not None
    with open_kv(str(runner.kv_path), cache=cache) as kv:
        signals = hypergraph_signals(
            kv, alias_surfaces=case.ngrams, ann_node_ids=list(case.rescue_seeds)
        )
        ordered = sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
        hyper = runner.chunks_from_hyperedges(ordered)
    return ([(c.id, c.score) for c in hyper], signals)


def timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """``(result, milliseconds)`` for one call."""
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - started) * 1000.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` (q in [0, 100])."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, int(round(q / 100.0 * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]
