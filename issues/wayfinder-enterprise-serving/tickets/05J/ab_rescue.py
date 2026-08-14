# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Wave 5-J's A/B oracle: the S5b bench pipeline vs the SHIPPED component, query by query.

The gate that licenses deleting ``benchmark/s5/rescue.py``'s duplicate pipeline (~600 lockstep
lines) in favour of ``latence_retrieval.rescue.GraphRescueRetrieval``. Both sides run over the
**same** :class:`~benchmark.s5.legs.Runner` — the same dense matrix, the same BM25 index, the same
``GraphRetriever``, the same hyperedge store — so the only variable is which pipeline code drove
them, and every comparison is ``==`` on full values rather than on a recall summary:

* **retrieval** — the whole candidate ``union``: ``(id, score, metadata)`` per candidate, so the
  ``rescued`` stamp, the four denormalised KG columns AND the precomputed ``redundancy`` row are
  all in the comparison (a diff in any of them is a diff);
* **the base** — ``(id, score)`` per candidate (the shipped base additionally carries the
  ``redundancy`` row the single-decorator seam computes ahead of the expander; it is overwritten
  on the union, and the union comparison is what proves that);
* **the gate** — every field of the ``GateDecision``, not just ``fired``;
* **the hypergraph signals** — ρ (ids, order, floats), fields, anchors, vertex confidences and
  both stage counters: what the ``coe`` selector spends its budget on;
* **selection** — for all three legs at **every** point of :data:`~benchmark.s5.rescue.BUDGET_CURVE`
  (the ranking leg once, since it has no selector): the selected id list, the packed-token count
  and the reported budget.

The three hazards the exploration named are each observable here rather than assumed away:
``_coe_select``'s budget threading (every curve point, both selectors), the module-level
``HeuristicTokenCounter`` vs the bench's per-call one (``tokens_packed`` is compared, not
recomputed), and redundancy-metadata population (the union comparison includes the full map).

Deterministic and resumable exactly like the ticket-01 parity harness: ``PYTHONHASHSEED=0`` is
asserted, query vectors come from the checkpointed embeddings ``benchmark/serving/cache`` already
holds (no embedder unless the checkpoint is missing), and ``--skip``/``--shard`` resume an
interrupted run. Any diff is printed with its query id and counted; a non-zero count exits 1.

Usage::

    PYTHONHASHSEED=0 .venv/bin/python .../05J/ab_rescue.py multihop_rag [--limit N] [--skip N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:  # pragma: no cover - script wiring
    sys.path.insert(0, str(REPO))

from benchmark.s5 import goldmap  # noqa: E402
from benchmark.s5.corpus import hyperedge_chunk_map, load_export  # noqa: E402
from benchmark.s5.datasets import DATASETS  # noqa: E402
from benchmark.s5.legs import Runner, build_runner  # noqa: E402
from benchmark.s5.rescue import (  # noqa: E402
    BUDGET_CURVE,
    ExportDecorator,
    RescuePipeline,
)
from benchmark.s5.run_matrix import sample  # noqa: E402
from latence_retrieval.orchestrator import QueryBundle  # noqa: E402
from latence_retrieval.rescue import GraphRescueRetrieval  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from benchmark.s5.corpus import Export
    from benchmark.s5.rescue import GateDecision
    from latence_retrieval.contracts import Candidate
    from latence_retrieval.hypergraph import HypergraphSignals

#: Where the serving-parity harness checkpoints its query embeddings. Reused verbatim (same
#: digest scheme) so a full-set run costs no GPU: the vectors for both full gold sets are already
#: there, and a miss re-embeds rather than silently ranking with the wrong rows.
CACHE_DIR = REPO / "benchmark" / "serving" / "cache"
OUT_DIR = Path(__file__).resolve().parent / "results"

#: The bench leg names and the shipped selector each one dispatches to.
LEGS = {"graph_rescue": "none", "graph_rescue_packed": "packed", "graph_rescue_coe": "coe"}

#: The rescue family's legs, which is what :func:`~benchmark.s5.legs.build_runner` needs to build
#: every component the pipeline uses (dense, BM25, traversal + seeder, hyperedge store).
RESCUE_LEGS = tuple(LEGS)


class HyperedgeChunks:
    """The shipped ``HyperedgeDocumentSource`` over the runner's own hyperedge→chunk mapping.

    Delegates to :meth:`~benchmark.s5.legs.Runner.chunks_from_hyperedges` — reused read-only, not
    re-implemented, so the shipped composition resolves Hyperedges byte-identically to the bench.
    """

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    def candidates(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        """The chunk candidates carrying ``hyperedge_ids``, best-first, deduplicated."""
        return self._runner.chunks_from_hyperedges(hyperedge_ids)


def shipped_pipeline(runner: Runner, export: Export) -> GraphRescueRetrieval:
    """The SHIPPED component over the bench's engines, in the bench's configuration.

    Every knob left at its default is a knob whose default IS the bench constant (pool_k 100,
    seed_docs_k 5, seed_node_cap 32, rescue_cap 50, rank_score_k 60, the expander weights, the
    gate thresholds, ``similarity_key="redundancy"``, ``fill="backfill"``) — that lockstep is
    what this harness exists to prove. The selector is configured as ``packed``; every leg's
    selection is taken through :meth:`GraphRescueRetrieval.select`'s per-call override.
    """
    assert runner.kv_path is not None, "the rescue family requires the hyperedge store"
    return GraphRescueRetrieval(
        {"graph_mode": "auto", "selector": "packed", "hyperedge_kv_path": str(runner.kv_path)},
        dense=runner.dense,
        bm25=runner.bm25,
        graph=runner.graph,
        hyperedge_documents=HyperedgeChunks(runner),
        decorate=ExportDecorator(export),
    )


def query_vectors(dataset: str, usable: Sequence[goldmap.ResolvedGold]) -> Any:  # noqa: ANN401
    """The checkpointed granite query vectors for ``usable`` (embedding only on a cache miss)."""
    digest = hashlib.sha256("\n".join(r.query_id for r in usable).encode("utf-8")).hexdigest()
    path = CACHE_DIR / f"{dataset}-queries-{digest[:16]}.npy"
    if path.is_file():
        vectors = np.load(path)
        if vectors.shape[0] == len(usable):
            return vectors
    from benchmark.s5.embed import GraniteQueryEmbedder

    print(f"[{dataset}] embedding {len(usable)} queries (checkpoint miss) …", flush=True)
    vectors = GraniteQueryEmbedder().encode([r.question for r in usable])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


# -- the comparison values (full, not summarised) ------------------------------------------------


def candidates_full(candidates: Sequence[Candidate]) -> list[tuple[str, float, dict[str, Any]]]:
    """``(id, score, metadata)`` per candidate — the strictest view of a ranked list."""
    return [(c.id, c.score, dict(c.metadata)) for c in candidates]


def candidates_ranked(candidates: Sequence[Candidate]) -> list[tuple[str, float]]:
    """``(id, score)`` per candidate — the ranking without the metadata."""
    return [(c.id, c.score) for c in candidates]


def gate_tuple(gate: GateDecision | Any) -> tuple[object, ...]:
    """Every field of a gate decision, dataclass or model alike."""
    return (
        gate.fired,
        tuple(gate.reasons),
        gate.vetoed_by,
        gate.agreement,
        gate.confidence,
        gate.query_seeds,
    )


def signals_tuple(signals: HypergraphSignals | None) -> tuple[object, ...] | None:
    """ρ, fields, anchors, vertex confidences and both stage counters (ticket-01's own view)."""
    if signals is None:
        return None
    return (
        signals.rho,
        signals.fields,
        signals.anchors,
        signals.vertex_confidence,
        signals.expanded_candidates,
        signals.pruned_candidates,
    )


def compare_query(
    question: str,
    vector: Any,  # noqa: ANN401 - a numpy row
    bench: RescuePipeline,
    shipped: GraphRescueRetrieval,
    seeds: tuple[str, ...],
) -> list[str]:
    """Every comparison for one query; returns the names of the lanes that differed (empty = 0)."""
    bench_retrieved = bench.retrieve(question, vector)
    shipped_retrieval = shipped.retrieve(
        QueryBundle(
            text=question,
            vector=tuple(float(x) for x in vector),
            seed_node_ids=seeds or None,
        )
    )

    diffs: list[str] = []
    if gate_tuple(bench_retrieved.gate) != gate_tuple(shipped_retrieval.gate):
        diffs.append("gate")
    if bench_retrieved.rescued != shipped_retrieval.rescued:
        diffs.append("rescued")
    if candidates_ranked(bench_retrieved.base) != candidates_ranked(shipped_retrieval.base):
        diffs.append("base")
    if candidates_full(bench_retrieved.union) != candidates_full(shipped_retrieval.union):
        diffs.append("union")
    if signals_tuple(bench_retrieved.signals) != signals_tuple(shipped_retrieval.signals):
        diffs.append("signals")

    for leg, selector in LEGS.items():
        # The ranking leg has no selector, so one comparison is the whole of it; repeating it per
        # budget would imply a measurement that was not made.
        budgets = (BUDGET_CURVE[0],) if selector == "none" else BUDGET_CURVE
        for budget in budgets:
            outcome = bench.select(question, bench_retrieved, leg, budget)
            result = shipped.select(question, shipped_retrieval, selector=selector, budget=budget)
            observed = (
                [c.id for c in result.candidates],
                result.tokens_packed,
                result.budget,
            )
            if (outcome.ranked, outcome.tokens_packed, outcome.budget) != observed:
                diffs.append(f"{leg}@{budget}")
            if gate_tuple(outcome.gate) != gate_tuple(result.gate):
                diffs.append(f"{leg}@{budget}:gate")
    return diffs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI (dataset, sampling, resume, output directory)."""
    parser = argparse.ArgumentParser(description="Wave 5-J bench-vs-shipped rescue A/B")
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--limit", type=int, default=None, help="fixed-seed query sample")
    parser.add_argument("--skip", type=int, default=0, help="resume at this query position")
    parser.add_argument("--shard", default=None, help="i/n — every n-th query, offset i")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the A/B for one dataset; write the result JSON and return 1 on any diff."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("Run with PYTHONHASHSEED=0 (the S5 determinism contract).", file=sys.stderr)
        return 2
    args = parse_args(argv)
    dataset = str(args.dataset)
    spec = DATASETS[dataset]

    queries = sample(spec.queries(), args.limit)
    page_docs = frozenset(
        Path(ref.file_name).stem
        for query in queries
        for ref in query.refs
        if ref.kind is goldmap.RefKind.PAGE
    )
    print(f"[{dataset}] loading export ({len(queries)} queries) …", flush=True)
    export = load_export(
        spec.export_dir,
        dataset,
        s2_root=spec.s2_root,
        with_embeddings=True,
        page_docs=page_docs,
        with_graph_features=True,
    )
    usable = [r for r in (goldmap.resolve(q, export) for q in queries) if r.is_usable]
    print(f"[{dataset}] {len(usable)}/{len(queries)} queries usable", flush=True)
    vectors = query_vectors(dataset, usable)

    runner = build_runner(export, RESCUE_LEGS)
    hyperedges = hyperedge_chunk_map(export.root / "graph-hyperedges.parquet")
    assert runner.hyperedge_chunks == hyperedges or runner.hyperedge_chunks, (
        "the runner must carry the hyperedge→chunk mapping the rescue lane resolves through"
    )
    bench = RescuePipeline(runner=runner)
    shipped = shipped_pipeline(runner, export)

    positions = list(range(args.skip, len(usable)))
    if args.shard is not None:
        part, total = (int(x) for x in str(args.shard).split("/"))
        positions = positions[part::total]
        print(f"[{dataset}] shard {part}/{total}: {len(positions)} queries", flush=True)

    diffs = 0
    compared = 0
    started = time.perf_counter()
    for done, position in enumerate(positions, start=1):
        item = usable[position]
        seeds = tuple(runner.seeder.seeds(item.question)) if runner.seeder is not None else ()
        lanes = compare_query(item.question, vectors[position], bench, shipped, seeds)
        if lanes:
            diffs += 1
            print(f"DIFF {item.query_id}: {lanes}", flush=True)
        else:
            compared += 1
        if done % 100 == 0 or done == len(positions):
            elapsed = time.perf_counter() - started
            print(
                f"[{dataset}] {done}/{len(positions)} compared ({elapsed:.0f}s, {diffs} diffs)",
                flush=True,
            )

    out: dict[str, Any] = {
        "dataset": dataset,
        "queries": len(positions),
        "usable_total": len(usable),
        "identical": compared,
        "diffs": diffs,
        "limit": args.limit,
        "skip": args.skip,
        "shard": args.shard,
        "legs": sorted(LEGS),
        "budget_curve": list(BUDGET_CURVE),
        "selections_per_query": 1 + 2 * len(BUDGET_CURVE),
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.shard is None else "-shard" + str(args.shard).replace("/", "of")
    if args.limit is not None:
        suffix += f"-limit{args.limit}"
    (args.out / f"ab-{dataset}{suffix}.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out), flush=True)
    return 1 if diffs else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
