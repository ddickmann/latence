# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Wave 5-J's A/B oracle for the S5b rescue pipeline: two ways of getting the same answer.

**Before the deletion** this harness ran the bench's own ``RescuePipeline`` against the SHIPPED
:class:`~latence_retrieval.rescue.GraphRescueRetrieval` over the same
:class:`~benchmark.s5.legs.Runner`, query by query, on the full gold sets — the gate that
licensed deleting the bench copy's ~600 lockstep lines. The bench copy is gone, so the
harness now pins the equation the deletion *relies* on, on the same real data:

* the **split path** — one :meth:`~latence_retrieval.rescue.GraphRescueRetrieval.retrieve` per
  query, then a :meth:`~latence_retrieval.rescue.GraphRescueRetrieval.select` at every operating
  point (exactly what ``run_matrix._rescue_cell`` does), against
* the **reference path** — a pipeline *configured* at that operating point, run end-to-end for
  that query alone (:data:`REFERENCE_POINTS`), which is what a budget curve would have cost
  before the split and what its numbers must still equal.

Both sides run over the **same** :class:`~benchmark.s5.legs.Runner` — the same dense matrix, the
same BM25 index, the same ``GraphRetriever``, the same hyperedge store — so the only variable is
how the pipeline was driven, and every comparison is ``==`` on full values rather than on a
recall summary:

* **the whole union** — the ``graph_rescue`` reference point has no selector, so its result *is*
  the reference pipeline's entire union: ``(id, score, metadata)`` per candidate, which puts the
  ``rescued`` stamp, the four denormalised KG columns AND the precomputed ``redundancy`` row
  inside the comparison (a diff in any of them is a diff);
* **selection** — the two selectors at their headline budgets and at the curve's extremes: the
  whole :class:`~latence_retrieval.rescue.RescueResult` (candidates with scores and metadata,
  gate, rescued count, packed tokens, budget), compared field by field *and* with ``==`` on the
  models themselves. The hypergraph signals are in the comparison through what consumes them —
  a ``coe`` selection is a function of ρ, so equal CoE selections at four budgets over the same
  union is what a signals diff would have to survive.

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
from benchmark.s5.corpus import load_export  # noqa: E402
from benchmark.s5.datasets import DATASETS  # noqa: E402
from benchmark.s5.legs import Runner, build_runner  # noqa: E402
from benchmark.s5.rescue import (  # noqa: E402
    BUDGET_CURVE,
    ExportDecorator,
    HyperedgeChunks,
    build_pipeline,
    bundle_for,
    headline_budget,
    selector_for,
)
from benchmark.s5.run_matrix import sample  # noqa: E402
from latence_retrieval.rescue import GraphRescueRetrieval, RescueResult  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from benchmark.s5.corpus import Export
    from latence_retrieval.contracts import Candidate
    from latence_retrieval.orchestrator import QueryBundle
    from latence_retrieval.rescue import GateDecision, RescueRetrieval

#: Where the serving-parity harness checkpoints its query embeddings. Reused verbatim (same
#: digest scheme) so a full-set run costs no GPU: the vectors for both full gold sets are already
#: there, and a miss re-embeds rather than silently ranking with the wrong rows.
CACHE_DIR = REPO / "benchmark" / "serving" / "cache"
OUT_DIR = Path(__file__).resolve().parent / "results"

#: The rescue family's legs, which is what :func:`~benchmark.s5.legs.build_runner` needs to build
#: every component the pipeline uses (dense, BM25, traversal + seeder, hyperedge store).
RESCUE_LEGS = ("graph_rescue", "graph_rescue_packed", "graph_rescue_coe")

#: The operating points a whole configured pipeline is run end-to-end at, per query — each one's
#: result must equal the split path's selection at the same point. Both headlines are here (the
#: numbers a cell reports) plus the two extreme curve points, because a budget that is NOT the
#: pipeline's configured default is exactly where a threading defect would hide.
REFERENCE_POINTS = (
    ("graph_rescue", None),
    ("graph_rescue_packed", 10000),
    ("graph_rescue_packed", 2000),
    ("graph_rescue_coe", 16000),
    ("graph_rescue_coe", 32000),
)


def reference_pipelines(
    runner: Runner, export: Export
) -> dict[tuple[str, int | None], GraphRescueRetrieval]:
    """One pipeline per :data:`REFERENCE_POINTS` entry, each CONFIGURED at that operating point.

    Configured, not overridden: this is the pre-split shape — a pipeline built for one selector
    at one budget and run end to end — so nothing it computes can be inherited from the split
    path's retrieval. Same runner, same engines, same decoration, so the only variable is how the
    pipeline was driven.
    """
    assert runner.kv_path is not None, "the rescue family requires the hyperedge store"
    pipelines: dict[tuple[str, int | None], GraphRescueRetrieval] = {}
    for leg, budget in REFERENCE_POINTS:
        selector = selector_for(leg)
        pipeline = build_pipeline(runner, export, selector=selector)
        if budget is not None:
            # `build_pipeline` leaves every measured default in place; a non-default budget is
            # the one thing a reference point may pin, so it is configured explicitly here.
            pipeline = GraphRescueRetrieval(
                {
                    "graph_mode": "auto",
                    "selector": selector,
                    "budget": budget,
                    "hyperedge_kv_path": str(runner.kv_path),
                },
                dense=runner.dense,
                bm25=runner.bm25,
                graph=runner.graph,
                hyperedge_documents=HyperedgeChunks(runner),
                decorate=ExportDecorator(export),
            )
        pipelines[(leg, budget)] = pipeline
    return pipelines


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


def selections(
    pipeline: GraphRescueRetrieval, question: str, retrieval: RescueRetrieval
) -> dict[tuple[str, int], RescueResult]:
    """The split path: every leg at every curve point, all from ONE retrieval.

    The ranking leg has no selector, so it is selected once — repeating one ranking under seven
    budget labels would imply a measurement that was not made.
    """
    out: dict[tuple[str, int], RescueResult] = {}
    for leg in RESCUE_LEGS:
        selector = selector_for(leg)
        budgets = (headline_budget(leg),) if selector == "none" else BUDGET_CURVE
        for budget in budgets:
            out[(leg, budget)] = pipeline.select(
                question, retrieval, selector=selector, budget=budget
            )
    return out


def compare_query(
    question: str,
    bundle: QueryBundle,
    split: GraphRescueRetrieval,
    references: dict[tuple[str, int | None], GraphRescueRetrieval],
) -> list[str]:
    """Every comparison for one query; returns the names of the lanes that differed (empty = 0)."""
    retrieval = split.retrieve(bundle)
    selected = selections(split, question, retrieval)

    diffs: list[str] = []
    for (leg, budget), pipeline in references.items():
        reference = pipeline.run(bundle)
        point = (leg, headline_budget(leg) if budget is None else budget)
        result = selected[point]
        if candidates_full(reference.candidates) != candidates_full(result.candidates):
            diffs.append(f"{leg}@{budget}")
        if (reference.tokens_packed, reference.budget, reference.rescued) != (
            result.tokens_packed,
            result.budget,
            result.rescued,
        ):
            diffs.append(f"{leg}@{budget}:spend")
        if gate_tuple(reference.gate) != gate_tuple(result.gate):
            diffs.append(f"{leg}@{budget}:gate")
        if reference != result:
            diffs.append(f"{leg}@{budget}:result")
    # The retrieval itself needs no separate comparison: the ``graph_rescue`` reference point is
    # the selector-less leg, so its result IS that pipeline's whole union — candidates, scores,
    # metadata (KG columns + redundancy), gate and rescued count — compared above against the
    # union this query's single retrieval produced.
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
    assert runner.hyperedge_chunks, (
        "the runner must carry the hyperedge→chunk mapping the rescue lane resolves through"
    )
    split = build_pipeline(runner, export)
    references = reference_pipelines(runner, export)

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
        bundle = bundle_for(runner, item.question, vectors[position])
        lanes = compare_query(item.question, bundle, split, references)
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
        "legs": sorted(RESCUE_LEGS),
        "budget_curve": list(BUDGET_CURVE),
        "selections_per_query": 1 + 2 * len(BUDGET_CURVE),
        "reference_points": [[leg, budget] for leg, budget in REFERENCE_POINTS],
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
