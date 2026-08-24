# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE rescue retrieval per query as a replayable trace — the experiment's only I/O.

The S5b question ("does graph traversal help, or is it only inert at ``k=10``?") is a question
about *rankings*, and a ranking metric needs three things per query: the gold chunk ids, the base
ranking, and the graph lane's own ranking. Everything the experiment asks — the oracle ceiling,
a larger ``k``, a larger ``select_pool_k``, an RRF weight sweep over the graph lane — is a pure
function of those three lists. So the expensive half runs **once** per (dataset, ``pool_k``) and
writes a JSONL trace; :mod:`analyse` then answers every arm offline in milliseconds.

Two things this captures that :meth:`GraphRescueRetrieval.retrieve` does not hand back:

* **the RAW rescue ranking** — ``retrieval.union``'s appendix is rescue *minus* base, because
  :func:`_append_and_dedupe` drops a rescue candidate the base already held. That dedupe is
  correct for the shipped append policy and fatal for a fusion arm, which needs the graph lane's
  opinion about candidates the base also found. :class:`TracingRescue` keeps the pre-dedupe list.
* **the per-lane split** — traversal candidates and hypergraph candidates arrive concatenated;
  the trace records where the boundary falls so an arm can weigh the two lanes separately.

Determinism is the matrix's contract, asserted here through the matrix's own function: a trace
written by an unpinned process would answer a different question than the cells it is compared to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5.datasets import DATASETS  # noqa: E402
from benchmark.s5.embed import GraniteQueryEmbedder  # noqa: E402
from benchmark.s5.legs import build_runner  # noqa: E402
from benchmark.s5.rescue import ExportDecorator, HyperedgeChunks, bundle_for  # noqa: E402
from benchmark.s5.run_matrix import (  # noqa: E402
    Plan,
    measure_resolution,
    require_deterministic,  # noqa: E402
)
from latence_retrieval.rescue import GraphRescueRetrieval  # noqa: E402

HERE = Path(__file__).resolve().parent


class TracingRescue(GraphRescueRetrieval):
    """The shipped pipeline, plus the two graph lanes BEFORE they are fused and deduped.

    Three private hooks, each of which only records what it already returns: the retrieval handed
    back is byte-identical to the shipped component's, so a trace and a matrix cell describe the
    same run. Two things the public ``RescueRetrieval`` cannot give an experiment, and both are
    needed here:

    * ``last_rescue`` — the fused rescue list *before* :func:`_append_and_dedupe` drops whatever
      the base already held. A fusion arm needs the graph lane's opinion about candidates the base
      also found; the appendix has, by construction, thrown exactly those away.
    * ``last_traversal`` / ``last_hyper`` — the two lanes separately, so an arm can weigh
      traversal against hypergraph instead of only against the base.

    Single-threaded by construction (the probe loop is a ``for``), which is what makes stashing
    state on ``self`` safe here and nowhere else.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.last_rescue: list = []
        self.last_traversal: list = []
        self.last_hyper: list = []

    def _traversal(self, seeds):  # noqa: ANN001, ANN202
        out = super()._traversal(seeds)
        self.last_traversal = list(out)
        return out

    def _hypergraph(self, question: str, seeds):  # noqa: ANN001, ANN202
        out = super()._hypergraph(question, seeds)
        self.last_hyper = list(out[0])
        return out

    def _rescue(self, question: str, seeds):  # noqa: ANN001, ANN202
        out = super()._rescue(question, seeds)
        self.last_rescue = list(out[0])
        return out


def build_tracing_pipeline(runner, export, *, pool_k: int, mode: str = "auto") -> TracingRescue:
    """:func:`benchmark.s5.rescue.build_pipeline`, with ``pool_k`` overridden and tracing on."""
    config: dict = {"graph_mode": mode, "selector": "none", "pool_k": pool_k}
    if runner.kv_path is not None:
        config["hyperedge_kv_path"] = str(runner.kv_path)
    return TracingRescue(
        config,
        dense=runner.dense,
        bm25=runner.bm25,
        graph=runner.graph,
        hyperedge_documents=None if runner.kv_path is None else HyperedgeChunks(runner),
        decorate=ExportDecorator(export),
    )


def trace_path(dataset: str, pool_k: int, limit: int | None) -> Path:
    """One trace file per (dataset, ``pool_k``, sample) — resumable, and never silently mixed."""
    tag = "full" if limit is None else f"n{limit}"
    return HERE / "traces" / f"{dataset}__pool{pool_k}__{tag}.jsonl"


def run(dataset: str, *, pool_k: int, limit: int | None, force: bool) -> Path:
    """Write (or reuse) the trace for one (dataset, ``pool_k``) cell."""
    require_deterministic()
    out = trace_path(dataset, pool_k, limit)
    if out.is_file() and not force:
        print(f"[{dataset}/pool{pool_k}] trace exists, skipping ({out.name})", flush=True)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    # ``out`` is THIS ticket, never the campaign stage: ``measure_resolution`` writes a
    # ``resolution/<dataset>.json`` under it, and the campaign's copy is committed evidence.
    plan = Plan(
        datasets=(dataset,),
        legs=("graph_rescue",),
        limit=limit,
        out=HERE,
        force=False,
    )
    _resolution, resolved, export = measure_resolution(
        dataset, plan, with_embeddings=True, with_graph_features=True
    )
    usable = [r for r in resolved if r.is_usable]
    print(f"[{dataset}] embedding {len(usable)} queries …", flush=True)
    vectors = GraniteQueryEmbedder().encode([r.question for r in usable])
    runner = build_runner(export, ["graph_rescue"])
    pipeline = build_tracing_pipeline(runner, export, pool_k=pool_k)

    started = time.perf_counter()
    with out.with_suffix(".partial").open("w", encoding="utf-8") as handle:
        for position, item in enumerate(usable):
            bundle = bundle_for(runner, item.question, vectors[position])
            query_started = time.perf_counter()
            retrieval = pipeline.retrieve(bundle)
            elapsed = (time.perf_counter() - query_started) * 1000.0
            handle.write(
                json.dumps(
                    {
                        "query_id": item.query_id,
                        "gold": sorted(item.chunk_ids),
                        "base": [c.id for c in retrieval.base],
                        "rescue": [c.id for c in pipeline.last_rescue],
                        "traversal": [c.id for c in pipeline.last_traversal],
                        "hyper": [c.id for c in pipeline.last_hyper],
                        "appendix": [c.id for c in retrieval.union if c.metadata.get("rescued")],
                        "gate_fired": retrieval.gate.fired,
                        "retrieve_ms": round(elapsed, 3),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if (position + 1) % 250 == 0:
                rate = (time.perf_counter() - started) / (position + 1)
                left = rate * (len(usable) - position - 1)
                print(
                    f"[{dataset}/pool{pool_k}] {position + 1}/{len(usable)} "
                    f"({rate * 1000:.0f} ms/query, ~{left / 60:.1f} min left)",
                    flush=True,
                )
            pipeline.last_rescue = []
            pipeline.last_traversal = []
            pipeline.last_hyper = []
    out.with_suffix(".partial").replace(out)
    print(
        f"[{dataset}/pool{pool_k}] wrote {out.name} "
        f"({len(usable)} queries, {(time.perf_counter() - started) / 60:.1f} min)",
        flush=True,
    )
    return out


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--pool-k", default="100")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    for name in names:
        for pool_k in (int(v) for v in str(args.pool_k).split(",")):
            run(name, pool_k=pool_k, limit=args.limit, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
