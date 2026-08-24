# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The one arm a ranking trace cannot answer: ``select_pool_k`` for the BUDGET selectors.

(The module is ``select_pool``, not ``selectors``: a script on ``sys.path[0]`` named for a stdlib
module shadows it, and ``socket`` imports ``selectors`` — so the obvious name breaks ``pyarrow``.)

`analyse.py` settles every ranking arm offline, because recall@k of a re-ordering is a function
of ids alone. The ``packed`` and ``coe`` selectors are not re-orderings — they spend a token
budget over a pool and return a *subset*, so what they do depends on candidate texts, token
costs and pairwise redundancy, none of which a trace carries. This runs them for real.

The structural claim under test is narrow and binary before it is quantitative:

    ``select()`` truncates to ``union[:select_pool_k]`` with ``select_pool_k = 100``, and the base
    alone is 100-200 candidates, so **the selector has never once seen a rescued candidate.**

So the first number this reports is not a recall — it is ``rescued_selected``, the count of
selected candidates stamped ``rescued``. At ``select_pool_k = 100`` it must be exactly 0; if
raising the pool does not move it off 0, the knob is not the constraint either.

One retrieval per query serves every (selector × pool) cell: :meth:`GraphRescueRetrieval.select`
is stateless and takes the retrieval as an argument, so a second pipeline configured with a larger
``select_pool_k`` can select over the *identical* retrieval the shipped one produced. That is the
seam ADR-0048 put there, used exactly as intended.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5.datasets import DATASETS  # noqa: E402
from benchmark.s5.embed import GraniteQueryEmbedder  # noqa: E402
from benchmark.s5.legs import build_runner  # noqa: E402
from benchmark.s5.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from benchmark.s5.rescue import (  # noqa: E402
    COE_SELECT_BUDGET,
    SELECT_BUDGET,
    ExportDecorator,
    HyperedgeChunks,
    bundle_for,
)
from benchmark.s5.run_matrix import Plan, measure_resolution, require_deterministic  # noqa: E402
from latence_retrieval.rescue import GraphRescueRetrieval  # noqa: E402

HERE = Path(__file__).resolve().parent

#: The pools compared. ``100`` is shipped; ``250`` clears a 200-candidate base plus the 50-item
#: appendix; ``1000`` is "no truncation at all" — the knob at its limit, so a null result here
#: cannot be answered with "you did not raise it far enough".
POOLS = (100, 250, 1000)

BUDGETS = {"packed": SELECT_BUDGET, "coe": COE_SELECT_BUDGET}


def pipelines(runner, export, pools) -> dict[int, GraphRescueRetrieval]:  # noqa: ANN001
    """One pipeline per pool — identical in every other knob, so the pool is the only variable."""
    out = {}
    for pool in pools:
        config: dict = {"graph_mode": "auto", "selector": "packed", "select_pool_k": pool}
        if runner.kv_path is not None:
            config["hyperedge_kv_path"] = str(runner.kv_path)
        out[pool] = GraphRescueRetrieval(
            config,
            dense=runner.dense,
            bm25=runner.bm25,
            graph=runner.graph,
            hyperedge_documents=None if runner.kv_path is None else HyperedgeChunks(runner),
            decorate=ExportDecorator(export),
        )
    return out


def run(dataset: str, *, limit: int | None) -> dict:
    """Every (selector × pool) cell for one dataset, over one retrieval per query."""
    require_deterministic()
    # Its OWN output root, not the ticket root: this may run concurrently with ``probe.py``
    # over the same dataset at a different ``--limit``, and ``measure_resolution`` writes a
    # ``resolution/<dataset>.json`` whose contents depend on the sample. Two writers, one path,
    # different truths — so they get different paths.
    plan = Plan(
        datasets=(dataset,),
        legs=("graph_rescue",),
        limit=limit,
        out=HERE / "selector-run",
        force=False,
    )
    _resolution, resolved, export = measure_resolution(
        dataset, plan, with_embeddings=True, with_graph_features=True
    )
    usable = [r for r in resolved if r.is_usable]
    print(f"[{dataset}] embedding {len(usable)} queries …", flush=True)
    vectors = GraniteQueryEmbedder().encode([r.question for r in usable])
    runner = build_runner(export, ["graph_rescue"])
    built = pipelines(runner, export, POOLS)
    driver = built[POOLS[0]]

    cells: dict[str, dict[str, list]] = {
        f"{selector}@pool{pool}": {
            "recall@10": [],
            "ndcg@10": [],
            "recall@selected": [],
            "rescued": [],
            "n": [],
        }
        for selector in BUDGETS
        for pool in POOLS
    }
    for position, item in enumerate(usable):
        bundle = bundle_for(runner, item.question, vectors[position])
        retrieval = driver.retrieve(bundle)
        gold = frozenset(item.chunk_ids)
        for selector, budget in BUDGETS.items():
            for pool in POOLS:
                result = built[pool].select(
                    item.question, retrieval, selector=selector, budget=budget
                )
                ranked = [c.id for c in result.candidates]
                cell = cells[f"{selector}@pool{pool}"]
                cell["recall@10"].append(recall_at_k(ranked, gold, 10))
                cell["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
                # The cut that matches what a SELECTOR is for. Its output is the context handed
                # to a model — ~23 (packed) to ~35 (coe) candidates here — so scoring it only at
                # rank 10 asks a budgeted subset to win a ranking contest it is not running.
                # A rescued candidate that lands at rank 15 of a 23-item context is IN the
                # context; recall@10 calls that a miss.
                cell["recall@selected"].append(recall_at_k(ranked, gold, len(ranked)))
                cell["rescued"].append(
                    sum(1 for c in result.candidates if c.metadata.get("rescued"))
                )
                cell["n"].append(len(ranked))
        if (position + 1) % 100 == 0:
            print(f"[{dataset}] {position + 1}/{len(usable)}", flush=True)

    return {
        "dataset": dataset,
        "queries": len(usable),
        "limit": limit,
        "cells": {
            name: {
                "recall@10": round(stats.fmean(cell["recall@10"]), 4),
                "ndcg@10": round(stats.fmean(cell["ndcg@10"]), 4),
                "recall@selected": round(stats.fmean(cell["recall@selected"]), 4),
                "rescued_selected_total": sum(cell["rescued"]),
                "queries_with_rescued_selected": sum(1 for v in cell["rescued"] if v),
                "selected_mean": round(stats.fmean(cell["n"]), 2),
            }
            for name, cell in cells.items()
        },
    }


def main(argv=None) -> int:  # noqa: ANN001
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        payload = run(name, limit=args.limit)
        tag = "full" if args.limit is None else f"n{args.limit}"
        path = out / f"select_pool__{name}__{tag}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload["cells"], indent=2, sort_keys=True), flush=True)
        print(f"[{name}] wrote {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
