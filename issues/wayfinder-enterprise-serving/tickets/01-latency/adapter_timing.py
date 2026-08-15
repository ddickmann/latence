# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""(Wave 1-A) Per-call latency of the SHIPPED parquet graph adapters, cached vs uncached.

FINDINGS F1 measured: a production caller wiring ``GraphRetriever`` over the shipped
``DuckDBGraphSource``/``NetworkxGraphSource`` + ``DuckDBGraphDocumentSource`` pays a per-query
file read (and, post-ticket-01, up to ``max_hops + 1`` full-parquet document scans per query via
the tiered resolution). This harness times ``GraphRetriever.search`` end-to-end per query on a
real export for each engine, ``cache=False`` (the pre-Wave-1A shipped behaviour, preserved
verbatim) vs ``cache=True`` (the file-identity read cache), plus the one-time decode cost the
cached path pays on its first call. Seeding runs OUTSIDE the timed region — the seed set is
byte-identical on both sides and its cost is not the adapters'.

Usage::

    PYTHONHASHSEED=0 .venv/bin/python .../adapter_timing.py <dataset> \\
        [--cached-limit=N] [--uncached-limit=N]
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.s5.legs import POOL_K  # noqa: E402
from harness import LaneCase, lane_cases, load_runner, percentile  # noqa: E402
from latence_retrieval.contracts import Capability, RetrievalQuery  # noqa: E402
from latence_retrieval.graph_retrieval import (  # noqa: E402
    DuckDBGraphDocumentSource,
    GraphRetriever,
    clear_document_source_cache,
)
from latence_retrieval.multihop import (  # noqa: E402
    DuckDBGraphSource,
    NetworkxGraphSource,
    clear_graph_source_caches,
)

_CORPUS_COLUMNS = {"id_column": "record_id", "node_ids_column": "kg_node_ids",
                   "text_column": "content"}


def _retriever(engine: str, root: Path, *, cache: bool) -> GraphRetriever:
    edges = str(root / "graph-edges.parquet")
    corpus = str(root / "corpus.parquet")
    documents = DuckDBGraphDocumentSource(corpus, cache=cache, **_CORPUS_COLUMNS)
    if engine == "duckdb":
        return GraphRetriever(
            graph_source=DuckDBGraphSource(edges, cache=cache), document_source=documents
        )
    return GraphRetriever(
        graph_source=NetworkxGraphSource(edges, cache=cache), document_source=documents
    )


def _lane(retriever: GraphRetriever, cases: list[LaneCase]) -> dict[str, float]:
    timings: list[float] = []
    for case in cases:
        seeds = list(case.seeds)
        if not seeds:
            continue
        query = RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=seeds, top_k=POOL_K)
        started = time.perf_counter()
        retriever.search(query)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "queries": len(timings),
        "median_ms": round(statistics.median(timings), 3) if timings else 0.0,
        "p95_ms": round(percentile(timings, 95), 3) if timings else 0.0,
    }


def main() -> None:
    dataset = sys.argv[1]
    cached_limit, uncached_limit = 200, 60
    for arg in sys.argv[2:]:
        if arg.startswith("--cached-limit"):
            cached_limit = int(arg.split("=", 1)[1])
        if arg.startswith("--uncached-limit"):
            uncached_limit = int(arg.split("=", 1)[1])
    runner = load_runner(dataset)
    root = runner.export.root
    cached_cases = lane_cases(runner, dataset, limit=cached_limit)
    uncached_cases = lane_cases(runner, dataset, limit=uncached_limit)

    engines = ["duckdb"] if importlib.util.find_spec("duckdb") is not None else []
    if importlib.util.find_spec("networkx") is not None and engines:
        engines.append("networkx")  # its document source needs duckdb too

    out: dict[str, object] = {"dataset": dataset}
    for engine in engines:
        clear_graph_source_caches()
        clear_document_source_cache()
        cached = _retriever(engine, root, cache=True)
        warm = next(c for c in cached_cases if c.seeds)
        started = time.perf_counter()
        cached.search(
            RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=list(warm.seeds), top_k=POOL_K)
        )
        build_ms = round((time.perf_counter() - started) * 1000.0, 1)
        out[engine] = {
            "cache_build_first_call_ms": build_ms,
            "cached": _lane(cached, cached_cases),
            "uncached": _lane(_retriever(engine, root, cache=False), uncached_cases),
        }
        print(f"[{dataset}] {engine}: {json.dumps(out[engine])}", flush=True)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"adapter-timing-{dataset}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
