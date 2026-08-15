# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""``parity.py`` — Qdrant serving-path parity vs the brute-force bench reference (ticket 02).

The question this answers, with live dependencies only: **"deployed with Qdrant, same recall?"**
The S5 bench's dense first stage is an exact brute-force cosine scan
(:class:`benchmark.s5.legs.DenseBackend`); production serves the dense lane through a real Qdrant
via the shipped :class:`latence_retrieval.QdrantBackend`, behind the same ``RetrievalBackend``
seam. This harness runs the S5 gold queries through the **production composition** —
:class:`latence_retrieval.rescue.GraphRescueRetrieval`, the same component an operator deploys —
twice per query:

* **reference**: the exact brute-force dense engine (:class:`KgDenseReference`, the bench's
  ``DenseBackend`` wrapped so its candidates carry the same KG payload columns the Qdrant lane
  serves — the *engine* is the only variable in the A/B, never the candidate contract);
* **served**: ``QdrantBackend`` over a collection ``latence-sink-qdrant`` loaded from the same
  export, with recorded HNSW search params.

Reported per dataset: chunk R@10 / docR@5 / docR@10 for both sides and their deltas, the
dense-lane-only ANN recall (Qdrant's top-k against the exact top-k — the classic ANN accuracy
number, uncontaminated by fusion), rank-list agreement of the full pipeline, gate-decision
agreement, and per-query serving latency (median/p95) for both compositions.

The selector is fixed at ``"none"`` (the ranked union): the selectors are deterministic functions
of the union, so identical unions imply identical selections, while a selector in the loop would
only *mask* first-stage deltas it happens to select around. ``graph_mode`` stays the production
default (``auto``).

Resumable: per-query rows append to a JSONL the moment they are computed, and a re-run resumes at
the first unscored query. Query vectors (the granite embed of up to 12,576 questions) are
checkpointed per dataset. Deterministic: ``PYTHONHASHSEED=0`` is asserted exactly as the S5
harness asserts it.

Usage::

    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.serving.parity \\
        --dataset wiki2multihop --url http://localhost:6333 --limit 200      # iteration sample
    PYTHONHASHSEED=0 .venv/bin/python -m benchmark.serving.parity \\
        --dataset wiki2multihop --url http://localhost:6333 --hnsw-ef 512    # the full set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.s5 import goldmap  # noqa: E402
from benchmark.s5.corpus import hyperedge_chunk_map, load_export, node_alias_map  # noqa: E402
from benchmark.s5.datasets import DATASETS  # noqa: E402
from benchmark.s5.embed import GraniteQueryEmbedder  # noqa: E402
from benchmark.s5.legs import (  # noqa: E402
    POOL_K,
    Bm25Backend,
    ChunkGraphDocumentSource,
    DenseBackend,
    EdgeGraphSource,
    Runner,
)
from benchmark.s5.metrics import documents_of, percentile, recall_at_k  # noqa: E402
from benchmark.s5.run_matrix import SAMPLE_SEED, sample  # noqa: E402
from latence_retrieval import QdrantBackend  # noqa: E402
from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery  # noqa: E402
from latence_retrieval.graph_retrieval import AliasQuerySeeder, GraphRetriever  # noqa: E402
from latence_retrieval.orchestrator import QueryBundle  # noqa: E402
from latence_retrieval.rescue import GraphRescueRetrieval  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.s5.corpus import Export

#: Where per-dataset run artifacts land. Summaries (small JSON) are committed; the per-query JSONL
#: and the query-vector checkpoints are local run state (gitignored).
OUT_DIR = REPO_ROOT / "benchmark" / "serving" / "results"
CACHE_DIR = REPO_ROOT / "benchmark" / "serving" / "cache"

#: The recall cut-offs of the ticket's gate artifact.
CHUNK_K = 10
DOC_KS = (5, 10)


class KgDenseReference:
    """The exact brute-force dense engine, serving the SAME candidate contract as the Qdrant lane.

    Wraps the bench's :class:`~benchmark.s5.legs.DenseBackend` (imported read-only — its maths is
    the reference) and attaches to every hit the four denormalised KG payload columns
    (``kg_node_ids``, ``neighbor_node_ids``, ``kg_node_community``, ``kg_node_centrality``) that
    ``latence-sink-qdrant`` writes into the point payload and ``QdrantBackend`` hands back as
    ``Candidate.metadata``. Without this the A/B would compare *engines and contracts at once*:
    the served lane's candidates would feed the Expander/Packer their KG columns while the
    reference's fed nothing, and every delta would be unattributable.
    """

    def __init__(self, export: Export) -> None:
        self._brute = DenseBackend(export)
        self._export = export

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{DENSE}`` — the same declaration the served lane makes."""
        return frozenset({Capability.DENSE})

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Exact top-``top_k`` by cosine, each candidate carrying its KG payload columns."""
        rows = self._export.rows
        out: list[Candidate] = []
        for candidate in self._brute.search(query):
            position = self._export.index_of(candidate.id)
            if position is None:  # pragma: no cover - the backend ranks only export rows
                out.append(candidate)
                continue
            row = rows[position]
            out.append(
                candidate.model_copy(
                    update={
                        "metadata": {
                            "kg_node_ids": list(row.node_ids),
                            "neighbor_node_ids": list(row.neighbor_node_ids),
                            "kg_node_community": list(row.community_ids),
                            "kg_node_centrality": list(row.centralities),
                        }
                    }
                )
            )
        return out


class HyperedgeChunks:
    """The rescue pipeline's hyperedge→chunk resolver over the export's own sidecar mapping.

    Satisfies :class:`latence_retrieval.rescue.HyperedgeDocumentSource` by delegating to the bench
    :meth:`~benchmark.s5.legs.Runner.chunks_from_hyperedges` — reused read-only rather than
    re-implemented, so the served composition resolves hyperedges byte-identically to the bench.
    """

    def __init__(self, export: Export, mapping: dict[str, str]) -> None:
        self._runner = Runner(
            export=export, dense=None, bm25=None, graph=None, kv_path=None,
            hyperedge_chunks=mapping,
        )

    def candidates(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        """The chunk candidates carrying ``hyperedge_ids``, best-first, deduplicated."""
        return self._runner.chunks_from_hyperedges(hyperedge_ids)


@dataclass(frozen=True)
class Plan:
    """One resolved parity run."""

    dataset: str
    url: str
    collection: str
    limit: int | None
    hnsw_ef: int | None
    exact: bool
    out: Path
    cache: Path
    force: bool

    @property
    def run_name(self) -> str:
        """The file stem carrying every knob that changes the numbers.

        The collection is a knob: two collections over the same export can differ in distance
        metric or index params, and a resume checkpoint keyed without it would silently serve one
        collection's rows as another's numbers.
        """
        knobs = [self.collection]
        knobs.append(f"limit{self.limit}" if self.limit else "full")
        if self.exact:
            knobs.append("exact")
        elif self.hnsw_ef is not None:
            knobs.append(f"ef{self.hnsw_ef}")
        else:
            knobs.append("ef-default")
        return "__".join(knobs)


def parse_args(argv: Sequence[str] | None = None) -> Plan:
    """Parse the CLI into a :class:`Plan`."""
    parser = argparse.ArgumentParser(description="Qdrant serving-path parity vs brute force")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--collection", default=None, help="default: the dataset name")
    parser.add_argument("--limit", type=int, default=None, help="fixed-seed query sample")
    parser.add_argument("--hnsw-ef", type=int, default=None, help="HNSW search beam width")
    parser.add_argument("--exact", action="store_true", help="exact (non-ANN) Qdrant scoring")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--force", action="store_true", help="rescore from query 0")
    args = parser.parse_args(argv)
    return Plan(
        dataset=str(args.dataset),
        url=str(args.url),
        collection=str(args.collection or args.dataset),
        limit=args.limit,
        hnsw_ef=args.hnsw_ef,
        exact=bool(args.exact),
        out=args.out,
        cache=args.cache,
        force=bool(args.force),
    )


def query_vectors(
    plan: Plan, usable: Sequence[goldmap.ResolvedGold]
) -> Any:  # noqa: ANN401 - numpy
    """The granite query vectors for ``usable``, checkpointed per (dataset, query-id set).

    The checkpoint records the sha256 of the query ids it covers, so a changed sample (a different
    ``--limit``, a changed gold mapping) re-embeds rather than silently reusing the wrong rows —
    the same digest discipline ``build_vectors``' shard store uses.
    """
    plan.cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\n".join(r.query_id for r in usable).encode("utf-8")).hexdigest()
    path = plan.cache / f"{plan.dataset}-queries-{digest[:16]}.npy"
    if path.is_file():
        vectors = np.load(path)
        if vectors.shape[0] == len(usable):
            return vectors
    print(f"[{plan.dataset}] embedding {len(usable)} queries …", flush=True)
    vectors = GraniteQueryEmbedder().encode([r.question for r in usable])
    np.save(path, vectors)
    return vectors


def build_pipelines(
    plan: Plan, export: Export
) -> tuple[GraphRescueRetrieval, GraphRescueRetrieval, KgDenseReference, QdrantBackend]:
    """The two production compositions: identical everywhere except the dense engine."""
    if export.embeddings is None:
        msg = f"{export.dataset}: corpus.parquet carries no 'embedding' column; nothing to serve."
        raise ValueError(msg)
    root = export.root
    bm25 = Bm25Backend(export)
    graph = GraphRetriever(
        graph_source=EdgeGraphSource(root / "graph-edges.parquet"),
        document_source=ChunkGraphDocumentSource(export),
    )
    kv_path = root / "graph-hyperedges.kv"
    hyperedges = HyperedgeChunks(export, hyperedge_chunk_map(root / "graph-hyperedges.parquet"))

    reference = KgDenseReference(export)
    search_params: dict[str, Any] | None = None
    if plan.exact:
        search_params = {"exact": True}
    elif plan.hnsw_ef is not None:
        search_params = {"hnsw_ef": plan.hnsw_ef}
    served = QdrantBackend(
        {
            "collection": plan.collection,
            "capabilities": ["dense", "filter"],
            "dimension": int(export.embeddings.shape[1]),
            "url": plan.url,
            **({"search_params": search_params} if search_params else {}),
        }
    )

    config: dict[str, Any] = {"selector": "none", "hyperedge_kv_path": str(kv_path)}
    pipeline_reference = GraphRescueRetrieval(
        config, dense=reference, bm25=bm25, graph=graph, hyperedge_documents=hyperedges
    )
    pipeline_served = GraphRescueRetrieval(
        config, dense=served, bm25=bm25, graph=graph, hyperedge_documents=hyperedges
    )
    return pipeline_reference, pipeline_served, reference, served


def overlap_at_k(reference: Sequence[str], candidate: Sequence[str], k: int) -> float:
    """``|exact top-k ∩ served top-k| / k`` — the standard ANN recall@k against ground truth."""
    if k < 1 or not reference:
        return 0.0
    cut = min(k, len(reference))
    return len(set(reference[:cut]) & set(candidate[:k])) / cut


def score_query(
    item: goldmap.ResolvedGold,
    vector: Any,  # noqa: ANN401 - numpy row
    export: Export,
    pipelines: tuple[GraphRescueRetrieval, GraphRescueRetrieval],
    engines: tuple[KgDenseReference, QdrantBackend],
    seeder: AliasQuerySeeder,
) -> dict[str, Any]:
    """One query through both compositions + both bare dense lanes; every number observed."""
    bundle = QueryBundle(
        text=item.question,
        vector=tuple(float(x) for x in vector),
        seed_node_ids=seeder.seeds(item.question) or None,
    )
    reference_pipe, served_pipe = pipelines
    reference_dense, served_dense = engines

    # The bare dense lanes first — the ANN-accuracy number, uncontaminated by fusion.
    dense_query = RetrievalQuery(mode=Capability.DENSE, vector=bundle.vector, top_k=POOL_K)
    exact_ids = [c.id for c in reference_dense.search(dense_query)]
    started = time.perf_counter()
    served_ids = [c.id for c in served_dense.search(dense_query)]
    dense_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    reference_result = reference_pipe.run(bundle)
    reference_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    served_result = served_pipe.run(bundle)
    served_ms = (time.perf_counter() - started) * 1000.0

    ranked_reference = [c.id for c in reference_result.candidates]
    ranked_served = [c.id for c in served_result.candidates]
    row: dict[str, Any] = {
        "query_id": item.query_id,
        "gold": len(item.chunk_ids),
        "ann_recall@10": overlap_at_k(exact_ids, served_ids, 10),
        "ann_recall@100": overlap_at_k(exact_ids, served_ids, POOL_K),
        "dense_ms": round(dense_ms, 3),
        "reference_ms": round(reference_ms, 3),
        "served_ms": round(served_ms, 3),
        "top10_identical": ranked_reference[:CHUNK_K] == ranked_served[:CHUNK_K],
        "gate_agrees": reference_result.gate.fired == served_result.gate.fired,
        "gate_fired": served_result.gate.fired,
    }
    for side, ranked in (("reference", ranked_reference), ("served", ranked_served)):
        row[f"{side}_r@{CHUNK_K}"] = recall_at_k(ranked, item.chunk_ids, CHUNK_K)
        documents = documents_of(ranked, export.document_of)
        gold_documents = frozenset(
            doc for doc in (export.document_of(c) for c in item.chunk_ids) if doc is not None
        )
        for k in DOC_KS:
            row[f"{side}_docr@{k}"] = recall_at_k(documents, gold_documents, k)
    return row


def aggregate(rows: list[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    """The dataset's summary block — means, deltas, latency tails, agreement rates."""

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else 0.0

    summary: dict[str, Any] = {
        "dataset": plan.dataset,
        "collection": plan.collection,
        "queries": len(rows),
        "limit": plan.limit,
        "search_params": {"exact": True} if plan.exact else {"hnsw_ef": plan.hnsw_ef},
        "ann_recall@10": round(mean("ann_recall@10"), 6),
        "ann_recall@100": round(mean("ann_recall@100"), 6),
        "top10_identical_rate": round(mean("top10_identical"), 6),
        "gate_agreement_rate": round(mean("gate_agrees"), 6),
        "gate_hit_rate": round(mean("gate_fired"), 6),
        "recall": {},
        "latency_ms": {},
    }
    for metric in (f"r@{CHUNK_K}", *(f"docr@{k}" for k in DOC_KS)):
        reference = mean(f"reference_{metric}")
        served = mean(f"served_{metric}")
        summary["recall"][metric] = {
            "reference": round(reference, 6),
            "served": round(served, 6),
            "delta": round(served - reference, 6),
        }
    for key in ("dense_ms", "reference_ms", "served_ms"):
        values = [row[key] for row in rows]
        summary["latency_ms"][key] = {
            "median": round(percentile(values, 0.5), 2),
            "p95": round(percentile(values, 0.95), 2),
            "mean": round(float(np.mean(values)), 2) if values else 0.0,
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run one dataset's parity measurement; print the summary JSON on stdout."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("Run with PYTHONHASHSEED=0 (the S5 determinism contract).", file=sys.stderr)
        return 2
    plan = parse_args(argv)
    spec = DATASETS[plan.dataset]

    queries = sample(spec.queries(), plan.limit)
    page_docs = frozenset(
        Path(ref.file_name).stem
        for query in queries
        for ref in query.refs
        if ref.kind is goldmap.RefKind.PAGE
    )
    print(f"[{plan.dataset}] loading export ({len(queries)} queries) …", flush=True)
    export = load_export(
        spec.export_dir,
        plan.dataset,
        s2_root=spec.s2_root,
        with_embeddings=True,
        page_docs=page_docs,
        with_graph_features=True,
    )
    resolved = [goldmap.resolve(q, export) for q in queries]
    usable = [r for r in resolved if r.is_usable]
    print(f"[{plan.dataset}] {len(usable)}/{len(queries)} queries usable", flush=True)

    vectors = query_vectors(plan, usable)
    print(f"[{plan.dataset}] building engines …", flush=True)
    reference_pipe, served_pipe, reference_dense, served_dense = build_pipelines(plan, export)
    seeder = AliasQuerySeeder(node_alias_map(export.root / "graph-nodes.parquet"))

    plan.out.mkdir(parents=True, exist_ok=True)
    rows_path = plan.out / f"{plan.run_name}.jsonl"
    rows: list[dict[str, Any]] = []
    if rows_path.is_file() and not plan.force:
        with rows_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        scored_ids = [row["query_id"] for row in rows]
        if scored_ids != [r.query_id for r in usable[: len(rows)]]:
            print(f"[{plan.dataset}] stale checkpoint {rows_path}; use --force.", file=sys.stderr)
            return 2
    elif plan.force and rows_path.is_file():
        rows_path.unlink()

    started = time.monotonic()
    with rows_path.open("a", encoding="utf-8") as handle:
        for position in range(len(rows), len(usable)):
            row = score_query(
                usable[position],
                vectors[position],
                export,
                (reference_pipe, served_pipe),
                (reference_dense, served_dense),
                seeder,
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            rows.append(row)
            done = position + 1
            if done % 100 == 0 or done == len(usable):
                elapsed = time.monotonic() - started
                print(
                    f"[{plan.dataset}] {done}/{len(usable)} "
                    f"({elapsed:.0f}s elapsed, sample seed {SAMPLE_SEED})",
                    flush=True,
                )

    summary = aggregate(rows, plan)
    summary_path = plan.out / f"{plan.run_name}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = [
    "CACHE_DIR",
    "CHUNK_K",
    "DOC_KS",
    "OUT_DIR",
    "HyperedgeChunks",
    "KgDenseReference",
    "Plan",
    "aggregate",
    "build_pipelines",
    "overlap_at_k",
    "parse_args",
    "query_vectors",
    "score_query",
]
