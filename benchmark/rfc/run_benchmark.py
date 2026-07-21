"""Run the fused retrieval + measure recall@10 per config (tutorial §6).

The RFC-retrieval-proof's whole claim is that **fusing** four retrieval signals unlocks a class of
queries no single signal can answer. §5 (``build_questions.py``) mined the super-hard questions +
gold chunk sets; this script is the **measured comparison** that proves (or refutes) the claim: it
runs each requested config against the run's real indexes and emits **real** recall@10 per config
and per question-type — never a fabricated number.

The four configs (tutorial §6):

* **dense** — the query is embedded (a sentence-transformer) and searched against the collection's
  dense vector via :class:`~latence_retrieval.qdrant.QdrantBackend` (the **real** Qdrant server).
* **bm25** — exact Okapi BM25 (:class:`~latence_retrieval.bm25.Bm25Rescorer`) over the exported
  chunk pool (pool-local IDF; no second index — ADR-0048).
* **splade** — the query is encoded to a sparse vector (the ADR-0053 SPLADE composition, reused
  query-side) and searched against the collection's **named sparse** vector (real Qdrant-native).
* **fused** — the real stack: dense-top-``pool`` + BM25-top-``pool`` + SPLADE-top-``pool`` →
  :class:`~latence_retrieval.fuser.RrfFuser` → **KG graph-injection** (denormalised ``kg_node_ids``
  re-attached to each candidate) + :class:`~latence_retrieval.multihop.MultiHopExpander` (2-hop,
  **hubs filtered**) → cross-encoder :class:`~latence_retrieval.rerank.Reranker` → knapsack
  :class:`~latence_retrieval.pack.Packer` @16k tokens.

**No fabricated numbers (tutorial §6).** Every recall is computed from a real ranked list a real
retrieval leg returned. The dense/splade legs run against the live Qdrant server named on the CLI;
the ``FakeQdrantClient`` is **forbidden** here (offline-CI-only). The heavy machinery
(``latence_retrieval`` processors, the encoders, ``qdrant-client``) is imported **lazily** in
:func:`build_real_engine` so importing this module — and the offline test that injects a stand-in
engine built from ``latence_retrieval``'s own faithful doubles — needs **no pod, GPU, network, or
Qdrant fake**.

The companion chart is **dependency-free SVG** (ADR-0016: no heavy chart lib in the pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from latence_retrieval import (
        Bm25Rescorer,
        MultiHopExpander,
        Packer,
        Reranker,
        RetrievalBackend,
        RrfFuser,
    )

# The four configs the proof compares (tutorial §6), in report order.
ALL_CONFIGS: tuple[str, ...] = ("dense", "bm25", "splade", "fused")
SINGLE_CONFIGS: tuple[str, ...] = ("dense", "bm25", "splade")
FUSED_CONFIG = "fused"

DEFAULT_POOL = 100
DEFAULT_TOP_K = 10
DEFAULT_PACK_BUDGET = 16_000
DEFAULT_MAX_HOPS = 2
DEFAULT_HUB_FRACTION = 0.02


# ---------------------------------------------------------------------------
# Data the benchmark consumes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One gated question from ``build_questions.py``: its type + gold chunk ids (§5 emit)."""

    qid: str
    qtype: str
    question: str
    gold_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CorpusChunk:
    """One exported chunk: its ``chunk_id`` (``record_id``), ``content``, and KG node ids.

    The ``kg_node_ids`` are the pipeline's denormalised entity node ids for the chunk (ADR-0051,
    the column ``MultiHopExpander`` reads). Absent on a corpus without a graph — then the multi-hop
    leg is a graceful no-op, surfaced honestly rather than faked.
    """

    chunk_id: str
    content: str
    kg_node_ids: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Readers — standard library only.
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every non-blank JSON object line of a file (``[]`` when the file is absent)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_questions(path: Path) -> tuple[dict[str, Any], list[Question]]:
    """Load the §5 benchmark JSON as ``(meta, questions)``.

    A malformed question (missing id/type/question, or no gold chunk ids) is a corrupt benchmark,
    not something to paper over — it raises. The ``meta`` block is returned untouched so the results
    can echo the provenance (``run``, ``top_k``, ``min_lift``) the questions were built under.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"benchmark JSON must be an object, got {type(payload).__name__}."
        raise ValueError(msg)
    raw_meta = payload.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list):
        msg = "benchmark JSON must carry a 'questions' array."
        raise ValueError(msg)
    questions: list[Question] = []
    for row in raw_questions:
        if not isinstance(row, dict):
            msg = f"each question must be an object, got {type(row).__name__}."
            raise ValueError(msg)
        qid = row.get("id")
        qtype = row.get("type")
        text = row.get("question")
        gold = row.get("gold_chunk_ids")
        if not isinstance(qid, str) or not qid:
            msg = f"question missing a non-empty 'id': {row!r}."
            raise ValueError(msg)
        if not isinstance(qtype, str) or not qtype:
            msg = f"question {qid!r} missing a non-empty 'type'."
            raise ValueError(msg)
        if not isinstance(text, str) or not text:
            msg = f"question {qid!r} missing a non-empty 'question'."
            raise ValueError(msg)
        if not isinstance(gold, list) or not gold:
            msg = f"question {qid!r} must carry a non-empty 'gold_chunk_ids' of strings."
            raise ValueError(msg)
        if not all(isinstance(g, str) and g for g in gold):
            msg = f"question {qid!r} 'gold_chunk_ids' must all be non-empty strings."
            raise ValueError(msg)
        questions.append(
            Question(qid=qid, qtype=qtype, question=text, gold_chunk_ids=tuple(gold))
        )
    return meta, questions


def _as_str_ids(raw: object) -> tuple[str, ...]:
    """Coerce a record's ``kg_node_ids`` cell to a tuple of non-empty id strings (else ``()``)."""
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(x) for x in raw if isinstance(x, str) and x)


def load_corpus(run: Path) -> list[CorpusChunk]:
    """Load the exported chunks (``export/records.jsonl``) as :class:`CorpusChunk`\\ s.

    A record is a chunk when it carries a ``record_id`` and textual ``content`` (a row with no text
    cannot be a gold chunk, so it is skipped). The denormalised ``kg_node_ids`` are read when
    present (the column the multi-hop leg traverses on) and default to empty otherwise.
    """
    chunks: list[CorpusChunk] = []
    for row in _read_jsonl(run / "export" / "records.jsonl"):
        record_id = row.get("record_id")
        content = row.get("content")
        if not isinstance(record_id, str) or not record_id:
            continue
        if not isinstance(content, str) or not content:
            continue
        chunks.append(
            CorpusChunk(
                chunk_id=record_id,
                content=content,
                kg_node_ids=_as_str_ids(row.get("kg_node_ids")),
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Metric — recall@k (identical semantics to build_questions.recall_at_k).
# ---------------------------------------------------------------------------


def recall_at_k(ranked: Sequence[str], gold: Collection[str], k: int) -> float:
    """Fraction of ``gold`` chunk ids in the top-``k`` of ``ranked`` (``0.0`` when no gold)."""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = set(ranked[:k])
    return len(top & gold_set) / len(gold_set)


# ---------------------------------------------------------------------------
# The hub-filtered graph source (tutorial §6: "MultiHopExpander (2-hop, hubs filtered)").
# ---------------------------------------------------------------------------


def hub_nodes_from_edges(
    edges: Iterable[tuple[str, str]], fraction: float
) -> frozenset[str]:
    """The highest-degree ``fraction`` of nodes — the graph's hubs.

    A hub-heavy induced KG (the German single-book failure the tutorial calls out) lets one
    over-connected node pull half the graph into a seed's 2-hop neighbourhood, drowning the real
    multi-hop signal. Degree is undirected (an edge lifts both endpoints), and the top ``fraction``
    by degree are named hubs and filtered from traversal. ``fraction <= 0`` names no hubs (the
    filter is off); ``fraction >= 1`` would name every node, so it is clamped below 1.
    """
    degree: Counter[str] = Counter()
    for src, dst in edges:
        degree[str(src)] += 1
        degree[str(dst)] += 1
    if fraction <= 0 or not degree:
        return frozenset()
    fraction = min(fraction, 0.999)
    count = max(1, int(len(degree) * fraction))
    # Highest degree first; ascending id breaks ties for determinism.
    ranked = sorted(degree, key=lambda node: (-degree[node], node))
    return frozenset(ranked[:count])


class HubFilteredGraphSource:
    """Wrap a :class:`~latence_retrieval.multihop.GraphSource`, removing hub nodes from the result.

    Faithful to the wrapped source's contract (undirected BFS, minimum hop distance, seeds
    excluded, the bounded-depth guard) — it only **subtracts the hubs at the boundary**: a hub is
    never a seed (so an over-connected chunk cannot inject its whole neighbourhood — the primary
    hub-heavy failure the tutorial calls out) and never a returned neighbour (so a hub node earns
    no hop-decayed bonus). The wrapped source's own connectivity is unchanged, so a non-hub node
    reachable only *through* a hub is still reached — hub filtering suppresses hubs as seeds and as
    lifted nodes, not the graph's transitive reachability. With an empty hub set it is exactly the
    wrapped source.
    """

    def __init__(self, inner: Any, hub_nodes: Collection[str]) -> None:
        self._inner = inner
        self._hubs: frozenset[str] = frozenset(hub_nodes)

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """Traverse from the non-hub seeds and drop hub nodes from the neighbourhood."""
        filtered_seeds = [s for s in seeds if s not in self._hubs]
        if not filtered_seeds:
            return {}
        neighbourhood = self._inner.neighbors(filtered_seeds, max_hops)
        return {
            node: dist for node, dist in neighbourhood.items() if node not in self._hubs
        }


# ---------------------------------------------------------------------------
# The retrieval engine seam.
# ---------------------------------------------------------------------------


DenseEncoder = Callable[[str], Sequence[float]]
SparseEncoder = Callable[[str], Mapping[str, float]]


class RetrievalEngine(Protocol):
    """Rank the run's chunks by each requested config for a query — the whole surface §6 drives.

    A single :meth:`rank` returns, per requested config, the ranked chunk-id list (already cut to
    ``top_k``). The production adapter runs it over the live Qdrant + encoders + the fused stack; a
    test injects a stand-in built from ``latence_retrieval``'s faithful doubles.
    """

    def rank(
        self, query: str, top_k: int, configs: Collection[str]
    ) -> dict[str, list[str]]: ...


class LatenceRetrievalEngine:
    """The real :class:`RetrievalEngine`: ``latence_retrieval`` over the running Qdrant (§6).

    Constructed with its collaborators (dependency injection): two
    :class:`~latence_retrieval.backend.RetrievalBackend`\\ s (dense + sparse, the Qdrant legs), the
    two query encoders, and the built fused-stack processors (:class:`RrfFuser`,
    :class:`MultiHopExpander`, :class:`Reranker`, :class:`Packer`) plus the corpus pool BM25
    rescores over. :func:`build_real_engine` wires the production (heavy) collaborators; a test
    wires the faithful fakes. The engine holds no durable index and re-runs every leg per query
    (ADR-0048).
    """

    def __init__(
        self,
        corpus: Sequence[CorpusChunk],
        *,
        dense_backend: RetrievalBackend,
        sparse_backend: RetrievalBackend,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        bm25: Bm25Rescorer,
        fuser: RrfFuser,
        multihop: MultiHopExpander | None,
        reranker: Reranker,
        packer: Packer,
        pool: int = DEFAULT_POOL,
    ) -> None:
        from latence_retrieval import Candidate

        self._corpus = list(corpus)
        self._dense_backend = dense_backend
        self._sparse_backend = sparse_backend
        self._dense_encoder = dense_encoder
        self._sparse_encoder = sparse_encoder
        self._bm25 = bm25
        self._fuser = fuser
        self._multihop = multihop
        self._reranker = reranker
        self._packer = packer
        self._pool = pool
        # The BM25 pool (whole corpus) and the id → (text, node ids) map for KG graph-injection.
        self._corpus_candidates = [
            Candidate(id=c.chunk_id, score=0.0, text=c.content) for c in self._corpus
        ]
        self._by_id: dict[str, CorpusChunk] = {c.chunk_id: c for c in self._corpus}

    def rank(
        self, query: str, top_k: int, configs: Collection[str]
    ) -> dict[str, list[str]]:
        """Rank by each requested config; only the legs a requested config needs are run."""
        want = set(configs)
        need_dense = "dense" in want or FUSED_CONFIG in want
        need_bm25 = "bm25" in want or FUSED_CONFIG in want
        need_splade = "splade" in want or FUSED_CONFIG in want

        dense_list = self._dense(query) if need_dense else []
        bm25_list = self._bm25_rank(query) if need_bm25 else []
        splade_list = self._splade(query) if need_splade else []

        out: dict[str, list[str]] = {}
        if "dense" in want:
            out["dense"] = [c.id for c in dense_list[:top_k]]
        if "bm25" in want:
            out["bm25"] = [c.id for c in bm25_list[:top_k]]
        if "splade" in want:
            out["splade"] = [c.id for c in splade_list[:top_k]]
        if FUSED_CONFIG in want:
            fused = self._fused(query, dense_list, bm25_list, splade_list)
            out[FUSED_CONFIG] = [c.id for c in fused[:top_k]]
        return out

    # -- single signals -----------------------------------------------------

    def _dense(self, query: str) -> list[Any]:
        from latence_retrieval import Capability, RetrievalQuery

        vector = tuple(float(x) for x in self._dense_encoder(query))
        hits = self._dense_backend.search(
            RetrievalQuery(mode=Capability.DENSE, vector=vector, top_k=self._pool)
        )
        return self._rekey(hits)

    def _splade(self, query: str) -> list[Any]:
        from latence_retrieval import Capability, RetrievalQuery

        sparse_terms = dict(self._sparse_encoder(query))
        hits = self._sparse_backend.search(
            RetrievalQuery(
                mode=Capability.SPARSE, sparse_terms=sparse_terms, top_k=self._pool
            )
        )
        return self._rekey(hits)

    def _bm25_rank(self, query: str) -> list[Any]:
        return list(self._bm25.process(query, self._corpus_candidates)[: self._pool])

    # -- the fused stack ----------------------------------------------------

    def _fused(
        self,
        query: str,
        dense_list: Sequence[Any],
        bm25_list: Sequence[Any],
        splade_list: Sequence[Any],
    ) -> list[Any]:
        """RRF → KG graph-injection + 2-hop hubs-filtered expand → rerank → pack@16k."""
        fused = self._fuser.fuse([list(dense_list), list(bm25_list), list(splade_list)])
        injected = self._inject_kg(fused)
        if self._multihop is not None:
            injected = self._multihop.process(query, injected)
        reranked = self._reranker.process(query, injected)
        packed = self._packer.process(query, reranked)
        return list(packed)

    def _inject_kg(self, candidates: Sequence[Any]) -> list[Any]:
        """Re-attach each candidate's corpus text + denormalised ``kg_node_ids`` (§6 injection).

        The fused list carries first-stage ids and scores; the multi-hop leg needs the entity node
        ids and the reranker/packer the authoritative chunk text. Both come from the exported
        corpus, keyed by id. A candidate absent from the corpus keeps what it had (surfaced, not
        hidden) and simply earns no graph bonus.
        """
        from latence_retrieval import Candidate

        out: list[Any] = []
        for cand in candidates:
            chunk = self._by_id.get(cand.id)
            if chunk is None:
                out.append(cand)
                continue
            out.append(
                Candidate(
                    id=cand.id,
                    score=cand.score,
                    text=chunk.content,
                    metadata={"kg_node_ids": list(chunk.kg_node_ids)},
                )
            )
        return out

    @staticmethod
    def _rekey(hits: Iterable[Any]) -> list[Any]:
        """Re-key backend hits to the corpus ``record_id`` so all signals fuse on one id space.

        A Qdrant point id is a derived UUIDv5; the chunk key the gold sets use is the ``record_id``
        the sink wrote into the payload. Re-key each hit to it (falling back to the point id only if
        the payload lacks it — a mis-loaded collection, surfaced not hidden).
        """
        from latence_retrieval import Candidate

        out: list[Any] = []
        for hit in hits:
            record_id = hit.metadata.get("record_id")
            cid = record_id if isinstance(record_id, str) and record_id else hit.id
            out.append(Candidate(id=cid, score=hit.score, text=hit.text))
        return out


# ---------------------------------------------------------------------------
# The production engine factory — heavy deps imported lazily (the pod path).
# ---------------------------------------------------------------------------


def _default_dimension_hint() -> int:
    """The Granite r2 dense width the enterprise-SOTA export writes (§2's ``embedding`` 768-d)."""
    return 768


def _read_graph_edges(edges_path: Path) -> list[tuple[str, str]]:  # pragma: no cover - pod path
    """Read ``graph-edges.parquet`` as ``(source, target)`` id pairs (for hub-degree counting)."""
    import duckdb

    rows = duckdb.sql(
        f"SELECT source, target FROM read_parquet('{edges_path}')"  # noqa: S608 - trusted local path
    ).fetchall()
    return [(str(s), str(t)) for s, t in rows]


def build_real_engine(  # pragma: no cover - exercised only on the pod (real deps)
    run: Path | None, corpus: Sequence[CorpusChunk], args: argparse.Namespace
) -> LatenceRetrievalEngine:
    """Wire the production :class:`LatenceRetrievalEngine` from the CLI args (the live-Qdrant path).

    Builds the two Qdrant backends, the dense (sentence-transformer) + sparse (SPLADE composition)
    query encoders, the exact-BM25 rescorer, the RRF fuser, the 2-hop hubs-filtered multi-hop
    expander over the emitted ``graph-edges.parquet`` (when present), the cross-encoder reranker,
    and the knapsack packer @``pack_budget``. Every heavy import lives here so the module — and the
    offline test — never needs a pod.
    """
    from latence_retrieval import (
        Bm25Rescorer,
        DuckDBGraphSource,
        MultiHopExpander,
        Packer,
        QdrantBackend,
        Reranker,
        RrfFuser,
        resolve_sparse_encoder,
    )

    dense_backend = QdrantBackend(
        {
            "collection": args.collection,
            "capabilities": ["dense"],
            "dimension": args.dimension,
            "vector_name": args.vector_name,
            "url": args.qdrant,
        }
    )
    sparse_backend = QdrantBackend(
        {
            "collection": args.collection,
            "capabilities": ["sparse"],
            "sparse_vector_name": args.sparse_vector_name,
            "url": args.qdrant,
        }
    )

    from sentence_transformers import SentenceTransformer

    dense_model = SentenceTransformer(args.dense_model)

    def dense_encoder(text: str) -> Sequence[float]:
        return [float(x) for x in dense_model.encode(text).tolist()]

    sparse_query_encoder = resolve_sparse_encoder(
        {"provider": "query_sparse.splade", "config": {"model_id": args.splade_model}}
    )

    def sparse_encoder(text: str) -> Mapping[str, float]:
        sv = sparse_query_encoder.encode(text)
        return {str(i): float(v) for i, v in zip(sv.indices, sv.values, strict=True)}

    edges_path = (run / "export" / "graph-edges.parquet") if run is not None else None
    multihop: MultiHopExpander | None = None
    if edges_path is not None and edges_path.is_file():
        hub_nodes = hub_nodes_from_edges(_read_graph_edges(edges_path), args.hub_fraction)
        graph_source = HubFilteredGraphSource(DuckDBGraphSource(str(edges_path)), hub_nodes)
        multihop = MultiHopExpander(
            {"max_hops": args.max_hops}, graph_source=graph_source
        )

    reranker = Reranker({"model": args.rerank_model})
    packer = Packer({"budget": args.pack_budget})

    return LatenceRetrievalEngine(
        corpus,
        dense_backend=dense_backend,
        sparse_backend=sparse_backend,
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder,
        bm25=Bm25Rescorer(),
        fuser=RrfFuser(),
        multihop=multihop,
        reranker=reranker,
        packer=packer,
        pool=args.pool,
    )


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionResult:
    """One question's measured recall@k for each config that was run."""

    qid: str
    qtype: str
    recall: dict[str, float]
    fused_only_win: bool = False


@dataclass(frozen=True)
class BenchmarkResults:
    """The measured comparison: per-question recalls + the aggregate story (means, per-type)."""

    configs: tuple[str, ...]
    top_k: int
    per_question: list[QuestionResult] = field(default_factory=list)

    def mean_recall(self) -> dict[str, float]:
        """Mean recall@k per config across all questions (``0.0`` for an empty benchmark)."""
        means: dict[str, float] = {}
        n = len(self.per_question)
        for config in self.configs:
            if n == 0:
                means[config] = 0.0
            else:
                means[config] = sum(q.recall[config] for q in self.per_question) / n
        return means

    def by_type(self) -> dict[str, dict[str, float]]:
        """Mean recall@k per config, broken down by question type (report legibility, §6)."""
        buckets: dict[str, list[QuestionResult]] = defaultdict(list)
        for q in self.per_question:
            buckets[q.qtype].append(q)
        out: dict[str, dict[str, float]] = {}
        for qtype in sorted(buckets):
            rows = buckets[qtype]
            out[qtype] = {
                config: sum(q.recall[config] for q in rows) / len(rows)
                for config in self.configs
            }
        return out

    def fused_only_wins(self) -> int:
        """Questions where fused finds gold in top-k that **no** single signal does (§6 bar)."""
        return sum(1 for q in self.per_question if q.fused_only_win)


def run_benchmark(
    questions: Sequence[Question],
    engine: RetrievalEngine,
    configs: Sequence[str],
    top_k: int,
) -> BenchmarkResults:
    """Rank every question by each config and measure recall@k — the real comparison (§6).

    For each question the engine is asked for the requested configs' ranked chunk-id lists, and
    recall@``top_k`` is computed against the question's gold set. When ``fused`` and the three
    single signals are all present, the per-question ``fused_only_win`` records the §6 proof event:
    fused surfaces a gold chunk in top-k that no single signal does.
    """
    ordered = tuple(c for c in ALL_CONFIGS if c in set(configs))
    results: list[QuestionResult] = []
    for q in questions:
        ranked = engine.rank(q.question, top_k, ordered)
        recall = {
            config: recall_at_k(ranked.get(config, []), q.gold_chunk_ids, top_k)
            for config in ordered
        }
        fused_only = _fused_only_win(ranked, q.gold_chunk_ids, top_k, ordered)
        results.append(
            QuestionResult(
                qid=q.qid, qtype=q.qtype, recall=recall, fused_only_win=fused_only
            )
        )
    return BenchmarkResults(configs=ordered, top_k=top_k, per_question=results)


def _fused_only_win(
    ranked: Mapping[str, Sequence[str]],
    gold: Collection[str],
    top_k: int,
    configs: Sequence[str],
) -> bool:
    """True when fused finds a gold chunk in top-k that every single signal misses (§6 proof)."""
    if FUSED_CONFIG not in configs or not all(c in configs for c in SINGLE_CONFIGS):
        return False
    gold_set = set(gold)
    fused_hits = gold_set & set(ranked.get(FUSED_CONFIG, [])[:top_k])
    if not fused_hits:
        return False
    single_hits: set[str] = set()
    for config in SINGLE_CONFIGS:
        single_hits |= gold_set & set(ranked.get(config, [])[:top_k])
    return bool(fused_hits - single_hits)


# ---------------------------------------------------------------------------
# Emit — the results JSON.
# ---------------------------------------------------------------------------


def results_to_json(
    results: BenchmarkResults,
    *,
    questions_path: Path,
    meta: Mapping[str, Any],
    qdrant: str,
    collection: str,
    chart: Path | None,
) -> dict[str, Any]:
    """The results payload: aggregate recall, per-type breakdown, and per-question detail."""
    return {
        "meta": {
            "questions": str(questions_path),
            "qdrant": qdrant,
            "collection": collection,
            "configs": list(results.configs),
            "top_k": results.top_k,
            "n_questions": len(results.per_question),
            "run": meta.get("run"),
            "fused_only_wins": results.fused_only_wins(),
            "chart": str(chart) if chart is not None else None,
        },
        "recall_at_k": {
            config: round(value, 4) for config, value in results.mean_recall().items()
        },
        "by_type": {
            qtype: {config: round(value, 4) for config, value in row.items()}
            for qtype, row in results.by_type().items()
        },
        "per_question": [
            {
                "id": q.qid,
                "type": q.qtype,
                "recall": {config: round(q.recall[config], 4) for config in results.configs},
                "fused_only_win": q.fused_only_win,
            }
            for q in results.per_question
        ],
    }


# ---------------------------------------------------------------------------
# Chart — dependency-free SVG (ADR-0016: no heavy chart lib).
# ---------------------------------------------------------------------------


_BG = "#0b0f14"
_FG = "#e6edf3"
_GRID = "#243040"
_FONT = "font-family='ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif'"
_CONFIG_COLOR: dict[str, str] = {
    "dense": "#4f83cc",
    "bm25": "#cc8a4f",
    "splade": "#9a6fcc",
    "fused": "#4fcc8a",
}


def _esc(text: str) -> str:
    """Minimal XML text escape (the only entities an SVG ``<text>`` body needs)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x: float, y: float, s: str, *, size: int, fill: str, weight: str, anchor: str) -> str:
    return (
        f"<text x='{x:.1f}' y='{y:.1f}' {_FONT} font-size='{size}' font-weight='{weight}' "
        f"fill='{fill}' text-anchor='{anchor}'>{_esc(s)}</text>"
    )


def _rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return (
        f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(w, 0.0):.1f}' "
        f"height='{max(h, 0.0):.1f}' fill='{fill}'/>"
    )


def render_recall_chart(results: BenchmarkResults) -> str:
    """A grouped-bar SVG: mean recall@k per config, plus a per-type cluster (self-contained).

    One bar per config in the "overall" group and in each question-type group, coloured per config,
    y-axis 0..1 recall. Pure string assembly — no external asset, font, or chart library (ADR-0016).
    """
    configs = results.configs
    groups: list[tuple[str, dict[str, float]]] = [("overall", results.mean_recall())]
    groups.extend(results.by_type().items())

    width, height = 900, 460
    pad_l, pad_r, pad_t, pad_b = 60, 20, 70, 90
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    group_w = plot_w / max(len(groups), 1)
    bar_w = group_w / (len(configs) + 1) if configs else group_w

    body: list[str] = [_rect(0, 0, width, height, _BG)]
    title = f"RFC fused-retrieval proof — recall@{results.top_k}"
    body.append(_text(width / 2, 34, title, size=20, fill=_FG, weight="700", anchor="middle"))
    # y grid + axis labels (0.0 .. 1.0).
    for i in range(6):
        frac = i / 5
        y = pad_t + plot_h * (1 - frac)
        body.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width - pad_r}' y2='{y:.1f}' "
                    f"stroke='{_GRID}' stroke-width='1'/>")
        body.append(_text(pad_l - 8, y + 4, f"{frac:.1f}", size=11, fill=_FG,
                          weight="400", anchor="end"))

    for gi, (label, recalls) in enumerate(groups):
        gx = pad_l + gi * group_w
        for ci, config in enumerate(configs):
            value = max(0.0, min(1.0, recalls.get(config, 0.0)))
            bx = gx + (ci + 0.5) * bar_w
            bh = plot_h * value
            by = pad_t + plot_h - bh
            body.append(_rect(bx, by, bar_w * 0.9, bh, _CONFIG_COLOR.get(config, "#888")))
            body.append(_text(bx + bar_w * 0.45, by - 4, f"{value:.2f}", size=10, fill=_FG,
                              weight="400", anchor="middle"))
        body.append(_text(gx + group_w / 2, pad_t + plot_h + 18, label, size=12, fill=_FG,
                          weight="600", anchor="middle"))

    # Legend.
    lx = pad_l
    ly = height - 26
    for config in configs:
        body.append(_rect(lx, ly - 10, 12, 12, _CONFIG_COLOR.get(config, "#888")))
        body.append(_text(lx + 18, ly, config, size=12, fill=_FG, weight="400", anchor="start"))
        lx += 18 + 10 * len(config) + 24

    joined = "".join(body)
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img'>{joined}</svg>\n"
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _resolve_run(meta: Mapping[str, Any], override: Path | None) -> Path | None:
    """The run dir the corpus + graph live under: the CLI override wins, else ``meta.run`` (§5)."""
    if override is not None:
        return override
    run = meta.get("run")
    if isinstance(run, str) and run:
        return Path(run)
    return None


def main(argv: Sequence[str] | None = None, *, engine_factory: Any = None) -> int:
    """CLI entry (tutorial §6). ``engine_factory(run, corpus, args) → RetrievalEngine`` injectable.

    With no ``engine_factory`` the production :func:`build_real_engine` is used — the real
    ``latence_retrieval`` path against the live Qdrant. A test passes a stand-in factory (built from
    ``latence_retrieval``'s own faithful doubles) to drive the whole rank → measure → emit pipeline
    offline, with no pod, network, or Qdrant fake.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--questions", required=True, type=Path, help="the §5 benchmark JSON (rfc-benchmark.json)"
    )
    ap.add_argument(
        "--qdrant",
        default=os.environ.get("LATENCE_QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (default $LATENCE_QDRANT_URL)",
    )
    ap.add_argument("--collection", default="rfc", help="the Qdrant collection to search")
    ap.add_argument(
        "--configs",
        default=",".join(ALL_CONFIGS),
        help="comma-separated subset of dense,bm25,splade,fused",
    )
    ap.add_argument("--out", required=True, type=Path, help="output results JSON path")
    ap.add_argument(
        "--charts", type=Path, default=None,
        help="output SVG chart path (default: <out>.svg alongside the results)",
    )
    ap.add_argument(
        "--run", type=Path, default=None,
        help="the run dir (default: the benchmark JSON's meta.run) — holds the corpus + graph",
    )
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k", help="recall@k cut")
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL, help="first-stage pool per signal")
    ap.add_argument(
        "--dimension", type=int, default=_default_dimension_hint(), help="dense vector width"
    )
    ap.add_argument(
        "--dense-model", dest="dense_model",
        default="ibm-granite/granite-embedding-278m-multilingual", help="dense query encoder",
    )
    ap.add_argument(
        "--splade-model", dest="splade_model", default="naver/splade-v3",
        help="SPLADE query encoder",
    )
    ap.add_argument(
        "--rerank-model", dest="rerank_model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="cross-encoder reranker",
    )
    ap.add_argument(
        "--vector-name", dest="vector_name", default=None,
        help="named dense vector (else the unnamed vector)",
    )
    ap.add_argument(
        "--sparse-vector-name", dest="sparse_vector_name", default="sparse",
        help="named sparse vector",
    )
    ap.add_argument(
        "--pack-budget", dest="pack_budget", type=int, default=DEFAULT_PACK_BUDGET,
        help="knapsack packer token budget",
    )
    ap.add_argument(
        "--max-hops", dest="max_hops", type=int, default=DEFAULT_MAX_HOPS,
        help="multi-hop traversal depth",
    )
    ap.add_argument(
        "--hub-fraction", dest="hub_fraction", type=float, default=DEFAULT_HUB_FRACTION,
        help="fraction of highest-degree nodes filtered as hubs before traversal",
    )
    args = ap.parse_args(argv)

    if not args.questions.is_file():
        print(f"error: benchmark JSON not found: {args.questions}", file=sys.stderr)
        return 2

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in requested if c not in ALL_CONFIGS]
    if unknown:
        print(f"error: unknown config(s) {unknown}; choose from {list(ALL_CONFIGS)}",
              file=sys.stderr)
        return 2
    if not requested:
        print("error: no configs requested", file=sys.stderr)
        return 2

    meta, questions = load_questions(args.questions)
    if not questions:
        print(f"error: no questions in {args.questions}", file=sys.stderr)
        return 2

    run = _resolve_run(meta, args.run)
    corpus: list[CorpusChunk] = load_corpus(run) if run is not None else []
    if not corpus and ("bm25" in requested or FUSED_CONFIG in requested):
        # bm25 and the fused stack rescore/rerank over the exported corpus pool — without it those
        # configs cannot run for real, and faking the pool would fabricate the numbers §6 forbids.
        print(
            f"error: no exported corpus under {run}/export/records.jsonl — "
            "bm25 and fused need the chunk pool (fix the run path, do not fake it)",
            file=sys.stderr,
        )
        return 2

    factory = engine_factory if engine_factory is not None else build_real_engine
    engine = factory(run, corpus, args)

    results = run_benchmark(questions, engine, requested, args.top_k)

    chart_path = args.charts if args.charts is not None else args.out.with_suffix(".svg")
    payload = results_to_json(
        results,
        questions_path=args.questions,
        meta=meta,
        qdrant=args.qdrant,
        collection=args.collection,
        chart=chart_path,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.write_text(render_recall_chart(results), encoding="utf-8")

    means = results.mean_recall()
    print(f"run_benchmark: measured {len(questions)} questions over {requested}")
    for config in results.configs:
        print(f"  recall@{args.top_k} {config:>7}: {means[config]:.4f}")
    if FUSED_CONFIG in results.configs:
        print(f"  fused-only wins (gold no single signal found): {results.fused_only_wins()}")
    print(f"  wrote {args.out} + {chart_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
