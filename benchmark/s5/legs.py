# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The cumulative leg ladder: dense → +bm25 → +KG → +hyperedges → +CoE → +packer.

Each leg is the previous one **plus** one signal, so every row of the S5 table attributes one
increment. All scoring is done by the shipped ``latence-retrieval`` components — this module
only supplies the two things a stateless library deliberately does not own (ADR-0048): the
caller's first-stage *engine*, and the caller's query-signal generation.

**What the harness legitimately implements, and why it is not "reimplemented scoring".**

* :class:`DenseBackend` — the S3 export ships vectors but no ANN index (the ticket's F-VEC
  residual). A ``RetrievalBackend`` is by contract *the caller's engine*; brute-force cosine
  over an L2-normalised matrix is one ``numpy`` matmul and is exact, so an approximate index
  would only add error. Memory is bounded by scoring in query blocks — never an all-queries ×
  all-chunks matrix.
* :class:`Bm25Backend` — the export ships an inverted index (``bm25-postings.parquet``) and
  corpus statistics (``bm25-stats.json``), but the library's BM25 is a *rescorer* over a
  candidate pool. So the backend uses the postings only to select the pool — the documents
  containing at least one query term — and hands that pool to the real
  :class:`~latence_retrieval.bm25.Bm25Rescorer` built with
  :meth:`~latence_retrieval.bm25.Bm25Rescorer.from_export`. Documents outside the pool score
  exactly ``0`` under Okapi BM25, so the ranking is **identical** to a full-corpus first stage;
  the prefilter changes cost, not results.
* :class:`EdgeGraphSource` — the shipped ``NetworkxGraphSource`` re-reads and rebuilds the edge
  graph on *every* call, which would dominate a per-query latency measurement. This holds the
  same networkx graph across a run and answers with the same call; ``tests/test_legs.py``
  asserts it returns exactly what ``NetworkxGraphSource`` returns on the real edge parquet.
* :class:`ChunkGraphDocumentSource` — the ``duckdb`` extra is not in this repo's dev environment,
  so the node→document index is served from the already-loaded ``kg_node_ids`` column, honouring
  the same ``GraphDocumentSource`` contract (each document at most once, ascending id).

Everything else is the real thing: :class:`~latence_retrieval.graph_retrieval.GraphRetriever`
and :class:`~latence_retrieval.graph_retrieval.AliasQuerySeeder` for leg 3,
:func:`~latence_retrieval.hypergraph.hypergraph_signals` for leg 4,
:func:`~latence_retrieval.coe.select` for leg 5,
:class:`~latence_retrieval.pack.Packer` for leg 6, and
:class:`~latence_retrieval.fuser.RrfFuser` to merge the legs — rank fusion, because the legs'
scores (cosine, Okapi, hop decay, ρ) are on flatly incomparable scales.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq

_RETRIEVAL_SRC = Path(__file__).resolve().parents[2] / "packages" / "latence-retrieval" / "src"
if str(_RETRIEVAL_SRC) not in sys.path:  # pragma: no cover - import wiring
    sys.path.insert(0, str(_RETRIEVAL_SRC))

from benchmark.s5.corpus import (  # noqa: E402
    CONTENT_ONLY_VECTORS,
    Export,
    hyperedge_chunk_map,
    node_alias_map,
)
from latence_retrieval.bm25 import Bm25Rescorer  # noqa: E402
from latence_retrieval.coe import select as coe_select  # noqa: E402
from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery  # noqa: E402
from latence_retrieval.fuser import RrfFuser  # noqa: E402
from latence_retrieval.graph_retrieval import (  # noqa: E402
    AliasQuerySeeder,
    GraphDocument,
    GraphRetriever,
)
from latence_retrieval.hypergraph import (  # noqa: E402
    HyperedgeKV,
    hypergraph_signals,
)
from latence_retrieval.pack import Packer  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

#: The legs, in ladder order. Each name is the *increment*; leg N runs signals 1..N.
LEG_NAMES = ("dense", "bm25", "kg", "hyperedges", "coe", "packer")

#: Ablation legs — deliberately NOT rungs of the ladder above. Each re-runs ONE rung against a
#: different *input* so the report can attribute that input, and each is reported in its own
#: section so the ladder's cumulative Δ chain stays a chain of single-signal increments.
#:
#: ``dense_content_only`` is the ``dense`` rung ranked with vectors built from each chunk's
#: CONTENT ONLY, against the ``dense`` rung's vectors which embed content **plus** the W13
#: ContextHeader (the chunk's KG entities and neighbour triples) — because that is what the S3
#: Export wrote into ``corpus.parquet``. Without this leg the header would be an invisible
#: advantage folded into every dense number, and worse, an *uneven* one: the S3 corpora carry
#: populated headers while the reused M6 exports carry ``context_header: null`` on every row, so
#: the same "dense" label would mean different things per dataset. Only the CORPUS side changes —
#: queries have no context header, so both legs rank the identical query vectors.
ABLATION_LEG_NAMES = ("dense_content_only",)

#: Every leg name the matrix accepts, ladder first.
ALL_LEG_NAMES = (*LEG_NAMES, *ABLATION_LEG_NAMES)

#: Which ladder leg each ablation leg is the counterfactual of — the pair the report subtracts.
ABLATION_BASELINE = {"dense_content_only": "dense"}


def vector_source(leg: str) -> str:
    """Where ``leg``'s dense vectors come from: ``"corpus"`` or ``"sidecar"``.

    ``corpus`` is ``corpus.parquet``'s ``embedding`` column (content + ContextHeader, as
    exported); ``sidecar`` is :data:`~benchmark.s5.corpus.CONTENT_ONLY_VECTORS` next to it. One
    function, so no call site has to re-derive the mapping from a leg name.
    """
    return "sidecar" if leg in ABLATION_LEG_NAMES else "corpus"

#: Candidates each first-stage leg returns before fusion. Wide enough that recall@10 is a
#: property of the legs rather than of the truncation.
POOL_K = 100

#: The packer's token budget (leg 6) — the "what actually fits in a prompt" question the ladder
#: ends on. Reported against tokens actually packed.
PACK_BUDGET = 4000

#: The CoE selector's token budget (leg 5), matching M6's mid-range operating point.
COE_BUDGET = 4000

_WORD = re.compile(r"[\w][\w&.\-]*")


def query_ngrams(question: str, max_n: int = 4) -> list[str]:
    """Descending-length word n-grams of a question — the tier-1 zero-model entry surfaces.

    Byte-identical to the M6 harness's ``question_ngrams`` (``benchmark/musique-subset``), so
    the hypergraph leg's entry behaves the same way it did in the M6 numbers this run is
    compared against.
    """
    words = _WORD.findall(question)
    grams: list[str] = []
    for n in range(max_n, 0, -1):
        grams.extend(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


# -- the caller's engines (ADR-0048: the backend is ours to supply, the scoring is not) --------


class DenseBackend:
    """Exact brute-force cosine over the export's L2-normalised vectors (a ``RetrievalBackend``).

    ``matrix`` overrides the export's own vectors with an alternative source (the content-only
    sidecar) while every other view of the corpus — rows, ids, text — stays the one loaded export.
    Whichever matrix it is given, the row count must equal the corpus's: a matrix off by one row
    still ranks, and every id it returns would be the wrong chunk's, so the mismatch is refused at
    construction rather than discovered in a recall number.
    """

    def __init__(
        self,
        export: Export,
        *,
        matrix: Any | None = None,
        source: str = "corpus.parquet:embedding",
    ) -> None:
        vectors = export.embeddings if matrix is None else matrix
        if vectors is None:
            msg = f"{export.dataset}: corpus.parquet carries no 'embedding' column."
            raise ValueError(msg)
        if int(vectors.shape[0]) != len(export.rows):
            msg = (
                f"{export.dataset}: {source} has {int(vectors.shape[0])} vectors for "
                f"{len(export.rows)} corpus rows; refusing to rank a misaligned matrix."
            )
            raise ValueError(msg)
        self._export = export
        self._matrix = vectors
        self._source = source

    @property
    def source(self) -> str:
        """A human-readable name for where this backend's vectors came from."""
        return self._source

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{DENSE}`` — a vector store, nothing more."""
        return frozenset({Capability.DENSE})

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Top-``top_k`` by cosine; ties broken by ascending id, like every leg here."""
        vector = np.asarray(query.vector, dtype=np.float32)
        scores = self._matrix @ vector
        top = min(query.top_k, scores.shape[0])
        part = np.argpartition(-scores, top - 1)[:top]
        order = part[np.argsort(-scores[part], kind="stable")]
        rows = self._export.rows
        return [
            Candidate(id=rows[i].record_id, score=float(scores[i]), text=rows[i].content)
            for i in order
        ]


class Bm25Backend:
    """Exact global Okapi BM25: postings pick the pool, the real ``Bm25Rescorer`` scores it."""

    def __init__(self, export: Export) -> None:
        self._export = export
        self._rescorer = Bm25Rescorer.from_export(str(export.root))
        table = pq.read_table(export.root / "bm25-postings.parquet", columns=["record_id", "terms"])  # type: ignore[no-untyped-call]
        postings: dict[str, list[int]] = {}
        for row, terms in enumerate(table.column("terms").to_pylist()):
            for term in terms or ():
                postings.setdefault(str(term), []).append(row)
        self._postings = postings
        self._row_of = {
            str(record_id): index
            for index, record_id in enumerate(table.column("record_id").to_pylist())
        }

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{BM25}``."""
        return frozenset({Capability.BM25})

    def _pool(self, text: str) -> list[Candidate]:
        rows = self._export.rows
        hit: set[int] = set()
        for term in set(_tokenize(text)):
            hit.update(self._postings.get(term, ()))
        return [
            Candidate(id=rows[i].record_id, score=0.0, text=rows[i].content) for i in sorted(hit)
        ]

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Score the term-matching pool with the shipped rescorer; truncate to ``top_k``."""
        pool = self._pool(query.text or "")
        if not pool:
            return []
        return self._rescorer.process(query.text or "", pool)[: query.top_k]


_BM25_TOKEN = re.compile(r"[^\W_]+")


def _tokenize(text: str) -> list[str]:
    """The index-side term unit (``latence_core.bm25.RegexWordTokenizer``), mirrored for lookup."""
    return _BM25_TOKEN.findall(text.lower())


class EdgeGraphSource:
    """A run-lifetime ``GraphSource`` over ``graph-edges.parquet`` (networkx, same semantics)."""

    def __init__(self, edges_parquet: Path) -> None:
        import networkx as nx

        table = pq.read_table(  # type: ignore[no-untyped-call]
            edges_parquet, columns=["source_node_id", "target_node_id"]
        )
        graph = nx.Graph()
        graph.add_edges_from(
            zip(
                (str(s) for s in table.column("source_node_id").to_pylist()),
                (str(t) for t in table.column("target_node_id").to_pylist()),
                strict=True,
            )
        )
        self._graph = graph

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """``node id → minimum hop distance`` within ``max_hops``, seeds excluded."""
        import networkx as nx

        seed_list = list(dict.fromkeys(seeds))
        seed_set = set(seed_list)
        best: dict[str, int] = {}
        for seed in seed_list:
            if seed not in self._graph:
                continue
            for node, distance in nx.single_source_shortest_path_length(
                self._graph, seed, cutoff=max_hops
            ).items():
                if distance == 0 or node in seed_set:
                    continue
                if node not in best or distance < best[node]:
                    best[node] = int(distance)
        return best


class ChunkGraphDocumentSource:
    """The node→document index over the export's own ``kg_node_ids`` column."""

    def __init__(self, export: Export) -> None:
        self._export = export

    def documents(self, node_ids: Collection[str]) -> list[GraphDocument]:
        """Documents mentioning any of ``node_ids``, each once, ascending id."""
        rows = self._export.rows
        hits = [rows[i] for i in self._export.rows_for_nodes(list(node_ids))]
        hits.sort(key=lambda row: row.record_id)
        return [
            GraphDocument(id=row.record_id, node_ids=row.node_ids, text=row.content)
            for row in hits
        ]


# -- the ladder ---------------------------------------------------------------------------------


@dataclass
class LegOutcome:
    """One query's result for one leg: the ranked ids plus the leg-specific observability."""

    ranked: list[str]
    latency_ms: float
    tokens_packed: int | None = None
    budget: int | None = None


@dataclass
class Runner:
    """Everything one dataset's ladder needs, built once and reused across legs and queries."""

    export: Export
    dense: DenseBackend | None
    bm25: Bm25Backend | None
    graph: GraphRetriever | None
    kv_path: Path | None
    #: The ablation's second dense engine — the content-only sidecar over the same corpus rows.
    dense_content_only: DenseBackend | None = None
    hyperedge_chunks: dict[str, str] = field(default_factory=dict)
    fuser: RrfFuser = field(default_factory=RrfFuser)
    packer: Packer = field(default_factory=lambda: Packer({"budget": PACK_BUDGET}))
    seeder: AliasQuerySeeder | None = None

    def _candidates_by_id(self, ids: Sequence[str]) -> list[Candidate]:
        rows = self.export.rows
        out: list[Candidate] = []
        for rank, chunk_id in enumerate(ids):
            position = self.export.index_of(chunk_id)
            if position is None:
                continue
            out.append(
                Candidate(
                    id=chunk_id,
                    score=1.0 / (1 + rank),
                    text=rows[position].content,
                )
            )
        return out

    def run(self, question: str, vector: Any, leg: str) -> LegOutcome:
        """Run the ladder up to and including ``leg`` for one query (or one ablation leg)."""
        started = time.perf_counter()
        if leg in ABLATION_LEG_NAMES:
            return self._ablation(vector, leg, started)
        rankings: list[list[Candidate]] = []

        if self.dense is not None:
            rankings.append(
                self.dense.search(
                    RetrievalQuery(
                        mode=Capability.DENSE, vector=tuple(float(x) for x in vector),
                        top_k=POOL_K,
                    )
                )
            )
        if leg != "dense" and self.bm25 is not None:
            rankings.append(
                self.bm25.search(
                    RetrievalQuery(mode=Capability.BM25, text=question, top_k=POOL_K)
                )
            )
        if leg in ("kg", "hyperedges", "coe", "packer") and self.graph is not None:
            rankings.append(self._graph_leg(question))
        if leg in ("hyperedges", "coe", "packer"):
            hyper, coe = self._hypergraph_legs(question, want_coe=leg in ("coe", "packer"))
            if hyper:
                rankings.append(hyper)
            if coe:
                rankings.append(coe)

        fused = self.fuser.fuse([r for r in rankings if r]) if rankings else []
        tokens = budget = None
        if leg == "packer" and fused:
            packed = self.packer.process(question, fused[:POOL_K])
            tokens = sum(_heuristic_tokens(c.text) for c in packed)
            budget = PACK_BUDGET
            fused = packed
        elapsed = (time.perf_counter() - started) * 1000.0
        return LegOutcome(
            ranked=[c.id for c in fused],
            latency_ms=elapsed,
            tokens_packed=tokens,
            budget=budget,
        )

    def _ablation(self, vector: Any, leg: str, started: float) -> LegOutcome:
        """One ablation leg: a SINGLE-signal run against the alternative vector source.

        It fuses nothing, exactly like the ``dense`` rung it is the counterfactual of — which is
        what makes the two numbers subtractable. An absent backend yields an empty ranking rather
        than falling back to the enriched vectors: a leg with no content-only sidecar must report
        nothing, never the very number it exists to contrast with.
        """
        backend = {"dense_content_only": self.dense_content_only}[leg]
        ranked: list[str] = []
        if backend is not None:
            hits = backend.search(
                RetrievalQuery(
                    mode=Capability.DENSE,
                    vector=tuple(float(x) for x in vector),
                    top_k=POOL_K,
                )
            )
            ranked = [c.id for c in hits]
        return LegOutcome(ranked=ranked, latency_ms=(time.perf_counter() - started) * 1000.0)

    def _graph_leg(self, question: str) -> list[Candidate]:
        if self.graph is None or self.seeder is None:
            return []
        seeds = self.seeder.seeds(question)
        if not seeds:
            return []
        return self.graph.search(
            RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=seeds, top_k=POOL_K)
        )

    def _hypergraph_legs(
        self, question: str, *, want_coe: bool
    ) -> tuple[list[Candidate], list[Candidate]]:
        if self.kv_path is None:
            return ([], [])
        with HyperedgeKV.open(str(self.kv_path)) as kv:
            signals = hypergraph_signals(kv, alias_surfaces=query_ngrams(question))
            hyper = self._chunks_from_hyperedges(
                sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
            )
            coe: list[Candidate] = []
            if want_coe and signals.rho:
                selection = coe_select(signals, budget=COE_BUDGET)
                coe = self._chunks_from_hyperedges(selection.selected)
        return (hyper, coe)

    def _chunks_from_hyperedges(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        rows = self.export.rows
        seen: set[str] = set()
        out: list[Candidate] = []
        for hyperedge_id in hyperedge_ids:
            chunk_id = self.hyperedge_chunks.get(hyperedge_id)
            if chunk_id is None or chunk_id in seen:
                continue
            position = self.export.index_of(chunk_id)
            if position is None:
                continue
            seen.add(chunk_id)
            out.append(
                Candidate(
                    id=chunk_id, score=1.0 / (1 + len(out)), text=rows[position].content
                )
            )
            if len(out) >= POOL_K:
                break
        return out


def _heuristic_tokens(text: str) -> int:
    """The packer's own dependency-free token estimate, reused for the packed-tokens report."""
    from latence_retrieval.pack import HeuristicTokenCounter

    return HeuristicTokenCounter().count(text)


def build_runner(
    export: Export,
    legs: Sequence[str],
    *,
    content_only_vectors: Any | None = None,
) -> Runner:
    """Assemble only the components the requested ``legs`` actually need.

    ``content_only_vectors`` is the matrix :func:`~benchmark.s5.corpus.read_vector_sidecar`
    returned for this export — supplied by the caller rather than loaded here, because the caller
    is the one that already knows the corpus row ids the sidecar has to align with.
    """
    root = export.root
    wanted = set(legs)
    dense = DenseBackend(export) if export.embeddings is not None else None
    dense_content_only = None
    if content_only_vectors is not None and wanted & set(ABLATION_LEG_NAMES):
        dense_content_only = DenseBackend(
            export, matrix=content_only_vectors, source=CONTENT_ONLY_VECTORS
        )
    bm25 = None
    if wanted & {"bm25", "kg", "hyperedges", "coe", "packer"} and (
        root / "bm25-stats.json"
    ).is_file():
        bm25 = Bm25Backend(export)
    graph = None
    seeder = None
    if wanted & {"kg", "hyperedges", "coe", "packer"} and (root / "graph-edges.parquet").is_file():
        graph = GraphRetriever(
            graph_source=EdgeGraphSource(root / "graph-edges.parquet"),
            document_source=ChunkGraphDocumentSource(export),
        )
        seeder = AliasQuerySeeder(node_alias_map(root / "graph-nodes.parquet"))
    kv_path = None
    hyperedges: dict[str, str] = {}
    if wanted & {"hyperedges", "coe", "packer"} and (root / "graph-hyperedges.kv").is_file():
        kv_path = root / "graph-hyperedges.kv"
        hyperedges = hyperedge_chunk_map(root / "graph-hyperedges.parquet")
    return Runner(
        export=export,
        dense=dense,
        bm25=bm25,
        graph=graph,
        kv_path=kv_path,
        dense_content_only=dense_content_only,
        hyperedge_chunks=hyperedges,
        seeder=seeder,
    )


def missing_inputs(export_dir: Path, leg: str) -> list[str]:
    """Artifacts ``leg`` needs that ``export_dir`` does not have — a refusal, never a fake run."""
    if leg in ABLATION_LEG_NAMES:
        # Not cumulative: an ablation leg is one rung re-run against one other input, so it needs
        # the corpus table (rows and ids) and its own vector file, nothing from the rungs above.
        return [
            f
            for f in ("corpus.parquet", CONTENT_ONLY_VECTORS)
            if not (export_dir / f).is_file()
        ]
    needs: dict[str, tuple[str, ...]] = {
        "dense": ("corpus.parquet",),
        "bm25": ("bm25-stats.json", "bm25-postings.parquet"),
        "kg": ("graph-edges.parquet", "graph-nodes.parquet"),
        "hyperedges": ("graph-hyperedges.kv", "graph-hyperedges.parquet"),
        "coe": ("graph-hyperedges.kv",),
        "packer": (),
    }
    cumulative: list[str] = []
    for name in LEG_NAMES:
        cumulative.extend(needs[name])
        if name == leg:
            break
    return [f for f in dict.fromkeys(cumulative) if not (export_dir / f).is_file()]


def has_embeddings(export_dir: Path) -> bool:
    """Whether the export's corpus table carries the dense ``embedding`` column."""
    return "embedding" in pq.read_schema(export_dir / "corpus.parquet").names  # type: ignore[no-untyped-call]


__all__ = [
    "ABLATION_BASELINE",
    "ABLATION_LEG_NAMES",
    "ALL_LEG_NAMES",
    "COE_BUDGET",
    "LEG_NAMES",
    "PACK_BUDGET",
    "POOL_K",
    "Bm25Backend",
    "ChunkGraphDocumentSource",
    "DenseBackend",
    "EdgeGraphSource",
    "LegOutcome",
    "Runner",
    "build_runner",
    "has_embeddings",
    "missing_inputs",
    "query_ngrams",
    "vector_source",
]
