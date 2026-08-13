"""latence-retrieval: stateless, query-time retrieval-quality tooling (ADR-0048/0049, #159).

The downstream, stateless half of the framework's retrieval tooling. It holds **no index** and runs
**no first-stage search** (ADR-0048): it only transforms candidate lists a caller's own engine
already fetched. This foundation ticket ships the seam every query-time capability builds on:

* :class:`~latence_retrieval.contracts.Candidate` — the retrieved-hit contract (id/score/text/
  metadata), round-tripping losslessly through dict + JSON.
* :class:`~latence_retrieval.contracts.Capability` /
  :class:`~latence_retrieval.contracts.RetrievalQuery` — the capability set a store declares and
  the request it is handed.
* :class:`~latence_retrieval.backend.RetrievalBackend` — the bring-your-own-engine seam, with two
  adapters: the faithful offline :class:`~latence_retrieval.testing.FakeBackend` for tests and the
  real reference :class:`~latence_retrieval.qdrant.QdrantBackend` over Qdrant (the example an
  adopter copies for their own store — ``{dense, sparse, filter}``, behind the ``[backend-qdrant]``
  extra, ADR-0052/#173).
* :class:`~latence_retrieval.processor.CandidateProcessor` +
  :class:`~latence_retrieval.processor.IdentityProcessor` — the query-time transform seam and its
  trivial reference, resolved by config through the
  :class:`~latence_retrieval.registry.RetrievalRegistry` under the new ``latence.retrieval``
  entry-point group (separate from the pipeline's ``latence.providers``, ADR-0049).
* :class:`~latence_retrieval.fuser.Fuser` + :class:`~latence_retrieval.fuser.RrfFuser` /
  :class:`~latence_retrieval.fuser.WeightedFuser` — the ``fuse`` seam (merge *N* ranked lists → one;
  RRF or weighted-score, selectable by config through
  :func:`~latence_retrieval.registry.resolve_fuser`).
* :class:`~latence_retrieval.rerank.Reranker` + the
  :class:`~latence_retrieval.rerank.CrossEncoderModel` seam
  (:class:`~latence_retrieval.rerank.SentenceTransformersCrossEncoder` real +
  :class:`~latence_retrieval.testing.FakeCrossEncoder` fake) — the ``rerank`` transform: re-score
  one candidate list with a config-selected, license-verified cross-encoder and reorder it. It is a
  :class:`~latence_retrieval.processor.CandidateProcessor`, resolved by config through
  :func:`~latence_retrieval.registry.resolve_processor` (``processor.rerank``).
* :class:`~latence_retrieval.pack.Packer` + the :class:`~latence_retrieval.pack.TokenCounter` seam
  (:class:`~latence_retrieval.pack.HeuristicTokenCounter` pure-Python day-one default +
  :class:`~latence_retrieval.pack.TiktokenCounter` real, behind the ``[pack]`` extra) — the ``pack``
  transform: select a token-budgeted subset maximising coverage/relevance minus redundancy (the
  ported, Apache-2.0 Knapsack QKP context-packer), with optional richness signals that degrade
  gracefully. A :class:`~latence_retrieval.processor.CandidateProcessor`, resolved by config through
  :func:`~latence_retrieval.registry.resolve_processor` (``processor.pack``).
* :class:`~latence_retrieval.expand.Expander` — the ``expand`` transform: 1-hop graph-augmented
  rescoring/reordering over the KG columns the pipeline denormalised onto each chunk
  (``kg_node_ids`` / ``neighbor_node_ids`` / ``kg_node_community`` / ``kg_node_centrality``). A
  pure-Python :class:`~latence_retrieval.processor.CandidateProcessor` — **no graph database**,
  works on any store (ADR-0051's default 1-hop path), resolved by config through
  :func:`~latence_retrieval.registry.resolve_processor` (``processor.expand``).
* :class:`~latence_retrieval.multihop.MultiHopExpander` + the
  :class:`~latence_retrieval.multihop.GraphSource` seam
  (:class:`~latence_retrieval.multihop.DuckDBGraphSource` zero-ops default over the emitted graph
  parquet, :class:`~latence_retrieval.multihop.NetworkxGraphSource` small-graph reference,
  :class:`~latence_retrieval.multihop.Neo4jGraphSource` optional enterprise backend +
  :class:`~latence_retrieval.testing.FakeGraphSource` faithful fake) — the ``expand`` **multi-hop**
  path (ADR-0051): rescore a candidate list by hop-decayed proximity to the seeds' bounded
  neighbourhood, traversed **read-only per call over ``graph-edges.parquet``** with no server. A
  :class:`~latence_retrieval.processor.CandidateProcessor` that **accepts** a
  :class:`~latence_retrieval.multihop.GraphSource` (dependency injection), resolved by config
  through
  :func:`~latence_retrieval.registry.resolve_processor` (``processor.multihop_expand``); the DuckDB
  /
  networkx / Neo4j engines ride the ``[graph-duckdb]`` / ``[graph-networkx]`` / ``[graph-neo4j]``
  extras and are imported lazily.
* :class:`~latence_retrieval.graph_retrieval.GraphRetriever` +
  :class:`~latence_retrieval.graph_retrieval.CompositeBackend` — the graph **retrieval leg**
  (T-RET-2): a :class:`~latence_retrieval.backend.RetrievalBackend` declaring
  :attr:`~latence_retrieval.contracts.Capability.GRAPH` that seeds from the query's linked entity
  nodes, traverses the emitted graph to a bounded depth through the
  :class:`~latence_retrieval.multihop.GraphSource` seam, resolves the reached nodes back to chunks
  through a :class:`~latence_retrieval.graph_retrieval.GraphDocumentSource`
  (:class:`~latence_retrieval.graph_retrieval.DuckDBGraphDocumentSource` real over the emitted
  chunk parquet + :class:`~latence_retrieval.testing.FakeGraphDocumentSource` faithful fake), and
  returns its own ranked list that enters **fusion as a peer** of dense/sparse/bm25.
  ``CompositeBackend`` presents the customer's store and the graph leg to the orchestrator as one
  backend (union of modes; ``FILTER`` only when every member can serve it). This is the only
  architecture in which "traversal recovered a document dense+sparse never retrieved" is
  expressible — the post-fusion expanders below are kept, but they can *reorder* only.
* :class:`~latence_retrieval.bm25.Bm25Rescorer` — the ``bm25`` transform: **exact** Okapi BM25
  rescoring of a caller-supplied candidate set (IDF from supplied corpus ``stats`` or the candidate
  pool), query-term-driven so it stays fast on large candidate sets — **never** a first-stage index
  (ADR-0048). A :class:`~latence_retrieval.processor.CandidateProcessor`, resolved by config through
  :func:`~latence_retrieval.registry.resolve_processor` (``processor.bm25``).
* :class:`~latence_retrieval.maxsim.MaxSimReranker` + the
  :class:`~latence_retrieval.maxsim.MultiVectorQueryEncoder` seam
  (:class:`~latence_retrieval.maxsim.HashingMultiVectorQueryEncoder` real, reusing the index-time
  ``multivector.hashing`` embedder +
  :class:`~latence_retrieval.testing.FakeMultiVectorEmbedder` fake) and the reference
  :func:`~latence_retrieval.maxsim.max_sim` scorer — the **experimental** MaxSim (late-interaction)
  rerank path (ADR-0050): re-score one candidate list by exact MaxSim over the raw multi-vectors the
  caller loaded from the ``multivectors.parquet`` sidecar, and query-side multi-vector generation
  matching the index-time contract. A :class:`~latence_retrieval.processor.CandidateProcessor`,
  resolved by config through :func:`~latence_retrieval.registry.resolve_processor`
  (``processor.maxsim``); the query encoder resolves through
  :func:`~latence_retrieval.registry.resolve_multivector_encoder` (``query_multivector.hashing``).
  Off by default; ``latence-core`` rides the optional ``[multivector]`` extra and is imported
  lazily.
* :class:`~latence_retrieval.sparse.SpladeQueryEncoder` +
  :class:`~latence_retrieval.sparse.SparseQueryEncoder` — query-side sparse generation: the
  ``query text → SparseVector`` transform that **reuses** the ADR-0053 index-time SPLADE composition
  (:class:`~latence_splade.provider.SpladeCompositionEmbedder`) so the query lands in the same
  sparse space as the corpus. Resolved by config through
  :func:`~latence_retrieval.registry.resolve_sparse_encoder` (``query_sparse.splade`` — named apart
  from the index-time ``sparse.splade`` Provider so the two seams stay disjoint, ADR-0049);
  ``latence-splade`` rides the optional ``[sparse]`` extra and is imported lazily.
* :func:`~latence_retrieval.orchestrator.retrieve` + its
  :class:`~latence_retrieval.orchestrator.QueryBundle` /
  :class:`~latence_retrieval.orchestrator.OrchestrationPipeline` /
  :class:`~latence_retrieval.orchestrator.RetrievalResult` — the **Level-2 orchestrator** (ADR-0052,
  #172): the opinionated, config-overridable ``fetch-per-modality → fuse → processors`` pipeline
  around a caller's :class:`~latence_retrieval.backend.RetrievalBackend`. It composes **only** the
  legs the backend's declared capabilities and the query's emitted signals support and drops the
  rest gracefully, running the backend's ``search`` as the single (stateless, caller-owned) fetch
  (ADR-0048). The recipe accepts *built* fuser/processors (DI) or resolves them from operator config
  through :meth:`~latence_retrieval.orchestrator.OrchestrationPipeline.from_config`; a planning
  failure (no runnable leg, or an unserveable filter) is a loud
  :class:`~latence_retrieval.errors.OrchestrationError`.
* :class:`~latence_retrieval.rescue.GraphRescueRetrieval` — the **graph-as-rescue** production
  composition (S5b): base dense+BM25 RRF (those two only, max-normalised), an
  :class:`~latence_retrieval.expand.Expander` graph rescore of the base, the graph lanes
  (:class:`~latence_retrieval.graph_retrieval.GraphRetriever` traversal +
  :func:`~latence_retrieval.hypergraph.hypergraph_signals`) **appended** below the base and
  deduplicated — never fused into it — and a token-budget selector
  (:class:`~latence_retrieval.pack.Packer` knapsack or :func:`~latence_retrieval.coe.select`)
  over the union, all behind a per-query policy gate (``graph_mode: off|auto|force``) whose
  :class:`~latence_retrieval.rescue.GateDecision` rides every
  :class:`~latence_retrieval.rescue.RescueResult`. Engines and signals are the caller's
  (ADR-0048); every default is the measured S5b operating point; zero LLM calls on the path.
"""

from __future__ import annotations

from latence_retrieval.backend import RetrievalBackend
from latence_retrieval.bm25 import Bm25Rescorer
from latence_retrieval.chains import (
    Chain,
    ChainAssembly,
    ChainStep,
    assemble_chains,
)
from latence_retrieval.coe import (
    CoeConfig,
    CoeSelection,
    EdgeFeatures,
    HeuristicUtility,
    SelectionState,
    UtilityModel,
    select,
)
from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery
from latence_retrieval.errors import (
    BackendCapabilityError,
    BackendDimensionError,
    BackendSearchError,
    CandidateProcessorError,
    GraphSourceError,
    MaxSimError,
    OrchestrationError,
    PackTokenizerError,
    QuerySignalError,
    RerankModelError,
    RetrievalError,
)
from latence_retrieval.expand import Expander
from latence_retrieval.fuser import Fuser, RrfFuser, WeightedFuser
from latence_retrieval.graph_retrieval import (
    AliasQuerySeeder,
    CompositeBackend,
    DuckDBGraphDocumentSource,
    GraphDocument,
    GraphDocumentSource,
    GraphRetriever,
    build_graph_document_source,
)
from latence_retrieval.hypergraph import (
    HyperedgeKV,
    HypergraphConfig,
    HypergraphSignals,
    Propagator,
    RWRPropagator,
    hypergraph_signals,
)
from latence_retrieval.kg_columns import KgColumnKeys, KgColumns
from latence_retrieval.maxsim import (
    HashingMultiVectorQueryEncoder,
    MaxSimReranker,
    MultiVectorQueryEncoder,
    max_sim,
)
from latence_retrieval.multihop import (
    DuckDBGraphSource,
    GraphSource,
    MultiHopExpander,
    Neo4jGraphSource,
    NetworkxGraphSource,
    build_graph_source,
)
from latence_retrieval.orchestrator import (
    OrchestrationPipeline,
    QueryBundle,
    RetrievalResult,
    retrieve,
)
from latence_retrieval.pack import (
    HeuristicTokenCounter,
    Packer,
    TiktokenCounter,
    TokenCounter,
)
from latence_retrieval.processor import CandidateProcessor, IdentityProcessor
from latence_retrieval.qdrant import QdrantBackend
from latence_retrieval.registry import (
    ENTRY_POINT_GROUP,
    RetrievalRegistry,
    resolve_fuser,
    resolve_multivector_encoder,
    resolve_processor,
    resolve_sparse_encoder,
)
from latence_retrieval.rerank import (
    CrossEncoderModel,
    Reranker,
    SentenceTransformersCrossEncoder,
)
from latence_retrieval.rescue import (
    CandidateDecorator,
    GateDecision,
    GraphRescueRetrieval,
    HyperedgeDocumentSource,
    KgColumnDecorator,
    RescueResult,
)
from latence_retrieval.sparse import SparseQueryEncoder, SpladeQueryEncoder
from latence_retrieval.tuning import (
    FuserComparisonRow,
    FusionRun,
    FusionScore,
    WeightSweep,
    WeightSweepRow,
    compare_fusers,
    evaluate_fusion,
    format_comparison_table,
    sweep_fusion_weights,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "AliasQuerySeeder",
    "BackendCapabilityError",
    "BackendDimensionError",
    "BackendSearchError",
    "Bm25Rescorer",
    "Candidate",
    "CandidateDecorator",
    "CandidateProcessor",
    "CandidateProcessorError",
    "Capability",
    "Chain",
    "ChainAssembly",
    "ChainStep",
    "CoeConfig",
    "CoeSelection",
    "CompositeBackend",
    "CrossEncoderModel",
    "DuckDBGraphDocumentSource",
    "DuckDBGraphSource",
    "EdgeFeatures",
    "Expander",
    "HyperedgeKV",
    "HypergraphConfig",
    "HypergraphSignals",
    "Fuser",
    "FuserComparisonRow",
    "FusionRun",
    "FusionScore",
    "GateDecision",
    "GraphDocument",
    "GraphDocumentSource",
    "GraphRescueRetrieval",
    "GraphRetriever",
    "GraphSource",
    "GraphSourceError",
    "HyperedgeDocumentSource",
    "HashingMultiVectorQueryEncoder",
    "HeuristicTokenCounter",
    "HeuristicUtility",
    "IdentityProcessor",
    "KgColumnDecorator",
    "KgColumnKeys",
    "KgColumns",
    "MaxSimError",
    "MaxSimReranker",
    "MultiHopExpander",
    "MultiVectorQueryEncoder",
    "Neo4jGraphSource",
    "NetworkxGraphSource",
    "OrchestrationError",
    "OrchestrationPipeline",
    "PackTokenizerError",
    "Packer",
    "Propagator",
    "RWRPropagator",
    "hypergraph_signals",
    "QdrantBackend",
    "QueryBundle",
    "QuerySignalError",
    "RerankModelError",
    "Reranker",
    "RescueResult",
    "RetrievalBackend",
    "RetrievalError",
    "RetrievalQuery",
    "RetrievalRegistry",
    "RetrievalResult",
    "RrfFuser",
    "SelectionState",
    "SentenceTransformersCrossEncoder",
    "SparseQueryEncoder",
    "SpladeQueryEncoder",
    "TiktokenCounter",
    "TokenCounter",
    "UtilityModel",
    "WeightSweep",
    "WeightSweepRow",
    "WeightedFuser",
    "assemble_chains",
    "build_graph_document_source",
    "build_graph_source",
    "compare_fusers",
    "evaluate_fusion",
    "format_comparison_table",
    "max_sim",
    "resolve_fuser",
    "resolve_multivector_encoder",
    "resolve_processor",
    "resolve_sparse_encoder",
    "retrieve",
    "select",
    "sweep_fusion_weights",
]
