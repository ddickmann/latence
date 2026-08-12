# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
#
# DELIBERATE, BOUNDED EXCEPTION -- not a header nobody updated at the relicence. This
# repository's outbound licence is PolyForm-Noncommercial-1.0.0 (root LICENSE); this module is
# one of seven the copyright holder keeps under Apache-2.0 ON PURPOSE, so the query-time
# retrieval primitives stay permissively reusable -- commercially included -- by anyone who
# wants only them. The grant is his to make and safe to make: sole copyright holder, no
# third-party and no copyleft code in this file. The exact set is authoritative in LICENSE,
# section "SCOPE: THE RETAINED APACHE-2.0 MODULES", and pinned by
# packages/latence-core/tests/test_publish_readiness.py, which fails the build if an eighth
# file acquires the grant or one of the seven loses it.
"""The graph **retrieval leg**: :class:`GraphRetriever` + :class:`CompositeBackend` (T-RET-2).

**The defect this module closes.** Until now the knowledge graph could only *reorder*. Both
:class:`~latence_retrieval.expand.Expander` and
:class:`~latence_retrieval.multihop.MultiHopExpander` are
:class:`~latence_retrieval.processor.CandidateProcessor`\\ s, and the orchestrator runs processors
**after** fusion (``orchestrator.py:266-269``). They add a hop-decayed bonus to already-fused scores
and, structurally, **cannot introduce a candidate the first stage missed**
(``multihop.py:621-628``). So the benchmark premise *"graph traversal recovers documents dense and
sparse never retrieved"* was not merely unproven — the architecture could not **express** it, and
every KG "win" it could produce was a reordering of a pool the graph did not choose.

**The fix, in one sentence:** the graph becomes a *first-stage leg*. :class:`GraphRetriever` is a
:class:`~latence_retrieval.backend.RetrievalBackend` declaring
:attr:`~latence_retrieval.contracts.Capability.GRAPH`: it seeds from the query's entity nodes,
traverses the emitted graph to a **bounded** depth through the existing
:class:`~latence_retrieval.multihop.GraphSource` seam, resolves the reached nodes back to documents
through a :class:`GraphDocumentSource`, and returns its **own ranked list**, which then enters
fusion as a *peer* of dense/sparse/bm25. The post-fusion expanders are kept unchanged — rescoring
after fusion is a legitimate *second* use of the graph, it is just not evidence for the recovery
claim.

**Where the seeds come from.** The query's entity node ids ride the query as data
(:attr:`~latence_retrieval.contracts.RetrievalQuery.seed_node_ids` /
:attr:`~latence_retrieval.orchestrator.QueryBundle.seed_node_ids`), exactly as the dense vector and
the sparse terms do: query-signal-gen is the caller's (ADR-0052) and entity linking is a *pipeline*
capability (the Disambiguate stage), not something this stateless library re-implements. The shipped
:class:`AliasQuerySeeder` is the dependency-free reference for callers who only have an alias map —
a deterministic longest-match over the query's tokens, no model, no network.

**Statelessness holds (ADR-0048/0051).** The retriever holds no index: the traversal is the
``GraphSource``'s (read-only, per call, over the emitted ``graph-edges.parquet`` — zero ops) and the
node→document resolution is the ``GraphDocumentSource``'s (read-only, per call, over the emitted
chunk table). Identical inputs always yield an identical ranked list.

**Composing it with the customer's store.** A vector store cannot traverse a graph, and the graph
leg cannot do ANN — so they are two backends, and :class:`CompositeBackend` presents them to the
orchestrator as one: it declares the **union** of its members' query modes and routes each query to
the member that declares its mode. It declares ``FILTER`` only when *every* member can serve a
predicate, so the orchestrator's refuse-if-unserveable path (``orchestrator.py:291-297``) still
fires **before any fetch** rather than a filter being silently dropped on one leg.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from latence_retrieval.contracts import Candidate, Capability
from latence_retrieval.errors import BackendCapabilityError, GraphSourceError
from latence_retrieval.multihop import _validate_max_hops

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from latence_retrieval.backend import RetrievalBackend
    from latence_retrieval.contracts import RetrievalQuery
    from latence_retrieval.multihop import GraphSource

_DEFAULT_MAX_HOPS = 2
_DEFAULT_HOP_DECAY = 0.5
_DEFAULT_SEED_SCORE = 1.0
_CONFIG_KEYS = frozenset(
    {"max_hops", "hop_decay", "seed_score", "graph", "documents", "include_seed_documents"}
)
# Unicode alphanumeric runs (underscore excluded) — the same tokenizer the shipped fakes and the
# BM25 rescorer use, so an alias map behaves identically on non-Latin scripts.
_TOKEN = re.compile(r"[^\W_]+")


class GraphDocument(BaseModel):
    """One document (chunk) the graph can reach, with the KG nodes it mentions.

    The unit a :class:`GraphDocumentSource` returns: a stable ``id`` (the same chunk key the vector
    store uses, so fusion can merge the graph leg's hits with the dense/sparse legs' by id), the
    ``node_ids`` it carries (the emitted denormalised ``kg_node_ids`` column), and the ``text`` /
    ``metadata`` a downstream reranker, packer or filter needs. Frozen and ``extra="forbid"`` like
    every contract here, so a malformed row fails at the boundary rather than corrupting a rank
    list.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    node_ids: tuple[str, ...] = ()
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class GraphDocumentSource(Protocol):
    """A read-only, stateless node→document index: which chunks mention these KG nodes.

    The second half of the graph leg (the first is
    :class:`~latence_retrieval.multihop.GraphSource`, which answers *which nodes are near the
    seeds*). One deep method: given the node ids a traversal reached, return the documents that
    mention any of them. Every adapter (embedded DuckDB over the emitted chunk parquet, an
    in-memory fake, a customer's own store) hides its engine entirely behind this call.

    The contract every adapter honours: an empty ``node_ids`` returns ``[]`` (nothing to resolve, no
    engine touched); each document is returned **at most once** however many of its nodes matched; a
    node with no documents simply contributes nothing; the order is deterministic (ascending
    ``id``); and a runtime engine fault is a
    :class:`~latence_retrieval.errors.GraphSourceError` carrying no raw engine text. It holds
    nothing across calls (ADR-0048).
    """

    def documents(self, node_ids: Collection[str]) -> list[GraphDocument]:
        """Return the documents mentioning any of ``node_ids``, each at most once."""
        ...


class DuckDBGraphDocumentSource:
    """The zero-ops :class:`GraphDocumentSource`: embedded DuckDB over the emitted chunk parquet.

    The retrieval-side twin of :class:`~latence_retrieval.multihop.DuckDBGraphSource` (ADR-0051):
    an in-process, serverless DuckDB reads the parquet the pipeline **already emitted** — no graph
    database, no new artifact, no server to operate — and selects the rows whose denormalised
    ``kg_node_ids`` list intersects the traversal's node set (``list_has_any``). Each call opens a
    fresh in-memory connection, reads read-only, and closes it, so the source holds nothing between
    calls (statelessness is structural, ADR-0048).

    Args:
        chunks_uri: path to the emitted chunk parquet (the Export's row table). Required, non-empty.
        id_column: the stable chunk key — must be the **same** key the vector store uses, or the
            graph leg's hits will not merge with the dense/sparse legs' at fusion. Default
            ``"chunk_id"``.
        node_ids_column: the denormalised KG node-id list column. Default ``"kg_node_ids"``.
        text_column: the chunk text column. Default ``"content"`` (the Export column name).

    Packaging: ``duckdb`` (MIT, an embedded engine — no server, no model) rides this package's
    ``[graph-duckdb]`` extra and is imported lazily on first :meth:`documents`, so constructing the
    source — which a registry/config does — never imports it or needs the extra.
    """

    def __init__(
        self,
        chunks_uri: str,
        *,
        id_column: str = "chunk_id",
        node_ids_column: str = "kg_node_ids",
        text_column: str = "content",
    ) -> None:
        self._chunks_uri = _require_non_empty_str(chunks_uri, "chunks_uri")
        self._id_column = _require_non_empty_str(id_column, "id_column")
        self._node_ids_column = _require_non_empty_str(node_ids_column, "node_ids_column")
        self._text_column = _require_non_empty_str(text_column, "text_column")

    def documents(self, node_ids: Collection[str]) -> list[GraphDocument]:
        """Select the chunk rows whose node-id list intersects ``node_ids``; deterministic order."""
        wanted = list(dict.fromkeys(node_ids))
        if not wanted:
            return []
        duckdb = self._load()
        # The column names are validated identifiers the operator configures (never query input);
        # the parquet URI and the node list travel as bound parameters.
        sql = (
            f"SELECT {self._id_column} AS id, {self._node_ids_column} AS node_ids, "
            f"       {self._text_column} AS text "
            "  FROM read_parquet(?) "
            f" WHERE list_has_any({self._node_ids_column}, ?::VARCHAR[]) "
            " ORDER BY id"
        )
        connection = duckdb.connect(database=":memory:")
        try:
            rows = connection.execute(sql, [self._chunks_uri, wanted]).fetchall()
        except Exception as exc:
            msg = f"duckdb graph document resolution failed: {type(exc).__name__}."
            raise GraphSourceError(msg) from exc
        finally:
            connection.close()
        return [
            GraphDocument(
                id=str(row[0]),
                node_ids=tuple(str(n) for n in (row[1] or ())),
                text="" if row[2] is None else str(row[2]),
            )
            for row in rows
        ]

    @staticmethod
    def _load() -> Any:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            msg = (
                "The 'duckdb' package is required for the zero-ops graph retrieval leg over the "
                "emitted chunk parquet. Install it with: "
                "pip install 'latence-retrieval[graph-duckdb]'"
            )
            raise ImportError(msg) from exc
        return duckdb


class AliasQuerySeeder:
    """The dependency-free reference for turning query text into seed node ids.

    Entity linking proper is a *pipeline* capability (the Disambiguate stage), and query-signal-gen
    is the caller's (ADR-0052) — so seeds normally arrive as data on the
    :class:`~latence_retrieval.orchestrator.QueryBundle`. This is the honest fallback for a caller
    who has only the emitted alias map: a deterministic **longest-match** scan over the query's
    tokens against ``aliases`` (surface form → node id), case-insensitive, no model and no network.
    It is a helper, not a seam: it invents no linking accuracy it does not have, and a caller with a
    real linker simply does not use it.

    Args:
        aliases: surface form → node id. Multi-word surfaces are matched as token runs, longest
            first, and a matched run is consumed so one mention seeds one node.
        max_span: the longest token run considered for a multi-word alias. Default 5.
    """

    def __init__(self, aliases: Mapping[str, str], max_span: int = 5) -> None:
        if not isinstance(aliases, Mapping):
            msg = f"AliasQuerySeeder 'aliases' must be a mapping, got {type(aliases).__name__}."
            raise TypeError(msg)
        if not isinstance(max_span, int) or isinstance(max_span, bool) or max_span < 1:
            msg = f"AliasQuerySeeder 'max_span' must be a positive integer, got {max_span!r}."
            raise ValueError(msg)
        self._aliases = {
            " ".join(_TOKEN.findall(str(surface).lower())): str(node)
            for surface, node in aliases.items()
        }
        self._aliases.pop("", None)
        self._max_span = max_span

    def seeds(self, text: str) -> tuple[str, ...]:
        """The distinct node ids the query text mentions, in first-occurrence order."""
        if not isinstance(text, str):
            msg = f"AliasQuerySeeder text must be a string, got {type(text).__name__}."
            raise TypeError(msg)
        tokens = _TOKEN.findall(text.lower())
        found: list[str] = []
        index = 0
        while index < len(tokens):
            for span in range(min(self._max_span, len(tokens) - index), 0, -1):
                key = " ".join(tokens[index : index + span])
                node = self._aliases.get(key)
                if node is not None:
                    found.append(node)
                    index += span
                    break
            else:
                index += 1
        return tuple(dict.fromkeys(found))


class GraphRetriever:
    """A first-stage **graph retrieval leg** — a :class:`~latence_retrieval.backend.\
RetrievalBackend` declaring ``GRAPH`` (T-RET-2).

    :meth:`search` takes the query's seed entity nodes, asks the injected
    :class:`~latence_retrieval.multihop.GraphSource` for their **bounded** neighbourhood, resolves
    every reached node (plus the seeds themselves) back to documents through the injected
    :class:`GraphDocumentSource`, scores each document by the **closest** hop at which any of its
    nodes was reached — a seed-mentioning document scores ``seed_score``, a hop-*h* document
    ``seed_score × hop_decay**h`` — and returns the list in descending score (ascending ``id``
    tiebreak), truncated to ``query.top_k``.

    That list is a **peer** of the dense/sparse/bm25 lists, so it can carry a document those legs
    never returned into fusion. This is the whole architectural point: a post-fusion
    ``CandidateProcessor`` structurally cannot do that.

    Both dependencies are **accepted, never constructed** (dependency injection, exactly as
    :class:`~latence_retrieval.multihop.MultiHopExpander` accepts a ``GraphSource``): a test hands
    it the shipped faithful fakes, production hands it ``{"graph": {...}, "documents": {...}}``
    config blocks that build the DuckDB adapters lazily on first search.

    Config:
        max_hops: int — the **bounded** traversal depth. Positive int within the ceiling (``64``),
            default ``2``.
        hop_decay: float — the per-hop geometric decay of the retrieval score, in ``(0, 1]``.
            Default ``0.5``.
        seed_score: float — the score of a document mentioning a **seed** node (hop 0). ``> 0``,
            default ``1.0``. The scale is arbitrary and query-independent, which is what a fuser
            needs.
        include_seed_documents: bool — whether hop-0 documents (those mentioning the seeds
            themselves) are returned at all. Default ``True``; set ``False`` for a *pure expansion*
            leg that only contributes what the traversal added.
        graph: mapping — a :func:`~latence_retrieval.multihop.build_graph_source` spec, used to
            build the traversal source lazily when none is injected.
        documents: mapping — a :func:`build_graph_document_source` spec, used to build the
            node→document index lazily when none is injected.

    It declares ``{GRAPH, FILTER}``: a metadata predicate is applied to the resolved documents
    **before** truncation (so a filtered graph search cannot silently return fewer hits than it
    should). A query whose mode is not ``GRAPH`` is a
    :class:`~latence_retrieval.errors.BackendCapabilityError`, exactly as a real store rejects a
    mode it never declared.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        graph_source: GraphSource | None = None,
        document_source: GraphDocumentSource | None = None,
    ) -> None:
        cfg = dict(config or {})
        unknown = set(cfg) - _CONFIG_KEYS
        if unknown:
            msg = f"GraphRetriever: unknown config key(s) {sorted(unknown)}."
            raise ValueError(msg)

        self._max_hops = _validate_max_hops_named(cfg.get("max_hops", _DEFAULT_MAX_HOPS))
        self._hop_decay = _unit_interval(cfg.get("hop_decay", _DEFAULT_HOP_DECAY), "hop_decay")
        self._seed_score = _positive_number(cfg.get("seed_score", _DEFAULT_SEED_SCORE),
                                            "seed_score")
        include_seeds = cfg.get("include_seed_documents", True)
        if not isinstance(include_seeds, bool):
            msg = (
                "GraphRetriever 'include_seed_documents' must be a boolean, "
                f"got {include_seeds!r}."
            )
            raise ValueError(msg)
        self._include_seed_documents = include_seeds

        graph_spec = cfg.get("graph")
        if graph_spec is not None and not isinstance(graph_spec, dict):
            msg = f"GraphRetriever 'graph' config must be a mapping if present, got {graph_spec!r}."
            raise TypeError(msg)
        documents_spec = cfg.get("documents")
        if documents_spec is not None and not isinstance(documents_spec, dict):
            msg = (
                "GraphRetriever 'documents' config must be a mapping if present, "
                f"got {documents_spec!r}."
            )
            raise TypeError(msg)
        self._graph_spec: dict[str, Any] | None = dict(graph_spec) if graph_spec else None
        self._documents_spec: dict[str, Any] | None = (
            dict(documents_spec) if documents_spec else None
        )
        self._graph_source = graph_source
        self._document_source = document_source

    @property
    def capabilities(self) -> frozenset[Capability]:
        """``{GRAPH, FILTER}`` — it retrieves by traversal and can honour a metadata predicate."""
        return frozenset({Capability.GRAPH, Capability.FILTER})

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Traverse from ``query.seed_node_ids`` and return the reached documents, ranked.

        Resolution is **tiered** whenever the hop scores are strictly decreasing (every
        ``hop_decay < 1.0`` that does not underflow): the reached nodes are resolved to
        documents closest hop first, and resolution STOPS as soon as ``top_k`` hits are
        ranked — a document first reachable at hop *h* mentions no closer node (a
        contract-honouring :class:`GraphDocumentSource` would have returned it at that
        closer tier), tier scores strictly decrease, and each tier arrives in ascending
        id, so the early stop returns byte-identical hits to ranking the full
        neighbourhood (ticket 01, parity-pinned on real exports). At real scale this is
        the difference between resolving ~10² and ~10⁵ documents per query. The
        document source is queried once per non-empty tier (≤ ``max_hops + 1`` calls)
        instead of once per search. Degenerate configs (``hop_decay == 1.0``, underflow)
        keep the exhaustive single-call path — same results, no early stop to reason
        about.
        """
        if query.mode is not Capability.GRAPH:
            msg = (
                f"graph retrieval leg does not declare capability {query.mode.value!r}; "
                "declared: ['filter', 'graph']."
            )
            raise BackendCapabilityError(msg)
        seeds = list(dict.fromkeys(query.seed_node_ids or ()))
        if not seeds:  # pragma: no cover - RetrievalQuery already rejects an empty seed set
            return []

        distances = dict(self._resolve_graph().neighbors(seeds, self._max_hops))
        # The seeds themselves are hop 0: a chunk that *mentions the query's own entity* is a
        # legitimate graph hit, not an expansion. `include_seed_documents=False` makes the leg pure
        # expansion (only what traversal added) for an ablation that isolates the recovered mass.
        for seed in seeds:
            distances.setdefault(seed, 0)
        reachable = {node: hop for node, hop in distances.items() if hop <= self._max_hops}
        if not self._include_seed_documents:
            reachable = {node: hop for node, hop in reachable.items() if hop > 0}
        if not reachable:
            return []

        if self._hop_scores_strictly_decrease():
            hits = self._tiered_hits(query, reachable)
        else:
            hits = self._exhaustive_hits(query, reachable)
        hits.sort(key=lambda c: (-c.score, c.id))
        return hits[: query.top_k]

    def _hop_scores_strictly_decrease(self) -> bool:
        """Whether every deeper hop scores strictly lower — the tiered path's guard.

        False for ``hop_decay == 1.0`` (all tiers tie, ranked purely by id across tiers)
        and for decays so small the score underflows to equal values within ``max_hops``.
        """
        scores = [self._seed_score * self._hop_decay**hop for hop in range(self._max_hops + 1)]
        return all(closer > deeper for closer, deeper in zip(scores, scores[1:], strict=False))

    def _tiered_hits(self, query: RetrievalQuery, reachable: Mapping[str, int]) -> list[Candidate]:
        """Resolve and rank tier by tier (closest hop first), stopping at ``top_k`` hits.

        Byte-identical to :meth:`_exhaustive_hits` under the strictly-decreasing guard —
        see :meth:`search`. The accumulated list is already in ``(-score, id)`` order, so
        the caller's sort is a stable no-op kept for uniformity with the exhaustive path.
        """
        source = self._resolve_documents()
        hits: list[Candidate] = []
        resolved: set[str] = set()
        for hop in sorted(set(reachable.values())):
            tier = sorted(node for node, distance in reachable.items() if distance == hop)
            for document in source.documents(tier):
                if document.id in resolved:
                    continue  # already ranked at a closer tier
                resolved.add(document.id)
                candidate = self._scored(document, reachable, query)
                if candidate is None:
                    continue
                hits.append(candidate)
                if len(hits) >= query.top_k:
                    return hits
        return hits

    def _exhaustive_hits(
        self, query: RetrievalQuery, reachable: Mapping[str, int]
    ) -> list[Candidate]:
        """Resolve the whole reached neighbourhood in one call and rank every document."""
        documents = self._resolve_documents().documents(sorted(reachable))
        return [
            candidate
            for document in documents
            if (candidate := self._scored(document, reachable, query)) is not None
        ]

    def _scored(
        self, document: GraphDocument, reachable: Mapping[str, int], query: RetrievalQuery
    ) -> Candidate | None:
        """``document`` as a scored hit, or ``None`` when it must not be ranked.

        Dropped when: none of its nodes was asked for (a real store would not return it,
        and scoring it would fabricate relevance); it is a seed document on a pure
        expansion leg; or it fails the metadata predicate. Filtering happens BEFORE
        truncation — filtering after top_k would silently return fewer (or zero) hits
        than the store holds, the same anti-false-green rule the orchestrator applies to
        a filter it cannot serve.
        """
        hops = [reachable[node] for node in document.node_ids if node in reachable]
        if not hops:
            return None
        if not self._include_seed_documents and min(hops) == 0:
            return None
        candidate = Candidate(
            id=document.id,
            score=self._seed_score * (self._hop_decay ** min(hops)),
            text=document.text,
            metadata=dict(document.metadata),
        )
        if query.filter and not _matches_filter(candidate, query.filter):
            return None
        return candidate

    def _resolve_graph(self) -> GraphSource:
        if self._graph_source is None:
            if not self._graph_spec:
                msg = (
                    "GraphRetriever needs a graph to traverse: pass a 'graph' source config (e.g. "
                    "{'source': 'duckdb', 'edges_uri': '.../graph-edges.parquet'}) or inject a "
                    "GraphSource."
                )
                raise ValueError(msg)
            from latence_retrieval.multihop import build_graph_source

            self._graph_source = build_graph_source(self._graph_spec)
        return self._graph_source

    def _resolve_documents(self) -> GraphDocumentSource:
        if self._document_source is None:
            if not self._documents_spec:
                msg = (
                    "GraphRetriever needs a node→document index: pass a 'documents' source config "
                    "(e.g. {'source': 'duckdb', 'chunks_uri': '.../chunks.parquet'}) or inject a "
                    "GraphDocumentSource."
                )
                raise ValueError(msg)
            self._document_source = build_graph_document_source(self._documents_spec)
        return self._document_source


_DOCUMENT_SOURCE_BUILDERS = ("duckdb",)


def build_graph_document_source(spec: Mapping[str, Any]) -> GraphDocumentSource:
    """Build a :class:`GraphDocumentSource` from a config block (the ``config["documents"]``).

    ``spec["source"]`` selects the adapter by name — currently ``"duckdb"`` (the zero-ops default
    over the emitted chunk parquet). The remaining keys are that adapter's constructor arguments.
    An unknown ``source`` fails loudly at wiring time, never at query time.
    """
    if not isinstance(spec, Mapping):
        msg = f"graph document source spec must be a mapping, got {type(spec).__name__}."
        raise TypeError(msg)
    source = spec.get("source")
    if source == "duckdb":
        return DuckDBGraphDocumentSource(
            _require_non_empty_str(spec.get("chunks_uri"), "chunks_uri"),
            id_column=str(spec.get("id_column", "chunk_id")),
            node_ids_column=str(spec.get("node_ids_column", "kg_node_ids")),
            text_column=str(spec.get("text_column", "content")),
        )
    msg = (
        f"unknown graph document source {source!r}; expected one of "
        f"{_DOCUMENT_SOURCE_BUILDERS} (or inject a GraphDocumentSource directly)."
    )
    raise ValueError(msg)


class CompositeBackend:
    """Present several :class:`~latence_retrieval.backend.RetrievalBackend`\\ s as one.

    A vector store cannot traverse a knowledge graph and a
    :class:`GraphRetriever` cannot run ANN — so they are two backends, and the orchestrator takes
    one. This composes them **honestly**:

    * :attr:`capabilities` is the **union** of its members' query modes, so the orchestrator plans
      one leg per mode any member can serve — that is how the graph leg becomes a fusion peer of
      dense/sparse/bm25 rather than a post-fusion rescorer.
    * ``FILTER`` is declared **only when every member declares it**. A composite that let one leg
      drop a predicate would return unfiltered hits from that leg, so instead the orchestrator's
      refuse-if-unserveable path (``orchestrator.py:291-297``) fires **before any fetch** — the
      loud failure the seam already promises for a filter no backend can honour.
    * :meth:`search` routes a query to the **first** member declaring its mode (declaration order is
      the operator's precedence, so the composition is deterministic), and raises
      :class:`~latence_retrieval.errors.BackendCapabilityError` when no member declares it — never
      a silent empty result.

    It holds nothing of its own (ADR-0048): every fetch is the member's, on the caller's engine.
    """

    def __init__(self, backends: Sequence[RetrievalBackend]) -> None:
        from latence_retrieval.backend import RetrievalBackend as _Backend

        members = list(backends)
        if not members:
            msg = "CompositeBackend requires at least one member backend."
            raise ValueError(msg)
        for member in members:
            if not isinstance(member, _Backend):
                msg = (
                    f"CompositeBackend member {type(member).__name__!r} does not satisfy the "
                    "RetrievalBackend seam (needs 'capabilities' and 'search(query)')."
                )
                raise TypeError(msg)
        self._members: tuple[RetrievalBackend, ...] = tuple(members)
        modes: set[Capability] = {
            capability
            for member in self._members
            for capability in member.capabilities
            if capability is not Capability.FILTER
        }
        if all(Capability.FILTER in member.capabilities for member in self._members):
            modes.add(Capability.FILTER)
        self._capabilities = frozenset(modes)

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The union of the members' query modes; ``FILTER`` iff *every* member can serve one."""
        return self._capabilities

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Route to the first member declaring ``query.mode``; its faults propagate unchanged."""
        for member in self._members:
            if query.mode in member.capabilities:
                return member.search(query)
        msg = (
            f"no composed backend declares capability {query.mode.value!r}; "
            f"declared: {sorted(c.value for c in self._capabilities)}."
        )
        raise BackendCapabilityError(msg)


# --- shared config parsing ---------------------------------------------------------------------


def _matches_filter(candidate: Candidate, predicate: Mapping[str, Any]) -> bool:
    """Exact per-field metadata match, mirroring a store's filter (missing field ⇒ no match)."""
    return all(candidate.metadata.get(key) == value for key, value in predicate.items())


def _validate_max_hops_named(value: object) -> int:
    try:
        return _validate_max_hops(value)  # type: ignore[arg-type]
    except ValueError as exc:
        msg = f"GraphRetriever 'max_hops' is invalid: {exc}"
        raise ValueError(msg) from None


def _unit_interval(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 < value <= 1):
        msg = f"GraphRetriever {name!r} must be a number in (0, 1], got {value!r}."
        raise ValueError(msg)
    return float(value)


def _positive_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        msg = f"GraphRetriever {name!r} must be a number > 0, got {value!r}."
        raise ValueError(msg)
    return float(value)


def _require_non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"{name!r} must be a non-empty string, got {value!r}."
        raise ValueError(msg)
    return value


__all__ = [
    "AliasQuerySeeder",
    "CompositeBackend",
    "DuckDBGraphDocumentSource",
    "GraphDocument",
    "GraphDocumentSource",
    "GraphRetriever",
    "build_graph_document_source",
]
