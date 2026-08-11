# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Graph-as-rescue retrieval: :class:`GraphRescueRetrieval` — the production composition (S5b).

The S5 benchmark ladder measured every graph leg *lowering* recall when graph results were fused
into the base ranking as a co-equal RRF list. The components were never the problem; the
*application* was (ticket ``s5b-retrieval-architecture-fix``). This module is the corrected
application as a reusable production component, built from the shipped query-time components only,
in the shape the framework's own seams and the graph-sidecar reference flow both describe — base
retrieval first, graph strictly additive on top, a solver spending the budget::

    1. BASE     dense + BM25, fused by RRF between THOSE TWO ONLY, max-normalised to 1.0.
                Two full-corpus first stages over the same corpus are peers, so rank fusion
                between them is legitimate. Nothing else enters this fusion.

    2. RESCORE  :class:`~latence_retrieval.expand.Expander` over the base list. It anchors on the
                top ``seed_top_k`` first-stage candidates and adds a bounded 1-hop graph bonus
                read off the KG columns denormalised onto every chunk (``kg_node_ids``,
                ``neighbor_node_ids``, ``kg_node_community``, ``kg_node_centrality``) — no graph
                database, ADR-0051's default path.

    3. RESCUE   the graph lanes — :class:`~latence_retrieval.graph_retrieval.GraphRetriever`
                traversal and :func:`~latence_retrieval.hypergraph.hypergraph_signals` — contribute
                candidates the base MISSED. They are fused with each other (co-equal graph lanes,
                same footing), then **appended** below the rescored base and deduplicated by id.
                The base ordering is never reordered by them. Entity resolution is seeded from the
                QUERY *and* the top first-stage documents' entities — the strongest entity signal
                available.

    4. SELECT   a token budget is spent over the deduped union:
                :class:`~latence_retrieval.pack.Packer` (the knapsack QKP with the graph feature as
                a ``value_signal`` and ``fill="backfill"``) or :func:`~latence_retrieval.coe.select`
                (chain-of-evidence) — or no selector at all (``selector="none"``), returning the
                ranked union.

**A policy gate decides whether the graph lane runs at all** (``graph_mode: off|auto|force``).
Known-item and navigational queries stay graph-off; relation cues, entity-heavy queries, low
dense/BM25 agreement and low first-stage confidence turn it on. Every decision is returned on the
result (:class:`GateDecision`), so the gate's hit rate is an observable number, never an assumption.

**Statelessness holds (ADR-0048).** The component owns *policy and composition only*: every fetch
runs on a caller-supplied :class:`~latence_retrieval.backend.RetrievalBackend`, every score is
computed by the shipped component named beside it, and nothing is held between queries — identical
inputs always yield an identical result. Query-signal-gen is the caller's, injected as data on the
:class:`~latence_retrieval.orchestrator.QueryBundle` (the dense ``vector``; the entity
``seed_node_ids`` from the caller's linker or the shipped
:class:`~latence_retrieval.graph_retrieval.AliasQuerySeeder`). There are **zero LLM calls**
anywhere on this path.

**Score scales, stated because they are load-bearing.** ``RrfFuser`` emits ``Σ w/(k+rank)`` —
about ``0.033`` at rank 0 for two lists — while an un-normalised expander bonus would let one graph
link outvote the entire base ranking, which is the very defect this module exists to remove. So the
base scores are **max-normalised to 1.0 first** (a strictly monotone transform: the base order is
untouched) and the expander bonuses default to measured fractions of that unit scale (hop ``0.05``,
community ``0.01``, centrality ``0.05``). The deduped union is then re-scored by its own rank as
``1/(rank_score_k + rank)`` — the same slowly-decaying shape the base already had, so a candidate
deep in the pool keeps enough value for the packer to consider it instead of the greedy stalling at
a fraction of the budget.

**Defaults are measured, not guessed.** Every default below is the S5b operating point measured on
the framework's own benchmark (commits 9b41d7d, e9d66c2, 6f3b379): the knapsack saturates ~5.5k
tokens so its budget is 10,000 with backfill; CoE keeps buying recall through 16k so its budget is
16,000; the seed caps and gate thresholds are the values the measured run shipped with.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from latence_retrieval.coe import select as coe_select
from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery
from latence_retrieval.expand import Expander
from latence_retrieval.fuser import RrfFuser
from latence_retrieval.hypergraph import HyperedgeKV, HypergraphSignals, hypergraph_signals
from latence_retrieval.pack import HeuristicTokenCounter, Packer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_retrieval.backend import RetrievalBackend
    from latence_retrieval.orchestrator import QueryBundle

#: The three graph modes (the sidecar's ``graph_mode``): ``off`` never pays for the graph lane,
#: ``auto`` runs the shipped policy per query, ``force`` runs the lane on every query (what makes
#: the gate's contribution separable from the lane's in an A/B).
GRAPH_MODES: tuple[str, ...] = ("off", "auto", "force")

#: The three selectors for step 4. ``packed`` is the knapsack QKP (precision operating point — it
#: saturates ~5.5k tokens and backfills the rest); ``coe`` is chain-of-evidence (recall operating
#: point — it keeps spending productively through 16k); ``none`` returns the ranked union.
SELECTORS: tuple[str, ...] = ("packed", "coe", "none")

#: Relation cues (the sidecar's shipped policy list, plus the two multi-hop connectives the
#: framework's own datasets are built on). A query carrying one of these is asking about a
#: RELATION, which is what the graph lane is for.
RELATION_CUES: frozenset[str] = frozenset(
    {
        "related",
        "relate",
        "relates",
        "relationship",
        "depends",
        "depend",
        "dependency",
        "impact",
        "impacts",
        "why",
        "caused",
        "cause",
        "causes",
        "because",
        "effect",
        "effects",
        "influence",
        "influenced",
        "between",
        "both",
        "compare",
        "compared",
    }
)

_DEFAULT_BUDGETS = {"packed": 10_000, "coe": 16_000}
_ALIAS_NGRAM_MAX = 4
_WORD = re.compile(r"[\w][\w&.\-]*")

_CONFIG_KEYS = frozenset(
    {
        "graph_mode",
        "selector",
        "budget",
        "pool_k",
        "select_pool_k",
        "seed_top_k",
        "hop_weight",
        "community_weight",
        "centrality_weight",
        "seed_docs_k",
        "seed_node_cap",
        "rescue_cap",
        "pack_centrality_weight",
        "similarity_key",
        "fill",
        "rank_score_k",
        "relation_cues",
        "entity_heavy_seeds",
        "agreement_k",
        "low_agreement",
        "high_agreement",
        "low_confidence",
        "navigational_max_words",
        "hyperedge_kv_path",
    }
)


class GateDecision(BaseModel):
    """Whether the graph lane runs for one query, and every reason that went into it.

    ``reasons`` are the triggers that fired (empty when none did); ``vetoed_by`` names the
    known-item guard — or the ``off`` mode — when it overrode them. Both are kept because "the gate
    did not fire" and "the gate fired and was vetoed" are different facts about a query, and a hit
    rate that cannot tell them apart cannot be debugged. Frozen and JSON-round-tripping like every
    contract in this library.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fired: bool
    reasons: tuple[str, ...] = ()
    vetoed_by: str | None = None
    agreement: float = 0.0
    confidence: float = 0.0
    query_seeds: int = 0


class RescueResult(BaseModel):
    """One query's graph-rescue retrieval: the final candidates plus what the pipeline decided.

    ``candidates`` is the selected (or, with ``selector="none"``, the full ranked) union — the base
    in its own order, rescue appendix below it, each stamped ``metadata["rescued"]``. ``gate`` is
    the policy decision made for this query; ``rescued`` counts the appendix candidates the graph
    carried in. ``tokens_packed``/``budget`` report the selector's spend (``None`` when no selector
    ran) so the operating point is observable per query, not assumed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[Candidate, ...]
    gate: GateDecision
    rescued: int
    tokens_packed: int | None = None
    budget: int | None = None


@runtime_checkable
class HyperedgeDocumentSource(Protocol):
    """A read-only, stateless hyperedge→chunk index: which chunks carry these Hyperedges.

    The rescue twin of :class:`~latence_retrieval.graph_retrieval.GraphDocumentSource`: given the
    Hyperedge ids the hypergraph lane retained (ordered by descending ρ), return the chunk
    candidates that carry them, best-first, each id at most once. The caller owns the mapping — it
    was emitted alongside the hyperedge store — so the component never guesses at it (ADR-0048).
    """

    def candidates(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        """Return the chunk candidates carrying ``hyperedge_ids``, best-first, deduplicated."""
        ...


class GraphRescueRetrieval:
    """The graph-as-rescue retrieval pipeline over caller-supplied engines (ADR-0048, S5b).

    The caller owns the engines (a dense backend, a BM25 backend, optionally a
    :class:`~latence_retrieval.graph_retrieval.GraphRetriever` and the hyperedge store); this owns
    the *policy* — the gate, the seeding, the append-and-dedupe invariant, and the selector. It
    holds no per-query state, so one instance serves a whole corpus and identical inputs always
    yield identical results.

    Config (every default is the measured S5b operating point):
        graph_mode: ``"off" | "auto" | "force"`` — the policy gate. Default ``"auto"``.
        selector: ``"packed" | "coe" | "none"`` — step 4. Default ``"packed"``.
        budget: int — the selector's token budget. Default ``10_000`` for ``packed`` (the knapsack
            saturates ~5.5k; backfill spends the rest), ``16_000`` for ``coe`` (it keeps buying
            recall through 16k). Ignored for ``"none"``.
        pool_k: int — first-stage depth per fetch leg. Default ``100``.
        select_pool_k: int — how many union candidates the selector may choose from. Default
            ``100``.
        seed_top_k / hop_weight / community_weight / centrality_weight — the
            :class:`~latence_retrieval.expand.Expander` rescore, in units of the max-normalised
            base score. Defaults ``5`` / ``0.05`` / ``0.01`` / ``0.05``: the base's own
            rank-to-rank gap near the head is ~``0.016``, so a corroborated candidate moves a few
            ranks and can never vault the head.
        seed_docs_k: int — how many first-stage documents seed entity resolution. Default ``5``.
        seed_node_cap: int — ceiling on seed node ids handed to the traversal (query seeds first,
            then doc entities in rank order). Default ``32`` — five news chunks carry ~68 entities,
            and an uncapped 1-hop expansion resolves back to a large fraction of the corpus.
        rescue_cap: int — rescue candidates appended per query. Default ``50`` — the rescue lane
            is a tail that carries in what the base missed, not a second ranking.
        pack_centrality_weight: float — weight on peak ``kg_node_centrality`` in the packer's
            value. Default ``0.25``. The one graph feature that reaches the solver as a *value*
            rather than an ordering.
        similarity_key: str | None — the packer's precomputed-redundancy metadata key. Default
            ``"redundancy"``; a pair absent from the map falls back to the packer's own lexical
            redundancy, so supplying the signal is an optimisation, never a requirement.
        fill: ``"stop" | "backfill"`` — what the packer does with budget the greedy strands.
            Default ``"backfill"`` (measured: pure greedy stranded ~45% of a 10k budget, and for
            retrieval unspent budget is unrecovered recall).
        rank_score_k: int — the RRF ``k`` the union's rank scores use. Default ``60`` (the fuser's
            own default, so the union's value decays in the shape the base already had).
        relation_cues: list[str] — the gate's relation-cue lexicon. Default
            :data:`RELATION_CUES`.
        entity_heavy_seeds: int — query seeds at/above which the query is entity-heavy. Default
            ``2`` (one entity is a look-up; two is a question about how they stand to each other).
        agreement_k: int — the top-``k`` window of the dense/BM25 agreement signal. Default ``10``.
        low_agreement: float — Jaccard agreement below which the first stages DISAGREE. Default
            ``0.30``.
        high_agreement: float — agreement at/above which they agree (the known-item shape, with a
            short cue-free query). Default ``0.50``.
        low_confidence: float — dense top-1 score below which the first stage is unconfident.
            Default ``0.60``. The one corpus-dependent knob; its firing is a named reason on the
            decision so it is visible per query.
        navigational_max_words: int — a query no longer than this, cue-free and with agreeing
            first stages, is known-item/navigational and stays graph-off. Default ``6``.
        hyperedge_kv_path: str | None — path to the emitted hyperedge sidecar store
            (``graph-hyperedges.kv``). Supplied together with ``hyperedge_documents`` it turns the
            hypergraph rescue lane on; supplying only one of the pair is a loud config error.

    Dependencies (accepted, never constructed — a test injects fakes, production injects real
    adapters):
        dense / bm25: the caller's first-stage :class:`~latence_retrieval.backend.RetrievalBackend`
            engines. At least one is required (a rescue pipeline with no base is a contradiction).
        graph: an optional ``GRAPH``-capable backend (typically
            :class:`~latence_retrieval.graph_retrieval.GraphRetriever`) — the traversal rescue
            lane.
        hyperedge_documents: an optional :class:`HyperedgeDocumentSource` — the hypergraph rescue
            lane's chunk resolver, paired with ``hyperedge_kv_path``.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        dense: RetrievalBackend | None = None,
        bm25: RetrievalBackend | None = None,
        graph: RetrievalBackend | None = None,
        hyperedge_documents: HyperedgeDocumentSource | None = None,
    ) -> None:
        cfg = dict(config or {})
        unknown = set(cfg) - _CONFIG_KEYS
        if unknown:
            msg = f"GraphRescueRetrieval: unknown config key(s) {sorted(unknown)}."
            raise ValueError(msg)

        self._mode = self._choice(cfg.get("graph_mode", "auto"), "graph_mode", GRAPH_MODES)
        self._selector = self._choice(cfg.get("selector", "packed"), "selector", SELECTORS)

        budget = cfg.get("budget")
        if budget is None:
            budget = _DEFAULT_BUDGETS.get(self._selector, 0)
        self._budget = self._positive_int(budget, "budget") if self._selector != "none" else 0

        self._pool_k = self._positive_int(cfg.get("pool_k", 100), "pool_k")
        self._select_pool_k = self._positive_int(cfg.get("select_pool_k", 100), "select_pool_k")

        self._seed_docs_k = self._positive_int(cfg.get("seed_docs_k", 5), "seed_docs_k")
        self._seed_node_cap = self._positive_int(cfg.get("seed_node_cap", 32), "seed_node_cap")
        self._rescue_cap = self._positive_int(cfg.get("rescue_cap", 50), "rescue_cap")
        self._rank_score_k = self._positive_int(cfg.get("rank_score_k", 60), "rank_score_k")

        # The Expander validates its own knobs; this only chooses the measured defaults.
        self._expander = Expander(
            {
                "seed_top_k": cfg.get("seed_top_k", 5),
                "hop_weight": cfg.get("hop_weight", 0.05),
                "community_weight": cfg.get("community_weight", 0.01),
                "centrality_weight": cfg.get("centrality_weight", 0.05),
            }
        )

        # The Packer validates budget/fill/value_signals/similarity_key itself.
        pack_centrality = cfg.get("pack_centrality_weight", 0.25)
        self._packer: Packer | None = None
        if self._selector == "packed":
            self._packer = Packer(
                {
                    "budget": self._budget,
                    "value_signals": {"kg_node_centrality": pack_centrality},
                    "similarity_key": cfg.get("similarity_key", "redundancy"),
                    "fill": cfg.get("fill", "backfill"),
                }
            )

        self._relation_cues = self._cue_set(cfg.get("relation_cues", RELATION_CUES))
        self._entity_heavy_seeds = self._positive_int(
            cfg.get("entity_heavy_seeds", 2), "entity_heavy_seeds"
        )
        self._agreement_k = self._positive_int(cfg.get("agreement_k", 10), "agreement_k")
        self._low_agreement = self._unit_float(cfg.get("low_agreement", 0.30), "low_agreement")
        self._high_agreement = self._unit_float(cfg.get("high_agreement", 0.50), "high_agreement")
        self._low_confidence = self._finite_float(
            cfg.get("low_confidence", 0.60), "low_confidence"
        )
        self._navigational_max_words = self._positive_int(
            cfg.get("navigational_max_words", 6), "navigational_max_words"
        )

        kv_path = cfg.get("hyperedge_kv_path")
        if kv_path is not None and (not isinstance(kv_path, str) or not kv_path):
            msg = (
                "GraphRescueRetrieval 'hyperedge_kv_path' must be a non-empty string path, "
                f"got {kv_path!r}."
            )
            raise ValueError(msg)
        if (kv_path is None) != (hyperedge_documents is None):
            msg = (
                "GraphRescueRetrieval hypergraph lane needs both 'hyperedge_kv_path' (config) and "
                "'hyperedge_documents' (injected), or neither — one without the other cannot "
                "resolve Hyperedges to chunks."
            )
            raise ValueError(msg)
        if hyperedge_documents is not None and not isinstance(
            hyperedge_documents, HyperedgeDocumentSource
        ):
            msg = (
                f"GraphRescueRetrieval 'hyperedge_documents' "
                f"{type(hyperedge_documents).__name__!r} does not satisfy the "
                "HyperedgeDocumentSource seam (needs 'candidates(hyperedge_ids)')."
            )
            raise TypeError(msg)
        self._kv_path = kv_path
        self._hyperedge_documents = hyperedge_documents

        if self._selector == "coe" and self._kv_path is None:
            msg = (
                "GraphRescueRetrieval selector 'coe' selects over the hypergraph lane's signals; "
                "configure 'hyperedge_kv_path' + 'hyperedge_documents' or choose another selector."
            )
            raise ValueError(msg)

        if dense is None and bm25 is None:
            msg = (
                "GraphRescueRetrieval needs at least one base first stage ('dense' and/or 'bm25') "
                "— a rescue pipeline with no base ranking is a contradiction."
            )
            raise ValueError(msg)
        for name, backend in (("dense", dense), ("bm25", bm25), ("graph", graph)):
            if backend is not None:
                self._require_backend(name, backend)
        self._dense = dense
        self._bm25 = bm25
        self._graph = graph
        self._fuser = RrfFuser()

    # -- the per-query pipeline -------------------------------------------------------------------

    def retrieve(self, bundle: QueryBundle) -> RescueResult:
        """Run the whole pipeline for one query: base, gate, rescore, rescue append, select.

        ``bundle`` carries the caller-generated query signals exactly as the Level-2 orchestrator's
        does: ``text`` (always), ``vector`` (the dense leg's signal — absent drops the leg) and
        ``seed_node_ids`` (the query's linked entity nodes — the gate's entity signal and the
        traversal's primary anchors). Identical bundles always yield identical results.
        """
        dense = self._search_dense(bundle)
        bm25 = self._search_bm25(bundle)
        query_seeds = tuple(dict.fromkeys(bundle.seed_node_ids or ()))
        gate = self._gate(bundle.text, dense, bm25, query_seeds)

        base = self._fuse_base(dense, bm25)
        rescue: list[Candidate] = []
        signals: HypergraphSignals | None = None
        if gate.fired:
            base = self._expander.process(bundle.text, base)
            seeds = self._graph_seeds(query_seeds, base)
            rescue, signals = self._rescue(bundle.text, seeds)

        union = _append_and_dedupe(base, rescue, cap=self._rescue_cap, k=self._rank_score_k)
        rescued = len(union) - len(base)
        return self._select(bundle.text, union, signals, gate, rescued)

    # -- step 1: the base -------------------------------------------------------------------------

    def _search_dense(self, bundle: QueryBundle) -> list[Candidate]:
        if self._dense is None or not bundle.vector:
            return []
        return self._dense.search(
            RetrievalQuery(mode=Capability.DENSE, vector=bundle.vector, top_k=self._pool_k)
        )

    def _search_bm25(self, bundle: QueryBundle) -> list[Candidate]:
        if self._bm25 is None:
            return []
        return self._bm25.search(
            RetrievalQuery(mode=Capability.BM25, text=bundle.text, top_k=self._pool_k)
        )

    def _fuse_base(self, dense: Sequence[Candidate], bm25: Sequence[Candidate]) -> list[Candidate]:
        """RRF over dense and BM25 — and over nothing else — then max-normalised to 1.0.

        The regression the S5 matrix measured came from extending this fusion to lists that are
        *not* peers; the max-normalisation is a strictly monotone change of units that puts the
        expander's absolute bonuses on a scale the base ranking survives.
        """
        rankings = [list(ranking) for ranking in (dense, bm25) if ranking]
        if not rankings:
            return []
        return _unit_scaled(self._fuser.fuse(rankings))

    # -- the policy gate (graph_mode) -------------------------------------------------------------

    def _gate(
        self,
        question: str,
        dense: Sequence[Candidate],
        bm25: Sequence[Candidate],
        query_seeds: Sequence[str],
    ) -> GateDecision:
        """Apply ``graph_mode`` to the policy's verdict.

        ``off`` and ``force`` still record what ``auto`` *would* have decided, so a forced run can
        still report which queries the policy would have paid for — a mode that erased the policy's
        opinion could not tell a gate that is too tight from a lane that is too weak.
        """
        decision = self._decide(question, dense, bm25, query_seeds)
        if self._mode == "off":
            return decision.model_copy(update={"fired": False, "vetoed_by": "mode-off"})
        if self._mode == "force":
            return decision.model_copy(
                update={
                    "fired": True,
                    "reasons": decision.reasons or ("mode-force",),
                    "vetoed_by": None,
                }
            )
        return decision

    def _decide(
        self,
        question: str,
        dense: Sequence[Candidate],
        bm25: Sequence[Candidate],
        query_seeds: Sequence[str],
    ) -> GateDecision:
        """``graph_mode="auto"``: should the graph lane be paid for on this query?

        The four triggers are the sidecar's shipped policy, evaluated against signals the pipeline
        actually has: a relation cue in the text, two or more entities the query itself names,
        first stages that disagree, and a first stage that is not confident. The known-item guard
        then vetoes a short, cue-free query whose two routes agree — the navigational shape that
        graph traversal can only make worse.
        """
        words = _WORD.findall(question)
        lowered = {word.lower() for word in words}
        agreement = top_k_agreement(dense, bm25, self._agreement_k)
        confidence = float(dense[0].score) if dense else 0.0

        reasons: list[str] = []
        if lowered & self._relation_cues:
            reasons.append("relation-cue")
        if len(query_seeds) >= self._entity_heavy_seeds:
            reasons.append("entity-heavy")
        if agreement < self._low_agreement:
            reasons.append("low-agreement")
        if confidence < self._low_confidence:
            reasons.append("low-confidence")

        navigational = (
            len(words) <= self._navigational_max_words
            and "relation-cue" not in reasons
            and agreement >= self._high_agreement
        )
        return GateDecision(
            fired=bool(reasons) and not navigational,
            reasons=tuple(reasons),
            vetoed_by="navigational" if navigational and reasons else None,
            agreement=round(agreement, 4),
            confidence=round(confidence, 4),
            query_seeds=len(query_seeds),
        )

    # -- step 3: seeding + the rescue lanes -------------------------------------------------------

    def _graph_seeds(
        self, query_seeds: Sequence[str], base: Sequence[Candidate]
    ) -> tuple[str, ...]:
        """Seed node ids from the QUERY **and** the top first-stage documents.

        Query seeds come first and are never displaced by the cap: the query naming an entity is a
        stronger signal than a top-``seed_docs_k`` document mentioning one. Document seeds follow
        in first-stage rank order (read off each candidate's own ``kg_node_ids`` column), so the
        cap trims the weakest anchors rather than an arbitrary set.
        """
        seeds: list[str] = list(dict.fromkeys(query_seeds))
        seen = set(seeds)
        for candidate in base[: self._seed_docs_k]:
            for node in _node_ids(candidate):
                if node not in seen:
                    seen.add(node)
                    seeds.append(node)
        return tuple(seeds[: self._seed_node_cap])

    def _rescue(
        self, question: str, seeds: Sequence[str]
    ) -> tuple[list[Candidate], HypergraphSignals | None]:
        """The graph lanes, fused with EACH OTHER, as one appendable rescue list.

        The traversal lane and the hypergraph lane are peers (both are graph reach over the same
        KG), so RRF between them is the same legitimate move as dense+bm25. What must never happen
        — and does not happen here — is fusing the result of that into the base.
        """
        traversal = self._traversal(seeds)
        hyper, signals = self._hypergraph(question, seeds)
        rankings = [list(ranking) for ranking in (traversal, hyper) if ranking]
        if not rankings:
            return ([], signals)
        return (self._fuser.fuse(rankings), signals)

    def _traversal(self, seeds: Sequence[str]) -> list[Candidate]:
        if self._graph is None or not seeds:
            return []
        return self._graph.search(
            RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=tuple(seeds), top_k=self._pool_k)
        )

    def _hypergraph(
        self, question: str, seeds: Sequence[str]
    ) -> tuple[list[Candidate], HypergraphSignals | None]:
        """The hypergraph lane, entered from the query's alias surfaces AND the seed nodes.

        ``ann_node_ids`` is the tier-0 entry — "the caller's own ANN supplies node ids" — and
        first-stage document entities are precisely that: node ids the caller's own first stage
        selected. The retained Hyperedges (descending ρ, id tiebreak) resolve to chunks through the
        injected :class:`HyperedgeDocumentSource`.
        """
        if self._kv_path is None or self._hyperedge_documents is None:
            return ([], None)
        with HyperedgeKV.open(self._kv_path) as kv:
            signals = hypergraph_signals(
                kv,
                alias_surfaces=_query_ngrams(question),
                ann_node_ids=list(seeds),
            )
        ordered = sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
        return (self._hyperedge_documents.candidates(ordered), signals)

    # -- step 4: selection over the deduped union -------------------------------------------------

    def _select(
        self,
        question: str,
        union: list[Candidate],
        signals: HypergraphSignals | None,
        gate: GateDecision,
        rescued: int,
    ) -> RescueResult:
        if self._selector == "none":
            return RescueResult(candidates=tuple(union), gate=gate, rescued=rescued)
        pool = union[: self._select_pool_k]
        if self._packer is not None:
            selected = self._packer.process(question, pool)
        else:
            selected = self._coe_select(pool, signals)
        return RescueResult(
            candidates=tuple(selected),
            gate=gate,
            rescued=rescued,
            tokens_packed=sum(_TOKENS.count(c.text) for c in selected),
            budget=self._budget,
        )

    def _coe_select(
        self, pool: Sequence[Candidate], signals: HypergraphSignals | None
    ) -> list[Candidate]:
        """The alternative selector: CoE decides the graph spend, the base fills the rest.

        :func:`~latence_retrieval.coe.select` selects *Hyperedges*, not arbitrary candidates, so it
        cannot rank the union directly. What it can do — and what this composes — is decide which
        of the RESCUE candidates are evidence worth budget. The base is then taken in its own rank
        order and the whole thing is filled to the budget.
        """
        allowed: set[str] = set()
        if (
            signals is not None
            and signals.rho
            and self._hyperedge_documents is not None
        ):
            selection = coe_select(signals, budget=self._budget)
            allowed = {
                candidate.id
                for candidate in self._hyperedge_documents.candidates(selection.selected)
            }
        selected: list[Candidate] = []
        spent = 0
        for candidate in pool:
            if candidate.metadata.get("rescued", False) and candidate.id not in allowed:
                continue
            cost = max(_TOKENS.count(candidate.text), 1)
            if spent + cost > self._budget:
                continue
            selected.append(candidate)
            spent += cost
        return selected

    # -- construction-time validation -------------------------------------------------------------

    @staticmethod
    def _require_backend(name: str, backend: object) -> None:
        from latence_retrieval.backend import RetrievalBackend as _Backend

        if not isinstance(backend, _Backend):
            msg = (
                f"GraphRescueRetrieval {name!r} backend {type(backend).__name__!r} does not "
                "satisfy the RetrievalBackend seam (needs 'capabilities' and 'search(query)')."
            )
            raise TypeError(msg)

    @staticmethod
    def _choice(value: object, name: str, allowed: tuple[str, ...]) -> str:
        if value not in allowed:
            msg = (
                f"GraphRescueRetrieval {name!r} must be one of {list(allowed)}, got {value!r}."
            )
            raise ValueError(msg)
        return str(value)

    @staticmethod
    def _positive_int(value: object, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            msg = f"GraphRescueRetrieval {name!r} must be a positive integer, got {value!r}."
            raise ValueError(msg)
        return value

    @staticmethod
    def _finite_float(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"GraphRescueRetrieval {name!r} must be a number, got {value!r}."
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def _unit_float(value: object, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (0.0 <= value <= 1.0)
        ):
            msg = f"GraphRescueRetrieval {name!r} must be a number in [0, 1], got {value!r}."
            raise ValueError(msg)
        return float(value)

    @staticmethod
    def _cue_set(value: object) -> frozenset[str]:
        if isinstance(value, str) or not isinstance(value, (list, tuple, set, frozenset)):
            msg = (
                "GraphRescueRetrieval 'relation_cues' must be a collection of cue words, "
                f"got {value!r}."
            )
            raise ValueError(msg)
        cues: set[str] = set()
        for cue in value:
            if not isinstance(cue, str) or not cue.strip():
                msg = f"GraphRescueRetrieval 'relation_cues' must contain words, got {cue!r}."
                raise ValueError(msg)
            cues.add(cue.strip().lower())
        return frozenset(cues)


# -- module-level helpers (pure, so a test can drive each on its own) ---------------------------

_TOKENS = HeuristicTokenCounter()


def top_k_agreement(a: Sequence[Candidate], b: Sequence[Candidate], k: int = 10) -> float:
    """Jaccard overlap of two lists' top-``k`` ids — the dense/BM25 route-agreement signal.

    ``0.0`` when either side is empty: a route that returned nothing agrees with nobody, which is
    itself a low-confidence signal rather than a perfect one.
    """
    left = {c.id for c in a[:k]}
    right = {c.id for c in b[:k]}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _query_ngrams(question: str, max_n: int = _ALIAS_NGRAM_MAX) -> list[str]:
    """Descending-length word n-grams of a question — the tier-1 zero-model alias surfaces."""
    words = _WORD.findall(question)
    grams: list[str] = []
    for n in range(max_n, 0, -1):
        grams.extend(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


def _node_ids(candidate: Candidate) -> tuple[str, ...]:
    """The candidate's own ``kg_node_ids`` column: absent → nothing, malformed → loud.

    The same graceful-absent / loud-malformed discipline the
    :class:`~latence_retrieval.expand.Expander` applies to the identical column: a store that never
    denormalised the KG simply contributes no doc seeds, while a present-but-corrupt column fails
    at the boundary instead of silently seeding nothing.
    """
    if "kg_node_ids" not in candidate.metadata:
        return ()
    raw = candidate.metadata["kg_node_ids"]
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        msg = (
            "graph-rescue metadata 'kg_node_ids' must be a list of node-id strings, "
            f"got {raw!r} for candidate {candidate.id!r}."
        )
        raise TypeError(msg)
    for element in raw:
        if not isinstance(element, str) or not element:
            msg = (
                "graph-rescue metadata 'kg_node_ids' must contain non-empty strings, "
                f"got {element!r} for candidate {candidate.id!r}."
            )
            raise ValueError(msg)
    return tuple(raw)


def _append_and_dedupe(
    base: Sequence[Candidate], rescue: Sequence[Candidate], *, cap: int, k: int
) -> list[Candidate]:
    """``base`` in its own order, then the rescue candidates it did not already contain.

    The architectural invariant, in one function: the base list's relative order is copied
    unchanged, a rescue candidate already present is dropped (never re-ranked to its rescue
    position), and what remains is appended below the whole base, capped at ``cap``. Every
    candidate is then scored by its position as ``1/(k + rank)`` — a strictly decreasing function
    of rank, so the score encodes the order rather than competing with it — and stamped
    ``metadata["rescued"]`` so a selector (and a caller) can tell what the base earned from what
    the graph carried in.
    """
    seen = {candidate.id for candidate in base}
    ordered: list[tuple[Candidate, bool]] = [(candidate, False) for candidate in base]
    for candidate in rescue:
        if candidate.id in seen:
            continue
        seen.add(candidate.id)
        ordered.append((candidate, True))
        if len(ordered) - len(base) >= cap:
            break
    return [
        candidate.model_copy(
            update={
                "score": 1.0 / (k + rank),
                "metadata": {**candidate.metadata, "rescued": rescued},
            }
        )
        for rank, (candidate, rescued) in enumerate(ordered)
    ]


def _unit_scaled(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Max-normalise scores to ``1.0`` — strictly monotone, so the ranking is untouched.

    Purely a change of units, and the reason it exists is in the module docstring: the expander's
    bonuses are absolute numbers while RRF scores are ~0.03, so without this one graph link would
    dominate the base ranking completely.
    """
    items = list(candidates)
    top = max((candidate.score for candidate in items), default=0.0)
    if top <= 0:
        return items
    return [candidate.model_copy(update={"score": candidate.score / top}) for candidate in items]


__all__ = [
    "GRAPH_MODES",
    "RELATION_CUES",
    "SELECTORS",
    "GateDecision",
    "GraphRescueRetrieval",
    "HyperedgeDocumentSource",
    "RescueResult",
    "top_k_agreement",
]
