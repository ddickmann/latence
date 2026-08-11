# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Graph as post-retrieval **rescue**, then knapsack **selection** — the S5b architecture.

The S5 ladder measured every graph leg lowering recall. The components were not the problem; the
*application* was (ticket ``s5b-retrieval-architecture-fix``). This module is the corrected
application, built from the same shipped ``latence-retrieval`` components and the same harness
engines, in the shape the framework's own seams and ``colsearch``'s shipped graph sidecar both
describe — base retrieval first, graph strictly additive on top, a solver spending the budget::

    1. BASE     dense + BM25, fused by RRF between THOSE TWO ONLY.
                Same footing (both are full-corpus first stages over the same corpus), so rank
                fusion between them is legitimate. Nothing else enters this fusion.

    2. RESCORE  `latence_retrieval.expand.Expander` over the base list. It anchors on the top
                `seed_top_k` first-stage candidates and adds a bounded 1-hop graph bonus read off
                the KG columns already denormalised onto every chunk (`kg_node_ids`,
                `neighbor_node_ids`, `kg_node_community`, `kg_node_centrality`) — no graph
                database, ADR-0051's default path. This is the seam the ladder should have used
                instead of running `GraphRetriever` as a competing list.

    3. RESCUE   the graph lanes — `GraphRetriever` traversal and
                `latence_retrieval.hypergraph.hypergraph_signals` — contribute candidates the base
                MISSED. They are fused with each other (co-equal graph lanes, same footing), then
                **appended** to the rescored base and deduplicated by id. The base ordering is
                never reordered by them. Entity resolution is seeded from the QUERY *and* the top
                first-stage documents, which is the strongest entity signal available and the one
                the ladder threw away.

    4. SELECT   a token budget is spent over the deduped union: `latence_retrieval.pack.Packer`
                (the knapsack QKP, with the graph features as `value_signals`) or
                `latence_retrieval.coe.select` (chain-of-evidence) as the alternative selector at
                the same budget. Both are measured.

**A policy gate decides whether step 3 runs at all** (``colsearch``'s ``graph_mode: auto``). Known-
item and navigational queries stay graph-off; entity-heavy queries, relation cues, low dense/BM25
agreement and low first-stage confidence turn it on. Every decision is recorded per query so the
gate's hit rate is a reported number rather than an assumption.

**What this module does NOT do.** It does not re-implement scoring: every score, bonus, fusion and
selection above is computed by the shipped component named beside it. It supplies exactly what
ADR-0048 says a stateless library does not own — the caller's engines (inherited from
:class:`~benchmark.s5.legs.Runner`), the caller's query-signal generation (seeds, alias surfaces),
the caller's redundancy signal, and the caller's policy. There are **zero LLM calls** anywhere on
this path.

**Score scales, stated because they are load-bearing.** ``RrfFuser`` emits ``Σ w/(k+rank)`` — about
``0.033`` at rank 0 for two lists — while ``Expander``'s ``hop_weight`` defaults to ``0.5``. Handing
raw RRF scores to the expander would let one graph link outvote the entire base ranking, which is
the very defect this module exists to remove, wearing a new hat. So the base scores are
**max-normalised to 1.0 first** (a strictly monotone transform: the base order is untouched), and
the bonuses are set as documented fractions of that unit scale. The union is then re-scored by its
own rank as ``1/(k + rank)`` with the fuser's own ``k`` — the same slowly-decaying shape the base
already had, so a candidate deep in the pool keeps enough value for the packer to consider it
instead of the greedy stalling at a third of the budget.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from benchmark.s5.legs import POOL_K, Runner, SignalRanking, fuse_signals, query_ngrams
from latence_retrieval.coe import select as coe_select
from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery
from latence_retrieval.expand import Expander
from latence_retrieval.hypergraph import HyperedgeKV, HypergraphSignals, hypergraph_signals
from latence_retrieval.pack import Packer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.s5.corpus import Export

#: The token budgets every selection leg reports a recall curve over. A selection stage optimises
#: *best context within a budget*, so a single ``recall@10`` is the wrong shape of number for it —
#: the curve is the native protocol. :data:`SELECT_BUDGET` is the operating point the leg's headline
#: ``metrics`` block is taken at, and it is a member of the curve so the two can never disagree.
BUDGET_CURVE = (2000, 4000, 8000, 10000, 16000, 24000, 32000)

#: The knapsack headline budget (operator ruling, s5b). The ladder's 4,000 came from neither the
#: HyCE-RAG paper (whose evidence budget is a COUNT cap of 6 segments) nor a measurement, and it
#: measurably packed only ~1,895 tokens on wiki2multihop. The knapsack saturates well below this
#: (mean spend ~5,545 on the smoke; 10k and 16k produced identical selections), so raising it
#: further changes nothing for ``graph_rescue_packed``.
SELECT_BUDGET = 10000

#: The CoE headline budget (operator ruling, 2026-08-11: "going up to 16k for a retrieval is not
#: a problem at all" — modern LLMs' smart zone is ~80–100k tokens). Unlike the knapsack, CoE keeps
#: spending productively as the budget grows (measured 9,944 at 10k → 15,942 at 16k with docR@10
#: rising 0.8380 → 0.8455), so its operating point sits where the spend still buys recall. The
#: 24k/32k curve points exist to locate its actual saturation, not to move the headline again.
COE_SELECT_BUDGET = 16000

#: How many first-stage documents seed the graph lane's entity resolution (colsearch step 3:
#: ``resolve_entities(query, docs)``). Small on purpose — these are the *anchors*, and anchoring
#: on a wide pool re-creates the "graph traversal as first-stage router" shape this rejects.
SEED_DOCS_K = 5

#: Ceiling on seed node ids handed to the traversal, query-alias seeds first, then first-stage-doc
#: entities in rank order. A cap is required, not cosmetic: five news chunks carry ~68 entities, and
#: an uncapped 1-hop expansion from all of them resolves back to a large fraction of the corpus,
#: which is a slow way of returning no information.
SEED_NODE_CAP = 32

#: Rescue candidates appended per query (after the two graph lanes are fused with each other).
#: The rescue lane is a tail, not a second ranking — it exists to carry in what the base missed.
RESCUE_CAP = 50

#: Candidates the selector may choose from. Matches the ladder's packer rung, so the two selection
#: numbers differ by the pipeline and not by how much was on offer.
SELECT_POOL_K = POOL_K

#: :class:`~latence_retrieval.expand.Expander` config, in units of the max-normalised base score
#: (see the module docstring). The base's own rank-to-rank gap near the head is about ``0.016``, so
#: ``hop_weight`` of ``0.05`` moves a corroborated candidate a few ranks and a candidate linked to
#: every one of the five seeds about fifteen — a real lift that still cannot vault the head.
EXPAND_SEED_TOP_K = 5
EXPAND_HOP_WEIGHT = 0.05
EXPAND_COMMUNITY_WEIGHT = 0.01
EXPAND_CENTRALITY_WEIGHT = 0.05

#: Weight on a candidate's PEAK ``kg_node_centrality`` in the packer's value. ``Packer`` reduces the
#: per-entity list to its peak identically to ``Expander`` (documented in both), so this is the one
#: graph feature that reaches the solver as a value signal rather than as an ordering.
PACK_CENTRALITY_WEIGHT = 0.25

#: The RRF ``k`` the union's rank scores use — the fuser's own default, so the union's value shape
#: is the shape the base already had rather than a second, faster-decaying convention.
RANK_SCORE_K = 60

#: Relation cues (``colsearch``'s shipped policy list, plus the two multi-hop connectives our own
#: datasets are built on). A query carrying one of these is asking about a RELATION, which is what
#: the graph lane is for.
RELATION_CUES = frozenset(
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

#: Distinct query-resolved entities at or above which a query counts as *entity-heavy*. Two, because
#: one entity is a look-up and two or more is a query about how they stand to each other.
ENTITY_HEAVY_SEEDS = 2

#: Top-``k`` overlap between the dense and the BM25 lists below which the two first stages are
#: judged to DISAGREE — the classic "neither route is confident" signal.
AGREEMENT_K = 10
LOW_AGREEMENT = 0.30

#: Top-``k`` overlap at or above which the first stages *agree*, which (together with a short query
#: and no relation cue) is what a known-item / navigational query looks like.
HIGH_AGREEMENT = 0.50

#: Dense top-1 cosine below which the first stage is judged unconfident. This is the one
#: corpus-dependent knob in the policy; it is recorded per query as a named reason so a gate that
#: fires for this reason alone is visible in the report rather than folded into a rate.
LOW_CONFIDENCE = 0.60

#: A query no longer than this, with no relation cue and agreeing first stages, is treated as
#: known-item/navigational and stays graph-off whatever else fired.
NAVIGATIONAL_MAX_WORDS = 6

_WORD = re.compile(r"[\w][\w&.\-]*")
#: The framework's term unit (``[^\W_]+``), mirrored for the caller-supplied redundancy signal.
#: Pinned against the packer's own lexical redundancy by ``tests/test_rescue.py``.
_TOKEN = re.compile(r"[^\W_]+")


# -- the policy gate (colsearch `graph_mode: auto`) ---------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    """Whether the graph lane runs for one query, and every reason that went into it.

    ``reasons`` are the triggers that fired (empty when none did); ``vetoed_by`` names the
    known-item guard when it overrode them. Both are kept because "the gate did not fire" and "the
    gate fired and was vetoed" are different facts about a query, and a hit rate that cannot tell
    them apart cannot be debugged.
    """

    fired: bool
    reasons: tuple[str, ...] = ()
    vetoed_by: str | None = None
    agreement: float = 0.0
    confidence: float = 0.0
    query_seeds: int = 0


@dataclass
class GateStats:
    """Running gate observability for one cell — the hit rate the ticket asks to report."""

    mode: str = "auto"
    queries: int = 0
    fired: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    vetoes: int = 0

    def add(self, decision: GateDecision) -> None:
        """Record one query's decision."""
        self.queries += 1
        if decision.fired:
            self.fired += 1
        if decision.vetoed_by is not None:
            self.vetoes += 1
        for reason in decision.reasons:
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        """The JSON block a cell carries."""
        return {
            "mode": self.mode,
            "queries": self.queries,
            "fired": self.fired,
            "hit_rate": round(self.fired / self.queries, 4) if self.queries else 0.0,
            "vetoed_navigational": self.vetoes,
            "reasons": dict(sorted(self.reasons.items())),
        }


def top_k_agreement(a: Sequence[Candidate], b: Sequence[Candidate], k: int = AGREEMENT_K) -> float:
    """Jaccard overlap of two lists' top-``k`` ids — the dense/BM25 route-agreement signal.

    ``0.0`` when either side is empty: a route that returned nothing agrees with nobody, which is
    itself a low-confidence signal rather than a perfect one.
    """
    left = {c.id for c in a[:k]}
    right = {c.id for c in b[:k]}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def decide_gate(
    question: str,
    dense: Sequence[Candidate],
    bm25: Sequence[Candidate],
    query_seeds: Sequence[str],
) -> GateDecision:
    """``graph_mode="auto"``: should the graph lane be paid for on this query?

    The four triggers are ``colsearch``'s shipped policy, evaluated against signals this harness
    actually has: a relation cue in the text, two or more entities the query itself names, first
    stages that disagree, and a first stage that is not confident. The known-item guard then vetoes
    a short, cue-free query whose two routes agree — the navigational shape that graph traversal can
    only make worse.
    """
    words = _WORD.findall(question)
    lowered = {word.lower() for word in words}
    agreement = top_k_agreement(dense, bm25)
    confidence = float(dense[0].score) if dense else 0.0

    reasons: list[str] = []
    if lowered & RELATION_CUES:
        reasons.append("relation-cue")
    if len(query_seeds) >= ENTITY_HEAVY_SEEDS:
        reasons.append("entity-heavy")
    if agreement < LOW_AGREEMENT:
        reasons.append("low-agreement")
    if confidence < LOW_CONFIDENCE:
        reasons.append("low-confidence")

    navigational = (
        len(words) <= NAVIGATIONAL_MAX_WORDS
        and "relation-cue" not in reasons
        and agreement >= HIGH_AGREEMENT
    )
    return GateDecision(
        fired=bool(reasons) and not navigational,
        reasons=tuple(reasons),
        vetoed_by="navigational" if navigational and reasons else None,
        agreement=round(agreement, 4),
        confidence=round(confidence, 4),
        query_seeds=len(query_seeds),
    )


# -- one query's retrieval, computed once and selected from at every budget ----------------------


@dataclass(frozen=True)
class Retrieved:
    """Everything one query's pipeline produced before a token budget is chosen.

    Held apart from selection because the recall-over-budget curve settles five budgets against the
    identical candidate union — the only way the curve measures the *selector* rather than five
    slightly different retrievals.
    """

    base: list[Candidate]
    union: list[Candidate]
    gate: GateDecision
    rescued: int
    signals: HypergraphSignals | None
    latency_ms: float


@dataclass(frozen=True)
class RescueOutcome:
    """One (query, leg, budget) result — the ranked/selected ids plus what it cost."""

    ranked: list[str]
    latency_ms: float
    gate: GateDecision
    tokens_packed: int | None = None
    budget: int | None = None


@dataclass
class RescuePipeline:
    """The graph-as-rescue pipeline over a built :class:`~benchmark.s5.legs.Runner`.

    The runner owns the engines (dense, BM25, the traversal retriever, the hyperedge store); this
    owns the *policy* — the gate, the seeding, the append-and-dedupe, and the two selectors. Nothing
    here holds per-query state, so a pipeline is reusable across a whole dataset and two pipelines
    over the same runner produce the same numbers.
    """

    runner: Runner
    #: ``colsearch``'s ``graph_mode``. ``auto`` is the shipped policy (:func:`decide_gate`);
    #: ``off`` is the base pipeline with no graph lane at all — the honest floor every rescue number
    #: is measured against — and ``force`` runs the lane on every query, which is what makes the
    #: gate's contribution separable from the lane's.
    mode: str = "auto"
    expander: Expander = field(
        default_factory=lambda: Expander(
            {
                "seed_top_k": EXPAND_SEED_TOP_K,
                "hop_weight": EXPAND_HOP_WEIGHT,
                "community_weight": EXPAND_COMMUNITY_WEIGHT,
                "centrality_weight": EXPAND_CENTRALITY_WEIGHT,
            }
        )
    )

    @property
    def export(self) -> Export:
        """The loaded export the runner ranks over."""
        return self.runner.export

    # -- step 1+2+3: base, rescore, rescue -------------------------------------------------------

    def retrieve(self, question: str, vector: Any) -> Retrieved:
        """Run steps 1–3 for one query: base ranking, graph rescore, graph rescue append.

        The returned ``union`` is the rescored base followed by the rescue candidates the base did
        not already contain, each carrying its rank score, its KG metadata and its redundancy row —
        i.e. exactly what a selector needs and nothing it has to recompute.
        """
        started = time.perf_counter()
        dense = self._dense(vector)
        bm25 = self._bm25(question)
        query_seeds = self._query_seeds(question)
        gate = self._gate(question, dense, bm25, query_seeds)

        base = self._fuse_base(dense, bm25)
        signals: HypergraphSignals | None = None
        rescue: list[Candidate] = []
        if gate.fired:
            base = self._rescore(question, base)
            seeds = self._graph_seeds(query_seeds, base)
            rescue, signals = self._rescue(question, seeds)

        union = _append_and_dedupe(base, rescue, cap=RESCUE_CAP)
        return Retrieved(
            base=base,
            union=self._decorate(union),
            gate=gate,
            rescued=len(union) - len(base),
            signals=signals,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _gate(
        self,
        question: str,
        dense: Sequence[Candidate],
        bm25: Sequence[Candidate],
        query_seeds: Sequence[str],
    ) -> GateDecision:
        """Apply :attr:`mode` to the shipped policy's verdict.

        ``off`` and ``force`` still record what ``auto`` *would* have decided, so a forced run's
        cell can still report which queries the policy would have paid for — a mode that erased the
        policy's opinion could not tell a gate that is too tight from a lane that is too weak.
        """
        decision = decide_gate(question, dense, bm25, query_seeds)
        if self.mode == "auto":
            return decision
        if self.mode == "off":
            return GateDecision(
                fired=False,
                reasons=decision.reasons,
                vetoed_by="mode-off",
                agreement=decision.agreement,
                confidence=decision.confidence,
                query_seeds=decision.query_seeds,
            )
        if self.mode == "force":
            return GateDecision(
                fired=True,
                reasons=decision.reasons or ("mode-force",),
                agreement=decision.agreement,
                confidence=decision.confidence,
                query_seeds=decision.query_seeds,
            )
        msg = f"graph mode must be one of 'off', 'auto', 'force'; got {self.mode!r}."
        raise ValueError(msg)

    def _dense(self, vector: Any) -> list[Candidate]:
        if self.runner.dense is None:
            return []
        return self.runner.dense.search(
            RetrievalQuery(
                mode=Capability.DENSE, vector=tuple(float(x) for x in vector), top_k=POOL_K
            )
        )

    def _bm25(self, question: str) -> list[Candidate]:
        if self.runner.bm25 is None:
            return []
        return self.runner.bm25.search(
            RetrievalQuery(mode=Capability.BM25, text=question, top_k=POOL_K)
        )

    def _fuse_base(self, dense: Sequence[Candidate], bm25: Sequence[Candidate]) -> list[Candidate]:
        """RRF over dense and BM25 — and over nothing else. Step 1.

        Two full-corpus first stages over the same corpus are peers, so rank fusion between them is
        the legitimate use of RRF. The regression the S5 matrix measured came from extending that
        same fusion to lists that are *not* peers.
        """
        rankings = [
            SignalRanking(name, list(candidates))
            for name, candidates in (("dense", dense), ("bm25", bm25))
            if candidates
        ]
        return _unit_scaled(fuse_signals(self.runner.fuser, rankings))

    def _rescore(self, question: str, base: Sequence[Candidate]) -> list[Candidate]:
        """Step 2 — the shipped 1-hop :class:`Expander` over the base list's own KG columns."""
        return self.expander.process(question, self._kg_metadata(base))

    def _query_seeds(self, question: str) -> tuple[str, ...]:
        """Entity node ids the QUERY itself names (the alias seeder over the emitted alias map)."""
        if self.runner.seeder is None:
            return ()
        return tuple(self.runner.seeder.seeds(question))

    def _graph_seeds(
        self, query_seeds: Sequence[str], base: Sequence[Candidate]
    ) -> tuple[str, ...]:
        """Seed node ids from the QUERY **and** the top first-stage documents (colsearch step 3).

        Query seeds come first and are never displaced by the cap: the query naming an entity is a
        stronger signal than a top-5 document mentioning one. Document seeds follow in first-stage
        rank order, so the cap trims the weakest anchors rather than an arbitrary set.
        """
        seeds: list[str] = list(dict.fromkeys(query_seeds))
        seen = set(seeds)
        rows = self.export.rows
        for candidate in base[:SEED_DOCS_K]:
            position = self.export.index_of(candidate.id)
            if position is None:
                continue
            for node in rows[position].node_ids:
                if node not in seen:
                    seen.add(node)
                    seeds.append(node)
        return tuple(seeds[:SEED_NODE_CAP])

    def _rescue(
        self, question: str, seeds: Sequence[str]
    ) -> tuple[list[Candidate], HypergraphSignals | None]:
        """Step 3 — the two graph lanes, fused with EACH OTHER, as one appendable rescue list.

        The traversal lane and the hypergraph lane are peers (both are graph reach over the same
        KG), so RRF between them is the same legitimate move as ``dense``+``bm25``. What must never
        happen — and does not happen here — is fusing the result of that into the base.
        """
        traversal = self._traversal(seeds)
        hyper, signals = self._hypergraph(question, seeds)
        rankings = [
            SignalRanking(name, candidates)
            for name, candidates in (("graph", traversal), ("hypergraph", hyper))
            if candidates
        ]
        return (fuse_signals(self.runner.fuser, rankings), signals)

    def _traversal(self, seeds: Sequence[str]) -> list[Candidate]:
        if self.runner.graph is None or not seeds:
            return []
        return self.runner.graph.search(
            RetrievalQuery(mode=Capability.GRAPH, seed_node_ids=tuple(seeds), top_k=POOL_K)
        )

    def _hypergraph(
        self, question: str, seeds: Sequence[str]
    ) -> tuple[list[Candidate], HypergraphSignals | None]:
        """The hypergraph lane, entered from the query's alias surfaces AND the doc-derived nodes.

        ``ann_node_ids`` is ADR-0059's tier-0 entry — "the caller's own ANN supplies node ids" —
        and first-stage document entities are precisely that: node ids the caller's own first stage
        selected. The ladder passed only ``alias_surfaces``, which is the second defect the ticket
        names, and the fix is one argument the shipped function already takes.
        """
        if self.runner.kv_path is None:
            return ([], None)
        with HyperedgeKV.open(str(self.runner.kv_path)) as kv:
            signals = hypergraph_signals(
                kv,
                alias_surfaces=query_ngrams(question),
                ann_node_ids=list(seeds),
            )
            ordered = sorted(signals.rho, key=lambda h: (-signals.rho[h], h))
            return (self.runner.chunks_from_hyperedges(ordered), signals)

    # -- step 4: selection over the deduped union ------------------------------------------------

    def select(
        self, question: str, retrieved: Retrieved, leg: str, budget: int
    ) -> RescueOutcome:
        """Spend ``budget`` tokens over ``retrieved.union`` with ``leg``'s selector.

        ``graph_rescue`` has no selector — it is the ranked union, the leg that is comparable to the
        ladder's ranking rungs at ``recall@k``. ``graph_rescue_packed`` selects with the knapsack
        QKP; ``graph_rescue_coe`` selects with chain-of-evidence at the same budget.
        """
        started = time.perf_counter()
        if leg == "graph_rescue":
            return RescueOutcome(
                ranked=[c.id for c in retrieved.union],
                latency_ms=retrieved.latency_ms,
                gate=retrieved.gate,
            )
        pool = retrieved.union[:SELECT_POOL_K]
        if leg == "graph_rescue_packed":
            selected = _packer(budget).process(question, pool)
        elif leg == "graph_rescue_coe":
            selected = self._coe_select(pool, retrieved.signals, budget)
        else:  # pragma: no cover - guarded by ALL_LEG_NAMES at the CLI boundary
            msg = f"{leg!r} is not a graph-rescue leg."
            raise ValueError(msg)
        return RescueOutcome(
            ranked=[c.id for c in selected],
            latency_ms=retrieved.latency_ms + (time.perf_counter() - started) * 1000.0,
            gate=retrieved.gate,
            tokens_packed=sum(_tokens(c.text) for c in selected),
            budget=budget,
        )

    def _coe_select(
        self, pool: Sequence[Candidate], signals: HypergraphSignals | None, budget: int
    ) -> list[Candidate]:
        """The alternative selector: CoE decides the graph spend, the base fills the rest.

        :func:`~latence_retrieval.coe.select` selects *hyperedges*, not arbitrary candidates, so it
        cannot rank the union directly. What it can do — and what this composes — is decide which of
        the RESCUE candidates are evidence worth budget. The base is then taken in its own rank
        order and the whole thing is filled to the budget. Against the knapsack that is a genuine
        A/B at one budget: a chain-aware evidence chooser plus a rank-order prefix, versus a
        redundancy-aware quadratic knapsack.
        """
        allowed: set[str] = set()
        if signals is not None and signals.rho:
            selection = coe_select(signals, budget=budget)
            allowed = {
                candidate.id
                for candidate in self.runner.chunks_from_hyperedges(selection.selected)
            }
        selected: list[Candidate] = []
        spent = 0
        for candidate in pool:
            if _is_rescue(candidate) and candidate.id not in allowed:
                continue
            cost = max(_tokens(candidate.text), 1)
            if spent + cost > budget:
                continue
            selected.append(candidate)
            spent += cost
        return selected

    # -- candidate decoration: the KG columns and the redundancy signal ---------------------------

    def _kg_metadata(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Attach each candidate's denormalised KG columns, verbatim off the export.

        A *caller-supplied signal* in the ADR-0048 sense: this carries the columns,
        :class:`~latence_retrieval.expand.Expander` and :class:`~latence_retrieval.pack.Packer`
        interpret them. Anything already on the candidate's metadata (the rescue marker) is kept.
        """
        rows = self.export.rows
        out: list[Candidate] = []
        for candidate in candidates:
            metadata: dict[str, Any] = dict(candidate.metadata)
            position = self.export.index_of(candidate.id)
            if position is not None:
                row = rows[position]
                metadata["kg_node_ids"] = list(row.node_ids)
                metadata["neighbor_node_ids"] = list(row.neighbor_node_ids)
                metadata["kg_node_community"] = list(row.community_ids)
                metadata["kg_node_centrality"] = list(row.centralities)
            out.append(candidate.model_copy(update={"metadata": metadata}))
        return out

    def _decorate(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """The KG columns **plus** the precomputed pairwise redundancy row the selector reads.

        ``redundancy`` is the lexical (Jaccard) similarity the packer would otherwise recompute
        ``O(selected × pool)`` times *per budget*, re-tokenising both texts on every call. Supplying
        it once per query through the documented ``similarity_key`` seam is what makes a five-budget
        curve affordable — and it is the identical number, pinned byte-for-byte against the packer's
        own default redundancy by ``tests/test_rescue.py``. Every pair is stored, including the
        zeros, so no pair can silently fall back to the packer's own recomputation.
        """
        items = self._kg_metadata(candidates)
        tokens = [_token_set(candidate.text) for candidate in items]
        similarity: list[dict[str, float]] = [{} for _ in items]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                value = _jaccard(tokens[i], tokens[j])
                similarity[i][items[j].id] = value
                similarity[j][items[i].id] = value
        return [
            candidate.model_copy(
                update={"metadata": {**candidate.metadata, "redundancy": similarity[index]}}
            )
            for index, candidate in enumerate(items)
        ]


# -- module-level helpers (pure, so a test can drive each on its own) ---------------------------


def _packer(budget: int) -> Packer:
    """The knapsack selector at ``budget``, reading the graph feature and supplied redundancy."""
    return Packer(
        {
            "budget": budget,
            "value_signals": {"kg_node_centrality": PACK_CENTRALITY_WEIGHT},
            "similarity_key": "redundancy",
            # Recall operating point: the pure greedy stall stranded ~45% of a 10k budget on the
            # smoke (mean spend ~5,545), and for a retrieval bench unspent budget is unrecovered
            # recall. Backfill spends it in value order without touching the QKP phase's picks.
            "fill": "backfill",
        }
    )


def _append_and_dedupe(
    base: Sequence[Candidate], rescue: Sequence[Candidate], *, cap: int
) -> list[Candidate]:
    """``base`` in its own order, then the rescue candidates it did not already contain.

    The architectural invariant, in one function: the base list's relative order is copied
    unchanged, a rescue candidate already present is dropped (never re-ranked to its rescue
    position), and what remains is appended below the whole base. Every candidate is then scored by
    its position as ``1/(k + rank)`` — a strictly decreasing function of rank, so the score encodes
    the order rather than competing with it — and an appended one is stamped ``rescued`` so a
    selector can tell what the base earned from what the graph carried in.
    """
    seen = {candidate.id for candidate in base}
    ordered = [(candidate, False) for candidate in base]
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
                "score": 1.0 / (RANK_SCORE_K + rank),
                "metadata": {**candidate.metadata, "rescued": rescued},
            }
        )
        for rank, (candidate, rescued) in enumerate(ordered)
    ]


def _unit_scaled(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Max-normalise scores to ``1.0`` — strictly monotone, so the ranking is untouched.

    Purely a change of units, and the reason it exists is in the module docstring: ``Expander``'s
    bonuses are absolute numbers, and RRF scores are ~0.03, so without this the graph bonus would
    dominate the base ranking completely.
    """
    items = list(candidates)
    top = max((candidate.score for candidate in items), default=0.0)
    if top <= 0:
        return items
    return [candidate.model_copy(update={"score": candidate.score / top}) for candidate in items]


def _is_rescue(candidate: Candidate) -> bool:
    """Whether a union candidate was appended by the rescue lane rather than earned by the base.

    Stamped onto the candidate by :func:`_append_and_dedupe` at the moment the distinction is
    actually known, so no downstream stage has to reconstruct it from a parallel id set that could
    drift out of step with the list it describes.
    """
    return bool(candidate.metadata.get("rescued", False))


def _token_set(text: str) -> frozenset[str]:
    """Lowercased word tokens of ``text`` — the framework's term unit."""
    return frozenset(_TOKEN.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """``|A ∩ B| / |A ∪ B|``; ``0.0`` when either side is empty (matching the packer's default)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _tokens(text: str) -> int:
    """The packer's own dependency-free token estimate, reused for the packed-tokens report."""
    from latence_retrieval.pack import HeuristicTokenCounter

    return HeuristicTokenCounter().count(text)


__all__ = [
    "AGREEMENT_K",
    "BUDGET_CURVE",
    "COE_SELECT_BUDGET",
    "ENTITY_HEAVY_SEEDS",
    "HIGH_AGREEMENT",
    "LOW_AGREEMENT",
    "LOW_CONFIDENCE",
    "NAVIGATIONAL_MAX_WORDS",
    "RELATION_CUES",
    "RESCUE_CAP",
    "SEED_DOCS_K",
    "SEED_NODE_CAP",
    "SELECT_BUDGET",
    "SELECT_POOL_K",
    "GateDecision",
    "GateStats",
    "RescueOutcome",
    "RescuePipeline",
    "Retrieved",
    "decide_gate",
    "top_k_agreement",
]
