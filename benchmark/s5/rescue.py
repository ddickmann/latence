# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The S5b graph-rescue family, as this bench wires the SHIPPED component to its own engines.

The architecture — base ``dense+bm25`` RRF (those two only, max-normalised), an ``Expander``
rescore off the denormalised KG columns, the traversal and hypergraph lanes **appended** below the
base and deduplicated (never fused into it), and a token-budget selector over the union behind a
per-query policy gate — is not implemented here. It is
:class:`latence_retrieval.rescue.GraphRescueRetrieval`, the component an operator deploys, and
this module is the four things a *benchmark* adds on top of it:

* **the leg dispatch** — which of the three S5 leg names selects how, and at which headline
  budget (:func:`selector_for`, :func:`headline_budget`);
* **the budget curve** — :data:`BUDGET_CURVE`, settled against ONE retrieval per query through
  :meth:`~latence_retrieval.rescue.GraphRescueRetrieval.select`, so the curve varies the budget
  and nothing else;
* **the gate report** — :class:`GateStats`, the per-cell hit rate the ticket asks for;
* **the caller-supplied signals** — this harness's engines wired to the component
  (:func:`build_pipeline`, :class:`HyperedgeChunks`) and its candidate decoration
  (:class:`ExportDecorator`: the export's KG columns plus the precomputed pairwise ``redundancy``
  row the packer reads through its ``similarity_key`` seam). ADR-0048's division exactly: the
  library owns policy and composition, the caller owns engines, signals and index.

Until Wave 5-J this module carried a second, line-for-line copy of the pipeline (``RescuePipeline``
and its ~600 lockstep lines). The copy existed for one reason — the bench retrieves once per query
and selects at every budget, while the shipped component fused retrieval and selection into a
single ``retrieve()``. Splitting the shipped component into
:meth:`~latence_retrieval.rescue.GraphRescueRetrieval.retrieve` +
:meth:`~latence_retrieval.rescue.GraphRescueRetrieval.select` removed the reason, and the copy was
deleted against a per-query A/B oracle over the full gold sets
(``issues/wayfinder-enterprise-serving/tickets/05J/ab_rescue.py``): full unions with metadata,
gate decisions, hypergraph signals and every leg × every budget's selection, compared with ``==``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_retrieval.orchestrator import QueryBundle
from latence_retrieval.rescue import GraphRescueRetrieval, KgColumnDecorator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmark.s5.corpus import Export
    from benchmark.s5.legs import Runner
    from latence_retrieval.contracts import Candidate
    from latence_retrieval.rescue import GateDecision

#: The token budgets every selection leg reports a recall curve over. A selection stage optimises
#: *best context within a budget*, so a single ``recall@10`` is the wrong shape of number for it —
#: the curve is the native protocol. Each leg's headline (:func:`headline_budget`) is a member of
#: the curve, so the two can never disagree.
BUDGET_CURVE = (2000, 4000, 8000, 10000, 16000, 24000, 32000)

#: The knapsack headline budget (operator ruling, s5b). The ladder's 4,000 came from neither the
#: HyCE-RAG paper (whose evidence budget is a COUNT cap of 6 segments) nor a measurement, and it
#: measurably packed only ~1,895 tokens on wiki2multihop. The knapsack saturates well below this
#: (mean spend ~5,545 on the smoke; 10k and 16k produced identical selections), so raising it
#: further changes nothing for ``graph_rescue_packed``. It is the shipped ``packed`` default.
SELECT_BUDGET = 10000

#: The CoE headline budget (operator ruling, 2026-08-11: "going up to 16k for a retrieval is not
#: a problem at all" — modern LLMs' smart zone is ~80–100k tokens). Unlike the knapsack, CoE keeps
#: spending productively as the budget grows (measured 9,944 at 10k → 15,942 at 16k with docR@10
#: rising 0.8380 → 0.8455), so its operating point sits where the spend still buys recall. The
#: 24k/32k curve points exist to locate its actual saturation, not to move the headline again. It
#: is the shipped ``coe`` default.
COE_SELECT_BUDGET = 16000

#: Which shipped selector each S5 leg name dispatches to. ``graph_rescue`` has no selector — it is
#: the ranked union, the leg comparable to the ladder's ranking rungs at ``recall@k``;
#: ``graph_rescue_packed`` is the knapsack QKP; ``graph_rescue_coe`` is chain-of-evidence.
LEG_SELECTORS = {
    "graph_rescue": "none",
    "graph_rescue_packed": "packed",
    "graph_rescue_coe": "coe",
}

#: The shipped ``seed_docs_k`` / ``seed_node_cap`` defaults, named here because the ticket-01
#: latency harness reproduces the seed harvest OUTSIDE the pipeline and has to use the same
#: numbers. Pinned against a constructed :class:`GraphRescueRetrieval` by ``tests/test_rescue.py``,
#: so they cannot drift from the component's own defaults.
SEED_DOCS_K = 5
SEED_NODE_CAP = 32


def selector_for(leg: str) -> str:
    """The shipped selector ``leg`` dispatches to; loud on a name that is not a rescue leg."""
    try:
        return LEG_SELECTORS[leg]
    except KeyError:
        msg = f"{leg!r} is not a graph-rescue leg."
        raise ValueError(msg) from None


def headline_budget(leg: str) -> int:
    """The budget ``leg``'s headline ``metrics`` block is taken at (a member of the curve).

    Each selector's headline sits at its own measured operating point: the knapsack saturates
    ~5.5k so 10k is generous, while CoE keeps buying recall through 16k.
    """
    return COE_SELECT_BUDGET if selector_for(leg) == "coe" else SELECT_BUDGET


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


# -- the caller's side of ADR-0048: engines, signals, decoration --------------------------------


class HyperedgeChunks:
    """The shipped ``HyperedgeDocumentSource`` over this runner's hyperedge→chunk mapping.

    Delegates to :meth:`~benchmark.s5.legs.Runner.chunks_from_hyperedges` — reused, not
    re-implemented, so the pipeline resolves retained Hyperedges to chunks exactly the way every
    other S5 leg does.
    """

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    def candidates(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        """The chunk candidates carrying ``hyperedge_ids``, best-first, deduplicated."""
        return self._runner.chunks_from_hyperedges(hyperedge_ids)


class ExportDecorator:
    """This harness's candidate decoration, as the shipped ``CandidateDecorator`` seam takes it.

    Two caller-supplied signals in one adapter (ADR-0048 — this carries them, the shipped
    components interpret them):

    * the export's four denormalised KG columns per candidate (``kg_node_ids``,
      ``neighbor_node_ids``, ``kg_node_community``, ``kg_node_centrality``), read verbatim off the
      export so a candidate whose engine did not carry them still reaches the ``Expander``, the
      traversal seed harvest and the packer's value signal with its own entities;
    * the precomputed pairwise ``redundancy`` row the :class:`~latence_retrieval.pack.Packer`
      reads through its documented ``similarity_key`` seam — the lexical Jaccard it would
      otherwise recompute ``O(selected × pool)`` times *per budget*, which is what makes a
      seven-point budget curve affordable.

    The columns mapping is built per call from the candidates in hand and handed to the shipped
    :class:`~latence_retrieval.rescue.KgColumnDecorator`, rather than materialised once for the
    whole corpus: a 220k-chunk export would cost hundreds of megabytes to hold as a dict of rows,
    and the redundancy numbers must be the packer's own — which the shipped adapter guarantees by
    computing them with the packer's own tokeniser (imported there, not mirrored here).
    """

    def __init__(self, export: Export) -> None:
        self._export = export

    def decorate(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """The KG columns of every candidate the export knows, plus the full redundancy map."""
        rows = self._export.rows
        columns: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            position = self._export.index_of(candidate.id)
            if position is None:
                continue
            row = rows[position]
            columns[candidate.id] = {
                "kg_node_ids": list(row.node_ids),
                "neighbor_node_ids": list(row.neighbor_node_ids),
                "kg_node_community": list(row.community_ids),
                "kg_node_centrality": list(row.centralities),
            }
        return KgColumnDecorator(columns, redundancy_key="redundancy").decorate(candidates)


def build_pipeline(
    runner: Runner, export: Export, *, mode: str = "auto", selector: str = "packed"
) -> GraphRescueRetrieval:
    """The shipped component over this runner's engines — the S5b cell's pipeline.

    Every knob left unset is a knob whose shipped default IS the value this bench measured
    (``pool_k`` 100, ``select_pool_k`` 100, ``seed_docs_k`` 5, ``seed_node_cap`` 32,
    ``rescue_cap`` 50, ``rank_score_k`` 60, the expander weights, the gate thresholds,
    ``pack_centrality_weight`` 0.25, ``similarity_key="redundancy"``, ``fill="backfill"``), so
    this function passes only what is genuinely the caller's: the mode, the engines, the hyperedge
    store, and the decoration.

    ``selector`` is the pipeline's configured one; a cell drives each leg through
    :meth:`~latence_retrieval.rescue.GraphRescueRetrieval.select`'s per-call override, so ONE
    pipeline (and one retrieval per query) serves every leg and every budget of the curve.
    """
    hyperedges = None if runner.kv_path is None else HyperedgeChunks(runner)
    config: dict[str, Any] = {"graph_mode": mode, "selector": selector}
    if runner.kv_path is not None:
        config["hyperedge_kv_path"] = str(runner.kv_path)
    return GraphRescueRetrieval(
        config,
        dense=runner.dense,
        bm25=runner.bm25,
        graph=runner.graph,
        hyperedge_documents=hyperedges,
        decorate=ExportDecorator(export),
    )


def bundle_for(runner: Runner, question: str, vector: Any) -> QueryBundle:  # noqa: ANN401 - numpy
    """The query-signal-gen this bench supplies per query: the dense vector and the alias seeds.

    The caller's side of the pipeline's inputs (ADR-0048): the linked entity nodes come from the
    shipped :class:`~latence_retrieval.graph_retrieval.AliasQuerySeeder` over the emitted alias
    map, exactly as every other seeded S5 leg gets them.
    """
    seeds = tuple(runner.seeder.seeds(question)) if runner.seeder is not None else ()
    return QueryBundle(
        text=question,
        vector=tuple(float(x) for x in vector),
        seed_node_ids=seeds or None,
    )


__all__ = [
    "BUDGET_CURVE",
    "COE_SELECT_BUDGET",
    "LEG_SELECTORS",
    "SEED_DOCS_K",
    "SEED_NODE_CAP",
    "SELECT_BUDGET",
    "ExportDecorator",
    "GateStats",
    "HyperedgeChunks",
    "build_pipeline",
    "bundle_for",
    "headline_budget",
    "selector_for",
]
