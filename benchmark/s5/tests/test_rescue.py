# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""The graph-as-rescue ARCHITECTURE, pinned — not just its plumbing.

Every test here asserts a property the S5 matrix's regression was caused by *violating*, so a
refactor that quietly restores the old shape fails here rather than in a recall number six hours
later:

* the base ranking survives a graph lane that adds nothing — **exactly**, not approximately;
* rescue candidates are APPENDED and DEDUPED, never fused into the base and never re-ranked to
  their rescue position;
* the selector sees the whole union and respects its token budget;
* the policy gate fires on a relation-shaped query and stays off on a known-item one;
* the caller-supplied redundancy signal is the packer's own number, so precomputing it is a change
  of cost and not of results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmark.s5.corpus import ChunkRow, Export
from benchmark.s5.datasets import DATASETS
from benchmark.s5.legs import POOL_K, ChunkGraphDocumentSource, DenseBackend, Runner
from benchmark.s5.rescue import (
    RANK_SCORE_K,
    SELECT_BUDGET,
    GateStats,
    RescuePipeline,
    _append_and_dedupe,
    _packer,
    _unit_scaled,
    decide_gate,
    top_k_agreement,
)
from latence_retrieval.contracts import Candidate
from latence_retrieval.graph_retrieval import AliasQuerySeeder, GraphRetriever

_MULTIHOP = DATASETS["multihop_rag"].export_dir

requires_export = pytest.mark.skipif(
    not (_MULTIHOP / "corpus.parquet").is_file(),
    reason="the S3 exports are not on this machine",
)


# -- a hand-built corpus whose graph structure is known exactly ----------------------------------

#: Six token vocabularies, cycled across the corpus so redundancy is real (some pairs overlap
#: heavily, some not at all) rather than uniformly zero — which is what makes the packer's
#: redundancy penalty an actual test rather than a no-op.
_TEXTS = (
    "alpha beta gamma delta epsilon zeta eta theta",
    "beta gamma iota kappa lambda mu nu xi",
    "omicron pi rho sigma tau upsilon phi chi",
    "psi omega alpha beta gamma delta epsilon",
    "aleph bet gimel dalet he vav zayin het",
    "tet yod kaf lamed mem nun samekh ayin",
)

#: The fixture corpus is deliberately WIDER than the first-stage pool (:data:`POOL_K`). A corpus
#: that fits inside the pool makes the base list the whole corpus, and then "rescue adds a
#: candidate the base missed" is untestable by construction — the base cannot miss anything.
_CORPUS_ROWS = POOL_K + 20

#: The entity only the tail chunks carry, so a query that names it can only reach them through the
#: graph lane. Placed past the pool cut-off on purpose.
_TAIL_NODE = "n-tail"
_TAIL_ROWS = (POOL_K + 5, POOL_K + 11)


def _rows(*, with_kg: bool = True) -> list[ChunkRow]:
    """``_CORPUS_ROWS`` chunks, three per document, with a known KG projection on each.

    ``with_kg=False`` is the *no graph signal whatsoever* corpus — the same rows with every KG
    column empty, which is what an export predating the projection looks like. It is what makes
    "the base survives a graph lane that adds nothing" an assertion about the architecture rather
    than about the fixture's particular entity layout.
    """
    rows: list[ChunkRow] = []
    for index in range(_CORPUS_ROWS):
        nodes = (f"n{index % 7}",)
        if index in _TAIL_ROWS:
            nodes = (_TAIL_NODE,)
        rows.append(
            ChunkRow(
                record_id=f"c{index:03d}",
                content=f"chunk{index} {_TEXTS[index % len(_TEXTS)]}",
                file_name=f"doc-{index // 3:03d}.md",
                char_start=(index % 3) * 40,
                char_end=(index % 3) * 40 + 39,
                node_ids=nodes if with_kg else (),
                neighbor_node_ids=(f"n{(index + 1) % 7}",) if with_kg else (),
                community_ids=(f"k{index % 3}",) if with_kg else (),
                centralities=(round(0.1 + (index % 9) / 10.0, 2),) if with_kg else (),
            )
        )
    return rows


def _matrix() -> np.ndarray:
    """Unit 2-D vectors on a narrow arc, so ``[1, 0]`` ranks row 0 … row N STRICTLY descending.

    A total order with no ties is what makes "the base head is preserved verbatim" an exact
    assertion instead of one that a tie-break could satisfy by luck.
    """
    angles = np.arange(_CORPUS_ROWS, dtype=np.float32) * 0.01
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def _export(root: Path, *, with_kg: bool = True) -> Export:
    return Export(
        dataset="fixture",
        root=root,
        rows=_rows(with_kg=with_kg),
        embeddings=_matrix(),
        pages={},
    )


class _NoEdges:
    """A ``GraphSource`` with no edges at all — so the traversal contributes seed documents only.

    Deliberately the *weakest possible* graph: whatever the rescue lane carries in here, it carried
    in through entity co-mention alone, which is the floor the architecture has to work at.
    """

    def neighbors(self, seeds: object, max_hops: int) -> dict[str, int]:
        """No node has a neighbour, so every rescue hit is a seed-mentioning document."""
        return {}


def _pipeline(
    root: Path, *, with_graph: bool, mode: str = "auto", with_kg: bool = True
) -> RescuePipeline:
    export = _export(root, with_kg=with_kg)
    graph = None
    seeder = None
    if with_graph:
        graph = GraphRetriever(
            graph_source=_NoEdges(), document_source=ChunkGraphDocumentSource(export)
        )
        seeder = AliasQuerySeeder({"kappa": _TAIL_NODE, "sigma": "n4"})
    runner = Runner(
        export=export,
        dense=DenseBackend(export),
        bm25=None,
        graph=graph,
        kv_path=None,
        seeder=seeder,
    )
    return RescuePipeline(runner=runner, mode=mode)


#: The query vector every test ranks with — row 0 first, then row 1, …, strictly descending.
_QUERY = np.array([1.0, 0.0], dtype=np.float32)

#: Carries a relation cue (`why`) and names two entities, so the gate fires and the graph lane runs.
#: `kappa` resolves to the tail entity, which lives ONLY outside the first-stage pool.
_RELATION_QUESTION = "why does kappa relate to sigma"


# -- the invariant: a graph lane that adds nothing leaves the base exactly as it was -------------


def _list(*ids: str) -> list[Candidate]:
    return [Candidate(id=i, score=0.9 - n / 100, text=i) for n, i in enumerate(ids)]


def test_base_ranking_is_preserved_when_the_graph_lane_adds_nothing(tmp_path: Path) -> None:
    """Gate ON, graph lane present, and NOT ONE graph signal in the corpus: the base is untouched.

    This is the property the ladder's fusion could not hold. Under RRF a second list changes the
    first list's order by construction, so a graph lane that knows nothing still costs recall.
    Under rescore-then-append it cannot: an expander with no KG column to read is an
    order-preserving no-op, and an empty rescue list appends nothing.
    """
    pipeline = _pipeline(tmp_path, with_graph=True, with_kg=False)
    retrieved = pipeline.retrieve(_RELATION_QUESTION, _QUERY)
    assert retrieved.gate.fired
    assert retrieved.rescued == 0
    assert [c.id for c in retrieved.union] == [c.id for c in retrieved.base]
    # …and the base IS the dense ranking, verbatim: the fixture's vectors give a strict total
    # order, so this is an exact claim about every position and not just about the head.
    assert [c.id for c in retrieved.union] == [f"c{i:03d}" for i in range(POOL_K)]


def test_the_gate_being_off_leaves_the_base_untouched_and_costs_nothing(tmp_path: Path) -> None:
    """``graph_mode="off"`` returns the base ranking — no rescue, no traversal, no signals.

    The floor every rescue number has to be measured against. It also makes "the gate declined" and
    "the lane found nothing" separable, which a pipeline with only ``auto`` cannot do.
    """
    off = _pipeline(tmp_path, with_graph=True, mode="off")
    retrieved = off.retrieve(_RELATION_QUESTION, _QUERY)
    assert not retrieved.gate.fired
    assert retrieved.gate.vetoed_by == "mode-off"
    assert retrieved.rescued == 0
    assert retrieved.signals is None
    assert [c.id for c in retrieved.union] == [f"c{i:03d}" for i in range(POOL_K)]
    # …and `force` runs the lane on the very same query the policy might have declined.
    # …and `force` runs the lane on a query `auto` would decline, while still recording what
    # `auto` said — so a forced cell can still tell a gate that is too tight from a weak lane.
    agreeing = _list(*[f"c{i:03d}" for i in range(10)])
    navigational = "annual report 2024"
    auto = _pipeline(tmp_path, with_graph=True)
    forced = _pipeline(tmp_path, with_graph=True, mode="force")
    assert not auto._gate(navigational, agreeing, agreeing, ()).fired  # noqa: SLF001
    decision = forced._gate(navigational, agreeing, agreeing, ())  # noqa: SLF001
    assert decision.fired
    assert decision.reasons == ("mode-force",)
    assert decision.agreement == 1.0


def test_an_unknown_graph_mode_is_refused(tmp_path: Path) -> None:
    """A typo'd mode must not silently degrade to `auto` — the cell would report the wrong policy."""
    pipeline = _pipeline(tmp_path, with_graph=True, mode="on")
    with pytest.raises(ValueError, match="graph mode must be one of"):
        pipeline.retrieve(_RELATION_QUESTION, _QUERY)


def test_rescue_candidates_are_appended_and_deduped_never_fused() -> None:
    """The base keeps its order and its head; a duplicate keeps its BASE position, not its rescue one."""
    base = [Candidate(id=f"b{i}", score=1.0 - i / 10, text=f"base {i}") for i in range(4)]
    rescue = [
        Candidate(id="b3", score=99.0, text="base 3"),  # already in the base, and scored higher
        Candidate(id="r0", score=0.5, text="rescue 0"),
        Candidate(id="r1", score=0.4, text="rescue 1"),
    ]
    union = _append_and_dedupe(base, rescue, cap=10)
    assert [c.id for c in union] == ["b0", "b1", "b2", "b3", "r0", "r1"]
    # A fusion would have lifted `b3` on the strength of its rescue score. Appending cannot.
    assert [c.id for c in union[:4]] == [c.id for c in base]
    # Scores encode the rank rather than competing with it: strictly decreasing, RRF's own shape.
    scores = [c.score for c in union]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0 / RANK_SCORE_K)
    # Only the appended ones are marked, and that marker is what the CoE selector reads.
    assert [c.metadata["rescued"] for c in union] == [False] * 4 + [True] * 2


def test_the_rescue_tail_is_capped() -> None:
    """The rescue lane is a tail, not a second ranking — the cap is enforced, not advisory."""
    base = [Candidate(id="b0", score=1.0, text="base")]
    rescue = [Candidate(id=f"r{i}", score=0.5, text=f"rescue {i}") for i in range(10)]
    union = _append_and_dedupe(base, rescue, cap=3)
    assert [c.id for c in union] == ["b0", "r0", "r1", "r2"]


def test_unit_scaling_is_a_change_of_units_not_of_order() -> None:
    """The expander needs commensurate scores; it must not get them at the cost of the ranking."""
    items = [Candidate(id=f"c{i}", score=0.03 - i * 0.001, text="t") for i in range(5)]
    scaled = _unit_scaled(items)
    assert [c.id for c in scaled] == [c.id for c in items]
    assert scaled[0].score == pytest.approx(1.0)
    assert scaled[-1].score == pytest.approx(items[-1].score / items[0].score)
    # An all-zero list has no scale to normalise to and is returned untouched rather than divided.
    zeros = [Candidate(id="z", score=0.0, text="t")]
    assert _unit_scaled(zeros) == zeros


def test_the_graph_lane_appends_candidates_the_base_ranking_did_not_have(tmp_path: Path) -> None:
    """Rescue is only rescue if it carries something in — and only below the base."""
    pipeline = _pipeline(tmp_path, with_graph=True)
    retrieved = pipeline.retrieve(_RELATION_QUESTION, _QUERY)
    assert retrieved.gate.fired
    assert retrieved.rescued > 0
    base_ids = [c.id for c in retrieved.base]
    union_ids = [c.id for c in retrieved.union]
    assert union_ids[: len(base_ids)] == base_ids
    assert len(set(union_ids)) == len(union_ids)


# -- the policy gate ----------------------------------------------------------------------------


def test_the_gate_fires_on_a_relation_cue_query() -> None:
    """`why` is a relation cue, so the graph lane is worth paying for even with agreeing routes."""
    agreeing = _list(*[f"c{i}" for i in range(10)])
    decision = decide_gate("why did the merger cause the outage", agreeing, agreeing, ())
    assert decision.fired
    assert "relation-cue" in decision.reasons
    assert decision.vetoed_by is None


def test_the_gate_stays_off_for_a_known_item_query() -> None:
    """Short, cue-free, both routes confident and agreeing: the navigational shape stays graph-off."""
    agreeing = _list(*[f"c{i}" for i in range(10)])
    decision = decide_gate("annual report 2024", agreeing, agreeing, ())
    assert not decision.fired
    assert decision.agreement == 1.0
    # It is off because nothing triggered, so there was nothing for the guard to veto.
    assert decision.reasons == ()
    assert decision.vetoed_by is None


def test_the_navigational_guard_vetoes_a_short_query_the_triggers_would_have_fired_on() -> None:
    """Two entities in a three-word look-up is still a look-up; the veto is recorded, not hidden."""
    agreeing = _list(*[f"c{i}" for i in range(10)])
    decision = decide_gate("granite embedding r2", agreeing, agreeing, ("n1", "n2"))
    assert "entity-heavy" in decision.reasons
    assert not decision.fired
    assert decision.vetoed_by == "navigational"


def test_the_gate_fires_when_the_two_first_stages_disagree() -> None:
    """Low dense/BM25 agreement is a first-stage saying it does not know."""
    dense = _list(*[f"d{i}" for i in range(10)])
    bm25 = _list(*[f"b{i}" for i in range(10)])
    decision = decide_gate(
        "the quarterly filing of the parent entity and its two subsidiaries", dense, bm25, ()
    )
    assert decision.fired
    assert "low-agreement" in decision.reasons
    assert decision.agreement == 0.0


def test_agreement_is_zero_when_a_route_returned_nothing() -> None:
    """A route that found nothing agrees with nobody — never vacuously with everybody."""
    assert top_k_agreement(_list("a", "b"), []) == 0.0
    assert top_k_agreement([], []) == 0.0
    assert top_k_agreement(_list("a", "b"), _list("b", "c")) == pytest.approx(1 / 3)


def test_gate_stats_report_the_hit_rate_and_every_reason() -> None:
    """The gate's behaviour is a reported number, not an assumption about the query mix."""
    stats = GateStats()
    agreeing = _list(*[f"c{i}" for i in range(10)])
    stats.add(decide_gate("why did it happen", agreeing, agreeing, ()))
    stats.add(decide_gate("annual report 2024", agreeing, agreeing, ()))
    summary = stats.summary()
    assert summary["mode"] == "auto"
    assert summary["queries"] == 2
    assert summary["fired"] == 1
    assert summary["hit_rate"] == 0.5
    assert summary["reasons"] == {"relation-cue": 1}


# -- selection over the union --------------------------------------------------------------------


def test_the_packer_receives_the_union_and_respects_the_budget(tmp_path: Path) -> None:
    """The selector sees everything the pipeline assembled, and never overspends."""
    from latence_retrieval.pack import HeuristicTokenCounter

    pipeline = _pipeline(tmp_path, with_graph=True)
    retrieved = pipeline.retrieve(_RELATION_QUESTION, _QUERY)
    union_ids = {c.id for c in retrieved.union}
    texts = {c.id: c.text for c in retrieved.union}
    counter = HeuristicTokenCounter()
    for budget in (10, 40, 200):
        outcome = pipeline.select(_RELATION_QUESTION, retrieved, "graph_rescue_packed", budget)
        assert outcome.ranked
        assert set(outcome.ranked) <= union_ids
        assert outcome.budget == budget
        spent = sum(counter.count(texts[cid]) for cid in outcome.ranked)
        assert spent <= budget
        assert outcome.tokens_packed == spent
    # A budget wide enough for everything can still select everything — the cap is the budget, not
    # some hidden count limit.
    generous = pipeline.select(_RELATION_QUESTION, retrieved, "graph_rescue_packed", 10_000)
    assert set(generous.ranked) <= union_ids


def test_the_ranking_leg_has_no_selector_and_returns_the_whole_union(tmp_path: Path) -> None:
    """`graph_rescue` is the leg that is comparable to the ladder's RANKING rungs at recall@k."""
    pipeline = _pipeline(tmp_path, with_graph=True)
    retrieved = pipeline.retrieve(_RELATION_QUESTION, _QUERY)
    outcome = pipeline.select(_RELATION_QUESTION, retrieved, "graph_rescue", SELECT_BUDGET)
    assert outcome.ranked == [c.id for c in retrieved.union]
    assert outcome.budget is None
    assert outcome.tokens_packed is None


def test_the_coe_selector_keeps_the_base_and_spends_the_budget_on_approved_evidence(
    tmp_path: Path,
) -> None:
    """With no hyperedge store, nothing in the rescue tail is CoE-approved — so nothing survives.

    The base is still taken in rank order and filled to the budget: CoE decides the GRAPH spend, it
    does not veto the first stage.
    """
    pipeline = _pipeline(tmp_path, with_graph=True)
    retrieved = pipeline.retrieve(_RELATION_QUESTION, _QUERY)
    assert retrieved.rescued > 0
    assert retrieved.signals is None
    outcome = pipeline.select(_RELATION_QUESTION, retrieved, "graph_rescue_coe", 10_000)
    rescued_ids = {c.id for c in retrieved.union if c.metadata["rescued"]}
    assert not set(outcome.ranked) & rescued_ids
    assert set(outcome.ranked) == {c.id for c in retrieved.base}


def test_an_unknown_leg_is_refused_rather_than_silently_selected(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, with_graph=False)
    retrieved = pipeline.retrieve(_RELATION_QUESTION, _QUERY)
    with pytest.raises(ValueError, match="not a graph-rescue leg"):
        pipeline.select("q", retrieved, "packer", SELECT_BUDGET)


def test_the_ladder_refuses_to_run_a_rescue_leg_through_its_fuse_path(tmp_path: Path) -> None:
    """A rescue leg on the ladder path would return the base stage wearing the rescue leg's name."""
    export = _export(tmp_path)
    runner = Runner(export=export, dense=DenseBackend(export), bm25=None, graph=None, kv_path=None)
    with pytest.raises(ValueError, match="does not fuse a signal list"):
        runner.run("q", _QUERY, "graph_rescue")


# -- the caller-supplied redundancy signal is the packer's own number ----------------------------


def test_the_supplied_redundancy_is_exactly_the_packers_own_lexical_similarity(
    tmp_path: Path,
) -> None:
    """Precomputing the redundancy map is a change of COST, not of results — pinned pair by pair."""
    from latence_retrieval.pack import _lexical_similarity  # noqa: PLC2701 - the oracle

    pipeline = _pipeline(tmp_path, with_graph=False)
    candidates = [
        Candidate(id=f"c{i}", score=1.0 - i / 10, text=text) for i, text in enumerate(_TEXTS)
    ]
    decorated = pipeline._decorate(candidates)  # noqa: SLF001 - the unit under test
    for i, left in enumerate(decorated):
        table = left.metadata["redundancy"]
        # Every pair is present, so no pair can silently fall through to a recomputation.
        assert set(table) == {c.id for c in decorated} - {left.id}
        for j, right in enumerate(decorated):
            if i == j:
                continue
            assert table[right.id] == pytest.approx(
                _lexical_similarity(left.text, right.text), abs=0.0
            )


def test_the_packer_selects_identically_with_and_without_the_supplied_redundancy(
    tmp_path: Path,
) -> None:
    """The equivalence that licenses supplying it at all, asserted on the selection itself."""
    from latence_retrieval.pack import Packer

    pipeline = _pipeline(tmp_path, with_graph=False)
    candidates = [
        Candidate(id=f"c{i}", score=1.0 / (RANK_SCORE_K + i), text=text)
        for i, text in enumerate(_TEXTS)
    ]
    decorated = pipeline._decorate(candidates)  # noqa: SLF001 - the unit under test
    for budget in (12, 30, 60, 400):
        supplied = _packer(budget).process("q", decorated)
        native = Packer(
            {"budget": budget, "value_signals": {"kg_node_centrality": 0.25}}
        ).process("q", decorated)
        assert [c.id for c in supplied] == [c.id for c in native], budget


# -- the whole pipeline on the real export -------------------------------------------------------


@requires_export
def test_the_architecture_holds_on_the_real_export() -> None:
    """Every invariant above, once more against 3,264 real chunks and a real KG.

    A fixture can be made to satisfy an invariant by accident; a real corpus with a real hypergraph
    store, real alias seeding and a real traversal cannot.
    """
    from benchmark.s5.corpus import load_export
    from benchmark.s5.legs import build_runner
    from latence_retrieval.pack import HeuristicTokenCounter

    legs = ("graph_rescue", "graph_rescue_packed", "graph_rescue_coe")
    export = load_export(_MULTIHOP, "multihop_rag", with_graph_features=True)
    assert any(row.neighbor_node_ids for row in export.rows)
    assert any(row.centralities for row in export.rows)
    pipeline = RescuePipeline(runner=build_runner(export, legs))

    question = DATASETS["multihop_rag"].queries()[0].question
    # A real corpus vector stands in for a query vector: the pipeline's invariants are properties of
    # the *shape* of the ranking, not of which chunk happens to top it, so this needs no embedder.
    assert export.embeddings is not None
    vector = export.embeddings[0]
    retrieved = pipeline.retrieve(question, vector)

    base_ids = [c.id for c in retrieved.base]
    union_ids = [c.id for c in retrieved.union]
    # 1. the base head is preserved verbatim, 2. the tail is appended and deduped.
    assert union_ids[: len(base_ids)] == base_ids
    assert len(set(union_ids)) == len(union_ids)
    assert len(base_ids) <= 2 * POOL_K

    counter = HeuristicTokenCounter()
    texts = {c.id: c.text for c in retrieved.union}
    for leg in ("graph_rescue_packed", "graph_rescue_coe"):
        outcome = pipeline.select(question, retrieved, leg, SELECT_BUDGET)
        assert set(outcome.ranked) <= set(union_ids), leg
        assert sum(counter.count(texts[cid]) for cid in outcome.ranked) <= SELECT_BUDGET, leg
        assert outcome.tokens_packed is not None
