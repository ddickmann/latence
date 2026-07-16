"""The ``expand`` seam (:class:`Expander`, 1-hop graph-augmented), tested at the boundary.

Every assertion is on the *observable output* of the transform — which candidates rise, the augmented
order, ``top_k`` truncation, graceful behaviour when KG columns are absent, and loud rejection when a
present column is malformed — never the internal bonus maths, so the tests survive a refactor of *how*
the graph score is computed. The fixtures are candidates carrying the denormalised KG columns
(``kg_node_ids`` / ``neighbor_node_ids`` / ``kg_node_community`` / ``kg_node_centrality``) exactly as a
customer's store would hand them back (the AC), and the whole suite is pure-Python and offline — expand
has no model or store dependency (ADR-0051's default 1-hop path holds no graph).
"""

from __future__ import annotations

from typing import Any

import pytest
from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    Expander,
    resolve_processor,
)


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


def _score_of(candidates: list[Candidate], cid: str) -> float:
    return next(c.score for c in candidates if c.id == cid)


def _cand(cid: str, score: float, **kg: Any) -> Candidate:
    """A candidate carrying whatever denormalised KG columns a test supplies as metadata."""
    return Candidate(id=cid, score=score, text=f"text-{cid}", metadata=dict(kg))


# --- seam identity -----------------------------------------------------------------------------


def test_expander_satisfies_the_candidate_processor_seam() -> None:
    assert isinstance(Expander(), CandidateProcessor)


def test_resolve_processor_selects_expand_by_name() -> None:
    processor = resolve_processor({"provider": "processor.expand", "config": {}})
    assert isinstance(processor, Expander)
    assert isinstance(processor, CandidateProcessor)


# --- the core 1-hop behaviour ------------------------------------------------------------------


def test_one_hop_neighbor_of_the_top_seed_is_lifted_above_a_higher_scored_stranger() -> None:
    # Seed S is the top hit; its triples reach node "n2". Candidate B owns node "n2" (a KG edge from
    # S to B), so B is a 1-hop graph neighbour of the seed. Stranger A outscores B at first stage but
    # has no graph connection. A big-enough hop_weight must lift the corroborated neighbour B over A.
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"])
    a = _cand("A", 0.6, kg_node_ids=["x9"], neighbor_node_ids=[])
    b = _cand("B", 0.4, kg_node_ids=["n2"], neighbor_node_ids=[])

    expander = Expander({"seed_top_k": 1, "hop_weight": 0.5})
    out = expander.process("q", [seed, a, b])

    assert out[0].id == "S"
    assert _ids(out).index("B") < _ids(out).index("A")
    # B earned exactly one hop bonus (one seed linked); A earned none.
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)
    assert _score_of(out, "A") == pytest.approx(0.6)


def test_shared_entity_co_mention_counts_as_a_one_hop_link() -> None:
    # No neighbor edges at all — B and the seed simply co-mention entity "n1". A co-mention is a 1-hop
    # graph connection, so B is still boosted.
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=[])
    b = _cand("B", 0.4, kg_node_ids=["n1"], neighbor_node_ids=[])
    other = _cand("C", 0.5, kg_node_ids=["z"], neighbor_node_ids=[])

    out = Expander({"seed_top_k": 1, "hop_weight": 1.0}).process("q", [seed, b, other])

    assert _score_of(out, "B") == pytest.approx(0.4 + 1.0)
    assert _score_of(out, "C") == pytest.approx(0.5)


def test_reverse_direction_edge_is_a_link() -> None:
    # The seed owns node "n1"; candidate B's triples reach "n1" (edge B → seed). The connection must be
    # detected from either direction of neighbor_node_ids.
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=[])
    b = _cand("B", 0.4, kg_node_ids=["b0"], neighbor_node_ids=["n1"])

    out = Expander({"seed_top_k": 1, "hop_weight": 0.5}).process("q", [seed, b])
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)


def test_bonus_scales_with_the_number_of_distinct_seeds_linked() -> None:
    # Two seeds S1, S2 both reach node "n" that candidate B owns → B links to two seeds → double hop
    # bonus. Candidate C links to only S1.
    s1 = _cand("S1", 1.0, kg_node_ids=["a"], neighbor_node_ids=["n"])
    s2 = _cand("S2", 0.9, kg_node_ids=["b"], neighbor_node_ids=["n"])
    b = _cand("B", 0.2, kg_node_ids=["n"], neighbor_node_ids=[])
    c = _cand("C", 0.2, kg_node_ids=["a"], neighbor_node_ids=[])  # co-mentions only S1's node "a"

    out = Expander({"seed_top_k": 2, "hop_weight": 0.5}).process("q", [s1, s2, b, c])
    assert _score_of(out, "B") == pytest.approx(0.2 + 2 * 0.5)
    assert _score_of(out, "C") == pytest.approx(0.2 + 1 * 0.5)


def test_seed_top_k_bounds_which_hits_anchor_the_expansion() -> None:
    # With seed_top_k=1 only S1 anchors, so a neighbour of S2 alone earns nothing.
    s1 = _cand("S1", 1.0, kg_node_ids=["a"], neighbor_node_ids=["p"])
    s2 = _cand("S2", 0.9, kg_node_ids=["b"], neighbor_node_ids=["q"])
    only_s2_neighbor = _cand("N", 0.3, kg_node_ids=["q"], neighbor_node_ids=[])

    out1 = Expander({"seed_top_k": 1, "hop_weight": 0.5}).process("q", [s1, s2, only_s2_neighbor])
    assert _score_of(out1, "N") == pytest.approx(0.3)  # S2 is not a seed → no bonus

    out2 = Expander({"seed_top_k": 2, "hop_weight": 0.5}).process("q", [s1, s2, only_s2_neighbor])
    assert _score_of(out2, "N") == pytest.approx(0.3 + 0.5)  # now S2 anchors


# --- optional signals: community + centrality --------------------------------------------------


def test_shared_community_bonus_is_opt_in_and_flat() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=[], kg_node_community=["c-red"])
    same = _cand("B", 0.4, kg_node_ids=["z"], neighbor_node_ids=[], kg_node_community=["c-red"])
    diff = _cand("C", 0.4, kg_node_ids=["y"], neighbor_node_ids=[], kg_node_community=["c-blue"])

    # Default community_weight=0 → no community effect.
    off = Expander({"seed_top_k": 1}).process("q", [seed, same, diff])
    assert _score_of(off, "B") == pytest.approx(0.4)

    on = Expander({"seed_top_k": 1, "hop_weight": 0.0, "community_weight": 0.3}).process(
        "q", [seed, same, diff]
    )
    assert _score_of(on, "B") == pytest.approx(0.4 + 0.3)
    assert _score_of(on, "C") == pytest.approx(0.4)  # different community → nothing


def test_centrality_bonus_uses_the_candidates_peak_and_is_opt_in() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=[])
    hub = _cand("H", 0.4, kg_node_ids=["a", "b"], neighbor_node_ids=[], kg_node_centrality=[0.2, 0.9])

    off = Expander({"seed_top_k": 1}).process("q", [seed, hub])
    assert _score_of(off, "H") == pytest.approx(0.4)

    on = Expander({"seed_top_k": 1, "hop_weight": 0.0, "centrality_weight": 2.0}).process(
        "q", [seed, hub]
    )
    assert _score_of(on, "H") == pytest.approx(0.4 + 2.0 * 0.9)  # peak centrality 0.9


# --- graceful degradation (the AC) -------------------------------------------------------------


def test_no_kg_columns_at_all_is_an_order_preserving_no_op() -> None:
    plain = [
        Candidate(id="A", score=0.9, text="a"),
        Candidate(id="B", score=0.5, text="b"),
        Candidate(id="C", score=0.1, text="c"),
    ]
    out = Expander().process("q", plain)
    assert _ids(out) == ["A", "B", "C"]
    assert [c.score for c in out] == [0.9, 0.5, 0.1]


def test_partial_columns_degrade_a_candidate_missing_keys_just_earns_nothing() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"])
    linked = _cand("B", 0.4, kg_node_ids=["n2"], neighbor_node_ids=[])
    bare = Candidate(id="C", score=0.5, text="c")  # no KG metadata at all

    out = Expander({"seed_top_k": 1, "hop_weight": 0.5}).process("q", [seed, linked, bare])
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)
    assert _score_of(out, "C") == pytest.approx(0.5)  # untouched, no crash


def test_custom_metadata_keys_are_honoured() -> None:
    # A store that denormalised the columns under different names is configured, not hard-coded.
    seed = _cand("S", 1.0, nodes=["n1"], reach=["n2"])
    b = _cand("B", 0.4, nodes=["n2"], reach=[])
    out = Expander(
        {"seed_top_k": 1, "hop_weight": 0.5, "node_ids_key": "nodes", "neighbor_ids_key": "reach"}
    ).process("q", [seed, b])
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)


# --- output shape, determinism, no mutation ----------------------------------------------------


def test_top_k_truncates_to_the_augmented_head() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"])
    b = _cand("B", 0.4, kg_node_ids=["n2"], neighbor_node_ids=[])  # lifted to 0.9
    c = _cand("C", 0.3, kg_node_ids=["z"], neighbor_node_ids=[])  # stays 0.3

    out = Expander({"seed_top_k": 1, "hop_weight": 0.5, "top_k": 2}).process("q", [seed, b, c])
    assert _ids(out) == ["S", "B"]  # C dropped by the budget


def test_ties_break_by_ascending_id_for_determinism() -> None:
    a = Candidate(id="zz", score=0.5, text="a")
    b = Candidate(id="aa", score=0.5, text="b")
    out = Expander().process("q", [a, b])
    assert _ids(out) == ["aa", "zz"]


def test_input_is_not_mutated_and_repeated_calls_are_identical() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"])
    b = _cand("B", 0.4, kg_node_ids=["n2"], neighbor_node_ids=[])
    inputs = [seed, b]

    first = Expander({"seed_top_k": 1, "hop_weight": 0.5}).process("q", inputs)
    second = Expander({"seed_top_k": 1, "hop_weight": 0.5}).process("q", inputs)

    assert [c.score for c in inputs] == [1.0, 0.4]  # originals untouched (frozen)
    assert _ids(first) == _ids(second)
    assert [c.score for c in first] == [c.score for c in second]


def test_empty_input_returns_empty() -> None:
    assert Expander().process("q", []) == []


# --- config + boundary rejection (anti-false-green) --------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"seed_top_k": 0},
        {"seed_top_k": -1},
        {"seed_top_k": 1.5},
        {"seed_top_k": True},
        {"hop_weight": -0.1},
        {"community_weight": -1},
        {"centrality_weight": -2},
        {"top_k": 0},
        {"node_ids_key": ""},
        {"neighbor_ids_key": 3},
    ],
)
def test_invalid_config_is_rejected_at_construction(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Expander(config)


def test_malformed_node_id_column_fails_loudly_not_silently() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"])
    # A dict where a list of ids is expected — a real graph-aware ranker cannot use this; reject it.
    bad = _cand("B", 0.4, kg_node_ids={"n2": 1}, neighbor_node_ids=[])
    with pytest.raises(TypeError):
        Expander({"seed_top_k": 1}).process("q", [seed, bad])


def test_non_string_node_id_element_is_rejected() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"])
    bad = _cand("B", 0.4, kg_node_ids=["n2", 7], neighbor_node_ids=[])
    with pytest.raises(ValueError):
        Expander({"seed_top_k": 1}).process("q", [seed, bad])


def test_non_finite_centrality_is_rejected() -> None:
    seed = _cand("S", 1.0, kg_node_ids=["n1"], neighbor_node_ids=[])
    bad = _cand("B", 0.4, kg_node_ids=["a"], neighbor_node_ids=[], kg_node_centrality=[float("inf")])
    with pytest.raises(ValueError):
        Expander({"seed_top_k": 1, "centrality_weight": 1.0}).process("q", [seed, bad])
