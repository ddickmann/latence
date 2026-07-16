"""The multi-hop ``expand`` seam (:class:`MultiHopExpander`), tested at the boundary.

Every assertion is on the *observable output* of the transform — which candidates a hop bound lifts,
the augmented order, hop-decay across depths, ``top_k`` truncation, graceful behaviour when the KG
column is absent, and loud rejection when it is malformed — never the internal bonus maths, so the
tests survive a refactor of *how* the graph score is computed. The transform is driven through the
shipped faithful :class:`FakeGraphSource` (the seam's always-available second adapter): a test plants
a graph as ``(source, target)`` edge tuples exactly as ``graph-edges.parquet`` holds them and asserts
which candidates the traversal lifts — the real DuckDB/networkx engines are proven identical to that
fake in ``test_graph_sources.py`` (anti-false-green), so driving the fake here tests the *processor*
without a graph engine. The whole suite is pure-Python and offline (no ``duckdb``/``networkx``).
"""

from __future__ import annotations

from typing import Any

import pytest
from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    GraphSource,
    MultiHopExpander,
    resolve_processor,
)
from latence_retrieval.testing import FakeGraphSource


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


def _score_of(candidates: list[Candidate], cid: str) -> float:
    return next(c.score for c in candidates if c.id == cid)


def _cand(cid: str, score: float, *nodes: str) -> Candidate:
    """A candidate carrying its own entity node ids under the denormalised ``kg_node_ids`` column."""
    metadata: dict[str, Any] = {"kg_node_ids": list(nodes)} if nodes else {}
    return Candidate(id=cid, score=score, text=f"text-{cid}", metadata=metadata)


# A line graph anchored at the top seed's node "n0":  n0 - n1 - n2 - n3 ,  plus n0 - k1 (a second
# 1-hop branch) and an isolated edge p - q unreachable from the seed.
_LINE_EDGES = [("n0", "n1"), ("n1", "n2"), ("n2", "n3"), ("n0", "k1"), ("p", "q")]


# --- seam identity -----------------------------------------------------------------------------


def test_multihop_expander_satisfies_the_candidate_processor_seam() -> None:
    assert isinstance(MultiHopExpander(graph_source=FakeGraphSource([])), CandidateProcessor)


def test_fake_graph_source_satisfies_the_graph_source_seam() -> None:
    assert isinstance(FakeGraphSource([]), GraphSource)


def test_resolve_processor_selects_multihop_expand_by_name() -> None:
    processor = resolve_processor(
        {"provider": "processor.multihop_expand", "config": {"graph": {"source": "duckdb", "edges_uri": "x"}}}
    )
    assert isinstance(processor, MultiHopExpander)
    assert isinstance(processor, CandidateProcessor)


# --- the core multi-hop behaviour --------------------------------------------------------------


def test_a_two_hop_neighbor_is_lifted_over_a_higher_scored_stranger() -> None:
    # Seed S owns node n0. Candidate B owns n2, a TWO-hop graph neighbour of n0 (n0-n1-n2) — a link
    # NO 1-hop signal on the candidates could reveal (B's own reach is only its own 1-hop). With a
    # big-enough hop_weight the corroborated 2-hop neighbour B must rise over the unconnected A.
    seed = _cand("S", 1.0, "n0")
    a = _cand("A", 0.6, "x9")
    b = _cand("B", 0.4, "n2")

    expander = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 1.0, "hop_decay": 0.5, "max_hops": 2},
        graph_source=FakeGraphSource(_LINE_EDGES),
    )
    out = expander.process("q", [seed, a, b])

    assert out[0].id == "S"
    assert _ids(out).index("B") < _ids(out).index("A")
    # B is at hop 2 → hop_weight * decay**(2-1) = 1.0 * 0.5 = 0.5.
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)
    assert _score_of(out, "A") == pytest.approx(0.6)


def test_bonus_decays_geometrically_with_hop_distance() -> None:
    seed = _cand("S", 1.0, "n0")
    h1 = _cand("H1", 0.0, "n1")  # 1 hop
    h2 = _cand("H2", 0.0, "n2")  # 2 hops
    h3 = _cand("H3", 0.0, "n3")  # 3 hops

    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.8, "hop_decay": 0.5, "max_hops": 3},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, h1, h2, h3])

    assert _score_of(out, "H1") == pytest.approx(0.8)  # decay**0
    assert _score_of(out, "H2") == pytest.approx(0.8 * 0.5)  # decay**1
    assert _score_of(out, "H3") == pytest.approx(0.8 * 0.25)  # decay**2


def test_a_candidate_takes_its_closest_hop_when_it_owns_several_nodes() -> None:
    seed = _cand("S", 1.0, "n0")
    # Owns both a 3-hop node (n3) and a 1-hop node (k1) → the CLOSEST (1 hop) drives the bonus.
    multi = _cand("M", 0.0, "n3", "k1")

    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 1.0, "hop_decay": 0.5, "max_hops": 3},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, multi])
    assert _score_of(out, "M") == pytest.approx(1.0)  # 1-hop wins over the 3-hop node


def test_max_hops_bounds_the_traversal_depth() -> None:
    # n3 is 3 hops from the seed; with max_hops=2 it is out of reach and earns nothing.
    seed = _cand("S", 1.0, "n0")
    far = _cand("F", 0.3, "n3")

    shallow = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 1.0, "max_hops": 2},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, far])
    assert _score_of(shallow, "F") == pytest.approx(0.3)  # beyond the bound → no bonus

    deep = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 1.0, "hop_decay": 0.5, "max_hops": 3},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, far])
    assert _score_of(deep, "F") == pytest.approx(0.3 + 1.0 * 0.25)  # now within reach


def test_seed_top_k_bounds_which_hits_anchor_the_traversal() -> None:
    # Two disjoint stars: seed S1 (node a) reaches m; seed S2 (node b) reaches w. With seed_top_k=1
    # only S1 anchors, so a neighbour of S2 alone earns nothing until S2 also becomes a seed.
    edges = [("a", "m"), ("b", "w")]
    s1 = _cand("S1", 1.0, "a")
    s2 = _cand("S2", 0.9, "b")
    only_s2_neighbor = _cand("N", 0.3, "w")

    out1 = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 1},
        graph_source=FakeGraphSource(edges),
    ).process("q", [s1, s2, only_s2_neighbor])
    assert _score_of(out1, "N") == pytest.approx(0.3)  # S2 not a seed → no bonus

    out2 = MultiHopExpander(
        {"seed_top_k": 2, "hop_weight": 0.5, "max_hops": 1},
        graph_source=FakeGraphSource(edges),
    ).process("q", [s1, s2, only_s2_neighbor])
    assert _score_of(out2, "N") == pytest.approx(0.3 + 0.5)  # now S2 anchors


def test_the_seed_itself_earns_no_self_bonus() -> None:
    # The seed owns n0; n0 is excluded from its own neighbourhood, so the seed keeps its raw score.
    seed = _cand("S", 1.0, "n0")
    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 1.0, "max_hops": 2},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed])
    assert _score_of(out, "S") == pytest.approx(1.0)


# --- graceful degradation + no-op fast paths ---------------------------------------------------


def test_no_node_columns_at_all_is_an_order_preserving_no_op_without_touching_the_graph() -> None:
    # A graph source that would explode if queried proves the no-op path never touches it.
    class _Exploding:
        def neighbors(self, seeds: Any, max_hops: int) -> dict[str, int]:
            raise AssertionError("the graph source must not be queried when there are no node ids")

    plain = [
        Candidate(id="A", score=0.9, text="a"),
        Candidate(id="B", score=0.5, text="b"),
        Candidate(id="C", score=0.1, text="c"),
    ]
    out = MultiHopExpander({"seed_top_k": 2}, graph_source=_Exploding()).process("q", plain)
    assert _ids(out) == ["A", "B", "C"]
    assert [c.score for c in out] == [0.9, 0.5, 0.1]


def test_hop_weight_zero_is_a_no_op_that_never_touches_the_graph() -> None:
    class _Exploding:
        def neighbors(self, seeds: Any, max_hops: int) -> dict[str, int]:
            raise AssertionError("hop_weight=0 must short-circuit before any traversal")

    seed = _cand("S", 1.0, "n0")
    b = _cand("B", 0.4, "n1")
    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.0}, graph_source=_Exploding()
    ).process("q", [seed, b])
    assert _ids(out) == ["S", "B"]


def test_a_candidate_missing_the_node_key_just_earns_nothing() -> None:
    seed = _cand("S", 1.0, "n0")
    linked = _cand("B", 0.4, "n1")
    bare = Candidate(id="C", score=0.5, text="c")  # no KG metadata at all

    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 1},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, linked, bare])
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)
    assert _score_of(out, "C") == pytest.approx(0.5)  # untouched, no crash


def test_custom_node_ids_key_is_honoured() -> None:
    seed = Candidate(id="S", score=1.0, text="s", metadata={"nodes": ["n0"]})
    b = Candidate(id="B", score=0.4, text="b", metadata={"nodes": ["n1"]})
    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 1, "node_ids_key": "nodes"},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, b])
    assert _score_of(out, "B") == pytest.approx(0.4 + 0.5)


# --- output shape, determinism, no mutation ----------------------------------------------------


def test_top_k_truncates_to_the_augmented_head() -> None:
    seed = _cand("S", 1.0, "n0")
    b = _cand("B", 0.4, "n1")  # lifted to 0.9
    c = _cand("C", 0.3, "zz")  # unconnected, stays 0.3

    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 1, "top_k": 2},
        graph_source=FakeGraphSource(_LINE_EDGES),
    ).process("q", [seed, b, c])
    assert _ids(out) == ["S", "B"]  # C dropped by the budget


def test_ties_break_by_ascending_id_for_determinism() -> None:
    a = Candidate(id="zz", score=0.5, text="a")
    b = Candidate(id="aa", score=0.5, text="b")
    out = MultiHopExpander(graph_source=FakeGraphSource([])).process("q", [a, b])
    assert _ids(out) == ["aa", "zz"]


def test_input_is_not_mutated_and_repeated_calls_are_identical() -> None:
    seed = _cand("S", 1.0, "n0")
    b = _cand("B", 0.4, "n1")
    inputs = [seed, b]

    cfg: dict[str, Any] = {"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 1}
    first = MultiHopExpander(cfg, graph_source=FakeGraphSource(_LINE_EDGES)).process("q", inputs)
    second = MultiHopExpander(cfg, graph_source=FakeGraphSource(_LINE_EDGES)).process("q", inputs)

    assert [c.score for c in inputs] == [1.0, 0.4]  # originals untouched (frozen)
    assert _ids(first) == _ids(second)
    assert [c.score for c in first] == [c.score for c in second]


def test_empty_input_returns_empty() -> None:
    assert MultiHopExpander(graph_source=FakeGraphSource([])).process("q", []) == []


# --- dependency injection + lazy source build --------------------------------------------------


def test_an_injected_graph_source_is_used_and_no_config_source_is_needed() -> None:
    # No 'graph' config at all — the injected source is enough (tests never build a real engine).
    seed = _cand("S", 1.0, "n0")
    b = _cand("B", 0.4, "n1")
    out = MultiHopExpander(
        {"seed_top_k": 1, "hop_weight": 0.5, "max_hops": 1}, graph_source=FakeGraphSource(_LINE_EDGES)
    ).process("q", [seed, b])
    assert _score_of(out, "B") == pytest.approx(0.9)


def test_missing_both_injected_source_and_config_fails_loudly_only_when_traversal_is_needed() -> None:
    # Construction must NOT build a source (the registry constructs with no engine installed)…
    expander = MultiHopExpander({"seed_top_k": 1, "hop_weight": 0.5})
    # …and a no-op call (no node ids) still never needs one.
    assert expander.process("q", [Candidate(id="A", score=1.0, text="a")]) == [
        Candidate(id="A", score=1.0, text="a")
    ]
    # But a call that actually needs to traverse raises a clear config error.
    with pytest.raises(ValueError, match="needs a graph to traverse"):
        expander.process("q", [_cand("S", 1.0, "n0"), _cand("B", 0.4, "n1")])


# --- config + boundary rejection (anti-false-green) --------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        {"seed_top_k": 0},
        {"seed_top_k": -1},
        {"seed_top_k": 1.5},
        {"seed_top_k": True},
        {"hop_weight": -0.1},
        {"max_hops": 0},
        {"max_hops": -1},
        {"max_hops": 1.5},
        {"max_hops": True},
        {"max_hops": 65},  # above the bounded-depth ceiling
        {"hop_decay": 0},  # must be > 0
        {"hop_decay": 1.5},  # must be <= 1
        {"hop_decay": -0.5},
        {"top_k": 0},
        {"node_ids_key": ""},
        {"graph": "not-a-mapping"},
    ],
)
def test_invalid_config_is_rejected_at_construction(config: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        MultiHopExpander(config, graph_source=FakeGraphSource([]))


def test_malformed_node_column_fails_loudly_not_silently() -> None:
    seed = _cand("S", 1.0, "n0")
    bad = Candidate(id="B", score=0.4, text="b", metadata={"kg_node_ids": {"n1": 1}})
    with pytest.raises(TypeError):
        MultiHopExpander({"seed_top_k": 1}, graph_source=FakeGraphSource(_LINE_EDGES)).process(
            "q", [seed, bad]
        )


def test_non_string_node_element_is_rejected() -> None:
    seed = _cand("S", 1.0, "n0")
    bad = Candidate(id="B", score=0.4, text="b", metadata={"kg_node_ids": ["n1", 7]})
    with pytest.raises(ValueError):
        MultiHopExpander({"seed_top_k": 1}, graph_source=FakeGraphSource(_LINE_EDGES)).process(
            "q", [seed, bad]
        )
