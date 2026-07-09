"""Unit tests for the type-aware tiered 2-hop candidate generator (bounded, rare-recall)."""

from __future__ import annotations

import logging

import pytest
from latence_linkpred_ultra.candidates import (
    CandidateConfig,
    CandidateGenerator,
    ObservedEdge,
)


def _org_graph() -> tuple[dict[str, str], list[ObservedEdge]]:
    node_types = {"acme": "org", "beta": "org", "gamma": "org", "alice": "person"}
    observed = [
        ObservedEdge("acme", "partner_of", "beta"),
        ObservedEdge("beta", "partner_of", "gamma"),
    ]
    return node_types, observed


def test_type_filter_excludes_wrong_type_endpoints() -> None:
    """A relation observed org->org never proposes a candidate touching a person node."""
    node_types, observed = _org_graph()
    cands = CandidateGenerator().generate(node_types, observed)
    assert cands, "should propose at least the transitive-style acme->gamma candidate"
    for c in cands:
        assert node_types[c.head] == "org"
        assert node_types[c.tail] == "org"
    # The person node never appears.
    assert all(c.head != "alice" and c.tail != "alice" for c in cands)


def test_never_proposes_observed_or_selfloop() -> None:
    node_types, observed = _org_graph()
    observed_keys = {(e.head, e.relation, e.tail) for e in observed}
    cands = CandidateGenerator().generate(node_types, observed)
    for c in cands:
        assert (c.head, c.relation, c.tail) not in observed_keys
        assert c.head != c.tail


def test_deterministic_sorted_output() -> None:
    node_types, observed = _org_graph()
    a = CandidateGenerator().generate(node_types, observed)
    b = CandidateGenerator().generate(node_types, observed)
    assert a == b
    assert a == sorted(a, key=lambda c: c.key)


def test_rare_relation_keeps_full_recall_but_dense_is_capped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rare relation is never capped; a dense one is capped AND logs the drop (no silent trunc)."""
    # A dense relation r_dense observed many times over a clique so its candidate set is large.
    nodes = [f"n{i}" for i in range(8)]
    node_types = {n: "org" for n in nodes}
    dense = [
        ObservedEdge(nodes[i], "r_dense", nodes[j])
        for i in range(len(nodes))
        for j in range(len(nodes))
        if i != j
    ][:20]
    # A rare relation observed twice.
    rare = [
        ObservedEdge("n0", "r_rare", "n1"),
        ObservedEdge("n1", "r_rare", "n2"),
    ]
    cfg = CandidateConfig(max_candidates_per_relation=5, rare_relation_max_edges=4)
    gen = CandidateGenerator(cfg)

    with caplog.at_level(logging.INFO):
        cands = gen.generate(node_types, dense + rare)

    dense_cands = [c for c in cands if c.relation == "r_dense"]
    rare_cands = [c for c in cands if c.relation == "r_rare"]
    # The dense relation is capped to exactly the cap (a drop happened and was recorded).
    assert len(dense_cands) == 5
    assert gen.capped.get("r_dense", 0) > 0
    # The cap is LOGGED (no silent truncation).
    assert any("capping" in rec.message for rec in caplog.records)
    # The rare relation keeps full recall (never in capped map); its candidates are marked rare.
    assert "r_rare" not in gen.capped
    assert rare_cands and all(c.rare for c in rare_cands)


def test_type_aware_off_admits_cross_type() -> None:
    node_types = {"acme": "org", "alice": "person"}
    observed = [
        ObservedEdge("acme", "rel", "alice"),
        ObservedEdge("alice", "rel", "acme"),
    ]
    cfg = CandidateConfig(type_aware=False)
    cands = CandidateGenerator(cfg).generate(node_types, observed)
    # With type-awareness off and both directions observed, no NEW candidate remains for this tiny
    # graph, but the generator must not crash and must stay deterministic.
    assert cands == CandidateGenerator(cfg).generate(node_types, observed)


def test_two_hop_pool_local_but_completable() -> None:
    """The 2-hop pool is local to a relation's endpoints; the reverse candidate is proposed."""
    node_types = {"a": "t", "b": "t", "c": "t"}
    observed = [ObservedEdge("a", "r", "b")]
    cands = CandidateGenerator().generate(node_types, observed)
    keys = {(c.head, c.relation, c.tail) for c in cands}
    # a<->b are the only nodes in the 2-hop neighbourhood; c is isolated (excluded by locality),
    # so the reverse edge b->r->a is the completable candidate — the tiered locality holds.
    assert ("b", "r", "a") in keys
    assert ("a", "r", "b") not in keys  # never re-proposes the observed edge


def test_two_hop_off_reaches_isolated_node() -> None:
    """With two_hop disabled the whole type-valid pool is considered (isolated c becomes reachable)."""
    node_types = {"a": "t", "b": "t", "c": "t"}
    observed = [ObservedEdge("a", "r", "b")]
    cfg = CandidateConfig(two_hop=False)
    cands = CandidateGenerator(cfg).generate(node_types, observed)
    keys = {(c.head, c.relation, c.tail) for c in cands}
    assert ("a", "r", "c") in keys or ("c", "r", "a") in keys
