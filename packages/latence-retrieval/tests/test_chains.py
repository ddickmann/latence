# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""M5 (ticket 16): chain assembly + structured emission (spec 3.4) and selection traces (3.5).

The spec-M5 acceptance criteria: connected components over shared entities are the chains
(singletons included); intra-chain order is BFS hop distance from the anchors with a ρ-desc
tie-break and identical inputs yield identical output BYTES; the spec-6.3 directed-relation
preference reorders a chain only when a ``relations_lookup`` is supplied (without it the
undirected order is untouched); peripheral components render last under their own header; an
empty selection renders an empty context without crashing; the selection trace records every
candidate exactly once as accepted/rejected with trimmed listed, the default ``trace_sink=None``
path leaves the selector's behaviour identical (test_coe.py runs UNCHANGED beside this file);
no trace ever carries raw ``tau_text``; and the e2e over the M3/M4 fixture corpus renders the
planted 2-hop chain as ONE chain with the bridge ordered after the direct evidence.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from latence_retrieval.chains import ChainAssembly, assemble_chains
from latence_retrieval.coe import CoeConfig, CoeSelection, select
from latence_retrieval.hypergraph import HyperedgeKV, HypergraphSignals, hypergraph_signals
from latence_retrieval.pack import HeuristicTokenCounter
from test_hypergraph import _build_kv

_STORED_COUNTER = "latence:estimate_tokens:v1"  # what the _build_kv fixture records per edge


def _signals(
    edges: dict[str, dict[str, Any]], anchors: dict[str, float]
) -> HypergraphSignals:
    """Build a HypergraphSignals directly (the M3 output contract) for unit tests."""
    rho = {hid: float(edge["rho"]) for hid, edge in edges.items()}
    fields = {
        hid: {
            "entity_node_ids": list(edge["nodes"]),
            "gamma": float(edge.get("gamma", 0.8)),
            "token_cost": int(edge.get("cost", 50)),
            "token_counter": edge.get("counter", _STORED_COUNTER),
            "support": int(edge.get("support", 1)),
            "has_relations": bool(edge.get("has_relations", True)),
        }
        for hid, edge in edges.items()
    }
    return HypergraphSignals(
        rho=rho,
        fields=fields,
        anchors=anchors,
        vertex_confidence={},
        expanded_candidates=len(edges),
        pruned_candidates=len(edges),
    )


def _selection(ids: list[str]) -> CoeSelection:
    """A CoeSelection carrying exactly ``ids`` (chain assembly reads only ``selected``)."""
    return CoeSelection(
        selected=list(ids),
        marginals=dict.fromkeys(ids, 1.0),
        total_cost=0,
        budget=1000,
        verified=False,
        trimmed=[],
    )


def _tau(hid: str) -> str:
    return f"text of {hid}"


def _assemble(
    edges: dict[str, dict[str, Any]],
    anchors: dict[str, float],
    selected: list[str],
    **kwargs: Any,
) -> ChainAssembly:
    return assemble_chains(
        _selection(selected), _signals(edges, anchors), tau_lookup=_tau, **kwargs
    )


# -- spec 3.4: connected components over shared entities are the chains -----------


def test_components_follow_shared_entity_connectivity_including_singletons() -> None:
    edges = {
        "e1": {"rho": 0.9, "nodes": ["n_q", "a"]},
        "e2": {"rho": 0.5, "nodes": ["a", "b"]},  # shares 'a' with e1 → same chain
        "e3": {"rho": 0.7, "nodes": ["x", "y"]},  # shares nothing → its own singleton chain
    }
    assembly = _assemble(edges, {"n_q": 1.0}, ["e1", "e2", "e3"])
    assert [len(c.steps) for c in assembly.chains] == [2, 1]
    assert [s.hyperedge_id for s in assembly.chains[0].steps] == ["e1", "e2"]
    assert [s.hyperedge_id for s in assembly.chains[1].steps] == ["e3"]
    assert assembly.chains[0].anchor_node_ids == ["n_q"]
    assert assembly.chains[1].peripheral, "a singleton with no anchor entity is peripheral"


# -- spec 3.4: BFS-from-anchor order, rho tie-break, deterministic bytes ----------


def test_bfs_hop_order_with_rho_tiebreak_and_deterministic_bytes() -> None:
    """Hop distance from V0 is primary — the hop-2 edge loses to LOWER-ρ hop-1 edges — and
    within a hop layer ρ descends. Two identical calls emit identical bytes."""
    edges = {
        "e_root": {"rho": 0.8, "nodes": ["n_q", "m"], "gamma": 0.91},
        "e_hi": {"rho": 0.9, "nodes": ["m", "c1"]},
        "e_lo": {"rho": 0.4, "nodes": ["m", "c2"]},
        "e_far": {"rho": 0.99, "nodes": ["c2", "d"]},  # highest ρ, but 2 hops out
    }
    selected = ["e_far", "e_lo", "e_hi", "e_root"]  # acceptance order must not matter
    first = _assemble(edges, {"n_q": 1.0}, selected)
    second = _assemble(edges, {"n_q": 1.0}, selected)
    order = [s.hyperedge_id for s in first.chains[0].steps]
    assert order == ["e_root", "e_hi", "e_lo", "e_far"]
    assert [s.hop for s in first.chains[0].steps] == [0, 1, 1, 2]
    assert first.text.encode() == second.text.encode(), "identical inputs ⇒ identical bytes"
    assert "## Chain 1 (anchor: n_q)" in first.text
    assert "[step 1] (rho=0.80, gamma=0.91, src=e_root) text of e_root" in first.text


def test_src_lookup_names_the_source_and_absent_falls_back_to_the_id() -> None:
    edges = {"e1": {"rho": 0.9, "nodes": ["n_q", "a"]}}
    named = _assemble(edges, {"n_q": 1.0}, ["e1"], src_lookup={"e1": "doc_17#c4"}.get)
    plain = _assemble(edges, {"n_q": 1.0}, ["e1"])
    assert "src=doc_17#c4" in named.text
    assert "src=e1" in plain.text


# -- spec 6.3: the directed-relation ordering preference --------------------------


def test_directed_preference_follows_head_to_tail_only_when_lookup_is_supplied() -> None:
    """e_a and e_z tie on (hop, ρ); undirected order breaks by id (e_a first). e_a's only
    relation points BACKWARD into the resolved evidence (head x is unresolved, tail m is),
    e_z's follows head→tail forward (head m already resolved) — so the directed preference
    must put e_z first, and only when the lookup is supplied."""
    edges = {
        "e_first": {"rho": 1.0, "nodes": ["n_q", "m"]},
        "e_a": {"rho": 0.6, "nodes": ["m", "x"]},
        "e_z": {"rho": 0.6, "nodes": ["m", "y"]},
    }
    triples = {
        "e_first": [("n_q", "owns", "m")],
        "e_a": [("x", "made", "m")],  # backward: arrives INTO the resolved set
        "e_z": [("m", "led", "y")],  # forward: continues head→tail from resolved m
    }
    undirected = _assemble(edges, {"n_q": 1.0}, ["e_first", "e_a", "e_z"])
    directed = _assemble(
        edges, {"n_q": 1.0}, ["e_first", "e_a", "e_z"], relations_lookup=lambda h: triples[h]
    )
    assert [s.hyperedge_id for s in undirected.chains[0].steps] == ["e_first", "e_a", "e_z"]
    assert [s.hyperedge_id for s in directed.chains[0].steps] == ["e_first", "e_z", "e_a"]
    assert "m --led--> y" in directed.text, "relations render on the step's detail line"
    assert "relations:" not in undirected.text, "no lookup ⇒ no relation lines"


def test_directed_preference_never_overrides_hop_distance() -> None:
    """Direction refines WITHIN a hop layer; a forward-directed hop-2 edge must still come
    after an undirected hop-1 edge — spec 3.4's evidence distance stays primary."""
    edges = {
        "e_first": {"rho": 1.0, "nodes": ["n_q", "m"]},
        "e_mid": {"rho": 0.5, "nodes": ["m", "c"]},
        "e_far": {"rho": 0.9, "nodes": ["c", "d"]},
    }
    triples: dict[str, list[tuple[str, str, str]]] = {
        "e_first": [],
        "e_mid": [("c", "made", "m")],  # backward at hop 1
        "e_far": [("c", "led", "d")],  # forward at hop 2
    }
    assembly = _assemble(
        edges,
        {"n_q": 1.0},
        ["e_first", "e_mid", "e_far"],
        relations_lookup=lambda h: triples[h],
    )
    assert [s.hyperedge_id for s in assembly.chains[0].steps] == ["e_first", "e_mid", "e_far"]


# -- spec 3.4: peripheral components render last under their own header -----------


def test_peripheral_components_render_last_under_the_peripheral_header() -> None:
    """The peripheral chain outranks the anchored one on ρ — it must STILL come last."""
    edges = {
        "e_anchored": {"rho": 0.5, "nodes": ["n_q", "a"]},
        "e_orphan": {"rho": 0.99, "nodes": ["x", "y"]},
        "e_orphan2": {"rho": 0.8, "nodes": ["y", "z"]},
    }
    assembly = _assemble(edges, {"n_q": 1.0}, ["e_orphan", "e_anchored", "e_orphan2"])
    assert [c.peripheral for c in assembly.chains] == [False, True]
    orphan_chain = assembly.chains[1]
    assert [s.hyperedge_id for s in orphan_chain.steps] == ["e_orphan", "e_orphan2"], (
        "a peripheral chain seeds at its best-ρ hyperedge and orders outward from it"
    )
    header = assembly.text.index("## Peripheral evidence")
    assert assembly.text.index("src=e_anchored") < header < assembly.text.index("src=e_orphan")
    assert "(peripheral)" in assembly.text


def test_empty_selection_yields_empty_context_without_crashing() -> None:
    traces: list[dict[str, Any]] = []
    assembly = _assemble({}, {}, [], trace_sink=traces.append)
    assert assembly.chains == [] and assembly.text == ""
    assert traces[0]["chain_count"] == 0 and traces[0]["peripheral_count"] == 0


def test_selected_id_missing_from_signals_fails_loudly() -> None:
    with pytest.raises(ValueError, match="absent from the"):
        _assemble({"e1": {"rho": 0.9, "nodes": ["n_q"]}}, {"n_q": 1.0}, ["e1", "ghost"])


def test_missing_tau_text_fails_loudly() -> None:
    signals = _signals({"e1": {"rho": 0.9, "nodes": ["n_q", "a"]}}, {"n_q": 1.0})
    with pytest.raises(ValueError, match="tau_text"):
        assemble_chains(_selection(["e1"]), signals, tau_lookup=lambda _hid: None)


# -- spec 3.5: the selection trace ------------------------------------------------


def _trace_scenario() -> tuple[HypergraphSignals, dict[str, str]]:
    ten_words = "one two three four five six seven eight nine ten"  # 10 heuristic tokens
    edges: dict[str, dict[str, Any]] = {
        "e_strong": {"rho": 1.0, "nodes": ["n1", "n2"], "cost": 5},
        "e_weakling": {"rho": 0.3, "nodes": ["m1", "m2"], "cost": 5},
        "e_filler": {"rho": 0.001, "nodes": ["f1", "f2"], "gamma": 0.05, "cost": 5},
    }
    taus = {hid: ten_words for hid in edges}
    return _signals(edges, {"n1": 1.0}), taus


def test_trace_records_every_candidate_exactly_once_with_trimmed_listed() -> None:
    """e_strong is accepted and survives, e_weakling is accepted then trimmed by the ADR-0058
    pass, e_filler is rejected below theta_min — the trace must carry each exactly once as
    accepted/rejected, list the trim, and serialise as JSON."""
    signals, taus = _trace_scenario()
    traces: list[dict[str, Any]] = []
    result = select(
        signals,
        budget=12,
        tau_lookup=taus.get,
        token_counter=HeuristicTokenCounter(),
        counter_identity="tiktoken:cl100k_base",  # differs from stored → pass is MANDATORY
        config=CoeConfig(verify_on_match=False),
        trace_sink=traces.append,
    )
    assert result.selected == ["e_strong"] and result.trimmed == ["e_weakling"]
    assert len(traces) == 1, "ONE dict per query"
    trace = traces[0]
    json.dumps(trace)  # JSON-serialisable, or this raises
    assert trace["stage"] == "select"
    assert trace["anchors"] == {"n1": 1.0}
    assert trace["candidate_count"] == 3 and trace["budget"] == 12
    decisions = {step["hyperedge_id"]: step["decision"] for step in trace["steps"]}
    assert len(trace["steps"]) == len(decisions) == 3, "every candidate exactly once"
    assert decisions == {"e_strong": "accepted", "e_weakling": "accepted", "e_filler": "rejected"}
    trimmed_flags = {step["hyperedge_id"]: step["trimmed"] for step in trace["steps"]}
    assert trimmed_flags == {"e_strong": False, "e_weakling": True, "e_filler": False}
    assert trace["trimmed"] == ["e_weakling"] and trace["selected"] == ["e_strong"]
    assert trace["verified"] is True and trace["total_cost"] == result.total_cost
    assert all(
        isinstance(step["v0"], float) and isinstance(step["marginal"], float)
        for step in trace["steps"]
    )


def test_default_none_sink_leaves_the_selection_identical() -> None:
    """The sink is pure observation: with and without it, select returns the same result.
    (test_coe.py running UNCHANGED beside this file pins the default path itself.)"""
    signals, taus = _trace_scenario()

    def run(sink: Any) -> Any:
        return select(
            signals,
            budget=12,
            tau_lookup=taus.get,
            token_counter=HeuristicTokenCounter(),
            counter_identity="tiktoken:cl100k_base",
            config=CoeConfig(verify_on_match=False),
            trace_sink=sink,
        )

    assert run(None) == run(lambda _t: None)


def test_no_tau_text_ever_reaches_a_trace() -> None:
    """Traces may travel further than the corpus: ids/counts/scores only, by design."""
    marker = "TOPSECRET_SPAN_CONTENT"
    edges: dict[str, dict[str, Any]] = {
        "e1": {"rho": 1.0, "nodes": ["n1", "n2"], "cost": 5},
        "e2": {"rho": 0.8, "nodes": ["n2", "n3"], "cost": 5},
    }
    signals = _signals(edges, {"n1": 1.0})
    taus = {hid: f"{marker} {hid}" for hid in edges}
    traces: list[dict[str, Any]] = []
    result = select(
        signals,
        budget=100,
        tau_lookup=taus.get,
        token_counter=HeuristicTokenCounter(),
        counter_identity=_STORED_COUNTER,
        config=CoeConfig(verify_on_match=True),
        trace_sink=traces.append,
    )
    assembly = assemble_chains(
        result, signals, tau_lookup=lambda hid: taus[hid], trace_sink=traces.append
    )
    assert marker in assembly.text, "the CONTEXT carries the span text — that is its job"
    assert traces and marker not in json.dumps(traces), "the TRACES never do"


def test_chains_trace_reports_counts_sizes_and_order_through_the_same_sink_shape() -> None:
    edges = {
        "e1": {"rho": 0.9, "nodes": ["n_q", "a"]},
        "e2": {"rho": 0.5, "nodes": ["a", "b"]},
        "e3": {"rho": 0.7, "nodes": ["x", "y"]},
    }
    traces: list[dict[str, Any]] = []
    _assemble(edges, {"n_q": 1.0}, ["e1", "e2", "e3"], trace_sink=traces.append)
    assert len(traces) == 1
    trace = traces[0]
    json.dumps(trace)
    assert trace["stage"] == "chains"
    assert trace["chain_count"] == 2 and trace["chain_sizes"] == [2, 1]
    assert trace["anchored_count"] == 1 and trace["peripheral_count"] == 1
    assert trace["directed_preference"] is False
    assert trace["chains"][0]["hyperedge_ids"] == ["e1", "e2"]
    assert trace["chains"][1]["peripheral"] is True


# -- e2e: KV fixture → signals → select → chains ----------------------------------


def test_e2e_two_hop_corpus_renders_one_chain_with_the_bridge_after_the_direct() -> None:
    """The full M3→M4→M5 path over the fixture corpus: the planted 2-hop chain is selected
    (M4's e2e), and chain assembly renders it as ONE anchored chain whose step 1 is the
    direct evidence and step 2 the bridge — the reading order generation wants."""
    hyperedges: dict[str, dict[str, Any]] = {
        "h_direct": {"nodes": ["n_seed", "n_mid"], "gamma": 0.80, "support": 3},
        "h_bridge": {"nodes": ["n_mid", "n_answer"], "gamma": 0.80, "support": 3},
        "h_distract": {"nodes": ["n_seed", "n_x"], "gamma": 0.95, "has_relations": False},
        **{
            f"h_noise{i}": {"nodes": [f"n_a{i}", f"n_b{i}"], "gamma": 0.9} for i in range(20)
        },
    }
    kv = HyperedgeKV.from_bytes(_build_kv(hyperedges, {"Seed Corp": ["n_seed"]}))
    eu_sims = {"h_distract": 1.0, "h_direct": 0.6, "h_bridge": 0.7}
    signals = hypergraph_signals(kv, alias_surfaces=["Seed Corp"], eu_sims=eu_sims)

    traces: list[dict[str, Any]] = []
    result = select(
        signals,
        budget=100,
        eu_sims=eu_sims,
        relation_bonus={"h_direct": 0.9, "h_bridge": 0.9},
        tau_lookup=kv.tau_text,
        token_counter=HeuristicTokenCounter(),
        counter_identity=_STORED_COUNTER,
        config=CoeConfig(verify_on_match=True),
        trace_sink=traces.append,
    )
    assert set(result.selected) == {"h_direct", "h_bridge"}

    first = assemble_chains(result, signals, tau_lookup=kv.tau_text, trace_sink=traces.append)
    second = assemble_chains(result, signals, tau_lookup=kv.tau_text)
    assert len(first.chains) == 1, "the 2-hop evidence is ONE chain, not two fragments"
    chain = first.chains[0]
    assert not chain.peripheral and chain.anchor_node_ids == ["n_seed"]
    assert [s.hyperedge_id for s in chain.steps] == ["h_direct", "h_bridge"], (
        "the bridge reads AFTER the direct evidence (BFS hop order from the anchor)"
    )
    assert [s.hop for s in chain.steps] == [0, 1]
    assert first.text.encode() == second.text.encode()
    assert "## Chain 1 (anchor: n_seed)" in first.text
    assert "text h_direct" in first.text and "text h_bridge" in first.text
    assert [t["stage"] for t in traces] == ["select", "chains"]
