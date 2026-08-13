# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""M4 (ticket 15): the chain-of-evidence selector over M3's HypergraphSignals.

The spec-M4 acceptance criteria, plus the ADR-0058 verification contract: the budget is never
exceeded — INCLUDING under a verification counter that counts higher than the stored index-time
weights (the under-count → trim case); marginal-aware greedy beats a plain value-density greedy
on chain value; theta_min stops filler with budget left; a near-duplicate is redundancy-discounted
and loses to an equally-valued diverse candidate; chain-adjacency wins over isolation; the weak
discount and the absent-signal weight redistribution hold; selection is deterministic; the
hard-redundancy case (identical relation signature selected twice) is UNREPRESENTABLE by
construction (M1's content-addressed dedup, ADR-0057); and the end-to-end selection-level ranking
test M3 explicitly deferred here — the planted 2-hop chain beats the lexical distractor in
SELECTION, not just in ρ.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from latence_retrieval.coe import (
    CoeConfig,
    EdgeFeatures,
    HeuristicUtility,
    SelectionState,
    select,
)
from latence_retrieval.hypergraph import HyperedgeKV, HypergraphSignals, hypergraph_signals
from latence_retrieval.pack import HeuristicTokenCounter
from test_hypergraph import _build_kv

_STORED_COUNTER = "latence:estimate_tokens:v1"  # what the _build_kv fixture records per edge


def _signals(
    edges: dict[str, dict[str, Any]], anchors: dict[str, float]
) -> HypergraphSignals:
    """Build a HypergraphSignals directly (the M3 output contract) for solver unit tests."""
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


# -- spec M4: the budget is never exceeded ----------------------------------------


def test_budget_is_never_exceeded_on_stored_costs() -> None:
    edges = {
        f"e{i:02d}": {"rho": 1.0 - i * 0.05, "nodes": [f"n{i}", f"m{i}"], "cost": 30 + i * 7}
        for i in range(10)
    }
    signals = _signals(edges, {"n0": 1.0})
    result = select(signals, budget=120)
    total = sum(int(signals.fields[h]["token_cost"]) for h in result.selected)
    assert result.selected, "a feasible budget must select something"
    assert total <= 120 and result.total_cost == total <= result.budget
    assert not result.verified and result.trimmed == []


def test_verification_trims_when_the_index_counter_under_counted() -> None:
    """ADR-0058's core case: stored weights say the set fits, the generation tokenizer says it
    does not — the pass re-counts the SELECTED set exactly and trims lowest-marginal-first, so
    the emitted set provably fits. Mandatory here because the identities differ; no scale
    factor is ever applied."""
    ten_words = "one two three four five six seven eight nine ten"  # 10 heuristic tokens
    edges = {
        "e_strong": {"rho": 1.0, "nodes": ["n1", "n2"], "cost": 5},
        "e_weakling": {"rho": 0.3, "nodes": ["m1", "m2"], "cost": 5},
    }
    signals = _signals(edges, {"n1": 1.0})
    taus = {"e_strong": ten_words, "e_weakling": ten_words}
    result = select(
        signals,
        budget=12,
        tau_lookup=taus.get,
        token_counter=HeuristicTokenCounter(),
        counter_identity="tiktoken:cl100k_base",  # differs from stored → pass is MANDATORY
        config=CoeConfig(verify_on_match=False),
    )
    assert result.verified
    assert result.selected == ["e_strong"] and result.trimmed == ["e_weakling"]
    assert result.total_cost == 10 <= result.budget  # the EXACT re-counted total fits
    # both were accepted by the greedy loop before the trim — the trace keeps their marginals
    assert set(result.marginals) == {"e_strong", "e_weakling"}


def test_verification_is_mandatory_on_mismatch_and_optional_on_match() -> None:
    edges = {"e_a": {"rho": 1.0, "nodes": ["n1", "n2"], "cost": 5}}
    signals = _signals(edges, {"n1": 1.0})
    taus = {"e_a": "three token text"}

    def run(identity: str, verify_on_match: bool) -> bool:
        return select(
            signals,
            budget=50,
            tau_lookup=taus.get,
            token_counter=HeuristicTokenCounter(),
            counter_identity=identity,
            config=CoeConfig(verify_on_match=verify_on_match),
        ).verified

    assert run("tiktoken:cl100k_base", False), "identity mismatch must force the pass"
    assert not run(_STORED_COUNTER, False), "matching identities leave the pass optional"
    assert run(_STORED_COUNTER, True), "config may opt the matching case in"


def test_partial_verification_triple_is_a_loud_config_error() -> None:
    signals = _signals({"e_a": {"rho": 1.0, "nodes": ["n1"]}}, {"n1": 1.0})
    with pytest.raises(ValueError, match="counter_identity"):
        select(signals, budget=10, token_counter=HeuristicTokenCounter())


def test_budget_must_be_a_positive_integer() -> None:
    signals = _signals({"e_a": {"rho": 1.0, "nodes": ["n1"]}}, {"n1": 1.0})
    with pytest.raises(ValueError, match="positive integer"):
        select(signals, budget=0)


# -- spec M4: marginal-aware greedy beats plain value-density greedy --------------


def test_greedy_beats_plain_value_density_baseline_on_chain_value() -> None:
    """Three same-entity-set near-duplicates (distinct relation signatures — the representable
    kind of near-duplicate, ADR-0057) carry the top v0: a plain value-density greedy takes all
    three; the marginal loop takes one plus two diverse candidates. Compared on chain value
    Σ_i v(e_i | {e_1..e_{i-1}}), the marginal selection must win — knapsack_refine is dropped
    (see the module docstring), so THIS is the value guarantee the milestone pins."""
    dup_nodes = ["n_q", "d1", "d2"]
    edges: dict[str, dict[str, Any]] = {
        "dup_a": {"rho": 1.0, "nodes": dup_nodes, "gamma": 0.9},
        "dup_b": {"rho": 1.0, "nodes": dup_nodes, "gamma": 0.9},
        "dup_c": {"rho": 1.0, "nodes": dup_nodes, "gamma": 0.9},
        "div_a": {"rho": 0.7, "nodes": ["a1", "a2", "a3"], "gamma": 0.9},
        "div_b": {"rho": 0.7, "nodes": ["b1", "b2", "b3"], "gamma": 0.9},
    }
    anchors = {"n_q": 1.0}
    signals = _signals(edges, anchors)
    result = select(signals, budget=150)  # room for exactly three 50-cost items

    assert len([h for h in result.selected if h.startswith("dup")]) == 1
    assert {"div_a", "div_b"} <= set(result.selected)
    greedy_value = sum(result.marginals[h] for h in result.selected)

    # the plain baseline: highest v0 per token, redundancy-blind → the three duplicates
    utility = HeuristicUtility()

    def base_value(hid: str) -> float:
        nodes = edges[hid]["nodes"]
        return utility.base(
            EdgeFeatures(
                rho_norm=float(edges[hid]["rho"]) / 1.0,
                gamma=0.9,
                anchor_overlap=sum(anchors.get(n, 0.0) for n in nodes) / len(nodes),
                support_norm=1.0,  # all supports equal → log1p ratio is 1 for every edge
                sim=None,
                relation_bonus=None,
                has_relations=True,
            )
        )

    baseline = sorted(edges, key=lambda h: (-base_value(h) / 50, h))[:3]
    assert baseline == ["dup_a", "dup_b", "dup_c"]
    baseline_value = 0.0
    chosen: list[str] = []
    for hid in baseline:
        nodes = set(edges[hid]["nodes"])
        red = max(
            (len(nodes & set(edges[a]["nodes"])) / len(nodes) for a in chosen), default=0.0
        )
        conn = bool(nodes & set(anchors)) or any(
            nodes & set(edges[a]["nodes"]) for a in chosen
        )
        feats = EdgeFeatures(
            rho_norm=1.0,
            gamma=0.9,
            anchor_overlap=sum(anchors.get(n, 0.0) for n in nodes) / len(nodes),
            support_norm=1.0,
            sim=None,
            relation_bonus=None,
            has_relations=True,
        )
        baseline_value += utility.marginal(feats, SelectionState(redundancy=red, connected=conn))
        chosen.append(hid)
    assert greedy_value > baseline_value


# -- spec 3.3: theta_min stops filler with budget left ----------------------------


def test_theta_min_rejects_filler_even_with_budget_left() -> None:
    edges = {
        "e_strong": {"rho": 1.0, "nodes": ["n_q", "n2"], "gamma": 0.9, "support": 3},
        **{
            f"e_fill{i}": {"rho": 0.001, "nodes": [f"f{i}", f"g{i}"], "gamma": 0.05, "cost": 5}
            for i in range(5)
        },
    }
    signals = _signals(edges, {"n_q": 1.0})
    result = select(signals, budget=10_000)
    assert result.selected == ["e_strong"], (
        "sub-threshold filler must not be added just because budget remains"
    )


# -- spec 3.2: redundancy discount + connectivity bonus ---------------------------


def test_near_duplicate_is_discounted_and_loses_to_diverse_equal_v0() -> None:
    edges = {
        "e_first": {"rho": 1.0, "nodes": ["n_q", "n1", "n2", "n3"], "gamma": 0.9},
        # e_dup shares 3 of its 4 entities with e_first; e_div shares none. Same rho/gamma/
        # support and zero anchor overlap for both → identical v0; redundancy must decide.
        "e_dup": {"rho": 0.6, "nodes": ["n1", "n2", "n3", "x1"], "gamma": 0.8},
        "e_div": {"rho": 0.6, "nodes": ["c1", "c2", "c3", "c4"], "gamma": 0.8},
    }
    signals = _signals(edges, {"n_q": 1.0})
    result = select(signals, budget=100)  # two 50-cost items
    assert result.selected == ["e_first", "e_div"]
    assert "e_dup" not in result.selected


def test_connectivity_bonus_prefers_the_chain_adjacent_candidate() -> None:
    edges = {
        "e_first": {"rho": 1.0, "nodes": ["n_q", "n1"], "gamma": 0.9},
        # e_chain touches e_first through ONE of its five entities (mild redundancy, but
        # chain-adjacent); e_isolated has identical v0 and no connection at all.
        "e_chain": {"rho": 0.6, "nodes": ["n1", "b1", "b2", "b3", "b4"], "gamma": 0.8},
        "e_isolated": {"rho": 0.6, "nodes": ["c1", "c2", "c3", "c4", "c5"], "gamma": 0.8},
    }
    signals = _signals(edges, {"n_q": 1.0})
    result = select(signals, budget=100)
    assert result.selected == ["e_first", "e_chain"]
    assert "e_isolated" not in result.selected


# -- spec 3.2: weak discount + absent-signal weight redistribution ----------------


def test_weak_hyperedge_base_value_is_discounted() -> None:
    utility = HeuristicUtility()
    strong = EdgeFeatures(
        rho_norm=0.8,
        gamma=0.7,
        anchor_overlap=0.5,
        support_norm=1.0,
        sim=0.4,
        relation_bonus=0.2,
        has_relations=True,
    )
    weak = EdgeFeatures(
        rho_norm=0.8,
        gamma=0.7,
        anchor_overlap=0.5,
        support_norm=1.0,
        sim=0.4,
        relation_bonus=0.2,
        has_relations=False,
    )
    assert utility.base(weak) == pytest.approx(0.6 * utility.base(strong))


def test_absent_signals_redistribute_their_weight() -> None:
    """With every signal pinned to the same level x, the weighted mean is x no matter which
    optional signals are absent — the redistribution invariant. A deflating implementation
    (treating an absent signal as a zero score) would drop the value instead."""
    utility = HeuristicUtility()

    def features(sim: float | None, relation_bonus: float | None) -> EdgeFeatures:
        return EdgeFeatures(
            rho_norm=0.5,
            gamma=0.5,
            anchor_overlap=0.5,
            support_norm=0.5,
            sim=sim,
            relation_bonus=relation_bonus,
            has_relations=True,
        )

    full = utility.base(features(0.5, 0.5))
    assert full == pytest.approx(0.5)
    assert utility.base(features(None, 0.5)) == pytest.approx(0.5)
    assert utility.base(features(0.5, None)) == pytest.approx(0.5)
    assert utility.base(features(None, None)) == pytest.approx(0.5)


# -- determinism ------------------------------------------------------------------


def test_identical_inputs_yield_identical_selection_twice() -> None:
    edges = {
        "e_b": {"rho": 0.8, "nodes": ["n_q", "n1"], "gamma": 0.8},
        "e_a": {"rho": 0.8, "nodes": ["n_q", "n2"], "gamma": 0.8},  # exact tie with e_b
        "e_c": {"rho": 0.5, "nodes": ["n3", "n4"], "gamma": 0.8},
    }
    signals = _signals(edges, {"n_q": 1.0})
    first = select(signals, budget=100, eu_sims={"e_c": 0.3})
    second = select(signals, budget=100, eu_sims={"e_c": 0.3})
    assert first == second
    assert first.selected[0] == "e_a", "exact ties must break by ascending hyperedge id"


# -- the dedup invariant and its true limits (corrected by the close-out audit) ---


def test_dedup_invariant_makes_the_hard_redundancy_case_unrepresentable() -> None:
    """The IDENTICAL case only: M1 content-addresses the Hyperedge id over (corpus,
    entity-node set, sorted relation signature) — NOT the span position (ADR-0057,
    latence_core.hyperedge ``make_graph_id``) — so a duplicate extraction of the SAME
    (entity set, signature) pair produces the SAME id and merges into one record with
    ``support += 1``. This does NOT cover subset signatures or identical signatures over
    different entity sets — see the companion test below, which pins that those coexist
    and are handled by the entity-overlap discount (the declared spec-3.2 deviation)."""

    def content_id(nodes: list[str], signature: list[str]) -> str:
        payload = json.dumps([sorted(nodes), sorted(signature)], separators=(",", ":"))
        return "h_" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    nodes = ["n_a", "n_b"]
    signature = ["n_a:supplier_of:n_b"]
    first_extraction = content_id(nodes, signature)
    second_extraction = content_id(nodes, signature)  # a second span stating the same fact
    assert first_extraction == second_extraction, "same content ⇒ same id: no second record"

    # the two extractions therefore land on ONE record whose support was incremented on merge
    kv = HyperedgeKV.from_bytes(
        _build_kv(
            {first_extraction: {"nodes": nodes, "gamma": 0.8, "support": 2}},
            {"A Corp": ["n_a"]},
        )
    )
    signals = hypergraph_signals(kv, alias_surfaces=["A Corp"])
    assert list(signals.fields) == [first_extraction]
    assert signals.fields[first_extraction]["support"] == 2
    result = select(signals, budget=1000)
    assert result.selected == [first_extraction]  # one candidate — nothing to hard-skip


def test_subset_signature_records_coexist_and_get_the_overlap_discount() -> None:
    """The DECLARED spec-3.2 deviation, pinned (close-out audit, 2026-08-02): a record whose
    relation signature is a strict subset of another's has a DIFFERENT content id (the id is
    addressed over entity set + signature jointly), so both coexist in any store — the earlier
    'unrepresentable by construction' claim was false. The KV solver fields carry no
    signatures (ADR-0056), so the selector cannot hard-skip on the subset condition; the
    shipped handling is the entity-overlap discount: with all of the subset record's entities
    inside the selected record, red = 1.0 and its marginal collapses to exactly the
    connectivity bonus ``w_conn`` — which CAN still admit it (deliberate: full entity
    containment does not imply fact redundancy; a distinct predicate over the same pair must
    stay selectable, and skipping without data would violate the standing rule)."""

    def content_id(nodes: list[str], signature: list[str]) -> str:
        payload = json.dumps([sorted(nodes), sorted(signature)], separators=(",", ":"))
        return "h_" + hashlib.sha256(payload.encode()).hexdigest()[:12]

    big_id = content_id(["n_a", "n_b", "n_c"], ["n_a:acq:n_b", "n_b:appr:n_c"])
    sub_id = content_id(["n_a", "n_b"], ["n_a:acq:n_b"])
    assert big_id != sub_id, "subset signature ⇒ different content id ⇒ both records exist"

    edges = {
        big_id: {"nodes": ["n_a", "n_b", "n_c"], "rho": 0.9, "cost": 50},
        sub_id: {"nodes": ["n_a", "n_b"], "rho": 0.5, "cost": 50},
    }
    signals = _signals(edges, {"n_a": 1.0})
    cfg = CoeConfig()
    result = select(signals, budget=1000, config=cfg)
    assert result.selected[0] == big_id
    # red(sub | {big}) = |{a,b} ∩ {a,b,c}| / 2 = 1.0 → marginal = w_conn exactly; theta_min
    # = theta_ratio·max(v0) < w_conn here, so the subset record IS admitted — the pinned
    # residual of the declared deviation, visible rather than hidden.
    assert sub_id in result.selected


# -- end-to-end: the 2-hop chain beats the lexical distractor in SELECTION --------


def test_e2e_selection_picks_the_planted_chain_over_the_lexical_distractor() -> None:
    """The selection-level ranking test M3 explicitly deferred to M4 (see
    test_two_hop_evidence_gains_confidence_from_zero): ρ alone cannot outrank an
    anchor-touching distractor, the utility function must. The distractor is the classic
    lexical-noise shape — it mentions the queried entity, hogs ALL the lexical signal
    (sim 1.0) and carries the highest γ, but states no relation (a weak Hyperedge,
    ADR-0057); the planted chain carries relations, corpus support, and the realistic
    bridge-text signal ticket 20's addendum established (the question names the chain's
    endpoints). Under a 2-item budget the chain must win the seat the distractor contests."""
    hyperedges: dict[str, dict[str, Any]] = {
        "h_direct": {"nodes": ["n_seed", "n_mid"], "gamma": 0.80, "support": 3},
        "h_bridge": {"nodes": ["n_mid", "n_answer"], "gamma": 0.80, "support": 3},
        "h_distract": {
            "nodes": ["n_seed", "n_x"],
            "gamma": 0.95,
            "has_relations": False,
        },
        **{
            f"h_noise{i}": {"nodes": [f"n_a{i}", f"n_b{i}"], "gamma": 0.9} for i in range(20)
        },
    }
    kv = HyperedgeKV.from_bytes(_build_kv(hyperedges, {"Seed Corp": ["n_seed"]}))
    eu_sims = {"h_distract": 1.0, "h_direct": 0.6, "h_bridge": 0.7}
    signals = hypergraph_signals(kv, alias_surfaces=["Seed Corp"], eu_sims=eu_sims)
    assert {"h_direct", "h_bridge", "h_distract"} <= set(signals.rho)

    result = select(
        signals,
        budget=100,  # two 50-cost seats: the anchor evidence plus ONE contested seat
        eu_sims=eu_sims,
        relation_bonus={"h_direct": 0.9, "h_bridge": 0.9},  # the query asks for this relation
        tau_lookup=kv.tau_text,
        token_counter=HeuristicTokenCounter(),
        counter_identity=_STORED_COUNTER,
        config=CoeConfig(verify_on_match=True),
    )
    assert set(result.selected) == {"h_direct", "h_bridge"}, (
        "selection must assemble the planted 2-hop chain, not spend the seat on the "
        "lexical distractor"
    )
    assert "h_distract" not in result.selected
    assert result.verified and result.trimmed == []
    assert result.total_cost <= result.budget
