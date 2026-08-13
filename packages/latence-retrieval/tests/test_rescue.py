# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Graph-as-rescue retrieval (:class:`GraphRescueRetrieval`), tested at the boundary.

Every assertion is on the *observable output* of the composition — the base order surviving the
graph lane, the append-dedupe invariant, the gate's per-query decision, the selector's budget, and
loud config validation — never on the internal maths, so the tests survive a refactor of *how* any
step is computed. The engines are the shipped faithful fakes
(:class:`~latence_retrieval.testing.FakeBackend`) and the hypergraph fixtures are byte-for-byte
ADR-0056 stores, so everything flows through the shipped contracts and the whole suite is
pure-Python and offline.
"""

from __future__ import annotations

import json
import struct
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from latence_retrieval import (
    Candidate,
    Capability,
    GraphRescueRetrieval,
    HyperedgeDocumentSource,
    QueryBundle,
    RescueResult,
)
from latence_retrieval.pack import HeuristicTokenCounter
from latence_retrieval.testing import FakeBackend

# -- fixtures -----------------------------------------------------------------------------------

_DIM = 2
_VEC = (1.0, 0.0)


def _cand(cid: str, score: float, text: str | None = None, **meta: Any) -> Candidate:
    return Candidate(id=cid, score=score, text=text if text is not None else f"text {cid}",
                     metadata=dict(meta))


def _dense(candidates: list[Candidate]) -> FakeBackend:
    return FakeBackend(candidates, {Capability.DENSE}, dimension=_DIM)


def _bm25(candidates: list[Candidate]) -> FakeBackend:
    return FakeBackend(candidates, {Capability.BM25})


def _graph(candidates: list[Candidate]) -> FakeBackend:
    return FakeBackend(candidates, {Capability.GRAPH})


def _bundle(text: str, seeds: tuple[str, ...] = ()) -> QueryBundle:
    return QueryBundle(text=text, vector=_VEC, seed_node_ids=seeds or None)


def _ids(result: RescueResult) -> list[str]:
    return [c.id for c in result.candidates]


# A minimal, independent ADR-0056 store writer (fixture-side, mirroring test_hypergraph).

_MAGIC = b"LHKV1\n"
_U64 = struct.Struct("<Q")


def _norm(surface: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", surface).casefold().split())


def _build_kv(hyperedges: dict[str, dict[str, Any]], aliases: dict[str, list[str]]) -> bytes:
    entries: dict[bytes, bytes] = {}
    postings: dict[str, list[tuple[float, str]]] = {}
    for hid, h in hyperedges.items():
        solver = {
            "entity_node_ids": h["nodes"],
            "gamma": h["gamma"],
            "has_relations": h.get("has_relations", True),
            "support": h.get("support", 1),
            "token_cost": h.get("token_cost", 50),
            "token_counter": "latence:estimate_tokens:v1",
        }
        entries[b"e|" + hid.encode()] = json.dumps(
            solver, sort_keys=True, separators=(",", ":")
        ).encode()
        entries[b"t|" + hid.encode()] = h.get("tau", f"text {hid}").encode()
        for v in h["nodes"]:
            postings.setdefault(v, []).append((h["gamma"], hid))
    for v, items in postings.items():
        items.sort(key=lambda t: (-t[0], t[1]))
        entries[b"p|" + v.encode()] = json.dumps(
            [[hid for _, hid in items], [g for g, _ in items]], separators=(",", ":")
        ).encode()
    for surface, node_ids in aliases.items():
        entries[b"a|" + _norm(surface).encode()] = json.dumps(
            sorted(node_ids), separators=(",", ":")
        ).encode()
    sorted_entries = sorted(entries.items())
    keys = [k for k, _ in sorted_entries]
    values = [val for _, val in sorted_entries]
    key_offs, val_offs = [0], [0]
    for k in keys:
        key_offs.append(key_offs[-1] + len(k))
    for val in values:
        val_offs.append(val_offs[-1] + len(val))
    out = bytearray()
    out += _MAGIC
    out += _U64.pack(len(sorted_entries))
    for off in key_offs:
        out += _U64.pack(off)
    for off in val_offs:
        out += _U64.pack(off)
    out += b"".join(keys)
    out += b"".join(values)
    return bytes(out)


def _write_kv(tmp_path: Path) -> str:
    """A seed→mid→answer hypergraph whose Hyperedges resolve to the r* rescue chunks."""
    hyperedges = {
        "h_direct": {"nodes": ["n_seed", "n_mid"], "gamma": 0.80},
        "h_bridge": {"nodes": ["n_mid", "n_answer"], "gamma": 0.80},
    }
    path = tmp_path / "graph-hyperedges.kv"
    path.write_bytes(_build_kv(hyperedges, {"Seed Corp": ["n_seed"]}))
    return str(path)


class _HyperedgeDocs:
    """A faithful in-memory HyperedgeDocumentSource: hyperedge id → chunk candidates."""

    def __init__(self, mapping: dict[str, list[Candidate]]) -> None:
        self._mapping = mapping

    def candidates(self, hyperedge_ids: Sequence[str]) -> list[Candidate]:
        out: list[Candidate] = []
        seen: set[str] = set()
        for hid in hyperedge_ids:
            for candidate in self._mapping.get(hid, []):
                if candidate.id not in seen:
                    seen.add(candidate.id)
                    out.append(candidate)
        return out


# -- construction-time validation ---------------------------------------------------------------


class TestConfigValidation:
    def test_an_unknown_config_key_is_loud(self) -> None:
        with pytest.raises(ValueError, match="budgett"):
            GraphRescueRetrieval({"budgett": 100}, dense=_dense([]))

    def test_an_unknown_graph_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="graph_mode"):
            GraphRescueRetrieval({"graph_mode": "always"}, dense=_dense([]))

    def test_an_unknown_selector_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="selector"):
            GraphRescueRetrieval({"selector": "knapsackk"}, dense=_dense([]))

    def test_no_base_first_stage_is_a_contradiction(self) -> None:
        with pytest.raises(ValueError, match="base"):
            GraphRescueRetrieval({})

    def test_a_non_backend_engine_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="RetrievalBackend"):
            GraphRescueRetrieval({}, dense=object())  # type: ignore[arg-type]

    def test_half_a_hypergraph_lane_is_rejected_both_ways(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="hyperedge"):
            GraphRescueRetrieval(
                {"hyperedge_kv_path": _write_kv(tmp_path)}, dense=_dense([])
            )
        with pytest.raises(ValueError, match="hyperedge"):
            GraphRescueRetrieval({}, dense=_dense([]), hyperedge_documents=_HyperedgeDocs({}))

    def test_coe_without_the_hypergraph_lane_is_a_contradiction(self) -> None:
        with pytest.raises(ValueError, match="coe"):
            GraphRescueRetrieval({"selector": "coe"}, dense=_dense([]))

    def test_a_bad_expander_knob_fails_at_construction_not_at_query_time(self) -> None:
        with pytest.raises(ValueError, match="hop_weight"):
            GraphRescueRetrieval({"hop_weight": -0.1}, dense=_dense([]))

    def test_a_bad_budget_fails_at_construction(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            GraphRescueRetrieval({"budget": 0}, dense=_dense([]))

    def test_a_gate_threshold_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="low_agreement"):
            GraphRescueRetrieval({"low_agreement": 1.5}, dense=_dense([]))

    def test_the_measured_defaults_construct_cleanly(self) -> None:
        GraphRescueRetrieval({}, dense=_dense([]), bm25=_bm25([]))


# -- the policy gate ----------------------------------------------------------------------------


def _gate_pipeline(
    dense: list[Candidate], bm25: list[Candidate], **cfg: Any
) -> GraphRescueRetrieval:
    return GraphRescueRetrieval(
        {"selector": "none", **cfg}, dense=_dense(dense), bm25=_bm25(bm25)
    )


_CONFIDENT = [_cand(f"d{i}", 0.9 - i * 0.05) for i in range(10)]


class TestGateDecisions:
    def test_a_relation_cue_fires_the_gate(self) -> None:
        pipeline = _gate_pipeline(_CONFIDENT, _CONFIDENT)
        result = pipeline.retrieve(_bundle("why did the merger between the companies collapse"))
        assert result.gate.fired
        assert "relation-cue" in result.gate.reasons

    def test_two_query_entities_fire_the_entity_heavy_trigger(self) -> None:
        pipeline = _gate_pipeline(_CONFIDENT, _CONFIDENT)
        result = pipeline.retrieve(
            _bundle("what does the annual filing say about the supplier contract terms",
                    seeds=("n1", "n2"))
        )
        assert result.gate.fired
        assert "entity-heavy" in result.gate.reasons
        assert result.gate.query_seeds == 2

    def test_disagreeing_first_stages_fire_the_low_agreement_trigger(self) -> None:
        other = [_cand(f"b{i}", 0.9 - i * 0.05) for i in range(10)]
        pipeline = _gate_pipeline(_CONFIDENT, other)
        result = pipeline.retrieve(
            _bundle("what does the annual filing say about the supplier contract terms")
        )
        assert result.gate.fired
        assert "low-agreement" in result.gate.reasons
        assert result.gate.agreement == 0.0

    def test_an_unconfident_first_stage_fires_the_low_confidence_trigger(self) -> None:
        weak = [_cand(f"d{i}", 0.3 - i * 0.01) for i in range(10)]
        pipeline = _gate_pipeline(weak, weak)
        result = pipeline.retrieve(
            _bundle("what does the annual filing say about the supplier contract terms")
        )
        assert result.gate.fired
        assert "low-confidence" in result.gate.reasons

    def test_a_short_agreeing_cue_free_query_is_vetoed_as_navigational(self) -> None:
        weak = [_cand(f"d{i}", 0.3 - i * 0.01) for i in range(10)]
        pipeline = _gate_pipeline(weak, weak)
        result = pipeline.retrieve(_bundle("acme revenue 2024"))
        assert not result.gate.fired
        assert result.gate.vetoed_by == "navigational"
        assert "low-confidence" in result.gate.reasons  # the veto kept the fired trigger visible

    def test_no_trigger_means_no_graph_lane(self) -> None:
        pipeline = _gate_pipeline(_CONFIDENT, _CONFIDENT)
        result = pipeline.retrieve(
            _bundle("what does the annual filing say about the supplier contract terms")
        )
        assert not result.gate.fired
        assert result.gate.reasons == ()
        assert result.gate.vetoed_by is None

    def test_mode_off_vetoes_but_still_records_the_policy_opinion(self) -> None:
        pipeline = _gate_pipeline(_CONFIDENT, _CONFIDENT, graph_mode="off")
        result = pipeline.retrieve(_bundle("why did the merger between the companies collapse"))
        assert not result.gate.fired
        assert result.gate.vetoed_by == "mode-off"
        assert "relation-cue" in result.gate.reasons

    def test_mode_force_fires_and_names_itself_when_the_policy_would_not_have(self) -> None:
        pipeline = _gate_pipeline(_CONFIDENT, _CONFIDENT, graph_mode="force")
        result = pipeline.retrieve(
            _bundle("what does the annual filing say about the supplier contract terms")
        )
        assert result.gate.fired
        assert result.gate.reasons == ("mode-force",)


# -- the append-dedupe invariant ----------------------------------------------------------------


_BASE = [_cand("d1", 1.0), _cand("d2", 0.9), _cand("d3", 0.8)]
_RESCUE = [_cand("d2", 0.99), _cand("r1", 0.7), _cand("r2", 0.6)]


def _forced(**cfg: Any) -> GraphRescueRetrieval:
    return GraphRescueRetrieval(
        {"graph_mode": "force", "selector": "none", **cfg},
        dense=_dense(_BASE),
        bm25=_bm25(_BASE),
        graph=_graph(_RESCUE),
    )


class TestAppendAndDedupe:
    def test_rescue_appends_below_the_base_and_never_reorders_it(self) -> None:
        result = _forced().retrieve(_bundle("q", seeds=("n_seed",)))
        assert _ids(result) == ["d1", "d2", "d3", "r1", "r2"]
        assert result.rescued == 2

    def test_a_rescue_candidate_the_base_already_holds_is_dropped_not_promoted(self) -> None:
        # d2 arrives from the graph lane with a HIGHER score than its base standing; it must
        # keep its base position and its base identity (rescued=False), never appear twice.
        result = _forced().retrieve(_bundle("q", seeds=("n_seed",)))
        assert _ids(result).count("d2") == 1
        d2 = next(c for c in result.candidates if c.id == "d2")
        assert d2.metadata["rescued"] is False

    def test_every_candidate_is_stamped_with_its_provenance(self) -> None:
        result = _forced().retrieve(_bundle("q", seeds=("n_seed",)))
        flags = {c.id: c.metadata["rescued"] for c in result.candidates}
        assert flags == {"d1": False, "d2": False, "d3": False, "r1": True, "r2": True}

    def test_union_scores_encode_rank_in_the_fusers_own_decay_shape(self) -> None:
        result = _forced().retrieve(_bundle("q", seeds=("n_seed",)))
        scores = [c.score for c in result.candidates]
        assert scores == [pytest.approx(1.0 / (60 + rank)) for rank in range(len(scores))]

    def test_the_rescue_cap_bounds_the_appendix_not_the_base(self) -> None:
        result = _forced(rescue_cap=1).retrieve(_bundle("q", seeds=("n_seed",)))
        assert _ids(result) == ["d1", "d2", "d3", "r1"]
        assert result.rescued == 1

    def test_an_unfired_gate_appends_nothing(self) -> None:
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "off", "selector": "none"},
            dense=_dense(_BASE),
            bm25=_bm25(_BASE),
            graph=_graph(_RESCUE),
        )
        result = pipeline.retrieve(_bundle("q", seeds=("n_seed",)))
        assert _ids(result) == ["d1", "d2", "d3"]
        assert result.rescued == 0

    def test_identical_bundles_yield_identical_results(self) -> None:
        pipeline = _forced()
        bundle = _bundle("q", seeds=("n_seed",))
        assert pipeline.retrieve(bundle) == pipeline.retrieve(bundle)


# -- the rescore step (Expander over the base's own KG columns) ---------------------------------


class TestRescore:
    def test_a_graph_corroborated_base_candidate_rises_a_few_ranks_but_never_vaults_the_head(
        self,
    ) -> None:
        # d10 (last of ten) co-mentions the top seed's entity; nobody else carries any KG
        # column. At the MEASURED default hop weight (0.05, in units of the max-normalised
        # base score) the corroborated candidate rises a few ranks — and the head stays the
        # head: the rescore is a bounded lift, not a re-ranking.
        base = [
            _cand("d1", 1.0, kg_node_ids=["n1"], neighbor_node_ids=[]),
            *[_cand(f"d{i}", 1.0 - i * 0.05) for i in range(2, 10)],
            _cand("d10", 0.5, kg_node_ids=["n1"], neighbor_node_ids=[]),
        ]
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "force", "selector": "none"},
            dense=_dense(base),
            bm25=_bm25(base),
        )
        result = pipeline.retrieve(_bundle("q"))
        ids = _ids(result)
        assert ids[0] == "d1"
        assert 0 < ids.index("d10") < 9  # lifted off the tail, nowhere near the head

    def test_without_kg_columns_the_rescore_is_an_order_preserving_no_op(self) -> None:
        result = _forced().retrieve(_bundle("q", seeds=("n_seed",)))
        assert _ids(result)[:3] == ["d1", "d2", "d3"]


# -- the hypergraph rescue lane (real ADR-0056 store through the shipped reader) ----------------


class TestHypergraphLane:
    def test_hyperedge_chunks_the_base_missed_are_appended(self, tmp_path: Path) -> None:
        docs = _HyperedgeDocs(
            {"h_direct": [_cand("r_direct", 0.0)], "h_bridge": [_cand("r_bridge", 0.0)]}
        )
        pipeline = GraphRescueRetrieval(
            {
                "graph_mode": "force",
                "selector": "none",
                "hyperedge_kv_path": _write_kv(tmp_path),
            },
            dense=_dense(_BASE),
            bm25=_bm25(_BASE),
            hyperedge_documents=docs,
        )
        result = pipeline.retrieve(_bundle("q", seeds=("n_seed",)))
        assert _ids(result)[:3] == ["d1", "d2", "d3"]
        appended = {c.id for c in result.candidates if c.metadata["rescued"]}
        assert appended == {"r_direct", "r_bridge"}

    def test_the_seam_protocol_accepts_the_faithful_fake(self) -> None:
        assert isinstance(_HyperedgeDocs({}), HyperedgeDocumentSource)


# -- selection: budgets are respected -----------------------------------------------------------


_COUNTER = HeuristicTokenCounter()


class TestSelection:
    def test_selector_none_reports_no_budget(self) -> None:
        result = _forced().retrieve(_bundle("q", seeds=("n_seed",)))
        assert result.tokens_packed is None
        assert result.budget is None

    def test_the_packed_selector_respects_its_budget(self) -> None:
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "force", "selector": "packed", "budget": 5},
            dense=_dense(_BASE),
            bm25=_bm25(_BASE),
            graph=_graph(_RESCUE),
        )
        result = pipeline.retrieve(_bundle("q", seeds=("n_seed",)))
        assert result.budget == 5
        assert result.tokens_packed is not None
        assert result.tokens_packed <= 5
        assert sum(_COUNTER.count(c.text) for c in result.candidates) <= 5

    def test_the_packed_selector_defaults_to_the_measured_operating_point(self) -> None:
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "force"},
            dense=_dense(_BASE),
            bm25=_bm25(_BASE),
            graph=_graph(_RESCUE),
        )
        result = pipeline.retrieve(_bundle("q", seeds=("n_seed",)))
        assert result.budget == 10_000
        # Everything fits at 10k, and packing never invents candidates.
        assert set(_ids(result)) == {"d1", "d2", "d3", "r1", "r2"}

    def test_the_coe_selector_respects_its_budget_and_defaults_to_16k(
        self, tmp_path: Path
    ) -> None:
        docs = _HyperedgeDocs(
            {"h_direct": [_cand("r_direct", 0.0)], "h_bridge": [_cand("r_bridge", 0.0)]}
        )
        pipeline = GraphRescueRetrieval(
            {
                "graph_mode": "force",
                "selector": "coe",
                "hyperedge_kv_path": _write_kv(tmp_path),
            },
            dense=_dense(_BASE),
            bm25=_bm25(_BASE),
            hyperedge_documents=docs,
        )
        result = pipeline.retrieve(_bundle("q", seeds=("n_seed",)))
        assert result.budget == 16_000
        assert result.tokens_packed is not None
        assert result.tokens_packed <= 16_000
        assert _ids(result)[:3] == ["d1", "d2", "d3"]  # base rank order survives selection

    def test_the_coe_selector_excludes_rescue_candidates_coe_did_not_buy(
        self, tmp_path: Path
    ) -> None:
        # Budget 20 cannot afford any stored hyperedge (token_cost 50 each), so CoE buys no
        # graph evidence — the rescue appendix must be excluded entirely, and the base fills
        # the budget in its own rank order.
        docs = _HyperedgeDocs(
            {"h_direct": [_cand("r_direct", 0.0)], "h_bridge": [_cand("r_bridge", 0.0)]}
        )
        pipeline = GraphRescueRetrieval(
            {
                "graph_mode": "force",
                "selector": "coe",
                "budget": 20,
                "hyperedge_kv_path": _write_kv(tmp_path),
            },
            dense=_dense(_BASE),
            bm25=_bm25(_BASE),
            hyperedge_documents=docs,
        )
        result = pipeline.retrieve(_bundle("q", seeds=("n_seed",)))
        assert not any(c.metadata["rescued"] for c in result.candidates)
        assert result.tokens_packed is not None
        assert result.tokens_packed <= 20
