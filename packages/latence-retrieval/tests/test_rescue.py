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
    CandidateDecorator,
    Capability,
    GraphRescueRetrieval,
    HyperedgeDocumentSource,
    KgColumnDecorator,
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


# -- decoration: the opt-in CandidateDecorator seam ---------------------------------------------


class _RecordingGraph:
    """A GRAPH backend spy: records every seed set it is asked to traverse, returns nothing."""

    def __init__(self) -> None:
        self.seed_sets: list[tuple[str, ...]] = []

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.GRAPH})

    def search(self, query: Any) -> list[Candidate]:
        self.seed_sets.append(tuple(query.seed_node_ids or ()))
        return []


def _decoration_fixture() -> tuple[list[Candidate], list[Candidate], list[Candidate]]:
    """The pinned pre-seam fixture: a dual-retrieved base, a bm25-only candidate, one rescue."""
    base = [
        _cand("d1", 1.0, kg_node_ids=["n1"], neighbor_node_ids=["n2"], kg_node_community=["c1"],
              kg_node_centrality=[0.9, 0.2]),
        _cand("d2", 0.9, kg_node_ids=["n3"], neighbor_node_ids=["n1"]),
        _cand("d3", 0.8),
        _cand("d4", 0.7, kg_node_ids=["n2"], neighbor_node_ids=[], kg_node_community=["c1"]),
    ]
    bm25 = [_cand("d2", 3.0), _cand("b1", 2.0), _cand("d4", 1.0)]
    rescue = [_cand("r1", 0.7, kg_node_ids=["n9"], neighbor_node_ids=[]), _cand("d3", 0.6)]
    return base, bm25, rescue


class TestDecoration:
    def test_decorate_none_is_byte_identical_to_the_pre_seam_pipeline(self) -> None:
        # The FULL RescueResult (ids, scores, metadata, gate, spend) of this fixture, captured
        # verbatim on the pre-seam code (commit 85b8e59, PYTHONHASHSEED=0). `decorate=None` must
        # reproduce it exactly — the seam's default is the identity, not "roughly the same".
        base, bm25, rescue = _decoration_fixture()
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "force", "selector": "packed", "budget": 40},
            dense=_dense(base),
            bm25=_bm25(bm25),
            graph=_graph(rescue),
        )
        result = pipeline.retrieve(
            QueryBundle(
                text="why did d1 relate to d4", vector=_VEC, seed_node_ids=("n1", "n2")
            )
        )
        expected = {
            "budget": 40,
            "candidates": [
                {"id": "d2", "score": 0.016666666666666666, "text": "text d2",
                 "metadata": {"kg_node_ids": ["n3"], "neighbor_node_ids": ["n1"],
                              "rescued": False}},
                {"id": "d4", "score": 0.01639344262295082, "text": "text d4",
                 "metadata": {"kg_node_community": ["c1"], "kg_node_ids": ["n2"],
                              "neighbor_node_ids": [], "rescued": False}},
                {"id": "d1", "score": 0.016129032258064516, "text": "text d1",
                 "metadata": {"kg_node_centrality": [0.9, 0.2], "kg_node_community": ["c1"],
                              "kg_node_ids": ["n1"], "neighbor_node_ids": ["n2"],
                              "rescued": False}},
                {"id": "b1", "score": 0.015873015873015872, "text": "text b1",
                 "metadata": {"rescued": False}},
                {"id": "d3", "score": 0.015625, "text": "text d3",
                 "metadata": {"rescued": False}},
                {"id": "r1", "score": 0.015384615384615385, "text": "text r1",
                 "metadata": {"kg_node_ids": ["n9"], "neighbor_node_ids": [], "rescued": True}},
            ],
            "gate": {"agreement": 0.4, "confidence": 1.0, "fired": True, "query_seeds": 2,
                     "reasons": ["relation-cue", "entity-heavy"], "vetoed_by": None},
            "rescued": 1,
            "tokens_packed": 12,
        }
        assert json.loads(result.model_dump_json()) == expected

    def test_the_seam_protocol_accepts_the_reference_adapter(self) -> None:
        assert isinstance(KgColumnDecorator({}), CandidateDecorator)

    def test_a_non_decorator_is_rejected_at_construction(self) -> None:
        with pytest.raises(TypeError, match="CandidateDecorator"):
            GraphRescueRetrieval({}, dense=_dense([]), decorate=object())  # type: ignore[arg-type]

    def test_a_bm25_only_candidate_produces_graph_seeds_only_when_decorated(self) -> None:
        # THE divergence the seam exists to close: RrfFuser keeps first-seen metadata and the
        # base fusion is (dense, bm25), so bm25-only b1 reaches the seed harvest with no
        # kg_node_ids — the traversal is seeded without its entities. Decorated, they arrive.
        dense = [_cand("d1", 1.0, kg_node_ids=["n_d1"], neighbor_node_ids=[])]
        bm25 = [_cand("b1", 2.0)]
        columns = {"b1": {"kg_node_ids": ["n_b1"], "neighbor_node_ids": []}}

        def seeds_reaching_traversal(decorate: CandidateDecorator | None) -> tuple[str, ...]:
            spy = _RecordingGraph()
            pipeline = GraphRescueRetrieval(
                {"graph_mode": "force", "selector": "none"},
                dense=_dense(dense),
                bm25=_bm25(bm25),
                graph=spy,  # type: ignore[arg-type]
                decorate=decorate,
            )
            pipeline.retrieve(_bundle("q", seeds=("n_q",)))
            assert len(spy.seed_sets) == 1
            return spy.seed_sets[0]

        undecorated = seeds_reaching_traversal(None)
        assert "n_b1" not in undecorated  # the shipped default: b1's entities are invisible
        decorated = seeds_reaching_traversal(KgColumnDecorator(columns))
        assert "n_b1" in decorated
        assert decorated[0] == "n_q"  # query seeds still come first, never displaced

    def test_the_decorated_union_carries_columns_on_the_rescue_appendix_too(self) -> None:
        base, bm25, rescue = _decoration_fixture()
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "force", "selector": "none"},
            dense=_dense(base),
            bm25=_bm25(bm25),
            graph=_graph(rescue),
            decorate=KgColumnDecorator(
                {"b1": {"kg_node_ids": ["n_b1"]}, "r1": {"kg_node_community": ["c9"]}}
            ),
        )
        result = pipeline.retrieve(_bundle("why does d1 relate to d4", seeds=("n1", "n2")))
        by_id = {c.id: c for c in result.candidates}
        assert by_id["b1"].metadata["kg_node_ids"] == ["n_b1"]
        assert by_id["r1"].metadata["kg_node_community"] == ["c9"]
        assert by_id["r1"].metadata["rescued"] is True  # decoration preserved the stamp

    def test_a_supplied_redundancy_map_short_circuits_the_packers_lexical_jaccard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from latence_retrieval import pack as pack_module

        calls = {"n": 0}
        real = pack_module._lexical_similarity  # noqa: SLF001 - the spy target

        def counting(a: str, b: str) -> float:
            calls["n"] += 1
            return real(a, b)

        monkeypatch.setattr(pack_module, "_lexical_similarity", counting)

        def run(decorate: CandidateDecorator | None) -> None:
            base, bm25, rescue = _decoration_fixture()
            pipeline = GraphRescueRetrieval(
                {"graph_mode": "force", "selector": "packed"},
                dense=_dense(base),
                bm25=_bm25(bm25),
                graph=_graph(rescue),
                decorate=decorate,
            )
            pipeline.retrieve(_bundle("why does d1 relate to d4", seeds=("n1", "n2")))

        run(None)
        assert calls["n"] > 0  # the shipped default recomputes redundancy inside the solver
        calls["n"] = 0
        run(KgColumnDecorator({}, redundancy_key="redundancy"))
        assert calls["n"] == 0  # every pair was precomputed; the packer never re-tokenises

    def test_the_adapters_redundancy_is_exactly_the_packers_own_lexical_similarity(self) -> None:
        from latence_retrieval.pack import _lexical_similarity  # noqa: SLF001 - the oracle

        candidates = [
            _cand("a", 1.0, text="the quick brown fox"),
            _cand("b", 0.9, text="the quick red panda"),
            _cand("c", 0.8, text="völlig anderes Thema — Umsätze"),
        ]
        decorated = KgColumnDecorator({}, redundancy_key="redundancy").decorate(candidates)
        for left in decorated:
            for right in decorated:
                if left.id == right.id:
                    continue
                stored = left.metadata["redundancy"][right.id]
                assert stored == pytest.approx(
                    _lexical_similarity(left.text, right.text), abs=0.0
                )

    def test_the_adapters_row_overwrites_stale_engine_columns_but_not_other_keys(self) -> None:
        candidate = _cand("a", 1.0, kg_node_ids=["stale"], rescued=True)
        decorated = KgColumnDecorator({"a": {"kg_node_ids": ["fresh"]}}).decorate([candidate])
        assert decorated[0].metadata["kg_node_ids"] == ["fresh"]
        assert decorated[0].metadata["rescued"] is True

    def test_a_decorator_that_reranks_is_loud_not_silent(self) -> None:
        class _Reorderer:
            def decorate(self, candidates: Sequence[Candidate]) -> list[Candidate]:
                return list(reversed(candidates))

        base, bm25, rescue = _decoration_fixture()
        pipeline = GraphRescueRetrieval(
            {"graph_mode": "force", "selector": "none"},
            dense=_dense(base),
            bm25=_bm25(bm25),
            graph=_graph(rescue),
            decorate=_Reorderer(),
        )
        with pytest.raises(ValueError, match="only enrich metadata"):
            pipeline.retrieve(_bundle("why does d1 relate to d4", seeds=("n1", "n2")))
