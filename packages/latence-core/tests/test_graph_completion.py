"""Unit tests for the W2-linkpred reference GraphCompleter (engine + Provider + contract).

Covers the pure :mod:`latence_core.graph_completion` engine (2-hop transitive/symmetric closure,
scoring, bounded fan-out, asserted-collision drop, determinism) and the
:class:`~latence_core.stages.graph_completion.ReferenceGraphCompleter` Provider that maps its
predictions onto the inferred-edge contract (``scope=EDGE``, ``properties["inferred"]=True``,
deterministic edge_id, honest model-derived Evidence with EMPTY mention_ids, collisions dropped).
"""

from __future__ import annotations

from latence_core.contracts import (
    Classification,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from latence_core.graph import make_graph_id
from latence_core.graph_completion import (
    CompletionConfig,
    GraphCompletionEngine,
    ObservedEdge,
)
from latence_core.stages.graph_completion import ReferenceGraphCompleter

# ---------------------------------------------------------------------------
# Engine: the pure 2-hop closure.
# ---------------------------------------------------------------------------


def _prov(doc: str = "doc-a") -> Provenance:
    return Provenance(
        source_uri="file://x",
        document_id=doc,
        file_name="x.txt",
        file_type="txt",
        file_size=1,
    )


def _class() -> Classification:
    return Classification(language="en", category="general", sensitivity="internal")


def test_engine_transitive_closure_proposes_the_two_hop_edge() -> None:
    """A partner_of B and B partner_of C ⇒ propose A partner_of C (the 2-hop transitive edge)."""
    observed = [
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("B", "partner_of", "C"),
    ]
    cfg = CompletionConfig(transitive_labels=frozenset({"partner_of"}))
    preds = GraphCompletionEngine(cfg).predict(observed)
    keys = {(p.source_node_id, p.label, p.target_node_id) for p in preds}
    assert ("A", "partner_of", "C") in keys
    ac = next(p for p in preds if p.target_node_id == "C" and p.source_node_id == "A")
    assert ac.rule == "transitive"
    assert ac.support == 1
    assert 0.0 <= ac.score <= 1.0


def test_engine_symmetric_proposes_the_reverse_edge() -> None:
    """A sibling_of B (symmetric) ⇒ propose B sibling_of A; the reverse when not already asserted."""
    observed = [ObservedEdge("A", "sibling_of", "B")]
    cfg = CompletionConfig(symmetric_labels=frozenset({"sibling_of"}))
    preds = GraphCompletionEngine(cfg).predict(observed)
    keys = {(p.source_node_id, p.label, p.target_node_id) for p in preds}
    assert ("B", "sibling_of", "A") in keys
    assert all(p.rule == "symmetric" for p in preds)


def test_engine_never_reproposes_an_asserted_edge() -> None:
    """If A--p-->C is ALREADY observed, the transitive closure does not re-propose it (collision)."""
    observed = [
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("B", "partner_of", "C"),
        ObservedEdge("A", "partner_of", "C"),  # already asserted
    ]
    cfg = CompletionConfig(transitive_labels=frozenset({"partner_of"}))
    preds = GraphCompletionEngine(cfg).predict(observed)
    keys = {(p.source_node_id, p.label, p.target_node_id) for p in preds}
    assert ("A", "partner_of", "C") not in keys


def test_engine_symmetric_skips_when_reverse_already_asserted() -> None:
    """A--p-->B and B--p-->A both observed ⇒ no symmetric proposal (both directions asserted)."""
    observed = [
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("B", "partner_of", "A"),
    ]
    cfg = CompletionConfig(symmetric_labels=frozenset({"partner_of"}), transitive_labels=frozenset())
    preds = GraphCompletionEngine(cfg).predict(observed)
    # The only symmetric candidates (B->A, A->B) are both already asserted.
    sym = [p for p in preds if p.rule == "symmetric"]
    assert sym == []


def test_engine_no_self_loops() -> None:
    """A cycle A->B->A never proposes the self-loop A->A."""
    observed = [
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("B", "partner_of", "A"),
    ]
    cfg = CompletionConfig(transitive_labels=frozenset({"partner_of"}))
    preds = GraphCompletionEngine(cfg).predict(observed)
    assert all(p.source_node_id != p.target_node_id for p in preds)


def test_engine_min_score_threshold_drops_weak_proposals() -> None:
    """A min_score above every proposal's normalized score drops them all."""
    observed = [
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("B", "partner_of", "C"),
    ]
    # support_norm=10 ⇒ the single-path proposal scores 0.1; min_score 0.5 drops it.
    cfg = CompletionConfig(
        transitive_labels=frozenset({"partner_of"}), support_norm=10.0, min_score=0.5
    )
    preds = GraphCompletionEngine(cfg).predict(observed)
    assert preds == []


def test_engine_score_reflects_path_support() -> None:
    """A pair supported by TWO distinct 2-hop paths outscores a pair supported by one."""
    observed = [
        # A -> {B, D} ; B -> C ; D -> C  => A->C has TWO paths (via B and via D).
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("A", "partner_of", "D"),
        ObservedEdge("B", "partner_of", "C"),
        ObservedEdge("D", "partner_of", "C"),
        # E -> F -> G  => E->G has ONE path.
        ObservedEdge("E", "partner_of", "F"),
        ObservedEdge("F", "partner_of", "G"),
    ]
    cfg = CompletionConfig(transitive_labels=frozenset({"partner_of"}))
    preds = GraphCompletionEngine(cfg).predict(observed)
    ac = next(p for p in preds if (p.source_node_id, p.target_node_id) == ("A", "C"))
    eg = next(p for p in preds if (p.source_node_id, p.target_node_id) == ("E", "G"))
    assert ac.support == 2
    assert eg.support == 1
    assert ac.score >= eg.score


def test_engine_bounded_fan_out() -> None:
    """A hub source with many reachable targets keeps at most max_candidates_per_node proposals."""
    # Hub H -> M ; M -> T0..T9  => H reaches ten targets transitively.
    observed = [ObservedEdge("H", "partner_of", "M")]
    observed += [ObservedEdge("M", "partner_of", f"T{i}") for i in range(10)]
    cfg = CompletionConfig(
        transitive_labels=frozenset({"partner_of"}), max_candidates_per_node=3
    )
    preds = GraphCompletionEngine(cfg).predict(observed)
    from_h = [p for p in preds if p.source_node_id == "H"]
    assert len(from_h) == 3  # bounded


def test_engine_is_deterministic_byte_identical() -> None:
    """Two runs over the same observed edges yield identical predictions in identical order."""
    observed = [
        ObservedEdge("A", "partner_of", "B"),
        ObservedEdge("B", "partner_of", "C"),
        ObservedEdge("C", "partner_of", "D"),
    ]
    cfg = CompletionConfig(transitive_labels=frozenset({"partner_of"}))
    a = GraphCompletionEngine(cfg).predict(observed)
    b = GraphCompletionEngine(cfg).predict(observed)
    assert [(p.source_node_id, p.label, p.target_node_id, p.score, p.rank) for p in a] == [
        (p.source_node_id, p.label, p.target_node_id, p.score, p.rank) for p in b
    ]
    # Ranks are dense and ascending in the emitted order.
    assert [p.rank for p in a] == list(range(len(a)))


def test_engine_empty_input_yields_no_predictions() -> None:
    assert GraphCompletionEngine().predict([]) == []


def test_engine_zero_config_treats_every_label_as_transitive_and_symmetric() -> None:
    """With no label config, every observed label gets both the transitive + symmetric rule."""
    observed = [
        ObservedEdge("A", "knows", "B"),
        ObservedEdge("B", "knows", "C"),
    ]
    preds = GraphCompletionEngine().predict(observed)  # default config
    keys = {(p.source_node_id, p.label, p.target_node_id) for p in preds}
    assert ("A", "knows", "C") in keys  # transitive
    assert ("B", "knows", "A") in keys  # symmetric


# ---------------------------------------------------------------------------
# Provider: the inferred-edge contract mapping.
# ---------------------------------------------------------------------------


def _node_record(node_id: str, name: str, docs: list[str]) -> GraphRecord:
    node = GraphNode(
        record_id=f"node:{node_id}",
        node_id=node_id,
        entity_id=f"ent-{node_id}",
        label="ORG",
        canonical_name=name,
        member_mention_ids=[f"m-{node_id}"],
        source_document_ids=docs,
        confidence=0.9,
        evidence=Evidence(mention_ids=[f"m-{node_id}"], document_ids=docs, confidence=0.9),
        provenance=_prov(docs[0]),
        classification=_class(),
    )
    return GraphRecord(
        record_id=f"node:{node_id}#g",
        scope=GraphScope.NODE,
        node=node,
        provenance=_prov(docs[0]),
        classification=_class(),
    )


def _edge_record(src: str, label: str, tgt: str, rel_id: str) -> GraphRecord:
    edge = GraphEdge(
        record_id=f"edge:{rel_id}",
        edge_id=make_graph_id("corpus", "edge", rel_id),
        relation_id=rel_id,
        source_node_id=src,
        target_node_id=tgt,
        label=label,
        confidence=0.8,
        properties={"is_predicted": False},
        evidence=Evidence(mention_ids=[f"m-{rel_id}"], document_ids=["doc-a"], confidence=0.8),
        provenance=_prov(),
        classification=_class(),
    )
    return GraphRecord(
        record_id=f"edge:{rel_id}#g",
        scope=GraphScope.EDGE,
        edge=edge,
        provenance=_prov(),
        classification=_class(),
    )


def _assembled_chain() -> list[GraphRecord]:
    """A --partner_of--> B --partner_of--> C over three nodes with distinct source docs."""
    return [
        _node_record("nA", "Acme", ["doc-a"]),
        _node_record("nB", "Beta", ["doc-b"]),
        _node_record("nC", "Gamma", ["doc-c"]),
        _edge_record("nA", "partner_of", "nB", "rel-ab"),
        _edge_record("nB", "partner_of", "nC", "rel-bc"),
    ]


def test_provider_emits_inferred_marked_predicted_edge() -> None:
    """The predicted edge is scope=EDGE, marked inferred=True, with the scorer metadata."""
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"], "symmetric_labels": []}
    )
    out = list(provider.complete(_assembled_chain()))
    # Only NEW predicted edges, never the input nodes/edges.
    assert all(r.scope is GraphScope.EDGE for r in out)
    predicted = [r for r in out if r.edge is not None]
    ac = next(r.edge for r in predicted if r.edge and r.edge.source_node_id == "nA")
    assert ac.properties["inferred"] is True
    assert ac.properties["scorer"] == "reference.adjacency"
    assert ac.properties["calibrated"] is False
    assert "score" in ac.properties and "rank" in ac.properties
    assert ac.source_node_id == "nA" and ac.target_node_id == "nC"
    assert ac.label == "partner_of"


def test_provider_evidence_is_honest_empty_mentions_union_docs() -> None:
    """Predicted-edge Evidence has EMPTY mention_ids (no fabricated offsets) + union of endpoint docs."""
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"], "symmetric_labels": []}
    )
    ac = next(
        r.edge
        for r in provider.complete(_assembled_chain())
        if r.edge and r.edge.source_node_id == "nA"
    )
    # No textual mention — asserting one would be a lie.
    assert ac.evidence.mention_ids == []
    # document_ids = union of head (nA: doc-a) + tail (nC: doc-c) source docs.
    assert ac.evidence.document_ids == ["doc-a", "doc-c"]
    assert "Predicted edge (not asserted in text)" in ac.evidence.snippet
    assert "reference.adjacency" in ac.evidence.snippet
    # Confidence tracks the calibrated score.
    assert ac.evidence.confidence == ac.confidence


def test_provider_edge_id_is_deterministic_and_distinct_from_asserted() -> None:
    """edge_id = sha256 over (corpus, src, label, tgt, 'predicted'); relation_id names the scorer."""
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"], "symmetric_labels": []}
    )
    ac = next(
        r.edge
        for r in provider.complete(_assembled_chain())
        if r.edge and r.edge.source_node_id == "nA"
    )
    expected = make_graph_id("corpus", "predicted", "nA", "partner_of", "nC")
    assert ac.edge_id == expected
    assert ac.relation_id == f"predicted:reference.adjacency:{expected}"
    # Distinct from any asserted edge id (asserted uses the 'edge' discriminator, not 'predicted').
    asserted_ab = make_graph_id("corpus", "edge", "rel-ab")
    assert ac.edge_id != asserted_ab


def test_provider_drops_collision_with_asserted_edge() -> None:
    """If A--partner_of-->C is already asserted, the completer emits NO predicted A->C edge."""
    records = _assembled_chain()
    records.append(_edge_record("nA", "partner_of", "nC", "rel-ac"))  # already asserted
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"], "symmetric_labels": []}
    )
    predicted = [r.edge for r in provider.complete(records) if r.edge is not None]
    ac = [e for e in predicted if e.source_node_id == "nA" and e.target_node_id == "nC"]
    assert ac == []


def test_provider_never_reemits_the_input() -> None:
    """The completer output contains only NEW predicted edges — none of the input node/edge ids."""
    records = _assembled_chain()
    input_edge_ids = {r.edge.edge_id for r in records if r.edge is not None}
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"]}
    )
    out_edge_ids = {r.edge.edge_id for r in provider.complete(records) if r.edge is not None}
    assert out_edge_ids.isdisjoint(input_edge_ids)


def test_provider_deterministic_byte_identical_over_two_runs() -> None:
    records = _assembled_chain()
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"]}
    )
    a = [r.model_dump_json() for r in provider.complete(records)]
    b = [r.model_dump_json() for r in provider.complete(records)]
    assert a == b


def test_provider_empty_graph_yields_no_predictions() -> None:
    provider = ReferenceGraphCompleter({"corpus_id": "corpus"})
    assert list(provider.complete([])) == []


def test_provider_predicted_edges_in_edge_id_order() -> None:
    """The Provider yields predicted edges sorted by edge_id (the Baseline determinism bar)."""
    # A longer chain A->B->C->D under zero-config (transitive) yields several 2-hop proposals.
    records = [
        _node_record("nA", "A", ["doc-a"]),
        _node_record("nB", "B", ["doc-b"]),
        _node_record("nC", "C", ["doc-c"]),
        _node_record("nD", "D", ["doc-d"]),
        _edge_record("nA", "partner_of", "nB", "rel-ab"),
        _edge_record("nB", "partner_of", "nC", "rel-bc"),
        _edge_record("nC", "partner_of", "nD", "rel-cd"),
    ]
    provider = ReferenceGraphCompleter(
        {"corpus_id": "corpus", "transitive_labels": ["partner_of"], "symmetric_labels": []}
    )
    edge_ids = [r.edge.edge_id for r in provider.complete(records) if r.edge is not None]
    assert edge_ids == sorted(edge_ids)
    assert len(edge_ids) >= 2


def test_provider_profile_is_honest_cpu_deterministic() -> None:
    """The declared ProviderProfile is pure-code, CPU, deterministic, PolyForm-Noncommercial-1.0.0, no model."""
    profile = ReferenceGraphCompleter.profile()
    assert profile.compute == "cpu"
    assert profile.deterministic is True
    assert profile.model_id is None
    assert profile.license_verified is True
