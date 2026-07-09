"""Unit tests for UltraGraphCompleter: the Slice-1 inferred-edge contract + failure modes."""

from __future__ import annotations

import pytest
from _linkpred_fakes import FakeScorer, UnavailableScorer, edge_record, node_record
from latence_core.capability import GraphCompleter
from latence_core.contracts import GraphRecord
from latence_core.errors import ProviderError
from latence_linkpred_ultra import provider as provider_mod
from latence_linkpred_ultra.provider import UltraGraphCompleter

_CORPUS = "test-corpus"


def _chain_graph() -> list[GraphRecord]:
    """A transitive org chain: acme->beta, beta->gamma (partner_of). Completion: acme->gamma."""
    n_acme = node_record(_CORPUS, "acme", "org", ["d1"])
    n_beta = node_record(_CORPUS, "beta", "org", ["d1", "d2"])
    n_gamma = node_record(_CORPUS, "gamma", "org", ["d2"])
    ids = {r.node.canonical_name: r.node.node_id for r in (n_acme, n_beta, n_gamma) if r.node}
    e1 = edge_record(_CORPUS, ids["acme"], "partner_of", ids["beta"], ["d1"])
    e2 = edge_record(_CORPUS, ids["beta"], "partner_of", ids["gamma"], ["d2"])
    return [n_acme, n_beta, n_gamma, e1, e2]


@pytest.fixture
def use_fake_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_mod, "UltraScorer", FakeScorer)


def test_satisfies_graphcompleter_protocol() -> None:
    p = UltraGraphCompleter({"corpus_id": _CORPUS})
    assert isinstance(p, GraphCompleter)


def test_profile_license_verified_mit() -> None:
    prof = UltraGraphCompleter.profile()
    assert prof.license == "MIT"
    assert prof.license_verified is True
    assert prof.license_code == "MIT"
    assert prof.license_weights == "MIT"
    assert prof.compute == "either"
    assert prof.deterministic is False  # learned float inference — not overclaimed


def test_emits_inferred_edges_via_slice1_contract(use_fake_scorer: None) -> None:
    p = UltraGraphCompleter(
        {"corpus_id": _CORPUS, "scorer": "ultra", "min_probability": 0.5, "target_precision": 0.9}
    )
    predicted = list(p.complete(_chain_graph()))
    assert predicted, "the fake ULTRA scorer completes the transitive acme->gamma edge"
    for r in predicted:
        assert r.scope.value == "edge" and r.edge is not None
        props = r.edge.properties
        # The inferred-edge contract EXACTLY (ADR-0037):
        assert props["inferred"] is True
        assert props["scorer"] == "ultra"
        assert props["calibrated"] is True  # temperature-scaled prob, not a raw fraction
        assert 0.0 <= props["score"] <= 1.0
        assert r.edge.confidence == props["score"]
        # Honest, model-derived Evidence: NO fabricated mention offsets.
        assert r.edge.evidence.mention_ids == []
        assert r.edge.evidence.document_ids  # union of endpoint docs (non-empty here)
        assert "not asserted in text" in r.edge.evidence.snippet
        # relation_id clearly marks it predicted, not a source relation.
        assert r.edge.relation_id.startswith("predicted:ultra:")
        assert r.record_id.endswith("#predicted")


def test_deterministic_byte_identical(use_fake_scorer: None) -> None:
    p1 = UltraGraphCompleter({"corpus_id": _CORPUS})
    p2 = UltraGraphCompleter({"corpus_id": _CORPUS})
    a = [r.model_dump_json() for r in p1.complete(_chain_graph())]
    b = [r.model_dump_json() for r in p2.complete(_chain_graph())]
    assert a == b
    # Emitted in edge-id order.
    ids = [r.edge.edge_id for r in p1.complete(_chain_graph()) if r.edge]
    assert ids == sorted(ids)


def test_asserted_collision_dropped(use_fake_scorer: None) -> None:
    """A candidate matching an asserted edge is never emitted (already asserted)."""
    graph = _chain_graph()
    # Add the acme->gamma edge as ASSERTED; the completer must then emit NOTHING new for it.
    ids = {r.node.canonical_name: r.node.node_id for r in graph if r.node}
    graph.append(edge_record(_CORPUS, ids["acme"], "partner_of", ids["gamma"], ["d1"]))
    p = UltraGraphCompleter({"corpus_id": _CORPUS})
    predicted = list(p.complete(graph))
    asserted_key = (ids["acme"], "partner_of", ids["gamma"])
    for r in predicted:
        assert r.edge is not None
        assert (r.edge.source_node_id, r.edge.label, r.edge.target_node_id) != asserted_key


def test_typed_error_when_no_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    """If neither ULTRA nor PyKEEN is available → a typed ProviderError, never a silent no-op."""
    monkeypatch.setattr(provider_mod, "UltraScorer", UnavailableScorer)
    monkeypatch.setattr(provider_mod, "PykeenScorer", UnavailableScorer)
    p = UltraGraphCompleter({"corpus_id": _CORPUS, "scorer": "auto"})
    with pytest.raises(ProviderError, match="no learned scorer available"):
        list(p.complete(_chain_graph()))


def test_skip_if_unavailable_yields_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional-trailing-stage posture: skip-with-flag (no edges) instead of raising."""
    monkeypatch.setattr(provider_mod, "UltraScorer", UnavailableScorer)
    monkeypatch.setattr(provider_mod, "PykeenScorer", UnavailableScorer)
    p = UltraGraphCompleter(
        {"corpus_id": _CORPUS, "scorer": "auto", "skip_if_unavailable": True}
    )
    assert list(p.complete(_chain_graph())) == []


def test_auto_falls_back_to_pykeen(monkeypatch: pytest.MonkeyPatch) -> None:
    """scorer=auto uses PyKEEN when ULTRA is unavailable (graceful fallback)."""

    class PykeenLikeFake(FakeScorer):
        name = "pykeen"

    monkeypatch.setattr(provider_mod, "UltraScorer", UnavailableScorer)
    monkeypatch.setattr(provider_mod, "PykeenScorer", PykeenLikeFake)
    p = UltraGraphCompleter({"corpus_id": _CORPUS, "scorer": "auto"})
    predicted = list(p.complete(_chain_graph()))
    assert p.last_scorer == "pykeen"
    assert predicted and all(
        r.edge and r.edge.properties["scorer"] == "pykeen" for r in predicted
    )


def test_bad_scorer_config_rejected() -> None:
    with pytest.raises(ValueError, match="scorer must be"):
        UltraGraphCompleter({"scorer": "nonsense"})


def test_no_observed_edges_yields_nothing(use_fake_scorer: None) -> None:
    p = UltraGraphCompleter({"corpus_id": _CORPUS})
    only_nodes = [node_record(_CORPUS, "solo", "org", ["d1"])]
    assert list(p.complete(only_nodes)) == []


def _holdout_graph() -> list[GraphRecord]:
    """A chain long enough that the eval split has held-out edges."""
    nodes = [node_record(_CORPUS, f"n{i}", "org", [f"d{i}"]) for i in range(6)]
    ids = [r.node.node_id for r in nodes if r.node]
    edges = [
        edge_record(_CORPUS, ids[i], "partner_of", ids[i + 1], [f"d{i}"])
        for i in range(5)
    ]
    return [*nodes, *edges]


def test_metrics_and_leak_detection_ran(use_fake_scorer: None) -> None:
    """The held-out MRR/Hits@k eval ran (leak-checked) and produced real numbers."""
    p = UltraGraphCompleter({"corpus_id": _CORPUS, "holdout_fraction": 0.4})
    list(p.complete(_holdout_graph()))
    assert p.last_metrics is not None
    assert p.last_metrics.leak_checked is True
    assert p.last_metrics.num_queries > 0
    assert 0.0 <= p.last_metrics.mrr <= 1.0


def test_provider_wires_the_leak_guard_into_evaluate(
    use_fake_scorer: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider actually CALLS assert_no_leak in _evaluate — deleting the call is caught.

    Monkeypatch the leak guard the provider imported to raise. If _evaluate invokes it (as it must),
    the raise propagates and guard() surfaces it as a typed LatenceError. If someone removed the
    assert_no_leak call from _evaluate, the patched guard never fires and NO error is raised, turning
    this test RED — closing the reviewer's "leak-guard integration is untested" gap.
    """
    from latence_core.errors import LatenceError
    from latence_linkpred_ultra.evaluation import LeakDetectionError

    def _boom(observed: object, held_out: object) -> None:
        raise LeakDetectionError("injected: leak guard was invoked by _evaluate")

    monkeypatch.setattr(provider_mod, "assert_no_leak", _boom)
    p = UltraGraphCompleter({"corpus_id": _CORPUS, "holdout_fraction": 0.4})
    # guard() maps the raised LeakDetectionError to a typed LatenceError (never a silent swallow);
    # the assertion is that SOME typed error propagates, proving _evaluate called the guard.
    with pytest.raises(LatenceError):
        list(p.complete(_holdout_graph()))
