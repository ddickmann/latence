"""The Level-2 orchestrator :func:`retrieve`, tested end-to-end through the faithful fake backend.

Every assertion is at the seam — a :class:`QueryBundle` + a
:class:`~latence_retrieval.testing.FakeBackend` (+ real fuser/processors, or the shipped fake
cross-encoder) go in, a :class:`RetrievalResult` comes out — never the orchestrator's internals, so
the tests survive any refactor of how it plans or chains. They pin the three acceptance criteria: the
opinionated config-overridable pipeline runs (AC1), it composes only the legs the backend + bundle
support and drops the rest (AC2), and orchestration is stateless — the fake backend runs every
first-stage search (AC3).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from latence_retrieval import (
    Candidate,
    Capability,
    OrchestrationError,
    OrchestrationPipeline,
    QueryBundle,
    RetrievalResult,
    WeightedFuser,
    retrieve,
)
from latence_retrieval.processor import CandidateProcessor, IdentityProcessor
from latence_retrieval.rerank import Reranker
from latence_retrieval.testing import FakeBackend, FakeCrossEncoder

# A small corpus with graph columns so `expand` has something to bite on, and text a cross-encoder
# fake can score by lexical overlap.
_CORPUS = [
    Candidate(id="a", score=0.90, text="alpha beta", metadata={"kg_node_ids": ["n1"]}),
    Candidate(id="b", score=0.50, text="beta gamma", metadata={"kg_node_ids": ["n2"]}),
    Candidate(id="c", score=0.70, text="gamma delta", metadata={"kg_node_ids": ["n1"]}),
]


def _hybrid_backend() -> FakeBackend:
    """A store declaring dense + sparse + bm25 + filter (dimension 3 for its dense index)."""
    return FakeBackend(
        _CORPUS,
        capabilities={
            Capability.DENSE,
            Capability.SPARSE,
            Capability.BM25,
            Capability.FILTER,
        },
        dimension=3,
    )


def _full_bundle(**over: object) -> QueryBundle:
    base: dict[str, object] = {
        "text": "beta",
        "vector": (0.1, 0.2, 0.3),
        "sparse_terms": {"beta": 1.0},
    }
    base.update(over)
    return QueryBundle(**base)  # type: ignore[arg-type]


# --- AC1: the opinionated, config-overridable pipeline runs -------------------------------------


def test_default_pipeline_fetches_and_returns_a_result() -> None:
    result = retrieve(_full_bundle(), _hybrid_backend())
    assert isinstance(result, RetrievalResult)
    # Every corpus id comes back (all three lists cover the corpus; fusion unions them).
    assert {c.id for c in result.candidates} == {"a", "b", "c"}


def test_default_pipeline_is_fetch_plus_fuse_no_processor_legs() -> None:
    # No rerank/pack by default (they need a model/budget, no safe default — ADR-0012): the result
    # is the fused first-stage lists, ordered by fused score.
    result = retrieve(_full_bundle(), _hybrid_backend())
    assert result.candidates[0].id == "a"  # highest first-stage score across every list


def test_config_overridable_pipeline_runs_a_processor_chain() -> None:
    # A rerank leg (with the faithful fake cross-encoder) reorders by lexical overlap with the query
    # "beta": ids "a" and "b" both contain "beta" (score 1.0), "c" does not (0.0) — so "c" sinks.
    pipeline = OrchestrationPipeline(
        processors=(Reranker(model=FakeCrossEncoder()),),
    )
    result = retrieve(_full_bundle(), _hybrid_backend(), pipeline)
    assert result.candidates[-1].id == "c"
    assert result.candidates[-1].score == pytest.approx(0.0)


def test_final_top_k_truncates_the_composed_list() -> None:
    pipeline = OrchestrationPipeline(final_top_k=1)
    result = retrieve(_full_bundle(), _hybrid_backend(), pipeline)
    assert len(result.candidates) == 1


def test_from_config_builds_an_equivalent_pipeline() -> None:
    # The operator-config path resolves the fuser + processor chain through the registry and runs.
    pipeline = OrchestrationPipeline.from_config(
        {
            "fuser": {"provider": "fuser.rrf"},
            "processors": [{"provider": "processor.identity"}],
            "final_top_k": 2,
        }
    )
    result = retrieve(_full_bundle(), _hybrid_backend(), pipeline)
    assert len(result.candidates) == 2
    assert result.fetched_modes == (Capability.BM25, Capability.DENSE, Capability.SPARSE)


# --- AC2: composes only the supported legs, drops the rest --------------------------------------


def test_drops_a_mode_the_backend_does_not_declare() -> None:
    # Bundle carries a sparse signal, but a dense+bm25-only store cannot serve sparse — that leg is
    # dropped even though the signal is available.
    backend = FakeBackend(
        _CORPUS, capabilities={Capability.DENSE, Capability.BM25}, dimension=3
    )
    result = retrieve(_full_bundle(), backend)
    assert Capability.SPARSE not in result.fetched_modes
    assert result.fetched_modes == (Capability.BM25, Capability.DENSE)


def test_drops_a_declared_mode_the_bundle_has_no_signal_for() -> None:
    # Store declares dense+bm25, but the bundle carries no vector — the dense leg is dropped; only
    # bm25 (text, always present) fires.
    backend = FakeBackend(
        _CORPUS, capabilities={Capability.DENSE, Capability.BM25}, dimension=3
    )
    result = retrieve(_full_bundle(vector=None), backend)
    assert result.fetched_modes == (Capability.BM25,)


def test_single_leg_skips_fusion_and_preserves_store_scores() -> None:
    # bm25-only store + text-only bundle → one leg, no fusion: the store's own scores survive intact
    # (fusing a lone list by RRF would overwrite them). Descending store order.
    backend = FakeBackend(_CORPUS, capabilities={Capability.BM25})
    result = retrieve(QueryBundle(text="beta"), backend)
    assert result.fetched_modes == (Capability.BM25,)
    assert [c.id for c in result.candidates] == ["a", "c", "b"]
    assert [c.score for c in result.candidates] == [0.90, 0.70, 0.50]


def test_no_runnable_leg_raises_before_any_fetch() -> None:
    # Dense-only store, but the bundle carries no vector → nothing to search: loud, not empty.
    backend = FakeBackend(_CORPUS, capabilities={Capability.DENSE}, dimension=3)
    with pytest.raises(OrchestrationError, match="no runnable fetch leg"):
        retrieve(QueryBundle(text="beta"), backend)


def test_filter_rides_every_leg_when_the_backend_declares_it() -> None:
    backend = FakeBackend(
        [
            Candidate(id="a", score=0.9, metadata={"tenant": "acme"}),
            Candidate(id="c", score=0.7, metadata={"tenant": "globex"}),
        ],
        capabilities={Capability.DENSE, Capability.BM25, Capability.FILTER},
        dimension=3,
    )
    result = retrieve(
        _full_bundle(sparse_terms=None, filter={"tenant": "acme"}), backend
    )
    assert {c.id for c in result.candidates} == {"a"}


def test_unserveable_filter_raises_rather_than_silently_dropping() -> None:
    # A filter the store cannot serve is NOT silently dropped (that would return unfiltered hits) —
    # anti-false-green: fail loudly instead.
    backend = FakeBackend(
        _CORPUS, capabilities={Capability.DENSE, Capability.BM25}, dimension=3
    )
    with pytest.raises(OrchestrationError, match="does not declare the 'filter' capability"):
        retrieve(_full_bundle(filter={"tenant": "acme"}), backend)


def test_weighted_fusion_weights_align_to_fired_modes() -> None:
    # Give the dense leg all the weight and bm25/sparse none: the result order follows the dense
    # list's scores (fake dense returns the corpus in descending score → a, c, b), proving the
    # per-mode weights are aligned to the fetched-mode order.
    pipeline = OrchestrationPipeline(
        fuser=WeightedFuser(),
        fuse_weights={"dense": 1.0, "bm25": 0.0, "sparse": 0.0},
    )
    result = retrieve(_full_bundle(), _hybrid_backend(), pipeline)
    assert [c.id for c in result.candidates] == ["a", "c", "b"]


# --- AC3: stateless — the backend runs every first-stage search ---------------------------------


class _CountingBackend:
    """Wraps a FakeBackend and counts search calls, to prove the backend is the only fetch."""

    def __init__(self, inner: FakeBackend) -> None:
        self._inner = inner
        self.search_calls = 0

    @property
    def capabilities(self) -> frozenset[Capability]:
        return self._inner.capabilities

    def search(self, query: object) -> list[Candidate]:
        self.search_calls += 1
        return self._inner.search(query)  # type: ignore[arg-type]


def test_one_backend_search_per_fired_leg() -> None:
    backend = _CountingBackend(_hybrid_backend())
    result = retrieve(_full_bundle(), backend)
    # dense + sparse + bm25 all fire → exactly three first-stage searches, all on the backend.
    assert backend.search_calls == 3
    assert result.fetched_modes == (Capability.BM25, Capability.DENSE, Capability.SPARSE)


def test_retrieve_is_pure_same_inputs_same_result() -> None:
    backend = _hybrid_backend()
    bundle = _full_bundle()
    first = retrieve(bundle, backend)
    second = retrieve(bundle, backend)
    assert first.model_dump() == second.model_dump()


def test_input_candidates_are_never_mutated() -> None:
    # A processor cannot mutate the frozen candidates the backend returned; the corpus is untouched.
    class _Touch:
        def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
            return list(candidates)

    scores_before = [c.score for c in _CORPUS]
    retrieve(_full_bundle(), _hybrid_backend(), OrchestrationPipeline(processors=(_Touch(),)))
    assert [c.score for c in _CORPUS] == scores_before


# --- planning-config validation (fails loud at construction) ------------------------------------


def test_pipeline_rejects_non_positive_fetch_top_k() -> None:
    with pytest.raises(ValueError, match="fetch_top_k"):
        OrchestrationPipeline(fetch_top_k=0)


def test_pipeline_rejects_unknown_fuse_weight_mode() -> None:
    with pytest.raises(ValueError, match="fuse_weights"):
        OrchestrationPipeline(fuse_weights={"filter": 1.0})


def test_from_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        OrchestrationPipeline.from_config({"nonsense": 1})


def test_query_bundle_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        QueryBundle(text="")


def test_identity_processor_is_accepted_as_a_leg() -> None:
    # A plain CandidateProcessor (not just a Reranker) chains cleanly — the chain is seam-typed.
    proc: CandidateProcessor = IdentityProcessor()
    result = retrieve(_full_bundle(), _hybrid_backend(), OrchestrationPipeline(processors=(proc,)))
    assert {c.id for c in result.candidates} == {"a", "b", "c"}
