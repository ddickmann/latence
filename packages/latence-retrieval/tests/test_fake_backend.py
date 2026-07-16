"""The faithful :class:`FakeBackend`, tested at the :class:`RetrievalBackend` seam (anti-false-green).

These tests assert the fake rejects **exactly** what a real vector store rejects — an undeclared
capability, an undeclared filter, a wrong-width dense vector — and, when the query is valid, returns
the right hits in the right order. Behaviour is asserted through the seam (``search`` in, candidates
out, typed errors), never internals: the tests survive any refactor of how the fake scores or filters.
"""

from __future__ import annotations

import pytest
from latence_retrieval import (
    BackendCapabilityError,
    BackendDimensionError,
    Candidate,
    Capability,
    RetrievalBackend,
    RetrievalQuery,
)
from latence_retrieval.testing import FakeBackend

_CORPUS = [
    Candidate(id="a", score=0.9, text="alpha", metadata={"tenant": "acme", "year": 2024}),
    Candidate(id="b", score=0.5, text="bravo", metadata={"tenant": "acme", "year": 2023}),
    Candidate(id="c", score=0.7, text="charlie", metadata={"tenant": "globex", "year": 2024}),
]


def _dense_backend(**kw: object) -> FakeBackend:
    return FakeBackend(_CORPUS, capabilities={Capability.DENSE, Capability.FILTER}, dimension=3)


def test_fake_backend_satisfies_the_seam() -> None:
    # runtime_checkable Protocol: the fake IS a RetrievalBackend without inheriting from it.
    assert isinstance(_dense_backend(), RetrievalBackend)


def test_declares_its_capabilities() -> None:
    backend = _dense_backend()
    assert backend.capabilities == frozenset({Capability.DENSE, Capability.FILTER})


def test_search_returns_hits_in_descending_score_order() -> None:
    backend = _dense_backend()
    hits = backend.search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))
    assert [c.id for c in hits] == ["a", "c", "b"]


def test_search_honours_top_k() -> None:
    backend = _dense_backend()
    hits = backend.search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), top_k=2))
    assert [c.id for c in hits] == ["a", "c"]


def test_search_applies_metadata_filter() -> None:
    backend = _dense_backend()
    hits = backend.search(
        RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), filter={"tenant": "acme"})
    )
    assert [c.id for c in hits] == ["a", "b"]


def test_filter_excludes_candidates_missing_the_field() -> None:
    backend = FakeBackend(
        [Candidate(id="a", score=1.0, metadata={"tenant": "acme"}), Candidate(id="d", score=0.9)],
        capabilities={Capability.DENSE, Capability.FILTER},
        dimension=2,
    )
    hits = backend.search(
        RetrievalQuery(mode=Capability.DENSE, vector=(1.0, 0.0), filter={"tenant": "acme"})
    )
    assert [c.id for c in hits] == ["a"]


def test_rejects_undeclared_capability() -> None:
    # Backend declares DENSE only; a BM25 query must be rejected, as a dense-only store would.
    backend = FakeBackend(_CORPUS, capabilities={Capability.DENSE}, dimension=3)
    with pytest.raises(BackendCapabilityError):
        backend.search(RetrievalQuery(mode=Capability.BM25, text="alpha"))


def test_rejects_filter_when_filter_not_declared() -> None:
    backend = FakeBackend(_CORPUS, capabilities={Capability.DENSE}, dimension=3)
    with pytest.raises(BackendCapabilityError):
        backend.search(
            RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), filter={"tenant": "acme"})
        )


def test_rejects_wrong_dimension_vector() -> None:
    backend = _dense_backend()  # dimension=3
    with pytest.raises(BackendDimensionError):
        backend.search(RetrievalQuery(mode=Capability.DENSE, vector=(0.1, 0.2)))


def test_dense_capability_requires_a_dimension_at_construction() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FakeBackend(_CORPUS, capabilities={Capability.DENSE})


def test_multivector_capability_also_requires_a_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FakeBackend(_CORPUS, capabilities={Capability.MULTIVECTOR})


def test_bm25_backend_needs_no_dimension() -> None:
    backend = FakeBackend(_CORPUS, capabilities={Capability.BM25})
    hits = backend.search(RetrievalQuery(mode=Capability.BM25, text="alpha", top_k=1))
    assert [c.id for c in hits] == ["a"]
