"""The statelessness invariant (ADR-0048), asserted by the seam's shape and behaviour.

The package must hold no durable index and run no first-stage search. That is enforced structurally,
not documented: a candidate processor's *only* input is a caller-supplied candidate list (it has no
method to fetch or store), the one fetch in the library is a backend ``search`` that runs the
*caller's* engine, and no component accumulates state across calls. These tests pin those shapes so a
future change that smuggled in an index or a stored corpus would break them.
"""

from __future__ import annotations

import inspect

from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    Capability,
    IdentityProcessor,
    RetrievalBackend,
    RetrievalQuery,
)
from latence_retrieval.testing import FakeBackend


def test_processor_seam_only_accepts_candidates_it_never_fetches() -> None:
    # The processor Protocol's single method takes (query, candidates) and returns candidates —
    # there is no fetch/index/store method through which it could acquire its own candidates.
    members = {name for name, _ in inspect.getmembers(CandidateProcessor, inspect.isfunction)}
    public = {m for m in members if not m.startswith("_")}
    assert public == {"process"}
    params = list(inspect.signature(IdentityProcessor.process).parameters)
    assert params == ["self", "query", "candidates"]


def test_backend_search_requires_a_caller_supplied_query() -> None:
    # The only fetch in the library is RetrievalBackend.search, and it cannot run without a query
    # the caller hands in — the tooling never originates a search on its own.
    params = list(inspect.signature(RetrievalBackend.search).parameters)
    assert params == ["self", "query"]


def test_processor_does_not_retain_candidates_across_calls() -> None:
    proc = IdentityProcessor()
    proc.process("q", [Candidate(id="a", score=1.0)])
    # A second call with a different (empty) input must reflect only that input — nothing from the
    # first call leaked into a hidden store.
    assert proc.process("q", []) == []


def test_backend_holds_only_the_corpus_it_was_handed() -> None:
    # The fake stands in for a customer engine: it returns only what was explicitly provided, never
    # anything it fetched or persisted itself.
    backend = FakeBackend(
        [Candidate(id="only", score=1.0)], capabilities={Capability.BM25}
    )
    hits = backend.search(RetrievalQuery(mode=Capability.BM25, text="q"))
    assert [c.id for c in hits] == ["only"]


def test_fake_backend_is_the_second_adapter_of_the_seam() -> None:
    # A real seam has two adapters; the faithful fake is the offline one (Qdrant is the prod one,
    # later ticket). This pins that the fake genuinely satisfies the same Protocol.
    assert isinstance(FakeBackend([], capabilities={Capability.BM25}), RetrievalBackend)
