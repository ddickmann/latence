"""The :class:`IdentityProcessor`, tested through the :class:`CandidateProcessor` seam.

Drives the processor over candidates a :class:`FakeBackend` returned — the end-to-end query-time
shape (backend search → candidate processor) — and asserts the identity contract: same candidates,
same order, a fresh list (input never mutated), and no accumulated state across calls.
"""

from __future__ import annotations

from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    Capability,
    IdentityProcessor,
    RetrievalQuery,
)
from latence_retrieval.testing import FakeBackend


def _candidates_from_backend() -> list[Candidate]:
    backend = FakeBackend(
        [
            Candidate(id="a", score=0.9, text="alpha"),
            Candidate(id="b", score=0.5, text="bravo"),
        ],
        capabilities={Capability.DENSE},
        dimension=2,
    )
    return backend.search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0)))


def test_identity_processor_satisfies_the_seam() -> None:
    assert isinstance(IdentityProcessor(), CandidateProcessor)


def test_identity_returns_input_unchanged() -> None:
    candidates = _candidates_from_backend()
    out = IdentityProcessor().process("query", candidates)
    assert out == candidates


def test_identity_returns_a_new_list_not_the_input_object() -> None:
    candidates = _candidates_from_backend()
    out = IdentityProcessor().process("query", candidates)
    # A caller mutating the returned list must not disturb the list it passed in.
    assert out is not candidates
    out.clear()
    assert len(candidates) == 2


def test_identity_is_stateless_across_calls() -> None:
    proc = IdentityProcessor()
    candidates = _candidates_from_backend()
    first = proc.process("q", candidates)
    second = proc.process("q", candidates)
    # No accumulation: the same input yields byte-identical output every call.
    assert first == second == candidates


def test_identity_accepts_and_ignores_config() -> None:
    candidates = _candidates_from_backend()
    out = IdentityProcessor({"anything": 1}).process("q", candidates)
    assert out == candidates


def test_identity_on_empty_list() -> None:
    assert IdentityProcessor().process("q", []) == []
