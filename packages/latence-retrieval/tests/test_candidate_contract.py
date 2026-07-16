"""The :class:`Candidate` + :class:`RetrievalQuery` contracts, tested at their serialization boundary.

Asserts external behaviour only: what round-trips, what is rejected. A candidate must survive a
dict and a JSON round-trip losslessly (a store's response crosses process/file boundaries), and both
contracts must reject malformed input at construction (``extra="forbid"`` + the query's mode/payload
invariant) so a bad record never silently corrupts a rank list.
"""

from __future__ import annotations

import pytest
from latence_retrieval import Candidate, Capability, RetrievalQuery
from pydantic import ValidationError


def test_candidate_carries_id_score_text_metadata() -> None:
    c = Candidate(id="doc-1", score=0.87, text="hello", metadata={"page": 3, "source": "a.pdf"})
    assert c.id == "doc-1"
    assert c.score == pytest.approx(0.87)
    assert c.text == "hello"
    assert c.metadata == {"page": 3, "source": "a.pdf"}


def test_candidate_defaults_text_and_metadata() -> None:
    c = Candidate(id="x", score=1.0)
    assert c.text == ""
    assert c.metadata == {}


def test_candidate_round_trips_through_dict() -> None:
    c = Candidate(id="doc-1", score=0.5, text="body", metadata={"k": [1, 2], "n": None})
    assert Candidate.model_validate(c.model_dump()) == c


def test_candidate_round_trips_through_json() -> None:
    c = Candidate(id="doc-1", score=-0.25, text="body", metadata={"nested": {"a": 1}})
    assert Candidate.model_validate_json(c.model_dump_json()) == c


def test_candidate_is_frozen() -> None:
    c = Candidate(id="x", score=1.0)
    with pytest.raises(ValidationError):
        c.score = 2.0


def test_candidate_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Candidate.model_validate({"id": "x", "score": 1.0, "surprise": True})


def test_candidate_rejects_empty_id() -> None:
    with pytest.raises(ValidationError):
        Candidate(id="", score=1.0)


def test_query_dense_requires_vector() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(mode=Capability.DENSE)


def test_query_sparse_requires_terms() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(mode=Capability.SPARSE)


def test_query_bm25_requires_text() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(mode=Capability.BM25, text="")


def test_query_filter_mode_is_rejected() -> None:
    # FILTER is a predicate, never a scoring mode — a store cannot "search by filter".
    with pytest.raises(ValidationError):
        RetrievalQuery(mode=Capability.FILTER)


def test_query_well_formed_dense_query_builds() -> None:
    q = RetrievalQuery(mode=Capability.DENSE, vector=(0.1, 0.2, 0.3), top_k=5)
    assert q.mode is Capability.DENSE
    assert q.vector == (0.1, 0.2, 0.3)
    assert q.top_k == 5


def test_query_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(mode=Capability.BM25, text="q", top_k=0)


def test_query_round_trips_through_json() -> None:
    q = RetrievalQuery(mode=Capability.DENSE, vector=(1.0, 2.0), filter={"tenant": "acme"})
    assert RetrievalQuery.model_validate_json(q.model_dump_json()) == q
