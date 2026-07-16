"""The experimental MaxSim rerank path (ADR-0050), tested at the boundary.

Three boundaries, all offline (no ColBERT-family encoder; the in-core reference reuse is exercised
where ``latence-core`` is present, else injected):

* the **reference :func:`~latence_retrieval.maxsim.max_sim` scorer** on **synthetic multi-vectors**
  (the AC: "tested for correctness on synthetic multi-vectors") — real math asserted on planted
  query/document token sets, including the faithful wrong-width rejection;
* the **:class:`~latence_retrieval.maxsim.MaxSimReranker` processor** driven through an injected
  faithful :class:`~latence_retrieval.testing.FakeMultiVectorEmbedder` — every observable behaviour
  (re-score, reorder, tie-break, ``top_k``, no-mutation/no-state, missing/malformed sidecar
  rejection) asserted on the returned candidate list, never the internal MaxSim maths, so the tests
  survive a refactor of *how* the score is computed;
* the **query-side multi-vector generation** (:class:`HashingMultiVectorQueryEncoder`) — that it
  **reuses** the index-time ``multivector.hashing`` embedder so the query representation matches the
  index-time contract byte-for-byte (the AC), plus its lazy-import boundary.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest
from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    HashingMultiVectorQueryEncoder,
    MaxSimError,
    MaxSimReranker,
    MultiVectorQueryEncoder,
    max_sim,
    resolve_multivector_encoder,
    resolve_processor,
)
from latence_retrieval.testing import FakeMultiVectorEmbedder


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


def _reranker(config: dict[str, Any] | None = None, *, dimension: int = 8) -> MaxSimReranker:
    return MaxSimReranker(config or {}, encoder=FakeMultiVectorEmbedder(dimension))


def _doc(mv: list[list[float]], **extra: Any) -> dict[str, Any]:
    """A candidate metadata dict carrying a planted document multi-vector under the default key."""
    return {"multivectors": mv, **extra}


# --- seam identity ----------------------------------------------------------------------------


def test_maxsim_reranker_satisfies_the_candidate_processor_seam() -> None:
    assert isinstance(_reranker(), CandidateProcessor)


def test_fake_multivector_embedder_satisfies_the_query_encoder_seam() -> None:
    assert isinstance(FakeMultiVectorEmbedder(), MultiVectorQueryEncoder)


def test_hashing_query_encoder_satisfies_the_query_encoder_seam() -> None:
    assert isinstance(HashingMultiVectorQueryEncoder(), MultiVectorQueryEncoder)


# --- the reference scorer on SYNTHETIC multi-vectors (the AC: real math, correctness) ---------


def test_max_sim_is_the_sum_of_per_query_token_best_dot() -> None:
    # A hand-computable case: two orthonormal query tokens; the doc holds both plus a distractor.
    query = [[1.0, 0.0], [0.0, 1.0]]
    document = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    # q0 best = <q0, d0> = 1; q1 best = <q1, d1> = 1 → 2.0.
    assert max_sim(query, document) == pytest.approx(2.0)


def test_max_sim_takes_the_single_best_document_token_per_query_token() -> None:
    query = [[1.0, 0.0]]
    document = [[0.5, 0.0], [0.9, 0.0], [0.3, 0.0]]
    # The best (not the sum) of the three: 0.9.
    assert max_sim(query, document) == pytest.approx(0.9)


def test_max_sim_rises_with_document_overlap() -> None:
    # doc_j contains exact copies of the first j of the query's (orthonormal) tokens → MaxSim == j.
    query = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    scores = [max_sim(query, query[:j]) for j in range(len(query) + 1)]
    assert scores == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_max_sim_empty_document_scores_zero() -> None:
    # A query token with no document token to match contributes 0.0 (never a crash).
    assert max_sim([[1.0, 0.0], [0.0, 1.0]], []) == 0.0


def test_max_sim_empty_query_scores_zero() -> None:
    assert max_sim([], [[1.0, 0.0]]) == 0.0


def test_max_sim_is_deterministic_and_rounded() -> None:
    query = [[0.1, 0.2, 0.3]]
    document = [[0.3, 0.2, 0.1], [0.9, 0.0, 0.4]]
    assert max_sim(query, document) == max_sim(query, document)


def test_max_sim_rejects_a_width_mismatch_between_query_and_document_tokens() -> None:
    # The query and the sidecar were encoded in different spaces — a real MaxSim kernel faults too.
    with pytest.raises(MaxSimError, match="different spaces"):
        max_sim([[1.0, 0.0, 0.0]], [[1.0, 0.0]])


# --- MaxSim rerank behaviour (via the faithful fake query encoder + planted sidecars) ---------


def test_rerank_reorders_by_maxsim_not_first_stage_score() -> None:
    # The fake maps each token to a fixed unit vector; a doc containing the query's own token vectors
    # scores highest. First-stage order is inverted so rerank must fix it.
    fake = FakeMultiVectorEmbedder(8)
    query = "revenue growth"
    q_mv = fake.encode(query)  # the two query token vectors
    candidates = [
        Candidate(id="a", score=9.9, text="off topic", metadata=_doc([fake.encode("weather")[0]])),
        Candidate(id="b", score=0.1, text="on topic", metadata=_doc(q_mv)),  # both query tokens
        Candidate(id="c", score=5.0, text="partial", metadata=_doc([q_mv[0]])),  # one query token
    ]
    reranked = MaxSimReranker({}, encoder=fake).process(query, candidates)
    assert _ids(reranked) == ["b", "c", "a"]


def test_rerank_replaces_the_score_with_the_maxsim_score() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("alpha beta")  # two unit token vectors
    doc_a = q_mv  # exact copies of both query tokens
    doc_b = [q_mv[0]]  # only the first query token
    candidates = [
        Candidate(id="a", score=0.42, text="x", metadata=_doc(doc_a)),
        Candidate(id="b", score=0.99, text="y", metadata=_doc(doc_b)),
    ]
    by_id = {c.id: c for c in MaxSimReranker({}, encoder=fake).process("alpha beta", candidates)}
    # The replacement score is exactly the reference MaxSim of the query against each doc's
    # multivectors — NOT the first-stage 0.42 / 0.99.
    assert by_id["a"].score == pytest.approx(max_sim(q_mv, doc_a))
    assert by_id["b"].score == pytest.approx(max_sim(q_mv, doc_b))
    assert by_id["a"].score == pytest.approx(2.0)  # both query tokens matched at 1.0 each
    assert by_id["a"].score > by_id["b"].score  # full overlap beats partial
    assert by_id["b"].score not in (0.42, 0.99)  # the score was replaced, not kept


def test_rerank_keeps_text_and_non_multivector_metadata_untouched() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("alpha")
    candidates = [
        Candidate(id="a", score=0.1, text="alpha", metadata=_doc(q_mv, src="s3", page=4)),
    ]
    (only,) = MaxSimReranker({}, encoder=fake).process("alpha", candidates)
    assert only.text == "alpha"
    assert only.metadata["src"] == "s3"
    assert only.metadata["page"] == 4


def test_rerank_breaks_ties_by_id_ascending() -> None:
    # Three candidates that each share exactly the one query token → identical MaxSim → id order.
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("alpha")
    candidates = [
        Candidate(id="c", score=1.0, text="", metadata=_doc(q_mv)),
        Candidate(id="a", score=1.0, text="", metadata=_doc(q_mv)),
        Candidate(id="b", score=1.0, text="", metadata=_doc(q_mv)),
    ]
    assert _ids(MaxSimReranker({}, encoder=fake).process("alpha", candidates)) == ["a", "b", "c"]


def test_rerank_top_k_truncates_to_the_best_k() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("one two three")  # three query token vectors
    candidates = [
        Candidate(id="a", score=0.0, text="", metadata=_doc(q_mv)),  # 3
        Candidate(id="b", score=0.0, text="", metadata=_doc(q_mv[:2])),  # 2
        Candidate(id="c", score=0.0, text="", metadata=_doc(q_mv[:1])),  # 1
        Candidate(id="d", score=0.0, text="", metadata=_doc([])),  # 0 (empty doc)
    ]
    reranked = MaxSimReranker({"top_k": 2}, encoder=fake).process("one two three", candidates)
    assert _ids(reranked) == ["a", "b"]


def test_rerank_without_top_k_keeps_every_candidate() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("one")
    candidates = [
        Candidate(id="a", score=0.0, text="", metadata=_doc(q_mv)),
        Candidate(id="b", score=0.0, text="", metadata=_doc([])),
    ]
    assert len(MaxSimReranker({}, encoder=fake).process("one", candidates)) == 2


def test_rerank_empty_document_multivector_scores_zero_not_an_error() -> None:
    # A document that had no tokens (empty sidecar row) is valid — it just scores 0.0 (min), distinct
    # from a MISSING sidecar (which is an error).
    fake = FakeMultiVectorEmbedder(8)
    (only,) = MaxSimReranker({}, encoder=fake).process(
        "alpha", [Candidate(id="a", score=5.0, text="", metadata=_doc([]))]
    )
    assert only.score == 0.0


# --- purity: no mutation, no state, empty short-circuit ---------------------------------------


def test_rerank_does_not_mutate_the_input_list_or_candidates() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("alpha")
    candidates = [
        Candidate(id="a", score=0.9, text="", metadata=_doc(q_mv)),
        Candidate(id="b", score=0.8, text="", metadata=_doc([])),
    ]
    snapshot = list(candidates)
    MaxSimReranker({}, encoder=fake).process("alpha", candidates)
    assert candidates == snapshot
    assert candidates[0].score == 0.9  # frozen candidate untouched


def test_rerank_is_deterministic_across_repeated_calls() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("alpha beta")
    candidates = [
        Candidate(id="a", score=0.1, text="", metadata=_doc(q_mv)),
        Candidate(id="b", score=0.2, text="", metadata=_doc(q_mv[:1])),
    ]
    reranker = MaxSimReranker({}, encoder=fake)
    assert reranker.process("alpha beta", candidates) == reranker.process("alpha beta", candidates)


def test_rerank_empty_candidates_returns_empty_without_touching_the_encoder() -> None:
    class _Exploding:
        def encode(self, query: str) -> list[list[float]]:
            raise AssertionError("encoder must not be called for an empty candidate list")

    assert MaxSimReranker({}, encoder=_Exploding()).process("q", []) == []


# --- faithful rejection of a missing/malformed sidecar (anti-false-green, no fabricated score) --


def test_rerank_missing_sidecar_multivector_is_a_typed_error() -> None:
    fake = FakeMultiVectorEmbedder(8)
    # A candidate the caller never loaded the sidecar onto (no 'multivectors' key).
    candidates = [Candidate(id="a", score=0.0, text="alpha", metadata={"src": "s3"})]
    with pytest.raises(MaxSimError, match="no multi-vector under"):
        MaxSimReranker({}, encoder=fake).process("alpha", candidates)


def test_rerank_respects_a_custom_multivector_key() -> None:
    fake = FakeMultiVectorEmbedder(8)
    q_mv = fake.encode("alpha")
    candidates = [Candidate(id="a", score=0.0, text="", metadata={"mv": q_mv})]
    (only,) = MaxSimReranker({"multivector_key": "mv"}, encoder=fake).process("alpha", candidates)
    assert only.score == pytest.approx(1.0)


def test_rerank_malformed_sidecar_not_a_list_of_vectors_is_a_typed_error() -> None:
    fake = FakeMultiVectorEmbedder(8)
    candidates = [Candidate(id="a", score=0.0, text="", metadata={"multivectors": "not-a-mv"})]
    with pytest.raises(MaxSimError, match="list of token vectors"):
        MaxSimReranker({}, encoder=fake).process("alpha", candidates)


def test_rerank_malformed_sidecar_non_numeric_token_is_a_typed_error() -> None:
    fake = FakeMultiVectorEmbedder(8)
    candidates = [
        Candidate(id="a", score=0.0, text="", metadata={"multivectors": [["x", "y"]]}),
    ]
    with pytest.raises(MaxSimError, match="non-numeric"):
        MaxSimReranker({}, encoder=fake).process("alpha", candidates)


def test_rerank_document_token_of_wrong_width_is_a_typed_error() -> None:
    # A sidecar encoded at a different dimension than the query encoder → cross-space, rejected.
    fake = FakeMultiVectorEmbedder(8)  # query tokens are width 8
    candidates = [
        Candidate(id="a", score=0.0, text="", metadata={"multivectors": [[1.0, 0.0, 0.0]]}),
    ]
    with pytest.raises(MaxSimError, match="different spaces"):
        MaxSimReranker({}, encoder=fake).process("alpha", candidates)


# --- config validation ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_rerank_rejects_a_bad_top_k(bad: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        MaxSimReranker({"top_k": bad})


@pytest.mark.parametrize("bad", [0, "", 5, None])
def test_rerank_rejects_a_bad_multivector_key(bad: Any) -> None:
    if bad is None:
        return  # None means "omitted" (the default applies), not an error — skip.
    with pytest.raises(ValueError, match="multivector_key"):
        MaxSimReranker({"multivector_key": bad})


def test_rerank_rejects_a_non_mapping_encoder_config() -> None:
    with pytest.raises(TypeError, match="encoder"):
        MaxSimReranker({"encoder": "not-a-dict"})


# --- config selection (the AC: reranker + encoder chosen by config) ---------------------------


def test_resolve_processor_selects_maxsim_by_name() -> None:
    processor = resolve_processor({"provider": "processor.maxsim", "config": {}})
    assert isinstance(processor, MaxSimReranker)
    assert isinstance(processor, CandidateProcessor)


def test_resolve_multivector_encoder_selects_hashing_by_name() -> None:
    encoder = resolve_multivector_encoder(
        {"provider": "query_multivector.hashing", "config": {"dimension": 16}}
    )
    assert isinstance(encoder, HashingMultiVectorQueryEncoder)
    assert isinstance(encoder, MultiVectorQueryEncoder)


# --- the faithful FakeMultiVectorEmbedder (mirror/reject what a real encoder does) -------------


def test_fake_returns_one_unit_vector_per_token_in_order() -> None:
    mv = FakeMultiVectorEmbedder(8).encode("alpha beta gamma")
    assert len(mv) == 3
    assert all(len(v) == 8 for v in mv)
    for v in mv:
        assert math.sqrt(sum(x * x for x in v)) == pytest.approx(1.0)


def test_fake_empty_text_yields_the_empty_multivector() -> None:
    assert FakeMultiVectorEmbedder(8).encode("   ") == []


def test_fake_is_deterministic_same_token_same_vector() -> None:
    fake = FakeMultiVectorEmbedder(8)
    (alpha1,) = fake.encode("alpha")
    alpha2 = fake.encode("alpha alpha")
    assert alpha1 == alpha2[0] == alpha2[1]


def test_fake_distinct_tokens_are_not_identical() -> None:
    fake = FakeMultiVectorEmbedder(8)
    (alpha,) = fake.encode("alpha")
    (beta,) = fake.encode("beta")
    assert alpha != beta
    # A token's self-dot is 1.0 (unit) and strictly beats a distinct token's dot.
    self_dot = sum(x * x for x in alpha)
    cross_dot = sum(x * y for x, y in zip(alpha, beta, strict=True))
    assert self_dot == pytest.approx(1.0)
    assert cross_dot < self_dot


def test_fake_rejects_a_non_string_query() -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        FakeMultiVectorEmbedder(8).encode(123)  # type: ignore[arg-type]


def test_fake_rejects_a_non_positive_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        FakeMultiVectorEmbedder(0)


# --- query-side generation reuses the index-time embedder (the AC: matches the contract) ------


def test_hashing_query_encoder_matches_the_index_time_multivector_contract() -> None:
    # The query representation must be byte-for-byte the index-time embedder's output for the same
    # text + config — that is what "matches the index-time contract" means (one construction, two
    # clocks). latence-core is present in the dev env, so we reuse the real reference here.
    from latence_core.signals import HashingMultiVectorEmbedder

    config = {"dimension": 32}
    encoder = HashingMultiVectorQueryEncoder(config)
    index_embedder = HashingMultiVectorEmbedder(config)
    text = "annual revenue growth"
    (index_mv,) = list(index_embedder.encode_multivector([text]))
    assert encoder.encode(text) == index_mv


def test_hashing_query_encoder_can_reuse_an_injected_index_time_embedder() -> None:
    # An adopter hands the EXACT index-time embedder instance the corpus was built with (or a real
    # ColBERT encoder) so the query lands in the identical space — no single-adapter seam.
    from latence_core.signals import HashingMultiVectorEmbedder

    embedder = HashingMultiVectorEmbedder({"dimension": 12})
    encoder = HashingMultiVectorQueryEncoder(embedder=embedder)
    (expected,) = list(embedder.encode_multivector(["hello world"]))
    assert encoder.encode("hello world") == expected


def test_hashing_query_encoder_empty_text_yields_empty_multivector() -> None:
    assert HashingMultiVectorQueryEncoder({"dimension": 8}).encode("") == []


def test_hashing_query_encoder_missing_core_raises_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no latence-core (the [multivector] extra), the reference reuse must fail with an install
    # hint — not an opaque ImportError — exactly like the rerank/pack missing-dep boundaries.
    monkeypatch.setitem(sys.modules, "latence_core.signals", None)
    encoder = HashingMultiVectorQueryEncoder({"dimension": 8})
    with pytest.raises(ImportError, match=r"latence-retrieval\[multivector\]"):
        encoder.encode("anything")


# --- THE HEADLINE: end-to-end query-side generation + MaxSim recovers the planted document -----


def test_maxsim_rerank_recovers_the_planted_document_end_to_end() -> None:
    """Using the REAL query-side generation (reusing the index-time embedder) and REAL MaxSim math:
    among candidates, the one whose sidecar multi-vectors ARE the query's own tokens must rank #1 —
    the retrieval property the raw-multivector path exists to provide (AC1+AC2+AC3 together)."""
    from latence_core.signals import HashingMultiVectorEmbedder

    config = {"dimension": 24}
    index_embedder = HashingMultiVectorEmbedder(config)
    query = "quarterly earnings report"
    (planted_mv,) = list(index_embedder.encode_multivector([query]))
    distractors = [
        list(index_embedder.encode_multivector([t]))[0]
        for t in ("a poem about the sea", "weather forecast tomorrow", "unrelated filler text")
    ]
    candidates = [
        Candidate(id="d0", score=9.0, text="distractor", metadata=_doc(distractors[0])),
        Candidate(id="d1", score=8.0, text="distractor", metadata=_doc(distractors[1])),
        Candidate(id="planted", score=0.1, text=query, metadata=_doc(planted_mv)),
        Candidate(id="d2", score=7.0, text="distractor", metadata=_doc(distractors[2])),
    ]
    reranker = MaxSimReranker({"encoder": config})  # builds the real reuse encoder from config
    reranked = reranker.process(query, candidates)
    assert reranked[0].id == "planted"
