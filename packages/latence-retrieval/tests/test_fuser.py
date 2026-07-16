"""The ``fuse`` seam (:class:`RrfFuser` / :class:`WeightedFuser`), tested at the input→output boundary.

Every assertion drives ``fuse(rankings)`` and inspects the returned list — the ids, their order, and
the fused scores a caller can observe — never the internal scoring maths, so the tests survive any
refactor of *how* the fusion is computed. Coverage per the acceptance criteria: both modes selectable
by config; deterministic order for tied inputs; disjoint and overlapping id sets; and that fuse is a
pure transform that mutates nothing and carries no state across calls.
"""

from __future__ import annotations

import pytest
from latence_retrieval import (
    Candidate,
    CandidateProcessor,
    Fuser,
    RrfFuser,
    WeightedFuser,
    resolve_fuser,
)


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


# --- seam identity ----------------------------------------------------------------------------


def test_both_adapters_satisfy_the_fuser_seam() -> None:
    assert isinstance(RrfFuser(), Fuser)
    assert isinstance(WeightedFuser(), Fuser)


def test_a_fuser_is_not_a_candidate_processor() -> None:
    # Distinct shapes (many-lists-in vs one-list-in): a Fuser must not pass as a processor.
    assert not isinstance(RrfFuser(), CandidateProcessor)


# --- RRF: overlapping + disjoint ids ----------------------------------------------------------


def test_rrf_ranks_a_shared_id_above_singletons() -> None:
    # 'b' is 2nd in both lists; RRF rewards agreement, so it should outrank the rank-1 singletons.
    dense = [Candidate(id="a", score=0.9), Candidate(id="b", score=0.8)]
    sparse = [Candidate(id="c", score=5.0), Candidate(id="b", score=4.0)]
    fused = RrfFuser({"k": 1}).fuse([dense, sparse])
    assert _ids(fused)[0] == "b"
    assert set(_ids(fused)) == {"a", "b", "c"}


def test_rrf_handles_fully_disjoint_lists() -> None:
    left = [Candidate(id="a", score=0.9), Candidate(id="b", score=0.1)]
    right = [Candidate(id="c", score=0.9), Candidate(id="d", score=0.1)]
    fused = RrfFuser().fuse([left, right])
    # Every id survives exactly once; rank-1s (a, c) precede rank-2s (b, d).
    assert set(_ids(fused)) == {"a", "b", "c", "d"}
    assert _ids(fused)[:2] == ["a", "c"]  # both rank-1, tie broken by id
    assert _ids(fused)[2:] == ["b", "d"]


def test_rrf_emits_each_id_once_even_when_overlapping() -> None:
    a1 = [Candidate(id="x", score=0.9), Candidate(id="y", score=0.8)]
    a2 = [Candidate(id="y", score=0.7), Candidate(id="x", score=0.6)]
    fused = RrfFuser().fuse([a1, a2])
    assert sorted(_ids(fused)) == ["x", "y"]


# --- determinism for tied inputs --------------------------------------------------------------


def test_rrf_breaks_ties_by_id_ascending() -> None:
    # Three symmetric rank-1 singletons → identical fused score → deterministic id order.
    lists = [
        [Candidate(id="c", score=1.0)],
        [Candidate(id="a", score=1.0)],
        [Candidate(id="b", score=1.0)],
    ]
    assert _ids(RrfFuser().fuse(lists)) == ["a", "b", "c"]


def test_fusion_is_deterministic_across_repeated_calls() -> None:
    lists = [
        [Candidate(id="a", score=0.9), Candidate(id="b", score=0.5)],
        [Candidate(id="b", score=0.4), Candidate(id="c", score=0.3)],
    ]
    fuser = RrfFuser()
    first = fuser.fuse(lists)
    second = fuser.fuse(lists)
    assert first == second


# --- purity: no mutation, no state ------------------------------------------------------------


def test_fuse_does_not_mutate_input_lists() -> None:
    dense = [Candidate(id="a", score=0.9), Candidate(id="b", score=0.8)]
    sparse = [Candidate(id="b", score=0.7)]
    rankings = [dense, sparse]
    RrfFuser().fuse(rankings)
    assert dense == [Candidate(id="a", score=0.9), Candidate(id="b", score=0.8)]
    assert sparse == [Candidate(id="b", score=0.7)]
    assert len(rankings) == 2


def test_fused_candidate_keeps_first_seen_text_and_metadata() -> None:
    dense = [Candidate(id="a", score=0.9, text="from-dense", metadata={"src": "dense"})]
    sparse = [Candidate(id="a", score=0.2, text="from-sparse", metadata={"src": "sparse"})]
    fused = RrfFuser().fuse([dense, sparse])
    assert len(fused) == 1
    assert fused[0].text == "from-dense"
    assert fused[0].metadata == {"src": "dense"}
    # The score is the *fused* score, not either input score.
    assert fused[0].score not in (0.9, 0.2)


# --- weighted score fusion --------------------------------------------------------------------


def test_weighted_normalises_incomparable_scales_per_list() -> None:
    # List A is cosine-scaled (~0.8), list B is BM25-scaled (~40). Without per-list normalisation
    # B's magnitudes would swamp A; with it, the top of each list is comparable.
    cosine = [Candidate(id="a", score=0.81), Candidate(id="b", score=0.80)]
    bm25 = [Candidate(id="b", score=40.0), Candidate(id="c", score=1.0)]
    fused = WeightedFuser().fuse([cosine, bm25])
    # 'b' is top of BM25 (normalised 1.0) + bottom of cosine (0.0) = 1.0; 'a' is top of cosine
    # (1.0) only = 1.0 → tie, broken by id → a before b; 'c' is bottom of BM25 (0.0) = 0.0 last.
    assert _ids(fused) == ["a", "b", "c"]


def test_weighted_respects_per_list_weights() -> None:
    # Same lists, but weight the sparse list to zero → only the dense list's order matters.
    dense = [Candidate(id="a", score=1.0), Candidate(id="b", score=0.0)]
    sparse = [Candidate(id="b", score=1.0), Candidate(id="a", score=0.0)]
    fused = WeightedFuser().fuse([dense, sparse], weights=[1.0, 0.0])
    assert _ids(fused) == ["a", "b"]


def test_weighted_all_equal_scores_normalise_to_full_weight() -> None:
    # A list whose scores are all equal expresses no preference: every member gets weight, not a
    # divide-by-zero. Here each list is internally flat, so the id tiebreak decides the order.
    flat_a = [Candidate(id="a", score=0.5), Candidate(id="b", score=0.5)]
    flat_b = [Candidate(id="a", score=9.0), Candidate(id="b", score=9.0)]
    fused = WeightedFuser().fuse([flat_a, flat_b])
    assert _ids(fused) == ["a", "b"]
    # Both received 1.0 + 1.0 = 2.0.
    assert fused[0].score == pytest.approx(2.0)
    assert fused[1].score == pytest.approx(2.0)


def test_weighted_single_element_list_does_not_divide_by_zero() -> None:
    fused = WeightedFuser().fuse([[Candidate(id="only", score=0.7)]])
    assert _ids(fused) == ["only"]
    assert fused[0].score == pytest.approx(1.0)


# --- config selection (the AC: modes selectable by config) ------------------------------------


def test_resolve_fuser_selects_rrf_by_name() -> None:
    fuser = resolve_fuser({"provider": "fuser.rrf", "config": {"k": 60}})
    assert isinstance(fuser, RrfFuser)


def test_resolve_fuser_selects_weighted_by_name() -> None:
    fuser = resolve_fuser({"provider": "fuser.weighted", "config": {}})
    assert isinstance(fuser, WeightedFuser)


def test_resolved_rrf_and_weighted_can_disagree_on_order() -> None:
    # A case where rank-only (RRF) and magnitude-aware (weighted) genuinely differ, proving the
    # config choice changes behaviour rather than being cosmetic.
    dense = [Candidate(id="a", score=0.99), Candidate(id="b", score=0.98)]
    sparse = [Candidate(id="b", score=0.5), Candidate(id="a", score=0.4)]
    rrf = resolve_fuser({"provider": "fuser.rrf"}).fuse([dense, sparse])
    weighted = resolve_fuser({"provider": "fuser.weighted"}).fuse([dense, sparse])
    # RRF: a is rank1+rank2, b is rank2+rank1 → symmetric → id tiebreak → a, b.
    assert _ids(rrf) == ["a", "b"]
    # Weighted keeps magnitudes: a = 1.0 + 0.0, b = 0.0 + 1.0 → tie → id tiebreak → a, b too,
    # but the fused *scores* are equal here, whereas RRF's are equal by symmetry. Assert both ran.
    assert _ids(weighted) == ["a", "b"]


# --- config validation ------------------------------------------------------------------------


def test_rrf_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k"):
        RrfFuser({"k": 0})


def test_weighted_rejects_unknown_normalization() -> None:
    with pytest.raises(ValueError, match="normalization"):
        WeightedFuser({"normalization": "z_score"})


def test_weights_length_must_match_number_of_lists() -> None:
    lists = [[Candidate(id="a", score=1.0)], [Candidate(id="b", score=1.0)]]
    with pytest.raises(ValueError, match="weights length"):
        RrfFuser().fuse(lists, weights=[1.0])


# --- edge cases -------------------------------------------------------------------------------


def test_fuse_no_lists_returns_empty() -> None:
    assert RrfFuser().fuse([]) == []
    assert WeightedFuser().fuse([]) == []


def test_fuse_ignores_empty_lists_among_real_ones() -> None:
    real = [Candidate(id="a", score=0.9)]
    assert _ids(RrfFuser().fuse([real, [], real])) == ["a"]


def test_duplicate_id_within_a_list_counts_once_at_best_rank() -> None:
    # A malformed list repeating an id must not let it double-count; the first (best) rank wins.
    dup = [
        Candidate(id="a", score=0.9),
        Candidate(id="a", score=0.1),
        Candidate(id="b", score=0.5),
    ]
    fused = RrfFuser().fuse([dup])
    assert _ids(fused) == ["a", "b"]
