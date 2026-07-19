"""The ``bm25`` rescorer (:class:`Bm25Rescorer`), tested at the ``(query, candidates) → list`` seam.

Every assertion is on the *observable* rescored candidate list — order, replaced scores, the effect
of the IDF source (supplied stats vs. the candidate pool), truncation, purity — never the internal
BM25 arithmetic, so the tests survive a refactor of *how* the score is computed. One exact-number
test pins the formula itself (anti-false-green: assert the real BM25 value, not that a knob is set).
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from latence_retrieval import Bm25Rescorer, Candidate, CandidateProcessor, resolve_processor


def _ids(candidates: list[Candidate]) -> list[str]:
    return [c.id for c in candidates]


# --- seam identity ----------------------------------------------------------------------------


def test_bm25_satisfies_the_candidate_processor_seam() -> None:
    assert isinstance(Bm25Rescorer(), CandidateProcessor)


# --- exact BM25 (the formula, pinned to a hand-derived number) --------------------------------


def test_single_term_score_reduces_to_idf_when_length_equals_average() -> None:
    # One candidate 'z' (dl=1) against supplied stats with avg_doc_len=1 → the length norm is 1 and
    # the tf=1 saturation is (k1+1)/(1+k1)=1, so the score collapses to exactly the BM25 IDF —
    # a real, checkable number, not a proxy.
    rescorer = Bm25Rescorer(
        {"stats": {"doc_count": 10, "avg_doc_len": 1.0, "doc_freq": {"z": 1}}}
    )
    (scored,) = rescorer.process("z", [Candidate(id="a", score=0.0, text="z")])
    expected_idf = math.log(1.0 + (10 - 1 + 0.5) / (1 + 0.5))
    assert scored.score == pytest.approx(expected_idf)
    assert scored.score == pytest.approx(1.9924301646902063)


# --- rescoring behaviour ----------------------------------------------------------------------


def test_bm25_reorders_by_lexical_relevance_not_first_stage_score() -> None:
    query = "quarterly revenue"
    candidates = [
        Candidate(id="a", score=9.9, text="a poem about the sea and the sky"),
        Candidate(id="b", score=0.1, text="quarterly revenue rose sharply this quarter"),
        Candidate(id="c", score=5.0, text="the revenue line was flat"),
    ]
    rescored = Bm25Rescorer().process(query, candidates)
    # 'b' carries both query terms (revenue twice), 'c' carries one, 'a' none → strict BM25 order.
    assert _ids(rescored) == ["b", "c", "a"]


def test_a_candidate_without_any_query_term_scores_zero() -> None:
    # Query-term-driven: a document sharing no query term is never scored and keeps exactly 0.0.
    candidates = [
        Candidate(id="hit", score=0.0, text="apple pie"),
        Candidate(id="miss", score=0.0, text="nothing relevant here"),
    ]
    rescored = Bm25Rescorer().process("apple", candidates)
    by_id = {c.id: c for c in rescored}
    assert by_id["miss"].score == 0.0
    assert by_id["hit"].score > 0.0


def test_bm25_keeps_text_and_metadata_untouched() -> None:
    candidates = [Candidate(id="a", score=0.1, text="alpha", metadata={"src": "s3", "page": 4})]
    (only,) = Bm25Rescorer().process("alpha", candidates)
    assert only.text == "alpha"
    assert only.metadata == {"src": "s3", "page": 4}


def test_bm25_breaks_ties_by_id_ascending() -> None:
    # Three symmetric single-term docs → identical BM25 scores → deterministic ascending-id order.
    candidates = [
        Candidate(id="c", score=1.0, text="alpha"),
        Candidate(id="a", score=1.0, text="beta"),
        Candidate(id="b", score=1.0, text="gamma"),
    ]
    rescored = Bm25Rescorer().process("alpha beta gamma", candidates)
    assert _ids(rescored) == ["a", "b", "c"]


def test_bm25_top_k_truncates_to_the_best_k() -> None:
    query = "one two three"
    candidates = [
        Candidate(id="a", score=0.0, text="one two three"),
        Candidate(id="b", score=0.0, text="one two"),
        Candidate(id="c", score=0.0, text="one"),
        Candidate(id="d", score=0.0, text="unrelated"),
    ]
    rescored = Bm25Rescorer({"top_k": 2}).process(query, candidates)
    assert _ids(rescored) == ["a", "b"]


def test_empty_query_scores_everything_zero_and_orders_by_id() -> None:
    candidates = [
        Candidate(id="b", score=5.0, text="beta"),
        Candidate(id="a", score=9.0, text="alpha"),
    ]
    rescored = Bm25Rescorer().process("", candidates)
    assert all(c.score == 0.0 for c in rescored)
    assert _ids(rescored) == ["a", "b"]


def test_empty_candidates_returns_empty() -> None:
    assert Bm25Rescorer().process("anything", []) == []


# --- the IDF source is genuinely used (anti-false-green: it changes the ranking, not a flag) ---


def test_supplied_corpus_stats_change_the_ranking_versus_the_pool() -> None:
    # Pool-local IDF ranks the higher-tf 'banana' doc first; but with true corpus stats where
    # 'apple' is globally rare and 'banana' common, the rare term dominates and the order flips.
    candidates = [
        Candidate(id="apple", score=0.0, text="apple"),
        Candidate(id="banana", score=0.0, text="banana banana"),
    ]
    query = "apple banana"

    pool = Bm25Rescorer().process(query, candidates)
    assert _ids(pool) == ["banana", "apple"]  # higher tf wins under pool-local (equal) IDF

    stats = {
        "doc_count": 1000,
        "avg_doc_len": 1.5,  # same avgdl so ONLY the IDF differs between the two runs
        "doc_freq": {"apple": 1, "banana": 900},
    }
    with_stats = Bm25Rescorer({"stats": stats}).process(query, candidates)
    assert _ids(with_stats) == ["apple", "banana"]  # the globally-rare term now dominates


def test_a_query_term_absent_from_supplied_doc_freq_is_treated_as_rare() -> None:
    # A term missing from doc_freq has df=0 → the maximum IDF, exactly as a real BM25 index treats
    # an unseen term (it is not silently dropped).
    stats = {"doc_count": 100, "avg_doc_len": 1.0, "doc_freq": {}}
    (scored,) = Bm25Rescorer({"stats": stats}).process("zebra", [Candidate(id="a", score=0.0, text="zebra")])
    expected_idf = math.log(1.0 + (100 - 0 + 0.5) / (0 + 0.5))
    assert scored.score == pytest.approx(expected_idf)


# --- purity: no mutation, deterministic -------------------------------------------------------


def test_bm25_does_not_mutate_the_input_list_or_candidates() -> None:
    candidates = [
        Candidate(id="a", score=0.9, text="alpha"),
        Candidate(id="b", score=0.8, text="beta"),
    ]
    snapshot = list(candidates)
    Bm25Rescorer().process("alpha", candidates)
    assert candidates == snapshot
    assert candidates[0].score == 0.9  # frozen candidate untouched


def test_bm25_is_deterministic_across_repeated_calls() -> None:
    candidates = [
        Candidate(id="a", score=0.1, text="alpha beta"),
        Candidate(id="b", score=0.2, text="beta"),
    ]
    rescorer = Bm25Rescorer()
    assert rescorer.process("alpha beta", candidates) == rescorer.process("alpha beta", candidates)


# --- config selection (the AC: rescorer chosen by config) -------------------------------------


def test_resolve_processor_selects_bm25_by_name() -> None:
    processor = resolve_processor({"provider": "processor.bm25", "config": {}})
    assert isinstance(processor, Bm25Rescorer)
    assert isinstance(processor, CandidateProcessor)


def test_config_selected_bm25_rescoring_works_end_to_end() -> None:
    rescorer = resolve_processor({"provider": "processor.bm25", "config": {"top_k": 1}})
    candidates = [
        Candidate(id="a", score=9.0, text="unrelated"),
        Candidate(id="b", score=0.0, text="the query terms"),
    ]
    assert _ids(rescorer.process("the query terms", candidates)) == ["b"]


# --- config validation ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1.0, True, "1.5"])
def test_bm25_rejects_a_bad_k1(bad: Any) -> None:
    with pytest.raises(ValueError, match="'k1'"):
        Bm25Rescorer({"k1": bad})


@pytest.mark.parametrize("bad", [-0.1, 1.1, True, "0.75"])
def test_bm25_rejects_a_bad_b(bad: Any) -> None:
    with pytest.raises(ValueError, match="'b'"):
        Bm25Rescorer({"b": bad})


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_bm25_rejects_a_bad_top_k(bad: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        Bm25Rescorer({"top_k": bad})


def test_bm25_rejects_non_mapping_stats() -> None:
    with pytest.raises(TypeError, match="'stats' must be a mapping"):
        Bm25Rescorer({"stats": [1, 2, 3]})


@pytest.mark.parametrize("bad", [0, -1, 1.0, True, "10"])
def test_bm25_rejects_a_bad_doc_count(bad: Any) -> None:
    with pytest.raises(ValueError, match="doc_count"):
        Bm25Rescorer({"stats": {"doc_count": bad, "avg_doc_len": 1.0, "doc_freq": {}}})


@pytest.mark.parametrize("bad", [0, -1.0, True, "1.0"])
def test_bm25_rejects_a_bad_avg_doc_len(bad: Any) -> None:
    with pytest.raises(ValueError, match="avg_doc_len"):
        Bm25Rescorer({"stats": {"doc_count": 5, "avg_doc_len": bad, "doc_freq": {}}})


def test_bm25_rejects_non_mapping_doc_freq() -> None:
    with pytest.raises(TypeError, match="doc_freq"):
        Bm25Rescorer({"stats": {"doc_count": 5, "avg_doc_len": 1.0, "doc_freq": [1]}})


@pytest.mark.parametrize("bad", [-1, 1.0, True, "2"])
def test_bm25_rejects_a_bad_doc_freq_value(bad: Any) -> None:
    with pytest.raises(ValueError, match="doc_freq"):
        Bm25Rescorer({"stats": {"doc_count": 5, "avg_doc_len": 1.0, "doc_freq": {"t": bad}}})


def test_bm25_rejects_a_doc_freq_above_doc_count() -> None:
    # A term cannot occur in more documents than the corpus holds. The classic df-vs-collection-
    # frequency mixup (df=999 with doc_count=5) drives the Lucene/Robertson IDF NEGATIVE, which
    # would silently invert relevance at query time. It must be rejected loudly at construction —
    # not accepted here to produce quietly-wrong scores later.
    with pytest.raises(ValueError, match="doc_freq.*<= 'doc_count'"):
        Bm25Rescorer(
            {"stats": {"doc_count": 5, "avg_doc_len": 10.0, "doc_freq": {"apple": 999}}}
        )


def test_bm25_accepts_a_doc_freq_equal_to_doc_count() -> None:
    # df == doc_count is legitimate (a term in every document); its IDF is the smallest
    # NON-negative value, not an error — the boundary must stay open so real corpora work.
    rescorer = Bm25Rescorer(
        {"stats": {"doc_count": 5, "avg_doc_len": 1.0, "doc_freq": {"z": 5}}}
    )
    (scored,) = rescorer.process("z", [Candidate(id="a", score=0.0, text="z")])
    assert scored.score >= 0.0


def test_bm25_never_ranks_a_term_bearing_doc_below_a_non_bearing_one_under_supplied_stats() -> None:
    # The end-to-end guarantee the rejection protects: with the offending stats now refused at
    # construction, a corrected (df <= doc_count) wiring must keep the doc that CONTAINS the query
    # term ranked at or above the one that does not — a positive-IDF, non-inverted 'exact BM25'.
    rescorer = Bm25Rescorer(
        {"stats": {"doc_count": 5, "avg_doc_len": 10.0, "doc_freq": {"apple": 1}}}
    )
    ranked = rescorer.process(
        "apple",
        [
            Candidate(id="has", score=0.0, text="apple apple apple"),
            Candidate(id="hasnot", score=0.0, text="banana cherry"),
        ],
    )
    assert _ids(ranked) == ["has", "hasnot"]
    by_id = {c.id: c.score for c in ranked}
    assert by_id["has"] > 0.0
    assert by_id["hasnot"] == 0.0


# --- multilingual queries (R5-wizard_profile-13) -----------------------------------------------


def test_non_ascii_query_is_not_a_dead_no_op() -> None:
    """A Cyrillic/CJK query must actually rescore — pre-fix EVERY candidate scored exactly 0.0.

    The index-side tokenizer the blessed `enterprise-sota` stack emits and this rescorer share one
    reference tokenisation. While it was ASCII-only (`[a-z0-9]+`) a non-Latin query tokenised to
    `[]`, so the BM25 sum was empty for every candidate: the sparse leg of the hybrid fuser was a
    permanent no-op for exactly the multilingual corpora that profile targets.
    """
    rescorer = Bm25Rescorer()
    candidates = [
        Candidate(id="miss", score=0.9, text="Export control classification code"),
        Candidate(id="hit", score=0.1, text="Таможенный кодекс и классификация товаров"),
    ]
    ranked = rescorer.process("таможенный кодекс", candidates)
    assert _ids(ranked) == ["hit", "miss"]
    assert ranked[0].score > 0.0
    assert ranked[1].score == 0.0


def test_accented_terms_are_matched_whole_not_shattered() -> None:
    """'für'/'Größe' must be single terms — pre-fix they shattered into 'f','r','gr','e'.

    The shrapnel both lost the real term and matched unrelated documents that merely contained the
    stray letters, so the ranking inverted.
    """
    rescorer = Bm25Rescorer()
    candidates = [
        Candidate(id="junk", score=0.0, text="e f r m gr ller"),
        Candidate(id="german", score=0.0, text="Müller GmbH Ausfuhr für Größe"),
    ]
    ranked = rescorer.process("für Größe Müller", candidates)
    assert _ids(ranked) == ["german", "junk"]
    assert ranked[0].score > 0.0
    assert ranked[1].score == 0.0


def test_ascii_tokenisation_is_unchanged() -> None:
    """The ASCII term unit is byte-identical to the pre-fix behaviour (punctuation still drops)."""
    from latence_retrieval.bm25 import _tokenize

    assert _tokenize("Alpha, BETA alpha! v2 foo_bar") == [
        "alpha",
        "beta",
        "alpha",
        "v2",
        "foo",
        "bar",
    ]
