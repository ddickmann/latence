"""BM25 term-statistics + the Tokenizer seam, tested THROUGH the interface (#164).

The two reference Tokenizers and the :class:`~latence_core.bm25.Bm25Accumulator` exercised through
their public surfaces — ``tokenize`` / ``add`` / ``add_batch`` / ``stats`` — asserting observable
behaviour (the term statistics, the canonical posting invariants, the two tokenisations genuinely
differing, determinism) rather than internal state. These survive a refactor of the fold.
"""

from __future__ import annotations

import math

import pytest
from latence_core.bm25 import (
    BM25_ARTIFACT,
    BM25_ARTIFACT_VERSION,
    Bm25Accumulator,
    Bm25Posting,
    Bm25Stats,
    RegexWordTokenizer,
    WhitespaceTokenizer,
)
from latence_core.capability import Tokenizer

# ---------------------------------------------------------------------------
# The reference Tokenizers satisfy the seam and are genuinely different
# ---------------------------------------------------------------------------


def test_references_satisfy_the_tokenizer_protocol() -> None:
    assert isinstance(RegexWordTokenizer({}), Tokenizer)
    assert isinstance(WhitespaceTokenizer({}), Tokenizer)


def test_regex_tokenizer_lowercases_and_drops_punctuation() -> None:
    [toks] = list(RegexWordTokenizer({}).tokenize(["Alpha, BETA alpha!"]))
    assert toks == ["alpha", "beta", "alpha"]


def test_whitespace_tokenizer_keeps_punctuation_attached() -> None:
    [toks] = list(WhitespaceTokenizer({}).tokenize(["Alpha, BETA alpha!"]))
    assert toks == ["alpha,", "beta", "alpha!"]


def test_the_two_tokenizers_produce_different_vocabularies() -> None:
    """The second adapter is not a clone: the same text yields different terms (a real seam)."""
    text = "foo, foo bar."
    regex = list(RegexWordTokenizer({}).tokenize([text]))[0]
    ws = list(WhitespaceTokenizer({}).tokenize([text]))[0]
    assert regex == ["foo", "foo", "bar"]
    assert ws == ["foo,", "foo", "bar."]
    assert set(regex) != set(ws)


def test_tokenizers_empty_and_whitespace_only_are_empty() -> None:
    assert list(RegexWordTokenizer({}).tokenize(["", "   ", ".,;"])) == [[], [], []]
    assert list(WhitespaceTokenizer({}).tokenize(["", "   "])) == [[], []]


# ---------------------------------------------------------------------------
# Bm25Posting canonical invariants (anti-false-green: a malformed posting cannot exist)
# ---------------------------------------------------------------------------


def test_posting_rejects_unsorted_or_mismatched_or_bad_len() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        Bm25Posting("d", 2, ["a", "b"], [1])
    with pytest.raises(ValueError, match="strictly ascending"):
        Bm25Posting("d", 2, ["b", "a"], [1, 1])
    with pytest.raises(ValueError, match=">= 1"):
        Bm25Posting("d", 1, ["a", "b"], [1, 0])
    with pytest.raises(ValueError, match="doc_len"):
        Bm25Posting("d", 5, ["a"], [1])


# ---------------------------------------------------------------------------
# The accumulator's postings + corpus statistics
# ---------------------------------------------------------------------------


def _regex_acc() -> Bm25Accumulator:
    return Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")


def test_posting_counts_term_frequency_and_doc_len() -> None:
    posting = _regex_acc().add("d1", "alpha beta alpha gamma")
    assert posting.record_id == "d1"
    assert posting.doc_len == 4
    assert posting.terms == ["alpha", "beta", "gamma"]  # canonical ascending
    assert posting.term_frequencies == [2, 1, 1]  # aligned with terms


def test_empty_document_is_counted_with_an_empty_posting() -> None:
    acc = _regex_acc()
    posting = acc.add("d1", "")
    assert posting.doc_len == 0 and posting.terms == [] and posting.term_frequencies == []
    # It still counts as a document (avgdl denominator), just contributes no tokens/terms.
    assert acc.stats().doc_count == 1 and acc.stats().total_tokens == 0


def test_document_frequency_counts_documents_not_occurrences() -> None:
    """df(term) is the number of documents containing it — once per doc, not per occurrence."""
    acc = _regex_acc()
    acc.add_batch([("d1", "alpha alpha alpha"), ("d2", "alpha beta"), ("d3", "gamma")])
    stats = acc.stats()
    assert stats.document_frequencies["alpha"] == 2  # d1 + d2 (three occurrences in d1 still = 1)
    assert stats.document_frequencies["beta"] == 1
    assert stats.document_frequencies["gamma"] == 1
    assert stats.doc_count == 3


def test_avgdl_is_mean_document_length_in_tokens() -> None:
    acc = _regex_acc()
    acc.add_batch([("d1", "a b c d"), ("d2", "a b")])  # lengths 4 and 2
    stats = acc.stats()
    assert stats.total_tokens == 6
    assert stats.avgdl == 3.0


def test_empty_corpus_stats_are_valid_and_zero() -> None:
    stats = _regex_acc().stats()
    assert stats.doc_count == 0 and stats.total_tokens == 0 and stats.avgdl == 0.0
    assert stats.to_artifact()["terms"] == []


def test_idf_matches_the_robertson_lucene_formula_and_is_nonnegative() -> None:
    acc = _regex_acc()
    # "alpha" in 2 of 3 docs, "gamma" in 1 of 3.
    acc.add_batch([("d1", "alpha"), ("d2", "alpha"), ("d3", "gamma")])
    stats = acc.stats()
    n, df_alpha = 3, 2
    expected = round(math.log(1.0 + (n - df_alpha + 0.5) / (df_alpha + 0.5)), 8)
    assert stats.idf("alpha") == expected
    # Every term's idf is non-negative (the +1 form) — even a term in every document.
    assert all(stats.idf(t) >= 0.0 for t in stats.document_frequencies)


def test_stats_returns_a_defensive_copy() -> None:
    """A caller cannot corrupt the accumulator's df map through the returned stats."""
    acc = _regex_acc()
    acc.add("d1", "alpha")
    stats = acc.stats()
    stats.document_frequencies["alpha"] = 999
    assert acc.stats().document_frequencies["alpha"] == 1


# ---------------------------------------------------------------------------
# The artifact object: self-describing, versioned, deterministic
# ---------------------------------------------------------------------------


def test_artifact_is_self_describing_and_terms_are_ascending() -> None:
    acc = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    acc.add_batch([("d1", "gamma alpha"), ("d2", "beta alpha")])
    art = acc.stats().to_artifact()
    assert art["artifact"] == BM25_ARTIFACT
    assert art["version"] == BM25_ARTIFACT_VERSION
    assert art["tokenizer"] == "tokenizer.regex"
    assert art["doc_count"] == 2
    terms = [t["term"] for t in art["terms"]]
    assert terms == sorted(terms) == ["alpha", "beta", "gamma"]
    alpha = next(t for t in art["terms"] if t["term"] == "alpha")
    assert alpha["df"] == 2 and "idf" in alpha


def test_artifact_is_deterministic_regardless_of_document_order() -> None:
    """df/idf/avgdl depend on the multiset of docs, not the order — the byte-reproducible bar."""
    a = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    a.add_batch([("d1", "alpha beta"), ("d2", "beta gamma")])
    b = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    b.add_batch([("d2", "beta gamma"), ("d1", "alpha beta")])
    assert a.stats().to_artifact()["terms"] == b.stats().to_artifact()["terms"]


def test_tokenizer_choice_changes_the_statistics() -> None:
    """The whitespace tokenizer yields a different vocabulary → different df, through the seam."""
    regex = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    regex.add("d1", "foo, foo bar.")
    ws = Bm25Accumulator(WhitespaceTokenizer({}), "tokenizer.whitespace")
    ws.add("d1", "foo, foo bar.")
    assert set(regex.stats().document_frequencies) == {"foo", "bar"}
    assert set(ws.stats().document_frequencies) == {"foo,", "foo", "bar."}


def test_stats_value_is_constructible_directly() -> None:
    """Bm25Stats is a plain value: derivable idf/avgdl without an accumulator."""
    stats = Bm25Stats(tokenizer="t", doc_count=2, total_tokens=6, document_frequencies={"a": 1})
    assert stats.avgdl == 3.0
    assert stats.idf("a") == round(math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5)), 8)


# ---------------------------------------------------------------------------
# Multilingual corpora (R5-wizard_profile-13) — the blessed enterprise-sota
# profile emits `tokenizer.regex` over an explicitly multilingual stack, so the
# reference tokenisation must not be ASCII-only.
# ---------------------------------------------------------------------------

# The mixed corpus the enterprise-sota profile targets: a German chunk with umlauts/eszett, a
# Chinese chunk (customs vocabulary), and a plain-ASCII English chunk as the control.
_MULTILINGUAL_CORPUS: tuple[tuple[str, str], ...] = (
    ("c1", "Müller GmbH Ausfuhr für Größe Zollgüter Straße"),
    ("c2", "关税 分类 编码 出口 管制"),
    ("c3", "Export control classification code"),
)


def test_regex_tokenizer_preserves_non_ascii_scripts() -> None:
    """`tokenizer.regex` keeps non-ASCII letters INSIDE the term, never as a word boundary.

    Pre-fix (`[A-Za-z0-9]+`) 'Müller' shattered into 'm' + 'ller', 'für' into 'f' + 'r' and
    'Größe' into 'gr' + 'e' — junk single-letter terms polluting the corpus IDF.
    """
    [german] = list(RegexWordTokenizer({}).tokenize(["Müller GmbH Ausfuhr für Größe, Straße!"]))
    assert german == ["müller", "gmbh", "ausfuhr", "für", "größe", "straße"]
    [cyrillic] = list(RegexWordTokenizer({}).tokenize(["Таможенный кодекс 2026"]))
    assert cyrillic == ["таможенный", "кодекс", "2026"]
    # Punctuation is STILL a boundary and STILL dropped (the adapter's identity is unchanged),
    # and the underscore stays a boundary too — it is not an alphanumeric.
    [mixed] = list(RegexWordTokenizer({}).tokenize(["größe_wert, für."]))
    assert mixed == ["größe", "wert", "für"]


def test_regex_tokenizer_is_byte_identical_on_ascii() -> None:
    """The ASCII tokenisation is UNCHANGED — no existing artifact or golden fixture moves."""
    [toks] = list(RegexWordTokenizer({}).tokenize(["Alpha, BETA alpha! v2 foo_bar"]))
    assert toks == ["alpha", "beta", "alpha", "v2", "foo", "bar"]


def test_multilingual_corpus_does_not_produce_empty_or_shattered_documents() -> None:
    """The blessed export block over a mixed corpus: no dead document, no junk single-letter terms.

    Pre-fix the Chinese chunk indexed as `doc_len == 0` with an EMPTY postings row — contributing
    nothing to bm25-postings.parquet while still dragging `avgdl` down for every other document.
    """
    acc = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    postings = acc.add_batch(list(_MULTILINGUAL_CORPUS))
    by_id = {p.record_id: p for p in postings}

    # Every document contributes real terms — none is a dead, empty posting.
    assert all(p.doc_len > 0 for p in postings), {p.record_id: p.doc_len for p in postings}
    assert by_id["c2"].terms == sorted(["关税", "分类", "出口", "管制", "编码"])

    vocabulary = set(acc.stats().document_frequencies)
    # The real German terms survive whole...
    assert {"müller", "für", "größe", "zollgüter", "straße"} <= vocabulary
    # ...and none of the pre-fix single-letter shrapnel ('m', 'ller', 'f', 'r', 'gr', 'e') is there.
    assert not [term for term in vocabulary if len(term) == 1], sorted(vocabulary)


def test_zero_length_documents_do_not_skew_the_length_normaliser() -> None:
    """A non-Latin document must carry its true length into `avgdl` (it normalises EVERY score).

    Pre-fix the CJK chunk contributed 0 tokens, so the corpus `avgdl` was measurably wrong and the
    BM25 score of every Latin document in the same corpus shifted with it.
    """
    acc = Bm25Accumulator(RegexWordTokenizer({}), "tokenizer.regex")
    postings = acc.add_batch(list(_MULTILINGUAL_CORPUS))
    by_id = {p.record_id: p for p in postings}
    stats = acc.stats()

    # Each document's TRUE length, per document — the corpus total alone is not enough: pre-fix the
    # German chunk was OVER-counted (7 words → 12 shrapnel tokens) by exactly as much as the
    # Chinese chunk was under-counted (5 words → 0), so the corpus total coincidentally agreed
    # while both documents' length-normalisation was wrong.
    assert by_id["c1"].doc_len == 7  # pre-fix: 12
    assert by_id["c2"].doc_len == 5  # pre-fix: 0
    assert by_id["c3"].doc_len == 4  # ASCII control, unchanged
    assert stats.total_tokens == sum(p.doc_len for p in postings) == 16
    assert stats.avgdl == 16 / 3


def test_export_and_query_tokenisation_agree_on_non_ascii() -> None:
    """The index-side and query-side BM25 term units must agree — or the sparse leg is a no-op.

    `latence-retrieval` is deliberately independent of `latence-core` (separate downstream
    library, no shared constant to import), so the two spellings of the reference pattern can only
    be held together by a test. A query term the index never emits scores nothing, forever.
    """
    from latence_retrieval.bm25 import _tokenize as query_tokenize

    for text in ("Müller GmbH für Größe", "关税 分类 编码", "Taxes 2026, code_v2"):
        [index_terms] = list(RegexWordTokenizer({}).tokenize([text]))
        assert query_tokenize(text) == index_terms, text
