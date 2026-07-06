"""Profiling engine (ported/consolidated enricher set, S7): pure-function unit tests.

Covers the per-document feature functions (density, readability, Zipf α, compression,
type-token ratio, averages), the empty/degenerate degradations, and the streaming
CorpusAggregator (source coverage, entity frequency, type consensus, cross-document
co-occurrence, corpus Zipf α). All deterministic and dependency-free.
"""

from __future__ import annotations

from latence_core.profiling import (
    CorpusAggregator,
    MentionView,
    document_features,
    readability_score,
    zipf_alpha,
)

# -- per-document features ----------------------------------------------------


def test_document_features_basic_counts() -> None:
    text = "The cat sat. The dog ran."
    feats = document_features(text)
    assert feats.char_count == len(text)
    # 8 word tokens: the cat sat the dog ran (+ the two "the"), sentences = 2.
    assert feats.word_count == 6
    assert feats.sentence_count == 2
    # distinct lowercased types: the, cat, sat, dog, ran = 5.
    assert feats.unique_word_count == 5
    assert feats.type_token_ratio == round(5 / 6, 6)
    assert 0.0 <= feats.density <= 1.0
    assert 0.0 <= feats.readability <= 1.0
    assert feats.avg_sentence_length == round(6 / 2, 6)


def test_empty_document_is_all_zero_not_error() -> None:
    feats = document_features("")
    assert feats.word_count == 0
    assert feats.sentence_count == 0
    assert feats.unique_word_count == 0
    assert feats.type_token_ratio == 0.0
    assert feats.density == 0.0
    assert feats.readability == 0.0
    assert feats.zipf_alpha == 0.0
    assert feats.compression_ratio == 0.0
    assert feats.word_frequencies == {}


def test_whitespace_only_document_is_clean() -> None:
    feats = document_features("   \n\t  \n")
    assert feats.word_count == 0
    # density is low but defined (all whitespace -> 0.0).
    assert feats.density == 0.0


def test_density_reflects_non_whitespace_fraction() -> None:
    assert document_features("abcd").density == 1.0
    # 2 non-ws of 4 chars.
    assert document_features("a b ").density == 0.5


def test_readability_easy_text_scores_higher_than_hard_text() -> None:
    easy = "The cat sat on the mat. The dog ran to the park."
    hard = (
        "Notwithstanding the aforementioned considerations, the "
        "epistemological ramifications remain fundamentally intractable."
    )
    assert document_features(easy).readability > document_features(hard).readability


def test_readability_score_bounds() -> None:
    assert readability_score(0, 0, 0) == 0.0
    score = readability_score(word_count=100, sentence_count=10, syllable_count=150)
    assert 0.0 <= score <= 1.0


def test_compression_ratio_low_for_repetitive_text() -> None:
    repetitive = "spam spam spam spam spam spam spam spam spam spam " * 20
    diverse = "The quick brown fox jumps over the lazy dog near the riverbank."
    assert document_features(repetitive).compression_ratio < document_features(diverse).compression_ratio


def test_zipf_alpha_positive_for_skewed_distribution() -> None:
    # A strongly skewed frequency distribution fits a positive Zipf slope.
    alpha = zipf_alpha([100, 50, 33, 25, 20, 10, 5, 2, 1])
    assert alpha > 0.0


def test_zipf_alpha_too_few_points_is_zero() -> None:
    assert zipf_alpha([]) == 0.0
    assert zipf_alpha([5]) == 0.0


def test_document_features_deterministic() -> None:
    text = "Alpha beta gamma. Alpha beta. Alpha."
    assert document_features(text) == document_features(text)


# -- streaming corpus aggregation ---------------------------------------------


def test_corpus_aggregator_source_coverage() -> None:
    agg = CorpusAggregator()
    agg.add_document({"the": 2, "cat": 1}, language="en", category="memo", sensitivity="public")
    agg.add_document({"der": 2, "hund": 1}, language="de", category=None, sensitivity="internal")
    agg.add_document({"the": 1}, language="en", category="memo", sensitivity="public")
    result = agg.finalize()
    assert result.document_count == 3
    assert result.language_coverage == {"de": 1, "en": 2}
    assert result.category_coverage == {"memo": 2}  # the None category is not counted
    assert result.sensitivity_coverage == {"internal": 1, "public": 2}
    assert result.total_words == 2 + 1 + 2 + 1 + 1
    assert result.total_unique_words == 4  # the, cat, der, hund


def test_corpus_aggregator_entity_frequency_and_type_frequency() -> None:
    agg = CorpusAggregator()
    agg.add_mentions(
        [
            MentionView(document_id="d1", text="Acme", label="ORG"),
            MentionView(document_id="d1", text="Alice", label="PERSON"),
        ]
    )
    agg.add_mentions([MentionView(document_id="d2", text="Acme", label="ORG")])
    result = agg.finalize()
    assert result.entity_frequency == {"Acme": 2, "Alice": 1}
    assert result.entity_type_frequency == {"ORG": 2, "PERSON": 1}


def test_corpus_aggregator_type_consensus() -> None:
    agg = CorpusAggregator()
    # "Mercury" typed inconsistently: 2x PLANET, 1x ELEMENT -> consensus 2/3.
    agg.add_mentions([MentionView(document_id="d1", text="Mercury", label="PLANET")])
    agg.add_mentions([MentionView(document_id="d2", text="Mercury", label="PLANET")])
    agg.add_mentions([MentionView(document_id="d3", text="Mercury", label="ELEMENT")])
    # "Acme" seen once -> not scored (single mention is trivially 1.0).
    agg.add_mentions([MentionView(document_id="d4", text="Acme", label="ORG")])
    result = agg.finalize()
    assert result.entity_type_consensus == {"Mercury": round(2 / 3, 6)}


def test_corpus_aggregator_cross_document_co_occurrence() -> None:
    agg = CorpusAggregator()
    # Alice + Acme co-occur in d1 and d2 (2 docs); Alice + Bob only in d1 (1 doc).
    agg.add_mentions(
        [
            MentionView(document_id="d1", text="Alice", label="PERSON"),
            MentionView(document_id="d1", text="Acme", label="ORG"),
            MentionView(document_id="d1", text="Bob", label="PERSON"),
        ]
    )
    agg.add_mentions(
        [
            MentionView(document_id="d2", text="Alice", label="PERSON"),
            MentionView(document_id="d2", text="Acme", label="ORG"),
        ]
    )
    result = agg.finalize()
    pairs = {(left, right): count for left, right, count in result.co_occurrences}
    assert pairs[("Acme", "Alice")] == 2
    assert pairs[("Alice", "Bob")] == 1
    assert pairs[("Acme", "Bob")] == 1


def test_co_occurrence_deduplicates_within_document() -> None:
    agg = CorpusAggregator()
    # Alice mentioned twice in the same doc alongside Acme -> the pair counts d1 once.
    agg.add_mentions(
        [
            MentionView(document_id="d1", text="Alice", label="PERSON"),
            MentionView(document_id="d1", text="Alice", label="PERSON"),
            MentionView(document_id="d1", text="Acme", label="ORG"),
        ]
    )
    result = agg.finalize()
    pairs = {(left, right): count for left, right, count in result.co_occurrences}
    assert pairs[("Acme", "Alice")] == 1


def test_co_occurrence_report_is_capped() -> None:
    agg = CorpusAggregator(max_co_occurrences=2)
    agg.add_mentions(
        [MentionView(document_id="d1", text=name, label="X") for name in ("A", "B", "C", "D")]
    )
    result = agg.finalize()
    assert len(result.co_occurrences) == 2


def test_aggregation_is_order_independent() -> None:
    def build(order: list[str]) -> object:
        agg = CorpusAggregator()
        docs = {
            "a": ({"the": 3, "cat": 1}, "en", "memo", "public"),
            "b": ({"der": 2}, "de", None, "internal"),
        }
        for key in order:
            freqs, lang, cat, sens = docs[key]
            agg.add_document(freqs, language=lang, category=cat, sensitivity=sens)
        return agg.finalize()

    assert build(["a", "b"]) == build(["b", "a"])


def test_empty_mentions_is_noop() -> None:
    agg = CorpusAggregator()
    agg.add_mentions([])
    result = agg.finalize()
    assert result.entity_frequency == {}
    assert result.co_occurrences == []
