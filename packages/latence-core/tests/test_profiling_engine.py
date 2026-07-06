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


# -- bounded aggregate state (algorithmic-complexity DoS guards, issue #33) ----


def test_per_document_pair_fanout_is_capped() -> None:
    """A pathological document with many distinct entities cannot build O(n²) pairs.

    Adversarial regression: without the per-document cap, one ``add_mentions`` call over
    N distinct entities builds N*(N-1)/2 transient pairs (a memory-DoS lever with a large
    NER Provider). With ``max_entities_per_document=k`` only k*(k-1)/2 pairs are tracked.
    Would FAIL (pair table would hold 499_500 entries) if the cap were reverted.
    """
    cap = 10
    agg = CorpusAggregator(max_entities_per_document=cap)
    # 1000 distinct entity surfaces in a single document.
    agg.add_mentions(
        [MentionView(document_id="d1", text=f"e{i:04d}", label="X") for i in range(1000)]
    )
    # Only k*(k-1)/2 pairs can exist across the whole (single-document) corpus.
    assert len(agg._pair_docs) == cap * (cap - 1) // 2  # noqa: SLF001
    # Every mention still counted in frequency (only the O(n²) pairing is capped).
    result = agg.finalize()
    assert len(result.entity_frequency) == 1000


def test_per_document_cap_keeps_most_frequent_entities() -> None:
    """The per-document cap deterministically keeps the most-frequent-in-document entities."""
    agg = CorpusAggregator(max_entities_per_document=2)
    # "hot" x3, "warm" x2, "cold" x1 — the cap keeps {cold? no} the top two by frequency.
    mentions = (
        [MentionView(document_id="d1", text="hot", label="X")] * 3
        + [MentionView(document_id="d1", text="warm", label="X")] * 2
        + [MentionView(document_id="d1", text="cold", label="X")] * 1
    )
    agg.add_mentions(mentions)
    pairs = {(left, right) for left, right, _ in agg.finalize().co_occurrences}
    # Only the (hot, warm) pair survives; the least-frequent "cold" is dropped from pairing.
    assert pairs == {("hot", "warm")}


def test_accumulated_pair_table_is_capped() -> None:
    """The corpus-wide pair table cannot grow past ``max_tracked_pairs``.

    Adversarial regression: a stream of documents each introducing fresh entity pairs would
    grow ``_pair_docs`` without bound. With the cap, new pairs are dropped once full while
    already-tracked pairs still accumulate their document sets. Would FAIL if reverted.
    """
    agg = CorpusAggregator(max_tracked_pairs=3, max_entities_per_document=2)
    for i in range(50):
        agg.add_mentions(
            [
                MentionView(document_id=f"d{i}", text=f"a{i}", label="X"),
                MentionView(document_id=f"d{i}", text=f"b{i}", label="X"),
            ]
        )
    assert len(agg._pair_docs) == 3  # noqa: SLF001


def test_pair_cap_still_accumulates_tracked_pairs_across_docs() -> None:
    """Once full, an already-tracked pair keeps counting new documents (not frozen out)."""
    agg = CorpusAggregator(max_tracked_pairs=1, max_entities_per_document=2)
    # First doc registers the (acme, alice) pair (fills the 1-slot table).
    agg.add_mentions(
        [
            MentionView(document_id="d1", text="acme", label="ORG"),
            MentionView(document_id="d1", text="alice", label="PERSON"),
        ]
    )
    # Second doc's NEW pair (bob, carol) is dropped (table full)...
    agg.add_mentions(
        [
            MentionView(document_id="d2", text="bob", label="PERSON"),
            MentionView(document_id="d2", text="carol", label="PERSON"),
        ]
    )
    # ...but a repeat of the tracked pair still adds its document.
    agg.add_mentions(
        [
            MentionView(document_id="d3", text="acme", label="ORG"),
            MentionView(document_id="d3", text="alice", label="PERSON"),
        ]
    )
    pairs = {(left, right): count for left, right, count in agg.finalize().co_occurrences}
    assert pairs == {("acme", "alice"): 2}


def test_caps_do_not_change_result_below_threshold() -> None:
    """Below the caps, behaviour is byte-identical to an uncapped aggregator (no regression)."""
    capped = CorpusAggregator(max_entities_per_document=1000, max_tracked_pairs=1000)
    default = CorpusAggregator()
    for agg in (capped, default):
        agg.add_mentions(
            [
                MentionView(document_id="d1", text="Alice", label="PERSON"),
                MentionView(document_id="d1", text="Acme", label="ORG"),
                MentionView(document_id="d1", text="Bob", label="PERSON"),
            ]
        )
    assert capped.finalize() == default.finalize()
