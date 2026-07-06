"""Corpus profiling engine — the consolidated, ported enricher set (S7).

Ported by RECAST from the ``latenceai-dataset-intelligence`` enrichment stack
(ASSET-INVENTORY.md / PORTING-MAP.md: "9 enrichers as a registry → consolidate; the
registry is heavier than a fixed set needs" and "Feature disk-spilling under memory
pressure → KEEP, Profiling Stage"). The private stack's nine enrichers — density,
readability, Zipf, compression, structure, entity frequency, co-occurrence, type
consensus, source coverage — are re-homed here as **one fixed feature set** behind
two pure functions, dropping the dynamic registry indirection. What is preserved
verbatim is the algorithmic value:

1. **Per-document statistics** (:func:`document_features`) — density, a
   dependency-free Flesch-style readability normalised to ``[0, 1]``, a least-squares
   Zipfian-α fit over the word-frequency distribution, and a zlib compression ratio as
   a redundancy proxy. Deterministic and CPU-only (no tokenizer, no model download),
   so a seeded run is byte-identical (Baseline bar).

2. **Cross-document aggregation** (:class:`CorpusAggregator`) — a **streaming**
   accumulator: documents are folded in one at a time and the corpus text is never
   held whole in RAM, so the feature pass never buffers the whole corpus (S7 AC). The
   corpus Zipf α, per-language/category/sensitivity source coverage, cross-document
   entity frequency, entity-type consensus, and the top entity co-occurrence pairs are
   computed incrementally and finalised once.

   The aggregate state that *is* retained is explicitly bounded so it cannot be driven
   to unbounded growth by a hostile or pathological mention stream (a memory-DoS lever
   with a large NER Provider): the per-document co-occurrence fan-out is capped at
   ``max_entities_per_document`` distinct entities (so one ``add_mentions`` call is
   ``O(cap²)``, not ``O(distinct-entities²)``) and the accumulated pair table is capped
   at ``max_tracked_pairs`` entries. The frequency/consensus counters and the corpus
   vocabulary grow with the corpus *vocabulary* (distinct words / distinct entity
   surfaces), which is inherent to the feature set — this is a per-vocabulary bound, not
   a spill-to-disk one; nothing here spills to disk.

This is a pure engine (no contracts, no IO); the reference Profiler Provider
(:mod:`latence_core.stages.profiling`) wraps it and stamps the framework contracts.
It lives in core (like :mod:`latence_core.chunking` and :mod:`latence_core.pagemap`)
because corpus statistics are a Provenance/Quality concern the framework owns, not a
single Provider's private detail.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass, field

# Word / sentence tokenisation: deterministic, dependency-free (no NLTK/spaCy
# download), so a seeded run is hash-stable and CPU-cheap (ADR-0007). ``_WORD_RE``
# is alphanumeric runs (apostrophes kept so "don't" is one word); ``_SENTENCE_RE``
# splits on terminal punctuation.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*")
_SENTENCE_RE = re.compile(r"[.!?]+(?:\s|$)")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")

# The co-occurrence report is capped so the corpus feature record stays bounded
# regardless of corpus size (the report is a summary, not the full O(n²) matrix).
_MAX_CO_OCCURRENCES = 50

# Per-document co-occurrence fan-out cap: at most this many distinct entities from one
# document enter the pairing, so a single ``add_mentions`` call is O(cap²) rather than
# O(distinct-entities²). Without it a document with a large NER Provider (thousands of
# distinct surfaces) builds millions of transient pairs in one call — an algorithmic
# memory-DoS lever. The retained entities are the most frequent within the document
# (ties broken lexicographically), so the cap keeps the strongest co-occurrence signal.
_MAX_ENTITIES_PER_DOCUMENT = 200

# Accumulated pair-table cap: the total number of distinct co-occurring pairs held
# across the whole corpus is bounded, so a large corpus cannot grow ``_pair_docs``
# without limit. Once the table is full, a pair not already tracked is dropped (the
# report is a capped summary anyway — see ``_MAX_CO_OCCURRENCES``). Sized generously
# above the report cap so the finalised top-K is unaffected on any realistic corpus.
_MAX_TRACKED_PAIRS = 100_000


@dataclass(frozen=True)
class _Tokenised:
    """A document's tokenisation, computed once and shared across feature functions."""

    words: list[str]
    sentence_count: int


def _tokenise(text: str) -> _Tokenised:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    # At least one sentence when there is any word but no terminal punctuation, so
    # avg-sentence-length is never a divide-by-zero for a headline/fragment.
    sentences = len(_SENTENCE_RE.findall(text))
    if sentences == 0 and words:
        sentences = 1
    return _Tokenised(words=words, sentence_count=sentences)


def _count_syllables(word: str) -> int:
    """A deterministic English-ish syllable estimate (vowel groups, min 1).

    A dependency-free stand-in for a pronunciation dictionary: count vowel groups,
    drop a trailing silent 'e', floor at 1. Used only to shape the readability score;
    identical input ⇒ identical count.
    """
    groups = _VOWEL_GROUP_RE.findall(word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_score(word_count: int, sentence_count: int, syllable_count: int) -> float:
    """Flesch reading-ease mapped into ``[0, 1]`` (higher = easier to read).

    The classic Flesch reading-ease formula
    ``206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)`` yields a score
    roughly in ``[0, 100]`` for natural text (higher = easier); we clamp it to that
    range and divide by 100 so the feature is a bounded ``[0, 1]`` quality signal.
    Empty text scores 0.0.
    """
    if word_count == 0 or sentence_count == 0:
        return 0.0
    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    clamped = min(max(flesch, 0.0), 100.0)
    return round(clamped / 100.0, 6)


def zipf_alpha(frequencies: list[int]) -> float:
    """Fit the Zipfian exponent α by least-squares on ``log(rank)`` vs ``log(freq)``.

    A word-frequency distribution obeying Zipf's law has ``freq ∝ rank**(-α)`` with
    α near 1.0 for well-formed natural language. Ranking the frequencies descending
    and regressing ``log(freq)`` on ``log(rank)`` gives a slope whose negation is α.
    Returns 0.0 when there are fewer than two distinct frequencies to fit (no line
    through one point), so a one-word or empty document has a defined feature.
    Deterministic: the ranks come from a stable descending sort.
    """
    ranked = sorted(frequencies, reverse=True)
    # Need at least two points with positive frequency to fit a slope.
    points = [
        (math.log(rank), math.log(freq))
        for rank, freq in enumerate(ranked, start=1)
        if freq > 0
    ]
    if len(points) < 2:
        return 0.0
    n = len(points)
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xx = sum(x * x for x, _ in points)
    sum_xy = sum(x * y for x, y in points)
    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    # α is the negated slope; a well-formed corpus is ~1.0. Never report a negative α
    # (a degenerate/near-uniform distribution) — clamp at 0.0.
    return round(max(-slope, 0.0), 6)


@dataclass(frozen=True)
class DocFeatureValues:
    """The computed per-document feature values (engine output, contract-free).

    Mirrors the contract :class:`~latence_core.contracts.DocumentFeatures` field for
    field, so the Provider maps it across without re-deriving anything.
    """

    char_count: int
    word_count: int
    sentence_count: int
    unique_word_count: int
    type_token_ratio: float
    density: float
    readability: float
    zipf_alpha: float
    compression_ratio: float
    avg_word_length: float
    avg_sentence_length: float
    # The lowercased word-frequency counts, surfaced so the corpus aggregator folds
    # them in without re-tokenising the document text.
    word_frequencies: dict[str, int] = field(default_factory=dict)


def document_features(text: str) -> DocFeatureValues:
    """Compute the fixed per-document feature set over one document's assembled markdown.

    Deterministic and dependency-free. An empty (or whitespace-only) document yields a
    well-defined all-zero feature set rather than an error, so every parsed document is
    profilable.
    """
    char_count = len(text)
    tok = _tokenise(text)
    words = tok.words
    word_count = len(words)

    if word_count == 0:
        return DocFeatureValues(
            char_count=char_count,
            word_count=0,
            sentence_count=0,
            unique_word_count=0,
            type_token_ratio=0.0,
            density=_density(text),
            readability=0.0,
            zipf_alpha=0.0,
            compression_ratio=_compression_ratio(text),
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            word_frequencies={},
        )

    freqs = Counter(words)
    unique_word_count = len(freqs)
    total_word_chars = sum(len(w) for w in words)
    syllables = sum(_count_syllables(w) for w in words)

    return DocFeatureValues(
        char_count=char_count,
        word_count=word_count,
        sentence_count=tok.sentence_count,
        unique_word_count=unique_word_count,
        type_token_ratio=round(unique_word_count / word_count, 6),
        density=_density(text),
        readability=readability_score(word_count, tok.sentence_count, syllables),
        zipf_alpha=zipf_alpha(list(freqs.values())),
        compression_ratio=_compression_ratio(text),
        avg_word_length=round(total_word_chars / word_count, 6),
        avg_sentence_length=(
            round(word_count / tok.sentence_count, 6) if tok.sentence_count else 0.0
        ),
        # Sorted for a deterministic, stable payload.
        word_frequencies=dict(sorted(freqs.items())),
    )


def _density(text: str) -> float:
    """Non-whitespace characters as a fraction of all characters (0..1)."""
    if not text:
        return 0.0
    non_ws = sum(1 for c in text if not c.isspace())
    return round(non_ws / len(text), 6)


def _compression_ratio(text: str) -> float:
    """zlib-compressed size / raw size — a redundancy proxy (0.0 for empty text).

    A repetitive/templated document compresses hard (low ratio); diverse prose stays
    near 1.0. Deterministic (fixed zlib level).
    """
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    compressed = zlib.compress(raw, level=6)
    return round(len(compressed) / len(raw), 6)


# -- cross-document streaming aggregation ------------------------------------


@dataclass(frozen=True)
class MentionView:
    """The minimal view of an EntityMention the corpus aggregator folds in.

    Contract-free so the engine stays IO/contract-independent: the Provider projects
    each :class:`~latence_core.contracts.EntityMention` into this before folding.
    """

    document_id: str
    text: str
    label: str


@dataclass(frozen=True)
class CorpusFeatureValues:
    """The finalised cross-document feature values (engine output, contract-free)."""

    document_count: int
    total_words: int
    total_unique_words: int
    corpus_zipf_alpha: float
    language_coverage: dict[str, int]
    category_coverage: dict[str, int]
    sensitivity_coverage: dict[str, int]
    entity_frequency: dict[str, int]
    entity_type_frequency: dict[str, int]
    entity_type_consensus: dict[str, float]
    co_occurrences: list[tuple[str, str, int]]


class CorpusAggregator:
    """Streaming, memory-bounded accumulator for the cross-document feature set.

    Documents are folded in one at a time via :meth:`add_document` and their mentions
    via :meth:`add_mentions`; the corpus text is never held whole in RAM — the S7
    "streaming, no whole-corpus-in-RAM assumption" acceptance criterion. :meth:`finalize`
    fits the corpus Zipf α and derives the type-consensus and the capped co-occurrence
    report once.

    **Bounded aggregate state (no unbounded-growth DoS lever).** The retained state is
    explicitly capped so a hostile or pathological mention stream cannot drive it to
    unbounded growth:

    * the per-document co-occurrence fan-out is capped at ``max_entities_per_document``
      distinct entities, so a single :meth:`add_mentions` call is ``O(cap²)`` rather than
      ``O(distinct-entities²)`` — the most frequent entities in the document are kept;
    * the accumulated pair table is capped at ``max_tracked_pairs`` entries — once full,
      a pair not already tracked is dropped (the report is a capped top-K summary anyway).

    The frequency/consensus counters and the corpus vocabulary still grow with the corpus
    *vocabulary* (distinct words / distinct entity surfaces) — inherent to the feature
    set, a per-vocabulary bound, not a spill-to-disk one; nothing spills to disk.

    Determinism: every finalised map is sorted, and the co-occurrence report is ranked
    by ``(-document_count, left, right)``, so a seeded run is byte-identical regardless
    of document arrival order (the aggregation is commutative). The per-document cap keeps
    the most-frequent entities with lexicographic tie-breaks, so it too is order-stable.
    """

    def __init__(
        self,
        max_co_occurrences: int = _MAX_CO_OCCURRENCES,
        *,
        max_entities_per_document: int = _MAX_ENTITIES_PER_DOCUMENT,
        max_tracked_pairs: int = _MAX_TRACKED_PAIRS,
    ) -> None:
        self._document_count = 0
        self._total_words = 0
        self._corpus_word_freq: Counter[str] = Counter()
        self._language_coverage: Counter[str] = Counter()
        self._category_coverage: Counter[str] = Counter()
        self._sensitivity_coverage: Counter[str] = Counter()
        self._entity_freq: Counter[str] = Counter()
        self._entity_type_freq: Counter[str] = Counter()
        # Per entity text, the distribution of labels it was tagged with (for consensus).
        self._entity_labels: dict[str, Counter[str]] = {}
        # Per unordered entity pair, the set of documents both appeared in (so the
        # co-occurrence count is cross-document, deduplicated within a document). Capped
        # at ``_max_tracked_pairs`` so it cannot grow without bound on a large corpus.
        self._pair_docs: dict[tuple[str, str], set[str]] = {}
        self._max_co_occurrences = max_co_occurrences
        self._max_entities_per_document = max_entities_per_document
        self._max_tracked_pairs = max_tracked_pairs

    def add_document(
        self,
        word_frequencies: dict[str, int],
        language: str,
        category: str | None,
        sensitivity: str,
    ) -> None:
        """Fold one profiled document's word frequencies + coverage attributes in.

        ``word_frequencies`` is the per-document lowercased count already computed by
        :func:`document_features` (not re-tokenised here). Coverage attributes come from
        the document's :class:`~latence_core.contracts.Classification`.
        """
        self._document_count += 1
        for word, count in word_frequencies.items():
            self._corpus_word_freq[word] += count
            self._total_words += count
        self._language_coverage[language] += 1
        if category is not None:
            self._category_coverage[category] += 1
        self._sensitivity_coverage[sensitivity] += 1

    def add_mentions(self, mentions: list[MentionView]) -> None:
        """Fold one document's entity mentions into the cross-document tallies.

        Entity frequency and type frequency count every mention; type consensus tracks
        the per-entity label distribution; co-occurrence records the DISTINCT entity
        pairs of *this* document against the set of documents each pair has co-occurred
        in (so a pair repeated within one document counts that document once). The
        per-document mention set is consumed here and not retained.

        The co-occurrence fan-out is bounded: at most ``max_entities_per_document``
        distinct entities from this document enter the pairing (the most frequent within
        the document, lexicographic tie-break), so one call is ``O(cap²)`` rather than
        ``O(distinct-entities²)`` — a large NER Provider cannot make a single document
        build an unbounded pair explosion. Frequency/consensus still count every mention.
        """
        if not mentions:
            return
        # add_mentions is called per document, so every mention shares one document_id.
        document_id = mentions[0].document_id
        # Frequency, type frequency, and consensus count EVERY mention (unbounded fan-out
        # applies only to the O(n²) pairing, not to these O(n) tallies).
        doc_entity_freq: Counter[str] = Counter()
        for mention in mentions:
            self._entity_freq[mention.text] += 1
            self._entity_type_freq[mention.label] += 1
            self._entity_labels.setdefault(mention.text, Counter())[mention.label] += 1
            doc_entity_freq[mention.text] += 1
        # Cap the distinct entities that enter the pairing to the most-frequent-in-document
        # (ties broken lexicographically), so the pairing is O(cap²) and deterministic.
        if len(doc_entity_freq) > self._max_entities_per_document:
            kept = sorted(doc_entity_freq.items(), key=lambda kv: (-kv[1], kv[0]))
            ordered = sorted(text for text, _ in kept[: self._max_entities_per_document])
        else:
            ordered = sorted(doc_entity_freq)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                pair = (left, right)
                # Record this pair as co-occurring in this document (deduplicated by
                # document_id, so a pair repeated within one document counts once). Once
                # the accumulated pair table is full, only already-tracked pairs are
                # updated — a new pair is dropped to keep the corpus-wide state bounded.
                existing = self._pair_docs.get(pair)
                if existing is not None:
                    existing.add(document_id)
                elif len(self._pair_docs) < self._max_tracked_pairs:
                    self._pair_docs[pair] = {document_id}

    def finalize(self) -> CorpusFeatureValues:
        """Fit corpus Zipf α, derive type consensus, and build the capped co-occurrence report."""
        consensus: dict[str, float] = {}
        for text, labels in self._entity_labels.items():
            total = sum(labels.values())
            # Only score entities seen at least twice; a single mention is trivially
            # 1.0 consensus and would drown the signal.
            if total < 2:
                continue
            top = max(labels.values())
            consensus[text] = round(top / total, 6)

        co_occurrences = sorted(
            ((left, right, len(docs)) for (left, right), docs in self._pair_docs.items()),
            key=lambda t: (-t[2], t[0], t[1]),
        )[: self._max_co_occurrences]

        return CorpusFeatureValues(
            document_count=self._document_count,
            total_words=self._total_words,
            total_unique_words=len(self._corpus_word_freq),
            corpus_zipf_alpha=zipf_alpha(list(self._corpus_word_freq.values())),
            language_coverage=dict(sorted(self._language_coverage.items())),
            category_coverage=dict(sorted(self._category_coverage.items())),
            sensitivity_coverage=dict(sorted(self._sensitivity_coverage.items())),
            entity_frequency=dict(sorted(self._entity_freq.items())),
            entity_type_frequency=dict(sorted(self._entity_type_freq.items())),
            entity_type_consensus=dict(sorted(consensus.items())),
            co_occurrences=co_occurrences,
        )
