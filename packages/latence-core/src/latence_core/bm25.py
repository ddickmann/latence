"""The optional BM25 term-statistics artifact exporter (#164, ADR-0048/0049).

BM25 is the one signal in the retrieval-tooling layer that is **not** a per-text vector: it is a
corpus-wide *lexical statistic*. A customer's store already runs BM25 first-stage search, but to
score it needs the corpus statistics — per-term document frequency (``df``), per-document term
frequency (``tf``), each document's length, and the corpus average length (``avgdl``). This module
generates exactly those, **as files** (ADR-0017/0049), so a store can load them and drive lexical
search over the corpus latence produced — without latence ever holding an index or running a
first-stage search itself (ADR-0048: BM25 here is an artifact exporter and a query-time rescorer,
**never** a first-stage inverted index).

The generator is strictly stateless and additive. It is off unless an Export names a
:class:`~latence_core.capability.Tokenizer` under its ``bm25`` config; when off the corpus is
byte-identical (ADR-0017). The only thing that varies is the tokenisation, which is the injected
:class:`~latence_core.capability.Tokenizer` seam — this module ships its two in-core reference
adapters (``tokenizer.regex`` + ``tokenizer.whitespace``).

**The emitted artifact** (two additive, versioned files a store loads; see
:meth:`Bm25Stats.to_artifact` and :meth:`Bm25Posting.as_row`):

* ``bm25-stats.json`` — the corpus-level term statistics: ``doc_count`` (N), ``total_tokens``,
  ``avgdl``, the tokenizer name, and one ``{term, df, idf}`` entry per vocabulary term (terms
  ascending, so two runs are byte-identical). ``idf`` is the Robertson/Lucene BM25 idf
  ``ln(1 + (N - df + 0.5) / (df + 0.5))`` (always non-negative), precomputed so a store need not.
* ``bm25-postings.parquet`` — the per-document forward postings a store inverts: one row per record
  (``record_id``, ``doc_len``, parallel ``terms`` / ``term_frequencies`` sorted by term). Streamed
  one row group per Export batch, so its memory footprint is O(batch) — only the ``df`` map
  (O(vocab)) is retained across the corpus.

The statistics are generated over the redacted corpus text (``masked_content`` when present), so the
BM25 signal never re-leaks PII the corpus removed (ADR-0044/0053), and over the *stored* corpus text
(not the embedding-only W13 context header — that inflates the vector, never the lexical index), so
the artifact matches exactly what a store indexes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.capability import Tokenizer

#: The artifact-format identifier + version stamped into ``bm25-stats.json`` (contract discipline:
#: additive + versioned, ADR-0006). Bump only on a breaking shape change.
BM25_ARTIFACT = "latence.bm25.termstats"
BM25_ARTIFACT_VERSION = 1

_WORD = re.compile(r"[A-Za-z0-9]+")


def _reference_profile() -> ProviderProfile:
    """The shared descriptor for a pure-code, CPU, deterministic, Apache-2.0 reference tokenizer.

    Each reference Tokenizer is a pure-Python part of ``latence-core`` (Apache-2.0, verified per
    ADR-0012 — the package's own code, weights-and-code trivially the same): no model weights
    (``model_id=None``), deterministic, CPU-only, no batching, no per-call cost — so a
    conformance / bake-off run reads honest, non-fabricated columns off it (ADR-0036 §2).
    """
    return ProviderProfile(
        compute="cpu",
        memory_mb=8,
        model_id=None,
        license="Apache-2.0",
        license_verified=True,
        license_code="Apache-2.0",
        license_source=REPO_LICENSE_SOURCE,
        license_verified_on="2026-07-16",
        deterministic=True,
        batch=False,
        cost_per_1k=None,
    )


class RegexWordTokenizer:
    """Lowercase alphanumeric-run :class:`~latence_core.capability.Tokenizer` (the reference).

    Each text is lowercased and split into maximal ``[A-Za-z0-9]+`` runs, so punctuation and
    whitespace are boundaries and are dropped — the *reference tokenization* the BM25 tests assert
    against, and the exact tokenisation the in-core dense (``embedding.hashing``) and sparse
    (``sparse.hashing``) references use, so the lexical statistics line up with those signals.
    Deterministic; empty / term-free text yields ``[]``.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # No configurable knobs (the tokenisation IS the identity of this adapter); accepting a
        # config keeps the {provider, config} construction contract uniform with every Provider.
        _ = config

    def tokenize(self, texts: Iterable[str]) -> Iterator[list[str]]:
        for text in texts:
            yield _WORD.findall(text.lower())


class WhitespaceTokenizer:
    """Lowercase whitespace-split :class:`~latence_core.capability.Tokenizer` (the second adapter).

    Each text is lowercased and split on runs of whitespace (``str.split``), so punctuation stays
    **attached** to the token — ``"foo, bar."`` tokenises to ``["foo,", "bar."]``, not ``["foo",
    "bar"]``. This is a genuinely different tokenisation from :class:`RegexWordTokenizer` (different
    vocabulary, different df/tf), the second real adapter that proves the Tokenizer seam varies —
    not a clone. Deterministic; empty / whitespace-only text yields ``[]``.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        _ = config

    def tokenize(self, texts: Iterable[str]) -> Iterator[list[str]]:
        for text in texts:
            yield text.lower().split()


@dataclass(frozen=True)
class Bm25Posting:
    """One document's forward BM25 postings: its length + its per-term frequencies (canonical).

    Built by :meth:`Bm25Accumulator.add`. Invariants (enforced at construction so a malformed
    posting can never reach the parquet writer, and so the emission is byte-deterministic regardless
    of a caller's dict ordering):

    * ``terms`` are **strictly ascending** (hence unique) — the canonical order two runs agree on;
    * ``term_frequencies`` is the same length as ``terms`` and each is ``>= 1`` (a posting lists a
      term only if it occurs);
    * ``doc_len`` equals the sum of the term frequencies (the length BM25 length-normalises by).

    A document with no terms is the empty posting (``doc_len == 0``, empty parallel lists).
    """

    record_id: str
    doc_len: int
    terms: list[str] = field(default_factory=list)
    term_frequencies: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.terms) != len(self.term_frequencies):
            msg = (
                f"Bm25Posting terms/term_frequencies length mismatch: "
                f"{len(self.terms)} vs {len(self.term_frequencies)}."
            )
            raise ValueError(msg)
        prev: str | None = None
        for term in self.terms:
            if prev is not None and term <= prev:
                msg = (
                    "Bm25Posting terms must be strictly ascending (unique) — the canonical "
                    f"postings order; got {term!r} after {prev!r}."
                )
                raise ValueError(msg)
            prev = term
        if any(tf < 1 for tf in self.term_frequencies):
            msg = "Bm25Posting term frequencies must each be >= 1 (a posting lists a term it has)."
            raise ValueError(msg)
        if self.doc_len != sum(self.term_frequencies):
            msg = (
                f"Bm25Posting doc_len {self.doc_len} must equal the sum of term frequencies "
                f"{sum(self.term_frequencies)} (the length BM25 normalises by)."
            )
            raise ValueError(msg)

    def as_row(self) -> dict[str, Any]:
        """The parquet row for ``bm25-postings.parquet`` (native ``list`` columns)."""
        return {
            "record_id": self.record_id,
            "doc_len": self.doc_len,
            "terms": list(self.terms),
            "term_frequencies": list(self.term_frequencies),
        }


@dataclass(frozen=True)
class Bm25Stats:
    """The corpus-level BM25 term statistics — the ``bm25-stats.json`` artifact value.

    Produced by :meth:`Bm25Accumulator.stats` after the whole corpus is folded. Holds the corpus
    scalars (``doc_count`` = N, ``total_tokens``) plus the per-term document frequencies; ``avgdl``
    and each term's ``idf`` are derived (see :attr:`avgdl` / :meth:`idf`), so nothing is stored
    redundantly. ``tokenizer`` records which :class:`~latence_core.capability.Tokenizer` produced
    the statistics, so the artifact is self-describing (a store must analyse queries the same way).
    """

    tokenizer: str
    doc_count: int
    total_tokens: int
    document_frequencies: dict[str, int] = field(default_factory=dict)

    @property
    def avgdl(self) -> float:
        """The average document length in tokens — the BM25 length-normaliser (0.0 for no docs)."""
        if self.doc_count == 0:
            return 0.0
        return self.total_tokens / self.doc_count

    def idf(self, term: str) -> float:
        """The Robertson/Lucene BM25 idf for ``term``: ``ln(1 + (N - df + 0.5) / (df + 0.5))``.

        The ``+1`` form is always non-negative (a term in every document scores ``idf ~ 0``, never
        the classic negative idf), and it is the idf modern stores (Lucene/Elasticsearch/Qdrant)
        use — so a store can adopt the precomputed value directly. A term absent from the corpus has
        ``df == 0`` and the maximal idf. Rounded to 8 decimals so the artifact is byte-reproducible.
        """
        df = self.document_frequencies.get(term, 0)
        return round(math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5)), 8)

    def to_artifact(self) -> dict[str, Any]:
        """The full ``bm25-stats.json`` object (self-describing, versioned, deterministic).

        Terms are emitted in ascending order and every derived value is rounded, so two runs over
        the same corpus produce byte-identical JSON (the Baseline bar). ``avgdl`` is rounded to 8
        decimals for the same reason.
        """
        return {
            "artifact": BM25_ARTIFACT,
            "version": BM25_ARTIFACT_VERSION,
            "tokenizer": self.tokenizer,
            "doc_count": self.doc_count,
            "total_tokens": self.total_tokens,
            "avgdl": round(self.avgdl, 8),
            "terms": [
                {"term": term, "df": self.document_frequencies[term], "idf": self.idf(term)}
                for term in sorted(self.document_frequencies)
            ],
        }


class Bm25Accumulator:
    """Folds a corpus into its BM25 term statistics + per-document postings (the deep module).

    Constructed with an injected :class:`~latence_core.capability.Tokenizer` (the varying seam) and
    the tokenizer's registered name (stamped into the artifact so it is self-describing). It accepts
    the dependency, holds no store and runs no search (ADR-0048), and its interface is tiny: feed it
    documents with :meth:`add` / :meth:`add_batch` (each returns that document's canonical
    :class:`Bm25Posting`, which the caller streams to the postings file), then call :meth:`stats`
    for the corpus-level :class:`Bm25Stats`.

    Memory is O(vocabulary): only the per-term ``df`` map + two scalars are retained across the
    corpus; the heavy per-document postings are **returned** for the caller to stream, never
    buffered here — so a BM25 export stays within the Export's O(batch) footprint (the postings are
    one parquet row group per batch). Deterministic: tokenisation drives everything and both
    tokenizers are deterministic, so the whole artifact is byte-reproducible.
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_name: str) -> None:
        self._tokenizer = tokenizer
        self._tokenizer_name = tokenizer_name
        self._df: dict[str, int] = {}
        self._doc_count = 0
        self._total_tokens = 0

    def add(self, record_id: str, text: str) -> Bm25Posting:
        """Fold one document in; return its canonical :class:`Bm25Posting`.

        Tokenises ``text`` once, counts term frequencies, updates the corpus document frequency (one
        increment per *distinct* term, never per occurrence — the df definition) and the corpus
        scalars, and returns the document's posting (terms ascending, tf parallel). An empty /
        term-free document still counts toward ``doc_count`` (it is a document) with an empty
        posting and ``doc_len == 0``.
        """
        [tokens] = list(self._tokenizer.tokenize([text]))
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for term in tf:
            self._df[term] = self._df.get(term, 0) + 1
        self._doc_count += 1
        self._total_tokens += len(tokens)
        ordered = sorted(tf)
        return Bm25Posting(
            record_id=record_id,
            doc_len=len(tokens),
            terms=ordered,
            term_frequencies=[tf[t] for t in ordered],
        )

    def add_batch(self, documents: Iterable[tuple[str, str]]) -> list[Bm25Posting]:
        """Fold a batch of ``(record_id, text)`` documents in order; return their postings."""
        return [self.add(record_id, text) for record_id, text in documents]

    def stats(self) -> Bm25Stats:
        """The corpus-level :class:`Bm25Stats` over every document folded so far.

        A defensive copy of the ``df`` map is taken, so a caller cannot mutate the accumulator's
        state through the returned value. Calling it before any document yields the empty stats
        (``doc_count == 0``, ``avgdl == 0.0``, no terms) — a valid, still-emittable artifact.
        """
        return Bm25Stats(
            tokenizer=self._tokenizer_name,
            doc_count=self._doc_count,
            total_tokens=self._total_tokens,
            document_frequencies=dict(self._df),
        )


__all__ = [
    "BM25_ARTIFACT",
    "BM25_ARTIFACT_VERSION",
    "Bm25Accumulator",
    "Bm25Posting",
    "Bm25Stats",
    "RegexWordTokenizer",
    "WhitespaceTokenizer",
]
