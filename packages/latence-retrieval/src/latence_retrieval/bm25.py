# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The ``bm25`` rescorer: :class:`Bm25Rescorer`, a :class:`CandidateProcessor` (ADR-0048, #170).

``bm25`` is the query-time transform that adds **exact lexical scoring** to an already-fetched
candidate list — the "rescore what my engine returned, without standing up a second index" move
(SPEC user story 16). It is a :class:`~latence_retrieval.processor.CandidateProcessor`, not a new
seam: a caller fetched the candidates from its engine, and this only re-scores/reorders them with
the Okapi BM25 formula. It holds nothing and fetches nothing (ADR-0048) — most importantly it
**never builds or holds a first-stage index**: it scores exactly the candidate set it is handed, so
it is a rescorer, never a retriever.

**The score.** For a query's distinct terms ``Q`` and a candidate document ``d`` of length ``|d|``
tokens, the score is the standard Okapi BM25 sum over the terms the document actually contains::

    score(d) = Σ_{t ∈ Q ∩ d}  idf(t) · ( tf(t,d) · (k1 + 1) )
                              ─────────────────────────────────────────
                               tf(t,d) + k1 · (1 − b + b · |d| / avgdl)

with the non-negative Lucene/Robertson IDF ``idf(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))``.
``tf(t,d)`` is the term's frequency in the candidate, ``|d|`` the candidate's token count. It is
**exact** — the same number a real BM25 index computes for that document given the same statistics —
not a heuristic proxy.

**Where the corpus statistics (``N``, ``df``, ``avgdl``) come from — the deep choice this hides.**
BM25's IDF needs document frequencies over a *corpus*, but a rescorer only sees a candidate set. Two
honest sources, selected by config and hidden behind the one :meth:`process` interface (the AC):

* **supplied ``stats``** — the caller passes the true corpus statistics its engine already holds
  (``doc_count``, ``avg_doc_len``, ``doc_freq``). Then the rescore reproduces the engine's *global*
  BM25 exactly — the right thing when you have the numbers.
* **the candidate pool itself** — with no ``stats``, ``N`` = the number of candidates, ``df(t)`` =
  how many candidates contain ``t``, and ``avgdl`` = the mean candidate length. This is a genuine,
  self-contained BM25 over the pool (a documented *pool-local* IDF, not the global corpus IDF), so
  the transform works with zero setup and gets *exact-global* the moment real ``stats`` are fed in.

**Optimized for large candidate sets.** The scoring is **query-term-driven**, not
candidate-driven: one pass tokenises each candidate and, in the same pass, builds postings **only**
for the query's terms (a document with none of the query terms is never touched again and keeps
score ``0``). The BM25 sum is then accumulated over those postings — work proportional to the actual
``(query-term, matching-document)`` pairs, not to ``|candidates| × |query terms|``. Pure-Python,
CPU-only, and deterministic (ADR-0049); a compiled-Cython accelerator over this same exact formula
is a documented follow-on (like the SPLADE torch-reduce and the ported Knapsack solver), not a
shipped fake.

Like every candidate processor it never mutates its input
(:class:`~latence_retrieval.contracts.Candidate` is frozen; a fresh list is returned) and holds no
state across calls, so identical inputs always yield an identical rescored order.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from latence_retrieval.contracts import Candidate

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

# The filename the Export stage writes the corpus-level BM25 term statistics to
# (``latence_core.bm25.Bm25Stats.to_artifact``). This package is deliberately independent of
# `latence-core` (it holds no index and runs no pipeline, ADR-0048), so the agreement on this
# artifact's name + shape is held here by :meth:`Bm25Rescorer.from_export` and its test, not an
# import — exactly as the tokenizer agreement on `_TOKEN` below is.
_BM25_STATS_FILENAME = "bm25-stats.json"

# Unicode alphanumeric runs (underscore excluded). This MUST stay the same term unit the index-side
# tokenizer emits (`latence_core.bm25.RegexWordTokenizer`, the one the `enterprise-sota` setup
# profile blesses): a query term the index never emits scores nothing, so an ASCII-only pattern made
# the whole sparse leg a permanent no-op for non-Latin corpora (R5-wizard_profile-13). This package
# is deliberately independent of `latence-core`, so the agreement is held by a test, not an import.
# ASCII-identical to the previous `[a-z0-9]+` (the text is lowercased first).
_TOKEN = re.compile(r"[^\W_]+")
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens — the dependency-free, deterministic BM25 term unit."""
    return _TOKEN.findall(text.lower())


class _CorpusStats:
    """The BM25 corpus statistics: ``N`` documents, ``df`` per term, and ``avgdl``.

    Either supplied by the caller (the engine's true global numbers) or derived from the candidate
    pool. Holds nothing durable — it is built per :meth:`Bm25Rescorer.process` call and discarded.
    """

    __slots__ = ("avgdl", "doc_count", "doc_freq")

    def __init__(self, doc_count: int, avgdl: float, doc_freq: dict[str, int]) -> None:
        self.doc_count = doc_count
        self.avgdl = avgdl
        self.doc_freq = doc_freq

    def idf(self, term: str) -> float:
        """Non-negative Lucene/Robertson IDF; an unseen term has ``df = 0`` (its maximum IDF)."""
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))


class Bm25Rescorer:
    """Exactly re-score a candidate list with Okapi BM25 and reorder it (a ``CandidateProcessor``).

    :meth:`process` computes the BM25 score (see the module docstring) of every candidate's ``text``
    against the ``query``, replaces each candidate's ``score`` with it, and returns a new list
    sorted by descending score (ties broken by ascending ``id`` so identical inputs always yield an
    identical order — the determinism the ``rerank``/``fuse`` seams hold too). The input list is
    never mutated and the rescorer keeps no state across calls.

    Config:
        k1: float — term-frequency saturation. Default ``1.5``; must be ``> 0``.
        b: float — length-normalisation strength. Default ``0.75``; must be in ``[0, 1]``.
        top_k: int | None — if set (a positive int), the rescored list is truncated to its top
            ``top_k``. Default ``None`` keeps every candidate, reordered.
        stats: dict | None — the caller's true corpus statistics, so the rescore reproduces the
            engine's *global* BM25 exactly. When present it must carry ``doc_count`` (int ``>= 1``),
            ``avg_doc_len`` (number ``> 0``), and ``doc_freq`` (a ``{term: int in [0, doc_count]}``
            mapping — a term cannot occur in more documents than the corpus holds, and a df above
            ``doc_count`` would flip the IDF negative; terms absent from it are treated as
            ``df = 0``). When omitted, the statistics are
            derived from the candidate pool itself (``N`` = pool size, ``df`` = in-pool document
            frequency, ``avgdl`` = mean candidate length) — a self-contained pool-local BM25.

    The rescorer is pure and stateless; ``stats`` are validated at construction so a malformed
    corpus-statistics block fails at wiring time, never silently at query time.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})

        k1 = cfg.get("k1", _DEFAULT_K1)
        if not isinstance(k1, (int, float)) or isinstance(k1, bool) or k1 <= 0:
            msg = f"bm25 'k1' must be a number > 0, got {k1!r}."
            raise ValueError(msg)
        self._k1 = float(k1)

        b = cfg.get("b", _DEFAULT_B)
        if not isinstance(b, (int, float)) or isinstance(b, bool) or not 0.0 <= b <= 1.0:
            msg = f"bm25 'b' must be a number in [0, 1], got {b!r}."
            raise ValueError(msg)
        self._b = float(b)

        top_k = cfg.get("top_k")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
        ):
            msg = f"bm25 'top_k' must be a positive integer or omitted, got {top_k!r}."
            raise ValueError(msg)
        self._top_k: int | None = top_k

        self._supplied_stats = self._parse_stats(cfg.get("stats"))

    @classmethod
    def from_export(
        cls,
        export_dir: str | PathLike[str],
        config: dict[str, Any] | None = None,
    ) -> Bm25Rescorer:
        """Build a rescorer from a pipeline export's ``bm25-stats.json`` — **no re-tokenisation**.

        The Export stage writes the corpus-level BM25 term statistics to
        ``<export_dir>/bm25-stats.json`` (:meth:`latence_core.bm25.Bm25Stats.to_artifact`): the
        corpus scalars ``doc_count`` (N) and ``avgdl`` plus one ``{term, df, idf}`` entry per
        vocabulary term. This reads **only that compact stats file** and constructs a rescorer whose
        supplied ``stats`` reproduce the corpus's *global* BM25 exactly — the same numbers the index
        computed — without re-reading or re-tokenising the corpus (the postings parquet is never
        touched: global IDF needs only ``doc_count`` + per-term ``df``). The artifact's ``avgdl``
        maps to the rescorer's ``avg_doc_len`` and its ``terms[].df`` fold into ``doc_freq``; the
        stored per-term ``idf`` is *derived*, so it is not carried across — the rescorer recomputes
        the identical Lucene/Robertson IDF from ``doc_count`` + ``df``.

        ``export_dir`` may be the export directory (the file ``bm25-stats.json`` is joined onto it)
        or the stats file itself. ``config`` may carry the usual ``k1`` / ``b`` / ``top_k`` knobs;
        passing a ``stats`` key in it is rejected (the statistics come from the export, never two
        sources at once). Reading is stateless: the returned rescorer holds only the parsed
        statistics (ADR-0048), never the file handle or the corpus. A missing, unreadable, or
        malformed artifact raises :class:`ValueError` at construction — never a silent,
        quietly-wrong rescore at query time (ADR-0034).
        """
        base = Path(export_dir)
        stats_path = base if base.is_file() else base / _BM25_STATS_FILENAME
        try:
            raw_text = stats_path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"bm25 from_export: could not read the stats artifact at '{stats_path}': {exc}."
            raise ValueError(msg) from exc
        try:
            artifact = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = f"bm25 from_export: '{stats_path}' is not valid JSON: {exc}."
            raise ValueError(msg) from exc
        if not isinstance(artifact, dict):
            got = type(artifact).__name__
            msg = f"bm25 from_export: '{stats_path}' must hold a JSON object, got {got}."
            raise ValueError(msg)

        terms = artifact.get("terms")
        if not isinstance(terms, list):
            msg = f"bm25 from_export: '{stats_path}' 'terms' must be a list, got {terms!r}."
            raise ValueError(msg)
        doc_freq: dict[str, int] = {}
        for entry in terms:
            if not isinstance(entry, dict):
                msg = f"bm25 from_export: each 'terms' entry must be an object, got {entry!r}."
                raise ValueError(msg)
            term = entry.get("term")
            df = entry.get("df")
            if not isinstance(term, str) or not term:
                msg = f"bm25 from_export: a 'terms' entry has a non-string/empty 'term': {entry!r}."
                raise ValueError(msg)
            if isinstance(df, bool) or not isinstance(df, int):
                msg = f"bm25 from_export: 'terms' entry {term!r} has a non-integer 'df': {df!r}."
                raise ValueError(msg)
            doc_freq[term] = df

        merged = dict(config or {})
        if "stats" in merged:
            msg = (
                "bm25 from_export: the corpus statistics come from the export; do not also pass a "
                "'stats' block in config."
            )
            raise ValueError(msg)
        # Hand the raw scalars straight to _parse_stats so the export path shares the ONE validation
        # gate (doc_count int >= 1, avg_doc_len > 0, every df in [0, doc_count]) — a malformed or
        # empty-corpus artifact fails loudly here, never silently at query time.
        merged["stats"] = {
            "doc_count": artifact.get("doc_count"),
            "avg_doc_len": artifact.get("avgdl"),
            "doc_freq": doc_freq,
        }
        return cls(merged)

    @staticmethod
    def _parse_stats(raw: object) -> _CorpusStats | None:
        """Validate an optional caller-supplied corpus-statistics block into :class:`_CorpusStats`.

        A real BM25 index rejects nonsensical global statistics; so does this (anti-false-green),
        loudly at construction rather than yielding quietly-wrong scores at query time.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            msg = (
                "bm25 'stats' must be a mapping with 'doc_count', 'avg_doc_len' and 'doc_freq', "
                f"got {raw!r}."
            )
            raise TypeError(msg)

        doc_count = raw.get("doc_count")
        if not isinstance(doc_count, int) or isinstance(doc_count, bool) or doc_count < 1:
            msg = f"bm25 stats 'doc_count' must be an integer >= 1, got {doc_count!r}."
            raise ValueError(msg)

        avgdl = raw.get("avg_doc_len")
        if not isinstance(avgdl, (int, float)) or isinstance(avgdl, bool) or avgdl <= 0:
            msg = f"bm25 stats 'avg_doc_len' must be a number > 0, got {avgdl!r}."
            raise ValueError(msg)

        doc_freq_raw = raw.get("doc_freq")
        if not isinstance(doc_freq_raw, dict):
            msg = f"bm25 stats 'doc_freq' must be a mapping of term → int, got {doc_freq_raw!r}."
            raise TypeError(msg)
        doc_freq: dict[str, int] = {}
        for term, df in doc_freq_raw.items():
            if not isinstance(term, str) or not term:
                msg = f"bm25 stats 'doc_freq' keys must be non-empty strings, got {term!r}."
                raise ValueError(msg)
            if not isinstance(df, int) or isinstance(df, bool) or df < 0:
                msg = f"bm25 stats 'doc_freq[{term!r}]' must be an integer >= 0, got {df!r}."
                raise ValueError(msg)
            if df > doc_count:
                # A term cannot occur in more documents than the corpus holds. Letting df > N
                # through yields a NEGATIVE Lucene/Robertson IDF (the log's argument drops below
                # 1), which silently inverts relevance — a term-bearing doc scoring below a
                # non-bearing one. Reject the impossible statistic loudly at construction (the
                # classic df-vs-collection-frequency mixup) rather than quietly-wrong at query time.
                msg = (
                    f"bm25 stats 'doc_freq[{term!r}]' must be <= 'doc_count' ({doc_count}), "
                    f"got {df!r}."
                )
                raise ValueError(msg)
            doc_freq[term] = df

        return _CorpusStats(doc_count=doc_count, avgdl=float(avgdl), doc_freq=doc_freq)

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Re-score ``candidates`` with exact BM25 against ``query``, reorder, optionally truncate.

        One tokenising pass builds postings for the query's terms only (documents lacking every
        query term keep score ``0``); the BM25 sum is accumulated over those postings; the list is
        returned newly-sorted by descending score with an ascending-``id`` tiebreak. An empty input
        returns ``[]``. Uses the caller-supplied ``stats`` when configured, else the candidate pool
        as its own corpus.
        """
        items = list(candidates)
        if not items:
            return []

        query_terms = set(_tokenize(query))
        # Per-query-term postings: term → list of (candidate index, term frequency in that doc).
        postings: dict[str, list[tuple[int, int]]] = {term: [] for term in query_terms}
        doc_lengths: list[int] = [0] * len(items)

        for index, candidate in enumerate(items):
            length = 0
            local_tf: dict[str, int] = {}
            for token in _tokenize(candidate.text):
                length += 1
                if token in query_terms:
                    local_tf[token] = local_tf.get(token, 0) + 1
            doc_lengths[index] = length
            for term, tf in local_tf.items():
                postings[term].append((index, tf))

        stats = self._supplied_stats or self._pool_stats(doc_lengths, postings)

        scores = [0.0] * len(items)
        for term, term_postings in postings.items():
            idf = stats.idf(term)
            if idf == 0.0 or not term_postings:
                continue
            for index, tf in term_postings:
                scores[index] += idf * self._term_saturation(tf, doc_lengths[index], stats.avgdl)

        rescored = [
            candidate.model_copy(update={"score": score})
            for candidate, score in zip(items, scores, strict=True)
        ]
        rescored.sort(key=lambda c: (-c.score, c.id))
        if self._top_k is not None:
            return rescored[: self._top_k]
        return rescored

    def _term_saturation(self, tf: int, doc_length: int, avgdl: float) -> float:
        """The BM25 tf-saturation/length-normalisation factor for one term in one document."""
        length_norm = 1.0 - self._b + self._b * (doc_length / avgdl if avgdl > 0.0 else 0.0)
        return (tf * (self._k1 + 1.0)) / (tf + self._k1 * length_norm)

    @staticmethod
    def _pool_stats(
        doc_lengths: Sequence[int], postings: dict[str, list[tuple[int, int]]]
    ) -> _CorpusStats:
        """Derive BM25 statistics from the candidate pool itself (pool-local IDF).

        ``N`` = pool size, ``avgdl`` = mean candidate length, and ``df(t)`` = the number of
        candidates in the pool that contain ``t`` (exactly the length of the query term's postings
        list, already built by the single scoring pass — no extra scan).
        """
        n = len(doc_lengths)
        total = sum(doc_lengths)
        avgdl = total / n if n else 0.0
        doc_freq = {term: len(term_postings) for term, term_postings in postings.items()}
        return _CorpusStats(doc_count=n, avgdl=avgdl, doc_freq=doc_freq)


__all__ = ["Bm25Rescorer"]
