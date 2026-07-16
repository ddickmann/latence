"""The faithful fake backend — the second, offline adapter of the :class:`RetrievalBackend` seam.

A real seam needs two implementations; the reference production one is Qdrant (a later ticket), and
:class:`FakeBackend` is the offline one every query-time test drives through (the spec's "faithful
fake backend"). It is shipped (not hidden in one package's tests) so *every* downstream package —
the orchestrator, the MCP server, each new processor — tests against the same honest double.

"Faithful" is the whole point (anti-false-green): the fake rejects **exactly** what a real vector
store rejects — a query mode or filter the store never declared, and a dense query vector of the
wrong width — and it never returns candidates outside its declared capabilities. It does not
re-implement a store's ANN maths (it returns its seeded corpus in score order, honouring ``top_k``
and the filter); it faithfully mirrors the store's *contract* — shape, ordering, and rejections —
which is what the tooling is tested against.

The same module ships :class:`FakeCrossEncoder`, the offline second adapter of the
:class:`~latence_retrieval.rerank.CrossEncoderModel` seam, so ``rerank`` (and, later, the
orchestrator and MCP server) is tested against an honest double instead of downloading weights, and
:class:`FakeMultiVectorEmbedder`, the offline second adapter of the
:class:`~latence_retrieval.maxsim.MultiVectorQueryEncoder` seam, so the experimental MaxSim rerank
path (ADR-0050) is tested against an honest double instead of a real ColBERT-family encoder.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import TYPE_CHECKING

from latence_retrieval.contracts import Candidate, Capability
from latence_retrieval.errors import BackendCapabilityError, BackendDimensionError
from latence_retrieval.multihop import _traverse, _validate_max_hops

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from latence_retrieval.contracts import RetrievalQuery

_VECTOR_MODES = frozenset({Capability.DENSE, Capability.MULTIVECTOR})
_TOKEN = re.compile(r"[a-z0-9]+")


class FakeBackend:
    """An offline, faithful :class:`~latence_retrieval.backend.RetrievalBackend` over a set corpus.

    Constructed with the candidates it would "return", the capabilities it declares, and (when it
    declares a vector mode) the fixed ``dimension`` of its index. :meth:`search` validates the
    query against those declarations exactly as a real store would, then returns the matching
    candidates in descending score order, truncated to ``top_k``. It holds only the fixed corpus it
    was handed — no durable index, no first-stage ANN — so it can stand in for any store's
    *contract* without being one.

    Args:
        candidates: the fixed corpus this backend can return (a real store's indexed rows).
        capabilities: the modes/predicates it declares — a subset of :class:`Capability`.
        dimension: the fixed vector width of its index; required iff it declares ``DENSE`` or
            ``MULTIVECTOR`` (a real dense index cannot exist without one), rejected otherwise.
    """

    def __init__(
        self,
        candidates: Iterable[Candidate],
        capabilities: Iterable[Capability],
        dimension: int | None = None,
    ) -> None:
        self._corpus: list[Candidate] = list(candidates)
        self._capabilities: frozenset[Capability] = frozenset(capabilities)
        declares_vector = bool(self._capabilities & _VECTOR_MODES)
        if declares_vector and dimension is None:
            msg = "FakeBackend declaring a dense/multivector capability requires a 'dimension'."
            raise ValueError(msg)
        if dimension is not None and dimension < 1:
            msg = f"FakeBackend 'dimension' must be >= 1, got {dimension}."
            raise ValueError(msg)
        self._dimension = dimension

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The first-stage capabilities this fake declares (what a real store would advertise)."""
        return self._capabilities

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Validate the query against the declarations, then return matched hits in score order.

        Rejects (as a real store would): a ``mode`` outside :attr:`capabilities`
        (:class:`~latence_retrieval.errors.BackendCapabilityError`); a ``filter`` when ``FILTER``
        was not declared (same); and a dense/multivector ``vector`` whose width is not the index
        ``dimension`` (:class:`~latence_retrieval.errors.BackendDimensionError`). Then applies the
        metadata filter (exact per-field equality; a candidate lacking a filtered field is excluded,
        as a store's predicate would), sorts by descending score, and truncates to ``query.top_k``.
        """
        if query.mode not in self._capabilities:
            msg = (
                f"backend does not declare capability {query.mode.value!r}; "
                f"declared: {sorted(c.value for c in self._capabilities)}."
            )
            raise BackendCapabilityError(msg)
        if query.filter is not None and Capability.FILTER not in self._capabilities:
            msg = "backend does not declare the 'filter' capability but a filter was supplied."
            raise BackendCapabilityError(msg)
        if query.mode in _VECTOR_MODES:
            # A vector is guaranteed present by RetrievalQuery's own validation for these modes.
            width = len(query.vector or ())
            if width != self._dimension:
                msg = (
                    f"query vector has width {width} but the index dimension is {self._dimension}."
                )
                raise BackendDimensionError(msg)
        matched = [c for c in self._corpus if _matches_filter(c, query.filter)]
        matched.sort(key=lambda c: c.score, reverse=True)
        return matched[: query.top_k]


def _matches_filter(candidate: Candidate, predicate: dict[str, object] | None) -> bool:
    """Exact per-field metadata match, mirroring a store's filter (missing field ⇒ no match)."""
    if not predicate:
        return True
    return all(candidate.metadata.get(key) == value for key, value in predicate.items())


class FakeCrossEncoder:
    """A faithful offline :class:`~latence_retrieval.rerank.CrossEncoderModel` — the 2nd adapter.

    The real adapter is :class:`~latence_retrieval.rerank.SentenceTransformersCrossEncoder` (a
    learned checkpoint via ``sentence-transformers``); this is the offline double every ``rerank``
    test drives (shipped, not test-local, so downstream packages test against the same honest
    double). It is **faithful to the contract, not the semantics** (exactly as the SPLADE/backend
    fakes are): it does not re-implement a learned cross-encoder — it scores query↔text **lexical
    token overlap**, a deterministic, weights-free proxy where more shared terms means a higher
    score.

    What it mirrors exactly (anti-false-green — it rejects what the real API rejects and returns
    what a real call returns):

    * **one score per text, in input order** — a real ``CrossEncoder.predict`` returns exactly one
      score per pair; this returns ``len(texts)`` floats, never more or fewer.
    * **string inputs only** — a real tokenizer raises on a non-string ``query`` or text; this
      raises :class:`TypeError` rather than silently coercing, so a mis-typed input fails offline.
    * **empty batch → empty list** — ``predict([])`` returns nothing; so does this, no model touch.

    It does **not** invent extra behaviour a real call lacks (no kwargs it would reject, no metadata
    a real score has no place to carry).
    """

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score each text by lexical token overlap with ``query``; one float per text, in order."""
        if not isinstance(query, str):
            msg = f"FakeCrossEncoder query must be a string, got {type(query).__name__}."
            raise TypeError(msg)
        texts = list(texts)
        query_terms = set(_TOKEN.findall(query.lower()))
        scores: list[float] = []
        for text in texts:
            if not isinstance(text, str):
                msg = f"FakeCrossEncoder text must be a string, got {type(text).__name__}."
                raise TypeError(msg)
            text_terms = set(_TOKEN.findall(text.lower()))
            scores.append(float(len(query_terms & text_terms)))
        return scores


class FakeMultiVectorEmbedder:
    """A faithful offline :class:`~latence_retrieval.maxsim.MultiVectorQueryEncoder` (2nd adapter).

    The real adapter is :class:`~latence_retrieval.maxsim.HashingMultiVectorQueryEncoder` (which
    reuses the index-time ``multivector.hashing`` reference, or a real ColBERT-family encoder an
    adopter injects); this is the offline double every MaxSim test drives (shipped, not test-local,
    so downstream packages test against the same honest double). It is **faithful to the contract,
    not the semantics of a learned encoder** (exactly as the cross-encoder/backend fakes are): it
    does not re-implement ColBERT — it maps each token to a deterministic **unit** vector by stable
    hash, so the same token always yields the same vector (a self-dot of ``1.0``) and distinct
    tokens are near-orthogonal. That makes MaxSim over a document that shares *k* of the query's
    tokens ``≈ k`` — a deterministic, weights-free planted signal a correctness test can assert on.

    What it mirrors exactly (anti-false-green — it rejects what a real encoder rejects and returns
    what a real call returns):

    * **one ``dimension``-wide vector per token, in order** — a real multi-vector is exactly the
      sequence of per-token vectors; this returns one unit vector per whitespace/alphanumeric token.
    * **empty/degenerate text → the empty multi-vector** ``[]`` — a text with no tokens has no token
      vectors; so does this (never a crash).
    * **string input only** — a real tokenizer raises on a non-string ``query``; this raises
      :class:`TypeError` rather than silently coercing, so a mis-typed input fails offline.

    It does **not** invent behaviour a real call lacks (no metadata a real token vector has no place
    to carry).
    """

    def __init__(self, dimension: int = 8) -> None:
        if dimension < 1:
            msg = f"FakeMultiVectorEmbedder dimension must be >= 1, got {dimension}."
            raise ValueError(msg)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """The fixed width of every token vector (what a real MultiVectorEmbedder declares)."""
        return self._dimension

    def encode(self, query: str) -> list[list[float]]:
        """Map ``query`` to its multi-vector: one deterministic unit vector per token, in order."""
        if not isinstance(query, str):
            msg = f"FakeMultiVectorEmbedder query must be a string, got {type(query).__name__}."
            raise TypeError(msg)
        return [self._token_vector(token) for token in _TOKEN.findall(query.lower())]

    def _token_vector(self, token: str) -> list[float]:
        raw: list[float] = []
        for i in range(self._dimension):
            digest = hashlib.sha256(f"{token}:{i}".encode()).digest()
            # First 8 bytes → a stable float in [-1, 1): the same token always hashes identically.
            value = struct.unpack(">Q", digest[:8])[0]
            raw.append(value / float(1 << 63) - 1.0)
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


class FakeGraphSource:
    """A faithful offline :class:`~latence_retrieval.multihop.GraphSource` — the seam's test double.

    The real adapters are the embedded :class:`~latence_retrieval.multihop.DuckDBGraphSource` (over
    the emitted graph parquet), the :class:`~latence_retrieval.multihop.NetworkxGraphSource`
    reference, and the optional :class:`~latence_retrieval.multihop.Neo4jGraphSource`; this is the
    always-available double every offline multi-hop ``expand`` test drives (shipped, not test-local,
    so downstream packages — the orchestrator, the MCP server — test against the same honest
    double).

    It is **faithful because it *is* the canonical semantics** the real engines must reproduce: it
    runs :func:`~latence_retrieval.multihop._traverse` (undirected BFS, minimum hop distance, seeds
    excluded) over a fixed in-memory edge list — so it is not an independent re-implementation that
    could drift from the engines, it is the reference each engine is proven equal to on a fixture
    graph (anti-false-green). A test plants a small graph as ``(source, target)`` edge tuples
    exactly
    as ``graph-edges.parquet`` holds them and asserts which candidates a hop bound lifts.

    What it mirrors exactly (rejects what the real sources reject, returns what a real call
    returns):

    * **empty seeds → ``{}``** — nothing to expand; no engine is touched, so neither is this.
    * **a hop bound below 1 or above the ceiling → :class:`ValueError`** — the bounded-depth guard
      every adapter enforces at the boundary
      (:func:`~latence_retrieval.multihop._validate_max_hops`).
    * **a seed absent from the graph contributes nothing** — a real traversal from a missing node
      yields no neighbours; so does this (never a crash).
    * **the seeds themselves are excluded** from the returned neighbourhood, keyed to the *minimum*
      hop distance — the exact contract a caller relies on.

    It invents no behaviour a real engine lacks (no node a traversal would not reach, no distance a
    shortest path would not report).
    """

    def __init__(self, edges: Iterable[tuple[str, str]]) -> None:
        self._edges: list[tuple[str, str]] = [(str(s), str(t)) for s, t in edges]

    def neighbors(self, seeds: Collection[str], max_hops: int) -> dict[str, int]:
        """Traverse the planted edge list; ``node id → minimum hop distance``, seeds excluded."""
        _validate_max_hops(max_hops)
        seed_list = list(dict.fromkeys(seeds))
        if not seed_list:
            return {}
        return _traverse(self._edges, seed_list, max_hops)
