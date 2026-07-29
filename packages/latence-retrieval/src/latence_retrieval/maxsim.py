# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
#
# DELIBERATE, BOUNDED EXCEPTION -- not a header nobody updated at the relicence. This
# repository's outbound licence is PolyForm-Noncommercial-1.0.0 (root LICENSE); this module is
# one of seven the copyright holder keeps under Apache-2.0 ON PURPOSE, so the query-time
# retrieval primitives stay permissively reusable -- commercially included -- by anyone who
# wants only them. The grant is his to make and safe to make: sole copyright holder, no
# third-party and no copyleft code in this file. The exact set is authoritative in LICENSE,
# section "SCOPE: THE RETAINED APACHE-2.0 MODULES", and pinned by
# packages/latence-core/tests/test_publish_readiness.py, which fails the build if an eighth
# file acquires the grant or one of the seven loses it.
"""The experimental MaxSim rerank path: :class:`MaxSimReranker` + query-side multi-vector gen.

MaxSim (Chamfer late-interaction) is the *experimental* rerank a caller runs over the **raw
multi-vectors** the index-time :class:`~latence_core.capability.MultiVectorEmbedder` emitted to the
``multivectors.parquet`` sidecar (ADR-0050): for a query multi-vector ``Q`` and a document
multi-vector ``D``, ``MaxSim(Q, D) = Σ_q max_d <q, d>`` — every query token's best match against any
document token, summed. It is the *exact* late-interaction score the MUVERA ``fde_embedding`` column
only *approximates* (ADR-0050) — so a caller who kept the raw multi-vectors can rerank the FDE
first-stage hits with the true score.

This module ships three things, one small seam each:

* :func:`max_sim` — the **reference scorer**: real, dependency-free MaxSim math (ADR-0050: real, not
  faked). A numpy/SIMD kernel is the deferred SOTA accelerator (ADR-0049/0050); the reference is
  pure-Python, deterministic (rounded), and byte-reproducible.
* :class:`MultiVectorQueryEncoder` — the **query-side multi-vector generator** seam
  (``query text → multi-vector``), the query clock of the index-time
  :class:`~latence_core.capability.MultiVectorEmbedder`. It is a genuine two-adapter seam (the
  codebase-design bar): :class:`HashingMultiVectorQueryEncoder` is the real adapter that **reuses**
  the index-time embedder so the query lands in the identical space as the corpus (matching the
  index-time contract), and :class:`~latence_retrieval.testing.FakeMultiVectorEmbedder` is the
  shipped faithful fake every offline test drives.
* :class:`MaxSimReranker` — the **rerank** transform (CONTEXT ``Rerank``, the MaxSim path): a
  :class:`~latence_retrieval.processor.CandidateProcessor` that scores each candidate's sidecar
  multi-vectors against the query's multi-vector, replaces the score, and reorders. It **accepts** a
  :class:`MultiVectorQueryEncoder` (dependency injection) — a test hands it a fake, production the
  real reuse adapter — so there is no seam with a single implementation. It holds nothing and
  fetches nothing (ADR-0048): the raw document multi-vectors are the sidecar rows the *caller*
  loaded onto each :class:`~latence_retrieval.contracts.Candidate`'s ``metadata``, not something
  this transform retrieves.

Experimental and off by default (ADR-0050): the multi-vector path is real and evaluable but blesses
no per-corpus quality guarantee, and nothing wires it unless a caller selects ``processor.maxsim``.
The heavy reuse (``latence-core``'s index-time embedder, and any real ColBERT-family encoder an
adopter injects) rides this package's ``[multivector]`` extra and is imported lazily, so the
pure-Python retrieval core — and registry discovery — never needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_retrieval.contracts import Candidate
from latence_retrieval.errors import MaxSimError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_core.capability import MultiVectorEmbedder

_ROUND_NDIGITS = 8


def max_sim(query: Sequence[Sequence[float]], document: Sequence[Sequence[float]]) -> float:
    """The reference MaxSim (Chamfer) score of a query multi-vector against a document multi-vector.

    ``MaxSim(Q, D) = Σ over query tokens q of max over document tokens d of <q, d>`` — each query
    token contributes its single best dot-product with any document token; the sum is the exact
    late-interaction relevance (ADR-0050). Real math, no dependency, deterministic (the result is
    rounded to :data:`_ROUND_NDIGITS` so a seeded run is byte-reproducible, mirroring the FDE
    reference). A query token with **no** document token to match (an empty document multi-vector)
    contributes ``0.0`` rather than crashing; an empty query multi-vector scores ``0.0``.

    A document token vector whose width does not match the query token it is dotted against is
    rejected with a :class:`~latence_retrieval.errors.MaxSimError` (the two were encoded in
    different spaces — a real MaxSim kernel cannot dot them either; anti-false-green, never a silent
    truncation).
    """
    total = 0.0
    for q in query:
        best: float | None = None
        for d in document:
            dot = _dot(q, d)
            if best is None or dot > best:
                best = dot
        if best is not None:
            total += best
    return round(total, _ROUND_NDIGITS)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-width vectors; a width mismatch is a typed :class:`MaxSimError`."""
    if len(a) != len(b):
        msg = (
            f"MaxSim cannot dot a query token of width {len(a)} with a document token of width "
            f"{len(b)}: the query and the sidecar multi-vectors were encoded in different spaces "
            "(use the same MultiVectorEmbedder at index and query time)."
        )
        raise MaxSimError(msg)
    return sum(float(x) * float(y) for x, y in zip(a, b, strict=True))


@runtime_checkable
class MultiVectorQueryEncoder(Protocol):
    """A query-side multi-vector generator: ``query text → multi-vector`` (ADR-0050).

    The whole seam the MaxSim rerank depends on for the query representation — deliberately one
    method. Given the query text it returns the query's multi-vector (one fixed-width token vector
    per token), in the **same** vector space the index-time
    :class:`~latence_core.capability.MultiVectorEmbedder` wrote the corpus sidecar in, so the query
    and document tokens are dot-comparable. It holds no state, so identical inputs yield the same
    multi-vector.
    """

    def encode(self, query: str) -> list[list[float]]:
        """Return the query's multi-vector (one ``dimension``-wide vector per token)."""
        ...


class HashingMultiVectorQueryEncoder:
    """Generate a query multi-vector by **reusing** the index-time ``MultiVectorEmbedder`` (0050).

    The query clock of the index-time signal generator: it does **not** re-implement the encoding
    (which would drift from the emitted sidecar) — it *reuses* a
    :class:`~latence_core.capability.MultiVectorEmbedder` directly (default: the in-core
    ``multivector.hashing`` reference,
    :class:`~latence_core.signals.HashingMultiVectorEmbedder`), so the query multi-vector is
    byte-for-byte the same construction the corpus was built with (the AC: query representations
    match the index-time contract). A real ColBERT-family encoder is the pluggable alternative an
    adopter **injects** (same Protocol) so the seam is never single-adapter.

    Config (passed straight through to the reused index-time embedder, so query- and index-time are
    configured identically — user story 36's one-line swap):
        dimension: int — the fixed token-vector width. Default matches the in-core reference (64).

    Packaging: ``latence-core`` (the home of the reference embedder) rides this package's
    ``[multivector]`` extra and is imported lazily on first :meth:`encode`, so constructing the
    encoder — which the registry does — never imports it or needs the extra. An adopter who wants
    query-side multi-vector generation installs ``latence-retrieval[multivector]`` (or injects their
    own embedder).
    """

    def __init__(
        self, config: dict[str, Any] | None = None, *, embedder: MultiVectorEmbedder | None = None
    ) -> None:
        self._config: dict[str, Any] = dict(config or {})
        # Dependency injection: an explicit index-time embedder (tests, or reuse of the exact
        # instance the corpus was built with) wins; otherwise the in-core reference is built lazily
        # from config on first encode. No latence_core import happens until then.
        self._embedder: MultiVectorEmbedder | None = embedder

    def _resolve_embedder(self) -> MultiVectorEmbedder:
        if self._embedder is None:
            try:
                from latence_core.signals import HashingMultiVectorEmbedder
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                msg = (
                    "The 'latence-core' package is required for query-side multi-vector generation "
                    "with the reference encoder. Install it with: "
                    "pip install 'latence-retrieval[multivector]' (or inject your own "
                    "MultiVectorEmbedder)."
                )
                raise ImportError(msg) from exc
            self._embedder = HashingMultiVectorEmbedder(self._config)
        return self._embedder

    def encode(self, query: str) -> list[list[float]]:
        """Return the query's multi-vector via the reused index-time embedder (one row per token).

        Runs the reused ``encode_multivector`` over the single query text and returns the one
        resulting multi-vector. Empty/degenerate query text yields the empty multi-vector ``[]`` (no
        tokens), exactly as the index-time seam does — never a crash.
        """
        (multivector,) = list(self._resolve_embedder().encode_multivector([query]))
        return multivector


class MaxSimReranker:
    """Re-score a candidate list by MaxSim over its sidecar multi-vectors and reorder it (ADR-0050).

    The experimental late-interaction :class:`~latence_retrieval.processor.CandidateProcessor`
    (CONTEXT ``Rerank``, MaxSim path): :meth:`process` encodes the query into a multi-vector (via
    the injected :class:`MultiVectorQueryEncoder`), scores **every** candidate's raw document
    multi-vectors — the ``multivectors.parquet`` sidecar rows the caller loaded onto each
    candidate's ``metadata`` — against it with :func:`max_sim`, replaces each candidate's ``score``
    with that MaxSim score, and returns a new list sorted by descending score (ties broken by
    ascending ``id``, the determinism the fuse/rerank seams hold). The input list is never mutated
    (:class:`~latence_retrieval.contracts.Candidate` is frozen; a fresh list is returned), and the
    reranker keeps no state across calls (ADR-0048): the document multi-vectors are the *caller's*
    loaded sidecar, never something this transform fetches.

    Config:
        multivector_key: str — the ``candidate.metadata`` key under which the caller placed each
            document's sidecar multi-vector (``list[list[float]]``, keyed by record id in
            ``multivectors.parquet``). Default ``"multivectors"``. A candidate lacking it is a
            :class:`~latence_retrieval.errors.MaxSimError` (the caller did not load the sidecar) —
            never a silently-fabricated score (ADR-0050 anti-false-green).
        top_k: int | None — if set (a positive int), the reranked list is truncated to its top
            ``top_k``. Default ``None`` keeps every candidate, reordered.
        encoder: dict — the config for the default :class:`HashingMultiVectorQueryEncoder` (e.g.
            ``{"dimension": 64}``), used only when no ``encoder`` is injected. The query encoder
            must match the index-time embedder that wrote the sidecar (same class, same config), or
            the dot-products are cross-space and :func:`max_sim` rejects them.

    The query encoder is resolved lazily on first :meth:`process` (constructing the
    ``MaxSimReranker`` — which the registry does — never imports ``latence-core`` or the
    ``[multivector]`` extra).

    Experimental and off by default (ADR-0050): nothing wires this unless a caller selects
    ``processor.maxsim``; it makes no per-corpus quality guarantee.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        encoder: MultiVectorQueryEncoder | None = None,
    ) -> None:
        cfg = dict(config or {})
        top_k = cfg.get("top_k")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1
        ):
            msg = f"maxsim 'top_k' must be a positive integer or omitted, got {top_k!r}."
            raise ValueError(msg)
        self._top_k: int | None = top_k
        key = cfg.get("multivector_key", "multivectors")
        if not isinstance(key, str) or not key:
            msg = f"maxsim 'multivector_key' must be a non-empty string, got {key!r}."
            raise ValueError(msg)
        self._key = key
        encoder_config = cfg.get("encoder", {})
        if not isinstance(encoder_config, dict):
            got = type(encoder_config).__name__
            msg = f"maxsim 'encoder' config must be a mapping if present, got {got}."
            raise TypeError(msg)
        self._encoder_config: dict[str, Any] = dict(encoder_config)
        # Dependency injection: an explicit query encoder (tests / a pre-built reuse adapter) wins;
        # otherwise the reference reuse adapter is built lazily on first use (no import at build).
        self._encoder: MultiVectorQueryEncoder | None = encoder

    def _resolve_encoder(self) -> MultiVectorQueryEncoder:
        if self._encoder is None:
            self._encoder = HashingMultiVectorQueryEncoder(self._encoder_config)
        return self._encoder

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Re-score ``candidates`` by MaxSim against ``query``, reorder, and (optionally) truncate.

        Encodes the query once into a multi-vector, scores every candidate's sidecar multi-vectors
        against it with :func:`max_sim`, replaces each candidate's ``score``, and sorts descending
        with an ascending-``id`` tiebreak. Returns a new list; the input is never mutated. An empty
        input returns ``[]`` without touching the encoder (so no ``[multivector]`` extra is needed
        to rerank nothing). A candidate missing its sidecar multi-vector — or one whose token width
        does not match the query's — is a :class:`~latence_retrieval.errors.MaxSimError`, not a
        fabricated score.
        """
        candidates = list(candidates)
        if not candidates:
            return []
        query_mv = self._resolve_encoder().encode(query)
        rescored: list[Candidate] = []
        for candidate in candidates:
            document_mv = candidate.metadata.get(self._key)
            if document_mv is None:
                msg = (
                    f"candidate {candidate.id!r} has no multi-vector under "
                    f"metadata[{self._key!r}]: MaxSim rerank scores the raw multivectors.parquet "
                    "sidecar the caller loads onto each candidate (ADR-0050), so this candidate "
                    "cannot be scored."
                )
                raise MaxSimError(msg)
            score = max_sim(query_mv, _as_multivector(document_mv, candidate.id))
            rescored.append(candidate.model_copy(update={"score": score}))
        rescored.sort(key=lambda c: (-c.score, c.id))
        if self._top_k is not None:
            return rescored[: self._top_k]
        return rescored


def _as_multivector(value: object, candidate_id: str) -> list[list[float]]:
    """Coerce a candidate's sidecar payload into a multi-vector, rejecting a malformed shape.

    A real ``multivectors.parquet`` row is a ``list<list<double>>``. A payload that is not a list of
    token vectors of numbers is a :class:`~latence_retrieval.errors.MaxSimError` (anti-false-green:
    the caller loaded the wrong column, and a real MaxSim kernel would fault too — never a coerced,
    invented score). Per-token *width* mismatches against the query are caught later, in
    :func:`_dot`, where the query token is known.
    """
    if not isinstance(value, (list, tuple)):
        msg = (
            f"candidate {candidate_id!r} multi-vector must be a list of token vectors "
            f"(list<list<float>>), got {type(value).__name__}."
        )
        raise MaxSimError(msg)
    multivector: list[list[float]] = []
    for token in value:
        if not isinstance(token, (list, tuple)):
            msg = (
                f"candidate {candidate_id!r} multi-vector token must be a list of floats, got "
                f"{type(token).__name__}."
            )
            raise MaxSimError(msg)
        try:
            multivector.append([float(x) for x in token])
        except (TypeError, ValueError) as exc:
            msg = f"candidate {candidate_id!r} multi-vector token holds a non-numeric value."
            raise MaxSimError(msg) from exc
    return multivector


__all__ = [
    "HashingMultiVectorQueryEncoder",
    "MaxSimReranker",
    "MultiVectorQueryEncoder",
    "max_sim",
]
