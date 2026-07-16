"""Reference index-time signal generators — deterministic, dependency-free (ADR-0049/0050).

The retrieval-tooling clock-split (ADR-0049) grows the pipeline three new index-time
:class:`~latence_core.capability` signal generators alongside the existing dense
:class:`~latence_core.capability.Embedder`:

* :class:`SparseHashingEmbedder` (``sparse.hashing``) — a :class:`SparseEmbedder` reference;
* :class:`HashingMultiVectorEmbedder` (``multivector.hashing``) — a :class:`MultiVectorEmbedder`
  reference;
* :class:`ReferenceFdeConverter` (``fde.reference``) — an :class:`FdeConverter` reference.

Each is a **pure transform that holds nothing** (ADR-0048) and names **no model** — the CPU-viable,
zero-dependency reference so the seam + its additive Export emission are exercisable end to end
offline (ADR-0007/0016) with no weights and no license to verify (ADR-0012), exactly as
``embedding.hashing`` is for the dense seam. The blessed *real* Providers — the multilingual SPLADE
composition (ADR-0053) and the canonical MUVERA FDE + ColBERT-family multi-vector encoder (ADR-0050)
— are pluggable packages, not core dependencies. These are **references, not those models**: they
are honest (deterministic, real math, never a passthrough that lies — the anti-false-green bar), but
they make no retrieval-quality claim.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING, Any

from latence_core.capability import SparseVector
from latence_core.embedding import _TOKEN, HashingEmbedder, _hash_token
from latence_core.errors import ConfigError
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_DEFAULT_VOCAB_SIZE = 4096
_DEFAULT_MULTIVECTOR_DIMENSION = 64
_DEFAULT_FDE_BUCKETS = 8


def _reference_profile(memory_mb: int) -> ProviderProfile:
    """The shared descriptor for a pure-code, CPU, deterministic, Apache-2.0 reference generator.

    Every reference signal generator is a pure-Python part of ``latence-core`` (Apache-2.0, verified
    per ADR-0012 — the package's own code, weights-and-code trivially the same): it ships no model
    weights (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call
    cost, so a bake-off / conformance run reads honest, non-fabricated columns off it (ADR-0036 §2).
    """
    return ProviderProfile(
        compute="cpu",
        memory_mb=memory_mb,
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


class SparseHashingEmbedder:
    """Deterministic hashing :class:`~latence_core.capability.SparseEmbedder` (the CPU reference).

    Config:
        vocab_size: int  — the term-index space; every emitted index lies in ``[0, vocab_size)``.
            Default 4096.

    Each text is lowercased and tokenised on alphanumerics; each token is folded into a term id by
    stable hash (``sha256`` mod ``vocab_size``, so it is byte-stable across processes — never the
    salted builtin ``hash``) and its **term frequency** is the weight. Colliding tokens sum their
    weights. The result is a canonical :class:`~latence_core.capability.SparseVector` (ascending
    indices), so a seeded run is byte-reproducible. Empty text yields the empty ``SparseVector()``.

    This is a *reference*, not SPLADE: it exercises the sparse seam + emission offline. The blessed
    multilingual-SPLADE composition (model term-weights ⊕ confidence-gated NER ⊕ metadata, ADR-0053)
    is the pluggable ``latence-splade`` package.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile(memory_mb=16)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        vocab_size = int(config.get("vocab_size", _DEFAULT_VOCAB_SIZE))
        if vocab_size < 1:
            msg = f"SparseEmbedder vocab_size must be >= 1, got {vocab_size}."
            raise ConfigError(msg)
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode_sparse(self, texts: Iterable[str]) -> Iterator[SparseVector]:
        for text in texts:
            yield self._encode_one(text)

    def _encode_one(self, text: str) -> SparseVector:
        terms: dict[int, float] = {}
        for token in _TOKEN.findall(text.lower()):
            idx = _hash_token(token) % self._vocab_size
            terms[idx] = terms.get(idx, 0.0) + 1.0
        return SparseVector.from_terms(terms)


class HashingMultiVectorEmbedder:
    """Deterministic per-token hashing :class:`~latence_core.capability.MultiVectorEmbedder`.

    Config:
        dimension: int  — the fixed width of every token vector. Default 64.

    Each text is lowercased and tokenised on alphanumerics; **each token** becomes one
    ``dimension``-wide vector — the token's own :class:`~latence_core.embedding.HashingEmbedder`
    embedding — so the multi-vector is exactly the sequence of per-token dense vectors (a faithful
    ColBERT-shaped signal without a model). Deterministic, so a seeded run is byte-reproducible.
    Empty text yields an empty multi-vector (no tokens).

    This is a *reference*, not a ColBERT encoder; it exercises the multi-vector seam + the sidecar
    emission path offline. Experimental and off by default (ADR-0050).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile(memory_mb=32)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        dimension = int(config.get("dimension", _DEFAULT_MULTIVECTOR_DIMENSION))
        if dimension < 1:
            msg = f"MultiVectorEmbedder dimension must be >= 1, got {dimension}."
            raise ConfigError(msg)
        # Reuse the dense hashing embedder for the per-token vector so a token's multi-vector row is
        # byte-identical to its ``embedding.hashing`` embedding — one tokenisation + hash, one
        # source of truth for determinism.
        self._token_embedder = HashingEmbedder({"dimension": dimension})

    @property
    def dimension(self) -> int:
        return self._token_embedder.dimension

    def encode_multivector(self, texts: Iterable[str]) -> Iterator[list[list[float]]]:
        for text in texts:
            tokens = _TOKEN.findall(text.lower())
            yield list(self._token_embedder.embed(tokens))


class ReferenceFdeConverter:
    """Deterministic data-oblivious :class:`~latence_core.capability.FdeConverter` (CPU reference).

    Config:
        input_dimension: int  — the width of each input token vector (required; the converter
            REJECTS a token vector of any other width — the faithful contract a real converter
            enforces, anti-false-green).
        buckets: int  — the number of SimHash space partitions; must be a power of two. Default 8.
        seed: int  — the deterministic seed for the fixed hyperplanes. Default 0.

    A canonical-shape, **training-free, data-oblivious** MUVERA-style Fixed-Dimensional Encoding: a
    fixed set of ``log2(buckets)`` random hyperplanes (deterministic from ``seed``) partitions each
    token vector into one of ``buckets`` cells by the sign pattern of its projections; the token
    vectors in a cell are **summed**; the output is the concatenation of the per-cell sums — one
    dense vector of width ``buckets × input_dimension`` whose dot-product approximates the
    Chamfer/MaxSim similarity, so it lands in an ordinary dense index (ADR-0050). An empty
    multi-vector yields the zero vector. Deterministic (fixed hyperplanes, rounded sums), so a
    seeded run is byte-reproducible.

    This is a genuine FDE reduction — NOT a passthrough that lies — but it is a *reference*: it
    makes no MaxSim-approximation quality guarantee on any given corpus. The quality-blessed
    canonical MUVERA reference (and the deferred regularizer-refined converter) is the pluggable
    ``fde.muvera`` Provider (ADR-0050). Experimental and off by default.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return _reference_profile(memory_mb=32)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        input_dimension = int(config.get("input_dimension", 0))
        if input_dimension < 1:
            msg = (
                "FdeConverter requires a positive 'input_dimension' (the width of each input "
                f"token vector); got {input_dimension}."
            )
            raise ConfigError(msg)
        buckets = int(config.get("buckets", _DEFAULT_FDE_BUCKETS))
        if buckets < 1 or (buckets & (buckets - 1)) != 0:
            msg = f"FdeConverter 'buckets' must be a power of two >= 1, got {buckets}."
            raise ConfigError(msg)
        self._input_dimension = input_dimension
        self._buckets = buckets
        self._bits = buckets.bit_length() - 1  # log2(buckets); 0 bits ⇒ a single bucket
        self._seed = int(config.get("seed", 0))
        # The fixed hyperplanes: one ``input_dimension``-wide plane per SimHash bit, deterministic
        # from the seed (so two runs — and two processes — share the exact same partition).
        self._planes = [self._hyperplane(bit) for bit in range(self._bits)]

    @property
    def dimension(self) -> int:
        return self._buckets * self._input_dimension

    def convert(self, multivectors: Iterable[list[list[float]]]) -> Iterator[list[float]]:
        for multivector in multivectors:
            yield self._convert_one(multivector)

    def _convert_one(self, multivector: list[list[float]]) -> list[float]:
        sums = [0.0] * self.dimension
        for token_vector in multivector:
            if len(token_vector) != self._input_dimension:
                msg = (
                    "FdeConverter received a token vector of width "
                    f"{len(token_vector)}, expected {self._input_dimension} "
                    "(the declared 'input_dimension')."
                )
                raise ConfigError(msg)
            bucket = self._bucket_of(token_vector)
            base = bucket * self._input_dimension
            for i, value in enumerate(token_vector):
                sums[base + i] += value
        return [round(v, 8) for v in sums]

    def _bucket_of(self, token_vector: list[float]) -> int:
        bucket = 0
        for bit, plane in enumerate(self._planes):
            projection = sum(a * b for a, b in zip(plane, token_vector, strict=True))
            if projection >= 0.0:
                bucket |= 1 << bit
        return bucket

    def _hyperplane(self, bit: int) -> list[float]:
        """A deterministic ``input_dimension``-wide hyperplane for one SimHash bit.

        Each component is a stable pseudo-random value in ``[-1, 1)`` derived from ``(seed, bit,
        component)`` via ``sha256`` — no numpy, no PRNG state, so two independent processes build
        the identical partition (the determinism the byte-identical export relies on).
        """
        plane: list[float] = []
        for component in range(self._input_dimension):
            payload = f"{self._seed}:{bit}:{component}".encode()
            digest = hashlib.sha256(payload).digest()
            # Map the first 8 bytes to a float in [-1, 1): unsigned 64-bit / 2**63 - 1.
            raw = struct.unpack(">Q", digest[:8])[0]
            plane.append(raw / float(1 << 63) - 1.0)
        return plane


__all__ = [
    "HashingMultiVectorEmbedder",
    "ReferenceFdeConverter",
    "SparseHashingEmbedder",
]
