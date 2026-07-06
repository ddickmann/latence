"""Reference Embedder — a deterministic, dependency-free hashing embedder (opt-in, S9).

The S9 opt-in Embedder Capability (ADR-0017: embeddings are optional so any vector DB can
ingest the RAG corpus) needs a CPU-viable, zero-dependency reference so the seam is
exercisable end to end offline (ADR-0007/0016) — with **no model, no weights, and no license
to verify** (ADR-0012). This is that reference: a hashing embedder that maps text to a fixed
``dimension``-wide L2-normalised vector via a token-hash bag-of-words. It is deterministic
(the same text always yields the same vector, so a seeded run is byte-identical) but it is
**not** a semantic embedder — the default learned Embedder (``multilingual-e5-small``, MIT)
is a pluggable Provider package, not a core dependency.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_DIMENSION = 64


def _hash_token(token: str) -> int:
    """A stable non-negative hash of a token (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class HashingEmbedder:
    """Deterministic hashing Embedder Provider (the CPU reference; opt-in, not required).

    Config:
        dimension: int  — the fixed embedding width. Default 64.

    Each text is lowercased, tokenised on alphanumerics, and folded into a ``dimension``-wide
    bag-of-words vector by token hash, then L2-normalised (a zero vector for empty text). The
    vector is byte-stable across processes (the token hash is sha256-based, not the salted
    builtin ``hash``), so a seeded run is reproducible.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        dimension = int(config.get("dimension", _DEFAULT_DIMENSION))
        if dimension < 1:
            msg = f"Embedder dimension must be >= 1, got {dimension}."
            raise ValueError(msg)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        for text in texts:
            yield self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            vector[_hash_token(token) % self._dimension] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [round(v / norm, 8) for v in vector]
