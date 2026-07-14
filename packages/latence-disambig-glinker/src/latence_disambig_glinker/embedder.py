"""The STS label-embedding seam the resolver's embedding-merge rung uses.

The source ``resolution/entity_resolver.py`` merged semantically-equivalent canonical entities by
calling ``ontology.embeddings.embed_labels(texts, task="STS")`` (EmbeddingGemma / a
sentence-transformers fallback) and ``cosine_similarity_matrix``. In the framework the STS
embedder is the resolved :class:`~latence_core.capability.Embedder` Provider — the default learned
one is **IBM Granite Embedding r2** (Apache-2.0, ADR-0045), NOT EmbeddingGemma / vllm-factory.

:class:`LabelEmbedder` adapts ANY framework Embedder to the ``embed_labels(texts) -> list of
L2-normalized vectors`` shape the ported resolver's CPU-reachable path expects (cosine == dot
product), so the enterprise merge algorithm is byte-for-byte identical — only the model behind the
STS vectors changed. Kept **pure-Python + numpy-free** on this path (like the in-core
``disambiguation.embedding`` Provider), so the CPU-first dev env stays numpy-free (ADR-0016); numpy
enters ONLY in the OPTIONAL FAISS-GPU path (:mod:`.entity_resolver`), imported lazily with faiss.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable


class LabelEncoder(Protocol):
    """The narrow seam the resolver's embedding-merge rung depends on (test-fakeable)."""

    def embed_labels(self, texts: list[str]) -> list[list[float]]:
        """Embed labels to a list of L2-normalized vectors (cosine == dot product)."""
        ...


class _FrameworkEmbedder(Protocol):
    """Structural view of the framework Embedder capability (the ``embed`` method)."""

    def embed(self, texts: Iterable[str]) -> Iterable[list[float]]: ...


class LabelEmbedder:
    """Adapt a framework :class:`~latence_core.capability.Embedder` to the resolver's STS seam.

    Wraps the resolved Embedder (e.g. ``embedding.sentence_transformers`` = Granite r2) and
    L2-normalizes each per-text vector so cosine similarity is a plain dot product — exactly the
    invariant the source ``embed_labels`` guaranteed. Output order matches input order (the Embedder
    contract), so the resolver's ``embeddings[i]`` aligns to ``group[i]``.
    """

    def __init__(self, embedder: _FrameworkEmbedder) -> None:
        self._embedder = embedder

    def embed_labels(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_l2_normalize(list(v)) for v in self._embedder.embed(texts)]


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector so cosine similarity reduces to the dot product (zero-safe)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-9:
        return vec
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is degenerate/length-mismatched).

    The pure-Python stand-in for the source's ``cosine_similarity_matrix`` cell (the vectors the
    resolver compares are L2-normalized, so this is the dot product); the path stays numpy-free.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)
