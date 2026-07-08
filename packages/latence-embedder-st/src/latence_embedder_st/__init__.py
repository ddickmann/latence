"""latence-embedder-st: the learned sentence-transformers Embedder Provider (W1-embed, #93).

Exposes :class:`~latence_embedder_st.provider.SentenceTransformersEmbedder`, registered under the
``latence.providers`` entry-point group as ``embedding.sentence_transformers`` (ADR-0004). Importing
this package never pulls torch — the heavy ``sentence-transformers`` import is deferred to first use
(ADR-0016).
"""

from __future__ import annotations

from latence_embedder_st.provider import SentenceTransformersEmbedder

__all__ = ["SentenceTransformersEmbedder"]
