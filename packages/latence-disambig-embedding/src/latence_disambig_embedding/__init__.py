"""latence-disambig-embedding: the learned (embedding-based) Disambiguation Provider (W1, #94).

Exposes :class:`~latence_disambig_embedding.provider.EmbeddingDisambiguator`, registered under the
``latence.providers`` entry-point group as ``disambiguation.embedding`` (ADR-0004) — a true drop-in
for the in-core ``disambiguation.cascade`` reference (same ``DisambiguationRecord`` contract). It
embeds each mention's **surface + surrounding context** (the pod proved a bare-surface embedding
over-merges distinct short surfaces; surface+context separates them cleanly) and merges by cosine
similarity with the contract's no-silent-over-merge audit. It resolves its Embedder from the plugin
registry (default the in-core, deterministic ``embedding.hashing``), so importing this package pulls
no heavy dep (ADR-0016).
"""

from __future__ import annotations

from latence_disambig_embedding.provider import EmbeddingDisambiguator

__all__ = ["EmbeddingDisambiguator"]
