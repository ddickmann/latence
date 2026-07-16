"""latence-splade: the structured-composition SPLADE ``SparseEmbedder`` Provider (ADR-0053, #162).

Exposes :class:`~latence_splade.provider.SpladeCompositionEmbedder`, registered under the
``latence.providers`` entry-point group as ``sparse.splade`` (ADR-0004/0049). It is the *real*
multilingual SPLADE sibling of the in-core deterministic ``sparse.hashing`` reference: its emitted
sparse vector is not plain ``SPLADE(text)`` but the ADR-0053 **structured composition** — the
model's term-weights unioned with the pipeline's own confidence-gated NER entity surface terms and
selected metadata terms, generated over the corpus's *redacted* text by default so it never
re-leaks PII the corpus removed.

The composition itself (:mod:`~latence_splade.composition`) is pure, dependency-free, and the same
whether it runs at index time (this Provider, emitting ``sparse_indices``/``sparse_values``) or at
query time over a query string — ADR-0053 applies one composition to both clocks.

Importing this package never pulls torch/transformers — the heavy model import is deferred to first
use (ADR-0016), and the model seam (:class:`~latence_splade.model.SpladeModel`) is dependency-
injected so the composition + Provider are exercised offline with a faithful fake.
"""

from __future__ import annotations

from latence_splade.composition import EntityTerm, SpladeInput, compose
from latence_splade.model import SpladeModel, TransformersSpladeModel
from latence_splade.provider import SpladeCompositionEmbedder

__all__ = [
    "EntityTerm",
    "SpladeCompositionEmbedder",
    "SpladeInput",
    "SpladeModel",
    "TransformersSpladeModel",
    "compose",
]
