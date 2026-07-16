"""latence-muvera: the canonical, training-free MUVERA FDE ``FdeConverter`` Provider (ADR-0050).

Exposes :class:`~latence_muvera.converter.MuveraFdeConverter`, registered under the
``latence.providers`` entry-point group as ``fde.muvera`` (ADR-0004/0049, #163). It is the *real*,
quality-blessed sibling of the in-core deterministic ``fde.reference``: its dense vector is a real
MUVERA Fixed-Dimensional Encoding whose dot-product **approximates the Chamfer / MaxSim similarity**
(the query = sum / document = average-and-fill-empty asymmetry the reference omits), so
late-interaction (ColBERT-style) quality can be approximated inside a customer's ordinary **dense**
ANN index with no special store, emitted at Export as the ``fde_embedding`` column.

MUVERA FDE is **data-oblivious and training-free**, so importing this package pulls no model and no
weights — the converter is pure-Python, dependency-free, and deterministic, exercised for real in
the CPU-first offline dev + conformance environment (ADR-0007/0016). Experimental and off by default
(ADR-0050): it ships real and evaluable, but blesses no per-corpus quality guarantee — whether
FDE-alone suffices versus the deferred regularizer-refined converter is the open research question.
"""

from __future__ import annotations

from latence_muvera.converter import MuveraFdeConverter

__all__ = ["MuveraFdeConverter"]
