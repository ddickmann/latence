"""The one SparseVector → query-terms conversion both RFC harnesses use (T-RET-4).

:meth:`~latence_retrieval.sparse.SpladeQueryEncoder.encode` returns a
:class:`~latence_core.capability.SparseVector` — two parallel columns, ``indices`` and ``values``,
the shape a vector DB's sparse index ingests. :attr:`RetrievalQuery.sparse_terms` wants
``dict[str, float]``.

That mismatch was written twice and got it right once: ``run_benchmark.py`` converted,
``build_questions.py`` passed the vector straight through. Nothing offline constructs a real
encoder, so the type error only ever fired on a pod. One shared function removes the opportunity to
disagree, which is the actual fix — the second call site would otherwise have been free to drift
again the next time either file was touched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def sparse_vector_to_terms(vector: Any) -> dict[str, float]:
    """Convert a :class:`SparseVector` (or an already-converted mapping) to query terms.

    Term ids become string keys because that is what the query contract and every sparse backend
    expect; weights are coerced to ``float`` so a numpy scalar from a real model cannot leak a
    non-JSON-serialisable type into the query.

    Accepting a mapping unchanged makes this **idempotent**: a caller that already converted cannot
    double-convert into nonsense, and neither call site has to know which side of the seam it is on.
    An empty vector maps to an empty mapping — never a run of zeros (a ``SparseVector``
    invariant).
    """
    if isinstance(vector, Mapping):
        return {str(k): float(v) for k, v in vector.items()}
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is None or values is None:
        msg = (
            f"cannot convert {type(vector).__name__} to sparse query terms: expected a "
            "SparseVector (parallel .indices/.values) or a mapping."
        )
        raise TypeError(msg)
    return {str(i): float(v) for i, v in zip(indices, values, strict=True)}
