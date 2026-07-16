"""The query-time data contracts: :class:`Candidate`, :class:`Capability`, :class:`RetrievalQuery`.

These are the *interface* every query-time component speaks — the whole surface a caller must know
to drive the seam. They are Pydantic v2 models (the framework's serialization spine, ADR-0006), so
they round-trip losslessly through dict and JSON and reject unknown fields (``extra="forbid"``) — a
malformed candidate or query fails loudly at the boundary rather than corrupting a rank list.

Nothing here holds state or fetches anything (ADR-0048): a :class:`Candidate` is one already-fetched
search hit the *caller's* engine returned, and a :class:`RetrievalQuery` is a request the caller
hands to *its own* :class:`~latence_retrieval.backend.RetrievalBackend`. This module is pure data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Capability(StrEnum):
    """A first-stage capability a :class:`~latence_retrieval.backend.RetrievalBackend` may declare.

    The closed set a store advertises (CONTEXT ``RetrievalBackend`` — ``{dense, sparse, bm25,
    filter, multivector}``). ``DENSE``/``SPARSE``/``BM25``/``MULTIVECTOR`` are *query modes* (how
    the first-stage search is scored); ``FILTER`` is *orthogonal* — a metadata predicate any mode
    may carry. The orchestrator composes only the legs a backend declares and drops the rest
    (ADR-0052); the tooling itself runs none of them.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    BM25 = "bm25"
    FILTER = "filter"
    MULTIVECTOR = "multivector"


class Candidate(BaseModel):
    """One already-retrieved search hit — the unit every candidate processor transforms.

    A caller's first-stage engine produced this (ADR-0048: we never fetch it ourselves). It carries
    the minimum a downstream transform needs: a stable ``id`` (the record/chunk key in the caller's
    store), a ``score`` (the first-stage relevance the engine assigned — higher is more relevant),
    the ``text`` (the chunk content a reranker re-scores or a packer selects), and free-form
    ``metadata`` (provenance, page, offsets, filter fields — whatever the caller loaded, opaque to
    the seam). It is frozen so a processor cannot mutate a shared candidate in place (statelessness
    is structural, not by convention) and round-trips losslessly through :meth:`model_dump` /
    :meth:`model_validate` and their JSON forms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    score: float
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    """A first-stage search request a caller hands to *its own* backend (ADR-0048).

    The tooling never runs this — a :class:`~latence_retrieval.backend.RetrievalBackend` adapter
    passes it to the customer's engine. It is modelled here so the seam can validate it: ``mode`` is
    the scoring capability the search uses (dense/sparse/bm25/multivector — **not** ``FILTER``,
    which is a predicate, not a mode); the mode's payload must be present (a dense/multivector
    search needs ``vector``, a sparse search needs ``sparse_terms``, a lexical search needs
    ``text``); and an optional ``filter`` is a metadata predicate that additionally requires the
    backend to declare
    :attr:`Capability.FILTER`. ``top_k`` bounds the returned list. Construction is validated so a
    mode/payload contradiction (a dense query with no vector) fails at the boundary, leaving the
    backend to enforce only capability and dimension — the things only *it* knows.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Capability
    text: str | None = None
    vector: tuple[float, ...] | None = None
    sparse_terms: dict[str, float] | None = None
    top_k: int = Field(default=10, ge=1)
    filter: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _payload_matches_mode(self) -> RetrievalQuery:
        """Reject a mode whose payload is absent — a real store cannot search without it.

        ``FILTER`` is not a query mode (it is a predicate) so it is invalid as ``mode``; ``DENSE``
        and ``MULTIVECTOR`` require ``vector``; ``SPARSE`` requires ``sparse_terms``; ``BM25``
        requires a non-empty ``text``. This keeps mode/payload consistency out of every backend so
        each adapter enforces only what is store-specific (declared capabilities + vector width).
        """
        if self.mode is Capability.FILTER:
            msg = "RetrievalQuery.mode cannot be FILTER — filter is a predicate, not a query mode."
            raise ValueError(msg)
        if self.mode in (Capability.DENSE, Capability.MULTIVECTOR) and not self.vector:
            msg = f"RetrievalQuery.mode={self.mode.value!r} requires a non-empty 'vector'."
            raise ValueError(msg)
        if self.mode is Capability.SPARSE and not self.sparse_terms:
            msg = "RetrievalQuery.mode='sparse' requires non-empty 'sparse_terms'."
            raise ValueError(msg)
        if self.mode is Capability.BM25 and not self.text:
            msg = "RetrievalQuery.mode='bm25' requires non-empty 'text'."
            raise ValueError(msg)
        return self
