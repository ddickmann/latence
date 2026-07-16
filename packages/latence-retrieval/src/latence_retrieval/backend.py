"""The bring-your-own-engine seam: :class:`RetrievalBackend` (ADR-0048, ADR-0052).

This is the one place first-stage search enters the query-time tooling — and it enters as the
*caller's* engine, never ours. A ``RetrievalBackend`` is a thin adapter over a store the customer
already runs (Qdrant, Weaviate, Azure AI Search, Elastic, …): it declares *what that store can do*
(:attr:`~RetrievalBackend.capabilities`) and runs one first-stage :meth:`~RetrievalBackend.search`,
returning the candidates the tooling then transforms. The library holds no index and owns no store;
this Protocol is the whole contract, and the reference production adapter (Qdrant) plus the faithful
:class:`~latence_retrieval.testing.FakeBackend` are its two implementations (a real seam has two).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery


@runtime_checkable
class RetrievalBackend(Protocol):
    """A capability-declaring adapter over the customer's first-stage retrieval engine.

    Two members, both about the customer's store:

    * :attr:`capabilities` — the closed set of :class:`~latence_retrieval.contracts.Capability` the
      store supports (dense/sparse/bm25/filter/multivector). The orchestrator reads this to compose
      only the legs the store can run and skip the rest (ADR-0052); it is the store's honest
      self-description, so a backend must not advertise a mode it cannot serve.
    * :meth:`search` — run one first-stage lookup for a
      :class:`~latence_retrieval.contracts.RetrievalQuery` and return the hits as
      :class:`~latence_retrieval.contracts.Candidate`s. This is the *only* fetch in the whole
      library, and it runs against the caller's engine, not a store we own (ADR-0048).

    An adapter must reject a query the store cannot serve — a mode or filter outside
    :attr:`capabilities` (:class:`~latence_retrieval.errors.BackendCapabilityError`) or a dense
    vector of the wrong width (:class:`~latence_retrieval.errors.BackendDimensionError`) — exactly
    as the real store would, never a silent empty result.
    """

    @property
    def capabilities(self) -> frozenset[Capability]:
        """The first-stage capabilities the underlying store declares it can serve."""
        ...

    def search(self, query: RetrievalQuery) -> list[Candidate]:
        """Run one first-stage search on the customer's engine; return up to ``query.top_k`` hits.

        Rejects a query the store cannot serve (capability or dimension mismatch) rather than
        returning wrong or empty results. Returns candidates in descending score order.
        """
        ...
