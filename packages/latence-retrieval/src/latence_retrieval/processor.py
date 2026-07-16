"""The candidate-processor seam: :class:`CandidateProcessor` + reference :class:`IdentityProcessor`.

A candidate processor is the query-time half of the tooling (CONTEXT ``Candidate processor``): a
pure transform ``(query, candidate list) → transformed list``. The caller already fetched the
candidates from its engine; a processor only reorders / rescores / selects / widens them and returns
a new list. It holds nothing and fetches nothing (ADR-0048) — the single-list transform shape shared
by ``rerank``, ``pack``, and ``expand`` (``fuse``, which merges *N* lists, is a distinct shape added
in its own ticket).

:class:`IdentityProcessor` is the trivial reference that proves the whole seam end to end —
register (entry point ``processor.identity``) → resolve by config
(:func:`~latence_retrieval.registry.resolve_processor`) → transform — while changing nothing about
the list. It is also the honest no-op baseline a pipeline can slot in to disable a stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from latence_retrieval.contracts import Candidate


@runtime_checkable
class CandidateProcessor(Protocol):
    """A pure query-time transform over a caller-supplied candidate list (ADR-0048).

    One method, :meth:`process`, taking the ``query`` (the text the candidates were retrieved for —
    a reranker/packer scores against it) and the ``candidates`` the caller already fetched, and
    returning a new list. The processor never fetches candidates itself and never mutates the input
    (:class:`~latence_retrieval.contracts.Candidate` is frozen), so the same inputs always yield the
    same output — statelessness is the contract, not a convention.
    """

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Transform ``candidates`` (retrieved for ``query``) into a new list; no input mutation."""
        ...


class IdentityProcessor:
    """The no-op reference processor: returns its input candidates unchanged (in a fresh list).

    The minimal implementation that exercises the full seam — discovery, config resolution, and the
    ``(query, candidates) → candidates`` transform — without any scoring logic. It returns a *new*
    list containing the same (frozen) candidates in the same order, so a caller can never observe a
    mutation of the list it passed in, and processing the same input twice yields identical output
    (no accumulated state). ``config`` is accepted (and ignored) so it is constructed exactly like
    every other registry component, ``Processor(config)``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Accepted for a uniform construction signature across the registry; the identity
        # transform has nothing to configure. Kept (not dropped) so a mis-passed config is a
        # visible no-op rather than a constructor TypeError that diverges from other processors.
        self._config: dict[str, Any] = dict(config or {})

    def process(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Return the candidates unchanged, order-preserved, in a new list (input never mutated)."""
        return list(candidates)
