"""The ``latence.retrieval`` component registry — query-time discovery + resolution (ADR-0049).

Query-time components (candidate processors now; the swappable reranker/fusion/sparse models later)
register under the ``latence.retrieval`` entry-point group — **separate from and parallel to** the
pipeline's ``latence.providers``. They are deliberately *not* pipeline Stage Capabilities and are
kept off the Stage DAG; a one-line config block selects one at query time.

:class:`RetrievalRegistry` discovers and lazily loads them (mirroring the pipeline's
``ProviderRegistry`` shape but self-contained, so this library carries no dependency on the pipeline
core). :func:`resolve_processor` turns a ``{"provider", "config"}`` block into a built, seam-checked
:class:`~latence_retrieval.processor.CandidateProcessor` — the same resolution shape the pipeline's
Export uses for its opt-in Embedder, so wiring a processor reads identically to wiring a Provider.
"""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

from latence_retrieval.fuser import Fuser
from latence_retrieval.maxsim import MultiVectorQueryEncoder
from latence_retrieval.processor import CandidateProcessor
from latence_retrieval.sparse import SparseQueryEncoder

if TYPE_CHECKING:
    from collections.abc import Mapping

ENTRY_POINT_GROUP = "latence.retrieval"


class RetrievalRegistry:
    """Discovers query-time components from the ``latence.retrieval`` entry-point group.

    Discovery is lazy and cached: entry points are enumerated once, then each is imported only when
    its name is first :meth:`load`-ed, so the registry stays cheap even with many components (some
    behind heavy extras) installed. This is the query-time twin of the pipeline's provider registry;
    it is intentionally a separate group so a query-time component is never mistaken for a Stage
    Capability (ADR-0049).
    """

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self._group = group
        self._entry_points: dict[str, metadata.EntryPoint] | None = None
        self._loaded: dict[str, type] = {}

    def _discover(self) -> dict[str, metadata.EntryPoint]:
        if self._entry_points is None:
            eps = metadata.entry_points(group=self._group)
            self._entry_points = {ep.name: ep for ep in eps}
        return self._entry_points

    def names(self) -> list[str]:
        """Every registered component name, sorted (stable ordering for listings/tests)."""
        return sorted(self._discover())

    def load(self, name: str) -> type:
        """Load and return the component class registered under ``name``.

        Raises ``KeyError`` (naming the available components) for an unknown name, and ``TypeError``
        if the entry point resolves to something that is not a class — a mis-registered entry point
        surfaces loudly rather than as a downstream ``()``-call failure.
        """
        if name in self._loaded:
            return self._loaded[name]
        eps = self._discover()
        if name not in eps:
            available = ", ".join(self.names()) or "<none>"
            msg = (
                f"No component registered as {name!r} in group {self._group!r}. "
                f"Available: {available}"
            )
            raise KeyError(msg)
        obj = eps[name].load()
        if not isinstance(obj, type):
            msg = f"Component {name!r} did not resolve to a class (got {type(obj)!r})."
            raise TypeError(msg)
        self._loaded[name] = obj
        return obj


def _build_from_spec(
    spec: Mapping[str, Any],
    registry: RetrievalRegistry | None,
    kind: str,
) -> object:
    """Load + construct the component named in ``spec`` (``{"provider", "config"}``).

    Shared by every ``resolve_*`` helper: it validates the ``provider`` name and the optional
    ``config`` mapping (``kind`` names the seam in the error text), loads the class from the
    ``latence.retrieval`` registry, and constructs it with its ``config``. The *seam check* (whether
    the built object satisfies a specific Protocol) is left to each caller, so a component is
    resolvable only through the resolver for the seam it actually implements.
    """
    provider_name = spec.get("provider")
    if not isinstance(provider_name, str) or not provider_name:
        msg = f"{kind} spec requires a non-empty string 'provider' entry-point name."
        raise ValueError(msg)
    component_config = spec.get("config", {})
    if not isinstance(component_config, dict):
        msg = f"{kind} spec 'config' must be a mapping if present."
        raise TypeError(msg)
    registry = registry or RetrievalRegistry()
    component_cls = registry.load(provider_name)
    return component_cls(component_config)


def resolve_processor(
    spec: Mapping[str, Any],
    registry: RetrievalRegistry | None = None,
) -> CandidateProcessor:
    """Build a :class:`~latence_retrieval.processor.CandidateProcessor` from a config block.

    ``spec`` is a ``{"provider": <entry-point name>, "config": {...}}`` mapping — the same shape the
    pipeline's Export uses to resolve its opt-in Embedder. The named component is loaded from the
    ``latence.retrieval`` registry, constructed with its ``config`` sub-mapping, and checked against
    the :class:`~latence_retrieval.processor.CandidateProcessor` seam so a component that does not
    implement ``process(query, candidates)`` fails at wiring time — never silently at query time.
    """
    processor = _build_from_spec(spec, registry, "processor")
    if not isinstance(processor, CandidateProcessor):
        provider_name = spec.get("provider")
        msg = (
            f"Component {provider_name!r} does not satisfy the CandidateProcessor seam "
            f"(needs a 'process(query, candidates)' method)."
        )
        raise TypeError(msg)
    return processor


def resolve_fuser(
    spec: Mapping[str, Any],
    registry: RetrievalRegistry | None = None,
) -> Fuser:
    """Build a :class:`~latence_retrieval.fuser.Fuser` from a config block.

    ``spec`` is a ``{"provider": <entry-point name>, "config": {...}}`` mapping (the same shape as
    every other query-time component). It selects the fusion algorithm by name —
    ``"fuser.rrf"`` or ``"fuser.weighted"`` — so swapping RRF for weighted fusion is a one-line
    config change (user story 36). The built component is checked against the
    :class:`~latence_retrieval.fuser.Fuser` seam, so a component that does not implement
    ``fuse(rankings, weights)`` fails at wiring time rather than silently at query time — and a
    :class:`~latence_retrieval.processor.CandidateProcessor` (a different shape) is *not* accepted
    here.
    """
    fuser = _build_from_spec(spec, registry, "fuser")
    if not isinstance(fuser, Fuser):
        provider_name = spec.get("provider")
        msg = (
            f"Component {provider_name!r} does not satisfy the Fuser seam "
            f"(needs a 'fuse(rankings, weights)' method)."
        )
        raise TypeError(msg)
    return fuser


def resolve_sparse_encoder(
    spec: Mapping[str, Any],
    registry: RetrievalRegistry | None = None,
) -> SparseQueryEncoder:
    """Build a :class:`~latence_retrieval.sparse.SparseQueryEncoder` from a config block.

    ``spec`` is a ``{"provider": <entry-point name>, "config": {...}}`` mapping (the same shape as
    every other query-time component). It selects the query-side sparse generator by name —
    ``"query_sparse.splade"`` reuses the ADR-0053 SPLADE composition — so swapping the sparse model
    is a one-line config change (user story 36) with the *same* config the index-time signal used
    (the name is kept distinct from the index-time ``sparse.splade`` Provider so the two seams'
    names stay disjoint, ADR-0049). The
    built component is checked against the
    :class:`~latence_retrieval.sparse.SparseQueryEncoder` seam, so a component that does not
    implement ``encode(query, ...)`` fails at wiring time rather than silently at query time — and a
    :class:`~latence_retrieval.processor.CandidateProcessor` (a different shape) is *not* accepted
    here.
    """
    encoder = _build_from_spec(spec, registry, "sparse encoder")
    if not isinstance(encoder, SparseQueryEncoder):
        provider_name = spec.get("provider")
        msg = (
            f"Component {provider_name!r} does not satisfy the SparseQueryEncoder seam "
            f"(needs an 'encode(query, ...)' method)."
        )
        raise TypeError(msg)
    return encoder


def resolve_multivector_encoder(
    spec: Mapping[str, Any],
    registry: RetrievalRegistry | None = None,
) -> MultiVectorQueryEncoder:
    """Build a :class:`~latence_retrieval.maxsim.MultiVectorQueryEncoder` from a config block.

    ``spec`` is a ``{"provider": <entry-point name>, "config": {...}}`` mapping (the same shape as
    every other query-time component). It selects the query-side multi-vector generator by name —
    ``"query_multivector.hashing"`` reuses the ADR-0050 index-time ``multivector.hashing`` reference
    — so the query multi-vector lands in the same space as the emitted corpus sidecar (the name is
    kept distinct from the index-time ``multivector.hashing`` Provider under ``latence.providers``
    so the two seams' names stay disjoint, ADR-0049). The built component is checked against the
    :class:`~latence_retrieval.maxsim.MultiVectorQueryEncoder` seam, so a component that does not
    implement ``encode(query)`` fails at wiring time rather than silently at query time — and a
    :class:`~latence_retrieval.processor.CandidateProcessor` (a different shape) is *not* accepted
    here.
    """
    encoder = _build_from_spec(spec, registry, "multi-vector encoder")
    if not isinstance(encoder, MultiVectorQueryEncoder):
        provider_name = spec.get("provider")
        msg = (
            f"Component {provider_name!r} does not satisfy the MultiVectorQueryEncoder seam "
            f"(needs an 'encode(query)' method)."
        )
        raise TypeError(msg)
    return encoder
