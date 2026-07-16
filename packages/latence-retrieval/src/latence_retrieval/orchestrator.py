"""The Level-2 orchestrator: :func:`retrieve` — capability-adaptive, stateless (ADR-0052/0048).

The framework's two consumption levels (CONTEXT ``Retrieval tooling``): **Level 1** is à-la-carte
— call ``fuse`` / ``rerank`` / ``pack`` / ``expand`` on candidates you already fetched. **Level 2**
is this optional orchestrator: hand it the query signals and *your own* engine (a
:class:`~latence_retrieval.backend.RetrievalBackend`), and it chains the whole opinionated,
config-overridable pipeline around it —

    query-signal-gen → **fetch-per-modality** → **fuse** → *(your candidate processors)* → pack

— composing **only the legs the backend's declared capabilities and the query's emitted signals
support, and dropping the rest gracefully** (ADR-0052). It is the exact orchestration the MCP
server (a later ticket) will expose with an MCP face.

**Strictly stateless (ADR-0048).** The orchestrator holds no index and runs no search of its own:
the *only* fetch in the whole flow is :meth:`~latence_retrieval.backend.RetrievalBackend.search`, on
the customer's engine. It accepts its dependencies (the backend, the built fuser/processors) and
returns a :class:`RetrievalResult` — no hidden state, no side effects, so identical inputs yield an
identical result.

**Query-signal-gen is the caller's, injected as data.** The orchestrator does not construct query
encoders (a dense query embedder is not even in this library, and the sparse/multi-vector encoders
carry heavy extras). The caller runs query-signal-gen — the Level-1 query-side encoders
(:class:`~latence_retrieval.sparse.SpladeQueryEncoder`, a dense embedder of its choice) — and hands
the results in a :class:`QueryBundle`. The orchestrator's job is the *adaptive composition*: map
each signal in the bundle onto the fetch legs the backend can actually serve.

**How the pipeline adapts (AC2).** For every first-stage query mode the backend declares
(dense/sparse/bm25/multivector — ``FILTER`` is a predicate, not a mode), a fetch leg runs **iff the
bundle carries that mode's signal**; a declared mode with no signal, or a signal for a mode the
store does not declare, is dropped. The surviving legs each run one :meth:`search` on the backend;
their ranked lists are **fused** into one (skipped when a single leg fired — one list needs no
fusion, and re-ranking it by RRF would discard the store's own scores); then the caller-configured
**candidate processors** (``expand`` / ``rerank`` / ``pack`` …) run in order, each a stateless
``(query, candidates) → candidates`` transform. Which modes actually fired is returned on
:attr:`RetrievalResult.fetched_modes`, so the adaptive drop is observable through the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from latence_retrieval.contracts import Candidate, Capability, RetrievalQuery
from latence_retrieval.errors import OrchestrationError
from latence_retrieval.fuser import RrfFuser

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from latence_retrieval.backend import RetrievalBackend
    from latence_retrieval.fuser import Fuser
    from latence_retrieval.processor import CandidateProcessor
    from latence_retrieval.registry import RetrievalRegistry

_DEFAULT_FETCH_TOP_K = 50

# The first-stage query modes a fetch leg can run — every Capability except FILTER, which is an
# orthogonal predicate (a mode's payload, not a mode). Ordered by value so the fetch legs, the fused
# input order, and RetrievalResult.fetched_modes are deterministic regardless of set iteration.
_QUERY_MODES: tuple[Capability, ...] = tuple(
    sorted((c for c in Capability if c is not Capability.FILTER), key=lambda c: c.value)
)


class QueryBundle(BaseModel):
    """The query-signal-gen output the orchestrator fans out across a backend's fetch legs.

    One request, all the per-modality signals the caller generated for it. ``text`` is the raw query
    (always required — ``rerank`` scores against it, a ``bm25`` leg searches it) and each optional
    field is the signal for one first-stage mode: ``vector`` (a dense — or multi-vector — query
    embedding), ``sparse_terms`` (a SPLADE/BM25 term-weight map from
    :class:`~latence_retrieval.sparse.SpladeQueryEncoder`), and an optional metadata ``filter``
    predicate. A mode's leg fires only when its signal is present *and non-empty*, so leaving a
    field unset (or empty) is how a caller says "I did not generate that signal" — the orchestrator
    drops that leg (AC2). It is frozen and rejects unknown fields (``extra="forbid"``), so a
    malformed bundle fails at the boundary rather than silently mis-routing a fetch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    vector: tuple[float, ...] | None = None
    sparse_terms: dict[str, float] | None = None
    filter: dict[str, Any] | None = None


class RetrievalResult(BaseModel):
    """The orchestrator's output: the final candidates plus *which* fetch legs actually fired.

    ``candidates`` is the fully-composed, ordered result (fetched → fused → processed → truncated).
    ``fetched_modes`` records the first-stage modes the orchestrator actually ran on the backend, in
    deterministic order — the observable proof of the capability-adaptive composition (AC2): a
    caller (and a test) reads it to see that a mode the store lacked, or one the query had no signal
    for, was dropped, and that ``len > 1`` legs were fused. It is frozen and round-trips losslessly
    through :meth:`model_dump` / JSON like every contract in this library.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[Candidate, ...]
    fetched_modes: tuple[Capability, ...]


@dataclass(frozen=True)
class OrchestrationPipeline:
    """The opinionated, config-overridable recipe :func:`retrieve` runs (AC1).

    It carries the *built* components the orchestrator composes — accept dependencies, do not create
    them — so a test injects fakes/references and production injects real adapters, and neither the
    orchestrator nor this recipe hides a constructor:

    * ``fuser`` — the :class:`~latence_retrieval.fuser.Fuser` that merges the per-mode ranked lists
      when more than one leg fires. Default: :class:`~latence_retrieval.fuser.RrfFuser` (rank-only,
      scale-free — the robust default across incomparable dense/sparse/bm25 score scales).
    * ``processors`` — the ordered :class:`~latence_retrieval.processor.CandidateProcessor` chain
      run after fusion (``expand`` → ``rerank`` → ``pack``, or any subset, in the caller's order).
      Each is a stateless single-list transform; the default is empty (fetch + fuse only), because
      ``rerank`` needs a license-cleared model and ``pack`` needs a token budget — neither has a
      safe hidden default (ADR-0012), so a processor leg is a deliberate opt-in.
    * ``fetch_top_k`` — how many hits each fetch leg asks the backend for (the first-stage depth the
      processors then refine). Positive int, default ``50``.
    * ``fuse_weights`` — optional per-mode pull for weighted fusion, keyed by mode name
      (``{"dense": 2.0, "bm25": 1.0}``); a fired mode with no entry weighs ``1.0``. Ignored when a
      single leg fires (nothing to fuse).
    * ``final_top_k`` — optional truncation of the fully-composed list to its top ``n``. Default
      ``None`` keeps every candidate.

    :meth:`from_config` builds the whole recipe from a plain config mapping by resolving the fuser
    and each processor through the :class:`~latence_retrieval.registry.RetrievalRegistry` — the
    config-driven path the MCP server loads from operator files, reusing the same ``resolve_*``
    seams the rest of the library uses.
    """

    fuser: Fuser = field(default_factory=RrfFuser)
    processors: tuple[CandidateProcessor, ...] = ()
    fetch_top_k: int = _DEFAULT_FETCH_TOP_K
    fuse_weights: Mapping[str, float] | None = None
    final_top_k: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.fetch_top_k, int)
            or isinstance(self.fetch_top_k, bool)
            or self.fetch_top_k < 1
        ):
            msg = (
                "OrchestrationPipeline 'fetch_top_k' must be a positive integer, "
                f"got {self.fetch_top_k!r}."
            )
            raise ValueError(msg)
        if self.final_top_k is not None and (
            not isinstance(self.final_top_k, int)
            or isinstance(self.final_top_k, bool)
            or self.final_top_k < 1
        ):
            msg = (
                "OrchestrationPipeline 'final_top_k' must be a positive integer or None, "
                f"got {self.final_top_k!r}."
            )
            raise ValueError(msg)
        if self.fuse_weights is not None:
            valid = {c.value for c in _QUERY_MODES}
            for name, weight in self.fuse_weights.items():
                if name not in valid:
                    msg = (
                        f"OrchestrationPipeline 'fuse_weights' key {name!r} is not a first-stage "
                        f"query mode; expected one of {sorted(valid)}."
                    )
                    raise ValueError(msg)
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    msg = (
                        f"OrchestrationPipeline 'fuse_weights[{name!r}]' must be a number, "
                        f"got {weight!r}."
                    )
                    raise ValueError(msg)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        registry: RetrievalRegistry | None = None,
    ) -> OrchestrationPipeline:
        """Build the recipe from a config mapping, resolving components through the registry.

        ``config`` keys (all optional): ``fuser`` — a ``{"provider", "config"}`` block resolved via
        :func:`~latence_retrieval.registry.resolve_fuser` (default ``fuser.rrf``); ``processors``
        — a list of such blocks resolved **in order** via
        :func:`~latence_retrieval.registry.resolve_processor` (the ordered chain, default empty);
        ``fetch_top_k`` / ``fuse_weights`` / ``final_top_k`` — passed through. This is the
        operator-config path (the MCP server's), reusing the same discovery + seam-check the rest of
        the library uses, so a component that does not satisfy its seam fails at build time — never
        at query time.
        """
        from latence_retrieval.registry import resolve_fuser, resolve_processor

        cfg = dict(config or {})
        known = {"fuser", "processors", "fetch_top_k", "fuse_weights", "final_top_k"}
        unknown = set(cfg) - known
        if unknown:
            msg = f"OrchestrationPipeline.from_config got unknown keys: {sorted(unknown)}."
            raise ValueError(msg)

        fuser_spec = cfg.get("fuser")
        fuser: Fuser = resolve_fuser(fuser_spec, registry) if fuser_spec is not None else RrfFuser()

        processors_spec = cfg.get("processors", [])
        if not isinstance(processors_spec, (list, tuple)):
            got = type(processors_spec).__name__
            msg = f"OrchestrationPipeline.from_config 'processors' must be a list, got {got}."
            raise TypeError(msg)
        processors = tuple(resolve_processor(spec, registry) for spec in processors_spec)

        fuse_weights = cfg.get("fuse_weights")
        if fuse_weights is not None and not isinstance(fuse_weights, Mapping):
            got = type(fuse_weights).__name__
            msg = f"OrchestrationPipeline.from_config 'fuse_weights' must be a mapping, got {got}."
            raise TypeError(msg)

        return cls(
            fuser=fuser,
            processors=processors,
            fetch_top_k=cfg.get("fetch_top_k", _DEFAULT_FETCH_TOP_K),
            fuse_weights=dict(fuse_weights) if fuse_weights is not None else None,
            final_top_k=cfg.get("final_top_k"),
        )


def retrieve(
    bundle: QueryBundle,
    backend: RetrievalBackend,
    pipeline: OrchestrationPipeline | None = None,
) -> RetrievalResult:
    """Run the capability-adaptive Level-2 pipeline over ``backend`` for ``bundle`` (ADR-0052/0048).

    Composes and runs, in order: **fetch-per-modality** (one backend :meth:`search` per query mode
    the backend declares *and* the bundle has a signal for — every other mode is dropped), **fuse**
    (merge the ranked lists into one; skipped when a single leg fired), then the pipeline's ordered
    **candidate processors** (``expand`` / ``rerank`` / ``pack`` …), then an optional
    ``final_top_k`` truncation. Returns a :class:`RetrievalResult` carrying the composed candidates
    and the modes
    that actually fired.

    Stateless (ADR-0048): the backend's :meth:`search` is the only fetch, and it runs on the
    *customer's* engine; the orchestrator holds nothing and mutates nothing. A backend fault during
    a leg it accepted (a wrong-width vector, a store failure) propagates as the backend's own typed
    error; a *planning* failure — no runnable fetch leg, or a filter the backend cannot serve — is
    an :class:`~latence_retrieval.errors.OrchestrationError` raised before any fetch.
    """
    pipeline = pipeline or OrchestrationPipeline()
    legs = _plan_fetch_legs(bundle, backend, pipeline.fetch_top_k)
    fetched_modes = tuple(mode for mode, _query in legs)

    # The ONLY fetch in the whole flow, and it runs on the caller's engine (ADR-0048). Backend
    # faults (dimension/capability/search) propagate unchanged — they are the store's to raise.
    rankings = [backend.search(query) for _mode, query in legs]

    if len(rankings) == 1:
        # One list needs no fusion; fusing a lone list by RRF would discard the store's own scores
        # that the downstream processors (pack's value, expand's anchor) rely on.
        candidates: list[Candidate] = list(rankings[0])
    else:
        weights = _fuse_weights_for(fetched_modes, pipeline.fuse_weights)
        candidates = pipeline.fuser.fuse(rankings, weights)

    for processor in pipeline.processors:
        candidates = processor.process(bundle.text, candidates)

    if pipeline.final_top_k is not None:
        candidates = candidates[: pipeline.final_top_k]

    return RetrievalResult(candidates=tuple(candidates), fetched_modes=fetched_modes)


def _plan_fetch_legs(
    bundle: QueryBundle,
    backend: RetrievalBackend,
    fetch_top_k: int,
) -> list[tuple[Capability, RetrievalQuery]]:
    """Plan one :class:`RetrievalQuery` per fireable mode; raise if nothing (or a filter) can run.

    A mode fires iff the backend declares it *and* the bundle carries its signal (AC2). A supplied
    ``filter`` rides every leg iff the backend declares ``FILTER``; a filter the backend cannot
    serve is a loud :class:`~latence_retrieval.errors.OrchestrationError` (never silently dropped —
    that would return unfiltered hits). Zero fireable legs is likewise an ``OrchestrationError``.
    """
    capabilities = backend.capabilities
    wants_filter = bundle.filter is not None
    if wants_filter and Capability.FILTER not in capabilities:
        msg = (
            "query bundle carries a metadata filter but the backend does not declare the 'filter' "
            f"capability (declared: {sorted(c.value for c in capabilities)}); refusing to drop the "
            "filter and return unfiltered results."
        )
        raise OrchestrationError(msg)
    leg_filter = bundle.filter if wants_filter else None

    legs: list[tuple[Capability, RetrievalQuery]] = []
    for mode in _QUERY_MODES:
        if mode not in capabilities:
            continue
        payload = _mode_payload(mode, bundle)
        if payload is None:
            continue
        query = RetrievalQuery(
            mode=mode,
            text=bundle.text,
            top_k=fetch_top_k,
            filter=leg_filter,
            **payload,
        )
        legs.append((mode, query))

    if not legs:
        declared = sorted(c.value for c in capabilities if c is not Capability.FILTER)
        msg = (
            f"no runnable fetch leg: the backend declares {declared} but the query bundle carries "
            "no matching signal (a dense/multivector mode needs 'vector', a sparse mode needs "
            "'sparse_terms', a bm25 mode needs 'text')."
        )
        raise OrchestrationError(msg)
    return legs


def _mode_payload(mode: Capability, bundle: QueryBundle) -> dict[str, Any] | None:
    """The mode's first-stage payload from the bundle, or ``None`` when its signal is absent/empty.

    Mirrors :class:`~latence_retrieval.contracts.RetrievalQuery`'s own mode→payload rule: dense and
    multivector need a non-empty ``vector``; sparse needs non-empty ``sparse_terms``; bm25 needs
    only the (always-present) ``text``, so its payload is empty.
    """
    if mode in (Capability.DENSE, Capability.MULTIVECTOR):
        return {"vector": bundle.vector} if bundle.vector else None
    if mode is Capability.SPARSE:
        return {"sparse_terms": bundle.sparse_terms} if bundle.sparse_terms else None
    if mode is Capability.BM25:
        return {}
    # Unreachable: _QUERY_MODES excludes FILTER and enumerates every other Capability. Guarding
    # rather than asserting keeps a future mode from silently defaulting to "always fires".
    msg = f"orchestrator has no fetch-payload rule for mode {mode.value!r}."  # pragma: no cover
    raise OrchestrationError(msg)  # pragma: no cover


def _fuse_weights_for(
    modes: Sequence[Capability],
    fuse_weights: Mapping[str, float] | None,
) -> list[float] | None:
    """Align ``fuse_weights`` (mode-name → weight) to the fired ``modes``; ``None`` for uniform.

    Built in the same order the ranked lists were fetched so weight *i* pulls list *i*; a fired mode
    with no configured weight defaults to ``1.0``.
    """
    if fuse_weights is None:
        return None
    return [float(fuse_weights.get(mode.value, 1.0)) for mode in modes]


__all__ = [
    "OrchestrationPipeline",
    "QueryBundle",
    "RetrievalResult",
    "retrieve",
]
