"""The stateless, hardened MCP server core: :class:`RetrievalServer` (ADR-0052, #174).

This is the deep module the whole ticket is about. Its **interface is tiny** — :meth:`list_tools`
(the rendered, config-driven tool surface) and :meth:`call` (one tool invocation → one
:class:`~latence_retrieval.mcp.envelope.ResponseEnvelope`) — and a lot sits behind it: bearer auth,
rate limiting, argument validation, the escalating-ladder dispatch over the customer's engine, the
uniform envelope, and structured logging. The MCP-protocol face
(:mod:`latence_retrieval.mcp.protocol`) is a thin translation of exactly these two methods, so the
*tool-call boundary* — not the wire — is the whole test surface (ADR-0052, spec §"what a good test
is here").

**It accepts its dependencies and creates none** (codebase-design): the customer's
:class:`~latence_retrieval.backend.RetrievalBackend`, the
:class:`~latence_retrieval.orchestrator.OrchestrationPipeline` recipe, the ``ServerCopy``, the
``BearerAuth`` and ``RateLimiter``, an optional query-signal generator, and an optional logger. It
**returns** envelopes and holds **no corpus and no cross-call state** (ADR-0048): the only fetch is
the backend's own ``search`` (via :func:`~latence_retrieval.orchestrator.retrieve`), and nothing a
tool call produces is retained for the next one — statelessness is structural.

**The escalating ladder (ADR-0052).** ``search`` (L1) runs the capability-adaptive orchestrator;
``more_like_this`` (L2) re-runs it on a hit's own content for breadth; ``get_context`` (L3) projects
hits the agent already holds into grounded, provenance-carrying chunks (no fetch — stateless);
``list_sources`` / ``index_status`` orient the agent over the configured backend. Agent-facing
names, descriptions, and parameter docs are all read from the (frozen, operator-trusted)
``ServerCopy`` — an argument can never reach the copy, so the tool descriptions cannot become a
prompt-injection surface.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from latence_retrieval.contracts import Candidate
from latence_retrieval.errors import (
    BackendCapabilityError,
    BackendDimensionError,
    CandidateProcessorError,
    OrchestrationError,
    QuerySignalError,
    RetrievalError,
)
from latence_retrieval.mcp.copy import (
    GET_CONTEXT,
    INDEX_STATUS,
    LIST_SOURCES,
    MORE_LIKE_THIS,
    SEARCH,
    ServerCopy,
)
from latence_retrieval.mcp.envelope import ErrorCode, ResponseEnvelope
from latence_retrieval.mcp.security import BearerAuth, RateLimiter, log_call
from latence_retrieval.orchestrator import (
    OrchestrationPipeline,
    QueryBundle,
    RetrievalResult,
    retrieve,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable, Mapping

    from latence_retrieval.backend import RetrievalBackend

# The injected query-signal generator: raw query text -> the extra bundle signals (a dense
# ``vector`` and/or ``sparse_terms``) the caller's own encoders produced. Kept a plain callable (not
# a Protocol) because the seam that *varies* is the two-adapter RetrievalBackend, not this — the
# operator wires whatever encoders their store needs, and ``None`` means text-only (a bm25 leg).
QuerySignalGen = Callable[[str], "Mapping[str, Any]"]


class SourceDescriptor(BaseModel):
    """One retrieval source the server is configured over (for ``list_sources``).

    Operator-supplied, purely descriptive orientation for the agent: a ``name``, an optional
    ``description``, and the ``capabilities`` (search modes) it offers. It is configuration, not
    corpus state — ``list_sources`` returns these unchanged, touching no index (ADR-0048).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    capabilities: tuple[str, ...] = ()


class _HitInput(BaseModel):
    """One hit an agent hands back to ``get_context`` — the id/text/metadata a prior search gave.

    Stateless by construction: the agent supplies the *content* (this server holds no corpus to look
    an id up in), and ``get_context`` projects it into a grounded chunk. Frozen, unknown-field-
    rejecting, so a malformed hit fails at the boundary.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class _SearchArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    filter: dict[str, Any] | None = None


class _MoreLikeThisArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    exclude: tuple[str, ...] = ()
    filter: dict[str, Any] | None = None


class _GetContextArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hits: tuple[_HitInput, ...]


class _DiscoveryArgs(BaseModel):
    """``list_sources`` / ``index_status`` take no arguments — any key is rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# tool id -> its argument model. The one place a tool's input schema lives; both ``call`` (parsing)
# and ``list_tools`` (the MCP inputSchema) read it, so the validated surface and the advertised
# surface cannot drift.
_ARG_MODELS: dict[str, type[BaseModel]] = {
    SEARCH: _SearchArgs,
    MORE_LIKE_THIS: _MoreLikeThisArgs,
    GET_CONTEXT: _GetContextArgs,
    LIST_SOURCES: _DiscoveryArgs,
    INDEX_STATUS: _DiscoveryArgs,
}

_DEFAULT_TOP_K = 10


class ToolDescriptor(BaseModel):
    """The rendered advertisement for one tool: dispatch id, agent copy, and JSON input schema.

    ``id`` is the canonical dispatch key; ``name`` / ``description`` / ``params`` are the
    operator-tailored copy the agent reads; ``level`` is the ladder rung; ``input_schema``
    is the JSON Schema (from the tool's Pydantic argument model, enriched with the copy's param
    docs) the MCP face publishes as ``inputSchema``. Frozen — a read-only projection of the server's
    immutable copy + argument models.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    description: str
    level: str
    params: dict[str, str]
    input_schema: dict[str, Any]


class RetrievalServer:
    """A stateless, hardened MCP server wrapping one backend + the orchestrator (ADR-0052).

    Construct it with the customer's engine and the operator's policy; then :meth:`list_tools` gives
    the rendered surface and :meth:`call` runs one tool. Every :meth:`call` is authenticated,
    rate-limited, argument-validated, dispatched, logged, and returned as one
    :class:`~latence_retrieval.mcp.envelope.ResponseEnvelope` — success or typed, leak-free failure.

    Args:
        backend: the customer's first-stage engine (the only place a fetch happens, ADR-0048).
        copy: the operator-trusted, capability-tailored agent-facing copy.
        auth: the bearer-token authenticator (every call must present a valid token).
        rate_limiter: the per-window request budget (a refused call never reaches the backend).
        pipeline: the orchestration recipe ``search`` / ``more_like_this`` run (default:
            fetch + fuse only).
        query_signal_gen: optional ``text -> {vector?, sparse_terms?}`` generator for a store whose
            modes need more than raw text; ``None`` means text-only bundles (a bm25 leg).
        sources: optional operator source descriptors for ``list_sources``; defaults to one derived
            from the backend's declared capabilities.
        logger: optional structured-JSON logger; when given, one non-secret record is emitted per
            call (never the token, never raw backend text).
        default_top_k: the result cap applied when a call omits ``top_k`` (default 10).
    """

    def __init__(
        self,
        *,
        backend: RetrievalBackend,
        copy: ServerCopy,
        auth: BearerAuth,
        rate_limiter: RateLimiter,
        pipeline: OrchestrationPipeline | None = None,
        query_signal_gen: QuerySignalGen | None = None,
        sources: Iterable[SourceDescriptor] | None = None,
        logger: logging.Logger | None = None,
        default_top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        top_k_ok = isinstance(default_top_k, int) and not isinstance(default_top_k, bool)
        if not top_k_ok or default_top_k < 1:
            msg = f"RetrievalServer 'default_top_k' must be a positive int, got {default_top_k!r}."
            raise ValueError(msg)
        self._backend = backend
        self._copy = copy
        self._auth = auth
        self._rate_limiter = rate_limiter
        self._pipeline = pipeline or OrchestrationPipeline()
        self._query_signal_gen = query_signal_gen
        self._sources = tuple(sources) if sources is not None else self._default_sources()
        self._logger = logger
        self._default_top_k = default_top_k

    # --- public interface (the whole tool-call boundary) -----------------------------------------

    @property
    def operating_policy(self) -> str:
        """The server's operating-policy guidance (the MCP server instructions), from the copy."""
        return self._copy.operating_policy

    def list_tools(self) -> list[ToolDescriptor]:
        """The rendered tool surface in escalating-ladder order — the MCP face publishes this."""
        descriptors: list[ToolDescriptor] = []
        for tool_copy in self._copy.ordered_tools():
            schema = _input_schema(tool_copy.id, tool_copy.params)
            descriptors.append(
                ToolDescriptor(
                    id=tool_copy.id,
                    name=tool_copy.name,
                    description=tool_copy.description,
                    level=tool_copy.level,
                    params=dict(tool_copy.params),
                    input_schema=schema,
                )
            )
        return descriptors

    def call(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        token: str | None,
    ) -> ResponseEnvelope:
        """Run one tool call, returning one envelope — auth → rate-limit → validate → dispatch.

        The ``name`` is the agent-facing tool name (resolved to its dispatch id via the copy);
        ``arguments`` is the raw agent input (validated against the tool's own schema — an unknown
        key, including any attempt to smuggle copy fields, is rejected as ``INVALID_ARGUMENTS``, so
        agent input can never mutate the copy); ``token`` is the presented bearer token. Failures
        are typed, leak-free envelopes; nothing is retained after the call returns (ADR-0048).
        """
        started = time.monotonic()
        envelope = self._dispatch(name, arguments, token=token)
        if self._logger is not None:
            duration_ms = round((time.monotonic() - started) * 1000, 3)
            record: dict[str, Any] = {
                "tool": envelope.tool,
                "ok": envelope.ok,
                "duration_ms": duration_ms,
                "result_count": _result_count(envelope),
            }
            if envelope.error is not None:
                record["error_code"] = envelope.error.code.value
            log_call(self._logger, record)
        return envelope

    # --- dispatch (private; the hidden complexity behind ``call``) -------------------------------

    def _dispatch(
        self,
        name: str,
        arguments: Mapping[str, Any] | None,
        *,
        token: str | None,
    ) -> ResponseEnvelope:
        if not self._auth.authenticate(token):
            return ResponseEnvelope.failure(
                name, ErrorCode.UNAUTHORIZED, "A valid bearer token is required."
            )
        if not self._rate_limiter.allow():
            return ResponseEnvelope.failure(
                name, ErrorCode.RATE_LIMITED, "Request rate limit exceeded; retry later."
            )

        tool_id = self._copy.name_to_id().get(name)
        if tool_id is None:
            available = ", ".join(sorted(self._copy.name_to_id())) or "<none>"
            return ResponseEnvelope.failure(
                name, ErrorCode.UNKNOWN_TOOL, f"Unknown tool {name!r}. Available: {available}."
            )

        arg_model = _ARG_MODELS[tool_id]
        try:
            args = arg_model.model_validate(dict(arguments or {}))
        except ValidationError as exc:
            return ResponseEnvelope.failure(
                name, ErrorCode.INVALID_ARGUMENTS, _validation_message(exc)
            )

        try:
            return self._run_tool(tool_id, name, args)
        except OrchestrationError as exc:
            return ResponseEnvelope.failure(name, ErrorCode.UNSUPPORTED, str(exc))
        except (BackendCapabilityError, BackendDimensionError) as exc:
            return ResponseEnvelope.failure(name, ErrorCode.UNSUPPORTED, str(exc))
        except RetrievalError as exc:
            # BackendSearchError and any other typed store/model runtime fault — already leak-free
            # by
            # contract (the adapter stripped raw store text), surfaced under one agent-facing code.
            return ResponseEnvelope.failure(name, ErrorCode.BACKEND_ERROR, str(exc))

    def _run_tool(self, tool_id: str, name: str, args: BaseModel) -> ResponseEnvelope:
        if tool_id == SEARCH:
            assert isinstance(args, _SearchArgs)
            return self._search(name, args)
        if tool_id == MORE_LIKE_THIS:
            assert isinstance(args, _MoreLikeThisArgs)
            return self._more_like_this(name, args)
        if tool_id == GET_CONTEXT:
            assert isinstance(args, _GetContextArgs)
            return self._get_context(name, args)
        if tool_id == LIST_SOURCES:
            return self._list_sources(name)
        if tool_id == INDEX_STATUS:
            return self._index_status(name)
        msg = f"no handler for tool id {tool_id!r}."  # pragma: no cover
        raise KeyError(msg)  # pragma: no cover

    def _search(self, name: str, args: _SearchArgs) -> ResponseEnvelope:
        bundle = self._build_bundle(args.query, args.filter)
        result = self._retrieve(bundle)
        top_k = args.top_k or self._default_top_k
        candidates = result.candidates[:top_k]
        return ResponseEnvelope.success(
            name,
            {"results": [_grounded(c) for c in candidates]},
            meta={
                "fetched_modes": [m.value for m in result.fetched_modes],
                "count": len(candidates),
            },
        )

    def _more_like_this(self, name: str, args: _MoreLikeThisArgs) -> ResponseEnvelope:
        bundle = self._build_bundle(args.text, args.filter)
        result = self._retrieve(bundle)
        exclude = set(args.exclude)
        distinct = [c for c in result.candidates if c.id not in exclude]
        top_k = args.top_k or self._default_top_k
        candidates = distinct[:top_k]
        return ResponseEnvelope.success(
            name,
            {"results": [_grounded(c) for c in candidates]},
            meta={
                "fetched_modes": [m.value for m in result.fetched_modes],
                "count": len(candidates),
                "excluded": len(exclude),
            },
        )

    def _get_context(self, name: str, args: _GetContextArgs) -> ResponseEnvelope:
        # Stateless projection: no fetch. The agent supplied the hits; we return each as a grounded
        # chunk with structured provenance (ADR-0048 — the server holds no corpus to look ids up
        # in).
        chunks = [
            {"id": h.id, "text": h.text, "provenance": dict(h.metadata)} for h in args.hits
        ]
        return ResponseEnvelope.success(
            name, {"chunks": chunks}, meta={"count": len(chunks)}
        )

    def _list_sources(self, name: str) -> ResponseEnvelope:
        return ResponseEnvelope.success(
            name,
            {"sources": [s.model_dump() for s in self._sources]},
            meta={"count": len(self._sources)},
        )

    def _index_status(self, name: str) -> ResponseEnvelope:
        capabilities = sorted(c.value for c in self._backend.capabilities)
        return ResponseEnvelope.success(
            name,
            {
                "capabilities": capabilities,
                "sources": len(self._sources),
                "stateless": True,
                "note": "First-stage search runs on the operator's own engine; this server holds "
                "no index.",
            },
        )

    def _retrieve(self, bundle: QueryBundle) -> RetrievalResult:
        """Run the orchestrator, containing a candidate-processor fault leak-free (ADR-0052).

        The orchestrator's only fetch runs on the caller's engine, and its typed store/planning
        faults (:class:`OrchestrationError`, :class:`BackendCapabilityError`,
        :class:`BackendDimensionError`, :class:`~latence_retrieval.errors.BackendSearchError`) are
        already leak-free and carry their own dispatch code — so a :class:`RetrievalError` is
        re-raised **unchanged** to keep its specific envelope code (``UNSUPPORTED`` /
        ``BACKEND_ERROR``).

        This is the other place the server runs **caller-supplied code over caller-supplied data**:
        the operator-configured candidate-processor chain (``expand`` / ``multihop_expand`` …) runs
        over the fetched candidates' metadata, and the graph-augmenting processors raise a plain
        :class:`TypeError` / :class:`ValueError` (their loud anti-false-green contract) on a
        malformed KG column — an exception whose message embeds the raw candidate metadata. Neither
        is a :class:`RetrievalError`, so it is contained here into a typed, leak-free
        :class:`CandidateProcessorError` (message = the exception *type* only, never its raw text),
        surfacing as one ``BACKEND_ERROR`` envelope through the dispatch guard rather than escaping
        uncaught across the tool-call boundary and leaking the store metadata (ADR-0052).
        """
        try:
            return retrieve(bundle, self._backend, self._pipeline)
        except RetrievalError:
            raise
        except Exception as exc:
            # A caller-configured candidate processor rejected malformed candidate metadata with a
            # plain TypeError/ValueError; its raw message carries the offending value + candidate id
            # — surface the type only, never str(exc).
            msg = f"candidate processing failed: {type(exc).__name__}."
            raise CandidateProcessorError(msg) from exc

    def _build_bundle(self, text: str, filter_: dict[str, Any] | None) -> QueryBundle:
        """Compose the query bundle: raw text plus the injected query-signal-gen's dense/sparse.

        The orchestrator drops any leg the backend does not declare or the bundle has no signal for
        (ADR-0052): a text-only bundle drives a bm25 leg, and a dense/sparse store gets its signal
        from the injected generator.

        This is where the server crosses into **caller-supplied code** run on the *raw agent query*:
        the ``query_signal_gen`` is a plain callable (an encoder), and :class:`QueryBundle`
        (``extra="forbid"``) rejects a signal key a mis-wired generator emits. A real encoder raises
        arbitrary native faults on some inputs, and pydantic raises ``ValidationError`` on a bad key
        — neither a :class:`RetrievalError`. Both are contained here into a typed, leak-free
        :class:`QuerySignalError` (message = the exception *type* only, never its raw text), so the
        fault surfaces as one ``BACKEND_ERROR`` envelope through the dispatch guard rather than
        escaping uncaught across the tool-call boundary and leaking internals (ADR-0052).
        """
        signals: dict[str, Any] = {}
        if self._query_signal_gen is not None:
            try:
                signals = dict(self._query_signal_gen(text))
            except Exception as exc:
                # A caller-supplied query encoder faulted on agent text; its raw message may carry a
                # model path/payload/driver stack — surface the type only, never str(exc).
                msg = f"query-signal generation failed: {type(exc).__name__}."
                raise QuerySignalError(msg) from exc
        try:
            return QueryBundle(text=text, filter=filter_, **signals)
        except Exception as exc:
            # A mis-wired generator produced a signal QueryBundle rejects (unknown key, wrong type,
            # or a colliding kwarg) — contain the pydantic/construction fault leak-free.
            msg = f"query-signal bundle rejected: {type(exc).__name__}."
            raise QuerySignalError(msg) from exc

    def _default_sources(self) -> tuple[SourceDescriptor, ...]:
        caps = tuple(sorted(c.value for c in self._backend.capabilities))
        return (
            SourceDescriptor(
                name="default",
                description="The operator's configured retrieval engine.",
                capabilities=caps,
            ),
        )


def _grounded(candidate: Candidate) -> dict[str, Any]:
    """Project a candidate into the agent-facing grounded shape: id/score/text/provenance."""
    return {
        "id": candidate.id,
        "score": candidate.score,
        "text": candidate.text,
        "provenance": dict(candidate.metadata),
    }


def _input_schema(tool_id: str, param_docs: Mapping[str, str]) -> dict[str, Any]:
    """The tool's JSON input schema, enriched with the copy's per-parameter docs for the agent."""
    schema: dict[str, Any] = _ARG_MODELS[tool_id].model_json_schema()
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for param, doc in param_docs.items():
            prop = properties.get(param)
            if isinstance(prop, dict):
                prop["description"] = doc
    return schema


def _validation_message(exc: ValidationError) -> str:
    """A short, agent-actionable summary of an argument-validation failure (no internals leaked)."""
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
        problems.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "Invalid arguments — " + "; ".join(problems[:5]) + "."


def _result_count(envelope: ResponseEnvelope) -> int | None:
    """The result count carried in the envelope meta, for the log record (None on failure)."""
    count = envelope.meta.get("count")
    return count if isinstance(count, int) else None


__all__ = [
    "RetrievalServer",
    "SourceDescriptor",
    "ToolDescriptor",
]
