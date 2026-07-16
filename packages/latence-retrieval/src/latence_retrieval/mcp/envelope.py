"""The uniform response contract every MCP tool call returns: :class:`ResponseEnvelope`.

One shape for *every* outcome — a success, a refused auth, a rate-limit, a bad argument, a backend
fault — so an agent (and a test) reads one contract, never a per-tool ad-hoc dict or a raw
exception. This is the ``memory_engine`` envelope/``Result``/``ToolError`` discipline ported
(ADR-0052): a call yields exactly one :class:`ResponseEnvelope`, and a failure is a **typed**
:class:`ToolError` whose ``message`` carries *no* raw store/model/driver text (no leak of a URL,
api-key, token, payload, or third-party stack) — the same no-leak rule the backend's
:class:`~latence_retrieval.errors.BackendSearchError` already enforces one layer down.

The models are frozen Pydantic v2 (the framework's serialization spine, ADR-0006), so an envelope
round-trips losslessly through :meth:`~pydantic.BaseModel.model_dump` / JSON — which is exactly what
the MCP-protocol face serialises onto the wire.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorCode(StrEnum):
    """The closed set of failure codes a tool call can carry — stable, agent-facing, leak-free.

    Each names *what* went wrong at the tool-call boundary without exposing *how*: an agent branches
    on the code, a test asserts on it, and neither depends on a backend's native exception class.

    * ``UNAUTHORIZED`` — no valid bearer token was presented (the tool never ran).
    * ``RATE_LIMITED`` — the per-window request budget was exhausted (the tool never ran).
    * ``UNKNOWN_TOOL`` — the requested tool name is not on this server's rendered surface.
    * ``INVALID_ARGUMENTS`` — the arguments failed the tool's own input schema (unknown key, wrong
      type, missing required field) — rejected at the boundary, never partially applied.
    * ``UNSUPPORTED`` — the request is well-formed but the wired backend cannot serve it (a filter a
      store did not declare, a mode with no query signal): the orchestrator's planning refusal
      (:class:`~latence_retrieval.errors.OrchestrationError`), surfaced without leaking internals.
    * ``BACKEND_ERROR`` — the customer's engine faulted *during* a search it accepted
      (:class:`~latence_retrieval.errors.BackendSearchError` and siblings), mapped to a leak-free
      message.
    """

    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNSUPPORTED = "unsupported"
    BACKEND_ERROR = "backend_error"


class ToolError(BaseModel):
    """A typed, leak-free failure — the only thing a non-``ok`` envelope carries in ``error``.

    ``code`` is a stable :class:`ErrorCode` an agent branches on; ``message`` is a short,
    operator-safe explanation that **must not** embed raw store/model/driver text (the no-leak rule,
    mirroring :class:`~latence_retrieval.errors.BackendSearchError`). It is frozen and round-trips
    losslessly like every contract in this library.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ErrorCode
    message: str


class ResponseEnvelope(BaseModel):
    """The single shape *every* tool call returns — success or failure, uniformly (ADR-0052).

    Exactly one of ``result`` / ``error`` is populated, keyed by ``ok``: an ``ok=True`` envelope
    carries the tool's ``result`` payload (a plain JSON-able mapping) and ``error=None``; an
    ``ok=False`` envelope carries a typed :class:`ToolError` and ``result=None``. ``tool`` echoes
    the called tool name and ``meta`` carries per-call, non-secret observability (e.g. the fetched
    modes, a result count) — never a token or raw backend text. Construct via :meth:`success` /
    :meth:`failure` so the ``ok``/``result``/``error`` invariant cannot be set inconsistently.

    It is frozen and round-trips losslessly through :meth:`~pydantic.BaseModel.model_dump` / JSON —
    the exact serialisation the MCP-protocol face puts on the wire.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    tool: str
    result: dict[str, Any] | None = None
    error: ToolError | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def success(
        cls,
        tool: str,
        result: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> ResponseEnvelope:
        """A well-formed success envelope (``ok=True``, ``error=None``)."""
        return cls(ok=True, tool=tool, result=result, error=None, meta=meta or {})

    @classmethod
    def failure(
        cls,
        tool: str,
        code: ErrorCode,
        message: str,
        meta: dict[str, Any] | None = None,
    ) -> ResponseEnvelope:
        """A well-formed failure envelope (``ok=False``, ``result=None``) with a typed error."""
        return cls(
            ok=False,
            tool=tool,
            result=None,
            error=ToolError(code=code, message=message),
            meta=meta or {},
        )


__all__ = ["ErrorCode", "ResponseEnvelope", "ToolError"]
