"""The optional, hardened, stateless MCP server for query-time retrieval (ADR-0052, #174).

A stack-agnostic MCP face over the query-time tooling: it wraps the *customer's* engine (the
two-adapter :class:`~latence_retrieval.backend.RetrievalBackend`) and the Level-2 orchestrator
(:func:`~latence_retrieval.orchestrator.retrieve`) and exposes retrieval as MCP tools for agentic
search, with **all agent-facing copy loaded from operator config at runtime** and the
``memory_engine`` security/observability core ported (ADR-0052). It holds **no index and no
cross-call state** (ADR-0048) — the only fetch is the backend's own ``search``.

The whole tool-call boundary is one deep module, :class:`RetrievalServer`
(:meth:`~latence_retrieval.mcp.server.RetrievalServer.list_tools` +
:meth:`~latence_retrieval.mcp.server.RetrievalServer.call`); everything else here is the machinery
it composes:

* :class:`~latence_retrieval.mcp.copy.ServerCopy` / :class:`~latence_retrieval.mcp.copy.ToolCopy`
  — the config-driven, capability-tailored, schema-validated, **not-agent-mutable** agent-facing
  copy (tool names/descriptions/param-docs/operating-policy/hierarchy).
* :class:`~latence_retrieval.mcp.envelope.ResponseEnvelope` /
  :class:`~latence_retrieval.mcp.envelope.ToolError` /
  :class:`~latence_retrieval.mcp.envelope.ErrorCode`
  — the uniform, leak-free response contract every call returns.
* :class:`~latence_retrieval.mcp.security.BearerAuth` /
  :class:`~latence_retrieval.mcp.security.RateLimiter` /
  :class:`~latence_retrieval.mcp.security.LoopbackTransport` /
  :func:`~latence_retrieval.mcp.security.build_rotating_json_logger` — the ported hardening
  (bearer-0600 + constant-time auth, per-window rate limit, loopback/UDS binding, structured
  rotating JSON logs), local/loopback v1.
* :func:`~latence_retrieval.mcp.protocol.build_mcp_server` /
  :func:`~latence_retrieval.mcp.protocol.run_stdio`
  — the thin ``mcp``-SDK face behind the ``[mcp]`` extra (imported lazily; the pure core needs no
  SDK).
"""

from __future__ import annotations

from latence_retrieval.mcp.copy import ServerCopy, ToolCopy
from latence_retrieval.mcp.envelope import ErrorCode, ResponseEnvelope, ToolError
from latence_retrieval.mcp.protocol import (
    build_mcp_server,
    dispatch_text,
    run_stdio,
    tool_specs,
)
from latence_retrieval.mcp.security import (
    BearerAuth,
    LoopbackTransport,
    RateLimiter,
    build_rotating_json_logger,
    log_call,
)
from latence_retrieval.mcp.server import RetrievalServer, SourceDescriptor, ToolDescriptor

__all__ = [
    "BearerAuth",
    "ErrorCode",
    "LoopbackTransport",
    "RateLimiter",
    "ResponseEnvelope",
    "RetrievalServer",
    "ServerCopy",
    "SourceDescriptor",
    "ToolCopy",
    "ToolDescriptor",
    "ToolError",
    "build_mcp_server",
    "build_rotating_json_logger",
    "dispatch_text",
    "log_call",
    "run_stdio",
    "tool_specs",
]
