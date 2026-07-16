"""The thin MCP-protocol face over :class:`~latence_retrieval.mcp.server.RetrievalServer`
(ADR-0052).

Everything hard already lives in the server core; this module is only the **translation** onto the
MCP low-level SDK, kept deliberately thin so the tested contract stays the tool-call boundary, not
the wire (ADR-0052, spec §"what a good test is here"). It has two pure, SDK-free helpers —
:func:`tool_specs` (rendered surface → MCP ``Tool`` payloads) and :func:`dispatch_text` (a tool call
→ the JSON envelope string an MCP text result carries) — and one lazily-imported binding,
:func:`build_mcp_server`, that registers those onto an ``mcp`` ``Server``.

The ``mcp`` SDK is a heavy optional dependency behind the ``[mcp]`` extra (ADR-0016/0049): it is
imported **only** inside :func:`build_mcp_server` / :func:`run_stdio`, so the pure core and the
offline tests never need it. The two pure helpers carry the whole translation, so a fake-``mcp``
test (the same faithful-fake-module boundary the qdrant/neo4j adapters use) exercises the binding
without the real SDK; live-SDK validation against a running MCP client is out of band.

**Local/loopback v1 (ADR-0052).** The v1 transport is stdio — the co-located agent/tool deployment
shape — with the bearer token supplied out-of-band (an operator env var by default). Remote
transport (OAuth 2.1 / mTLS) is deferred; because the whole face is these two helpers over the
server's ``call``, a later remote adapter reuses the core unchanged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from latence_retrieval.mcp.envelope import ResponseEnvelope
    from latence_retrieval.mcp.server import RetrievalServer

# The env var the default token provider reads for the stdio face's bearer token. The token is
# supplied out-of-band (never over the protocol), matching the local/loopback v1 trust boundary.
TOKEN_ENV_VAR = "LATENCE_RETRIEVAL_MCP_TOKEN"

TokenProvider = Callable[[], "str | None"]


def tool_specs(server: RetrievalServer) -> list[dict[str, Any]]:
    """The rendered surface as MCP ``Tool`` payloads: ``{name, description, inputSchema}`` each.

    Pure translation of :meth:`~latence_retrieval.mcp.server.RetrievalServer.list_tools` — the exact
    dicts :func:`build_mcp_server` feeds into ``mcp.types.Tool``. Kept SDK-free so the translation
    is tested directly.
    """
    return [
        {"name": d.name, "description": d.description, "inputSchema": d.input_schema}
        for d in server.list_tools()
    ]


def dispatch_text(
    server: RetrievalServer,
    name: str,
    arguments: dict[str, Any] | None,
    *,
    token: str | None,
) -> str:
    """Run a tool call and return the JSON envelope string an MCP text result carries.

    Pure translation: it calls :meth:`~latence_retrieval.mcp.server.RetrievalServer.call` and
    serialises the resulting :class:`~latence_retrieval.mcp.envelope.ResponseEnvelope` losslessly to
    JSON. Every outcome — success or a typed, leak-free failure — comes back as one text payload, so
    the MCP handler never raises across the protocol.
    """
    envelope: ResponseEnvelope = server.call(name, arguments, token=token)
    return json.dumps(envelope.model_dump(), sort_keys=True, default=str)


def _default_token_provider() -> str | None:
    """Read the bearer token from the operator env var (the out-of-band v1 supply)."""
    return os.environ.get(TOKEN_ENV_VAR)


def build_mcp_server(
    server: RetrievalServer,
    *,
    name: str = "latence-retrieval",
    token_provider: TokenProvider | None = None,
) -> Any:
    """Register the server's tools onto a lazily-imported ``mcp`` low-level ``Server`` (``[mcp]``).

    Wires two handlers — ``list_tools`` (from :func:`tool_specs`) and ``call_tool`` (from
    :func:`dispatch_text`, wrapping the JSON envelope in an ``mcp`` text result) — plus the server's
    operating policy as the MCP instructions. The bearer token for each call is drawn from
    ``token_provider`` (default: the :data:`TOKEN_ENV_VAR` env var), never from the protocol. The
    ``mcp`` import is deferred to here, so importing this module never requires the SDK.
    """
    provider = token_provider or _default_token_provider
    try:
        from mcp.server.lowlevel import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = (
            "The 'mcp' package is required to serve the MCP protocol face. Install it with: "
            "pip install 'latence-retrieval[mcp]'"
        )
        raise ImportError(msg) from exc

    mcp_server = Server(name, instructions=server.operating_policy)

    @mcp_server.list_tools()  # type: ignore[untyped-decorator]  # mcp Server is Any (lazy import)
    async def _list_tools() -> list[Any]:
        return [Tool(**spec) for spec in tool_specs(server)]

    @mcp_server.call_tool()  # type: ignore[untyped-decorator]  # mcp Server is Any (lazy import)
    async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> list[Any]:
        text = dispatch_text(server, tool_name, arguments, token=provider())
        return [TextContent(type="text", text=text)]

    return mcp_server


async def run_stdio(
    server: RetrievalServer,
    *,
    name: str = "latence-retrieval",
    token_provider: TokenProvider | None = None,
) -> None:  # pragma: no cover - requires the live mcp SDK + a stdio peer
    """Serve the MCP face over stdio (the local/loopback v1 transport, ADR-0052).

    Builds the ``mcp`` server via :func:`build_mcp_server` and runs it over the stdio transport —
    the co-located agent/tool deployment shape. Requires the ``[mcp]`` extra and a connected client,
    so it is validated out of band, not in the offline suite.
    """
    from mcp.server.stdio import stdio_server

    mcp_server = build_mcp_server(server, name=name, token_provider=token_provider)
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream, write_stream, mcp_server.create_initialization_options()
        )


__all__ = [
    "TOKEN_ENV_VAR",
    "build_mcp_server",
    "dispatch_text",
    "run_stdio",
    "tool_specs",
]
