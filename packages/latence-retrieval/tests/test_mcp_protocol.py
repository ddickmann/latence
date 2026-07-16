"""The thin MCP-protocol face — pure translation + a faithful fake ``mcp`` module (ADR-0052, #174).

Two boundaries, both offline (no real ``mcp`` SDK needed — ADR-0016/0049):

* the **pure translation helpers** — :func:`tool_specs` (rendered surface → MCP ``Tool`` payloads) and
  :func:`dispatch_text` (a tool call → the JSON envelope string a text result carries) — asserted
  directly, so the translation is covered without any SDK;
* the **binding** — :func:`build_mcp_server` — driven through a faithful fake ``mcp`` module (the same
  honest fake-module boundary the qdrant/neo4j adapters use): the fake ``Tool`` / ``TextContent``
  reject unknown kwargs exactly as the real Pydantic models do (anti-false-green — a spec key the
  real API lacks would fail here too), and the registered async handlers are awaited to assert they
  return faithful ``Tool`` / ``TextContent`` objects carrying the server's envelopes.

Live-SDK validation against a running MCP client is out of band (as the module docstring states).
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest
from latence_retrieval import Candidate, Capability
from latence_retrieval.mcp import (
    BearerAuth,
    RateLimiter,
    RetrievalServer,
    ServerCopy,
)
from latence_retrieval.mcp.protocol import (
    TOKEN_ENV_VAR,
    build_mcp_server,
    dispatch_text,
    tool_specs,
)
from latence_retrieval.testing import FakeBackend

CORPUS = [
    Candidate(id="a", score=0.9, text="alpha beta", metadata={"source": "s1"}),
    Candidate(id="b", score=0.7, text="beta delta", metadata={"source": "s2"}),
]


def _server(token: str = "tok") -> RetrievalServer:
    backend = FakeBackend(CORPUS, frozenset({Capability.BM25}))
    return RetrievalServer(
        backend=backend,
        copy=ServerCopy.build(backend.capabilities),
        auth=BearerAuth(token),
        rate_limiter=RateLimiter(capacity=100, window=60.0),
    )


# --- pure translation helpers --------------------------------------------------------------------


def test_tool_specs_render_name_description_and_schema() -> None:
    specs = tool_specs(_server())
    names = [s["name"] for s in specs]
    assert names == ["search", "more_like_this", "get_context", "list_sources", "index_status"]
    for spec in specs:
        assert set(spec) == {"name", "description", "inputSchema"}
        assert spec["inputSchema"]["type"] == "object"


def test_dispatch_text_returns_a_json_envelope() -> None:
    text = dispatch_text(_server(), "search", {"query": "beta"}, token="tok")
    payload = json.loads(text)
    assert payload["ok"] is True
    assert payload["tool"] == "search"
    assert [r["id"] for r in payload["result"]["results"]] == ["a", "b"]


def test_dispatch_text_serialises_a_failure_without_raising() -> None:
    text = dispatch_text(_server(), "search", {"query": "beta"}, token="wrong")
    payload = json.loads(text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unauthorized"


# --- the faithful fake ``mcp`` module + the binding ----------------------------------------------


class _FakeTool:
    """Mirrors ``mcp.types.Tool`` — an explicit signature that rejects unknown kwargs (no swallow)."""

    def __init__(self, *, name: str, description: str, inputSchema: dict[str, Any]) -> None:  # noqa: N803
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


class _FakeTextContent:
    """Mirrors ``mcp.types.TextContent`` — requires ``type`` + ``text``, rejects extras."""

    def __init__(self, *, type: str, text: str) -> None:  # noqa: A002
        if type != "text":
            msg = f"TextContent type must be 'text', got {type!r}."
            raise ValueError(msg)
        self.type = type
        self.text = text


class _FakeServer:
    """Mirrors ``mcp.server.lowlevel.Server``'s decorator-registration contract."""

    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.list_tools_handler: Any = None
        self.call_tool_handler: Any = None

    def list_tools(self) -> Any:
        def deco(fn: Any) -> Any:
            self.list_tools_handler = fn
            return fn

        return deco

    def call_tool(self) -> Any:
        def deco(fn: Any) -> Any:
            self.call_tool_handler = fn
            return fn

        return deco


@pytest.fixture
def fake_mcp() -> Any:
    """Install a faithful fake ``mcp`` package into ``sys.modules`` for the duration of a test."""
    saved = {k: sys.modules[k] for k in list(sys.modules) if k == "mcp" or k.startswith("mcp.")}
    for key in list(saved):
        del sys.modules[key]

    mcp_mod = types.ModuleType("mcp")
    server_mod = types.ModuleType("mcp.server")
    lowlevel_mod = types.ModuleType("mcp.server.lowlevel")
    types_mod = types.ModuleType("mcp.types")
    lowlevel_mod.Server = _FakeServer  # type: ignore[attr-defined]
    types_mod.Tool = _FakeTool  # type: ignore[attr-defined]
    types_mod.TextContent = _FakeTextContent  # type: ignore[attr-defined]
    server_mod.lowlevel = lowlevel_mod  # type: ignore[attr-defined]
    mcp_mod.server = server_mod  # type: ignore[attr-defined]
    sys.modules["mcp"] = mcp_mod
    sys.modules["mcp.server"] = server_mod
    sys.modules["mcp.server.lowlevel"] = lowlevel_mod
    sys.modules["mcp.types"] = types_mod
    try:
        yield
    finally:
        for key in ["mcp", "mcp.server", "mcp.server.lowlevel", "mcp.types"]:
            sys.modules.pop(key, None)
        sys.modules.update(saved)


def test_build_mcp_server_registers_tools_and_policy(fake_mcp: Any) -> None:
    server = _server()
    mcp_server = build_mcp_server(server)
    assert mcp_server.instructions == server.operating_policy

    tools = asyncio.run(mcp_server.list_tools_handler())
    assert [t.name for t in tools] == [
        "search",
        "more_like_this",
        "get_context",
        "list_sources",
        "index_status",
    ]
    assert all(isinstance(t, _FakeTool) for t in tools)


def test_registered_call_tool_returns_text_content_with_the_envelope(fake_mcp: Any) -> None:
    server = _server(token="tok")
    mcp_server = build_mcp_server(server, token_provider=lambda: "tok")
    result = asyncio.run(mcp_server.call_tool_handler("search", {"query": "beta"}))
    assert len(result) == 1
    assert isinstance(result[0], _FakeTextContent)
    payload = json.loads(result[0].text)
    assert payload["ok"] is True
    assert [r["id"] for r in payload["result"]["results"]] == ["a", "b"]


def test_default_token_provider_reads_the_env_var(
    fake_mcp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(token="env-tok")
    monkeypatch.setenv(TOKEN_ENV_VAR, "env-tok")
    mcp_server = build_mcp_server(server)  # no explicit token_provider → reads the env var
    result = asyncio.run(mcp_server.call_tool_handler("search", {"query": "beta"}))
    assert json.loads(result[0].text)["ok"] is True


def test_call_tool_without_the_env_token_is_unauthorized(
    fake_mcp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(token="env-tok")
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    mcp_server = build_mcp_server(server)
    result = asyncio.run(mcp_server.call_tool_handler("search", {"query": "beta"}))
    payload = json.loads(result[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "unauthorized"
