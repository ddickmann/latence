"""Operator-config-driven, capability-tailored agent-facing copy: :class:`ServerCopy` (ADR-0052).

MCP tool descriptions are how an agent decides what to call, so the copy *is* the product — and the
right copy depends on the customer's stack (a hybrid Qdrant reads differently from a dense-only
index). This module makes **all** agent-facing copy — tool names, descriptions, per-parameter docs,
the server operating-policy, and the escalating-ladder hierarchy — operator config resolved at
runtime, over **strong defaults auto-tailored to the backend's declared capabilities** (ADR-0052):

* :meth:`ServerCopy.build` starts from the ``memory_engine`` escalating-ladder defaults, tailors
  them to the backend's :class:`~latence_retrieval.contracts.Capability` set (a sparse+dense store
  gets "hybrid" copy; a dense-only store gets "semantic"; a filter-capable store's copy says so),
  then **merges the operator's overrides on top** — so the server is excellent before any
  customisation and speaks the operator's stack + domain after it.
* The override mapping is **schema-validated** (:meth:`ServerCopy.build` /
  :meth:`ServerCopy.from_file`): an unknown tool id, an unknown per-tool field, or an unknown
  parameter name is a loud ``ValueError`` at load time — the copy config is *operator-trusted*, so
  a typo fails closed rather than silently shipping stale guidance.
* The built :class:`ServerCopy` is **frozen and loaded once at construction**; nothing in the
  tool-call path can write to it. The agent only ever *reads* the rendered
  :class:`~latence_retrieval.mcp.copy.ToolCopy`; there is no argument that reaches the copy — the
  instruction-source-boundary principle applied to the server's own configuration (ADR-0052), so the
  tool descriptions can never become a prompt-injection surface.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from latence_retrieval.contracts import Capability

# --- the canonical, stack-agnostic tool surface (the escalating ladder, ADR-0052) ----------------
# Each entry is (canonical id, ladder level, ordered parameter names). The ids and params are the
# server's contract; the *copy* (name/description/param docs) is what the operator tailors. Ladder
# levels order the hierarchy an agent climbs: L1 entry → L2 expand → L3 full-context, plus
# discovery.
_LADDER_RANK: dict[str, int] = {"L1": 0, "L2": 1, "L3": 2, "discovery": 3}

SEARCH = "search"
MORE_LIKE_THIS = "more_like_this"
GET_CONTEXT = "get_context"
LIST_SOURCES = "list_sources"
INDEX_STATUS = "index_status"

# tool id -> (ladder level, parameter names). The one place the surface shape is declared; the
# server's handlers and the copy defaults both read it, so they cannot drift.
TOOL_SURFACE: dict[str, tuple[str, tuple[str, ...]]] = {
    SEARCH: ("L1", ("query", "top_k", "filter")),
    MORE_LIKE_THIS: ("L2", ("text", "top_k", "exclude", "filter")),
    GET_CONTEXT: ("L3", ("hits",)),
    LIST_SOURCES: ("discovery", ()),
    INDEX_STATUS: ("discovery", ()),
}

# Declaration order of the tool surface — the tiebreak within a ladder level, so two discovery tools
# render in their declared (list_sources → index_status) order rather than by id.
_TOOL_ORDER: dict[str, int] = {tool_id: i for i, tool_id in enumerate(TOOL_SURFACE)}

_PARAM_DOC_DEFAULTS: dict[str, dict[str, str]] = {
    SEARCH: {
        "query": "The natural-language question to search for.",
        "top_k": "Maximum number of results to return (optional; the server caps the default).",
        "filter": "Optional exact-match metadata predicate, e.g. {\"source\": \"handbook\"}.",
    },
    MORE_LIKE_THIS: {
        "text": "The content of a result you already have; finds more, distinct results like it.",
        "top_k": "Maximum number of results to return (optional).",
        "exclude": "Ids you already have, so they are not returned again (optional).",
        "filter": "Optional exact-match metadata predicate to scope the expansion.",
    },
    GET_CONTEXT: {
        "hits": (
            "The hits (id/text/metadata) you retrieved earlier; returns each as a grounded chunk "
            "with structured provenance (source, page, offsets, confidence) for citation."
        ),
    },
    LIST_SOURCES: {},
    INDEX_STATUS: {},
}

_DEFAULT_OPERATING_POLICY = (
    "Retrieval over the operator's own search engine. Start with 'search'; escalate to "
    "'more_like_this' for breadth on a promising hit and 'get_context' to assemble grounded, "
    "citable context. Use 'list_sources' and 'index_status' to orient. Results are read-only; this "
    "server holds no index and stores nothing across calls."
)

_ALLOWED_TOOL_FIELDS = frozenset({"name", "description", "params"})


class ToolCopy(BaseModel):
    """The rendered, agent-facing copy for one tool — name, description, ladder level, param docs.

    ``id`` is the stable canonical tool id the server dispatches on (never operator-renamed);
    ``name`` is the agent-facing name (defaults to ``id``, operator-overridable); ``description``
    and ``params`` (parameter name → doc) are what the agent reads to decide how to call it;
    ``level`` is the escalating-ladder rung (``L1``/``L2``/``L3``/``discovery``) that orders the
    hierarchy. Frozen: once built it is read-only, so no tool call can mutate the copy an agent
    reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    name: str
    description: str
    level: str
    params: dict[str, str] = Field(default_factory=dict)


class ServerCopy(BaseModel):
    """The full agent-facing copy set: the operating policy + one :class:`ToolCopy` per tool.

    Built by :meth:`build` (capability-tailored defaults ⊕ validated operator overrides) or
    :meth:`from_file` (the same, sourced from an operator-controlled, sandboxed file). Frozen and
    constructed once, so it is the server's immutable, operator-trusted configuration — the agent
    reads :meth:`tool` / :meth:`ordered_tools` and can never write back (ADR-0052).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operating_policy: str
    tools: dict[str, ToolCopy]

    def tool(self, tool_id: str) -> ToolCopy:
        """The rendered copy for ``tool_id`` (raises ``KeyError`` for an id off the surface)."""
        return self.tools[tool_id]

    def ordered_tools(self) -> list[ToolCopy]:
        """Every tool's copy in ladder order (L1 → L2 → L3 → discovery, then declared order)."""
        return sorted(
            self.tools.values(),
            key=lambda t: (_LADDER_RANK.get(t.level, 99), _TOOL_ORDER.get(t.id, 99)),
        )

    def name_to_id(self) -> dict[str, str]:
        """Map each agent-facing tool ``name`` back to its canonical dispatch ``id``."""
        return {t.name: t.id for t in self.tools.values()}

    @classmethod
    def build(
        cls,
        capabilities: Iterable[Capability],
        overrides: Mapping[str, Any] | None = None,
    ) -> ServerCopy:
        """Capability-tailored defaults merged under validated operator ``overrides``.

        ``capabilities`` are the backend's declared modes; they tailor the default ``search`` copy
        (hybrid vs semantic vs lexical) and the filter mention. ``overrides`` is the operator config
        mapping — ``{"operating_policy"?: str, "tools"?: {tool_id: {"name"?, "description"?,
        "params"?: {param: doc}}}}`` — schema-validated so an unknown tool id, tool field, or
        parameter name is a loud ``ValueError`` (the config is operator-trusted; a typo fails
        closed).
        Overrides win field-by-field; anything unspecified keeps the tailored default.
        """
        caps = frozenset(capabilities)
        raw_overrides = dict(overrides or {})
        _reject_unknown_keys(raw_overrides, {"operating_policy", "tools"}, "copy override")

        operating_policy = raw_overrides.get("operating_policy", _DEFAULT_OPERATING_POLICY)
        if not isinstance(operating_policy, str) or not operating_policy:
            msg = "copy override 'operating_policy' must be a non-empty string."
            raise ValueError(msg)

        tool_overrides = raw_overrides.get("tools", {})
        if not isinstance(tool_overrides, Mapping):
            msg = "copy override 'tools' must be a mapping of tool id -> copy."
            raise TypeError(msg)
        _reject_unknown_keys(tool_overrides, set(TOOL_SURFACE), "copy override 'tools'")

        tools: dict[str, ToolCopy] = {}
        for tool_id, (level, params) in TOOL_SURFACE.items():
            tools[tool_id] = _build_tool_copy(
                tool_id, level, params, caps, tool_overrides.get(tool_id)
            )
        return cls(operating_policy=operating_policy, tools=tools)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        capabilities: Iterable[Capability],
        *,
        base_dir: str | Path | None = None,
    ) -> ServerCopy:
        """Load operator override copy from a sandboxed JSON file, then :meth:`build` over defaults.

        The file must be a regular JSON file whose top level is a mapping. ``base_dir``, when given,
        sandboxes the load: the file's *resolved* path must lie inside ``base_dir`` (a
        path-traversal
        or symlink-escape attempt is a loud ``ValueError``), so an operator can confine the copy
        config to a trusted directory. The parsed mapping is handed to :meth:`build`, so the same
        schema validation and capability tailoring apply.
        """
        resolved = Path(path).resolve()
        if base_dir is not None:
            base = Path(base_dir).resolve()
            if not resolved.is_relative_to(base):
                msg = f"copy config {str(path)!r} resolves outside the sandbox base {str(base)!r}."
                raise ValueError(msg)
        if not resolved.is_file():
            msg = f"copy config path {str(path)!r} is not a regular file."
            raise ValueError(msg)
        try:
            parsed = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"copy config {str(path)!r} is not valid JSON: {exc.msg}."
            raise ValueError(msg) from None
        if not isinstance(parsed, dict):
            msg = f"copy config {str(path)!r} must be a JSON object at the top level."
            raise ValueError(msg)
        return cls.build(capabilities, parsed)


def _build_tool_copy(
    tool_id: str,
    level: str,
    params: tuple[str, ...],
    caps: frozenset[Capability],
    override: Any,
) -> ToolCopy:
    """One tool's tailored-default copy merged under its validated override block."""
    default_name = tool_id
    default_description = _default_description(tool_id, caps)
    default_params = dict(_PARAM_DOC_DEFAULTS[tool_id])

    if override is None:
        return ToolCopy(
            id=tool_id, name=default_name, description=default_description,
            level=level, params=default_params,
        )
    if not isinstance(override, Mapping):
        msg = f"copy override for tool {tool_id!r} must be a mapping."
        raise TypeError(msg)
    _reject_unknown_keys(override, _ALLOWED_TOOL_FIELDS, f"copy override for tool {tool_id!r}")

    name = override.get("name", default_name)
    if not isinstance(name, str) or not name:
        msg = f"copy override {tool_id!r}.name must be a non-empty string."
        raise ValueError(msg)
    description = override.get("description", default_description)
    if not isinstance(description, str) or not description:
        msg = f"copy override {tool_id!r}.description must be a non-empty string."
        raise ValueError(msg)

    param_override = override.get("params", {})
    if not isinstance(param_override, Mapping):
        msg = f"copy override {tool_id!r}.params must be a mapping of parameter -> doc."
        raise TypeError(msg)
    _reject_unknown_keys(param_override, set(params), f"copy override {tool_id!r}.params")
    merged_params = dict(default_params)
    for param, doc in param_override.items():
        if not isinstance(doc, str) or not doc:
            msg = f"copy override {tool_id!r}.params[{param!r}] must be a non-empty string."
            raise ValueError(msg)
        merged_params[param] = doc

    return ToolCopy(
        id=tool_id, name=name, description=description, level=level, params=merged_params
    )


def _default_description(tool_id: str, caps: frozenset[Capability]) -> str:
    """The strong default description for ``tool_id``, tailored to declared capabilities
    (ADR-0052)."""
    filter_clause = " Supports exact-match metadata filtering." if Capability.FILTER in caps else ""
    if tool_id == SEARCH:
        return _search_description(caps) + filter_clause
    if tool_id == MORE_LIKE_THIS:
        return (
            "Find more, distinct results similar to a result you already have (breadth). Pass the "
            "content of the promising hit; escalate here from 'search'." + filter_clause
        )
    if tool_id == GET_CONTEXT:
        return (
            "Assemble grounded, citable context from hits you already retrieved: returns each as a "
            "chunk with structured provenance (source, page, offsets, confidence)."
        )
    if tool_id == LIST_SOURCES:
        return "List the retrieval sources this server is configured over, and their search modes."
    if tool_id == INDEX_STATUS:
        return "Report the backend's declared search capabilities and configured source count."
    msg = f"no default description for unknown tool {tool_id!r}."  # pragma: no cover
    raise KeyError(msg)  # pragma: no cover


def _search_description(caps: frozenset[Capability]) -> str:
    """Tailor the ``search`` verb to the backend's modes: hybrid / semantic / lexical (ADR-0052)."""
    has_dense = Capability.DENSE in caps or Capability.MULTIVECTOR in caps
    has_lexical = Capability.SPARSE in caps or Capability.BM25 in caps
    if has_dense and has_lexical:
        return "Search the corpus with hybrid semantic + lexical retrieval."
    if has_dense:
        return "Search the corpus with semantic (vector) retrieval."
    if has_lexical:
        return "Search the corpus with lexical (keyword) retrieval."
    return "Search the corpus."


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: AbstractSet[str], context: str
) -> None:
    """Fail closed on any key outside ``allowed`` — the operator-trusted schema guard."""
    unknown = set(mapping) - allowed
    if unknown:
        msg = f"{context} has unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}."
        raise ValueError(msg)


__all__ = [
    "GET_CONTEXT",
    "INDEX_STATUS",
    "LIST_SOURCES",
    "MORE_LIKE_THIS",
    "SEARCH",
    "TOOL_SURFACE",
    "ServerCopy",
    "ToolCopy",
]
