"""Config-driven, capability-tailored, schema-validated copy — :class:`ServerCopy` (AC2, #174).

Asserts the observable copy an agent would read: that defaults auto-tailor to the backend's declared
capabilities, that operator overrides merge field-by-field, that the override schema fails closed on
anything unknown (operator-trusted config), and that a built copy is frozen (not agent-mutable).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from latence_retrieval import Capability
from latence_retrieval.mcp import ServerCopy
from latence_retrieval.mcp.copy import ToolCopy
from pydantic import ValidationError


def test_search_copy_is_hybrid_when_dense_and_lexical_declared() -> None:
    copy = ServerCopy.build(frozenset({Capability.DENSE, Capability.BM25}))
    assert "hybrid" in copy.tool("search").description.lower()


def test_search_copy_is_semantic_for_dense_only() -> None:
    copy = ServerCopy.build(frozenset({Capability.DENSE}))
    desc = copy.tool("search").description.lower()
    assert "semantic" in desc
    assert "hybrid" not in desc


def test_search_copy_is_lexical_for_bm25_only() -> None:
    copy = ServerCopy.build(frozenset({Capability.BM25}))
    desc = copy.tool("search").description.lower()
    assert "lexical" in desc or "keyword" in desc


def test_filter_capability_adds_a_filter_mention() -> None:
    with_filter = ServerCopy.build(frozenset({Capability.BM25, Capability.FILTER}))
    without = ServerCopy.build(frozenset({Capability.BM25}))
    assert "filter" in with_filter.tool("search").description.lower()
    assert "filter" not in without.tool("search").description.lower()


def test_defaults_render_the_full_ladder_ordered() -> None:
    copy = ServerCopy.build(frozenset({Capability.BM25}))
    assert [t.id for t in copy.ordered_tools()] == [
        "search",
        "more_like_this",
        "get_context",
        "list_sources",
        "index_status",
    ]


def test_operator_override_merges_field_by_field() -> None:
    copy = ServerCopy.build(
        frozenset({Capability.BM25}),
        {
            "operating_policy": "Search the ACME handbook.",
            "tools": {
                "search": {
                    "name": "handbook_search",
                    "description": "Search the ACME employee handbook.",
                    "params": {"query": "What to look up in the handbook."},
                }
            },
        },
    )
    search = copy.tool("search")
    assert copy.operating_policy == "Search the ACME handbook."
    assert search.name == "handbook_search"
    assert search.description == "Search the ACME employee handbook."
    assert search.params["query"] == "What to look up in the handbook."
    # An un-overridden param keeps its tailored default.
    assert "top_k" in search.params
    # An un-overridden tool keeps its default name.
    assert copy.tool("get_context").name == "get_context"
    # The rename is reflected in the dispatch map.
    assert copy.name_to_id()["handbook_search"] == "search"


def test_unknown_tool_id_in_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ServerCopy.build(frozenset({Capability.BM25}), {"tools": {"drop_table": {}}})


def test_unknown_tool_field_in_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ServerCopy.build(
            frozenset({Capability.BM25}), {"tools": {"search": {"colour": "blue"}}}
        )


def test_unknown_param_in_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ServerCopy.build(
            frozenset({Capability.BM25}),
            {"tools": {"search": {"params": {"not_a_param": "x"}}}},
        )


def test_top_level_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ServerCopy.build(frozenset({Capability.BM25}), {"policy": "typo"})


def test_empty_description_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ServerCopy.build(
            frozenset({Capability.BM25}), {"tools": {"search": {"description": ""}}}
        )


def test_built_copy_is_frozen_not_agent_mutable() -> None:
    copy = ServerCopy.build(frozenset({Capability.BM25}))
    with pytest.raises(ValidationError):
        copy.operating_policy = "hijacked"
    tool = copy.tool("search")
    with pytest.raises(ValidationError):
        tool.description = "hijacked"


# --- from_file: the sandboxed operator config path -----------------------------------------------


def test_from_file_loads_and_tailors(tmp_path: Path) -> None:
    cfg = tmp_path / "copy.json"
    cfg.write_text(
        json.dumps({"tools": {"search": {"name": "kb_search"}}}), encoding="utf-8"
    )
    copy = ServerCopy.from_file(cfg, frozenset({Capability.BM25}))
    assert copy.tool("search").name == "kb_search"
    # Capability tailoring still applies to the un-overridden description.
    assert "lexical" in copy.tool("search").description.lower() or "keyword" in copy.tool(
        "search"
    ).description.lower()


def test_from_file_rejects_a_path_outside_the_sandbox(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    cfg = outside / "copy.json"
    cfg.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the sandbox"):
        ServerCopy.from_file(cfg, frozenset({Capability.BM25}), base_dir=sandbox)


def test_from_file_rejects_non_object_json(tmp_path: Path) -> None:
    cfg = tmp_path / "copy.json"
    cfg.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        ServerCopy.from_file(cfg, frozenset({Capability.BM25}))


def test_from_file_rejects_invalid_json(tmp_path: Path) -> None:
    cfg = tmp_path / "copy.json"
    cfg.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        ServerCopy.from_file(cfg, frozenset({Capability.BM25}))


def test_tool_copy_is_frozen() -> None:
    tc = ToolCopy(id="x", name="x", description="d", level="L1", params={})
    with pytest.raises(ValidationError):
        tc.name = "y"
