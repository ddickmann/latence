"""Provider discovery via importlib.metadata entry points."""

from __future__ import annotations

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.stages.export import JsonlParquetExport
from latence_core.stages.source import LocalFolderSource


def test_builtin_providers_discovered() -> None:
    reg = ProviderRegistry()
    names = reg.names()
    assert "source.local_folder" in names
    assert "export.jsonl_parquet" in names
    # S4/S5 in-core reference extractors are discoverable too.
    assert "entity.gazetteer" in names
    assert "relation.pattern" in names


def test_relation_pattern_provider_loads() -> None:
    from latence_core.stages.relation import PatternRelationExtractor

    reg = ProviderRegistry()
    assert reg.load("relation.pattern") is PatternRelationExtractor


def test_load_returns_expected_class() -> None:
    reg = ProviderRegistry()
    assert reg.load("source.local_folder") is LocalFolderSource
    assert reg.load("export.jsonl_parquet") is JsonlParquetExport


def test_cross_package_parser_discovered() -> None:
    # This provider lives in latence-parser-plaintext, a SEPARATE package.
    reg = ProviderRegistry()
    assert "parser.plaintext" in reg.names()
    cls = reg.load("parser.plaintext")
    assert cls.__module__.startswith("latence_parser_plaintext")


def test_unknown_provider_raises() -> None:
    reg = ProviderRegistry()
    with pytest.raises(KeyError, match="No Provider registered"):
        reg.load("does.not.exist")


def test_load_is_cached() -> None:
    reg = ProviderRegistry()
    assert reg.load("source.local_folder") is reg.load("source.local_folder")
