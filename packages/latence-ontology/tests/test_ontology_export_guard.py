"""The export is checked against its own shipped schema before a byte of it is written: a host
reads ``graph-export/4`` by that schema, so a file that breaks it is refused, not written."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from _ontology_toy import OPTIONS, SENSITIVE, VIEWS
from latence_ontology import cli
from latence_ontology.cli import build
from latence_ontology.labels import with_identity


def _toy(tmp_path: Path) -> Path:
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({
        "name": "toy", "title": "Toy estate", "schema_prefix": "acme_", "schemas": ["shop", "marts"],
        "personal_objects": ["shop.owners", "shop.vehicles"], "resolve": ["make"],
        "sensitive": SENSITIVE, "ignore_columns": list(OPTIONS["ignore_columns"]),
        "derived_schemas": list(OPTIONS["derived_schemas"])}))
    (tmp_path / "views.txt").write_text(VIEWS)
    return cfg


def test_an_export_that_breaks_its_schema_is_refused_and_nothing_is_written(
    tmp_path: Path, toy_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = with_identity

    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        export = real(*args, **kwargs)
        return {**export, "schema": "graph-export/2"}

    monkeypatch.setattr(cli, "with_identity", broken)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="graph-export/4"):
        build(config=_toy(tmp_path), duckdb_path=toy_path, views=tmp_path / "views.txt", verified=None, out=out)
    assert not [p.name for p in out.glob("ontology*")]


def test_a_conforming_export_is_written(tmp_path: Path, toy_path: Path) -> None:
    out = tmp_path / "out"
    build(config=_toy(tmp_path), duckdb_path=toy_path, views=tmp_path / "views.txt", verified=None, out=out)
    assert sorted(p.name for p in out.glob("ontology*")) == [
        "ontology.json", "ontology.prov.ttl", "ontology.shacl.ttl", "ontology.ttl"]
