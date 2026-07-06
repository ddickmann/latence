"""Pipeline YAML load/validate + DAG topological order."""

from __future__ import annotations

import pytest
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.runner import topological_order

VALID = """
name: p
storage_uri: memory://bucket/run
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {path: 'memory://bucket/docs'}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""


def test_load_valid_yaml() -> None:
    p = Pipeline.from_yaml_text(VALID)
    assert p.name == "p"
    assert [s.capability for s in p.stages] == [
        CapabilityKind.SOURCE,
        CapabilityKind.PARSE,
        CapabilityKind.EXPORT,
    ]


def test_roundtrip_yaml() -> None:
    p = Pipeline.from_yaml_text(VALID)
    p2 = Pipeline.from_yaml_text(p.to_yaml_text())
    assert p2 == p


def test_duplicate_stage_names_rejected() -> None:
    bad = VALID.replace("name: parse", "name: source")
    with pytest.raises(ValueError, match="Duplicate Stage names"):
        Pipeline.from_yaml_text(bad)


def test_unknown_dependency_rejected() -> None:
    bad = VALID.replace("depends_on: [source]", "depends_on: [ghost]")
    with pytest.raises(ValueError, match="unknown Stage"):
        Pipeline.from_yaml_text(bad)


def test_unknown_capability_rejected() -> None:
    bad = VALID.replace("capability: parse", "capability: nonsense")
    with pytest.raises(ValueError):
        Pipeline.from_yaml_text(bad)


def test_topological_order_linear() -> None:
    p = Pipeline.from_yaml_text(VALID)
    ordered = [s.name for s in topological_order(list(p.stages))]
    assert ordered.index("source") < ordered.index("parse") < ordered.index("export")


def test_cycle_detected() -> None:
    stages = [
        Stage(name="a", capability=CapabilityKind.PARSE, provider="x", depends_on=["b"]),
        Stage(name="b", capability=CapabilityKind.PARSE, provider="x", depends_on=["a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        topological_order(stages)
