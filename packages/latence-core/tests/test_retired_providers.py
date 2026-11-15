"""Providers retired by a measurement (ADR-0075; the owner's rule of 14 Nov: an older module is
kept only where a measurement shows it adds value).

Measured on the text-side gold set of ``benchmark/resolution-modules/`` against the same run on
exact surfaces: the string cascade's fuzzy rungs (``disambiguation.cascade``) and the embedding
Disambiguator on its hashing default (``disambiguation.embedding`` with no ``embedder``) add no
value under the rule. No shipped stack and not the demo names them, and ``latence stack check``
says so of any stack that does, as a warning: a Pipeline that names one still runs."""

from __future__ import annotations

from pathlib import Path

import yaml
from latence_core.pipeline import Pipeline
from latence_core.stacks.check import Severity, check_stack
from latence_core.stacks.retired import RETIRED, retired_use

ROOT = Path(__file__).resolve().parents[3]
SHIPPED = sorted([*(ROOT / "stacks").glob("*.yaml"), *(ROOT / "deploy" / "stacks").glob("*.yaml")])


def _stack(provider: str, config: dict[str, object] | None = None) -> str:
    stages = [
        {"name": "parse", "capability": "parse", "provider": "parser.plaintext",
         "config": {"input_dir": "corpus"}},
        {"name": "chunk", "capability": "chunk", "provider": "chunk.sentence_window",
         "depends_on": ["parse"]},
        {"name": "entities", "capability": "entity_extraction", "provider": "entity.gazetteer",
         "depends_on": ["chunk"]},
        {"name": "disambiguate", "capability": "disambiguation", "provider": provider,
         "depends_on": ["entities"], **({"config": config} if config else {})},
    ]
    text: str = yaml.safe_dump(
        {"name": "retired-probe", "storage_uri": "file://out", "stages": stages}
    )
    return text


def test_no_shipped_stack_names_a_retired_provider() -> None:
    assert SHIPPED, "no shipped stack found"
    named = {
        str(path.relative_to(ROOT)): [
            f"{stage.name}: {stage.provider}"
            for stage in Pipeline.from_yaml_text(path.read_text(encoding="utf-8")).stages
            if retired_use(stage.provider, stage.config) is not None
        ]
        for path in SHIPPED
    }
    assert {k: v for k, v in named.items() if v} == {}


def test_stack_check_warns_on_a_retired_provider_and_names_the_measurement() -> None:
    found = check_stack(_stack("disambiguation.cascade"))
    retired = [f for f in found.findings if f.check == "retired"]
    assert [f.severity for f in retired] == [Severity.WARNING] and found.ok
    assert "disambiguation.exact_surface" in retired[0].message
    assert "benchmark/resolution-modules" in retired[0].message
    hashing = check_stack(_stack("disambiguation.embedding"))
    assert [f.check for f in hashing.findings if f.check == "retired"] == ["retired"]
    learned = check_stack(_stack("disambiguation.embedding", {
        "embedder": {"provider": "embedding.sentence_transformers", "config": {}}}))
    assert [f for f in learned.findings if f.check == "retired"] == []
    assert [f for f in check_stack(_stack("disambiguation.exact_surface")).findings
            if f.check == "retired"] == []


def test_every_retirement_cites_its_measurement() -> None:
    results = ROOT / "benchmark" / "resolution-modules" / "results.json"
    assert results.exists()
    assert set(RETIRED) == {"disambiguation.cascade", "disambiguation.embedding"}
    assert all("benchmark/resolution-modules" in why for why in RETIRED.values())
