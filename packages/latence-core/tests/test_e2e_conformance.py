"""End-to-end: a real LocalRunner run whose Quality Report carries the conformance roll-up.

P3-F3 requires ≥1 end-to-end test through the local Runner and that conformance metrics land in
the Quality Report (the Baseline "Observable" bar). This runs a real Source→Parse→Chunk→Entity→
Export pipeline through :class:`~latence_core.runner.LocalRunner`, then attaches the
:class:`~latence_core.conformance.report.ConformanceReport` (the enterprise-readiness gate over
every registered Provider) to the produced :class:`~latence_core.quality.QualityReport` and
asserts it round-trips through JSON — the machine-readable audit the "model-agnostic" claim rests
on, folded into the same first-class artifact every other metric lands in.

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest; ``LATENCE_CUDA=0``
pinned below so the device seam is reproducible).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import build_conformance_report
from latence_core.pipeline import Pipeline
from latence_core.quality import QUALITY_SCHEMA_VERSION, QualityReport
from latence_core.runner import LocalRunner

_LABELS_YAML = """
      labels:
        ORG:
          terms: [Acme Corp]
          confidence: 0.95
        PERSON:
          terms: [Alice]
          confidence: 0.8
"""


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: conformance-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
    config:{_LABELS_YAML}
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works for Acme Corp in Berlin.\n", encoding="utf-8"
    )
    return docs


def test_run_report_carries_conformance_rollup(tmp_path: Path) -> None:
    """A real LocalRunner run's Quality Report carries the attached conformance gate result."""
    store = f"file://{tmp_path / 'store'}"
    report = LocalRunner().run(_pipeline(store, _docs(tmp_path)), run_id="conf-e2e")

    # The run itself produced a valid report (the pipeline ran end to end).
    assert report.completeness.passed
    assert report.schema_version == QUALITY_SCHEMA_VERSION
    # A fresh run reports no conformance section until one is attached (it is a tooling sweep).
    assert report.conformance is None

    # Attach the enterprise-readiness gate over every registered Provider — the audit the
    # "model-agnostic" claim rests on, folded into the same first-class artifact.
    conformance = build_conformance_report(ProviderRegistry())
    report.conformance = conformance
    assert report.conformance is not None
    assert report.conformance.providers_total == 56
    assert report.conformance.all_passed
    assert report.conformance.providers_failed == 0

    # The whole report (run metrics + conformance roll-up) round-trips through JSON — the
    # machine-readable artifact stays valid with the new section (counts only, no PII).
    round_tripped = QualityReport.model_validate_json(report.to_json())
    assert round_tripped.conformance is not None
    assert round_tripped.conformance.providers_total == 56
    assert round_tripped.conformance.all_passed
    # The human-readable render still works (the conformance section is machine-only).
    assert "Quality Report" in round_tripped.render()


def test_conformance_rollup_is_counts_only_no_pii(tmp_path: Path) -> None:
    """The attached conformance report carries no record content / PII — safe to persist/share."""
    store = f"file://{tmp_path / 'store'}"
    report = LocalRunner().run(_pipeline(store, _docs(tmp_path)), run_id="conf-e2e-2")
    report.conformance = build_conformance_report(ProviderRegistry())
    serialized = report.to_json()
    # No raw PII from the canonical fixture leaks into the persisted report.
    assert "alice@example.com" not in serialized
    assert "victim@secret-corp.example" not in serialized
    assert "sk-live-" not in serialized
