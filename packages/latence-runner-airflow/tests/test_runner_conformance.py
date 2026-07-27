"""Runner conformance — the SAME Stage reported the SAME way under BOTH Runners (C3, ADR-0003).

ADR-0003 says the framework is a library of typed Stages and execution is delegated to
interchangeable Runners. That claim was only ever tested on the *records*: two Runners exported
byte-identical rows. It was never tested on what a Runner **reports** or on how a Runner
**behaves when a Stage cannot run** — and both had silently forked, because everything a Runner
needs to *be* a Runner lived inline in one Runner's loop:

* the local Runner built a 22-column ``StageMetrics``; the Airflow adapter hand-built 7 of them,
  so an Airflow run reported ``compute=None``, ``model_id=None``, ``license=None``,
  ``device=None``, ``device_selected=None`` and zero induction columns — not because the Stage
  lacked them, but because the second Runner never learned to ask;
* the skip-with-flag posture (a GPU-only Provider on a CPU-only host) was inline in the local
  ``_run`` loop, so the SAME Pipeline was skipped-with-flag locally and **crashed** under Airflow.

These two tests are parameterised over both Runners and assert they agree. Against the pre-C3
code they FAIL — the first on the profile/device columns, the second with the GPU Provider's own
``AssertionError`` escaping the Airflow adapter.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from latence_core import ProviderProfile, ProviderRegistry
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport, StageMetrics
from latence_core.runner import LocalRunner
from latence_core.stages.entity import GazetteerEntityExtractor
from latence_runner_airflow import AirflowRunner

# Columns that are ALLOWED to differ between Runners. ``runner`` is the whole point of the
# comparison; ``duration_seconds`` is wall clock; ``from_checkpoint`` is a local-Runner resume
# concept (Airflow keeps its state in Airflow, ADR-0010). Everything else must match exactly.
_RUNNER_SPECIFIC = frozenset({"runner", "duration_seconds", "from_checkpoint"})


class _ProfiledGazetteer(GazetteerEntityExtractor):
    """The stock gazetteer extractor, with a declared ProviderProfile bolted on.

    Deliberately profile-declaring: ``compute``/``model_id``/``license`` plus the routed
    ``device``/``device_selected`` are exactly the columns the Airflow adapter used to leave
    ``None``, because it hand-built its metrics row instead of asking the executor for one.
    Subclassing the real Provider keeps the *records* identical to a normal gazetteer run, so the
    test isolates the reporting difference.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id="test/profiled-gazetteer",
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source="https://example.invalid/profiled-gazetteer",
            license_verified_on="2026-07-26",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )


class _GpuOnlyExtractor:
    """A GPU-only EntityExtractor. ``extract`` raises if it is ever reached on a CPU host."""

    @classmethod
    def profile(cls) -> ProviderProfile:
        return ProviderProfile(
            compute="gpu",
            memory_mb=8000,
            model_id="vendor/gpu-ner-large",
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source="https://huggingface.co/vendor/gpu-ner-large",
            license_verified_on="2026-07-26",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def extract(self, chunks: Any) -> Any:
        msg = "a GPU-only Provider must never execute on a CPU-only host — it must be skipped"
        raise AssertionError(msg)


def _registry(name: str, cls: type) -> ProviderRegistry:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access
    ep = metadata.EntryPoint(name=name, value="test:injected", group="latence.providers")
    object.__setattr__(ep, "load", lambda: cls)
    discovered[name] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def _pipeline(store: str, docs_dir: Path, provider: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: runner-conformance
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
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
    provider: {provider}
    depends_on: [chunk]
    config:
      device: auto
      labels:
        ORG:
          terms: [Acme]
          confidence: 0.9
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text("Acme shipped the order to the depot.\n", encoding="utf-8")
    return docs


def _shape(metrics: StageMetrics) -> dict[str, object]:
    return {
        field: value
        for field, value in metrics.model_dump().items()
        if field not in _RUNNER_SPECIFIC
    }


def _by_stage(report: QualityReport) -> dict[str, StageMetrics]:
    return {m.name: m for m in report.stages}


def test_stage_metrics_identical_under_both_runners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every StageMetrics column except the Runner's own name must agree across both Runners.

    Pre-C3 this fails on the ``entities`` Stage: the local Runner reports
    ``compute='cpu'``/``model_id='test/profiled-gazetteer'``/``license='Apache-2.0'``/
    ``device='auto'``/``device_selected='cpu'``, the Airflow adapter reports ``None`` for all
    five, because it hand-built a 7-column metrics row instead of asking the executor for one.
    """
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    registry = _registry("entity.profiled", _ProfiledGazetteer)

    local = LocalRunner(registry=_registry("entity.profiled", _ProfiledGazetteer)).run(
        _pipeline(f"file://{tmp_path}/local", docs, "entity.profiled"), run_id="r"
    )
    airflow = AirflowRunner(registry=registry).run(
        _pipeline(f"file://{tmp_path}/airflow", docs, "entity.profiled"), run_id="r"
    )

    local_stages, airflow_stages = _by_stage(local), _by_stage(airflow)
    assert set(local_stages) == set(airflow_stages)
    for name in sorted(local_stages):
        assert _shape(local_stages[name]) == _shape(airflow_stages[name]), (
            f"Stage {name!r} is reported differently by the two Runners. ADR-0003 says the "
            f"identical Stages run under a pluggable Runner — a Runner that hand-assembles its "
            f"own metrics row forks the Quality Report."
        )

    # Guard the guard: the profile columns must actually be POPULATED, otherwise the equality
    # above would pass trivially on two equally-empty rows.
    entities = local_stages["entities"]
    assert entities.compute == "cpu"
    assert entities.model_id == "test/profiled-gazetteer"
    assert entities.license == "Apache-2.0"
    assert entities.device == "auto"
    assert entities.device_selected == "cpu"


def test_gpu_only_provider_is_skipped_with_flag_under_both_runners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GPU-only Provider on a CPU-only host is skipped-with-flag under EVERY Runner.

    Pre-C3 the local Runner skips it (an empty output, ``skipped=True``, a reason) while the
    Airflow adapter constructs and calls the Provider — so this test fails there with the
    Provider's own ``AssertionError``. Same Pipeline, same host, two different failure modes is
    exactly what ADR-0003 forbids: the posture belonged in the shared executor, not in one
    Runner's loop.
    """
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)

    local = LocalRunner(registry=_registry("entity.gpu_only", _GpuOnlyExtractor)).run(
        _pipeline(f"file://{tmp_path}/local", docs, "entity.gpu_only"), run_id="r"
    )
    airflow = AirflowRunner(registry=_registry("entity.gpu_only", _GpuOnlyExtractor)).run(
        _pipeline(f"file://{tmp_path}/airflow", docs, "entity.gpu_only"), run_id="r"
    )

    for report in (local, airflow):
        entities = _by_stage(report)["entities"]
        assert entities.skipped is True
        assert entities.skip_reason
        assert entities.records_out == 0
        assert entities.compute == "gpu"
    assert _shape(_by_stage(local)["entities"]) == _shape(_by_stage(airflow)["entities"])
