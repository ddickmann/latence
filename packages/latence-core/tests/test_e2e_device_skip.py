"""P3-F1 end-to-end — a GPU-declared Provider is skipped-with-flag by the routing path.

Source -> Parse -> Chunk -> Entity (a GPU-only Provider) -> Export through the local
Runner on THIS CPU host (``LATENCE_CUDA=0``). Asserts the G1 posture end to end: the
GPU-only Stage is skipped-with-flag (its StageMetrics carries ``skipped=True`` + a
reason + the declared profile columns), the run does NOT crash and does NOT fabricate a
GPU number, and the run resumes cleanly (the skip is sticky). Also asserts a CPU
Provider on the same host runs normally with its profile columns surfaced.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from latence_core import ProviderProfile, ProviderRegistry
from latence_core.contracts import EntityMention
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


class _GpuOnlyEntityExtractor:
    """A GPU-only EntityExtractor Provider — declared, never actually run on a CPU host.

    Its ``extract`` raises if ever called, proving the Runner SKIPS it (never runs it)
    on a CPU-only host rather than invoking it and crashing.
    """

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
            license_verified_on="2026-07-06",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def extract(self, chunks: Any) -> Any:
        msg = "GPU-only provider must never be executed on a CPU-only host (should be skipped)"
        raise AssertionError(msg)


def _registry_with_gpu_ner() -> ProviderRegistry:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access
    ep = metadata.EntryPoint(
        name="entity.gpu_only", value="test:injected", group="latence.providers"
    )
    object.__setattr__(ep, "load", lambda: _GpuOnlyEntityExtractor)
    discovered["entity.gpu_only"] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def _pipeline(store: str, docs_dir: Path, *, entity_provider: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: device-skip-slice
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
    provider: {entity_provider}
    depends_on: [chunk]
    config:
      device: auto
      labels:
        ORG:
          terms: [Acme Corp]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text("Acme Corp shipped the order.\n", encoding="utf-8")
    return docs


def test_gpu_provider_skipped_with_flag_on_cpu_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")  # deterministic CPU-only host
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = LocalRunner(registry=_registry_with_gpu_ner())

    # The run completes WITHOUT crashing, even though the entity Stage is GPU-only.
    report = runner.run(_pipeline(store, docs, entity_provider="entity.gpu_only"), run_id="skip")

    entities = next(m for m in report.stages if m.name == "entities")
    # Skipped-with-flag: the marker + reason + the declared profile columns are surfaced.
    assert entities.skipped is True
    assert entities.skip_reason is not None
    assert "GPU" in entities.skip_reason or "CUDA" in entities.skip_reason
    assert entities.device == "auto"
    assert entities.device_selected is None
    assert entities.compute == "gpu"
    assert entities.model_id == "vendor/gpu-ner-large"
    assert entities.license == "Apache-2.0"
    assert entities.records_out == 0  # no fabricated output

    # The skip is real: no entity records were produced (the GPU provider never ran).
    s = Storage.for_uri(store)
    ent_ckpt = f"{store}/_latence/runs/skip/checkpoints/entities.jsonl"
    assert s.exists(ent_ckpt)
    assert [x for x in s.read_text(ent_ckpt).splitlines() if x.strip()] == []


def test_gpu_skip_is_sticky_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs, entity_provider="entity.gpu_only")
    LocalRunner(registry=_registry_with_gpu_ner()).run(pipeline, run_id="resume")
    # A second run resumes every Stage from its checkpoint (the skip's empty checkpoint too):
    # the GPU-only extract() still raises if ever called, so a resume that re-ran it would
    # crash — proving the skip's empty checkpoint is authoritative for non-execution.
    report2 = LocalRunner(registry=_registry_with_gpu_ner()).run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)

    # The SKIP itself is sticky in the Quality Report, not just the non-re-execution: the
    # resumed entity Stage still surfaces skipped=True + its reason + the declared profile
    # columns, so a stack harness reading report2 sees the GPU stage as skipped, not as a
    # plain non-skipped Stage. (Regression guard: the resume path re-derives these columns.)
    entities2 = next(m for m in report2.stages if m.name == "entities")
    assert entities2.from_checkpoint is True
    assert entities2.skipped is True
    assert entities2.skip_reason is not None
    assert "GPU" in entities2.skip_reason or "CUDA" in entities2.skip_reason
    assert entities2.compute == "gpu"
    assert entities2.model_id == "vendor/gpu-ner-large"
    assert entities2.license == "Apache-2.0"
    assert entities2.device == "auto"
    assert entities2.device_selected is None
    assert entities2.records_out == 0


def test_cpu_provider_runs_with_profile_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same slice with the CPU gazetteer runs normally and surfaces its cpu profile
    # columns + a resolved device — the non-skipped half of the routing path.
    monkeypatch.setenv("LATENCE_CUDA", "0")
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(
        _pipeline(store, docs, entity_provider="entity.gazetteer"), run_id="cpu"
    )
    entities = next(m for m in report.stages if m.name == "entities")
    assert entities.skipped is False
    assert entities.compute == "cpu"
    assert entities.device == "auto"
    assert entities.device_selected == "cpu"
    assert entities.license == "Apache-2.0"
    assert entities.records_out >= 1  # the gazetteer actually extracted "Acme Corp"

    s = Storage.for_uri(store)
    rows = [
        EntityMention.model_validate_json(line)
        for line in s.read_text(f"{store}/_latence/runs/cpu/export/records.jsonl").splitlines()
        if line.strip()
    ]
    assert any(m.text == "Acme Corp" for m in rows)
