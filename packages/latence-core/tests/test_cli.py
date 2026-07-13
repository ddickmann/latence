"""CLI smoke test: `latence run pipeline.yaml` executes and prints a report."""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_run(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        f"""
name: cli-demo
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["run", str(pipeline_file), "--run-id", "cli"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pipeline_name"] == "cli-demo"
    assert payload["document_count"] == 2


def test_cli_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 2


def _plaintext_pipeline(docs_dir: Path, tmp_path: Path) -> Path:
    store = f"file://{tmp_path}/store"
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text(
        f"""
name: cli-tuned-demo
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}'}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
""",
        encoding="utf-8",
    )
    return pipeline_file


def _write_tuned(
    tmp_path: Path, *, fp: object, provider: str, knobs: dict[str, object]
) -> Path:
    """Write a minimal tuned/<device>.yaml with one provider tuning, fingerprinted to ``fp``."""
    from latence_core.tune.report import KnobSetting, ProviderTuning, TuneReport

    report = TuneReport(
        stack="cli-tuned-demo",
        sample_size=8,
        seed=0,
        device_fingerprint=fp,  # type: ignore[arg-type]
        providers=[ProviderTuning(provider=provider, winner=KnobSetting(knobs=knobs))],
    )
    path = tmp_path / "tuned.yaml"
    path.write_text(report.to_yaml(), encoding="utf-8")
    return path


def test_cli_run_tuned_applies_on_matching_fingerprint(
    docs_dir: Path, tmp_path: Path, monkeypatch: object
) -> None:
    """`latence run --tuned` overlays the tune when the device_fingerprint matches this host."""
    # Pin the fingerprint to a known device so the tuned config matches.
    monkeypatch.setenv("LATENCE_CUDA", "1")  # type: ignore[attr-defined]
    monkeypatch.setenv("LATENCE_DEVICE", "Fake Match GPU")  # type: ignore[attr-defined]
    from latence_core.tune.fingerprint import probe_device_fingerprint

    fp = probe_device_fingerprint()
    pipeline_file = _plaintext_pipeline(docs_dir, tmp_path)
    # parser.plaintext is not a heavy provider, but the overlay is provider-name-keyed; use it so
    # the
    # CLI exercises a real applied overlay end-to-end without a heavy dep (knobs land as extra
    # config).
    tuned = _write_tuned(
        tmp_path, fp=fp, provider="parser.plaintext", knobs={"encoding": "utf-8"}
    )
    result = runner.invoke(app, ["run", str(pipeline_file), "--tuned", str(tuned), "--run-id", "t"])
    assert result.exit_code == 0, result.output
    assert "applied tuned config" in result.output


def test_cli_run_tuned_warns_and_defaults_on_mismatch(
    docs_dir: Path, tmp_path: Path, monkeypatch: object
) -> None:
    """`latence run --tuned` from a DIFFERENT device is NOT applied — warns + uses stack defaults."""
    monkeypatch.setenv("LATENCE_CUDA", "1")  # type: ignore[attr-defined]
    monkeypatch.setenv("LATENCE_DEVICE", "This Host GPU")  # type: ignore[attr-defined]
    from latence_core.tune.fingerprint import DeviceFingerprint

    pipeline_file = _plaintext_pipeline(docs_dir, tmp_path)
    other = DeviceFingerprint(cuda=True, gpu_name="A Totally Different GPU", cpu_count=64)
    tuned = _write_tuned(tmp_path, fp=other, provider="parser.plaintext", knobs={"x": 1})
    result = runner.invoke(app, ["run", str(pipeline_file), "--tuned", str(tuned), "--run-id", "t"])
    # The run still succeeds (defaults), and the mismatch is announced (never silently applied).
    assert result.exit_code == 0, result.output
    assert "NOT applying" in result.output


def test_cli_run_tuned_missing_file(docs_dir: Path, tmp_path: Path) -> None:
    pipeline_file = _plaintext_pipeline(docs_dir, tmp_path)
    result = runner.invoke(
        app, ["run", str(pipeline_file), "--tuned", str(tmp_path / "nope.yaml")]
    )
    assert result.exit_code == 2
