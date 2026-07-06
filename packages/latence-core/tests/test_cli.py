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
