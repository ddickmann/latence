"""A Pipeline that still carries a template placeholder is refused, never run.

The stacks under ``stacks/`` and ``deploy/env/*/stack.yaml`` are templates: they carry
``{STORAGE_URI}`` and ``{CORPUS_DIR}`` (and, for a served-OCR stack, ``{OCR_BASE_URL}``) where a
machine-specific value belongs, and ``latence run`` reads a YAML verbatim. Run as shipped, a stack
used to exit 0: it created a directory literally named ``{STORAGE_URI}`` under the working
directory, found no documents at the relative path ``{CORPUS_DIR}`` and wrote an empty report.

These tests hold the refusal at every seam that executes a Pipeline: the CLI commands, the local
Runner and the Delta Runner. They also hold the two things the refusal must not break: a stack
whose placeholders were filled, and config text that only looks like a placeholder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from latence_core.cli import app
from latence_core.delta_runner import DeltaRunner
from latence_core.errors import ConfigError
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from typer.testing import CliRunner

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_STACKS = sorted((_REPO_ROOT / "stacks").glob("*.yaml"))
_DOC_ID = "sha256:" + "a" * 64

runner = CliRunner()


@pytest.fixture
def empty_cwd(tmp_path: Path) -> Iterator[Path]:
    """Run from an empty directory, so a literal ``{STORAGE_URI}`` directory would be visible."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        yield cwd
    finally:
        os.chdir(previous)


def _template() -> Pipeline:
    return Pipeline.from_yaml_text((_REPO_ROOT / "stacks" / "default.yaml").read_text("utf-8"))


def test_there_are_shipped_stacks_to_check() -> None:
    assert _SHIPPED_STACKS, f"no stacks found under {_REPO_ROOT / 'stacks'}"


# -- the CLI ---------------------------------------------------------------------------------


@pytest.mark.parametrize("stack", _SHIPPED_STACKS, ids=lambda path: path.name)
def test_latence_run_refuses_every_shipped_stack_as_shipped(stack: Path, empty_cwd: Path) -> None:
    result = runner.invoke(app, ["run", str(stack)])
    assert result.exit_code == 2, result.output
    assert "[config]" in result.output
    assert "{STORAGE_URI}" in result.output
    assert "{CORPUS_DIR}" in result.output
    # Refused before anything was written: no directory named after the placeholder, no run state.
    assert list(empty_cwd.iterdir()) == []


@pytest.mark.parametrize(
    "argv",
    [
        ["process", "--config", "STACK"],
        ["delta", "STACK"],
        ["delta", "STACK", "--dry-run"],
        ["retract", "STACK", "--doc", _DOC_ID],
        ["purge", "STACK", "--doc", _DOC_ID, "--yes"],
    ],
    ids=lambda argv: " ".join(argv[:1] + argv[2:]),
)
def test_every_command_that_executes_a_pipeline_refuses_a_template(
    argv: list[str], empty_cwd: Path
) -> None:
    stack = str(_REPO_ROOT / "stacks" / "default.yaml")
    result = runner.invoke(app, [stack if arg == "STACK" else arg for arg in argv])
    assert result.exit_code == 2, result.output
    assert "{STORAGE_URI}" in result.output
    assert list(empty_cwd.iterdir()) == []


# -- the Runners (library callers do not go through the CLI) ---------------------------------


def test_the_local_runner_raises_a_config_error_before_writing(empty_cwd: Path) -> None:
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}") as excinfo:
        LocalRunner().run(_template())
    message = str(excinfo.value)
    assert "storage_uri" in message
    assert "stages.source.config.path" in message and "{CORPUS_DIR}" in message
    assert list(empty_cwd.iterdir()) == []


def test_the_delta_runner_raises_a_config_error_before_writing(empty_cwd: Path) -> None:
    delta = DeltaRunner()
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}"):
        delta.preview(_template())
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}"):
        delta.apply_delta(_template(), "delta-1")
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}"):
        delta.retract(_template(), "retract-1", [_DOC_ID])
    with pytest.raises(ConfigError, match=r"\{STORAGE_URI\}"):
        delta.purge(_template(), "purge-1", [_DOC_ID])
    assert list(empty_cwd.iterdir()) == []


# -- what counts as a placeholder ------------------------------------------------------------


def _pipeline(*, storage_uri: str = "file:///tmp/store", config: str = "{path: /tmp/docs}") -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: placeholder-probe
storage_uri: '{storage_uri}'
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {config}
"""
    )


@pytest.mark.parametrize("stack", _SHIPPED_STACKS, ids=lambda path: path.name)
def test_a_filled_stack_carries_no_placeholder(stack: Path) -> None:
    text = (
        stack.read_text(encoding="utf-8")
        .replace("{CORPUS_DIR}", "file:///tmp/corpus")
        .replace("{STORAGE_URI}", "file:///tmp/storage")
    )
    pipeline = Pipeline.from_yaml_text(text)
    assert pipeline.unresolved_placeholders() == []
    pipeline.require_resolved()


def test_placeholders_are_found_wherever_a_string_sits() -> None:
    pipeline = Pipeline.from_yaml_text(
        """
name: placeholder-probe
storage_uri: 's3://{BUCKET}/latence'
gold_set_uri: '{STORAGE_URI}/gold.json'
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{CORPUS_DIR}/inbox'
      nested:
        endpoints: ['http://ok.example', '{OCR_BASE_URL}/v1']
"""
    )
    assert pipeline.unresolved_placeholders() == [
        ("storage_uri", "{BUCKET}"),
        ("gold_set_uri", "{STORAGE_URI}"),
        ("stages.source.config.path", "{CORPUS_DIR}"),
        ("stages.source.config.nested.endpoints[1]", "{OCR_BASE_URL}"),
    ]
    with pytest.raises(ConfigError, match=r"\{BUCKET\}"):
        pipeline.require_resolved()


@pytest.mark.parametrize(
    "value",
    [
        "{id}",  # a lower-case format field (the id_template default)
        "{{LABELS}}",  # an escaped brace in a format template is literal text
        r"\N{BULLET} item",  # a named character in a regular expression
        r"\d{3}-\d{4}",  # a repetition count
        '{"labels": ["ORG"]}',  # inline JSON
        "plain text with } and { braces",
    ],
)
def test_text_that_only_looks_like_a_placeholder_is_left_alone(value: str) -> None:
    pipeline = _pipeline()
    pipeline.stages[0].config["free_text"] = value
    assert pipeline.unresolved_placeholders() == []
    pipeline.require_resolved()


def test_the_message_names_where_and_what_but_no_other_config_value() -> None:
    pipeline = _pipeline(config="{path: '{CORPUS_DIR}', api_key: sk-do-not-print}")
    with pytest.raises(ConfigError) as excinfo:
        pipeline.require_resolved()
    message = str(excinfo.value)
    assert "stages.source.config.path" in message and "{CORPUS_DIR}" in message
    assert "sk-do-not-print" not in message
    assert "latence stack validate" in message
