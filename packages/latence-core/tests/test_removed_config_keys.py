"""A Stage config key that ADR-0068 removed is refused, never ignored.

ADR-0068 took out the query-time retrieval tooling and with it nine Stage config keys: two on
Graph Assembly and seven on Export. The Providers that used to read them simply stopped reading
them, so a stack that still carried ``hyperedges``, ``context_columns`` or ``hyperedge_kv`` ran to
completion and produced none of what the key asked for, without a message. ``latence stack check``
reported two of the nine (``graph_features`` and ``bm25``), and only when they were written as a
``{provider, config}`` block, because that shape tripped a different check.

These tests hold the refusal at both ends: the offline check reports every key, and every path
that executes a Pipeline raises a ``ConfigError`` naming the key, the Stage and ADR-0068 before
anything is written. They also hold what the refusal must NOT do: reject a key under a Stage kind
where it was never valid, or reject unknown keys in general.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from latence_core.cli import app
from latence_core.delta_runner import DeltaRunner
from latence_core.errors import ConfigError
from latence_core.pipeline import REMOVED_CONFIG_KEYS, CapabilityKind, Pipeline
from latence_core.runner import LocalRunner
from latence_core.stacks.check import check_stack
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOC_ID = "sha256:" + "a" * 64

cli = CliRunner()

# Where each removed key was once valid, read out of the tree before ADR-0068 (the Graph Assembly
# Stage, the corpus Export and the knowledge-graph Export), written out here rather than derived
# from the list under test so that a key dropped from the list turns a test red.
_GRAPH_KEYS = ("graph_features", "hyperedges")
_EXPORT_KEYS = (
    "bm25",
    "context_columns",
    "fde_converter",
    "hyperedge_kv",
    "keep_multivectors",
    "multivector_embedder",
    "sparse_embedder",
)
_STAGE_OF = {
    **dict.fromkeys(_GRAPH_KEYS, "graph"),
    **dict.fromkeys(_EXPORT_KEYS, "export"),
}
_ALL_KEYS = (*_GRAPH_KEYS, *_EXPORT_KEYS)

# A removed key is refused whatever value it carries: the flag form, the mapping form and the
# ``{provider, config}`` block form were all written in real stacks.
_VALUES: tuple[Any, ...] = (True, {"max_size": 8}, {"provider": "anything.at_all", "config": {}})


def _stack(storage: Path, docs: Path) -> dict[str, Any]:
    return {
        "name": "removed-key-probe",
        "storage_uri": f"file://{storage}",
        "stages": [
            {
                "name": "source",
                "capability": "source",
                "provider": "source.local_folder",
                "config": {"path": f"file://{docs}", "sensitivity": "internal"},
            },
            {
                "name": "parse",
                "capability": "parse",
                "provider": "parser.plaintext",
                "depends_on": ["source"],
            },
            {
                "name": "chunk",
                "capability": "chunk",
                "provider": "chunk.markdown",
                "depends_on": ["parse"],
            },
            {
                "name": "entities",
                "capability": "entity_extraction",
                "provider": "entity.gazetteer",
                "depends_on": ["chunk"],
                "config": {"labels": {"WORD": {"terms": ["notes"], "confidence": 0.9}}},
            },
            {
                "name": "resolve",
                "capability": "disambiguation",
                "provider": "disambiguation.cascade",
                "depends_on": ["entities"],
            },
            {
                "name": "graph",
                "capability": "graph_assembly",
                "provider": "graph.canonical",
                "depends_on": ["resolve"],
                "config": {"corpus_id": "probe"},
            },
            {
                "name": "export",
                "capability": "export",
                "provider": "export.jsonl_parquet",
                "depends_on": ["chunk"],
                "config": {"basename": "corpus"},
            },
        ],
    }


def _stage(stack: dict[str, Any], name: str) -> dict[str, Any]:
    stage: dict[str, Any] = next(s for s in stack["stages"] if s["name"] == name)
    stage.setdefault("config", {})
    return stage


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """``(storage, docs)``: an absent storage root and a one-document corpus."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.txt").write_text("Notes about the framework.\n", encoding="utf-8")
    return tmp_path / "store", docs


def _with_key(dirs: tuple[Path, Path], key: str, value: Any = True) -> dict[str, Any]:
    stack = _stack(*dirs)
    _stage(stack, _STAGE_OF[key])["config"][key] = value
    return stack


def _pipeline(stack: dict[str, Any]) -> Pipeline:
    return Pipeline.model_validate(stack)


# -- the list itself ---------------------------------------------------------------------------


def test_the_closed_list_holds_the_nine_keys_under_the_two_stage_kinds() -> None:
    assert {kind: tuple(sorted(keys)) for kind, keys in REMOVED_CONFIG_KEYS.items()} == {
        CapabilityKind.GRAPH_ASSEMBLY: _GRAPH_KEYS,
        CapabilityKind.EXPORT: _EXPORT_KEYS,
    }


def test_the_probe_stack_runs_when_it_carries_no_removed_key(dirs: tuple[Path, Path]) -> None:
    """The baseline: the same stack, without a removed key, checks clean and runs."""
    stack = _stack(*dirs)
    assert check_stack(_pipeline(stack)).ok
    LocalRunner().run(_pipeline(stack))
    assert dirs[0].exists()


# -- the offline check reports every key -------------------------------------------------------


@pytest.mark.parametrize("value", _VALUES, ids=["flag", "mapping", "block"])
@pytest.mark.parametrize("key", _ALL_KEYS)
def test_stack_check_reports_every_removed_key_once(
    key: str, value: Any, dirs: tuple[Path, Path]
) -> None:
    result = check_stack(_pipeline(_with_key(dirs, key, value)))

    assert not result.ok, result.render()
    naming = [f for f in result.errors if f"{key!r}" in f.message]
    assert len(naming) == 1, result.render()
    (finding,) = naming
    assert finding.check == "config_key"
    assert finding.stage == _STAGE_OF[key]
    assert "ADR-0068" in finding.message


def test_latence_stack_check_exits_one_and_names_the_key(
    dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    path = tmp_path / "stack.yaml"
    path.write_text(yaml.safe_dump(_with_key(dirs, "hyperedge_kv", False)), encoding="utf-8")

    result = cli.invoke(app, ["stack", "check", str(path)])

    assert result.exit_code == 1, result.output
    assert "'hyperedge_kv'" in result.output
    assert "ADR-0068" in result.output


# -- every path that executes a Pipeline refuses, before writing -------------------------------


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_the_local_runner_raises_a_config_error_before_writing(
    key: str, dirs: tuple[Path, Path]
) -> None:
    pipeline = _pipeline(_with_key(dirs, key))

    with pytest.raises(ConfigError) as excinfo:
        LocalRunner().run(pipeline)

    message = str(excinfo.value)
    assert f"{key!r}" in message
    assert f"Stage {_STAGE_OF[key]!r}" in message
    assert "ADR-0068" in message
    assert not dirs[0].exists(), "the refusal came after something was written to Storage"


def test_the_delta_runner_raises_a_config_error_before_writing(dirs: tuple[Path, Path]) -> None:
    pipeline = _pipeline(_with_key(dirs, "hyperedges", {"max_size": 8}))
    delta = DeltaRunner()

    with pytest.raises(ConfigError, match="'hyperedges'"):
        delta.preview(pipeline)
    with pytest.raises(ConfigError, match="'hyperedges'"):
        delta.apply_delta(pipeline, "delta-1")
    with pytest.raises(ConfigError, match="'hyperedges'"):
        delta.retract(pipeline, "retract-1", [_DOC_ID])
    with pytest.raises(ConfigError, match="'hyperedges'"):
        delta.purge(pipeline, "purge-1", [_DOC_ID])
    assert not dirs[0].exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "STACK"],
        ["process", "--config", "STACK"],
        ["delta", "STACK"],
        ["delta", "STACK", "--dry-run"],
        ["retract", "STACK", "--doc", _DOC_ID],
        ["purge", "STACK", "--doc", _DOC_ID, "--yes"],
    ],
    ids=lambda argv: " ".join(arg for arg in argv if arg != "STACK"),
)
def test_every_command_that_executes_a_pipeline_refuses_a_removed_key(
    argv: list[str], dirs: tuple[Path, Path], tmp_path: Path
) -> None:
    """Exit 2, the key and the ADR named, nothing written.

    ``latence process`` refuses one step earlier than the others: it runs the offline stack check
    before it loads the Pipeline, so its message is the check's finding rather than the loader's.
    """
    path = tmp_path / "stack.yaml"
    path.write_text(yaml.safe_dump(_with_key(dirs, "context_columns")), encoding="utf-8")

    result = cli.invoke(app, [str(path) if arg == "STACK" else arg for arg in argv])

    assert result.exit_code == 2, result.output
    assert ("[config_key]" if argv[0] == "process" else "[config]") in result.output
    assert "'context_columns'" in result.output
    assert "ADR-0068" in result.output
    assert not dirs[0].exists()


def test_the_message_names_the_key_and_stage_but_no_config_value(dirs: tuple[Path, Path]) -> None:
    stack = _with_key(dirs, "bm25", {"provider": "tokenizer.x", "api_key": "sk-do-not-print"})
    with pytest.raises(ConfigError) as excinfo:
        _pipeline(stack).require_runnable()
    message = str(excinfo.value)
    assert "'bm25'" in message and "Stage 'export'" in message
    assert "sk-do-not-print" not in message
    assert "tokenizer.x" not in message


def test_every_removed_key_of_a_pipeline_is_named_in_one_error(dirs: tuple[Path, Path]) -> None:
    stack = _stack(*dirs)
    _stage(stack, "graph")["config"].update({"hyperedges": {}, "graph_features": {}})
    _stage(stack, "export")["config"]["hyperedge_kv"] = True
    pipeline = _pipeline(stack)

    assert pipeline.removed_config_keys() == [
        ("graph", "graph_features"),
        ("graph", "hyperedges"),
        ("export", "hyperedge_kv"),
    ]
    with pytest.raises(ConfigError) as excinfo:
        pipeline.require_runnable()
    message = str(excinfo.value)
    for key in ("graph_features", "hyperedges", "hyperedge_kv"):
        assert f"{key!r}" in message


# -- what the refusal must not do --------------------------------------------------------------


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_a_removed_key_is_not_refused_under_a_stage_kind_that_never_read_it(
    key: str, dirs: tuple[Path, Path]
) -> None:
    """The list is per Stage kind: a Chunk Stage never had these keys, so they are not policed."""
    stack = _stack(*dirs)
    _stage(stack, "chunk")["config"][key] = True
    pipeline = _pipeline(stack)

    assert pipeline.removed_config_keys() == []
    pipeline.require_runnable()
    assert check_stack(pipeline).ok


def test_an_unknown_key_is_still_accepted(dirs: tuple[Path, Path]) -> None:
    """Only the closed list is refused. A key nobody has heard of is not this check's business."""
    stack = _stack(*dirs)
    _stage(stack, "graph")["config"]["a_key_nobody_reads"] = 3
    _stage(stack, "export")["config"]["another_one"] = "x"
    pipeline = _pipeline(stack)

    assert pipeline.removed_config_keys() == []
    assert check_stack(pipeline).ok
    LocalRunner().run(pipeline)


# -- nothing shipped carries one ---------------------------------------------------------------


def _shipped_pipelines() -> list[Path]:
    found = sorted((_REPO_ROOT / "stacks").glob("*.yaml"))
    found += sorted((_REPO_ROOT / "deploy" / "env").glob("*/stack.yaml"))
    found += sorted((_REPO_ROOT / "examples").rglob("*.yaml"))
    return found


def test_there_are_shipped_pipelines_to_check() -> None:
    assert len(_shipped_pipelines()) > 8


@pytest.mark.parametrize(
    "path", _shipped_pipelines(), ids=lambda p: str(p.relative_to(_REPO_ROOT))
)
def test_no_shipped_stack_or_example_carries_a_removed_key(path: Path) -> None:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not (isinstance(loaded, dict) and "stages" in loaded):
        pytest.skip("not a Pipeline declaration")
    assert Pipeline.model_validate(loaded).removed_config_keys() == []
