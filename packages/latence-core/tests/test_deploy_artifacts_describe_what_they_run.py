"""The deploy artifacts' own comments may not claim what the artifacts do not do.

``deploy/docker-compose.yml`` and the Helm chart are read by an operator who is about to run them,
and their comments and install notes are the first description of the result. Two of those
descriptions were false on the tree that introduced this module:

* the compose file, the chart's values and the chart's install notes said a run writes "the RAG
  corpus + KG + Quality Report", while the stack they run ends at the corpus export and has no
  graph Stage;
* the chart's values called the CronJob "incremental re-runs", while every run it starts is
  ``latence run`` with the one run id from ``pipeline.runId`` — which the Runner resumes from its
  checkpoints, so a later run does not read documents added since.

Both checks read the fact from the artifact first (does the stack have a graph Stage, does the pod
run ``delta`` or a fresh run id) and police the wording only while the fact is what it is today.
If a graph Stage or a delta run is added, the wording is free to follow.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from latence_core.pipeline import CapabilityKind, Pipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEPLOY = _REPO_ROOT / "deploy"
_COMPOSE = _DEPLOY / "docker-compose.yml"
_STACK = _DEPLOY / "stacks" / "served-ocr.yaml"
_CHART = _DEPLOY / "helm" / "latence"
_VALUES = _CHART / "values.yaml"
_NOTES = _CHART / "templates" / "NOTES.txt"
_POD = _CHART / "templates" / "_pipeline-pod.tpl"
_CRONJOB = _CHART / "templates" / "pipeline-cronjob.yaml"

# A mention of a knowledge graph that is not the statement that there is none.
_GRAPH_CLAIM = re.compile(r"(?<!no )\b(?:kg|knowledge graph)\b")
# A mention of incremental operation that is not the statement that this is not one.
_INCREMENTAL_CLAIM = re.compile(r"(?<!not an )\bincremental")


def _prose(path: Path) -> str:
    """The file as one lower-cased line of text, comment markers and line breaks removed.

    A claim wraps across comment lines (``writes no`` / ``# knowledge graph``), so the text is
    joined before it is searched.
    """
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        words.extend(word for word in line.split() if word != "#")
    return " ".join(words).lower()


def _claims(pattern: re.Pattern[str], paths: tuple[Path, ...]) -> list[str]:
    found: list[str] = []
    for path in paths:
        text = _prose(path)
        for match in pattern.finditer(text):
            start = max(0, match.start() - 60)
            snippet = text[start : match.end() + 20]
            found.append(f"{path.relative_to(_REPO_ROOT)}: ...{snippet}...")
    return found


def _shipped_stacks() -> list[Pipeline]:
    """The two stacks the deploy artifacts run: the compose-mounted file and the chart's copy."""
    values = yaml.safe_load(_VALUES.read_text(encoding="utf-8"))
    chart_stack = (
        str(values["stack"]["yaml"])
        .replace("{{ .storageUri }}", str(values["stack"]["storageUri"]))
        .replace("{{ .corpusPath }}", str(values["stack"]["corpusPath"]))
        .replace("{{ .ocrBaseUrl }}", "http://ocr.example:8000/v1")
    )
    assert "{{" not in chart_stack, "the chart's stack carries a template field this test does not fill"
    return [
        Pipeline.from_yaml_text(_STACK.read_text(encoding="utf-8")),
        Pipeline.from_yaml_text(chart_stack),
    ]


def test_the_chart_and_the_compose_file_run_the_same_stages() -> None:
    """The two copies of the served-OCR stack agree, so one description fits both."""
    compose_stack, chart_stack = _shipped_stacks()
    assert [(s.name, s.capability, s.provider) for s in compose_stack.stages] == [
        (s.name, s.capability, s.provider) for s in chart_stack.stages
    ]


def test_no_deploy_artifact_promises_a_knowledge_graph_the_stack_does_not_build() -> None:
    builds_a_graph = any(
        stage.capability is CapabilityKind.GRAPH_ASSEMBLY
        for pipeline in _shipped_stacks()
        for stage in pipeline.stages
    )
    if builds_a_graph:
        return  # then saying so is true, and the wording is free to say it
    claims = _claims(_GRAPH_CLAIM, (_COMPOSE, _STACK, _VALUES, _NOTES))
    assert claims == [], (
        "the shipped served-OCR stack has no graph Stage, but these say a run produces a "
        "knowledge graph:\n" + "\n".join(claims)
    )


def test_the_cronjob_is_not_called_incremental_while_it_resumes_one_run_id() -> None:
    pod = _POD.read_text(encoding="utf-8")
    args = pod[pod.index("args:") : pod.index("env:")]
    runs_plain_run = '- "run"' in args and '"delta"' not in args
    one_run_id = ".Values.pipeline.runId" in args
    assert "pipelinePodTemplate" in _CRONJOB.read_text(encoding="utf-8")
    if not (runs_plain_run and one_run_id):
        return  # a delta run, or a run id that changes per run, may be called incremental
    claims = _claims(_INCREMENTAL_CLAIM, (_VALUES, _NOTES, _POD, _CRONJOB))
    assert claims == [], (
        "every scheduled run is `latence run` with the one run id from pipeline.runId, which "
        "resumes the finished run; these call it incremental:\n" + "\n".join(claims)
    )
