"""Shared helpers for the guided-setup tests — ONE place that turns emitted YAML into Stages.

Five test modules used to re-implement the same ``_stage(text, name)`` one-liner (render the whole
config, parse it back into a :class:`~latence_core.pipeline.Pipeline`, find a Stage by name). That
duplication is a symptom, not a style problem: before the C7 Stage-contribution seam the ONLY way to
assert anything about one Stage's config was to render and re-parse the entire file.

Since C7 the preferred assertion is a direct one — ``contribution.contribute(answers, ctx)`` returns
exactly one block (see ``test_setup_recipe.py``). These helpers remain for the tests that are
genuinely about the whole emitted DOCUMENT: determinism, dry validation, the DAG, the comment
header. Not a pytest plugin and not in ``conftest.py`` — a plain module the setup tests import, so
it stays out of every other package's fixture namespace.
"""

from __future__ import annotations

from latence_core.pipeline import Pipeline, Stage

__all__ = ["header_of", "stage", "stage_names", "stages_by_name", "try_stage"]


def stages_by_name(text: str) -> dict[str, Stage]:
    """Every Stage in the emitted config, keyed by name (the DAG as a lookup)."""
    return {s.name: s for s in Pipeline.from_yaml_text(text).stages}


def stage(text: str, name: str) -> Stage:
    """The Stage called ``name`` — raises ``KeyError`` when the config does not carry it."""
    return stages_by_name(text)[name]


def try_stage(text: str, name: str) -> Stage | None:
    """The Stage called ``name``, or ``None`` — for asserting a Stage is ABSENT."""
    return stages_by_name(text).get(name)


def stage_names(text: str) -> list[str]:
    """Stage names in emitted order (order is part of the determinism contract)."""
    return [s.name for s in Pipeline.from_yaml_text(text).stages]


def header_of(text: str) -> str:
    """The generated file's comment header — everything above the pipeline mapping."""
    return text.split("\nname:")[0]
