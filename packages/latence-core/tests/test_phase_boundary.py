"""T1 phase-boundary invariant: no corpus-level Stage may run before the chunk-level front half.

The redesign's structural claim is that ``redact`` is the LAST chunk-level stage and the first
corpus-level operation is the resolution phase. These tests hold the claim mechanically:

* the wizard's emitted stacks (both profiles, PII on/off, masked/unmasked, predefined-only) satisfy
  it — so the shipped default cannot regress;
* the PRE-T1 shape (``induce`` at ``granularity: corpus`` feeding ``extract``/``redact``) is
  REJECTED, with a message naming both stages and the dependency chain between them;
* the check is a property of the DAG (not of one topological order), so it fires on a corpus-level
  ancestor however many hops away it is;
* the generator's own dry validation refuses to emit a violating config.

Everything here is static — no run, no providers, no network — so it is the cheap gate that keeps
the architecture honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from latence_core.pipeline import Pipeline
from latence_core.setup import (
    SetupAnswers,
    SetupConfigError,
    build_stack_config,
    validate_config_text,
)
from latence_core.setup.wizard import PARSER_OCR_VLLM, PARSER_UNIVERSAL, PROFILE_LITE, PROFILE_SOTA
from latence_core.stacks.phase_boundary import check_phase_boundary, is_corpus_level


def _src_dir(tmp_path: Path) -> str:
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.pdf").write_text("a", encoding="utf-8")
    (d / "b.txt").write_text("b", encoding="utf-8")
    return f"file://{d}"


def _answers(
    tmp_path: Path,
    *,
    pii: bool = True,
    profile: str = PROFILE_LITE,
    parser: str = PARSER_UNIVERSAL,
    extract_on: str = "unmasked",
    predefined_labels: dict[str, list[str]] | None = None,
) -> SetupAnswers:
    return SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=parser,
        device="auto",
        pii=pii,
        extract_on=extract_on,
        profile=profile,
        extensions=("pdf", "txt"),
        predefined_labels=predefined_labels,
    )


def _flip_induce_to_corpus(config_text: str) -> str:
    """Rewrite a generated stack back into the PRE-T1 shape: corpus-granularity induction.

    This is the exact config ``latence setup`` emitted before T1, reconstructed from the current
    one so the two differ in nothing else. It is what the check must reject.
    """
    doc = yaml.safe_load(config_text)
    for stage in doc["stages"]:
        if stage["name"] == "induce":
            stage["config"]["granularity"] = "corpus"
    return str(yaml.safe_dump(doc, sort_keys=False))


# -- the shipped stacks satisfy the invariant ---------------------------------


@pytest.mark.parametrize("profile", [PROFILE_LITE, PROFILE_SOTA])
@pytest.mark.parametrize("pii", [True, False])
def test_wizard_stacks_keep_the_front_half_chunk_level(
    tmp_path: Path, profile: str, pii: bool
) -> None:
    parser = PARSER_OCR_VLLM if profile == PROFILE_SOTA else PARSER_UNIVERSAL
    text = build_stack_config(_answers(tmp_path, pii=pii, profile=profile, parser=parser))
    assert check_phase_boundary(Pipeline.from_yaml_text(text)) == []


def test_masked_extract_and_predefined_only_also_hold_the_boundary(tmp_path: Path) -> None:
    masked = build_stack_config(_answers(tmp_path, extract_on="masked"))
    assert check_phase_boundary(Pipeline.from_yaml_text(masked)) == []
    fixed = build_stack_config(
        _answers(tmp_path, predefined_labels={"entity_types": ["clause", "obligation"]})
    )
    assert check_phase_boundary(Pipeline.from_yaml_text(fixed)) == []


# -- the pre-T1 shape is rejected ---------------------------------------------


def test_corpus_granularity_induction_before_extraction_is_a_violation(tmp_path: Path) -> None:
    """REGRESSION (T1): the PRE-T1 wizard output — corpus-merged induction feeding the chunk-level
    extractor and redactor — is a phase-boundary violation, named as one.

    Corpus mode collapses every document's labels into ONE salience-capped schema and stamps it on
    every chunk: a corpus-level operation sitting in the middle of the chunk-level front half. That
    is precisely the recall/precision loss T1 removes, so the check must SEE it — not merely prefer
    the new default.
    """
    text = _flip_induce_to_corpus(build_stack_config(_answers(tmp_path)))
    violations = check_phase_boundary(Pipeline.from_yaml_text(text))
    assert violations, "corpus-granularity induction before extract/redact must be rejected"
    offenders = {v.stage for v in violations}
    assert {"extract", "redact"} <= offenders
    assert all(v.corpus_stage == "induce" for v in violations)
    message = violations[0].describe()
    assert "induce" in message and "chunk-level" in message and "corpus-level" in message


def test_setup_dry_validation_refuses_a_phase_boundary_violation(tmp_path: Path) -> None:
    """The generator's own gate: a violating config cannot be emitted, only detected later."""
    good = build_stack_config(_answers(tmp_path))
    validate_config_text(good)  # the shipped shape passes
    with pytest.raises(SetupConfigError, match="phase boundary"):
        validate_config_text(_flip_induce_to_corpus(good))


def test_a_corpus_stage_many_hops_upstream_is_still_caught() -> None:
    """The check reasons over the DAG, so distance does not launder a violation.

    ``redact`` here does not name the corpus-level ``profiling`` stage directly — it reaches it
    through ``extract``. A check that only looked at direct ``depends_on`` (or at one topological
    order) would miss it; this one reports the whole chain.
    """
    pipeline = Pipeline.model_validate(
        {
            "name": "hops",
            "storage_uri": "file:///tmp/x",
            "stages": [
                {"name": "source", "capability": "source", "provider": "source.local_folder"},
                {
                    "name": "parse",
                    "capability": "parse",
                    "provider": "parser.plaintext",
                    "depends_on": ["source"],
                },
                {
                    "name": "profiling",
                    "capability": "profiling",
                    "provider": "profiling.statistical",
                    "depends_on": ["parse"],
                },
                {
                    "name": "extract",
                    "capability": "fused_entity_relation",
                    "provider": "fused_entity_relation.gliner2",
                    "depends_on": ["profiling"],
                },
                {
                    "name": "redact",
                    "capability": "redaction",
                    "provider": "redaction.gliner2",
                    "depends_on": ["extract"],
                },
            ],
        }
    )
    violations = check_phase_boundary(pipeline)
    by_stage = {v.stage: v for v in violations}
    assert set(by_stage) == {"extract", "redact"}
    assert by_stage["redact"].path == ("profiling", "extract", "redact")
    assert by_stage["extract"].path == ("profiling", "extract")


def test_check_is_deterministic_and_order_stable() -> None:
    """Identical input ⇒ identical violation list (message bytes included)."""
    pipeline = Pipeline.model_validate(
        {
            "name": "d",
            "storage_uri": "file:///tmp/x",
            "stages": [
                {"name": "source", "capability": "source", "provider": "source.local_folder"},
                {
                    "name": "graph",
                    "capability": "graph_assembly",
                    "provider": "graph.canonical",
                    "depends_on": ["source"],
                },
                {
                    "name": "profiling",
                    "capability": "profiling",
                    "provider": "profiling.statistical",
                    "depends_on": ["graph"],
                },
                {
                    "name": "redact",
                    "capability": "redaction",
                    "provider": "redaction.gliner2",
                    "depends_on": ["profiling"],
                },
            ],
        }
    )
    first = [v.describe() for v in check_phase_boundary(pipeline)]
    second = [v.describe() for v in check_phase_boundary(pipeline)]
    assert first == second
    # Declaration order, not hash order: graph is declared before profiling.
    assert [v.corpus_stage for v in check_phase_boundary(pipeline)] == ["graph", "profiling"]


# -- the level predicate ------------------------------------------------------


def test_induction_level_follows_its_granularity_not_its_kind() -> None:
    """``schema_induction`` is the one config-dependent kind — chunk/doc vs corpus."""
    stages = {
        "chunk": {"granularity": "chunk"},
        "doc": {"granularity": "doc"},
        "corpus": {"granularity": "CORPUS"},  # case/whitespace tolerant
        "absent": {},
    }
    levels = {}
    for key, config in stages.items():
        pipeline = Pipeline.model_validate(
            {
                "name": "g",
                "storage_uri": "file:///tmp/x",
                "stages": [
                    {
                        "name": "induce",
                        "capability": "schema_induction",
                        "provider": "label_inducer.llm",
                        "config": config,
                    }
                ],
            }
        )
        levels[key] = is_corpus_level(pipeline.stages[0])
    assert levels == {"chunk": False, "doc": False, "corpus": True, "absent": False}


# -- every shipped stack -------------------------------------------------------


def test_every_committed_stack_satisfies_the_phase_boundary() -> None:
    """`latence stack validate` now reports this check, so every stack in `stacks/` must pass it.

    Without this, adding the check to the harness could turn a committed stack red only once CI ran
    the (slow, device-gated) full validation. It is also the guard against the check itself drifting
    into a false positive: if a future kind lands in ``CORPUS_LEVEL_KINDS`` by mistake, the blessed
    stacks say so immediately.
    """
    root = Path(__file__).resolve().parents[3] / "stacks"
    stack_files = sorted(root.glob("*.yaml"))
    assert stack_files, f"no stacks found under {root}"
    offenders: dict[str, str] = {}
    for path in stack_files:
        text = (
            path.read_text(encoding="utf-8")
            .replace("{CORPUS_DIR}", "file:///tmp/corpus")
            .replace("{STORAGE_URI}", "file:///tmp/storage")
        )
        violations = check_phase_boundary(Pipeline.from_yaml_text(text))
        if violations:
            offenders[path.name] = "; ".join(v.describe() for v in violations)
    assert offenders == {}
