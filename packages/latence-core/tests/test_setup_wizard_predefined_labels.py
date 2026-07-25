"""T1 predefined-only mode: a fixed operator schema, and the `induce` Stage is GONE.

The escape hatch for a corpus with a KNOWN taxonomy (compliance codes, a fixed entity set), where
schema discovery is not merely unnecessary but undesirable. The contract these tests hold:

* the emitted stack contains **no `induce` Stage at all** — not a disabled one, not a
  short-circuited Provider. Absence is the only implementation of "no LLM call" that cannot quietly
  regress into one, and it is what the ZERO-induction-calls assertion actually measures;
* every Stage that used to read the induced stream is rewired to the Stage that fed induction, so
  the DAG is intact and the run still validates;
* extraction runs against the operator's labels;
* **the ADR-0044 financial-PII floor is NOT subsettable.** A fixed *relevance* schema is the
  operator's call; the *safety* schema is not. A leaked card number in a chunk the operator
  considers irrelevant must still be masked (README invariant 3);
* the fixed schema is parsed loudly — a typo'd slot raises rather than silently falling back to the
  profile defaults, which would ship a stack that LOOKS operator-configured and is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from latence_core.cli import app
from latence_core.pipeline import Pipeline, Stage
from latence_core.setup import (
    FINANCIAL_PII_LABELS,
    SetupAnswers,
    SetupConfigError,
    build_stack_config,
    parse_predefined_labels,
    validate_config_text,
)
from latence_core.setup.wizard import (
    PARSER_OCR_VLLM,
    PARSER_UNIVERSAL,
    PROFILE_LITE,
    PROFILE_SOTA,
)
from typer.testing import CliRunner

runner = CliRunner()

_FIXED: dict[str, list[str]] = {
    "entity_types": ["tariff heading", "commodity"],
    "relation_types": ["classified under"],
    "pii_types": ["customer reference"],
}


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
    predefined_labels: dict[str, list[str]] | None = None,
) -> SetupAnswers:
    return SetupAnswers(
        source=_src_dir(tmp_path),
        storage=f"file://{tmp_path}/out",
        parser=parser,
        device="auto",
        pii=pii,
        profile=profile,
        extensions=("pdf", "txt"),
        predefined_labels=predefined_labels,
    )


def _stages(text: str) -> dict[str, Stage]:
    return {s.name: s for s in Pipeline.from_yaml_text(text).stages}


# -- the induce Stage is absent, and the DAG survives its absence -------------


@pytest.mark.parametrize(
    ("profile", "parser"), [(PROFILE_LITE, PARSER_UNIVERSAL), (PROFILE_SOTA, PARSER_OCR_VLLM)]
)
@pytest.mark.parametrize("pii", [True, False])
def test_predefined_only_emits_no_induce_stage_and_still_validates(
    tmp_path: Path, profile: str, parser: str, pii: bool
) -> None:
    """REGRESSION (T1): with a fixed schema there is NO `induce` Stage — so ZERO induction calls.

    Before T1 the only way to run without induction was to hand-delete the Stage and re-wire every
    dependent by hand (the wizard always emitted it). Asserting the Stage's ABSENCE is what makes
    "no LLM call" a structural fact rather than a Provider promise.
    """
    text = build_stack_config(
        _answers(tmp_path, pii=pii, profile=profile, parser=parser, predefined_labels=_FIXED)
    )
    validate_config_text(text)  # raises on a broken DAG / capability wiring / phase boundary
    stages = _stages(text)
    assert "induce" not in stages
    assert not any(s.capability == "schema_induction" for s in stages.values())
    assert not any("induce" in s.depends_on for s in stages.values())
    # Everything that used to hang off `induce` now hangs off the Stage that FED induction, so the
    # front half is unbroken: with PII on that is `content_screen`, else `chunk`.
    upstream = "content_screen" if pii else "chunk"
    assert stages["extract"].depends_on == [upstream]
    if pii:
        assert stages["redact"].depends_on == [upstream]


def test_induction_is_on_by_default_and_off_only_by_choice(tmp_path: Path) -> None:
    """The default is still per-chunk induction — predefined-only is opt-in, never implicit."""
    default_stages = _stages(build_stack_config(_answers(tmp_path)))
    assert default_stages["induce"].config["granularity"] == "chunk"


# -- the operator's labels reach extraction ----------------------------------


@pytest.mark.parametrize(
    ("profile", "parser"), [(PROFILE_LITE, PARSER_UNIVERSAL), (PROFILE_SOTA, PARSER_OCR_VLLM)]
)
def test_extraction_runs_against_exactly_the_operator_labels(
    tmp_path: Path, profile: str, parser: str
) -> None:
    text = build_stack_config(
        _answers(tmp_path, profile=profile, parser=parser, predefined_labels=_FIXED)
    )
    extract = _stages(text)["extract"]
    # The operator's own order, not the profile default and not sorted behind their back.
    assert extract.config["labels"] == ["tariff heading", "commodity"]
    assert extract.config["relation_labels"] == ["classified under"]


def test_a_partial_schema_keeps_the_profile_default_for_the_unspecified_slot(
    tmp_path: Path,
) -> None:
    """An operator who fixes only the entity types still gets a working relation extractor."""
    text = build_stack_config(
        _answers(tmp_path, predefined_labels={"entity_types": ["clause"]})
    )
    extract = _stages(text)["extract"]
    assert extract.config["labels"] == ["clause"]
    assert extract.config["relation_labels"] == ["works for", "partner of"]


# -- the safety floor is universal, never subsetted ---------------------------


@pytest.mark.parametrize(
    ("profile", "parser"), [(PROFILE_LITE, PARSER_UNIVERSAL), (PROFILE_SOTA, PARSER_OCR_VLLM)]
)
def test_the_financial_pii_floor_survives_predefined_only(
    tmp_path: Path, profile: str, parser: str
) -> None:
    """REGRESSION: a fixed RELEVANCE schema must not shrink the SAFETY schema (README invariant 3).

    The naive implementation of "the operator's labels ARE the schema" replaces `pii_labels` with
    whatever they supplied — and a corpus whose taxonomy says nothing about credit cards then stops
    masking credit cards. The floor is a security invariant, not a relevance label.
    """
    text = build_stack_config(
        _answers(tmp_path, profile=profile, parser=parser, predefined_labels=_FIXED)
    )
    pii_labels = _stages(text)["redact"].config["pii_labels"]
    assert set(FINANCIAL_PII_LABELS) <= set(pii_labels)
    assert "customer reference" in pii_labels  # ...and the operator's own PII type rides on top
    # The floor comes FIRST and keeps its canonical order — a stable, diffable config.
    assert pii_labels[: len(FINANCIAL_PII_LABELS)] == list(FINANCIAL_PII_LABELS)


def test_an_operator_pii_type_that_duplicates_the_floor_is_not_repeated(tmp_path: Path) -> None:
    text = build_stack_config(
        _answers(tmp_path, predefined_labels={"entity_types": ["x"], "pii_types": ["iban"]})
    )
    pii_labels = _stages(text)["redact"].config["pii_labels"]
    assert pii_labels.count("iban") == 1


# -- the schema is parsed loudly ---------------------------------------------


def test_parse_predefined_labels_accepts_a_well_formed_schema() -> None:
    assert parse_predefined_labels(dict(_FIXED)) == _FIXED


def test_parse_predefined_labels_dedupes_and_keeps_operator_order() -> None:
    parsed = parse_predefined_labels({"entity_types": [" clause ", "clause", "party", ""]})
    assert parsed == {"entity_types": ["clause", "party"]}


@pytest.mark.parametrize(
    "bad",
    [
        ["entity_types"],  # not a mapping
        {"entities": ["a"]},  # typo'd slot — the silent-default trap
        {"entity_types": "clause"},  # not a list
        {"entity_types": [1, 2]},  # not strings
        {"entity_types": []},  # declares nothing usable
        {"entity_types": ["  "]},  # ...nor after stripping
    ],
)
def test_parse_predefined_labels_rejects_malformed_input(bad: object) -> None:
    with pytest.raises(SetupConfigError):
        parse_predefined_labels(bad)


# -- the CLI surface ----------------------------------------------------------


def test_cli_predefined_labels_file_drives_the_generated_stack(tmp_path: Path) -> None:
    """`latence setup --predefined-labels <file>` is the operator-facing switch, and it is
    deterministic (two identical invocations produce byte-identical configs)."""
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_FIXED), encoding="utf-8")
    out1, out2 = tmp_path / "a.yaml", tmp_path / "b.yaml"
    args = [
        "setup",
        "--non-interactive",
        "--source",
        _src_dir(tmp_path),
        "--storage",
        f"file://{tmp_path}/out",
        "--predefined-labels",
        str(schema),
        "--output",
    ]
    r1 = runner.invoke(app, [*args, str(out1)])
    r2 = runner.invoke(app, [*args, str(out2)])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    assert out1.read_bytes() == out2.read_bytes()
    stages = _stages(out1.read_text(encoding="utf-8"))
    assert "induce" not in stages
    assert stages["extract"].config["labels"] == ["tariff heading", "commodity"]


def test_cli_rejects_a_malformed_predefined_labels_file(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"entities": ["oops"]}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--predefined-labels",
            str(schema),
            "--output",
            str(tmp_path / "out.yaml"),
        ],
    )
    assert result.exit_code == 2
    assert "unknown slot" in result.output


def test_cli_rejects_a_missing_predefined_labels_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--predefined-labels",
            str(tmp_path / "nope.json"),
            "--output",
            str(tmp_path / "out.yaml"),
        ],
    )
    assert result.exit_code == 2
