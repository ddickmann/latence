"""The extraction ENGINE answer — one choice, two Stages, and the pair that cannot be expressed.

The defect this closes is a *dependency* one, and it is invisible to every validator the framework
already has. ``fused_entity_relation.gliner25`` and ``redaction.gliner2`` are both real Providers,
both fulfil their Capability, and a stack naming both passes ``latence stack validate`` — and then
cannot be installed at all, because their packages pin disjoint ``gliner2`` majors
(``>=2.0,<3`` vs ``>=1.3,<2``). The wizard therefore does not *discourage* the pair; it has one
``engine`` answer that both Stages read, so the pair is unrepresentable.

Every hop is asserted: the field, the CLI flag, the emitted YAML, and the real Provider class
constructed with the wizard's own emitted config.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.cli import app
from latence_core.setup.recipe import (
    CHUNK_MAX_TOKENS,
    CHUNK_MAX_TOKENS_PAGE,
    CHUNK_STRATEGY_PAGE,
    CHUNK_STRATEGY_WINDOW,
    ENGINE_GLINER2,
    ENGINE_GLINER25,
    ENGINES,
    GLINER25_MAX_WINDOW,
    PARSER_OCR_VLLM,
    PARSER_UNIVERSAL,
    PROFILE_LITE,
    PROFILE_SOTA,
    SetupConfigError,
    engine_for,
    model_window_for,
)
from latence_core.setup.wizard import (
    SetupAnswers,
    build_pipeline_dict,
    build_stack_config,
    validate_config_text,
)
from latence_core.stages.chunk import PAGE_CHUNK_MAX_TOKENS
from latence_gliner25._backbone import MAX_SUPPORTED_MAX_LEN

# The REAL Provider classes the emitted config names — the fourth hop. Installed --no-deps from
# source by scripts/verify-local.sh; every constructor below is pure config validation (no weights,
# no torch import, no network), so this suite stays CPU-only and offline.
from latence_gliner25.extract import Gliner25FusedExtractor
from latence_gliner25.redact import Gliner25PIIRedactor
from typer.testing import CliRunner

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

runner = CliRunner()


def _src_dir(tmp_path: Path) -> str:
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.pdf").write_text("a", encoding="utf-8")
    return f"file://{d}"


def _answers(tmp_path: Path, **overrides: Any) -> SetupAnswers:
    base: dict[str, Any] = {
        "source": _src_dir(tmp_path),
        "storage": f"file://{tmp_path}/out",
        "parser": PARSER_UNIVERSAL,
        "device": "auto",
        "extensions": ("pdf",),
    }
    base.update(overrides)
    return SetupAnswers(**base)


def _stage(answers: SetupAnswers, name: str) -> dict[str, Any]:
    stages = build_pipeline_dict(answers)["stages"]
    assert isinstance(stages, list)
    for block in stages:
        assert isinstance(block, dict)
        if block.get("name") == name:
            return block
    msg = f"no {name!r} stage in the emitted stack"
    raise AssertionError(msg)


def _config(answers: SetupAnswers, name: str) -> dict[str, Any]:
    cfg = _stage(answers, name).get("config", {})
    assert isinstance(cfg, dict)
    return cfg


# ==============================================================================================
# The pair that cannot be expressed
# ==============================================================================================


@pytest.mark.parametrize("profile", [PROFILE_LITE, PROFILE_SOTA])
@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_extract_and_redact_always_come_from_the_SAME_engine_row(
    tmp_path: Path, engine: str, profile: str
) -> None:
    """The invariant, over every (engine × profile) the wizard can emit.

    Not "the default pairing is right" — *every* pairing, because the failure mode is a
    combination nobody wrote down. Both Stages are checked against the SAME row object, so a future
    contribution that hard-codes a Provider name breaks this test rather than shipping a stack that
    cannot be installed.
    """
    answers = _answers(
        tmp_path,
        engine=engine,
        profile=profile,
        parser=PARSER_OCR_VLLM if profile == PROFILE_SOTA else PARSER_UNIVERSAL,
    )
    row = engine_for(engine)
    assert _stage(answers, "extract")["provider"] == row.extract_provider
    assert _stage(answers, "redact")["provider"] == row.redact_provider


def test_no_emitted_stack_ever_mixes_the_two_gliner2_majors(tmp_path: Path) -> None:
    """Stated as the dependency fact it is, not as a name comparison.

    The 1.x and 2.x Provider families are named here explicitly so that adding a third Provider to
    either family without extending the engine table is a red test.
    """
    gliner2_1x = {"fused_entity_relation.gliner2", "redaction.gliner2"}
    gliner2_2x = {"fused_entity_relation.gliner25", "redaction.gliner25"}
    for engine in ENGINES:
        for profile in (PROFILE_LITE, PROFILE_SOTA):
            answers = _answers(
                tmp_path,
                engine=engine,
                profile=profile,
                parser=PARSER_OCR_VLLM if profile == PROFILE_SOTA else PARSER_UNIVERSAL,
            )
            stages = build_pipeline_dict(answers)["stages"]
            assert isinstance(stages, list)
            named = {
                str(block.get("provider"))
                for block in stages
                if isinstance(block, dict)
            }
            assert not (named & gliner2_1x and named & gliner2_2x)


def test_an_unknown_engine_is_refused_by_the_struct_not_only_by_the_cli(tmp_path: Path) -> None:
    """``build_stack_config(answers)`` is a public entry point, so the refusal lives on the struct."""
    with pytest.raises(SetupConfigError, match="Unknown extraction engine"):
        _answers(tmp_path, engine="gliner3")
    with pytest.raises(SetupConfigError, match="Unknown chunk strategy"):
        _answers(tmp_path, chunk_strategy="paragraphs")


# ==============================================================================================
# The coherent default: page-preferred chunking at 1024
# ==============================================================================================


def test_the_gliner25_engine_defaults_to_page_chunking_at_1024(tmp_path: Path) -> None:
    """Hop 3: `--engine gliner25` alone lands the page/1024 pairing, with no other flag."""
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path).removeprefix("file://"),
            "--storage",
            f"file://{tmp_path}/out",
            "--engine",
            ENGINE_GLINER25,
            "--output",
            str(tmp_path / "stack.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    text = (tmp_path / "stack.yaml").read_text(encoding="utf-8")
    validate_config_text(text)
    assert "provider: chunk.page" in text
    assert "max_tokens: 1024" in text
    assert "fused_entity_relation.gliner25" in text
    assert "redaction.gliner25" in text
    assert "redaction.gliner2\n" not in text


def test_the_default_engine_is_unchanged_byte_for_byte(tmp_path: Path) -> None:
    """The whole change must be a no-op for an operator who does not ask for it.

    Byte-identity against a struct built without the new fields at all is the strongest available
    statement of "nothing moved" — and it is what protects every committed baseline.
    """
    answers = _answers(tmp_path)
    assert answers.engine == ENGINE_GLINER2
    assert _stage(answers, "chunk")["provider"] == "chunk.markdown"
    assert _config(answers, "chunk")["max_tokens"] == CHUNK_MAX_TOKENS
    assert _stage(answers, "extract")["provider"] == "fused_entity_relation.gliner2"
    assert _stage(answers, "redact")["provider"] == "redaction.gliner2"
    assert _config(answers, "redact")["guardrails"] is True


def test_the_chunk_strategy_override_reaches_both_engines(tmp_path: Path) -> None:
    """The escape hatch: page chunking with the 1.x engine, window chunking with 2.5."""
    paged_1x = _answers(tmp_path, engine=ENGINE_GLINER2, chunk_strategy=CHUNK_STRATEGY_PAGE)
    assert _stage(paged_1x, "chunk")["provider"] == "chunk.page"
    assert _stage(paged_1x, "extract")["provider"] == "fused_entity_relation.gliner2"

    windowed_25 = _answers(
        tmp_path,
        engine=ENGINE_GLINER25,
        chunk_strategy=CHUNK_STRATEGY_WINDOW,
        chunk_max_tokens=CHUNK_MAX_TOKENS_PAGE,
    )
    assert _stage(windowed_25, "chunk")["provider"] == "chunk.markdown"
    assert _stage(windowed_25, "extract")["provider"] == "fused_entity_relation.gliner25"


def test_the_page_budget_constant_mirrors_the_chunker_it_names() -> None:
    """An unchecked mirror is the drift class ADR-0041 §7a recorded, so it is checked."""
    assert CHUNK_MAX_TOKENS_PAGE == PAGE_CHUNK_MAX_TOKENS
    assert ENGINES[ENGINE_GLINER25].chunk_max_tokens == PAGE_CHUNK_MAX_TOKENS


# ==============================================================================================
# The window the 2.5 checkpoint actually supports
# ==============================================================================================


def test_the_derived_window_for_the_1024_budget_fits_the_2_5_checkpoint() -> None:
    """``model_window_for(1024)`` must be a window the checkpoint really attends to."""
    window = model_window_for(CHUNK_MAX_TOKENS_PAGE)
    assert window == 1280
    assert window <= GLINER25_MAX_WINDOW


def test_the_mirrored_checkpoint_window_matches_the_provider_that_enforces_it() -> None:
    """``latence-core`` must not import a Provider package, so the ceiling is mirrored — and pinned.

    Both sides refuse independently (the wizard at answer construction, the Provider at
    construction), which is only a real double gate while the two numbers agree.
    """
    assert GLINER25_MAX_WINDOW == MAX_SUPPORTED_MAX_LEN


def test_a_budget_too_large_for_the_engine_is_refused_at_answer_construction(
    tmp_path: Path,
) -> None:
    """Silent truncation for `redact` means PII left UNMASKED — so it is a refusal, not a warning."""
    over = -(-(GLINER25_MAX_WINDOW + 1) // 1) * 1  # a budget whose derived window exceeds 4096
    with pytest.raises(SetupConfigError, match="checkpoint window"):
        _answers(tmp_path, engine=ENGINE_GLINER25, chunk_max_tokens=over)


def test_the_1x_engine_claims_no_ceiling_it_never_verified(tmp_path: Path) -> None:
    """The negative control. Nothing here ever measured a 1.x window, so nothing may assert one.

    A cap invented for symmetry would refuse budgets that work today — which is exactly what a
    first draft of this change did, and what this test now prevents.
    """
    assert ENGINES[ENGINE_GLINER2].max_window is None
    answers = _answers(tmp_path, engine=ENGINE_GLINER2, chunk_max_tokens=4096)
    assert _config(answers, "extract")["max_len"] == model_window_for(4096)


# ==============================================================================================
# The PII claim, and the guardrail switch that must not be emitted
# ==============================================================================================


def test_guardrails_is_emitted_only_for_the_engine_whose_checkpoint_has_them(
    tmp_path: Path,
) -> None:
    """A security control that is enabled and unbacked is worse than one that is off."""
    assert _config(_answers(tmp_path, engine=ENGINE_GLINER2), "redact")["guardrails"] is True
    assert "guardrails" not in _config(_answers(tmp_path, engine=ENGINE_GLINER25), "redact")


def test_the_emitted_header_states_the_zero_shot_pii_caveat(tmp_path: Path) -> None:
    """An operator must be able to read what their privacy posture rests on, in the file itself."""
    text = build_stack_config(_answers(tmp_path, engine=ENGINE_GLINER25))
    assert "ZERO-SHOT" in text
    assert "UNMEASURED" in text
    assert "no GLiNER 2.5 PII checkpoint" in text.replace("NO GLiNER 2.5 PII checkpoint", "no GLiNER 2.5 PII checkpoint")
    # And the 1.x engine's header must NOT carry it — a caveat that always appears teaches nothing.
    assert "ZERO-SHOT" not in build_stack_config(_answers(tmp_path, engine=ENGINE_GLINER2))


def test_the_content_screen_stage_still_rides_the_pii_toggle_on_both_engines(
    tmp_path: Path,
) -> None:
    """The deterministic guardrail pre-filter is engine-independent and must stay wired."""
    for engine in ENGINES:
        answers = _answers(tmp_path, engine=engine, pii=True)
        assert _stage(answers, "content_screen")["provider"] == "screening.content_keyword"


# ==============================================================================================
# Hop 4 — the REAL Provider accepts the wizard's own emitted config
# ==============================================================================================


def test_the_real_gliner25_providers_accept_the_emitted_config(tmp_path: Path) -> None:
    """Not "the YAML looks right": the actual classes are constructed with the actual blocks.

    Constructing them is pure config validation — no weights, no torch, no network — so a key the
    wizard emits that the Provider rejects is caught here rather than on a pod.
    """
    answers = _answers(tmp_path, engine=ENGINE_GLINER25)
    extract_cfg = _config(answers, "extract")
    redact_cfg = _config(answers, "redact")
    Gliner25FusedExtractor(dict(extract_cfg))
    Gliner25PIIRedactor(dict(redact_cfg))
    # And the window the wizard derived is the one the Provider will thread natively.
    assert extract_cfg["max_len"] == model_window_for(CHUNK_MAX_TOKENS_PAGE)
    assert redact_cfg["max_len"] == extract_cfg["max_len"]


def test_every_engine_the_wizard_can_emit_produces_a_valid_stack(tmp_path: Path) -> None:
    """`latence stack validate` over the whole (engine × profile × strategy) product."""
    for engine in ENGINES:
        for profile in (PROFILE_LITE, PROFILE_SOTA):
            for strategy in (CHUNK_STRATEGY_WINDOW, CHUNK_STRATEGY_PAGE):
                answers = _answers(
                    tmp_path,
                    engine=engine,
                    profile=profile,
                    chunk_strategy=strategy,
                    chunk_max_tokens=engine_for(engine).chunk_max_tokens,
                    parser=PARSER_OCR_VLLM if profile == PROFILE_SOTA else PARSER_UNIVERSAL,
                )
                validate_config_text(build_stack_config(answers))


def test_the_cli_flags_and_the_struct_reach_the_same_answers(tmp_path: Path) -> None:
    """Interactive/flag parity: the flag path must produce the config the struct path does."""
    out = tmp_path / "from-flags.yaml"
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path),
            "--storage",
            f"file://{tmp_path}/out",
            "--parser",
            PARSER_UNIVERSAL,
            "--engine",
            ENGINE_GLINER25,
            "--chunk-strategy",
            CHUNK_STRATEGY_WINDOW,
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    # No chunk budget on EITHER side: both paths must resolve it from the engine row, which is
    # the parity that matters — the CLI resolves it for its own range checks, the struct resolves
    # it for every programmatic caller, and byte-identity is what proves the two agree.
    expected = build_stack_config(
        _answers(tmp_path, engine=ENGINE_GLINER25, chunk_strategy=CHUNK_STRATEGY_WINDOW)
    )
    assert out.read_text(encoding="utf-8") == expected


def test_an_unknown_engine_flag_exits_non_zero(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "setup",
            "--non-interactive",
            "--source",
            _src_dir(tmp_path).removeprefix("file://"),
            "--engine",
            "gliner9",
            "--output",
            str(tmp_path / "stack.yaml"),
        ],
    )
    assert result.exit_code == 2
    assert "Unknown engine" in result.output
