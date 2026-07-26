"""C7 — the Stage-contribution seam behind ``latence setup``.

What this locks down, and why each property needed a seam to be testable at all:

* **A Stage's block is assertable directly.** ``contribution.contribute(answers, ctx)`` returns one
  mapping. Before C7 the only way to check one Stage's config was to render the whole YAML, parse it
  back into a Pipeline and search for the Stage by name — which five test modules each
  re-implemented (see ``_setup_stack.py``).
* **The comment header may only claim what the stack actually emits.** ADR-0041 §7a records the
  generator drifting from its own design: the SOTA builder emitted ``redaction.gliner_pii`` while
  the module docstring, the ``FINANCIAL_PII_LABELS`` constant and the lite builder all named
  ``redaction.gliner2``. The header used to re-decide its claims from ``answers``; it now derives
  them from the assembled blocks, so a Stage that stops contributing a block silences the prose
  about it. Both tests below FAIL against the pre-C7 generator (verified by swapping the emitted
  block under the old ``_build_sota_stages`` and watching the header go on advertising the removed
  BM25 signal / the superseded redactor).
* **A mis-ordered or double-declared recipe fails loudly** (ADR-0034: typed errors, never silent) —
  the failure mode a hand-written ordered literal actually has.
* **Determinism (ADR-0041 §3) survives the seam**: the recipe is an ordered literal, so identical
  answers still yield byte-identical YAML.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import latence_core.setup.wizard as wizard
import pytest
from _setup_stack import header_of, stage_names
from latence_core.setup import (
    LITE_RECIPE,
    SOTA_RECIPE,
    RecipeContext,
    RecipeWiringError,
    SetupAnswers,
    StackRecipe,
    StageBlock,
    StageContribution,
    build_stack_config,
    recipe_for,
    validate_config_text,
)
from latence_core.setup.recipe import (
    PARSER_OCR_VLLM,
    PROFILE_LITE,
    PROFILE_SOTA,
    ChunkContribution,
    ExportCorpusContribution,
    ExtractContribution,
    InduceContribution,
    RedactContribution,
    SourceContribution,
)


def _answers(**overrides: Any) -> SetupAnswers:
    base = SetupAnswers(
        source="file:///tmp/corpus",
        storage="file:///tmp/out",
        parser=PARSER_OCR_VLLM,
        profile=PROFILE_SOTA,
        extensions=("pdf",),
    )
    return replace(base, **overrides)


# ---------------------------------------------------------------------------
# the testability win: one block, asserted directly
# ---------------------------------------------------------------------------


def test_a_stage_block_is_assertable_without_rendering_the_whole_config() -> None:
    """The chunk Stage's 640/80 calibration is one call away — no YAML, no Pipeline, no search."""
    block = ChunkContribution().contribute(_answers(), RecipeContext())
    assert block == {
        "name": "chunk",
        "capability": "chunk",
        "provider": "chunk.markdown",
        "depends_on": ["parse"],
        "config": {"max_tokens": 640, "overlap_tokens": 80, "min_tokens": 8},
    }


def test_the_profile_override_is_the_only_difference_between_the_two_extract_blocks() -> None:
    """lite and enterprise-sota extraction differ by exactly their declared overrides."""
    answers = _answers()
    lite = _only(LITE_RECIPE, ExtractContribution).contribute(answers, _ctx_at_labels())
    sota = _only(SOTA_RECIPE, ExtractContribution).contribute(answers, _ctx_at_labels())
    assert lite is not None
    assert sota is not None
    lite_config = _config(lite)
    sota_config = _config(sota)
    assert lite_config["threshold"] == 0.5
    assert sota_config["threshold"] == 0.3  # the SOTA recall floor
    assert sota_config["device"] == "cuda"  # the GPU-pod pin
    assert lite_config["device"] == answers.device  # the operator's resolved device
    assert "max_len" in lite_config
    assert "max_len" not in sota_config
    # Everything else is shared, so a change to fused extraction is a change to ONE declaration.
    assert lite["provider"] == sota["provider"] == "fused_entity_relation.gliner2"


def test_predefined_only_omits_the_induce_stage_and_re_anchors_its_consumers() -> None:
    """No Stage, no Provider, no client, no call — and the label anchor still resolves."""
    ctx = RecipeContext()
    ctx.set_anchor("chunk_stream", "content_screen")
    answers = _answers(predefined_labels={"entity_types": ["hs code"]})
    assert InduceContribution().contribute(answers, ctx) is None
    assert ctx.anchor("labels") == "content_screen"


# ---------------------------------------------------------------------------
# the header derives from the stack — the ADR-0041 7a drift class
# ---------------------------------------------------------------------------


def test_header_stops_claiming_bm25_when_the_stack_stops_emitting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop the BM25 block from ONE contribution; the comment header stops advertising it.

    The pre-C7 renderer keyed this paragraph off ``answers.profile``, so the same edit left the
    header advertising a `bm25-stats.json` artifact the stack no longer produced.
    """
    claim = "bm25-stats.json"
    assert claim in header_of(build_stack_config(_answers()))  # the shipped SOTA recipe does emit it

    dense_only = _with_contribution(
        SOTA_RECIPE, ExportCorpusContribution(gpu_embedder=True, bm25=False)
    )
    monkeypatch.setattr(wizard, "recipe_for", lambda _profile: dense_only)
    text = build_stack_config(_answers())

    assert claim not in header_of(text)
    assert "bm25" not in text
    validate_config_text(text)  # and the stack is still a valid one


def test_header_names_the_redactor_the_stack_actually_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The redactor in the prose is read off the emitted block, never off a module constant.

    This is ADR-0041 §7a made unrepresentable: the header cannot name a Provider the generator did
    not emit, because it has no other source for the name.
    """

    class OtherRedactor(RedactContribution):
        def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock | None:
            block = super().contribute(answers, ctx)
            if block is not None:
                block["provider"] = "redaction.gliner_pii"
            return block

    swapped = _with_contribution(SOTA_RECIPE, OtherRedactor(device="cuda"))
    monkeypatch.setattr(wizard, "recipe_for", lambda _profile: swapped)
    header = header_of(build_stack_config(_answers()))

    assert "`redaction.gliner_pii`" in header
    assert "`redaction.gliner2`" not in header


def test_a_stage_that_stays_out_still_owns_the_prose_about_its_absence() -> None:
    """Predefined-only mode's "there is NO induce stage" paragraph comes from ``induce`` itself."""
    header = header_of(build_stack_config(_answers(predefined_labels={"entity_types": ["a"]})))
    assert "Schema induction: OFF (predefined-only)" in header
    assert "Schema-induction LLM key" not in header  # no induce Stage ⇒ no key to name


def test_the_pii_summary_follows_the_redact_block_not_the_answer() -> None:
    """Both halves of the ONE toggle are described from what the stack carries."""
    on = header_of(build_stack_config(_answers(pii=True)))
    off = header_of(build_stack_config(_answers(pii=False)))
    assert "PII redaction + guardrail screening: ON" in on
    assert "PII redaction + guardrail screening: OFF" in off
    assert "Redactor:" in on
    assert "Redactor:" not in off


# ---------------------------------------------------------------------------
# a mis-written recipe fails loudly (ADR-0034)
# ---------------------------------------------------------------------------


def test_reading_an_anchor_nothing_has_set_is_a_typed_wiring_error() -> None:
    """A contribution ordered before the Stage it reads must not silently emit a broken DAG."""
    with pytest.raises(RecipeWiringError, match="anchor 'labels' was read before"):
        RedactContribution().contribute(_answers(), RecipeContext())


def test_reading_a_fanin_nothing_has_added_to_is_a_typed_wiring_error() -> None:
    with pytest.raises(RecipeWiringError, match="fan-in 'kg' was read before"):
        recipe_for(PROFILE_LITE).contributions[-2].contribute(_answers(), RecipeContext())


def test_two_contributions_claiming_the_same_stage_name_is_a_typed_wiring_error() -> None:
    doubled = StackRecipe(
        profile=PROFILE_LITE, contributions=(SourceContribution(), SourceContribution())
    )
    with pytest.raises(RecipeWiringError, match="both emitted a Stage named 'source'"):
        doubled.assemble(_answers())


def test_a_block_whose_name_disagrees_with_its_declaration_is_rejected() -> None:
    """The assembler will not wire a Stage under a name no other contribution can depend on."""

    class Liar(SourceContribution):
        def contribute(self, answers: SetupAnswers, ctx: RecipeContext) -> StageBlock:
            block = super().contribute(answers, ctx)
            block["name"] = "sauce"
            return block

    with pytest.raises(RecipeWiringError, match="declares name 'source' but emitted"):
        StackRecipe(profile=PROFILE_LITE, contributions=(Liar(),)).assemble(_answers())


# ---------------------------------------------------------------------------
# the recipes describe themselves, and still emit the same stack
# ---------------------------------------------------------------------------


def test_every_contributing_stage_carries_a_one_line_rationale() -> None:
    """The recipe can say what it built and why — no Stage lands in the stack unexplained."""
    stack = SOTA_RECIPE.assemble(_answers())
    assert [name for name, _why in stack.rationales] == [b["name"] for b in stack.stages]
    assert all(why.strip() for _name, why in stack.rationales)


@pytest.mark.parametrize("profile", [PROFILE_LITE, PROFILE_SOTA])
def test_the_recipe_is_an_ordered_literal_so_stage_order_is_deterministic(profile: str) -> None:
    answers = _answers(profile=profile, parser=PARSER_OCR_VLLM)
    first = build_stack_config(answers)
    assert first == build_stack_config(answers)
    declared = [c.name for c in recipe_for(profile).contributions]
    emitted = stage_names(first)
    assert emitted == [name for name in declared if name in set(emitted)]


def test_an_unknown_profile_falls_back_to_lite() -> None:
    """Unchanged from the pre-C7 generator: the CLI validates --profile long before this point."""
    assert recipe_for("no-such-tier") is LITE_RECIPE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _config(block: StageBlock) -> dict[str, object]:
    """A Stage block's ``config`` mapping (a block always carries one where a test asks for it)."""
    config = block["config"]
    assert isinstance(config, dict)
    return dict(config)


def _only(recipe: StackRecipe, kind: type[StageContribution]) -> StageContribution:
    """The single contribution of type ``kind`` in ``recipe``."""
    found = [c for c in recipe.contributions if type(c) is kind]
    assert len(found) == 1, f"{kind.__name__} appears {len(found)}x in {recipe.profile}"
    return found[0]


def _ctx_at_labels() -> RecipeContext:
    """A context wound forward to just after schema induction."""
    ctx = RecipeContext()
    ctx.set_anchor("chunk_stream", "content_screen")
    ctx.set_anchor("labels", "induce")
    return ctx


def _with_contribution(recipe: StackRecipe, replacement: StageContribution) -> StackRecipe:
    """``recipe`` with the contribution owning ``replacement.name`` swapped out, order preserved."""
    swapped = tuple(
        replacement if c.name == replacement.name else c for c in recipe.contributions
    )
    return replace(recipe, contributions=swapped)
