"""Guided setup (W15, ADR-0041).

``latence setup`` converges to a clean, validated, quality-baked stack config and
``latence process`` runs it. This package holds the deterministic config *generator* + the
dry validation reused by both the interactive and ``--non-interactive`` paths — no prompting
lives here (the CLI owns I/O), so the generator is pure and byte-for-byte testable.

Two modules:

* :mod:`latence_core.setup.wizard` — the operator-facing entry point. Reads the corpus
  (:func:`scan_source`, :func:`recommend_parser`, :func:`detect_header_family`), assembles the
  profile's recipe, renders the file, and dry-validates it. ``build_stack_config(answers) -> str``
  is the whole interface a caller needs.
* :mod:`latence_core.setup.recipe` — the Stage-contribution seam (C7). Each Stage declares the
  config block it contributes and the comment prose describing it; a PROFILE is an ordered literal
  of those declarations. In-core and profile-scoped by design (ADR-0041 constraint 1: "opinionated,
  not a blank canvas") — not a third-party registry.
"""

from __future__ import annotations

from latence_core.setup.recipe import (
    LITE_RECIPE,
    SOTA_RECIPE,
    AssembledStack,
    RecipeContext,
    RecipeWiringError,
    StackRecipe,
    StageBlock,
    StageContribution,
    recipe_for,
)
from latence_core.setup.wizard import (
    FINANCIAL_PII_LABELS,
    PREDEFINED_LABEL_SLOTS,
    CorpusScan,
    SetupAnswers,
    SetupConfigError,
    build_stack_config,
    detect_extensions,
    detect_header_family,
    detect_inline_family,
    intake_max_bytes,
    llm_key_env_var,
    normalize_domain_hint,
    parse_predefined_labels,
    recommend_parser,
    recommend_sota_parser,
    scan_source,
    validate_config_text,
)

__all__ = [
    "FINANCIAL_PII_LABELS",
    "LITE_RECIPE",
    "PREDEFINED_LABEL_SLOTS",
    "SOTA_RECIPE",
    "AssembledStack",
    "CorpusScan",
    "RecipeContext",
    "RecipeWiringError",
    "SetupAnswers",
    "SetupConfigError",
    "StackRecipe",
    "StageBlock",
    "StageContribution",
    "build_stack_config",
    "detect_extensions",
    "detect_header_family",
    "detect_inline_family",
    "intake_max_bytes",
    "llm_key_env_var",
    "normalize_domain_hint",
    "parse_predefined_labels",
    "recipe_for",
    "recommend_parser",
    "recommend_sota_parser",
    "scan_source",
    "validate_config_text",
]
