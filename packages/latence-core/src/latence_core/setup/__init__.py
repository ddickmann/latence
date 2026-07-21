"""Guided setup (W15, ADR-0041).

``latence setup`` converges to a clean, validated, quality-baked stack config and
``latence process`` runs it. This package holds the deterministic config *generator* + the
dry validation reused by both the interactive and ``--non-interactive`` paths — no prompting
lives here (the CLI owns I/O), so the generator is pure and byte-for-byte testable.
"""

from __future__ import annotations

from latence_core.setup.wizard import (
    FINANCIAL_PII_LABELS,
    CorpusScan,
    SetupAnswers,
    SetupConfigError,
    build_stack_config,
    detect_extensions,
    detect_header_family,
    intake_max_bytes,
    llm_key_env_var,
    normalize_domain_hint,
    recommend_parser,
    recommend_sota_parser,
    scan_source,
    validate_config_text,
)

__all__ = [
    "FINANCIAL_PII_LABELS",
    "CorpusScan",
    "SetupAnswers",
    "SetupConfigError",
    "build_stack_config",
    "detect_extensions",
    "detect_header_family",
    "intake_max_bytes",
    "llm_key_env_var",
    "normalize_domain_hint",
    "recommend_parser",
    "recommend_sota_parser",
    "scan_source",
    "validate_config_text",
]
