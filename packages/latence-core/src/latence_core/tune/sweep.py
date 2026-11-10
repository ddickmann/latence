"""The per-Provider knob grid the tuner sweeps — cartesian-but-pruned, torch-free (W11).

Each heavy/GPU Provider exposes a small set of perf knobs the autotuner turns per device
(dtype/quantize/compile/batch_size/ocr_batch_size/max_concurrency). This module declares, per
Provider *family*, WHICH knobs apply and their candidate values, and expands them into the ordered
list of :class:`~latence_core.tune.report.KnobSetting` candidates the tuner probes — the BASELINE
(fp32/plain, no compile) always first (it is the correctness reference every other candidate is
compared against), then the accelerated candidates in a **stable, deterministic order** so a seeded
sweep is byte-reproducible.

Pure + torch-free (ADR-0016): this is declarative data + list expansion, no model load, no torch.
The knob names + defaults mirror the W7/W8 seam exactly (``dtype``, ``compile``, ``batch_size``,
``ocr_batch_size``, ``max_concurrency``, ``quantize``) so a winning :class:`KnobSetting` overlays
straight onto a stage config with no translation.
"""

from __future__ import annotations

from latence_core.providers.perf import DTYPE_BF16, DTYPE_FP32
from latence_core.tune.report import KnobSetting

# The perf-knob *dimensions* a Provider family may expose. Each maps a config knob name to its
# ordered candidate values; the BASELINE value of each is listed FIRST (so the cartesian product's
# first element is the all-baseline setting — the correctness reference). A Provider that does not
# list a dimension simply never sweeps it.
#
# The families:
#   * "encoder"      — the learned GLiNER-family encoders (entity.gliner, redaction.gliner_pii,
#                      relation.gliner_relex): dtype + compile + batch_size (W7/W8).
#   * "encoder_quant"— gliner2 (redaction.gliner2): also a ``quantize`` knob (the pod compile-crash
#                      lesson lived here — gliner2-multi-v1 + torch 2.4.1).
#   * "embedding"    — embedding.sentence_transformers: dtype + compile + batch_size.
#   * "ocr"          — the generative OCR VLMs (parser.lighton, parser.lighton_vllm): dtype +
#                      compile + ocr_batch_size (W8 page-batching).
#   * "endpoint"     — endpoint/LLM Providers: max_concurrency only (W6) — no dtype/compile (the
#                      model runs server-side).
_DTYPE_VALUES = [DTYPE_FP32, DTYPE_BF16]  # fp32 first = the baseline/correctness reference.
_COMPILE_VALUES = [False, True]  # off first = the baseline (the pod compile-crash lives on `True`).

# Batch sizes are throughput-only knobs whose OUTPUT is byte-identical to the baseline (W7 proved
# batched == unbatched record-for-record); the smallest (1) is the reference, larger values are the
# candidates. Kept short so the sweep stays a *small* real sample, not a benchmark suite.
_BATCH_VALUES = [1, 8, 16]
_OCR_BATCH_VALUES = [1, 4]
_CONCURRENCY_VALUES = [1, 8]
_QUANTIZE_VALUES = [False, True]

# Provider name → knob-dimension dict (config-knob → ordered candidate values, baseline first).
_FAMILY_KNOBS: dict[str, dict[str, list[object]]] = {
    "encoder": {"dtype": list(_DTYPE_VALUES), "compile": list(_COMPILE_VALUES),
                "batch_size": list(_BATCH_VALUES)},
    "encoder_quant": {"dtype": list(_DTYPE_VALUES), "quantize": list(_QUANTIZE_VALUES),
                      "compile": list(_COMPILE_VALUES), "batch_size": list(_BATCH_VALUES)},
    "embedding": {"dtype": list(_DTYPE_VALUES), "compile": list(_COMPILE_VALUES),
                  "batch_size": list(_BATCH_VALUES)},
    "ocr": {"dtype": list(_DTYPE_VALUES), "compile": list(_COMPILE_VALUES),
            "ocr_batch_size": list(_OCR_BATCH_VALUES)},
    "endpoint": {"max_concurrency": list(_CONCURRENCY_VALUES)},
}

# The heavy/GPU Providers the tuner sweeps, mapped to their knob family. The spine of the roster the
# spec names (parser.lighton, fused_entity_relation.gliner2, redaction.gliner2,
# embedding.sentence_transformers, parser.lighton_vllm) plus the sibling GLiNER encoders they share
# a
# family with. A Provider NOT in this map is not swept (a CPU-deterministic Provider has no knobs).
_PROVIDER_FAMILY: dict[str, str] = {
    "entity.gliner": "encoder",
    "redaction.gliner_pii": "encoder",
    "relation.gliner_relex": "encoder",
    "fused_entity_relation.gliner2": "encoder_quant",
    "redaction.gliner2": "encoder_quant",
    "embedding.sentence_transformers": "embedding",
    "parser.lighton": "ocr",
    "parser.glm": "ocr",
    "parser.lighton_vllm": "ocr",
    "embedding.endpoint": "endpoint",
    "parser.endpoint": "endpoint",
    "entity.endpoint": "endpoint",
}


def is_tunable_provider(provider: str) -> bool:
    """Whether ``provider`` has perf knobs the tuner sweeps (a heavy/GPU/endpoint Provider)."""
    return provider in _PROVIDER_FAMILY


def _cartesian(dims: dict[str, list[object]]) -> list[dict[str, object]]:
    """Expand a knob-dimension dict into the ordered list of knob-value dicts (baseline first).

    Deterministic: dimensions are iterated in their declared order, values in their declared order
    (baseline first), so the first element is the all-baseline setting and the whole list is
    byte-reproducible. Empty ``dims`` yields a single empty setting (the baseline is "no knobs").
    """
    settings: list[dict[str, object]] = [{}]
    for knob, values in dims.items():
        expanded: list[dict[str, object]] = []
        for base in settings:
            for value in values:
                expanded.append({**base, knob: value})
        settings = expanded
    return settings


def knob_grid_for_provider(provider: str) -> list[KnobSetting]:
    """The ordered candidate :class:`KnobSetting` list for ``provider`` — baseline first, or ``[]``.

    The first element is ALWAYS the baseline (fp32/plain, compile off, batch 1 / concurrency 1) —
    the
    correctness reference the tuner compares every other candidate against and the fallback winner
    when nothing accelerates safely. A Provider with no known knob family yields ``[]`` (not swept).
    Deterministic + torch-free.
    """
    family = _PROVIDER_FAMILY.get(provider)
    if family is None:
        return []
    dims = _FAMILY_KNOBS[family]
    return [KnobSetting(knobs=knobs) for knobs in _cartesian(dims)]
