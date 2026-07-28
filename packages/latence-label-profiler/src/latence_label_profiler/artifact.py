"""Serialization of a :class:`~latence_label_profiler.profiler.CorpusProfile` to stable JSON.

Two rules the whole determinism contract rests on:

* **Nothing is iterated in container order.** Every mapping is built from an explicitly sorted or
  explicitly ranked sequence, and ``json.dumps(..., sort_keys=True)`` fixes the rest. A dict or set
  that leaked its iteration order here would make the output depend on the interpreter's string-hash
  seed, which is exactly what ``test_determinism_across_processes.py`` exists to catch.
* **Floats are rounded at the boundary.** Unrounded ratios are reproducible in CPython, but rounding
  makes the artifact diffable by a human, which is most of what it is for.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from latence_label_profiler.coverage import WORST_CASE_RATIO

if TYPE_CHECKING:
    from latence_label_profiler.profiler import CorpusProfile, SlotProfile

#: Artifact schema version. Bumped when a consumer would have to change to keep reading it.
ARTIFACT_VERSION = 1

_PRECISION = 6


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), _PRECISION)


def _slot_dict(profile: SlotProfile) -> dict[str, Any]:
    result = profile.result
    total = result.total_weight
    selection = [
        {
            "rank": step.rank,
            "label": step.label,
            "marginal_gain": _round(step.marginal_gain),
            "marginal_gain_fraction": _round(step.marginal_gain / total if total else 0.0),
            "cumulative_mass": _round(step.cumulative),
            "coverage_fraction": _round(step.coverage_fraction),
            "certificate_bound": _round(step.certificate_bound),
            "mention_mass": profile.candidates[step.label].mass,
            "document_frequency": profile.candidates[step.label].document_frequency,
            "detector_only": profile.candidates[step.label].detector_only,
            "cues": list(profile.candidates[step.label].cues),
        }
        for step in result.steps
    ]
    return {
        "budget_n": profile.budget,
        "candidates_considered": result.candidates_considered,
        "mention_mass_total": _round(total),
        "selection": selection,
        "marginal_gain_curve": [
            {
                "n": step.rank,
                "marginal_gain": _round(step.marginal_gain),
                "marginal_gain_fraction": _round(step.marginal_gain / total if total else 0.0),
            }
            for step in result.steps
        ],
        "coverage_curve": [
            {"n": step.rank, "coverage": _round(step.coverage_fraction)} for step in result.steps
        ],
        "certificate": {
            "greedy_value": _round(result.value),
            "upper_bound_on_optimum": _round(result.certificate_bound),
            "ratio_to_bound": _round(result.certificate_ratio),
            "worst_case_floor": WORST_CASE_RATIO,
            "tighter_than_worst_case": bool(result.certificate_ratio > WORST_CASE_RATIO),
            "meaning": (
                "f(OPT_N) <= f(A_i) + sum of the N largest remaining marginal gains at A_i, "
                "minimised over every prefix i and clamped to the total mention mass. "
                "ratio_to_bound is a per-corpus optimality guarantee; worst_case_floor is the "
                "Nemhauser-Wolsey-Fisher (1978) theorem's floor, which Feige (1998) proved no "
                "polynomial algorithm improves on."
            ),
        },
        "coverage": {
            "empirical_coverage": _round(result.coverage_fraction),
            "covered_mention_mass": _round(result.value),
            "total_mention_mass": _round(total),
        },
        "zipf": {
            "n_types": profile.zipf.n_types,
            "total_mass": profile.zipf.total_mass,
            "rank_slope": _round(profile.zipf.rank_slope),
            "mle_alpha": _round(profile.zipf.mle_alpha),
            "log_growth": profile.zipf.log_growth,
        },
        "residual": {
            "empirical_uncovered_mass": _round(profile.empirical_uncovered_mass),
            "good_turing_unseen_mass": _round(profile.good_turing_unseen_mass),
            "estimated_future_uncovered_mass": _round(profile.estimated_future_uncovered_mass),
            "meaning": (
                "good_turing_unseen_mass = N1/N (Good 1953): the estimated share of FUTURE "
                "mentions whose type this sample never observed. "
                "estimated_future_uncovered_mass composes it "
                "with the empirically uncovered share."
            ),
        },
        "knee": {
            "n_star": profile.knee.x,
            "max_difference": _round(profile.knee.max_difference),
            "sharp": profile.knee.sharp,
            "n_is_a_budget_decision": not profile.knee.sharp,
            "capped_n": profile.budget,
            "interpretation": (
                (
                    f"Kneedle locates the knee at N* = {profile.knee.x}; the difference peak is "
                    f"{profile.knee.max_difference:.3f}, an unambiguous elbow."
                )
                if profile.knee.sharp
                else (
                    "No sharp knee: the Kneedle difference peak is "
                    f"{profile.knee.max_difference:.3f}, below the 0.5 convention declared in "
                    "stats.SHARPNESS_THRESHOLD. N is a budget decision, not a discovered boundary. "
                    "There is no published standard for a coverage or marginal-gain cutoff."
                )
            ),
        },
        "notes": list(profile.notes),
    }


def render_artifact(profile: CorpusProfile) -> dict[str, Any]:
    """The whole artifact as plain data: labels, curves, certificate, fit, residual, caveats."""
    from latence_label_profiler.mining import SLOTS

    return {
        "artifact_version": ARTIFACT_VERSION,
        "method": {
            "name": profile.method.name,
            "network_calls": profile.method.network_calls,
            "model_id": profile.method.model_id,
            "cost_usd": profile.method.cost_usd,
            "references": list(profile.method.references),
        },
        "corpus": {
            "texts_seen": profile.corpus.texts_seen,
            "texts_sampled": profile.corpus.texts_sampled,
            "sample_seed": profile.corpus.sample_seed,
            "sample_strategy": profile.corpus.sample_strategy,
        },
        "config": {
            "max_types": profile.config.max_types,
            "max_phrase_tokens": profile.config.max_phrase_tokens,
            "max_relation_tokens": profile.config.max_relation_tokens,
            "min_mention_mass": profile.config.min_mention_mass,
            "min_document_frequency": profile.config.min_document_frequency,
            "max_cues": profile.config.max_cues,
            "sample_size": profile.config.sample_size,
        },
        "slots": {
            slot: _slot_dict(profile.slots[slot]) for slot in SLOTS if slot in profile.slots
        },
        "candidates": profile.lexicon_candidates(),
        "notes": list(profile.notes),
    }


def to_json(profile: CorpusProfile) -> str:
    """Byte-stable JSON: sorted keys, fixed indent, ASCII-escaped, trailing newline."""
    return (
        json.dumps(render_artifact(profile), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    )
