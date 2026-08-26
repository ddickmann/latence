"""Temperature scaling + a precision-targeted threshold for a link-prediction scorer.

The calibrator the learned :mod:`~latence_linkpred_ultra.provider` runs over raw scorer outputs so
what it emits as an edge's ``confidence`` is a **calibrated probability**, never a raw model score,
and so it only emits an edge whose calibrated probability clears a configurable **target
precision**. This is the predecessor stack's *calibrator* discipline, ported as-is from its
link-prediction stack: temperature scaling (Guo et al. 2017) + precision-targeted
thresholds. It is pure math (no torch), so it is unit-testable offline.

Enterprise honesty (why this exists): a raw ULTRA/PyKEEN score is an *unbounded logit-like* number
whose scale is arbitrary; presenting it as a 0..1 confidence would be a fabricated calibration. The
calibrator turns raw scores into probabilities two ways:

* **Temperature scaling** — divide the raw score by a learned/configured temperature ``T`` and pass
  it through a logistic sigmoid: ``p = 1 / (1 + exp(-score / T))``. ``T > 1`` softens
  over-confident scores; ``T = 1`` is the plain sigmoid. When a small held-out labelled set is
  available, ``T`` is fit by a deterministic 1-D search minimising negative-log-likelihood (a
  bounded golden-ish scan, no torch autograd), so the calibration is *learned from data*, not
  guessed. With no labelled set it falls back to the configured ``temperature``.
* **Precision-targeted threshold** — given the calibrated probabilities of a labelled held-out set,
  pick the LOWEST probability cut at which precision ≥ ``target_precision`` (so recall is
  maximised at the required precision). With no labelled set it falls back to a configured
  ``min_probability`` floor. Either way, a candidate below the chosen cut is **not emitted** — the
  enterprise "don't assert a low-precision belief" guarantee.

Determinism (Baseline bar): the temperature scan is a fixed grid, the threshold is the min
qualifying calibrated probability over a sorted set, and all outputs are rounded to a fixed
precision — identical input yields byte-identical calibration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The fixed temperature scan grid (deterministic). A log-spaced-ish set of positive temperatures the
# NLL fit searches over when a held-out labelled set is available — no autograd, fully reproducible.
_TEMPERATURE_GRID: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0,
)
# Rounding precision for every emitted probability / temperature, so runs are byte-identical.
_ROUND = 6


@dataclass(frozen=True)
class CalibratorConfig:
    """Calibrator knobs.

    ``temperature`` is the fallback ``T`` used when no held-out labelled set is supplied to fit it
    (``>0``). ``target_precision`` is the precision the emit threshold targets (``0..1``).
    ``min_probability`` is the fallback probability floor used when no labelled set is available to
    derive a precision-targeted cut, OR when the labelled set is too small to calibrate on
    (``0..1``). ``min_calibration_samples`` is the minimum labelled examples required before a
    precision-targeted threshold is fit from data — below it, the precision cut would be
    statistical noise (a single (positive, negative) pair cannot justify a threshold), so the
    calibrator falls back to ``min_probability`` rather than over-fitting a meaningless cut.
    """

    temperature: float = 1.0
    target_precision: float = 0.9
    min_probability: float = 0.5
    min_calibration_samples: int = 8


@dataclass(frozen=True)
class Calibration:
    """A fitted calibration: the chosen temperature and the emit probability threshold."""

    temperature: float
    threshold: float


def sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class Calibrator:
    """Temperature-scale raw scores into probabilities and pick a precision-targeted threshold."""

    def __init__(self, config: CalibratorConfig | None = None) -> None:
        self._config = config or CalibratorConfig()
        if self._config.temperature <= 0.0:
            msg = f"temperature must be > 0, got {self._config.temperature}"
            raise ValueError(msg)
        if not 0.0 <= self._config.target_precision <= 1.0:
            msg = f"target_precision must be in [0,1], got {self._config.target_precision}"
            raise ValueError(msg)
        if not 0.0 <= self._config.min_probability <= 1.0:
            msg = f"min_probability must be in [0,1], got {self._config.min_probability}"
            raise ValueError(msg)

    def probability(self, raw_score: float, temperature: float) -> float:
        """Calibrated probability of a raw score at a temperature (rounded, deterministic)."""
        return round(sigmoid(raw_score / temperature), _ROUND)

    def fit(
        self,
        labelled: list[tuple[float, bool]] | None = None,
    ) -> Calibration:
        """Fit the temperature + emit threshold from an OPTIONAL held-out labelled set.

        ``labelled`` is a list of ``(raw_score, is_true_edge)`` pairs from a held-out split. When
        present:

        * the temperature is the grid value minimising negative-log-likelihood over the set;
        * the threshold is the lowest calibrated probability cut achieving precision ≥
          ``target_precision`` (maximising recall at that precision). If no cut reaches the target
          (e.g. the scorer cannot separate the classes at this precision), the threshold is set
          just above the max positive probability so nothing is emitted — refusing to emit is the
          honest disposition, never lowering the bar silently.

        With no labelled set it returns the configured fallback ``(temperature, min_probability)``.
        """
        if not labelled:
            return Calibration(
                temperature=round(self._config.temperature, _ROUND),
                threshold=round(self._config.min_probability, _ROUND),
            )
        temperature = self._fit_temperature(labelled)
        if len(labelled) < self._config.min_calibration_samples:
            # Too few labelled examples to fit a precision-targeted cut without over-fitting noise:
            # keep the NLL-fit temperature (it is well-defined at any size) but fall back to the
            # configured ``min_probability`` floor for the emit threshold (honest under-calibration,
            # not a fabricated precision cut off one example).
            return Calibration(
                temperature=round(temperature, _ROUND),
                threshold=round(self._config.min_probability, _ROUND),
            )
        threshold = self._precision_threshold(labelled, temperature)
        return Calibration(
            temperature=round(temperature, _ROUND),
            threshold=round(threshold, _ROUND),
        )

    # -- temperature fit (deterministic grid NLL) ---------------------------

    def _fit_temperature(self, labelled: list[tuple[float, bool]]) -> float:
        best_t = self._config.temperature
        best_nll = math.inf
        for t in _TEMPERATURE_GRID:
            nll = 0.0
            for raw, is_true in labelled:
                p = sigmoid(raw / t)
                # Clamp to avoid log(0); a fixed epsilon keeps it deterministic.
                p = min(max(p, 1e-12), 1.0 - 1e-12)
                nll -= math.log(p) if is_true else math.log(1.0 - p)
            # Strict `<` so the first (smallest-T) minimiser wins on ties — deterministic.
            if nll < best_nll:
                best_nll = nll
                best_t = t
        return best_t

    # -- precision-targeted threshold ---------------------------------------

    def _precision_threshold(
        self,
        labelled: list[tuple[float, bool]],
        temperature: float,
    ) -> float:
        scored = sorted(
            ((sigmoid(raw / temperature), is_true) for raw, is_true in labelled),
            key=lambda pair: pair[0],
        )
        # Candidate cuts are the distinct calibrated probabilities (a cut keeps p >= cut). Evaluate
        # each; pick the LOWEST cut whose precision >= target (max recall at the target precision).
        cuts = sorted({p for p, _ in scored})
        best_cut: float | None = None
        for cut in cuts:
            kept = [(p, is_true) for p, is_true in scored if p >= cut]
            if not kept:
                continue
            true_positives = sum(1 for _, is_true in kept if is_true)
            precision = true_positives / len(kept)
            if precision >= self._config.target_precision:
                best_cut = cut
                break
        if best_cut is None:
            # No cut reaches the target precision: refuse to emit (threshold just above the max p).
            max_p = cuts[-1] if cuts else self._config.min_probability
            return min(max_p + 1e-6, 1.0)
        return best_cut
