"""The uncertainty the artifact is required to carry: Zipf fit, Good-Turing residual, Kneedle knee.

The ticket's honesty bar is the reason this module exists. There is **no published standard** for a
"90% coverage" or "marginal gain < x%" cutoff on a label budget. Under a Zipf distribution with
``alpha ~ 1`` the coverage of the top-N types grows like ``H(N) ~ ln N`` — logarithmically, with no
sharp knee anywhere — and in that regime N is a *budget decision*, not a discovered boundary. These
functions produce the numbers that let the artifact say which regime the corpus is in, instead of
implying a boundary that does not exist.

Everything is closed-form and pure-Python: no numpy, no scipy, no fitting library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: How far the fitted rank exponent may sit from 1.0 and still count as "logarithmic growth".
#: Declared here as a stated convention, not borrowed from a standard — there isn't one. A reader
#: who disagrees has ``rank_slope`` in the artifact and can draw their own line.
LOG_GROWTH_BAND: float = 0.25

#: The Kneedle difference-curve peak above which we are willing to call a knee SHARP. Kneedle
#: (Satopaa et al. 2011) defines where the knee IS; it does not define whether one EXISTS, so this
#: threshold is our convention and the raw ``max_difference`` is always reported alongside it.
#: 0.5 means the curve rises half the plot height above the diagonal — an unambiguous elbow.
SHARPNESS_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class ZipfFit:
    """A rank-frequency fit over the mined type distribution."""

    n_types: int
    total_mass: int
    #: OLS slope of ``log frequency`` on ``log rank`` (the classic Zipf exponent). ``None`` when
    #: fewer than three types were observed — a fit over two points asserts nothing.
    rank_slope: float | None
    #: Discrete power-law MLE exponent with ``x_min = 1`` (Clauset, Shalizi & Newman 2009, the
    #: continuous approximation ``alpha = 1 + n / sum(ln(x / (x_min - 0.5)))``). Reported ALONGSIDE
    #: the rank slope rather than instead of it: the two answer different questions and disagreeing
    #: is informative, whereas quietly picking one would hide that.
    mle_alpha: float | None
    #: True when the rank exponent sits within :data:`LOG_GROWTH_BAND` of 1.0, i.e. coverage grows
    #: logarithmically in N and no sharp knee can exist.
    log_growth: bool


@dataclass(frozen=True)
class Knee:
    """A Kneedle result over a normalised, concave-increasing curve."""

    x: int | None
    max_difference: float
    sharp: bool


def fit_zipf(counts: Mapping[str, int]) -> ZipfFit:
    """Fit the type frequency distribution, both by rank regression and by power-law MLE."""
    frequencies = sorted((int(v) for v in counts.values() if int(v) > 0), reverse=True)
    total = sum(frequencies)
    n = len(frequencies)
    if n < 3:
        return ZipfFit(
            n_types=n, total_mass=total, rank_slope=None, mle_alpha=None, log_growth=False
        )

    log_ranks = [math.log(rank) for rank in range(1, n + 1)]
    log_freqs = [math.log(freq) for freq in frequencies]
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    variance = sum((x - mean_x) ** 2 for x in log_ranks)
    slope: float | None = None
    if variance > 0:
        covariance = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs, strict=True)
        )
        # Reported as a POSITIVE exponent: frequency ~ rank ** -slope.
        slope = -covariance / variance

    denominator = sum(math.log(freq / 0.5) for freq in frequencies)
    mle = 1.0 + n / denominator if denominator > 0 else None

    log_growth = slope is not None and abs(slope - 1.0) <= LOG_GROWTH_BAND
    return ZipfFit(
        n_types=n, total_mass=total, rank_slope=slope, mle_alpha=mle, log_growth=log_growth
    )


def good_turing_missing_mass(counts: Mapping[str, int]) -> float:
    """Good (1953): the probability that the NEXT mention belongs to a type never yet observed.

    ``p0 = N1 / N`` — the share of total observed mass sitting in types seen exactly once. This is
    the estimate of what a bigger sample would still add, and it is the honest counterweight to a
    coverage number: ``C(40) = 0.86`` computed over a sample that has ``p0 = 0.19`` does not mean
    86% of the corpus's mentions are covered.

    An empty corpus returns ``1.0``: with no observations there is no evidence AGAINST the next
    mention being of an unseen type, and reporting ``0.0`` there would be a false reassurance.
    """
    values = [int(v) for v in counts.values() if int(v) > 0]
    total = sum(values)
    if total == 0:
        return 1.0
    singletons = sum(1 for v in values if v == 1)
    return singletons / total


def kneedle(xs: Sequence[int], ys: Sequence[float]) -> Knee:
    """Kneedle (Satopaa, Albrecht, Irwin & Raghavan, 2011) on a concave-increasing curve.

    Normalise both axes to ``[0, 1]``, take the difference curve ``y_norm - x_norm``, and call its
    argmax the knee. The published smoothing/spline step is deliberately omitted: submodularity
    already guarantees the marginal-gain sequence is monotonically non-increasing, so the cumulative
    coverage curve is concave by construction and there is no noise for a spline to remove.

    ``sharp`` is OUR convention (see :data:`SHARPNESS_THRESHOLD`), not Kneedle's — the algorithm
    locates a knee, it does not adjudicate whether one is real. ``max_difference`` is returned so a
    reader can apply a different line.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return Knee(x=None, max_difference=0.0, sharp=False)
    x_lo, x_hi = float(min(xs)), float(max(xs))
    y_lo, y_hi = float(min(ys)), float(max(ys))
    if x_hi <= x_lo or y_hi <= y_lo:
        return Knee(x=None, max_difference=0.0, sharp=False)
    best_diff = -math.inf
    best_x: int | None = None
    for x, y in zip(xs, ys, strict=True):
        diff = (float(y) - y_lo) / (y_hi - y_lo) - (float(x) - x_lo) / (x_hi - x_lo)
        if diff > best_diff:
            best_diff = diff
            best_x = int(x)
    difference = max(0.0, best_diff)
    return Knee(x=best_x, max_difference=difference, sharp=difference >= SHARPNESS_THRESHOLD)
