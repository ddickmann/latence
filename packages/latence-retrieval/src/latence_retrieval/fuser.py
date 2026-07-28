"""The fuse seam: :class:`Fuser` + its two adapters :class:`RrfFuser` / :class:`WeightedFuser`.

``fuse`` is the query-time transform that merges *N* ranked candidate lists into one (CONTEXT
``Fuse``). It is a **distinct shape** from :class:`~latence_retrieval.processor.CandidateProcessor`:
a processor reshapes *one* list ``(query, candidates) → candidates``; a fuser combines *many*
``[list, list, …] → list``. So it is its own seam, not a processor — but it obeys the same
statelessness contract (ADR-0048): a pure transform over caller-supplied lists that holds nothing,
fetches nothing, and never mutates its inputs (:class:`~latence_retrieval.contracts.Candidate` is
frozen).

The seam has two real adapters because the *combination rule* is exactly where behaviour varies:

* :class:`RrfFuser` — **Reciprocal Rank Fusion**: score by *rank position only*, scale-free, so a
  dense-cosine list and a BM25 list combine without their incomparable score ranges distorting the
  result. The robust default.
* :class:`WeightedFuser` — **weighted score fusion**: normalise each list's scores to ``[0, 1]``,
  then take a weighted sum, so the *magnitude* of a hit's relevance (not just its rank) carries
  through — the normalized-convex-combination family Bruch et al. (TOIS 2023) measure as beating
  RRF, with two normalisations: per-query ``min_max`` and query-independent
  ``theoretical_min_max`` (safe at serving time). RRF is **kept**, not deprecated: with no labelled
  queries there is nothing to tune, so it remains the no-labels fallback and the default.

:mod:`latence_retrieval.tuning` is the other half of that choice: it evaluates a fuser on a small
labelled slice, sweeps the weights on it, and renders the RRF-vs-tuned comparison table — so
"which fuser wins" is a measurement a harness runs, not a preference the library encodes.

Both are selected by config through :func:`~latence_retrieval.registry.resolve_fuser`
(``{"provider": "fuser.rrf" | "fuser.weighted", "config": {…}}``), so swapping the fusion algorithm
is a one-line config change (ADR-0049; user story 36). Both take optional per-list ``weights`` at
call time (the lists — and thus their weights — are known only per call, not at construction), and
both order ties deterministically by candidate ``id`` so identical inputs always yield an identical
output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from latence_retrieval.contracts import Candidate

if TYPE_CHECKING:
    from collections.abc import Sequence


@runtime_checkable
class Fuser(Protocol):
    """A pure transform that merges *N* ranked candidate lists into one ranked list (ADR-0048).

    One method, :meth:`fuse`. It takes the ``rankings`` (each an already-ranked list the caller's
    engine produced — dense, sparse, BM25, …, each in descending relevance) and optional per-list
    ``weights``, and returns a single list in descending fused score. The fuser fetches nothing and
    never mutates an input list or candidate, so identical inputs always produce an identical output
    — statelessness is the contract, not a convention. This is a *different* shape from
    :class:`~latence_retrieval.processor.CandidateProcessor` (many lists in, no ``query``), so it is
    its own seam.
    """

    def fuse(
        self,
        rankings: Sequence[Sequence[Candidate]],
        weights: Sequence[float] | None = None,
    ) -> list[Candidate]:
        """Merge ``rankings`` into one descending-score list; ``weights`` scale each list's pull."""
        ...


def _validate_weights(
    rankings: Sequence[Sequence[Candidate]],
    weights: Sequence[float] | None,
) -> list[float]:
    """Return a per-list weight vector: the caller's ``weights`` (length-checked) or all-ones.

    ``weights`` are call-time data (one per ranked list passed *this* call), so their length is
    validated here — against the actual ``rankings`` — not at construction. A length mismatch is a
    caller error and fails loudly rather than silently dropping or zero-padding a list's influence.
    """
    if weights is None:
        return [1.0] * len(rankings)
    weights = list(weights)
    if len(weights) != len(rankings):
        msg = (
            f"weights length {len(weights)} does not match the number of ranked lists "
            f"{len(rankings)}; supply exactly one weight per list (or none for uniform)."
        )
        raise ValueError(msg)
    return [float(w) for w in weights]


def _fuse_by_score(
    rankings: Sequence[Sequence[Candidate]],
    weights: Sequence[float],
    contribution: Any,
) -> list[Candidate]:
    """Shared fold: accumulate each id's weighted ``contribution`` across lists, then order.

    ``contribution(rank, candidate, list_index)`` yields one list's additive score for a candidate
    (RRF uses ``rank``; weighted uses the candidate's normalised score). Common to both adapters is
    everything *around* that choice — first-seen candidate identity, additive accumulation across
    overlapping lists, per-list dedup keeping the best (first) rank, and the deterministic
    ``(-score, id)`` final ordering — so the adapters differ only in the one line that actually
    defines the fusion.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Candidate] = {}
    for list_index, ranking in enumerate(rankings):
        weight = weights[list_index]
        seen_in_list: set[str] = set()
        for position, candidate in enumerate(ranking):
            if candidate.id in seen_in_list:
                # A well-formed ranked list has no repeated id; if one slips in, the first
                # (best) rank wins and the duplicate is ignored, so a malformed list cannot
                # double-count an id.
                continue
            seen_in_list.add(candidate.id)
            first_seen.setdefault(candidate.id, candidate)
            scores[candidate.id] = scores.get(candidate.id, 0.0) + weight * contribution(
                position + 1, candidate, list_index
            )
    fused = [
        first_seen[cid].model_copy(update={"score": score}) for cid, score in scores.items()
    ]
    # Deterministic for tied fused scores: descending score, then ascending id. Identical inputs
    # (in any equal-score arrangement) therefore always yield an identical output order.
    fused.sort(key=lambda c: (-c.score, c.id))
    return fused


class RrfFuser:
    """Reciprocal Rank Fusion: combine lists by rank position, ignoring raw score magnitudes.

    For each candidate, sums ``weight_i / (k + rank_i)`` over every list ``i`` it appears in (the
    rank is 1-based). Because it reads *only* the rank, lists with incomparable score scales
    (cosine, BM25, a reranker logit) fuse cleanly — the property that makes RRF the robust default.
    A candidate
    absent from a list simply contributes nothing from it, so disjoint and overlapping id sets are
    handled uniformly. The emitted candidate keeps the ``text``/``metadata`` of its first appearance
    (in input list order) with its ``score`` replaced by the fused score.

    Config: ``k`` (default 60, the standard RRF constant) damps the influence of top ranks; it must
    be positive.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        k = config.get("k", 60.0)
        if not isinstance(k, (int, float)) or isinstance(k, bool) or k <= 0:
            msg = f"RrfFuser 'k' must be a positive number, got {k!r}."
            raise ValueError(msg)
        self._k = float(k)

    def fuse(
        self,
        rankings: Sequence[Sequence[Candidate]],
        weights: Sequence[float] | None = None,
    ) -> list[Candidate]:
        """Fuse by reciprocal rank; see the class docstring for the exact score."""
        resolved = _validate_weights(rankings, weights)
        return _fuse_by_score(
            rankings,
            resolved,
            lambda rank, _candidate, _list_index: 1.0 / (self._k + rank),
        )


_NORMALIZATIONS = ("min_max", "theoretical_min_max")
_WEIGHTED_CONFIG_KEYS = frozenset({"normalization", "bounds"})


class WeightedFuser:
    """Weighted score fusion: normalise each list's scores to ``[0, 1]``, then take a weighted sum.

    Ranks throw away *how much* better a hit is; this adapter keeps that magnitude — the
    normalized-convex-combination family Bruch et al. (TOIS 2023) measure as beating RRF both
    in-domain and out-of-domain, and the one RRF was explicitly written to be a *baseline* for
    (D6). Each candidate's fused score is the weighted sum of its normalised scores across the
    lists it appears in; a candidate absent from a list contributes nothing from it. The emitted
    candidate keeps the ``text``/``metadata`` of its first appearance with its ``score`` replaced
    by the fused score.

    Config:
        normalization: ``"min_max"`` (default) or ``"theoretical_min_max"``. An unrecognised value
            fails loudly rather than silently changing the maths.
        bounds: **required for, and only for,** ``"theoretical_min_max"`` — either one ``[lo, hi]``
            pair applied to every list, or one pair per list (matched positionally at fuse time).

    **min_max** (per-query): within each list the scores are min-max-normalised against *that
    list's own* observed range, so a list scored 0.9–0.95 and one scored 2–40 contribute on the
    same footing. When a list's scores are all equal (including a single-element list) its range is
    zero and every member normalises to ``1.0`` — the list expresses no preference, so each gets
    that list's full weight rather than a divide-by-zero.

    **theoretical_min_max** (query-independent): the range comes from the operator's declared
    *theoretical* bounds of each scoring function (cosine ``[-1, 1]``, a lexical score ``[0, m]``),
    not from this query's hits. That is the property Bruch recommends for **serving time**: a
    document's normalised contribution no longer depends on which other documents happened to be
    retrieved for this particular query, so the fusion is stable across queries and a tuned weight
    transfers. A score outside its declared range is **clamped** into ``[0, 1]``: a theoretical
    bound is a claim about the score *function*, so an overshoot means the bound was mis-declared —
    clamping keeps the combination convex (and surfaces the mistake as saturation) instead of
    letting one list's overshoot silently dominate.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        unknown = set(config) - _WEIGHTED_CONFIG_KEYS
        if unknown:
            msg = f"WeightedFuser: unknown config key(s) {sorted(unknown)}."
            raise ValueError(msg)
        normalization = config.get("normalization", "min_max")
        if normalization not in _NORMALIZATIONS:
            msg = (
                f"WeightedFuser 'normalization' must be one of {list(_NORMALIZATIONS)}, "
                f"got {normalization!r}."
            )
            raise ValueError(msg)
        self._normalization = normalization
        raw_bounds = config.get("bounds")
        if normalization == "min_max":
            if raw_bounds is not None:
                msg = (
                    "WeightedFuser 'bounds' applies only to normalization="
                    "'theoretical_min_max'; per-query 'min_max' derives its range from each list. "
                    "Remove 'bounds' or switch the normalization — a silently ignored knob is how "
                    "a config stops reaching the provider."
                )
                raise ValueError(msg)
            self._uniform_bounds: tuple[float, float] | None = None
            self._per_list_bounds: tuple[tuple[float, float], ...] | None = None
        else:
            if raw_bounds is None:
                msg = (
                    "WeightedFuser normalization='theoretical_min_max' requires 'bounds': either "
                    "one [lo, hi] pair for every list, or one pair per list."
                )
                raise ValueError(msg)
            self._uniform_bounds, self._per_list_bounds = _parse_bounds(raw_bounds)

    def fuse(
        self,
        rankings: Sequence[Sequence[Candidate]],
        weights: Sequence[float] | None = None,
    ) -> list[Candidate]:
        """Fuse by weighted, normalised score; see the class docstring for the two modes."""
        resolved = _validate_weights(rankings, weights)
        if self._normalization == "min_max":
            normalised = [_min_max_normalised(ranking) for ranking in rankings]
        else:
            per_list = self._bounds_for(len(rankings))
            normalised = [
                _theoretical_normalised(ranking, per_list[index])
                for index, ranking in enumerate(rankings)
            ]
        return _fuse_by_score(
            rankings,
            resolved,
            lambda rank, candidate, list_index: normalised[list_index][candidate.id],
        )

    def _bounds_for(self, count: int) -> list[tuple[float, float]]:
        """Align the declared bounds to the ranked lists actually passed *this* call.

        A single declared pair applies to every list; per-list pairs must match the number of lists
        the caller passed, because a positional mapping that silently truncates or recycles would
        normalise a list against another list's score function.
        """
        if self._uniform_bounds is not None:
            return [self._uniform_bounds] * count
        per_list = list(self._per_list_bounds or ())
        if len(per_list) != count:
            msg = (
                f"WeightedFuser 'bounds' has {len(per_list)} pairs but {count} ranked lists were "
                "passed; supply exactly one pair per list (or a single pair for all)."
            )
            raise ValueError(msg)
        return per_list


def _parse_bounds(
    raw: Any,
) -> tuple[tuple[float, float] | None, tuple[tuple[float, float], ...] | None]:
    """Parse ``bounds`` into ``(uniform_pair, per_list_pairs)`` — exactly one of them is set."""
    if _is_pair(raw):
        return _validate_pair(raw), None
    if isinstance(raw, (list, tuple)) and raw and all(_is_pair(item) for item in raw):
        return None, tuple(_validate_pair(item) for item in raw)
    msg = (
        "WeightedFuser 'bounds' must be one [lo, hi] number pair, or a non-empty list of such "
        f"pairs (one per ranked list), got {raw!r}."
    )
    raise ValueError(msg)


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _validate_pair(pair: Any) -> tuple[float, float]:
    lo, hi = float(pair[0]), float(pair[1])
    if not lo < hi:
        msg = f"WeightedFuser 'bounds' pair must satisfy lo < hi, got [{lo}, {hi}]."
        raise ValueError(msg)
    return (lo, hi)


def _theoretical_normalised(
    ranking: Sequence[Candidate], bounds: tuple[float, float]
) -> dict[str, float]:
    """Map each candidate id to ``(score - lo) / (hi - lo)``, clamped into ``[0, 1]``.

    Query-independent by construction: the range is the score *function*'s declared one, not this
    list's observed one, so the same raw score always maps to the same normalised score.
    """
    lo, hi = bounds
    span = hi - lo
    result: dict[str, float] = {}
    for candidate in ranking:
        if candidate.id in result:
            continue
        result[candidate.id] = min(1.0, max(0.0, (candidate.score - lo) / span))
    return result


def _min_max_normalised(ranking: Sequence[Candidate]) -> dict[str, float]:
    """Map each candidate id in ``ranking`` to its min-max-normalised score in ``[0, 1]``.

    A zero range (all scores equal, or a single element) maps every member to ``1.0`` — the list
    voices no preference, so no member is penalised. On a duplicated id the first occurrence wins,
    matching the fold's per-list dedup.
    """
    if not ranking:
        return {}
    values = [c.score for c in ranking]
    lo, hi = min(values), max(values)
    span = hi - lo
    result: dict[str, float] = {}
    for candidate in ranking:
        if candidate.id in result:
            continue
        result[candidate.id] = 1.0 if span == 0 else (candidate.score - lo) / span
    return result
