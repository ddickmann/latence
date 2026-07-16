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
* :class:`WeightedFuser` — **weighted score fusion**: min-max-normalise each list's own scores, then
  take a weighted sum, so the *magnitude* of a hit's relevance (not just its rank) carries through —
  the stronger fusion the spec calls for when ranks alone throw information away.

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


class WeightedFuser:
    """Weighted score fusion: min-max-normalise each list's own scores, then take a weighted sum.

    Ranks throw away *how much* better a hit is; this adapter keeps that magnitude. Within each list
    the scores are min-max-normalised to ``[0, 1]`` (so a list scored 0.9–0.95 and one scored 2–40
    contribute on the same footing), then each candidate's fused score is the weighted sum of its
    normalised scores across the lists it appears in. A candidate absent from a list contributes
    nothing from it. When a list's scores are all equal (including a single-element list) its range
    is zero and every member normalises to ``1.0`` — the list expresses no preference, so each gets
    that list's full weight rather than a divide-by-zero. The emitted candidate keeps the
    ``text``/``metadata`` of its first appearance with its ``score`` replaced by the fused score.

    Config: currently none beyond the defaults; ``normalization`` accepts only ``"min_max"`` (the
    default) so an unrecognised value fails loudly rather than silently changing the maths.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        normalization = config.get("normalization", "min_max")
        if normalization != "min_max":
            msg = (
                f"WeightedFuser 'normalization' must be 'min_max' (the only supported mode), "
                f"got {normalization!r}."
            )
            raise ValueError(msg)

    def fuse(
        self,
        rankings: Sequence[Sequence[Candidate]],
        weights: Sequence[float] | None = None,
    ) -> list[Candidate]:
        """Fuse by weighted, per-list min-max-normalised score; see the class docstring."""
        resolved = _validate_weights(rankings, weights)
        normalised = [_min_max_normalised(ranking) for ranking in rankings]
        return _fuse_by_score(
            rankings,
            resolved,
            lambda rank, candidate, list_index: normalised[list_index][candidate.id],
        )


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
