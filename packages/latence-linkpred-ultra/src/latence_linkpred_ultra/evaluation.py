"""Held-out MRR / Hits@k evaluation with genuine leak detection.

The KG-quality lens (the issue's AC): split the observed edges into an *observed* (graph the scorer
sees) and a *held-out* (edges the scorer must rank back), score each held-out edge against all
type-valid tails, and report **MRR** and **Hits@k** — the standard link-prediction metrics. The
non-negotiable part is **leak detection**: the held-out edges AND their inverses must be removed
from the observed graph before scoring, and this module *asserts* that removal (a held-out edge, or
its inverse, still present in the observed graph would let the scorer trivially "rank" a leaked
answer — a fabricated MRR). It refuses to compute metrics over a leaking split.

This module is **pure/torch-free**: it drives any ``Scorer`` (or a plain callable returning a
per-entity score vector) and computes the ranking arithmetic itself, so it is unit-testable offline
against a deterministic fake scorer and runs identically for ULTRA or PyKEEN. The real ULTRA MRR is
re-validated on the pod; this harness computes the number honestly from whatever scorer it is given.
"""

from __future__ import annotations

from dataclasses import dataclass

from latence_linkpred_ultra.candidates import ObservedEdge


class LeakDetectionError(ValueError):
    """Raised when the held-out split leaks into the observed graph (a fabricated-MRR guard).

    A ``ValueError`` subtype so an existing ``except ValueError`` boundary still catches it; the
    provider maps it to the framework's typed error taxonomy.
    """


@dataclass(frozen=True)
class SplitResult:
    """An observed/held-out edge split with the entity vocab, ready to evaluate."""

    observed: list[ObservedEdge]
    held_out: list[ObservedEdge]
    entities: list[str]


@dataclass(frozen=True)
class RankingMetrics:
    """MRR + Hits@k over a held-out split (filtered ranking).

    ``mrr`` is the mean reciprocal rank; ``hits`` maps each requested ``k`` to the fraction of
    held-out edges ranked within the top ``k``; ``num_queries`` is how many held-out edges were
    scored — one TAIL query per held-out edge (the provider issues tail queries only, so
    ``num_queries == len(held_out)``); ``leak_checked`` records that the leak-detection assertion
    (:func:`assert_no_leak`) ran and passed BEFORE these ranks were computed. ``leak_checked`` has
    no default: the caller MUST pass it, so a code path that skipped the leak check cannot silently
    inherit ``True`` (the reviewer's tautology concern).
    """

    mrr: float
    hits: dict[int, float]
    num_queries: int
    leak_checked: bool


def split_edges(
    observed: list[ObservedEdge],
    *,
    holdout_fraction: float,
    seed: int = 0,
) -> SplitResult:
    """Deterministically split ``observed`` into observed/held-out edges.

    The split is a seeded, stable hash over each edge's identity so the SAME edges are held out for
    a
    given seed (reproducible), taking ``holdout_fraction`` of the edges (at least one when the graph
    is non-trivial). The held-out edges are removed from the observed set the scorer sees; leak
    detection (:func:`assert_no_leak`) verifies the removal — including inverses.
    """
    if not 0.0 < holdout_fraction < 1.0:
        msg = f"holdout_fraction must be in (0,1), got {holdout_fraction}"
        raise ValueError(msg)
    ordered = sorted(observed, key=lambda e: (e.relation, e.head, e.tail))
    entities = sorted({e.head for e in observed} | {e.tail for e in observed})
    n_hold = max(1, int(round(len(ordered) * holdout_fraction))) if len(ordered) >= 2 else 0
    # Deterministic selection: rank edges by a seeded hash and take the first n_hold as held-out.
    ranked = sorted(ordered, key=lambda e: _edge_hash(e, seed))
    held = set(map(_edge_key, ranked[:n_hold]))
    held_out = [e for e in ordered if _edge_key(e) in held]
    observed_rest = [e for e in ordered if _edge_key(e) not in held]
    return SplitResult(observed=observed_rest, held_out=held_out, entities=entities)


def assert_no_leak(observed: list[ObservedEdge], held_out: list[ObservedEdge]) -> None:
    """Assert no held-out edge (or its inverse) remains in the observed graph — else raise.

    The genuine leak-detection guard: a held-out edge ``(h, r, t)`` must NOT appear in the observed
    graph, and NEITHER must its inverse ``(t, r, h)`` (ULTRA doubles relations with inverses, so a
    surviving inverse leaks the answer just as badly). Raises :class:`LeakDetectionError` naming the
    offending edge if either is present.
    """
    observed_keys = {_edge_key(e) for e in observed}
    for e in held_out:
        if _edge_key(e) in observed_keys:
            msg = f"held-out edge leaks into the observed graph: {_edge_key(e)}"
            raise LeakDetectionError(msg)
        inverse = (e.tail, e.relation, e.head)
        if inverse in observed_keys:
            msg = (
                f"inverse of held-out edge leaks into the observed graph: "
                f"{_edge_key(e)} (inverse {inverse})"
            )
            raise LeakDetectionError(msg)


def rank_of(target_index: int, scores: list[float], filter_indices: set[int]) -> int:
    """Filtered rank (1-based) of ``target_index`` given per-entity ``scores``.

    Higher score = better. The FILTERED protocol: other true entities (``filter_indices``, e.g.
    other valid tails for this ``(h, r, ?)`` query) are not counted against the target, so a
    correct competitor never inflates the rank. The target itself is never in ``filter_indices``.
    """
    target_score = scores[target_index]
    rank = 1
    for i, s in enumerate(scores):
        if i == target_index or i in filter_indices:
            continue
        if s > target_score:
            rank += 1
    return rank


def metrics_from_ranks(
    ranks: list[int], ks: tuple[int, ...], *, leak_checked: bool
) -> RankingMetrics:
    """Aggregate 1-based ranks into MRR + Hits@k.

    ``leak_checked`` MUST be passed by the caller: it is True only when :func:`assert_no_leak` was
    run over the same held-out split BEFORE these ranks were computed. It is threaded through (not
    defaulted) so the recorded flag genuinely attests the leak guard ran — a caller that computed
    ranks without the guard would have to pass ``leak_checked=False`` and could not silently claim
    a clean eval.
    """
    if not ranks:
        return RankingMetrics(
            mrr=0.0, hits={k: 0.0 for k in ks}, num_queries=0, leak_checked=leak_checked
        )
    mrr = sum(1.0 / r for r in ranks) / len(ranks)
    hits = {k: sum(1 for r in ranks if r <= k) / len(ranks) for k in ks}
    return RankingMetrics(
        mrr=round(mrr, 6),
        hits={k: round(v, 6) for k, v in hits.items()},
        num_queries=len(ranks),
        leak_checked=leak_checked,
    )


# -- helpers -----------------------------------------------------------------


def _edge_key(e: ObservedEdge) -> tuple[str, str, str]:
    return (e.head, e.relation, e.tail)


def _edge_hash(e: ObservedEdge, seed: int) -> str:
    import hashlib

    raw = f"{seed}|{e.head}|{e.relation}|{e.tail}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
