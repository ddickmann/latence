"""Affected-set incremental disambiguation — the recast IncrementalLinker (S11, ADR-0018).

The private stack's ``incremental_linker`` (ASSET-INVENTORY.md, RECAST) linked a delta's new
mentions against the *existing* KB — good for attach-to-known, but it structurally could not
catch two cases ADR-0018 calls out:

* **merge-on-add** — two pre-existing entity clusters that should MERGE when a *bridging*
  document arrives (a doc whose mentions connect surfaces that were previously in separate
  blocks);
* **split-on-delete** — a merged cluster that should SPLIT when its only bridging document is
  deleted (the evidence that justified the merge is gone).

The recast here is the **affected set**: rather than re-resolve the whole corpus (O(corpus)
per change) or naively attach-to-known (misses merge/split), it re-resolves only the *blocking
neighborhoods* the delta touched, plus their immediate neighbors, and leaves every untouched
cluster verbatim. Because full incremental resolution is order-dependent, a cheap **drift
metric** (the fraction of the corpus changed) crossing a threshold triggers a full
**Reconciliation** (re-resolve everything) — the lean/enterprise hinge (incremental speed,
bounded-error backstop).

Blocking: a mention's blocking keys are its normalised alphanumeric tokens (the same tokeniser
the disambiguation cascade uses), so two mentions that could plausibly resolve together share a
key. The affected set is the transitive closure, over shared keys, of the mentions in changed/
deleted documents plus their key-neighbors — exactly the region whose resolution could change.
This is deterministic and dependency-free (the Baseline bar): sorted throughout, so a seeded
corpus yields the same affected set and the same re-resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from latence_core.disambiguation import EntityResolver, MentionInput, ResolvedCluster, _tokens

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class DriftMetric:
    """The cheap drift metric that decides incremental vs full Reconciliation (ADR-0018).

    ``changed_documents`` are the documents this delta added/updated/deleted; ``total_documents``
    the corpus size (post-delta, or the parent size for a pure deletion — the caller passes the
    larger). ``drift`` is their ratio, in ``[0, 1]``. ``triggered`` is ``drift >= threshold``:
    when True the orchestration recomputes ALL blocks (a full Reconciliation) rather than only
    the affected set, correcting the order-dependent error incremental recompute accumulates.
    """

    changed_documents: int
    total_documents: int
    threshold: float
    drift: float
    triggered: bool


def compute_drift(
    changed_documents: int, total_documents: int, threshold: float
) -> DriftMetric:
    """Compute the drift ratio and whether it crosses the reconciliation threshold.

    ``drift = changed / total`` (0.0 for an empty corpus). ``triggered`` is ``drift >=
    threshold``; a threshold of 0.0 always reconciles and a threshold > 1.0 never does — both
    valid configured extremes.
    """
    total = max(total_documents, 0)
    drift = (changed_documents / total) if total else 0.0
    drift = min(max(drift, 0.0), 1.0)
    return DriftMetric(
        changed_documents=changed_documents,
        total_documents=total,
        threshold=threshold,
        drift=round(drift, 6),
        triggered=drift >= threshold,
    )


@dataclass
class AffectedSet:
    """The blocking neighborhoods a delta touched — the re-resolve region (``Affected set``).

    ``mention_ids`` are every mention in the affected blocking neighborhoods (the seeds — mentions
    in changed/deleted documents — plus their key-neighbors). ``blocking_keys`` are the normalised
    tokens that define those neighborhoods. ``is_full`` marks a full-corpus affected set (a
    Reconciliation): every mention is affected, so the recompute is a full re-resolution.
    """

    mention_ids: set[str] = field(default_factory=set)
    blocking_keys: set[str] = field(default_factory=set)
    is_full: bool = False

    @property
    def size(self) -> int:
        return len(self.mention_ids)


@dataclass(frozen=True)
class MentionView:
    """The contract-free mention view the affected-set engine folds in (id/doc/text/label/conf)."""

    mention_id: str
    document_id: str
    text: str
    label: str
    confidence: float


def _blocking_keys(text: str) -> set[str]:
    """A mention's blocking keys: its normalised alphanumeric tokens (shared ⇒ same block)."""
    return set(_tokens(text))


def compute_affected_set(
    mentions: Iterable[MentionView],
    seed_document_ids: Iterable[str],
    *,
    full: bool = False,
) -> AffectedSet:
    """The transitive blocking-neighborhood closure of the seed documents' mentions.

    The seeds are every mention in a changed/deleted document. The affected set grows to
    every mention sharing a blocking key with a seed (a merge-on-add candidate two blocks
    over), then again to *their* keys' mentions — the "affected blocks plus immediate
    neighbors" of ADR-0018. ``full=True`` short-circuits to the whole mention set (a
    Reconciliation). Deterministic: keys and ids are sets, materialised sorted by the caller.
    """
    mention_list = list(mentions)
    if full:
        return AffectedSet(
            mention_ids={m.mention_id for m in mention_list},
            blocking_keys={k for m in mention_list for k in _blocking_keys(m.text)},
            is_full=True,
        )

    seeds = set(seed_document_ids)
    # Index: blocking key -> mention ids; mention id -> keys.
    keys_by_mention: dict[str, set[str]] = {}
    mentions_by_key: dict[str, set[str]] = {}
    for m in mention_list:
        keys = _blocking_keys(m.text)
        keys_by_mention[m.mention_id] = keys
        for k in keys:
            mentions_by_key.setdefault(k, set()).add(m.mention_id)

    # Seed with mentions in the changed/deleted documents.
    affected_ids: set[str] = {m.mention_id for m in mention_list if m.document_id in seeds}
    affected_keys: set[str] = set()
    frontier = set(affected_ids)
    # Grow the neighborhood: a mention pulls in every mention that shares any of its keys, and
    # that pull-in is itself a frontier, so the closure catches a bridge two blocks away.
    while frontier:
        next_frontier: set[str] = set()
        for mid in frontier:
            for key in keys_by_mention.get(mid, set()):
                if key in affected_keys:
                    continue
                affected_keys.add(key)
                for neighbor in mentions_by_key.get(key, set()):
                    if neighbor not in affected_ids:
                        affected_ids.add(neighbor)
                        next_frontier.add(neighbor)
        frontier = next_frontier
    return AffectedSet(mention_ids=affected_ids, blocking_keys=affected_keys)


class IncrementalDisambiguator:
    """Affected-set entity re-resolution: merge-on-add + split-on-delete (recast IncrementalLinker).

    Wraps the ported :class:`~latence_core.disambiguation.EntityResolver` and applies it to only
    the affected blocking neighborhoods, splicing the result back onto the untouched clusters. The
    same resolver over the affected mentions catches BOTH cases the naive attach-to-known missed:

    * a bridging mention's tokens now connect two previously-separate blocks ⇒ the resolver unions
      them ⇒ **merge-on-add**;
    * a deleted document's mentions leave the affected set ⇒ re-resolving the survivors no longer
      finds the bridge ⇒ the cluster **splits**.

    The resolver kwargs mirror the reference Disambiguator so the incremental result is identical
    to a from-scratch resolution over the same affected mentions (order-independence + the drift
    reconciliation backstop guarantee the incremental corpus does not diverge from a full one).
    """

    def __init__(self, **resolver_kwargs: object) -> None:
        self._resolver_kwargs = resolver_kwargs

    def resolve_affected(
        self,
        current_mentions: Mapping[str, MentionView],
        affected: AffectedSet,
    ) -> list[ResolvedCluster]:
        """Re-resolve ONLY the affected mentions; return their fresh canonical clusters.

        ``current_mentions`` is the post-delta mention set keyed by id; ``affected`` names the
        subset to re-resolve. The returned clusters replace any prior cluster over the affected
        region; the orchestration keeps every unaffected cluster verbatim and splices these in
        (:class:`~latence_core.state.graph_patcher.GraphPatcher.add`). Resolving the affected set
        with the full resolver is what makes merge-on-add and split-on-delete exact.
        """
        resolver = EntityResolver(**self._resolver_kwargs)  # type: ignore[arg-type]
        resolver.add_all(
            [
                MentionInput(
                    mention_id=view.mention_id,
                    document_id=view.document_id,
                    text=view.text,
                    label=view.label,
                    confidence=view.confidence,
                )
                for mid, view in sorted(current_mentions.items())
                if mid in affected.mention_ids
            ]
        )
        clusters, _audits = resolver.resolve()
        return clusters
