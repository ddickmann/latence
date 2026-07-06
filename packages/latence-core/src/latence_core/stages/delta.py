"""Built-in reference DeltaProcessor Provider (deterministic, CPU-viable, zero-dep).

The S11 Delta reference Provider (ADR-0004/0018; ships in core like every other reference
Provider so the delta demo runs end to end, ADR-0007 CPU-first, ADR-0016 no extra deps). It
composes the ported :class:`~latence_core.state.graph_patcher.GraphPatcher` (retract by
Evidence provenance) with the affected-set recompute to turn a parent Corpus Version + the
current run's freshly-extracted corpus-level records + a classified
:class:`~latence_core.contracts.Delta` into the next live derived set — surgically, so the cost
is proportional to the delta, not the corpus (ADR-0018).

How the affected-set splice works. The current run re-extracts and re-disambiguates the WHOLE
current corpus, so ``current_records`` are the correct fully-reconciled truth (a bridging doc's
mentions were resolved together ⇒ merge-on-add; a deleted doc's mentions are absent ⇒
split-on-delete). But blindly replacing the parent set with the current set is the O(corpus)
recompute ADR-0018 rejects. Instead:

1. **retract** the parent records whose source-document reach touches a changed (added/updated)
   or deleted document — these are the parent's stale view of the affected region (ports
   ``graph/patcher``);
2. **keep** every other parent record verbatim (the untouched blocks — the lean win);
3. **splice in** exactly the current records that cover the affected region (a current record
   whose reach touches a changed document, or that shares a canonical surface with a retracted
   parent record — the merge-on-add bridge), de-duplicated by ``record_id`` with the current
   record winning.

On a **Reconciliation** (the Delta's ``triggered_reconciliation``), the whole current set is
the affected region — every parent record is retracted and the current set replaces it — the
bounded-error backstop. The churn (entities created / merged / split, edges ±) is measured by
comparing the affected current entities against the retracted parent entities, so the Quality
Report reports what actually changed (the S11 AC).

Determinism: every partition and splice iterates sorted ``record_id``s, so a seeded corpus
state yields the same live set and the same churn (the Baseline bar).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.capability import DeltaOutcome
from latence_core.contracts import (
    DisambiguationRecord,
    DisambiguationScope,
    GraphRecord,
    GraphScope,
    Record,
)
from latence_core.disambiguation import normalize_text
from latence_core.state.graph_patcher import GraphPatcher, _record_reach

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import Delta


class AffectedSetDeltaProcessor:
    """DeltaProcessor Provider: retract-by-provenance + affected-set splice + churn counts.

    Config:
        (none required) — the processor is pure over records; the resolver/blocking config lives
        on the Disambiguation Stage that produced the records. Accepts and ignores a config dict
        so it satisfies the standard ``Provider(config)`` construction the Runner uses.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def apply_delta(
        self,
        parent_records: Iterable[Record],
        current_records: Iterable[Record],
        delta: Delta,
    ) -> DeltaOutcome:
        parent = list(parent_records)
        current = list(current_records)

        # The documents whose parent derivations are stale: changed (added/updated) + deleted.
        # (Added docs have no parent records, but including them keeps the reach test uniform.)
        touched_docs = set(delta.changed) | set(delta.deleted)

        if delta.triggered_reconciliation:
            # Full Reconciliation: every parent record is retracted, the current set is the
            # whole affected region. The bounded-error backstop.
            retracted = sorted(parent, key=lambda r: r.record_id)
            live = sorted(current, key=lambda r: r.record_id)
            affected_current = current
            kept_parent: list[Record] = []
        else:
            patch = GraphPatcher.retract(parent, touched_docs)
            # A partially-affected parent record (drew on both a deleted and a surviving doc) is
            # also stale — its surviving-side derivation is re-covered by the current set — so it
            # is retracted here and its surviving documents seed the affected region below.
            retracted = sorted(
                [*patch.retracted, *patch.partially_affected], key=lambda r: r.record_id
            )
            kept_parent = patch.kept
            # Surviving documents whose parent cluster was split by a deletion (split-on-delete):
            # the affected region must also pull current records for those documents.
            affected_docs = touched_docs | set(patch.affected_document_ids)
            # The canonical surfaces the retraction removed — a current record sharing one is the
            # merge-on-add bridge that reunites the region.
            retracted_surfaces = {
                surf for r in retracted for surf in _canonical_surfaces(r)
            }
            affected_current = [
                r
                for r in current
                if _touches(r, affected_docs)
                or (_canonical_surfaces(r) & retracted_surfaces)
            ]
            live = self._splice(kept_parent, affected_current)

        # The churn baseline is the parent's view of the affected region: every parent entity
        # whose canonical surface reappears in the affected current set (whether it was formally
        # retracted or kept-but-superseded by the splice). Comparing against only the retracted
        # set would miss a merge-on-add where the smaller parent cluster was kept (its reach did
        # not touch the deleted docs) yet is superseded by a larger current cluster of the same
        # surface — the merge must still be counted.
        current_surfaces = {
            surf for r in affected_current for surf in _canonical_surfaces(r)
        }
        parent_baseline = [
            r for r in parent if _canonical_surfaces(r) & current_surfaces
        ] + retracted
        entities_created, entities_merged, entities_split = _entity_churn(
            parent_baseline=parent_baseline, affected_current=affected_current
        )
        edges_added, edges_retracted = _edge_churn(
            all_parent=parent, retracted=retracted, affected_current=affected_current
        )
        return DeltaOutcome(
            records=live,
            retracted=retracted,
            entities_created=entities_created,
            entities_merged=entities_merged,
            entities_split=entities_split,
            edges_added=edges_added,
            edges_retracted=edges_retracted,
        )

    @staticmethod
    def _splice(kept_parent: list[Record], affected_current: list[Record]) -> list[Record]:
        """Splice the affected current records onto the kept parent set, de-duped by record_id.

        The current record wins on an id collision (it is the fresh derivation). The result is
        sorted by ``record_id`` for a byte-stable committed set.
        """
        merged: dict[str, Record] = {r.record_id: r for r in kept_parent}
        for record in affected_current:
            merged[record.record_id] = record
        return [merged[key] for key in sorted(merged)]


def _touches(record: Record, docs: set[str]) -> bool:
    """True if a record's source-document reach intersects ``docs``."""
    if not docs:
        return False
    return bool(set(_record_reach(record)) & docs)


def _canonical_surfaces(record: Record) -> set[str]:
    """The normalised canonical surface(s) a corpus-level record carries (for bridge matching).

    A :class:`~latence_core.contracts.DisambiguationRecord` ENTITY / :class:`GraphRecord` NODE
    carries a canonical text; a relation/edge carries head+tail surfaces. Empty for anything
    else. Used to find the merge-on-add bridge: a current record sharing a surface with a
    retracted parent record is part of the reunited region.
    """
    surfaces: set[str] = set()
    if isinstance(record, DisambiguationRecord):
        if record.scope is DisambiguationScope.ENTITY and record.entity is not None:
            surfaces.add(normalize_text(record.entity.canonical_text))
        elif record.scope is DisambiguationScope.RELATION and record.relation is not None:
            surfaces.add(normalize_text(record.relation.head_text))
            surfaces.add(normalize_text(record.relation.tail_text))
    elif (
        isinstance(record, GraphRecord)
        and record.scope is GraphScope.NODE
        and record.node is not None
    ):
        surfaces.add(normalize_text(record.node.canonical_name))
    return surfaces


def _entity_records(records: list[Record]) -> list[DisambiguationRecord]:
    """The ENTITY-scope DisambiguationRecords in a record set (the canonical entities)."""
    return [
        r
        for r in records
        if isinstance(r, DisambiguationRecord)
        and r.scope is DisambiguationScope.ENTITY
        and r.entity is not None
    ]


def _entity_churn(
    *, parent_baseline: list[Record], affected_current: list[Record]
) -> tuple[int, int, int]:
    """Count entities created / merged / split by comparing affected current vs the parent baseline.

    Keyed by normalised canonical surface, using each entity's member-mention count as the
    cluster size (the baseline keeps the LARGEST parent cluster per surface, so a merge that
    absorbed two smaller parent clusters of the same surface is still measured against the bigger):

    * **created** — a canonical surface present in the affected current set but not in the parent
      baseline (a genuinely new entity from an added document);
    * **merged** — a surface whose current cluster is LARGER than the parent's (mentions folded
      in from a bridging document — merge-on-add);
    * **split** — a surface whose current cluster is SMALLER than the parent's (a bridging
      document's mentions left — split-on-delete).
    """
    parent: dict[str, int] = {}
    for r in _entity_records(parent_baseline):
        if r.entity is None:
            continue
        surface = normalize_text(r.entity.canonical_text)
        parent[surface] = max(parent.get(surface, 0), r.entity.mention_count)
    current = {
        normalize_text(r.entity.canonical_text): r.entity.mention_count
        for r in _entity_records(affected_current)
        if r.entity is not None
    }
    created = merged = split = 0
    for surface in sorted(current):
        size = current[surface]
        if surface not in parent:
            created += 1
        elif size > parent[surface]:
            merged += 1
        elif size < parent[surface]:
            split += 1
    return created, merged, split


def _edge_ids(records: list[Record]) -> set[str]:
    return {
        r.edge.edge_id
        for r in records
        if isinstance(r, GraphRecord) and r.scope is GraphScope.EDGE and r.edge is not None
    }


def _edge_churn(
    *, all_parent: list[Record], retracted: list[Record], affected_current: list[Record]
) -> tuple[int, int]:
    """Count graph edges added vs retracted (by edge_id).

    ``edges_added`` — edges in the affected current set that no parent edge already had (a fresh
    edge from an added/re-resolved region). ``edges_retracted`` — edges the delta retracted that
    the current set does not re-derive (a graph edge whose justification was deleted).
    """
    parent_edges = _edge_ids(all_parent)
    retracted_edges = _edge_ids(retracted)
    current_edges = _edge_ids(affected_current)
    edges_added = len(current_edges - parent_edges)
    edges_retracted = len(retracted_edges - current_edges)
    return edges_added, edges_retracted
