"""Built-in reference DeltaProcessor Provider (deterministic, CPU-viable, zero-dep).

The S11 Delta reference Provider (ADR-0004/0018; ships in core like every other reference
Provider so the delta demo runs end to end, ADR-0007 CPU-first, ADR-0016 no extra deps). It
composes the ported :class:`~latence_core.state.graph_patcher.GraphPatcher` (retract by
Evidence provenance) with the blocking-neighborhood closure
(:func:`latence_core.blocking.closure`) to turn a parent Corpus Version +
the current run's freshly-extracted corpus-level records + a classified
:class:`~latence_core.contracts.Delta` into the next live derived set — surgically, so the
*committed* corpus-level set is proportional to the affected region, not the corpus (ADR-0018).

The scope boundary (issue #42): this processor makes the **committed corpus-level record set**
affected-set-scoped — untouched clusters are carried forward byte-identical and only the affected
neighborhood is re-resolved-and-spliced. Since W17/ADR-0043 the upstream *extraction* is also
incremental: the caller (:class:`~latence_core.delta_runner.DeltaRunner`) re-extracts only the
NEW/CHANGED documents and reuses the unchanged ones, then runs the corpus-level Stages over the
UNION — so ``current_records`` handed here are still the correct fully-reconciled truth (identical
to a from-scratch run), and this processor is unchanged. ``affected_set_size`` /
``corpus_record_count`` on the returned :class:`~latence_core.capability.DeltaOutcome` are the
measurable witness of the committed-side scoping.

How the affected-set splice works. The current run yields ``current_records`` for the WHOLE current
corpus (freshly-extracted new/changed documents disambiguated together with the reused ones), so
``current_records`` are the correct fully-reconciled truth (a bridging doc's
mentions were resolved together ⇒ merge-on-add; a deleted doc's mentions are absent ⇒
split-on-delete). But blindly replacing the parent set with the current set discards the
carry-forward and mis-reports the churn. Instead:

1. **retract** the parent records whose source-document reach touches a changed (added/updated)
   or deleted document — these are the parent's stale view of the affected region (ports
   ``graph/patcher``);
2. **keep** every other parent record verbatim (the untouched blocks — the lean win);
3. **splice in** exactly the current records in the affected blocking neighborhood — the
   *transitive* closure (:func:`latence_core.blocking.closure`) over shared blocking keys,
   seeded by the
   current records whose reach touches a changed document or that share a canonical surface with a
   retracted parent record (the merge-on-add bridge). The closure catches a bridge two blocks
   away, which a single-hop surface match misses. De-duplicated by ``record_id`` with the current
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

from latence_core.blocking import closure
from latence_core.capability import DeltaOutcome
from latence_core.contracts import (
    DisambiguationRecord,
    DisambiguationScope,
    GraphRecord,
    GraphScope,
    Record,
)
from latence_core.disambiguation import normalize_text
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)
from latence_core.state.graph_patcher import GraphPatcher, record_reach

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

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        The affected-set DeltaProcessor is the ported graph patcher + blocking recompute.
        A pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012
        — the package's own code, weights-and-code trivially the same), it ships no model weights
        (``model_id=None``), is deterministic, CPU-only, does not batch, and has no per-call cost,
        so a bake-off/conformance run reads honest, non-fabricated columns off it.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

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
            affected_current = _affected_current_records(
                current, affected_docs=affected_docs, retracted_surfaces=retracted_surfaces
            )
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
            # The perf-baseline witness (issue #42): how many corpus-level records the
            # blocking-neighborhood recompute actually re-resolved, out of the whole current
            # corpus-level set. On an incremental delta this is the affected-set closure
            # (<= the corpus); on a Reconciliation it is the whole corpus.
            affected_set_size=len(affected_current),
            corpus_record_count=len(current),
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


def _affected_current_records(
    current: list[Record],
    *,
    affected_docs: set[str],
    retracted_surfaces: set[str],
) -> list[Record]:
    """The current corpus-level records inside the affected blocking neighborhood (ADR-0018).

    Wires the shared blocking primitive (:func:`latence_core.blocking.closure`) onto the real
    delta path so the affected region is the genuine *transitive* closure over shared blocking
    keys — "the affected blocks plus their immediate neighbors" ADR-0018 promises — rather
    than a single-hop surface match. Each current corpus-level record enters the closure as
    one ``(record_id, blocking_text)`` pair, its blocking text the record's canonical
    surface(s); records that share a surface (⇒ shared tokens ⇒ shared blocking key) land in
    the same neighborhood, so a bridge two blocks away is caught.

    The seeds are the records whose provenance reach touches a changed/deleted document
    (``affected_docs``) OR that carry a surface the retraction removed (``retracted_surfaces``,
    the merge-on-add bridge). From those seeds the closure grows over shared blocking keys and
    returns exactly the current records in the neighborhood — a strict superset of the old
    one-hop match (so every merge-on-add / split-on-delete it caught is still caught), now
    genuinely affected-set-scoped rather than surface-equality-scoped.
    """
    items: list[tuple[str, str]] = []
    seed_ids: set[str] = set()
    for record in current:
        surfaces = _canonical_surfaces(record)
        # A record with no canonical surface (e.g. a relation carrier already split into its
        # head/tail surfaces, or an unscoped record) has empty blocking text — its neighborhood
        # is only itself; direct-reach seeding below still pulls it in when touched.
        items.append((record.record_id, " ".join(sorted(surfaces))))
        touched = _touches(record, affected_docs) or bool(surfaces & retracted_surfaces)
        if touched:
            seed_ids.add(record.record_id)
    if not seed_ids:
        return []
    affected_ids = closure(items, sorted(seed_ids))
    return [r for r in current if r.record_id in affected_ids]


def _touches(record: Record, docs: set[str]) -> bool:
    """True if a record's source-document reach intersects ``docs``."""
    if not docs:
        return False
    return bool(set(record_reach(record)) & docs)


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
