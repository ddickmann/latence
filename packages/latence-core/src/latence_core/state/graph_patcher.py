"""Graph patcher — the ported Evidence-provenance record retract/add engine (PORT-AS-IS, S11).

Ported faithfully from ``latenceai-dataset-intelligence/graph/patcher.py``
(ASSET-INVENTORY.md, 5/5, PORT-AS-IS). The algorithmic value is preserved verbatim:
**retract records by their source-document provenance**, so a deletion is surgical (touching
only the records a document actually contributed to) rather than a full-corpus rebuild. The
private stack retracted graph *edges* by ``source_document_ids``; here the same technique is
generalised to any derived :class:`~latence_core.contracts.Record` that carries a
document-reach signal, because S11 retracts across the whole derived set (canonical entities,
normalized relations, graph nodes/edges, chunks, mentions, redactions), not only edges.

The reach of a record — which source documents it was derived from — is read from, in order:

* an ``evidence.document_ids`` list (the CONTEXT ``Evidence`` cross-document reach carried by
  a :class:`~latence_core.contracts.CanonicalEntity`, :class:`~latence_core.contracts.GraphNode`,
  or :class:`~latence_core.contracts.GraphEdge`);
* a ``source_document_ids`` list (the affected-set key on a CanonicalEntity/GraphNode);
* otherwise the record's own ``provenance.document_id`` (a doc-level record — a chunk,
  mention, relation, or redaction — belongs to exactly one document).

**Retraction is a soft tombstone** (the S11 default deletion disposition): a retracted record
is removed from the *live* set but returned in :class:`PatchResult.retracted` so the caller
retains it for audit and can roll it back. **Purge** discards the retracted records instead of
retaining them (the caller does not persist them) — the physical-erasure path. A record whose
reach is *entirely* the deleted documents is retracted; a record that *also* draws on a
surviving document is retracted-and-re-derived by the affected-set recompute (it is returned in
``partially_affected`` so the orchestration re-resolves it rather than dropping it), which is
how a split-on-delete is caught (a merged cluster justified partly by a deleted bridging
document must be re-resolved, not blindly kept or blindly dropped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import Record


@dataclass
class PatchResult:
    """The outcome of a retract-by-provenance pass (the ported patcher result).

    ``kept`` are the records untouched by the deletion (no reach into any deleted document).
    ``retracted`` are records whose reach was *entirely* within the deleted documents — soft
    tombstones (retained for audit on a Retraction, discarded on a Purge). ``partially_affected``
    are records that drew on BOTH a deleted and a surviving document — they must be re-derived
    by the affected-set recompute (the split-on-delete case), never silently kept or dropped.
    """

    kept: list[Record] = field(default_factory=list)
    retracted: list[Record] = field(default_factory=list)
    partially_affected: list[Record] = field(default_factory=list)

    @property
    def retracted_count(self) -> int:
        return len(self.retracted)

    @property
    def affected_document_ids(self) -> list[str]:
        """Distinct surviving documents touched by a partially-affected record (sorted).

        These are the surviving documents whose derivations must be re-resolved after a
        deletion — the seed of the affected set for a split-on-delete.
        """
        seen: set[str] = set()
        for record in self.partially_affected:
            for doc_id in record_reach(record):
                seen.add(doc_id)
        return sorted(seen)


def _reach_of(obj: object) -> set[str]:
    """The reach signals directly on one object: its Evidence + source_document_ids + provenance."""
    reach: set[str] = set()
    evidence = getattr(obj, "evidence", None)
    if evidence is not None:
        for doc_id in getattr(evidence, "document_ids", []) or []:
            reach.add(doc_id)
    for doc_id in getattr(obj, "source_document_ids", []) or []:
        reach.add(doc_id)
    prov = getattr(obj, "provenance", None)
    if prov is not None and getattr(prov, "document_id", None):
        reach.add(prov.document_id)
    return reach


def record_reach(record: Record) -> list[str]:
    """Every source document a derived record draws on (its provenance reach).

    Reads the richest available reach signal: Evidence ``document_ids`` (a corpus-level
    record's cross-document justification) ∪ a ``source_document_ids`` field ∪ the record's
    own ``provenance.document_id`` (always present). A **scoped carrier**
    (:class:`~latence_core.contracts.DisambiguationRecord` /
    :class:`~latence_core.contracts.GraphRecord`) holds its Evidence and document reach on the
    inner payload (``entity``/``relation``/``node``/``edge``), not on the carrier itself, so those
    payloads are unwrapped and their reach unioned in — otherwise a canonical entity's
    cross-document reach would be invisible and the retract would be wrong. Deterministic (sorted).
    """
    reach = _reach_of(record)
    for attr in ("entity", "relation", "node", "edge"):
        payload = getattr(record, attr, None)
        if payload is not None:
            reach |= _reach_of(payload)
    return sorted(reach)


# Backwards-compatible alias: ``record_reach`` was born private and is imported under its
# old name by the corpus store and several delta test suites. Same function object.
_record_reach = record_reach


class GraphPatcher:
    """Retract/add derived records by their source-document provenance (ported, generalised).

    Pure and stateless: :meth:`retract` partitions a derived record set against a set of
    deleted document ids into kept / retracted / partially-affected, and :meth:`add` splices
    a re-derived record set back in (replacing any record for the same affected documents).
    Never touches Storage or re-runs a Stage — it is the surgical set algebra the delta
    orchestration drives, keeping the retract/add cost proportional to the delta, not the
    corpus (the ADR-0018 "surgical via Evidence provenance" property).
    """

    @staticmethod
    def retract(records: Iterable[Record], deleted_document_ids: Iterable[str]) -> PatchResult:
        """Partition ``records`` against the deleted documents (the ported retract-by-reach).

        A record whose reach is a subset of the deleted set is fully retracted; a record whose
        reach intersects the deleted set but also includes a surviving document is
        partially-affected (re-derive it); a record with no reach into the deleted set is kept.
        """
        deleted = set(deleted_document_ids)
        result = PatchResult()
        if not deleted:
            result.kept = list(records)
            return result
        for record in records:
            reach = set(record_reach(record))
            hit = reach & deleted
            if not hit:
                result.kept.append(record)
            elif reach <= deleted:
                result.retracted.append(record)
            else:
                result.partially_affected.append(record)
        return result

    @staticmethod
    def add(kept: list[Record], rederived: list[Record]) -> list[Record]:
        """Splice re-derived records back into the kept set (the ported add step).

        The kept records are those the delta did not touch; ``rederived`` are the affected-set
        recompute's fresh records for the affected region. Their concatenation is the patched
        derived set. The caller (the orchestration) is responsible for de-duplicating by
        ``record_id`` upstream; this is the pure splice.
        """
        return [*kept, *rederived]
