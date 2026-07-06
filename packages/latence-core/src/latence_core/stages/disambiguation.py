"""Built-in reference Disambiguator Provider (deterministic, CPU-viable, zero-dep).

The S8 Disambiguation reference Provider (ADR-0004/0026; ships in core like the S3
Chunker/Screeners, the S4 gazetteer, the S5 trigger extractor, the S6 hybrid redactor
and the S7 profiler so the demo runs end to end, ADR-0007 CPU-first, ADR-0016 no extra
deps). It wraps the ported disambiguation engine (:mod:`latence_core.disambiguation` —
the 5-strategy EntityResolver cascade + Union-Find + audited merge log, the GLinker-style
KB linking cascade with graceful fallback, and the RelationNormalizer) and turns the run's
entity mentions (+ optional relations) into typed
:class:`~latence_core.contracts.DisambiguationRecord`s:

* one ``ENTITY``-scope record per resolved
  :class:`~latence_core.contracts.CanonicalEntity` — the cross-document merged cluster, its
  audited merge decisions (no silent over-merge), its optional external-KB link (graceful
  fallback when unlinked), its :class:`~latence_core.contracts.Evidence`, and its
  page-accurate Provenance (inherited from the canonical mention, already resolved through
  :class:`~latence_core.pagemap.PageOffsetIndex` upstream — the S8 page-map-integration
  sub-task);
* one ``RELATION``-scope record per :class:`~latence_core.contracts.NormalizedRelation` —
  the relation remapped onto the canonical entities, its label normalised and its direction
  canonicalised.

Determinism: entities are emitted in canonical-text order and relations after them in
source-relation order; every member/document list is sorted by the engine — so a seeded run
is byte-identical (Baseline bar). A merge below the confidence policy is recorded on the
entity's ``merge_decisions`` with ``applied=False`` but never folded — the S8 "no silent
over-merge" acceptance criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    CanonicalEntity,
    DisambiguationRecord,
    DisambiguationScope,
    Evidence,
    MergeDecision,
    MergeStrategy,
    NormalizedRelation,
    Provenance,
)
from latence_core.disambiguation import (
    EntityResolver,
    KBEntry,
    KnowledgeBase,
    MentionInput,
    RelationInput,
    RelationNormalizer,
    ResolvedCluster,
    normalize_text,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import EntityMention, RelationMention
    from latence_core.disambiguation import MergeAudit

# The engine Strategy enum values line up 1:1 with the contract MergeStrategy values, so a
# lookup by value maps an audit to its contract enum without a hand-written table.
_STRATEGY_BY_VALUE = {s.value: s for s in MergeStrategy}


class CascadeDisambiguator:
    """Disambiguator Provider: mentions (+ relations) → canonical entities + normalized relations.

    Config:
        aliases: dict[str, str]  — surface → surface alias map for the ALIAS strategy.
        min_confidence: float  — the merge confidence policy; a proposed merge below it is
            logged (applied=False) but NOT folded (no silent over-merge, S8 AC). Default 0.82.
        alias_confidence / acronym_confidence: float  — confidence stamped by those strategies.
        embedding_threshold: float  — token-similarity cutoff for the (CPU lexical) EMBEDDING
            strategy. Default 0.75.
        min_substring_ratio: float  — length-ratio gate for the SUBSTRING strategy. Default 0.5.
        enable_embedding: bool  — turn the lexical EMBEDDING strategy off for the strict
            no-embedder path (a real embedder Provider is the pluggable upgrade). Default True.
        require_type_compatibility: bool  — gate the acronym + substring rungs on a
            same-entity-type compatibility check, so those low-precision strategies do not
            merge different-typed surfaces ("IT" ↔ "Information Technology", issue #35).
            Default True (safe: never blocks a same-typed cross-document merge).
        require_shared_document: bool  — gate the acronym + substring rungs on shared-document
            evidence (both surfaces appear in a common document), the original asset's stricter
            evidence requirement. Default False (opt-in: it also blocks the legitimate
            cross-document merges the Stage exists to make).
        kb: list[dict]  — the external knowledge base for linking: each entry is
            ``{"kb_id", "name", "entity_type"?, "aliases"?}``. Empty/omitted ⇒ every entity is
            unlinked (the graceful fallback, S8 AC).
        kb_threshold: float  — L0 acceptance threshold for a KB link. Default 0.5.
        label_aliases / inverses: dict[str, str]  — relation-label normalisation + inverse
            direction-swap maps for the RelationNormalizer.
        allowed_labels: list[str]  — when set, relations whose normalised label is not in the
            set are dropped (type filter) and counted.
        label_threshold: float  — fuzzy label-match cutoff. Default 0.6.
        corpus_sensitivity: str  — sensitivity stamped on the synthetic corpus-scope entity
            Provenance fallback (used only when an entity has no member mention Provenance —
            never in practice, since a cluster always has ≥1 mention). Default ``"internal"``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._resolver_kwargs: dict[str, Any] = {
            "aliases": _as_str_map(config.get("aliases")),
            "min_confidence": float(config.get("min_confidence", 0.82)),
            "alias_confidence": float(config.get("alias_confidence", 0.95)),
            "acronym_confidence": float(config.get("acronym_confidence", 0.9)),
            "embedding_threshold": float(config.get("embedding_threshold", 0.75)),
            "min_substring_ratio": float(config.get("min_substring_ratio", 0.5)),
            "enable_embedding": bool(config.get("enable_embedding", True)),
            "require_type_compatibility": bool(
                config.get("require_type_compatibility", True)
            ),
            "require_shared_document": bool(config.get("require_shared_document", False)),
        }
        self._kb = self._build_kb(config)
        self._normalizer = RelationNormalizer(
            label_aliases=_as_str_map(config.get("label_aliases")),
            inverses=_as_str_map(config.get("inverses")),
            allowed_labels=(
                {str(x) for x in config["allowed_labels"]}
                if config.get("allowed_labels")
                else None
            ),
            label_threshold=float(config.get("label_threshold", 0.6)),
        )

    @staticmethod
    def _build_kb(config: dict[str, Any]) -> KnowledgeBase:
        entries: list[KBEntry] = []
        for raw in config.get("kb", []) or []:
            entries.append(
                KBEntry(
                    kb_id=str(raw["kb_id"]),
                    name=str(raw["name"]),
                    entity_type=(
                        str(raw["entity_type"]) if raw.get("entity_type") is not None else None
                    ),
                    aliases=tuple(str(a) for a in raw.get("aliases", [])),
                )
            )
        return KnowledgeBase(entries, threshold=float(config.get("kb_threshold", 0.5)))

    def disambiguate(
        self, mentions: Iterable[EntityMention], relations: Iterable[RelationMention]
    ) -> Iterator[DisambiguationRecord]:
        mention_list = list(mentions)
        # Index each mention by record_id so a canonical entity can inherit the page-accurate
        # Provenance of its chosen representative mention (S8 page-map integration).
        by_id = {m.record_id: m for m in mention_list}

        resolver = EntityResolver(**self._resolver_kwargs)
        resolver.add_all(
            [
                MentionInput(
                    mention_id=m.record_id,
                    document_id=m.provenance.document_id,
                    text=m.text,
                    label=m.label,
                    confidence=m.confidence,
                )
                for m in mention_list
            ]
        )
        clusters, _audits = resolver.resolve()

        # Build the mention_id -> canonical entity record_id map (for relation remapping),
        # and emit the ENTITY-scope records first (canonical-text order, from the engine).
        entity_id_by_mention: dict[str, str] = {}
        entity_text_by_id: dict[str, str] = {}
        entity_records: list[DisambiguationRecord] = []
        for cluster in clusters:
            record = self._entity_record(cluster, by_id)
            entity = record.entity
            assert entity is not None  # ENTITY scope always carries an entity
            for mention_id in cluster.member_mention_ids:
                entity_id_by_mention[mention_id] = entity.record_id
            entity_text_by_id[entity.record_id] = entity.canonical_text
            entity_records.append(record)

        yield from entity_records

        # RELATION-scope records: normalise each relation's label/direction and remap its
        # endpoints to canonical entities. A relation whose head or tail did not resolve to a
        # canonical entity, or that the type filter drops, is skipped (not a broken relation).
        for relation in relations:
            relation_record = self._relation_record(
                relation, entity_id_by_mention, entity_text_by_id
            )
            if relation_record is not None:
                yield relation_record

    # -- entity records -----------------------------------------------------

    def _entity_record(
        self, cluster: ResolvedCluster, by_id: dict[str, EntityMention]
    ) -> DisambiguationRecord:
        # The representative mention: the first member (member ids are engine-sorted) that
        # still exists, so Provenance/Classification inherit a real, page-accurate mention.
        representative = next(
            (by_id[mid] for mid in cluster.member_mention_ids if mid in by_id), None
        )
        entity_id = f"entity:{normalize_text(cluster.canonical_text)}"
        evidence = Evidence(
            mention_ids=list(cluster.member_mention_ids),
            document_ids=list(cluster.document_ids),
            snippet=f"merged {len(cluster.member_mention_ids)} mention(s) as "
            f"{cluster.canonical_text!r}",
            confidence=cluster.confidence,
        )
        # Link the canonical surface to the external KB (L2→L3/L4→L0); unlinked ⇒ fallback.
        link = self._kb.link(cluster.canonical_text, cluster.entity_type)
        provenance = self._entity_provenance(representative)
        classification = (
            representative.classification if representative is not None
            else _corpus_classification()
        )
        entity = CanonicalEntity(
            record_id=entity_id,
            canonical_text=cluster.canonical_text,
            entity_type=cluster.entity_type,
            member_mention_ids=list(cluster.member_mention_ids),
            source_document_ids=list(cluster.document_ids),
            kb_id=link.kb_id if link.linked else None,
            merge_decisions=[_merge_decision(a, cluster) for a in cluster.merge_audits],
            evidence=evidence,
            provenance=provenance,
            classification=classification,
        )
        return DisambiguationRecord(
            record_id=f"{entity_id}#canonical",
            scope=DisambiguationScope.ENTITY,
            entity=entity,
            provenance=provenance,
            classification=classification,
        )

    @staticmethod
    def _entity_provenance(representative: EntityMention | None) -> Provenance:
        """The canonical entity's page-accurate Provenance: the representative mention's.

        The mention's Provenance already carries its ``char_start``/``char_end`` and its
        source ``page_start``/``page_end`` (resolved through PageOffsetIndex at S4), so a
        canonical entity cites the exact page of its representative mention — the S8
        page-map-integration sub-task. A cluster always has a member mention; the synthetic
        corpus fallback is defensive only.
        """
        if representative is not None:
            return representative.provenance
        return Provenance(
            source_uri="latence:corpus",
            file_name="corpus",
            file_type="corpus",
            file_size=0,
            document_id="latence:corpus",
            char_start=0,
            char_end=0,
        )

    # -- relation records ---------------------------------------------------

    def _relation_record(
        self,
        relation: RelationMention,
        entity_id_by_mention: dict[str, str],
        entity_text_by_id: dict[str, str],
    ) -> DisambiguationRecord | None:
        normalized = self._normalizer.normalize(
            RelationInput(
                relation_id=relation.record_id,
                document_id=relation.provenance.document_id,
                label=relation.label,
                head_mention_id=relation.head_mention_id,
                tail_mention_id=relation.tail_mention_id,
                head_text=relation.head_text,
                tail_text=relation.tail_text,
                confidence=relation.confidence,
            )
        )
        if normalized is None:
            return None  # dropped by the type filter
        head_entity = entity_id_by_mention.get(normalized.head_mention_id)
        tail_entity = entity_id_by_mention.get(normalized.tail_mention_id)
        # Both endpoints must resolve to a canonical entity, and to DISTINCT ones — a
        # relation whose head/tail collapsed into the same entity is a self-loop and dropped.
        if head_entity is None or tail_entity is None or head_entity == tail_entity:
            return None
        evidence = Evidence(
            mention_ids=[normalized.head_mention_id, normalized.tail_mention_id],
            document_ids=[relation.provenance.document_id],
            snippet=f"{normalized.label}: {entity_text_by_id[head_entity]!r} -> "
            f"{entity_text_by_id[tail_entity]!r}",
            confidence=normalized.confidence,
        )
        normalized_relation = NormalizedRelation(
            record_id=f"{relation.record_id}#normalized",
            source_relation_id=relation.record_id,
            label=normalized.label,
            head_entity_id=head_entity,
            tail_entity_id=tail_entity,
            head_text=entity_text_by_id[head_entity],
            tail_text=entity_text_by_id[tail_entity],
            confidence=normalized.confidence,
            inverted=normalized.inverted,
            evidence=evidence,
            provenance=relation.provenance,
            classification=relation.classification,
        )
        return DisambiguationRecord(
            record_id=f"{relation.record_id}#normalized-record",
            scope=DisambiguationScope.RELATION,
            relation=normalized_relation,
            provenance=relation.provenance,
            classification=relation.classification,
        )


def _merge_decision(audit: MergeAudit, cluster: ResolvedCluster) -> MergeDecision:
    """Map an engine MergeAudit to the contract MergeDecision, with its Evidence."""
    evidence = Evidence(
        mention_ids=list(cluster.member_mention_ids),
        document_ids=list(cluster.document_ids),
        snippet=audit.reason,
        confidence=audit.confidence,
    )
    return MergeDecision(
        source_text=audit.source_text,
        target_text=audit.target_text,
        strategy=_STRATEGY_BY_VALUE[audit.strategy.value],
        confidence=audit.confidence,
        applied=audit.applied,
        reason=audit.reason,
        evidence=evidence,
    )


def _corpus_classification() -> Any:
    from latence_core.contracts import Classification

    return Classification(language="und", category="corpus", sensitivity="internal")


def _as_str_map(value: Any) -> dict[str, str]:
    """Coerce a config mapping into a ``dict[str, str]`` (empty for None)."""
    if not value:
        return {}
    return {str(k): str(v) for k, v in dict(value).items()}
