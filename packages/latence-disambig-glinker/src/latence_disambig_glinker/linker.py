"""EntityLinker — orchestrates entity linking via the GLinker pipeline.

Ported (PORT-AS-IS) from ``dataset_intelligence.linking.linker``. Architecture:

  1. Build a self-referencing entity KB from pipeline mentions (canonical entries + aliases)
  2. Warm the backend — load KB + precompute BiEncoder label embeddings
  3. Group mentions by unique text (avoid redundant inference), feed each a ±window context
  4. Run the full GLinker pipeline (L2->L3->L4->L0) for ALL unique texts
  5. Fall back to a direct KB text match for mentions the pipeline misses
  6. Produce canonical entity clusters with full provenance

DI adaptation vs the source: the ±window context is precomputed by the Provider from the run's
ChunkRecords and carried on :attr:`MentionRecord.context` (the source read it from document
markdown), so this module consumes ``m.context`` directly — the linking ALGORITHM is unchanged.
Backend load/inference failure degrades to the direct-KB fallback (never a crash) — the enterprise
robustness requirement.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import Counter, defaultdict
from typing import Any

from latence_disambig_glinker.backends import (
    GLinkerBackend,
    InferenceBackend,
    LinkedEntity,
)
from latence_disambig_glinker.ids import make_entity_id
from latence_disambig_glinker.kb_builder import build_entity_kb, kb_to_glinker_format
from latence_disambig_glinker.models import CanonicalEntityRecord, MentionRecord

logger = logging.getLogger(__name__)


class EntityLinker:
    """Orchestrates entity linking across a corpus via the GLinker L2->L3->L4->L0 pipeline.

    Args:
        backend: inference backend (defaults to :class:`GLinkerBackend`).
        threshold: minimum linking confidence.
        batch_size: batch size for inference.
    """

    def __init__(
        self,
        backend: InferenceBackend | None = None,
        threshold: float = 0.3,
        batch_size: int = 32,
    ) -> None:
        self.backend: InferenceBackend = backend if backend is not None else GLinkerBackend()
        self.threshold = threshold
        self.batch_size = batch_size
        self._entity_kb: list[dict[str, Any]] = []

    def link_dataset(
        self,
        mentions: list[MentionRecord],
        dataset_id: str,
    ) -> tuple[list[MentionRecord], list[CanonicalEntityRecord]]:
        """Run entity linking on the whole corpus.

        Returns ``(updated_mentions, canonical_entities)``.
        """
        t0 = time.monotonic()

        # Step 1: Build entity KB from mentions.
        logger.info("Step 1/5: Building entity KB from %d mentions", len(mentions))
        self._entity_kb = build_entity_kb(
            mentions, dataset_id, min_mentions=1, min_score=self.threshold
        )
        logger.info("Entity KB: %d entries", len(self._entity_kb))

        glinker_kb: list[dict[str, Any]] = kb_to_glinker_format(self._entity_kb)

        # Step 2: Group mentions by unique text (avoid redundant inference).
        text_groups: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(mentions):
            key = m.text.strip().lower()
            text_groups[key].append(i)
        logger.info("%d unique mention texts across %d mentions", len(text_groups), len(mentions))

        # Step 3: Build a representative ±window-context input per unique text.
        representative_inputs: list[dict[str, object]] = []
        representative_keys: list[str] = []
        for key, indices in text_groups.items():
            best_idx = max(indices, key=lambda i: mentions[i].score)
            m = mentions[best_idx]
            representative_inputs.append(
                {
                    "text": m.text,
                    "label": m.label,
                    "start": m.start,
                    "end": m.end,
                    "mention_id": m.mention_id,
                    "document_id": m.document_id,
                    # Context precomputed by the Provider from ChunkRecords (±window); the source
                    # read it from doc markdown. Falls back to the surface if no context located.
                    "context": m.context or m.text,
                }
            )
            representative_keys.append(key)

        # Step 4: Neural linking for all unique texts (graceful backend-failure fallback).
        logger.info("Step 2/5: Neural entity linking for %d unique texts", len(representative_keys))
        neural_results = self._safe_neural_link(representative_inputs, glinker_kb)

        neural_map: dict[str, LinkedEntity] = {}
        for key, result in zip(representative_keys, neural_results, strict=True):
            if result is not None:
                neural_map[key] = result
        logger.info(
            "Neural linking: %d/%d unique texts resolved", len(neural_map), len(representative_keys)
        )

        # Step 5: Direct KB text-match fallback for unresolved mentions.
        logger.info("Step 3/5: Direct KB matching for unresolved mentions")
        kb_by_text: dict[str, dict[str, Any]] = {}
        for entry in self._entity_kb:
            key = str(entry["label"]).strip().lower()
            kb_by_text[key] = entry
            for alias in entry.get("aliases", []) or []:
                alias_key = str(alias).strip().lower()
                kb_by_text.setdefault(alias_key, entry)

        linked_results: list[LinkedEntity | None] = [None] * len(mentions)
        neural_count = direct_count = unresolved_count = 0
        for key, indices in text_groups.items():
            if key in neural_map:  # Priority 1: neural GLinker result.
                template = neural_map[key]
                for idx in indices:
                    linked_results[idx] = copy.copy(template)
                neural_count += len(indices)
            elif key in kb_by_text:  # Priority 2: direct KB text match.
                kb_entry = kb_by_text[key]
                le = LinkedEntity(
                    entity_id=str(kb_entry["entity_id"]),
                    label=str(kb_entry["label"]),
                    description=str(kb_entry.get("description", "")),
                    score=min(1.0, float(kb_entry.get("_avg_score", 0.5) or 0.5)),
                    entity_type=str(kb_entry.get("entity_type", "")),
                    aliases=list(kb_entry.get("aliases", []) or []),
                    source="kb_direct",
                )
                for idx in indices:
                    linked_results[idx] = copy.copy(le)
                direct_count += len(indices)
            else:
                unresolved_count += len(indices)

        logger.info(
            "Assignment: neural=%d, direct=%d, unresolved=%d",
            neural_count,
            direct_count,
            unresolved_count,
        )

        # Step 6: Apply results to mentions + build canonical entities.
        updated_mentions = _apply_linking_results(mentions, linked_results)
        canonical_entities = self._build_canonical_entities(updated_mentions, dataset_id)

        logger.info(
            "Entity linking complete: %d canonical entities from %d mentions in %.2fs (backend=%s)",
            len(canonical_entities),
            len(mentions),
            time.monotonic() - t0,
            self.backend.name,
        )
        return updated_mentions, canonical_entities

    def _safe_neural_link(
        self,
        representative_inputs: list[dict[str, object]],
        glinker_kb: list[dict[str, Any]],
    ) -> list[LinkedEntity | None]:
        """Warmup + neural link, degrading to all-unresolved on any backend fault (never crash).

        The enterprise robustness path (S8 AC / step 3): a backend that cannot load its model (no
        GPU, missing weights, import error) yields an all-``None`` result so every mention flows to
        the deterministic direct-KB fallback + string-cascade resolver, rather than raising.
        """
        try:
            self.backend.warmup(glinker_kb)
            return self.backend.link_mentions(
                representative_inputs,
                glinker_kb,
                threshold=self.threshold,
                batch_size=self.batch_size,
            )
        except Exception as e:  # noqa: BLE001 - backend fault -> direct-KB fallback (never crash)
            logger.warning(
                "Neural backend unavailable (%s: %s) — falling back to direct-KB linking",
                type(e).__name__,
                e,
            )
            return [None] * len(representative_inputs)

    def _build_canonical_entities(
        self,
        mentions: list[MentionRecord],
        dataset_id: str,
    ) -> list[CanonicalEntityRecord]:
        """Build canonical entity clusters from linked mentions (provisional for unresolved)."""
        clusters: dict[str, list[MentionRecord]] = defaultdict(list)
        for m in mentions:
            if m.canonical_entity_id:
                clusters[m.canonical_entity_id].append(m)
            else:
                prov_id = make_entity_id(dataset_id, m.text, m.label)
                clusters[prov_id].append(m)

        entities: list[CanonicalEntityRecord] = []
        for entity_id, cluster in clusters.items():
            dominant_label = Counter(m.label for m in cluster).most_common(1)[0][0]
            canonical_name = Counter(m.text for m in cluster).most_common(1)[0][0]
            aliases = sorted({m.text for m in cluster} - {canonical_name})
            mention_ids = [m.mention_id for m in cluster]
            document_ids = sorted({m.document_id for m in cluster})
            link_confidences = {
                m.mention_id: m.link_confidence
                for m in cluster
                if m.link_confidence is not None
            }
            confidence = sum(m.score for m in cluster) / len(cluster)
            entities.append(
                CanonicalEntityRecord(
                    entity_id=entity_id,
                    dataset_id=dataset_id,
                    canonical_name=canonical_name,
                    label=dominant_label,
                    aliases=aliases,
                    mention_ids=mention_ids,
                    document_ids=document_ids,
                    evidence_count=len(cluster),
                    confidence=round(confidence, 4),
                    link_confidences=link_confidences,
                    source="resolved" if len(cluster) > 1 else "single_mention",
                )
            )
        entities.sort(key=lambda e: e.evidence_count, reverse=True)
        return entities


def _apply_linking_results(
    mentions: list[MentionRecord],
    linked: list[LinkedEntity | None],
) -> list[MentionRecord]:
    """Apply linking results back to mentions (new list; originals not mutated).

    Ported from ``dataset_intelligence.linking.adapter.apply_linking_results``.
    """
    updated: list[MentionRecord] = []
    linked_count = 0
    for mention, result in zip(mentions, linked, strict=True):
        if result is not None:
            updated.append(
                mention.model_copy(
                    update={
                        "canonical_entity_id": result.entity_id,
                        # v20: persist the link score the linker always computed, so the
                        # canonical entity carries a per-mention link confidence.
                        "link_confidence": min(1.0, max(0.0, float(result.score))),
                    }
                )
            )
            linked_count += 1
        else:
            updated.append(mention.model_copy())
    logger.info(
        "Linking results applied: %d linked, %d unresolved",
        linked_count,
        len(mentions) - linked_count,
    )
    return updated
