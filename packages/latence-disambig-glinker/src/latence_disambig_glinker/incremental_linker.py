"""Incremental entity linker — links new mentions against an existing KB plus delta entities.

Ported (PORT-AS-IS) from ``dataset_intelligence.linking.incremental_linker``. During incremental
ingestion, new documents produce new mentions that must be linked against the *existing* canonical
entity KB (from prior ingestions) plus any truly new entities discovered in the delta batch — this
avoids rebuilding the full KB from scratch every time. Pairs with
:meth:`EntityResolver.resolve_incremental` for the resolution side.
"""

from __future__ import annotations

import logging

from latence_disambig_glinker.backends import InferenceBackend
from latence_disambig_glinker.linker import EntityLinker
from latence_disambig_glinker.models import CanonicalEntityRecord, MentionRecord

logger = logging.getLogger(__name__)


class IncrementalLinker:
    """Links new mentions against an existing KB plus delta entities."""

    def __init__(
        self,
        existing_entities: list[CanonicalEntityRecord],
        backend: InferenceBackend | None = None,
        threshold: float = 0.3,
        batch_size: int = 32,
    ) -> None:
        self._existing_entities = existing_entities
        self._existing_ids = {e.entity_id for e in existing_entities}
        self._backend = backend
        self._threshold = threshold
        self._batch_size = batch_size

    def link_delta(
        self,
        new_mentions: list[MentionRecord],
        dataset_id: str,
    ) -> tuple[list[MentionRecord], list[CanonicalEntityRecord], list[CanonicalEntityRecord]]:
        """Link new mentions against the merged KB (existing + delta).

        Returns ``(linked_mentions, new_canonicals, merged_entities)``.
        """
        if not new_mentions:
            logger.info("No new mentions to link")
            return [], [], list(self._existing_entities)

        linker = EntityLinker(
            backend=self._backend, threshold=self._threshold, batch_size=self._batch_size
        )
        linked_mentions, all_entities = linker.link_dataset(new_mentions, dataset_id=dataset_id)

        new_canonicals = [e for e in all_entities if e.entity_id not in self._existing_ids]
        merged_entities = list(self._existing_entities) + new_canonicals

        logger.info(
            "Incremental linking: %d mentions -> %d matched existing, %d new canonicals",
            len(new_mentions),
            len(all_entities) - len(new_canonicals),
            len(new_canonicals),
        )
        return linked_mentions, new_canonicals, merged_entities
