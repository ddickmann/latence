"""Contract-free engine models for the ported GLinker linking + resolution pipeline.

Ported (PORT-AS-IS) from the private ``latenceai-dataset-intelligence`` assets
(``models/mentions.py``, ``models/entities.py``). Kept as small, contract-free Pydantic
models so the ported engine (:mod:`.kb_builder`, :mod:`.backends`, :mod:`.linker`,
:mod:`.entity_resolver`, :mod:`.incremental_linker`) stays IO/framework-independent — exactly
as it was in the source repo. The Provider (:mod:`.provider`) projects each framework
:class:`~latence_core.contracts.EntityMention` into a :class:`MentionRecord` and lifts each
resolved :class:`CanonicalEntityRecord` back onto the framework's
:class:`~latence_core.contracts.CanonicalEntity` / ``ResolvedCluster`` — so the algorithm is
byte-for-byte the enterprise one, only the DI at the boundary is adapted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MentionRecord(BaseModel):
    """A single mention of an entity in a specific chunk/document (contract-free view).

    Mirrors the source ``dataset_intelligence.models.mentions.MentionRecord`` the linker +
    resolver were written against; ``context`` carries the ±window of surrounding text the
    Provider extracts from the run's ChunkRecords (the source read it from the document markdown).
    """

    mention_id: str
    dataset_id: str
    document_id: str
    chunk_id: str = ""

    text: str
    label: str
    start: int = 0
    end: int = 0
    score: float = 0.0
    context: str = ""
    source: str = "model"

    # Resolved after entity linking.
    canonical_entity_id: str | None = None
    # v20 (ADR-0057): the linker's confidence for THAT link, previously computed and dropped.
    # None = unlinked / no score, never 0.0 (missing is unknown, not zero confidence).
    link_confidence: float | None = None


class CanonicalEntityRecord(BaseModel):
    """A resolved, canonical entity aggregated across the corpus (contract-free view).

    Mirrors the source ``dataset_intelligence.models.entities.CanonicalEntityRecord``. Multiple
    mentions across documents may resolve to the same canonical entity after linking + resolution.
    """

    entity_id: str
    dataset_id: str

    canonical_name: str
    label: str
    aliases: list[str] = Field(default_factory=list)

    # --- Provenance ---
    mention_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0

    # --- Confidence ---
    confidence: float = 0.0
    # v20 (ADR-0057): per-member link confidence, mention_id -> 0..1, linked members only.
    link_confidences: dict[str, float] = Field(default_factory=dict)
    source: str = "resolved"

    # --- Metadata ---
    properties: dict[str, object] = Field(default_factory=dict)
    version: int = 1
