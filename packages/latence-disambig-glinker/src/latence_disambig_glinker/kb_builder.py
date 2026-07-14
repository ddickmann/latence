"""Entity Knowledge Base builder — constructs a linking KB from pipeline mentions.

Ported (PORT-AS-IS) from ``dataset_intelligence.linking.kb_builder``. The KB is
self-referencing: unique entity texts from the run become KB entries. GLinker then resolves
which mentions map to which canonical entry.

Strategy:
  1. Aggregate all mentions by normalized text
  2. For each unique entity text, pick the dominant label (type consensus)
  3. Build a GLinker-compatible entity dict with aliases
  4. Score by frequency (popularity) for disambiguation priority
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from latence_disambig_glinker.ids import make_entity_id
from latence_disambig_glinker.models import MentionRecord

logger = logging.getLogger(__name__)


def build_entity_kb(
    mentions: list[MentionRecord],
    dataset_id: str,
    min_mentions: int = 1,
    min_score: float = 0.3,
) -> list[dict[str, Any]]:
    """Build an entity knowledge base from pipeline mentions.

    Groups mentions by normalized text, resolves the dominant label, collects aliases, and
    produces GLinker-compatible entity dicts.

    Args:
        mentions: All mention records from the pipeline.
        dataset_id: Dataset identifier for ID generation.
        min_mentions: Minimum mention count to include in KB.
        min_score: Minimum average score to include in KB.

    Returns:
        List of entity dicts compatible with GLinker's ``load_entities()``.
    """
    # Group by normalized text.
    groups: dict[str, list[MentionRecord]] = defaultdict(list)
    for m in mentions:
        key = m.text.strip().lower()
        if key:
            groups[key].append(m)

    entities: list[dict[str, Any]] = []
    for _key, group in groups.items():
        if len(group) < min_mentions:
            continue

        avg_score = sum(m.score for m in group) / len(group)
        if avg_score < min_score:
            continue

        # Dominant label by weighted vote (count x score); deterministic tie-break by label.
        label_scores: dict[str, float] = defaultdict(float)
        for m in group:
            label_scores[m.label] += m.score

        dominant_label = sorted(label_scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        # Pick the most frequent surface form as canonical, with the SAME deterministic-sorted
        # tie-break as the dominant-label vote above (A-F3): most frequent first, ties by surface
        # (not Counter.most_common, whose ties fall to insertion order — inconsistent with the label
        # rung and fragile under any upstream reordering).
        surface_forms = sorted({m.text for m in group})
        form_counts = Counter(m.text for m in group)
        canonical_name = sorted(surface_forms, key=lambda f: (-form_counts[f], f))[0]
        aliases = [f for f in surface_forms if f != canonical_name]

        # Generate deterministic entity ID.
        entity_id = make_entity_id(dataset_id, canonical_name, dominant_label)

        # Build description from context.
        description = _build_description(canonical_name, dominant_label, group)

        entities.append(
            {
                "entity_id": entity_id,
                "label": canonical_name,
                "description": description,
                "entity_type": dominant_label,
                "popularity": len(group),
                "aliases": aliases,
                # Extra metadata for downstream.
                "_mention_count": len(group),
                "_avg_score": round(avg_score, 4),
                "_label_distribution": dict(label_scores),
                "_document_ids": list({m.document_id for m in group}),
            }
        )

    # Sort by popularity (most frequent first) for disambiguation priority.
    entities.sort(key=lambda e: e["popularity"], reverse=True)

    logger.info(
        "Built entity KB: %d entries from %d mentions (%d unique texts, %d filtered)",
        len(entities),
        len(mentions),
        len(groups),
        len(groups) - len(entities),
    )

    return entities


def _build_description(name: str, label: str, mentions: list[MentionRecord]) -> str:
    """Build a short description for the entity from its mentions.

    GLinker uses "{label}: {description}" format for linking. Includes first-seen context
    snippet for better BiEncoder discrimination.
    """
    doc_count = len({m.document_id for m in mentions})
    mention_count = len(mentions)

    parts = [f"{label}"]
    if doc_count > 1:
        parts.append(f"mentioned across {doc_count} documents")
    if mention_count > 3:
        parts.append(f"{mention_count} occurrences")

    first = mentions[0]
    ctx = first.context
    if ctx and len(ctx) > len(name) + 5:
        snippet = ctx[:120].replace("\n", " ").strip()
        if snippet:
            parts.append(f'context: "{snippet}"')

    return ". ".join(parts)


def kb_to_glinker_format(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our KB format to GLinker's entity format.

    GLinker expects: entity_id, label, description, entity_type, aliases, popularity.
    Our KB already matches this format by design.
    """
    return [
        {
            "entity_id": e["entity_id"],
            "label": e["label"],
            "description": e.get("description", ""),
            "entity_type": e.get("entity_type", ""),
            "popularity": e.get("popularity", 0),
            "aliases": e.get("aliases", []),
        }
        for e in entities
    ]
