"""Deterministic ID generation for the ported linking/resolution engine.

Ported (PORT-AS-IS) from ``dataset_intelligence.utils.ids``. All IDs are SHA-256 hashes of their
constituent parts so reruns on the same input produce identical IDs (the Baseline determinism bar).
"""

from __future__ import annotations

import hashlib


def make_entity_id(dataset_id: str, canonical_name: str, label: str) -> str:
    """Deterministic canonical entity ID from dataset + canonical name + label."""
    raw = f"entity:{dataset_id}:{canonical_name}:{label}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
