"""Corpus-state machinery for incremental deltas (S11, ADR-0018).

The ported, pure, contract-light engines behind the lean-yet-enterprise delta
machinery: the :class:`~latence_core.state.write_ahead_log.WriteAheadLog` that makes a
Corpus Version transaction crash-safe, the
:class:`~latence_core.state.delta_detector.DeltaDetector` that content-hashes the Source
manifest into an added/updated/deleted/unchanged :class:`~latence_core.contracts.Delta`,
and the :class:`~latence_core.state.graph_patcher.GraphPatcher` that retracts/adds derived
records surgically by their Evidence provenance (``source_document_ids``). The
:class:`~latence_core.state.corpus_store.CorpusStore` composes them into the read-N →
write-N+1 transaction, and :mod:`latence_core.state.incremental` recasts the disambiguation
engine into the affected-set merge/split.
"""

from __future__ import annotations

from latence_core.state.corpus_store import CorpusStore, CorpusStoreError
from latence_core.state.delta_detector import DeltaDetector, ManifestEntry
from latence_core.state.graph_patcher import GraphPatcher, PatchResult
from latence_core.state.incremental import (
    AffectedSet,
    DriftMetric,
    IncrementalDisambiguator,
    MentionView,
    compute_affected_set,
    compute_drift,
)
from latence_core.state.write_ahead_log import WalRecovery, WriteAheadLog

__all__ = [
    "AffectedSet",
    "CorpusStore",
    "CorpusStoreError",
    "DeltaDetector",
    "DriftMetric",
    "GraphPatcher",
    "IncrementalDisambiguator",
    "ManifestEntry",
    "MentionView",
    "PatchResult",
    "WalRecovery",
    "WriteAheadLog",
    "compute_affected_set",
    "compute_drift",
]
