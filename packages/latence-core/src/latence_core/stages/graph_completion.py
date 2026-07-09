"""Built-in reference GraphCompleter Provider (deterministic, CPU-viable, zero-dep; W2 Slice 1/2).

The OPTIONAL post-Assembly link-prediction reference Provider (ADR-0037; ships in core like every
other reference Provider so the seam runs end to end offline, ADR-0007 CPU-first, ADR-0016 no
extra deps). It consumes the whole run's assembled :class:`~latence_core.contracts.GraphRecord`s
(NODE + EDGE) and yields ADDITIONAL EDGE-scope GraphRecords for **predicted** edges, using the
pure :mod:`latence_core.graph_completion` engine's deterministic, bounded 2-hop transitive/
symmetric closure. This is NOT ULTRA — it is the honest reference that proves the inferred-edge
contract + runner wiring + determinism and gives Slice 2's learned scorer a drop-in bake-off
baseline.

The inferred-edge contract (the enterprise-honesty crux — never silently merged, ADR-0037):

* every predicted edge is a :class:`~latence_core.contracts.GraphEdge` at ``scope=EDGE`` marked
  ``properties["inferred"] = True`` (REQUIRED), plus ``scorer`` (``"reference.adjacency"``),
  ``score`` (raw path-support fraction), ``rank``, and ``calibrated`` (``False`` for the reference —
  the raw fraction is not temperature-calibrated; Slice 2's ULTRA sets it ``True``);
* ``edge_id`` is a deterministic sha256 over
  ``(corpus_id, source_node_id, label, target_node_id, "predicted")`` — distinct from any asserted
  edge id; ``relation_id`` = ``f"predicted:{scorer}:{edge_id}"`` (clearly NOT a source
  ``NormalizedRelation``, and satisfies the contract ``min_length``);
* **Evidence is honest and model-derived, NOT fabricated mention offsets:** ``mention_ids = []``
  (there is no textual mention — asserting one would be a lie), ``document_ids`` = the union of the
  head + tail nodes' ``source_document_ids``, ``snippet`` a human justification naming the scorer +
  score, ``confidence`` the calibrated score;
* a predicted edge that COLLIDES (same endpoints + label) with an asserted edge is DROPPED (it is
  already asserted) — logged, not emitted (the engine already drops observed pairs; the Provider
  double-guards against the asserted-edge set).

Determinism: predicted edges are yielded in edge-id order (the Baseline bar). The Provider never
re-emits the input nodes/edges and never mutates an asserted edge — it appends only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    Evidence,
    GraphEdge,
    GraphRecord,
    GraphScope,
)
from latence_core.graph import make_graph_id
from latence_core.graph_completion import (
    CompletionConfig,
    GraphCompletionEngine,
    ObservedEdge,
    PredictedEdge,
)
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_LOG = logging.getLogger(__name__)

#: The scorer name stamped on every predicted edge this Provider emits (ADR-0037). Slice 2's
#: learned Provider stamps ``"ultra"``.
_SCORER = "reference.adjacency"


class ReferenceGraphCompleter:
    """GraphCompleter Provider: assembled GraphRecords -> predicted inferred EDGE GraphRecords.

    Wraps the pure :class:`~latence_core.graph_completion.GraphCompletionEngine` (a deterministic,
    bounded 2-hop transitive/symmetric closure) and maps each :class:`PredictedEdge` onto the
    inferred-edge contract. Pure Python, deterministic, zero extra deps (ADR-0007/0016/0017).

    Config:
        corpus_id: str  — the stable corpus identity mixed into every deterministic predicted
            ``edge_id`` (so a run + a re-run over the same corpus share ids). Default ``"corpus"``.
        transitive_labels: list[str]  — relation labels that get the 2-hop transitive closure
            (empty + empty ``symmetric_labels`` ⇒ every label is transitive AND symmetric).
        symmetric_labels: list[str]  — relation labels that get the symmetric reverse.
        min_score: float  — keep threshold on the path-support fraction (default 0.0).
        max_candidates_per_node: int  — bounded fan-out per source node (default 32; 0 = unlimited).
        support_norm: float  — score denominator (0 = normalise by the run's max path-support).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        The reference Graph Completer proposes edges by a deterministic adjacency closure. A
        pure-Python part of ``latence-core`` (Apache-2.0, verified per ADR-0012 — the package's own
        code, weights-and-code trivially the same), it ships no model weights (``model_id=None``),
        is deterministic, CPU-only, does not batch, and has no per-call cost, so a bake-off/
        conformance run reads honest, non-fabricated columns off it. (Slice 2's ULTRA Provider
        declares a real ``model_id`` + weights license.)
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=32,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-09",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._corpus_id = str(config.get("corpus_id", "corpus"))
        self._config = CompletionConfig(
            transitive_labels=frozenset(config.get("transitive_labels", []) or []),
            symmetric_labels=frozenset(config.get("symmetric_labels", []) or []),
            min_score=float(config.get("min_score", 0.0)),
            max_candidates_per_node=int(config.get("max_candidates_per_node", 32)),
            support_norm=float(config.get("support_norm", 0.0)),
        )

    def complete(self, records: Iterable[GraphRecord]) -> Iterator[GraphRecord]:
        # Materialise the assembled graph: observed edges + the node -> source-document-ids map
        # (for honest Evidence) + a representative source record per endpoint (for inherited
        # Provenance/Classification) + the asserted-edge collision set. The whole assembled graph
        # is corpus-level and already bounded by the KG size (nodes + edges), like GraphAssembler's
        # input — not the corpus size.
        observed: list[ObservedEdge] = []
        node_docs: dict[str, list[str]] = {}
        node_source: dict[str, GraphRecord] = {}
        asserted_keys: set[tuple[str, str, str]] = set()
        for record in records:
            if record.scope is GraphScope.NODE and record.node is not None:
                node_docs[record.node.node_id] = list(record.node.source_document_ids)
                node_source[record.node.node_id] = record
            elif record.scope is GraphScope.EDGE and record.edge is not None:
                edge = record.edge
                observed.append(
                    ObservedEdge(
                        source_node_id=edge.source_node_id,
                        label=edge.label,
                        target_node_id=edge.target_node_id,
                    )
                )
                asserted_keys.add((edge.source_node_id, edge.label, edge.target_node_id))

        predicted = GraphCompletionEngine(self._config).predict(observed)

        emitted: list[GraphRecord] = []
        for pred in predicted:
            key = (pred.source_node_id, pred.label, pred.target_node_id)
            # Double-guard the asserted-edge collision drop (the engine already never proposes an
            # observed pair; this is the belt-and-braces contract guarantee — logged, not emitted).
            if key in asserted_keys:
                _LOG.debug(
                    "graph_completion: dropping predicted edge colliding with an asserted edge: %s",
                    key,
                )
                continue
            source = node_source.get(pred.source_node_id)
            if source is None:  # pragma: no cover - engine never proposes an unknown node
                _LOG.debug(
                    "graph_completion: dropping predicted edge with unknown source node: %s", key
                )
                continue
            emitted.append(self._predicted_record(pred, node_docs, source))

        # Deterministic: yield predicted edges in edge-id order (the Baseline bar).
        emitted.sort(key=lambda r: r.edge.edge_id if r.edge is not None else "")
        yield from emitted

    def _predicted_record(
        self,
        pred: PredictedEdge,
        node_docs: dict[str, list[str]],
        source: GraphRecord,
    ) -> GraphRecord:
        edge_id = make_graph_id(
            self._corpus_id,
            "predicted",
            pred.source_node_id,
            pred.label,
            pred.target_node_id,
        )
        relation_id = f"predicted:{_SCORER}:{edge_id}"
        # Honest, model-derived Evidence: NO fabricated mention offsets (mention_ids stays empty —
        # a predicted edge has no textual mention), document_ids = union of the head + tail nodes'
        # source documents (dedup + sort for determinism).
        document_ids = sorted(
            set(node_docs.get(pred.source_node_id, []))
            | set(node_docs.get(pred.target_node_id, []))
        )
        evidence = Evidence(
            mention_ids=[],
            document_ids=document_ids,
            snippet=(
                f"Predicted edge (not asserted in text); {_SCORER} "
                f"score {pred.score:.3f} ({pred.rule})"
            ),
            confidence=pred.score,
        )
        graph_edge = GraphEdge(
            record_id=f"edge:{edge_id}",
            edge_id=edge_id,
            relation_id=relation_id,
            source_node_id=pred.source_node_id,
            target_node_id=pred.target_node_id,
            label=pred.label,
            confidence=pred.score,
            properties={
                "inferred": True,
                "scorer": _SCORER,
                "score": pred.score,
                "rank": pred.rank,
                "calibrated": False,
                "support": pred.support,
                "rule": pred.rule,
            },
            evidence=evidence,
            provenance=source.provenance,
            classification=source.classification,
        )
        return GraphRecord(
            record_id=f"edge:{edge_id}#predicted",
            scope=GraphScope.EDGE,
            edge=graph_edge,
            provenance=source.provenance,
            classification=source.classification,
        )
