"""Shared test fixtures: a deterministic fake Scorer + assembled-graph builders (offline, no torch)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from latence_core.contracts import (
    Classification,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from latence_core.graph import make_graph_id
from latence_linkpred_ultra.scorer import BuiltKG


class FakeScorer:
    """A deterministic in-memory Scorer: scores a tail high iff it completes a 2-hop chain.

    For the observed KG it prepares an adjacency; then a ``(head, relation)`` tail query scores a
    tail highly when the tail is 2-hops-reachable from the head over that relation (a transitive
    completion). This makes the provider's end-to-end behaviour deterministic and offline-testable
    without torch — the scoring *policy* is a stand-in for ULTRA, but the Scorer INTERFACE
    (``prepare`` / ``score_tails`` returning a per-entity vector) is exactly the real one.
    """

    name = "ultra"  # stamp as 'ultra' so contract assertions match the real scorer name

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._num_entities = 0
        self._two_hop: set[tuple[int, int, int]] = set()

    def prepare(self, kg: BuiltKG) -> None:
        self._num_entities = kg.num_entities
        by_rel: dict[int, dict[int, set[int]]] = {}
        for h, r, t in kg.edges:
            by_rel.setdefault(r, {}).setdefault(h, set()).add(t)
        # A tail is "plausible" for (h, r) if it is a DIRECT observed tail (so a held-out true edge
        # ranks well in the MRR eval) OR a 2-hop-reachable tail (so a transitive completion is
        # proposed). This mirrors a real scorer: it believes both observed and closure edges,
        # which is what a faithful held-out ranking needs.
        for r, adj in by_rel.items():
            for h, mids in adj.items():
                for mid in mids:
                    self._two_hop.add((h, r, mid))  # direct
                    for t in adj.get(mid, set()):
                        if t != h:
                            self._two_hop.add((h, r, t))  # 2-hop

    def score_tails(self, queries: Sequence[tuple[int, int]]) -> list[list[float]]:
        out: list[list[float]] = []
        for h, r in queries:
            row = [0.0] * self._num_entities
            for t in range(self._num_entities):
                if (h, r, t) in self._two_hop:
                    row[t] = 8.0
                else:
                    row[t] = -8.0
            out.append(row)
        return out


class UnavailableScorer:
    """A Scorer whose prepare always reports unavailable (to drive the fallback / typed-error path)."""

    name = "ultra"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def prepare(self, kg: BuiltKG) -> None:
        from latence_linkpred_ultra.scorer import ScorerUnavailableError

        raise ScorerUnavailableError("test: scorer deliberately unavailable")

    def score_tails(self, queries: Sequence[tuple[int, int]]) -> list[list[float]]:  # pragma: no cover
        return []


def _prov() -> Provenance:
    return Provenance(
        source_uri="file://doc.txt",
        document_id="doc",
        file_name="doc.txt",
        file_type="text/plain",
        file_size=10,
    )


def _cls() -> Classification:
    return Classification(language="en", category="general", sensitivity="internal")


def node_record(corpus_id: str, node_id_seed: str, label: str, docs: list[str]) -> GraphRecord:
    node_id = make_graph_id(corpus_id, "node", node_id_seed)
    node = GraphNode(
        record_id=f"node:{node_id}",
        node_id=node_id,
        entity_id=f"ent:{node_id_seed}",
        label=label,
        canonical_name=node_id_seed,
        member_mention_ids=[],
        source_document_ids=docs,
        kb_id=None,
        confidence=1.0,
        properties={},
        evidence=Evidence(mention_ids=[], document_ids=docs, snippet="", confidence=1.0),
        provenance=_prov(),
        classification=_cls(),
    )
    return GraphRecord(
        record_id=f"node:{node_id}#n",
        scope=GraphScope.NODE,
        node=node,
        provenance=_prov(),
        classification=_cls(),
    )


def edge_record(corpus_id: str, head: str, label: str, tail: str, docs: list[str]) -> GraphRecord:
    edge_id = make_graph_id(corpus_id, "edge", head, label, tail)
    edge = GraphEdge(
        record_id=f"edge:{edge_id}",
        edge_id=edge_id,
        relation_id=f"rel:{head}:{label}:{tail}",
        source_node_id=head,
        target_node_id=tail,
        label=label,
        confidence=1.0,
        properties={"inferred": False},
        evidence=Evidence(mention_ids=["m"], document_ids=docs, snippet="asserted", confidence=1.0),
        provenance=_prov(),
        classification=_cls(),
    )
    return GraphRecord(
        record_id=f"edge:{edge_id}#e",
        scope=GraphScope.EDGE,
        edge=edge,
        provenance=_prov(),
        classification=_cls(),
    )
