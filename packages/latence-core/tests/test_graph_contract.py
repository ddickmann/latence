"""S9 Graph Assembly contract tests: GraphNode / GraphEdge / GraphRecord scope enforcement."""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from pydantic import ValidationError


def _prov() -> Provenance:
    return Provenance(
        source_uri="file:///c/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        file_size=10,
        document_id="doc-1",
        char_start=0,
        char_end=4,
        page_start=1,
        page_end=1,
    )


def _cls() -> Classification:
    return Classification(language="en", category="memo", sensitivity="internal")


def _ev(mentions: list[str] | None = None) -> Evidence:
    return Evidence(mention_ids=mentions or ["m1"], document_ids=["doc-1"], confidence=0.9)


def _node(node_id: str = "n1") -> GraphNode:
    return GraphNode(
        record_id=f"node:{node_id}",
        node_id=node_id,
        entity_id="entity:acme",
        label="ORG",
        canonical_name="Acme",
        member_mention_ids=["m1", "m2"],
        source_document_ids=["doc-1"],
        confidence=0.9,
        evidence=_ev(["m1", "m2"]),
        provenance=_prov(),
        classification=_cls(),
    )


def _edge(src: str = "n1", tgt: str = "n2") -> GraphEdge:
    return GraphEdge(
        record_id="edge:e1",
        edge_id="e1",
        relation_id="rel-1#normalized",
        source_node_id=src,
        target_node_id=tgt,
        label="works_for",
        confidence=0.8,
        evidence=_ev(["m1", "m3"]),
        provenance=_prov(),
        classification=_cls(),
    )


def test_node_carries_provenance_and_classification() -> None:
    node = _node()
    assert node.provenance.document_id == "doc-1"
    assert node.classification.language == "en"
    assert node.schema_version == SCHEMA_VERSION
    assert node.linked is False


def test_node_kb_link_sets_linked() -> None:
    node = _node().model_copy(update={"kb_id": "Q1"})
    assert node.linked is True


def test_edge_endpoints_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        _edge(src="n1", tgt="n1")


def test_edge_carries_evidence() -> None:
    edge = _edge()
    assert edge.evidence.mention_ids == ["m1", "m3"]
    assert edge.evidence.confidence == pytest.approx(0.9)


def test_graph_record_node_scope_requires_node() -> None:
    rec = GraphRecord(
        record_id="node:n1#graph",
        scope=GraphScope.NODE,
        node=_node(),
        provenance=_prov(),
        classification=_cls(),
    )
    assert rec.node is not None
    assert rec.edge is None


def test_graph_record_node_scope_rejects_edge_payload() -> None:
    with pytest.raises(ValidationError, match="must not carry an 'edge'"):
        GraphRecord(
            record_id="bad",
            scope=GraphScope.NODE,
            node=_node(),
            edge=_edge(),
            provenance=_prov(),
            classification=_cls(),
        )


def test_graph_record_edge_scope_requires_edge() -> None:
    with pytest.raises(ValidationError, match="must carry an 'edge'"):
        GraphRecord(
            record_id="bad",
            scope=GraphScope.EDGE,
            provenance=_prov(),
            classification=_cls(),
        )


def test_graph_record_edge_scope_rejects_node_payload() -> None:
    with pytest.raises(ValidationError, match="must not carry a 'node'"):
        GraphRecord(
            record_id="bad",
            scope=GraphScope.EDGE,
            edge=_edge(),
            node=_node(),
            provenance=_prov(),
            classification=_cls(),
        )
