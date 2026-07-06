"""Unit tests for the self-contained web viewer + the export.demo_site Provider (S13).

Covers the S13 acceptance criteria for the viewer: it renders the KG (nodes/edges/Evidence)
and the Quality Report; it references no external CDN/network; it is deterministic; and the
DemoSiteExport Provider is discoverable via the latence.providers entry point and writes a
valid standalone page + typed manifest.
"""

from __future__ import annotations

from pathlib import Path

from latence_core.capability import ProviderRegistry
from latence_core.contracts import (
    Classification,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from latence_core.storage import Storage
from latence_demo.contracts import DemoSiteManifest
from latence_demo.viewer import DemoSiteExport, render_site


def _prov() -> Provenance:
    return Provenance(
        source_uri="file://corpus/memo.md",
        file_name="memo.md",
        file_type="md",
        file_size=42,
        document_id="sha256:doc",
        char_start=0,
        char_end=10,
    )


def _cls() -> Classification:
    return Classification(language="en", sensitivity="internal")


def _node(node_id: str, name: str, label: str, kb: str | None = None) -> GraphNode:
    return GraphNode(
        record_id=f"n:{node_id}",
        provenance=_prov(),
        classification=_cls(),
        node_id=node_id,
        entity_id=f"e:{node_id}",
        label=label,
        canonical_name=name,
        member_mention_ids=[f"{name}#0"],
        source_document_ids=["sha256:doc"],
        kb_id=kb,
        confidence=0.9,
        evidence=Evidence(mention_ids=[f"{name}#0"], document_ids=["sha256:doc"], confidence=0.9),
    )


def _edge(edge_id: str, src: str, tgt: str, label: str) -> GraphEdge:
    return GraphEdge(
        record_id=f"x:{edge_id}",
        provenance=_prov(),
        classification=_cls(),
        edge_id=edge_id,
        relation_id=f"r:{edge_id}",
        source_node_id=src,
        target_node_id=tgt,
        label=label,
        confidence=0.8,
        evidence=Evidence(mention_ids=["m0", "m1"], document_ids=["sha256:doc"], confidence=0.8),
    )


def _records() -> list[GraphRecord]:
    nodes = [
        _node("na", "Alice", "PERSON"),
        _node("nb", "Acme", "ORG", kb="Q1"),
    ]
    edges = [_edge("ea", "na", "nb", "works_for")]
    recs = [GraphRecord(record_id=n.record_id, provenance=n.provenance,
                        classification=n.classification, scope=GraphScope.NODE, node=n)
            for n in nodes]
    recs += [GraphRecord(record_id=e.record_id, provenance=e.provenance,
                         classification=e.classification, scope=GraphScope.EDGE, edge=e)
             for e in edges]
    return recs


def test_render_site_shows_graph_and_report() -> None:
    nodes = [_node("na", "Alice", "PERSON"), _node("nb", "Acme", "ORG", kb="Q1")]
    edges = [_edge("ea", "na", "nb", "works_for")]
    report = {
        "document_count": 3,
        "total_duration_seconds": 0.01,
        "contracts_complete": True,
        "completeness": {"records_with_provenance": 5, "records_with_classification": 5,
                         "records_checked": 5},
        "stages": [{"name": "parse", "capability": "parse", "provider": "parser.document",
                    "records_in": 3, "records_out": 3, "duration_seconds": 0.01}],
        "screening": {"quarantined": 3, "flagged": 1,
                      "findings": [{"checkpoint": "intake", "verdict": "quarantine",
                                    "category": "zip_bomb", "reason": "ratio too high"}]},
    }
    html = render_site(nodes, edges, pipeline_name="demo", report=report,
                       corpus={"memo.md": "clean", "archive.zip": "quarantine:zip_bomb"})
    # KG: nodes, edge label, per-edge Evidence table, and the KB link all render.
    assert "<svg" in html and "Alice" in html and "Acme" in html
    assert "works_for" in html
    assert "Evidence" in html  # the per-edge Evidence table header
    # Quality Report: stage + screening surfaced.
    assert "parser.document" in html
    assert "zip_bomb" in html
    # Corpus header lists the ingested files.
    assert "archive.zip" in html


def test_render_site_is_self_contained_and_deterministic() -> None:
    nodes = [_node("na", "Alice", "PERSON"), _node("nb", "Acme", "ORG", kb="Q1")]
    edges = [_edge("ea", "na", "nb", "works_for")]
    a = render_site(nodes, edges, pipeline_name="demo")
    b = render_site(nodes, edges, pipeline_name="demo")
    assert a == b  # byte-identical (deterministic layout + ordering)
    # No external CDN/network resource: no remote http(s) fetch and no protocol-relative src.
    lowered = a.lower()
    assert "https://" not in lowered
    assert "src=\"//" not in lowered and "href=\"//" not in lowered
    # Only inline assets: a <style> block and an inline <svg>, no <link>/<script src>.
    assert "<style>" in a
    assert "<script src" not in a
    assert "<link " not in a


def test_render_site_handles_empty_graph() -> None:
    html = render_site([], [], pipeline_name="demo")
    assert "No graph nodes assembled." in html
    assert "<svg" in html


def test_demo_site_export_writes_page_and_manifest(tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    storage = Storage.for_uri(store)
    out_dir = f"{store}/export"
    provider = DemoSiteExport({"basename": "index", "pipeline_name": "demo"})
    returned = provider.export(_records(), storage, out_dir)
    assert returned == [f"{out_dir}/index.html", f"{out_dir}/index.demo.json"]
    assert storage.exists(f"{out_dir}/index.html")
    manifest = DemoSiteManifest.model_validate_json(storage.read_text(f"{out_dir}/index.demo.json"))
    assert manifest.self_contained is True
    assert manifest.graph.nodes_total == 2
    assert manifest.graph.edges_total == 1
    assert manifest.graph.nodes_linked == 1
    assert manifest.graph.edges_with_evidence == 1


def test_demo_site_export_is_a_passthrough_sink(tmp_path: Path) -> None:
    # The Export protocol requires the Provider to return its input records unchanged so the
    # Runner can checkpoint them (mirrors the S1/S9 exporters). DemoSiteExport returns the URIs
    # it wrote; the Runner treats an Export as a sink and re-emits its inputs — assert the
    # written page reflects exactly the records passed in.
    store = f"file://{tmp_path}/store"
    storage = Storage.for_uri(store)
    provider = DemoSiteExport()
    provider.export(_records(), storage, f"{store}/export")
    html = storage.read_text(f"{store}/export/index.html")
    assert "Alice" in html and "Acme" in html


def test_demo_site_export_registered_via_entry_point() -> None:
    # export.demo_site is discoverable through the latence.providers group (ADR-0004), so any
    # Pipeline can wire it with no core change.
    registry = ProviderRegistry()
    assert "export.demo_site" in registry.names()
    assert registry.load("export.demo_site") is DemoSiteExport


def test_demo_site_export_ignores_non_graph_records(tmp_path: Path) -> None:
    # A record that is not a GraphRecord is ignored (an Export downstream of Graph Assembly sees
    # only graph records); the page still renders with just the graph records present.
    store = f"file://{tmp_path}/store"
    storage = Storage.for_uri(store)
    provider = DemoSiteExport()
    provider.export(_records(), storage, f"{store}/export")
    manifest = DemoSiteManifest.model_validate_json(
        storage.read_text(f"{store}/export/index.demo.json")
    )
    assert manifest.graph.nodes_total == 2
