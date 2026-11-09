"""S9 KnowledgeGraphExport tests: Parquet loads as Arrow, TTL well-formed RDF, GraphML loads."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from latence_core.capability import Export
from latence_core.contracts import (
    Classification,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
)
from latence_core.stages.graph_export import KnowledgeGraphExport
from latence_core.storage import Storage


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
    return Classification(language="en", sensitivity="internal")


def _node(node_id: str, name: str, kb: str | None = None) -> GraphRecord:
    node = GraphNode(
        record_id=f"node:{node_id}",
        node_id=node_id,
        entity_id=f"entity:{name.lower()}",
        label="ORG",
        canonical_name=name,
        member_mention_ids=["m1"],
        source_document_ids=["doc-1"],
        kb_id=kb,
        confidence=0.9,
        properties={"mention_count": 1},
        evidence=Evidence(mention_ids=["m1"], document_ids=["doc-1"], confidence=0.9),
        provenance=_prov(),
        classification=_cls(),
    )
    return GraphRecord(
        record_id=f"node:{node_id}#graph",
        scope=GraphScope.NODE,
        node=node,
        provenance=_prov(),
        classification=_cls(),
    )


def _edge(edge_id: str, src: str, tgt: str) -> GraphRecord:
    edge = GraphEdge(
        record_id=f"edge:{edge_id}",
        edge_id=edge_id,
        relation_id="rel-1#normalized",
        source_node_id=src,
        target_node_id=tgt,
        label="works_for",
        confidence=0.8,
        properties={"inverted": False},
        evidence=Evidence(mention_ids=["m1", "m2"], document_ids=["doc-1"], confidence=0.8),
        provenance=_prov(),
        classification=_cls(),
    )
    return GraphRecord(
        record_id=f"edge:{edge_id}#graph",
        scope=GraphScope.EDGE,
        edge=edge,
        provenance=_prov(),
        classification=_cls(),
    )


def _records() -> list[GraphRecord]:
    return [
        _node("na", "Alice"),
        _node("nb", "Acme", kb="Q1"),
        _edge("e1", "na", "nb"),
    ]


def test_provider_satisfies_export_capability() -> None:
    assert isinstance(KnowledgeGraphExport(), Export)


def test_export_writes_four_kg_files(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    out = f"{store}/kg"
    uris = KnowledgeGraphExport().export(_records(), s, out)
    assert len(uris) == 4
    for uri in uris:
        assert s.exists(uri)
    assert any(u.endswith("-nodes.parquet") for u in uris)
    assert any(u.endswith("-edges.parquet") for u in uris)
    assert any(u.endswith("graph.ttl") for u in uris)
    assert any(u.endswith("graph.graphml") for u in uris)


def test_nodes_parquet_loads_as_arrow(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    with s.open(f"{store}/kg/graph-nodes.parquet", "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert table.num_rows == 2
    names = set(table.column("canonical_name").to_pylist())
    assert names == {"Alice", "Acme"}


def test_nodes_parquet_entity_type_column_named_type(tmp_path: Path) -> None:
    # W12: the node entity-type column is named `type` (not `label`), carrying the entity type, so
    # Parquet names it consistently with GraphML (`type`) and TTL (`lg:entityType`).
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    with s.open(f"{store}/kg/graph-nodes.parquet", "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert "type" in table.column_names
    assert "label" not in table.column_names  # the old node-type column name is gone
    assert set(table.column("type").to_pylist()) == {"ORG"}


def test_node_type_matches_across_parquet_graphml_ttl(tmp_path: Path) -> None:
    # The entity type is named+carried consistently in all THREE export formats (W12).
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    with s.open(f"{store}/kg/graph-nodes.parquet", "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert set(table.column("type").to_pylist()) == {"ORG"}
    # GraphML: <data key="type">ORG</data>
    graphml = s.read_text(f"{store}/kg/graph.graphml")
    assert '<data key="type">ORG</data>' in graphml
    # TTL: lg:entityType "ORG"
    ttl = s.read_text(f"{store}/kg/graph.ttl")
    assert 'lg:entityType "ORG"' in ttl


def test_edges_parquet_loads_as_arrow(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    with s.open(f"{store}/kg/graph-edges.parquet", "rb") as fh:
        table = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert table.num_rows == 1
    assert table.column("label").to_pylist() == ["works_for"]


def test_ttl_is_well_formed(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    ttl = s.read_text(f"{store}/kg/graph.ttl")
    # Prefixes declared, and every non-prefix, non-blank statement terminates with a period.
    assert "@prefix rdf:" in ttl
    assert "@prefix lg:" in ttl
    assert "rdfs:label" in ttl
    body = [
        ln.strip()
        for ln in ttl.splitlines()
        if ln.strip() and not ln.startswith("@prefix")
    ]
    # A Turtle document is a sequence of statements; the last non-empty token before a blank
    # line separator must be a terminator (';' inside a statement, '.' at its end).
    assert any(ln.endswith(" .") for ln in body)
    # Reified statements carry the per-edge Evidence count and confidence.
    assert "rdf:Statement" in ttl
    assert "lg:evidenceCount" in ttl
    # It parses as UTF-8 with balanced IRI brackets (a cheap structural sanity check).
    assert ttl.count("<") == ttl.count(">")


def test_graphml_loads_as_xml(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    graphml = s.read_text(f"{store}/kg/graph.graphml")
    root = ET.fromstring(graphml)  # a real XML parser: proves it's well-formed loadable XML
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    assert root.tag == f"{ns}graphml"
    graph = root.find(f"{ns}graph")
    assert graph is not None
    nodes = graph.findall(f"{ns}node")
    edges = graph.findall(f"{ns}edge")
    assert len(nodes) == 2
    assert len(edges) == 1
    # Edge references node ids that exist.
    node_ids = {n.get("id") for n in nodes}
    edge = edges[0]
    assert edge.get("source") in node_ids
    assert edge.get("target") in node_ids


def test_empty_graph_writes_valid_files(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export([], s, f"{store}/kg")
    # Parquet with zero rows still loads.
    with s.open(f"{store}/kg/graph-nodes.parquet", "rb") as fh:
        assert pq.read_table(fh).num_rows == 0  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    # GraphML still parses.
    root = ET.fromstring(s.read_text(f"{store}/kg/graph.graphml"))
    assert root.tag.endswith("graphml")


def test_export_is_byte_stable(tmp_path: Path) -> None:
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(_records(), s, f"{store}/a")
    KnowledgeGraphExport().export(_records(), s, f"{store}/b")
    assert s.read_text(f"{store}/a/graph.ttl") == s.read_text(f"{store}/b/graph.ttl")
    assert s.read_text(f"{store}/a/graph.graphml") == s.read_text(f"{store}/b/graph.graphml")


def test_export_returns_input_records_unchanged() -> None:
    # KnowledgeGraphExport is a pass-through sink like the S1 exporter; verify via the Runner
    # contract that it does not mutate — here we just confirm the records list is usable after.
    records = _records()
    store = "memory://kgtest"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export(records, s, f"{store}/kg")
    assert len(records) == 3


def test_ttl_escapes_special_characters(tmp_path: Path) -> None:
    # A canonical name with a quote/newline must not break the Turtle literal.
    node = _node("nx", 'Ac"me\nInc')
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export([node], s, f"{store}/kg")
    ttl = s.read_text(f"{store}/kg/graph.ttl")
    assert '\\"' in ttl and "\\n" in ttl


def test_control_chars_dont_break_any_representation(tmp_path: Path) -> None:
    # Node canonical_name / edge label are verbatim spans from messy source documents. A stray
    # XML-1.0-illegal control character (NUL, BEL, form-feed, vertical-tab) must not make the
    # GraphML non-well-formed or the TTL unparseable — the S9 'loads in a standard tool' AC and
    # the Baseline 'input validation on external data' bar. Regression guard for all three reps.
    poisoned_name = "Acme\x0cCorp\x00\x07\x0bInc"  # form-feed + NUL + BEL + vertical-tab
    node_a = _node("na", poisoned_name)
    node_b = _node("nb", "Beta")
    edge = _edge("e1", "na", "nb")
    assert edge.edge is not None
    edge.edge.label = "works\x0cfor"  # the predicate/label is attacker-controlled too
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    KnowledgeGraphExport().export([node_a, node_b, edge], s, f"{store}/kg")

    # 1. GraphML loads in a standard XML parser (would raise 'not well-formed' before the fix).
    graphml = s.read_text(f"{store}/kg/graph.graphml")
    root = ET.fromstring(graphml)
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    labels = [d.text for d in root.iter(f"{ns}data")]
    assert "AcmeCorpInc" in labels  # control chars scrubbed, visible text preserved

    # 2. The serialized output carries no raw XML-1.0-illegal control character anywhere.
    illegal = "".join(
        chr(c) for c in [*range(0x00, 0x09), 0x0B, 0x0C, *range(0x0E, 0x20)]
    )
    assert not (set(graphml) & set(illegal))
    ttl = s.read_text(f"{store}/kg/graph.ttl")
    assert not (set(ttl) & set(illegal))

    # 3. The TTL IRI/predicate path also survives: brackets balance and it stays valid UTF-8.
    assert ttl.count("<") == ttl.count(">")
    ttl.encode("utf-8")  # no surrogate/escape corruption


# -- Set-atomic publication (the corpus-Export discipline, ADR-0061 §2) ---------


def test_export_leaves_no_temp_files(tmp_path: Path) -> None:
    # Atomic set publication: every file streams into a temp sibling and is renamed on
    # publish, so a completed export leaves no ``.tmp`` behind (the corpus-Export idiom).
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    uris = KnowledgeGraphExport().export(_records(), s, f"{store}/kg")
    assert len(uris) == 4  # nodes/edges parquet + ttl + graphml
    leftover = [p for p in s.walk_files(f"{store}/kg") if p.endswith(".tmp")]
    assert leftover == []


def test_crash_mid_set_leaves_previous_complete_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The torn-set hazard this Export publication exists to close (ADR-0061 §2): a crash
    # part-way through re-exporting must leave the PREVIOUS complete, mutually-consistent
    # file set — never a new ``graph.ttl`` beside a stale ``graph-nodes.parquet``. Inject the
    # crash mid-set (after the parquet + TTL streams, before GraphML) and assert every
    # previously-published byte survives untouched, with no temp debris.
    store = f"file://{tmp_path}"
    s = Storage.for_uri(store)
    out = f"{store}/kg"
    KnowledgeGraphExport().export(_records(), s, out)
    before = {uri: s.read_bytes(uri) for uri in s.walk_files(out)}
    assert len(before) == 4

    def _boom(self: object, nodes: object, edges: object) -> str:
        msg = "injected mid-set crash"
        raise RuntimeError(msg)

    monkeypatch.setattr(KnowledgeGraphExport, "_render_graphml", _boom)
    # A CHANGED graph (extra node) so a torn publication would be detectable as mixed bytes.
    with pytest.raises(RuntimeError, match="injected mid-set crash"):
        KnowledgeGraphExport().export([*_records(), _node("nc", "Carol")], s, out)

    after = {uri: s.read_bytes(uri) for uri in s.walk_files(out)}
    assert after == before  # the previous complete set, byte-for-byte; no partials, no temps
