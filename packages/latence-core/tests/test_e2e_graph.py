"""End-to-end S9 Graph Assembly + Export through the local Runner.

The headline S9 demo — the first end-to-end messy-docs -> KG run:
Source -> Parse -> Chunk -> Entity -> Relation -> Disambiguation -> Graph Assembly, then a
FAN-OUT to two Exports off the one converging DAG: the KG Export (Parquet + TTL + GraphML)
downstream of Graph Assembly, and the RAG-ready-corpus Export (JSONL + Parquet) downstream of
Chunk. Asserts: the KG holds nodes with deterministic ids + edges that carry Evidence; cross-
document entities merge to one node; the TTL/GraphML/Parquet all materialize and load; the
RAG corpus exports the chunked text with Provenance/Classification; the Quality Report's
GraphQuality carries the KG statistics (the S9 AC); and the run is deterministic + resumes.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pyarrow.parquet as pq
from latence_core.contracts import ChunkRecord, GraphRecord, GraphScope
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: graph-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt, md]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
    config:
      labels:
        ORG:
          terms: [Acme Corporation, Acme, International Business Machines, IBM]
        PERSON:
          terms: [Alice, Bob]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      window_chars: 200
      relations:
        works_for:
          head: PERSON
          tail: ORG
          triggers: [works at, employed by, joined]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
    config:
      kb:
        - kb_id: Q1
          name: Acme
          entity_type: ORG
      kb_threshold: 0.5
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: graph-slice
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
    config:
      basename: corpus
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works at Acme. International Business Machines is a partner.\n",
        encoding="utf-8",
    )
    (docs / "note.txt").write_text(
        "Bob joined Acme last year. IBM announced results. "
        "Acme Corporation filed the report.\n",
        encoding="utf-8",
    )
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_graph_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s9e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s9e2e"

    # 1. The KG Export materialized all three KG representations (Parquet + TTL + GraphML).
    kg = f"{base}/export/graph"
    assert s.exists(f"{kg}-nodes.parquet")
    assert s.exists(f"{kg}-edges.parquet")
    assert s.exists(f"{base}/export/graph.ttl")
    assert s.exists(f"{base}/export/graph.graphml")

    # 2. Nodes load as Arrow; cross-document IBM/International Business Machines is ONE node.
    with s.open(f"{kg}-nodes.parquet", "rb") as fh:
        nodes_tbl = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    node_rows = nodes_tbl.to_pylist()
    node_names = [r["canonical_name"] for r in node_rows]
    node_ids = [r["node_id"] for r in node_rows]
    assert len(node_ids) == len(set(node_ids))  # deterministic, unique node ids
    # Pin the exact assembled node set: the two IBM surface forms ('IBM' in note.txt,
    # 'International Business Machines' in memo.txt) collapse to ONE node during
    # disambiguation, so the graph has exactly these five nodes. A merge regression that
    # left IBM split (6 nodes) would fail here instead of shipping green.
    assert sorted(node_names) == [
        "Acme",
        "Acme Corporation",
        "Alice",
        "Bob",
        "IBM",
    ]
    # That single IBM node carries mentions from BOTH source documents — the actual cross-doc
    # merge signal, not just 'some node contains the substring IBM'.
    ibm = next(r for r in node_rows if r["canonical_name"] == "IBM")
    ibm_mentions = json.loads(ibm["member_mention_ids"])
    ibm_docs = {m.split("#", 1)[0] for m in ibm_mentions}
    assert len(ibm_docs) == 2, "IBM node must merge mentions across both documents"
    # Exactly one node carries IBM's surface forms (no split remnant).
    assert sum(1 for n in node_names if n == "IBM") == 1
    # A node linked to the KB carries a non-empty kb_id.
    kb_ids = [k for k in nodes_tbl.column("kb_id").to_pylist() if k]
    assert "Q1" in kb_ids

    # 3. Edges load; the works_for edge carries Evidence.
    with s.open(f"{kg}-edges.parquet", "rb") as fh:
        edges_tbl = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    assert edges_tbl.num_rows >= 1
    assert "works_for" in edges_tbl.column("label").to_pylist()
    first_edge_evidence = json.loads(edges_tbl.column("evidence").to_pylist()[0])
    assert first_edge_evidence["mention_ids"]  # every edge carries Evidence (S9 AC)

    # 4. GraphML loads in a standard XML parser and its edges reference real nodes.
    root = ET.fromstring(s.read_text(f"{base}/export/graph.graphml"))
    ns = "{http://graphml.graphdrawing.org/xmlns}"
    graph_el = root.find(f"{ns}graph")
    assert graph_el is not None
    gml_node_ids = {n.get("id") for n in graph_el.findall(f"{ns}node")}
    for edge in graph_el.findall(f"{ns}edge"):
        assert edge.get("source") in gml_node_ids
        assert edge.get("target") in gml_node_ids

    # 5. TTL is well-formed Turtle with reified per-edge Evidence.
    ttl = s.read_text(f"{base}/export/graph.ttl")
    assert "@prefix lg:" in ttl
    assert "rdf:Statement" in ttl and "lg:evidenceCount" in ttl

    # 6. The RAG-ready corpus exported the chunked text with Provenance/Classification.
    corpus_uri = f"{base}/export/corpus.jsonl"
    assert s.exists(corpus_uri)
    corpus_rows = [
        json.loads(line) for line in s.read_text(corpus_uri).splitlines() if line.strip()
    ]
    corpus = [ChunkRecord.model_validate(r) for r in corpus_rows]
    assert corpus
    assert all(c.provenance.document_id for c in corpus)
    assert all(c.classification.language for c in corpus)

    # 7. Quality Report: GraphQuality present with the KG statistics (S9 AC).
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.graph is not None
    gq = persisted.graph
    assert gq.nodes_total >= 2
    assert gq.edges_total >= 1
    assert gq.nodes_linked >= 1
    # Every edge carries Evidence -> full coverage.
    assert gq.evidence_coverage == 1.0
    assert gq.edges_with_evidence == gq.edges_total
    assert "ORG" in gq.nodes_by_type
    assert "works_for" in gq.edges_by_label
    # The report returned by run() matches the persisted one.
    assert report.graph is not None
    assert report.graph.nodes_total == gq.nodes_total

    # 8. Contract completeness holds over the exported records.
    assert persisted.contracts_complete


def test_graph_slice_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    for name in ("graph.ttl", "graph.graphml"):
        a = s.read_text(f"{store}/_latence/runs/det-a/export/{name}")
        b = s.read_text(f"{store}/_latence/runs/det-b/export/{name}")
        assert a == b


def test_graph_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/resume/checkpoints/graph.jsonl"
    lines = [x for x in s.read_text(ckpt).splitlines() if x.strip()]
    assert lines
    scopes = {GraphRecord.model_validate_json(line).scope for line in lines}
    assert GraphScope.NODE in scopes


def test_graph_requires_disambiguation_wiring(tmp_path: Path) -> None:
    # A Graph Assembly Stage wired directly to Parse (no Disambiguation) must error clearly.
    store = f"file://{tmp_path}/store"
    docs = _docs(tmp_path)
    pipeline = Pipeline.from_yaml_text(
        f"""
name: bad-graph
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
"""
    )
    try:
        LocalRunner().run(pipeline, run_id="bad")
    except TypeError as exc:
        assert "DisambiguationRecord" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a TypeError for the mis-wired Graph Assembly Stage")
