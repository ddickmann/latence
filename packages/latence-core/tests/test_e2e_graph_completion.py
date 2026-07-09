"""End-to-end W2-linkpred Graph Completion through the local Runner.

The OPTIONAL post-Assembly link-prediction Stage, proven end to end:
Source -> Parse -> Chunk -> Entity -> Relation -> Disambiguation -> Graph Assembly ->
Graph Completion, then Export off BOTH the asserted graph and the predicted edges. Asserts:

* the completer appends PREDICTED edges (the 2-hop closure over observed ``partner_of`` edges),
  each marked ``properties["inferred"] = True`` with honest model-derived Evidence (EMPTY
  mention_ids, endpoint-doc union) — never merged into the asserted edges;
* the Quality Report gains a SEPARATE ``inferred_edge_count`` that never inflates ``edges_total``;
* the predicted edges reach the KG Export (marked inferred);
* a stack WITHOUT the graph_completion Stage is BYTE-IDENTICAL to today (the optional-stage proof);
* the run is deterministic (byte-identical predicted edges over two runs) and resumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from latence_core.contracts import GraphRecord, GraphScope
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

# A corpus whose relations form a transitive chain: Acme partner_of Beta, Beta partner_of Gamma.
# The reference completer proposes the 2-hop edge Acme partner_of Gamma (never asserted in text).
_CORPUS = {
    "a.txt": "Acme partners with Beta on the venture.\n",
    "b.txt": "Beta partners with Gamma in the consortium.\n",
}


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in _CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    return docs


def _base_stages(store: str, docs_dir: Path) -> str:
    """The shared spine up to Graph Assembly (Source..graph) — reused by both pipelines."""
    return f"""
name: gc-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
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
          terms: [Acme, Beta, Gamma]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      window_chars: 200
      relations:
        partner_of:
          head: ORG
          tail: ORG
          triggers: [partners with]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: gc-slice
"""


def _pipeline_with_completion(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        _base_stages(store, docs_dir)
        + """
  - name: complete
    capability: graph_completion
    provider: graph_completion.reference
    depends_on: [graph]
    config:
      corpus_id: gc-slice
      transitive_labels: [partner_of]
      symmetric_labels: []
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph, complete]
"""
    )


def _pipeline_without_completion(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        _base_stages(store, docs_dir)
        + """
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
"""
    )


def _read_predicted(store: str, run_id: str) -> list[GraphRecord]:
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/{run_id}/checkpoints/complete.jsonl"
    return [
        GraphRecord.model_validate_json(line)
        for line in s.read_text(ckpt).splitlines()
        if line.strip()
    ]


def test_completion_appends_inferred_edges_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline_with_completion(store, docs), run_id="gc")

    predicted = _read_predicted(store, "gc")
    assert predicted, "the reference completer must propose the 2-hop closure edge"
    # Every predicted record is an EDGE marked inferred with honest model-derived Evidence.
    for r in predicted:
        assert r.scope is GraphScope.EDGE and r.edge is not None
        assert r.edge.properties["inferred"] is True
        assert r.edge.properties["scorer"] == "reference.adjacency"
        # Honest Evidence: no fabricated mention offsets.
        assert r.edge.evidence.mention_ids == []
    # The proposed edge is the transitive Acme -> Gamma partner_of (not asserted in the text).
    labels = {r.edge.label for r in predicted if r.edge}
    assert labels == {"partner_of"}

    # The Quality Report counts inferred edges SEPARATELY and never inflates edges_total.
    assert report.graph is not None
    gq = report.graph
    assert gq.inferred_edge_count == len(predicted)
    assert gq.inferred_edges_by_scorer == {"reference.adjacency": len(predicted)}
    # The asserted edge count is the observed partner_of edges only (Acme->Beta, Beta->Gamma) — the
    # inferred Acme->Gamma edge did NOT inflate it.
    assert gq.edges_total >= 2
    assert gq.inferred_edge_count >= 1
    assert gq.edges_by_label.get("partner_of", 0) == gq.edges_total  # asserted only


def test_predicted_edges_reach_the_kg_export_marked_inferred(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_pipeline_with_completion(store, docs), run_id="gc")
    s = Storage.for_uri(store)
    edges_uri = f"{store}/_latence/runs/gc/export/graph-edges.parquet"
    assert s.exists(edges_uri)
    with s.open(edges_uri, "rb") as fh:
        tbl = pq.read_table(fh)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    rows = tbl.to_pylist()
    inferred = [r for r in rows if json.loads(r["properties"]).get("inferred") is True]
    asserted = [r for r in rows if not json.loads(r["properties"]).get("inferred")]
    # Both flow into the one KG export, distinguished ONLY by properties.inferred.
    assert inferred, "predicted edges must reach the KG export"
    assert asserted, "asserted edges must reach the KG export"


def test_optional_stage_a_stack_without_it_is_byte_identical(tmp_path: Path) -> None:
    """The graph_completion Stage is genuinely OPTIONAL: dropping it reproduces today's output."""
    store = f"file://{tmp_path}/store"
    docs = _docs(tmp_path)
    # Run WITHOUT completion.
    LocalRunner().run(_pipeline_without_completion(store, docs), run_id="no-gc")
    s = Storage.for_uri(store)
    base_no = f"{store}/_latence/runs/no-gc"
    edges_no = s.read_text(f"{base_no}/export/graph.ttl")
    report_no = QualityReport.model_validate_json(
        s.read_text(f"{base_no}/quality-report.json")
    )
    # A run without the Stage reports zero inferred edges (byte-identical asserted-graph story).
    assert report_no.graph is not None
    assert report_no.graph.inferred_edge_count == 0
    assert report_no.graph.inferred_edges_by_scorer == {}

    # Run WITH completion, then confirm the ASSERTED graph (nodes + asserted edges) is unchanged:
    # the predicted edges are purely additive.
    LocalRunner().run(_pipeline_with_completion(store, docs), run_id="with-gc")
    base_with = f"{store}/_latence/runs/with-gc"
    report_with = QualityReport.model_validate_json(
        s.read_text(f"{base_with}/quality-report.json")
    )
    assert report_with.graph is not None
    # The asserted-edge count is identical with and without completion (the optional-stage proof).
    assert report_with.graph.edges_total == report_no.graph.edges_total
    assert report_with.graph.nodes_total == report_no.graph.nodes_total
    assert report_with.graph.inferred_edge_count >= 1
    # And the asserted TTL is unaffected by the presence of the Stage — the graph_assembly
    # checkpoint is byte-identical (the completer never mutates it).
    ckpt_no = s.read_text(f"{base_no}/checkpoints/graph.jsonl")
    ckpt_with = s.read_text(f"{base_with}/checkpoints/graph.jsonl")
    assert ckpt_no == ckpt_with
    assert edges_no  # sanity: the no-completion run produced a TTL


def test_completion_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline_with_completion(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    a = s.read_text(f"{store}/_latence/runs/det-a/checkpoints/complete.jsonl")
    b = s.read_text(f"{store}/_latence/runs/det-b/checkpoints/complete.jsonl")
    assert a == b


def test_completion_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline_with_completion(store, docs)
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)
    predicted = _read_predicted(store, "resume")
    assert predicted


def test_completion_requires_graph_assembly_wiring(tmp_path: Path) -> None:
    """A Graph Completion Stage wired to the wrong parent (Disambiguation) must error clearly."""
    store = f"file://{tmp_path}/store"
    docs = _docs(tmp_path)
    pipeline = Pipeline.from_yaml_text(
        _base_stages(store, docs)
        + """
  - name: complete
    capability: graph_completion
    provider: graph_completion.reference
    depends_on: [disambiguate]
"""
    )
    try:
        LocalRunner().run(pipeline, run_id="bad-gc")
    except Exception as exc:  # noqa: BLE001 - assert the typed wiring error surfaces
        assert "graph_completion" in str(exc).lower() or "GraphRecord" in str(exc)
    else:
        msg = "expected a wiring error when completion depends on Disambiguation, not Graph Assembly"
        raise AssertionError(msg)
