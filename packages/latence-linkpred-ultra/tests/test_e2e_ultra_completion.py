"""End-to-end: graph_completion.ultra through the local Runner (offline, faithful fake scorer).

Source -> Parse -> Chunk -> Entity -> Relation -> Disambiguation -> Graph Assembly ->
Graph Completion (graph_completion.ultra) -> KG Export. The ULTRA/PyKEEN scorer deps are absent in
the CPU dev env, so the Scorer is monkeypatched with the FAITHFUL FakeScorer from conftest (the
Scorer INTERFACE is real; only the learned floats are stubbed). Asserts the learned Provider drops
into the Slice-1 seam and the inferred-edge contract exactly, the Quality Report counts inferred
edges separately, the predicted edges reach the KG export, and the run is deterministic.

The REAL ULTRA MRR/Hits@k are re-validated on the pod (scripts/validate-linkpred-on-gpu.md); this
e2e proves the wiring + contract + determinism offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from _linkpred_fakes import FakeScorer
from latence_core.contracts import GraphRecord, GraphScope
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage
from latence_linkpred_ultra import provider as provider_mod

_CORPUS = {
    "a.txt": "Acme partners with Beta on the venture.\n",
    "b.txt": "Beta partners with Gamma in the consortium.\n",
}


@pytest.fixture(autouse=True)
def _fake_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire the faithful fake Scorer in place of the (absent) real ULTRA/PyKEEN deps."""
    monkeypatch.setattr(provider_mod, "UltraScorer", FakeScorer)
    monkeypatch.setattr(provider_mod, "PykeenScorer", FakeScorer)


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in _CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    return docs


def _base_stages(store: str, docs_dir: Path) -> str:
    return f"""
name: gc-ultra-slice
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
      corpus_id: gc-ultra-slice
"""


def _pipeline_with_ultra(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        _base_stages(store, docs_dir)
        + """
  - name: complete
    capability: graph_completion
    provider: graph_completion.ultra
    depends_on: [graph]
    config:
      corpus_id: gc-ultra-slice
      scorer: ultra
      min_probability: 0.5
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph, complete]
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


def test_ultra_completion_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline_with_ultra(store, docs), run_id="gc")

    predicted = _read_predicted(store, "gc")
    assert predicted, "the learned completer proposes the transitive Acme->Gamma partner_of edge"
    for r in predicted:
        assert r.scope is GraphScope.EDGE and r.edge is not None
        props = r.edge.properties
        # Slice-1 inferred-edge contract EXACTLY, now with the learned scorer + calibration.
        assert props["inferred"] is True
        assert props["scorer"] == "ultra"
        assert props["calibrated"] is True
        assert r.edge.evidence.mention_ids == []  # honest empty-mention Evidence
        assert r.edge.label == "partner_of"

    # The Quality Report counts inferred edges SEPARATELY (never inflating asserted counts).
    assert report.graph is not None
    gq = report.graph
    assert gq.inferred_edge_count == len(predicted)
    assert gq.inferred_edges_by_scorer == {"ultra": len(predicted)}
    assert gq.edges_by_label.get("partner_of", 0) == gq.edges_total  # asserted only


def test_predicted_edges_reach_kg_export_marked_inferred(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_pipeline_with_ultra(store, docs), run_id="gc")
    s = Storage.for_uri(store)
    edges_uri = f"{store}/_latence/runs/gc/export/graph-edges.parquet"
    assert s.exists(edges_uri)
    with s.open(edges_uri, "rb") as fh:
        tbl = pq.read_table(fh)  # type: ignore[no-untyped-call]
    rows = tbl.to_pylist()
    inferred = [r for r in rows if json.loads(r["properties"]).get("inferred") is True]
    asserted = [r for r in rows if not json.loads(r["properties"]).get("inferred")]
    assert inferred, "predicted edges must reach the KG export"
    assert asserted, "asserted edges must reach the KG export"
    # The learned scorer stamped 'ultra' on the inferred edges.
    assert all(json.loads(r["properties"]).get("scorer") == "ultra" for r in inferred)


def test_deterministic_across_runs(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_pipeline_with_ultra(store, docs), run_id="r1")
    LocalRunner().run(_pipeline_with_ultra(store, docs), run_id="r2")
    a = _read_predicted(store, "r1")
    b = _read_predicted(store, "r2")
    assert [r.model_dump_json() for r in a] == [r.model_dump_json() for r in b]
