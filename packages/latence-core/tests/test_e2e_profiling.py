"""End-to-end S7 Profiling through the local Runner.

Source -> Parse -> Chunk -> Entity -> Profiling -> Export, using the in-core
deterministic ``profiling.statistical``. Asserts the exported records are
FeatureRecords (one DOCUMENT-scope per source doc + one CORPUS-scope), that the
per-document features are populated, that the corpus record carries the
cross-document aggregates (source coverage + entity frequency + co-occurrence), that
the Quality Report's ProfilingQuality carries the feature summary (the S7 AC), that
document_count is not inflated by the synthetic corpus record, and that the run is
deterministic and resumes cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

from latence_core.contracts import FeatureRecord, FeatureScope
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: profiling-slice
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
          terms: [Acme, Globex]
        PERSON:
          terms: [Alice, Bob]
  - name: profile
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, entities]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [profile]
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text(
        "Alice works at Acme. Bob also works at Acme. The team is productive.\n",
        encoding="utf-8",
    )
    (docs / "note.txt").write_text(
        "Acme announced a partnership with Globex. Alice leads the effort.\n",
        encoding="utf-8",
    )
    (docs / "empty.txt").write_text("", encoding="utf-8")
    return docs


def test_profiling_slice_end_to_end(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_pipeline(store, docs), run_id="s7e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/s7e2e"

    # 1. The export holds typed FeatureRecord rows.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    records = [FeatureRecord.model_validate(r) for r in rows]
    assert records
    doc_records = [r for r in records if r.scope is FeatureScope.DOCUMENT]
    corpus_records = [r for r in records if r.scope is FeatureScope.CORPUS]

    # 2. One DOCUMENT-scope record per non-empty parsed doc, exactly one CORPUS record.
    assert len(doc_records) == 2
    assert len(corpus_records) == 1
    for dr in doc_records:
        assert dr.document is not None
        assert dr.document.word_count > 0
        assert 0.0 <= dr.document.readability <= 1.0

    # 3. The corpus record carries the cross-document aggregates.
    corpus = corpus_records[0]
    assert corpus.corpus is not None
    cf = corpus.corpus
    assert cf.document_count == 2
    assert cf.sensitivity_coverage == {"internal": 2}
    assert cf.entity_frequency.get("Acme", 0) >= 2
    assert cf.entity_type_frequency.get("ORG", 0) >= 2
    # Alice + Acme co-occur across both documents.
    pairs = {(c.left, c.right): c.document_count for c in cf.co_occurrences}
    assert pairs.get(("Acme", "Alice"), 0) == 2

    # 4. Quality Report: ProfilingQuality present with the feature summary (S7 AC).
    persisted = QualityReport.model_validate_json(s.read_text(f"{base}/quality-report.json"))
    assert persisted.profiling is not None
    pq = persisted.profiling
    assert pq.documents_profiled == 2
    assert pq.corpus_total_words > 0
    assert pq.distinct_entities >= 2
    assert pq.sensitivity_coverage == {"internal": 2}
    assert "Acme|Alice" in pq.top_co_occurrences
    assert report.profiling is not None
    assert report.profiling.documents_profiled == persisted.profiling.documents_profiled

    # 5. document_count reflects the 2 real source documents, not the synthetic corpus.
    assert persisted.document_count == 2

    # 6. Contract completeness holds over the FeatureRecords (Provenance+Classification).
    assert persisted.contracts_complete


def test_profiling_slice_is_deterministic(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="det-a")
    LocalRunner().run(pipeline, run_id="det-b")
    s = Storage.for_uri(store)
    out_a = s.read_text(f"{store}/_latence/runs/det-a/export/records.jsonl")
    out_b = s.read_text(f"{store}/_latence/runs/det-b/export/records.jsonl")
    assert out_a == out_b


def test_profiling_slice_resumes_from_checkpoints(tmp_path: Path) -> None:
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    LocalRunner().run(pipeline, run_id="resume")
    report2 = LocalRunner().run(pipeline, run_id="resume")
    assert all(m.from_checkpoint for m in report2.stages)
    s = Storage.for_uri(store)
    ckpt = f"{store}/_latence/runs/resume/checkpoints/profile.jsonl"
    lines = [x for x in s.read_text(ckpt).splitlines() if x.strip()]
    assert lines
    scopes = {FeatureRecord.model_validate_json(line).scope for line in lines}
    assert FeatureScope.CORPUS in scopes
    assert FeatureScope.DOCUMENT in scopes
