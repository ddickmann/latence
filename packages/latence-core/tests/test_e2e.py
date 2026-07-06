"""End-to-end: the example pipeline over a temp folder through the local Runner.

Asserts JSONL + Parquet outputs exist and parse, records carry Provenance from
Stage 1, and a Quality Report JSON is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def test_end_to_end(docs_dir: Path, tmp_path: Path) -> None:
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: walking-skeleton
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [parse]
"""
    )

    report = LocalRunner().run(pipeline, run_id="e2e")
    s = Storage.for_uri(store)
    base = f"{store}/_latence/runs/e2e"

    # 1. JSONL exists and parses; records carry Provenance from Stage 1.
    jsonl_uri = f"{base}/export/records.jsonl"
    assert s.exists(jsonl_uri)
    rows = [json.loads(line) for line in s.read_text(jsonl_uri).splitlines() if line.strip()]
    assert len(rows) == 2  # the two text docs; ignore.bin is skipped
    for row in rows:
        prov = row["provenance"]
        assert prov["file_name"] in {"hello.txt", "notes.md"}
        assert prov["file_type"] in {"txt", "md"}
        assert prov["file_size"] > 0
        assert prov["document_id"].startswith("sha256:")
        assert row["classification"]["sensitivity"] == "internal"
        assert row["media_type"] == "text/markdown"  # set by the passthrough Parser

    # 2. Parquet exists and parses back to the same record ids.
    parquet_uri = f"{base}/export/records.parquet"
    assert s.exists(parquet_uri)
    with s.open(parquet_uri, "rb") as fh:
        table = pq.read_table(fh)
    assert table.num_rows == 2
    assert set(table.column_names) >= {"record_id", "provenance", "classification", "content"}

    # 3. Quality Report JSON produced and internally consistent.
    report_uri = f"{base}/quality-report.json"
    assert s.exists(report_uri)
    persisted = QualityReport.model_validate_json(s.read_text(report_uri))
    assert persisted.document_count == 2
    assert persisted.contracts_complete is True
    assert persisted.completeness.passed is True
    assert {m.name for m in persisted.stages} == {"source", "parse", "export"}
    # The returned report matches what was persisted.
    assert report.document_count == persisted.document_count
