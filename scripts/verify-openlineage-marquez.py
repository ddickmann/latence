#!/usr/bin/env python
"""End-to-end check of the OpenLineage seam (#185) against a REAL consumer (Marquez).

The offline test suite proves the *shape* of the events (``test_openlineage.py``); this proves the
other half — that a real OpenLineage backend **accepts** them and that the run, its Stage jobs, and
its datasets actually land in the lineage graph. It is deliberately NOT part of ``verify-local.sh``
(that gate is offline and dependency-free); it is the manual gate you run when touching the emitter.

Start a Marquez first::

    docker network create marquez-net
    docker run -d --name marquez-db --network marquez-net \
        -e POSTGRES_USER=marquez -e POSTGRES_PASSWORD=marquez -e POSTGRES_DB=marquez postgres:14
    docker run -d --name marquez-api --network marquez-net -p 5050:5000 -p 5051:5001 \
        -e POSTGRES_HOST=marquez-db -e POSTGRES_PORT=5432 -e POSTGRES_USER=marquez \
        -e POSTGRES_PASSWORD=marquez -e POSTGRES_DB=marquez marquezproject/marquez:0.51.1

then::

    uv run python scripts/verify-openlineage-marquez.py http://localhost:5050

Exits non-zero (and prints what was missing) unless Marquez reports the run's jobs *and* the
datasets they read/write. Marquez is Apache-2.0 (code); it ships no model weights.
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from latence_core.observability import build_lineage_emitter
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner

NAMESPACE = "latence-verify"
PIPELINE_NAME = "lineage-verify"
STAGES = ("source", "parse", "chunk", "export")


def _get(url: str) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - operator-supplied http
        payload: dict[str, object] = json.load(response)
    return payload


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: {PIPELINE_NAME}
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    config:
      max_tokens: 40
      overlap_tokens: 8
      min_tokens: 4
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
"""
    )


def main(base_url: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        docs = root / "docs"
        docs.mkdir()
        (docs / "a.txt").write_text("# Report\n\n" + ("Revenue figures. " * 40), encoding="utf-8")
        emitter = build_lineage_emitter(base_url, namespace=NAMESPACE)
        assert emitter is not None
        report = LocalRunner(lineage=emitter).run(
            _pipeline(f"file://{root}/store", docs), run_id="verify-0001"
        )
        print(f"run finished: {report.document_count} document(s)")

    jobs = _get(f"{base_url}/api/v1/namespaces/{NAMESPACE}/jobs")
    job_names = {str(j["name"]) for j in jobs.get("jobs", [])}  # type: ignore[union-attr]
    expected_jobs = {PIPELINE_NAME, *(f"{PIPELINE_NAME}.{s}" for s in STAGES)}
    missing_jobs = expected_jobs - job_names
    datasets = _get(f"{base_url}/api/v1/namespaces/file/datasets")
    dataset_names = {str(d["name"]) for d in datasets.get("datasets", [])}  # type: ignore[union-attr]
    checkpoints = {n for n in dataset_names if "/checkpoints/" in n}

    print(f"marquez jobs in {NAMESPACE!r}: {sorted(job_names)}")
    print(f"marquez file datasets (checkpoints): {sorted(checkpoints)}")
    if missing_jobs:
        print(f"FAIL: Marquez is missing jobs {sorted(missing_jobs)}", file=sys.stderr)
        return 1
    if len(checkpoints) < len(STAGES):
        print(f"FAIL: expected {len(STAGES)} checkpoint datasets, saw {len(checkpoints)}", file=sys.stderr)
        return 1
    runs = _get(f"{base_url}/api/v1/namespaces/{NAMESPACE}/jobs/{PIPELINE_NAME}/runs")
    states = [str(r["state"]) for r in runs.get("runs", [])]  # type: ignore[union-attr]
    print(f"pipeline run states: {states}")
    if "COMPLETED" not in states:
        print(f"FAIL: pipeline run never reached COMPLETED (states={states})", file=sys.stderr)
        return 1
    print("OK: run, per-Stage jobs, and datasets are all in the Marquez lineage graph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5050"))
