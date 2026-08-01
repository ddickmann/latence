# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Run the enterprise-gold pipeline through the real ``LocalRunner`` (ticket 21, part D).

    PYTHONHASHSEED=0 uv run python benchmark/enterprise-gold/run_pipeline.py

Reads ``pipeline.yaml`` next to this file, substitutes ``${REPO_ROOT}`` (core's YAML has
no env expansion), runs it under ``run_id`` ``gold-v0`` with storage under
``benchmark/enterprise-gold/store``, then reports what the export actually contains:
hyperedge count + strong fraction + node/edge counts + the sidecar KV — so a thin run is
visible instead of silently shipping.

Re-running is safe: the runner resumes from its checkpoints. Pass ``--fresh`` to delete
the store first (the honest way to re-measure after a corpus or config change).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUN_ID = "gold-v0"


def _load_pipeline():  # noqa: ANN202 - the concrete type lives in latence_core
    from latence_core.pipeline import Pipeline

    text = (HERE / "pipeline.yaml").read_text(encoding="utf-8")
    return Pipeline.from_yaml_text(text.replace("${REPO_ROOT}", str(REPO_ROOT)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true", help="delete the store before running (no resume)"
    )
    args = parser.parse_args(argv)

    from latence_core.runner import LocalRunner
    from latence_core.storage import Storage

    store_dir = HERE / "store"
    if args.fresh and store_dir.exists():
        shutil.rmtree(store_dir)

    pipeline = _load_pipeline()
    report = LocalRunner().run(pipeline, run_id=RUN_ID)

    storage = Storage.for_uri(pipeline.storage_uri)
    base = f"{pipeline.storage_uri}/_latence/runs/{RUN_ID}"
    export = f"{base}/export"

    import pyarrow.parquet as pq

    def _rows(uri: str) -> list[dict]:
        if not storage.exists(uri):
            return []
        with storage.open(uri, "rb") as fh:
            return pq.read_table(fh).to_pylist()  # type: ignore[no-untyped-call]

    hyperedges = _rows(f"{export}/graph-hyperedges.parquet")
    nodes = _rows(f"{export}/graph-nodes.parquet")
    edges = _rows(f"{export}/graph-edges.parquet")
    corpus_uri = f"{export}/records.jsonl"
    corpus_lines = storage.read_text(corpus_uri).splitlines() if storage.exists(corpus_uri) else []
    chunks = [json.loads(line) for line in corpus_lines if line.strip()]

    strong = [h for h in hyperedges if h["has_relations"]]
    masked = [c for c in chunks if c.get("masked_content")]
    kv_uri = f"{export}/graph-hyperedges.kv"
    print(f"export dir:        {export}")
    print(f"documents/chunks:  {len({c['document_record_id'] for c in chunks})} / {len(chunks)}")
    print(f"chunks masked:     {len(masked)}")
    print(f"nodes / edges:     {len(nodes)} / {len(edges)}")
    print(f"hyperedges:        {len(hyperedges)}")
    if hyperedges:
        print(f"  strong (has_relations): {len(strong)} ({len(strong) / len(hyperedges):.1%})")
        gammas = sorted(h["gamma"] for h in hyperedges)
        supports = sorted(h["support"] for h in hyperedges)
        mid = len(gammas) // 2
        print(f"  gamma  min/median/max:  {gammas[0]:.3f} / {gammas[mid]:.3f} / {gammas[-1]:.3f}")
        print(f"  support min/median/max: {supports[0]} / {supports[mid]} / {supports[-1]}")
    print(f"hyperedge KV:      {'present' if storage.exists(kv_uri) else 'ABSENT'}")
    if report.graph is not None and report.graph.hyperedges is not None:
        hq = report.graph.hyperedges
        print(
            f"quality roll-up:   hyperedges_total={hq.hyperedges_total} "
            f"relations_total={hq.relations_total} "
            f"token_cost {hq.token_cost_min}..{hq.token_cost_max}"
        )
    if len(hyperedges) < 200:
        print(
            f"WARNING: {len(hyperedges)} hyperedges — calibration (ticket 09) wants >= 200.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
