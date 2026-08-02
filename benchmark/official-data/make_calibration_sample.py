#!/usr/bin/env python3
"""Draw the DETERMINISTIC stratified gamma-calibration sample for an official export.

No new deps (pyarrow only, already a project dep). Usage:

    python3 benchmark/official-data/make_calibration_sample.py musique-full 250

Sampling rule (no cherry-picking, reproducible byte-for-byte):
  1. sort all hyperedges by (gamma, hyperedge_id) ascending
  2. step = len // n ; take indices 0, step, 2*step, ... until n rows are taken

Entity node ids are joined to canonical names + types via graph-nodes.parquet, and the
relations JSON is expanded to head/label/tail triples with canonical names. The result is
written as a JSONL judging worksheet next to the export.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "musique-full"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    export = ROOT / dataset / "export"

    he = pq.read_table(
        export / "graph-hyperedges.parquet",
        columns=[
            "hyperedge_id",
            "entity_node_ids",
            "relations",
            "tau_text",
            "gamma",
            "support",
            "has_relations",
            "arity",
            "properties",
        ],
    ).to_pylist()
    total = len(he)
    # A chunk can emit several Hyperedges (arity>max_arity splits, multiple relation
    # windows). Judging "materially incomplete" needs that context: an entity absent from
    # THIS Hyperedge but present in a sibling over the same chunk is a deliberate split,
    # not a recall miss.
    by_chunk: dict[str, list[dict]] = {}
    for r in he:
        cid = json.loads(r["properties"])["primary_chunk_record_id"]
        r["_chunk"] = cid
        by_chunk.setdefault(cid, []).append(r)
    he.sort(key=lambda r: (r["gamma"], r["hyperedge_id"]))
    step = total // n
    sample = [he[i * step] for i in range(n)]
    assert len({r["hyperedge_id"] for r in sample}) == n

    wanted: set[str] = set()
    for r in sample:
        wanted.update(json.loads(r["entity_node_ids"]))
        for sib in by_chunk[r["_chunk"]]:
            wanted.update(json.loads(sib["entity_node_ids"]))

    nodes: dict[str, dict] = {}
    pf = pq.ParquetFile(export / "graph-nodes.parquet")
    for batch in pf.iter_batches(
        batch_size=20000, columns=["node_id", "type", "canonical_name"]
    ):
        for row in batch.to_pylist():
            if row["node_id"] in wanted:
                nodes[row["node_id"]] = row

    out = ROOT / dataset / "calibration-sample.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for idx, r in enumerate(sample):
            ids = json.loads(r["entity_node_ids"])
            members = [
                {
                    "node_id": i,
                    "name": nodes.get(i, {}).get("canonical_name"),
                    "type": nodes.get(i, {}).get("type"),
                }
                for i in ids
            ]
            triples = []
            for rel in json.loads(r["relations"]):
                triples.append(
                    {
                        "head": nodes.get(rel.get("head_node_id"), {}).get(
                            "canonical_name"
                        ),
                        "label": rel.get("label"),
                        "tail": nodes.get(rel.get("tail_node_id"), {}).get(
                            "canonical_name"
                        ),
                        "confidence": rel.get("confidence"),
                    }
                )
            own = set(ids)
            elsewhere = sorted(
                {
                    nodes[i]["canonical_name"]
                    for sib in by_chunk[r["_chunk"]]
                    for i in json.loads(sib["entity_node_ids"])
                    if i not in own
                }
            )
            fh.write(
                json.dumps(
                    {
                        "chunk_siblings": len(by_chunk[r["_chunk"]]) - 1,
                        "extracted_elsewhere_in_chunk": elsewhere,
                        "i": idx,
                        "rank": idx * step,
                        "hyperedge_id": r["hyperedge_id"],
                        "gamma": r["gamma"],
                        "arity": r["arity"],
                        "support": r["support"],
                        "has_relations": r["has_relations"],
                        "members": members,
                        "relations": triples,
                        "tau_text": r["tau_text"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    grid = [sample[i]["gamma"] for i in (0, n // 4, n // 2, 3 * n // 4, n - 1)]
    print(f"{dataset}: {total} hyperedges, step={step}, wrote {n} rows -> {out}")
    print("gamma at 0/25/50/75/100 pct of sample:", grid)
    missing = sum(1 for i in wanted if i not in nodes)
    print("unresolved node ids:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
