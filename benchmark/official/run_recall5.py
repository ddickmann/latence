# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Ticket 08's remaining deliverable: paragraph-level recall@5 (HippoRAG-2 comparability).

Left as an open note by ticket 17; the close-out audit flagged the demotion, so it is
delivered here on the SAME pre-registered test files the official gate scored — no tuning,
no free parameters, one scoring. Baseline ranks paragraphs by best BM25 chunk (the
inverted index carries Bm25Rescorer's exact scoring math, validated in the official run);
CoE ranks by best hyperedge rho mapped through its primary chunk's source file. Reported
per entry tier: tier1 (zero-model default install) and tier2 (the tuned gate-winning
config, opt-in query NER — surfaces precomputed on the pod, query-surfaces.jsonl).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "musique-subset"))

from official_eval import chunk_file_map, entry_signals, hyperedge_file_map  # noqa: E402
from run_eval_final import load_chunks  # noqa: E402
from run_official_eval import BM25Index  # noqa: E402

from latence_retrieval.hypergraph import HyperedgeKV  # noqa: E402

DATA = HERE.parent / "official-data"
TIERS = ["tier1", "tier2"]

_W: dict = {}


def _init(base: str) -> None:
    base_p = Path(base)
    chunks = load_chunks(base_p / "export" / "corpus.jsonl")
    _W["bm25"] = BM25Index(chunks)
    _W["chunk_files"] = chunk_file_map(base_p / "export" / "corpus.jsonl")
    _W["he_files"] = hyperedge_file_map(
        str(base_p / "export" / "graph-hyperedges.parquet"), _W["chunk_files"]
    )
    _W["surfaces"] = {
        json.loads(l)["id"]: json.loads(l)["surfaces"]
        for l in (base_p / "query-surfaces.jsonl").open()
    }
    _W["kv_path"] = str(base_p / "export" / "graph-hyperedges.kv")


def _eval_q(q: dict) -> dict:
    sup_files = {p["file"] for p in q["paragraphs"] if p["is_supporting"]}
    out: dict = {"total": len(sup_files)}
    base_files: list[str] = []
    for c in _W["bm25"].rank(q["question"], top=400):
        f = _W["chunk_files"].get(c.id)
        if f and f not in base_files:
            base_files.append(f)
        if len(base_files) == 5:
            break
    out["base"] = len(sup_files & set(base_files))
    with HyperedgeKV.open(_W["kv_path"]) as kv:
        for tier in TIERS:
            sig = entry_signals(kv, q, _W["surfaces"], tier, None)
            coe_files: list[str] = []
            for hid in sorted(sig.rho, key=lambda h: -sig.rho[h]):
                f = _W["he_files"].get(hid)
                if f and f not in coe_files:
                    coe_files.append(f)
                if len(coe_files) == 5:
                    break
            out[tier] = len(sup_files & set(coe_files))
    return out


def run(name: str, workers: int, limit: int | None) -> dict:
    base = DATA / name
    stem = name.replace("-full", "")
    questions = [json.loads(l) for l in (base / f"{stem}-full-test.jsonl").open()]
    if limit:
        questions = questions[:limit]
    agg = {"total": 0, "base": 0, **{t: 0 for t in TIERS}}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init, initargs=(str(base),)
    ) as pool:
        for i, res in enumerate(pool.map(_eval_q, questions, chunksize=8)):
            for k in agg:
                agg[k] += res[k]
            if (i + 1) % 200 == 0:
                print(f"  {name} …{i + 1}/{len(questions)} "
                      f"({(time.time() - t0) / (i + 1):.2f}s/q)", flush=True)
    out = {
        "n_questions": len(questions),
        "n_supporting": agg["total"],
        "baseline_recall@5": round(agg["base"] / agg["total"], 4),
        **{f"coe_{t}_recall@5": round(agg[t] / agg["total"], 4) for t in TIERS},
    }
    print(name, json.dumps(out), flush=True)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "recall5-results.json"))
    args = ap.parse_args()
    results = {name: run(name, args.workers, args.limit)
               for name in ["musique-full", "hotpot-full"]}
    if not args.limit:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
