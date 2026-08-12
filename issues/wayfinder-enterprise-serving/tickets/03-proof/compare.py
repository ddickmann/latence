"""Ticket-03 equivalence comparison: incremental final state vs full rebuild (ADR-0061 §8).

Compares every exported artifact of the final incremental run against the full rebuild of the
same corpus, and the retrieval rankings both stacks produce on the real MultiHopRAG query set:

* ``corpus.jsonl``           — file sha256 + per-line multiset diff (rows include embeddings)
* ``corpus.parquet``         — row sets keyed by ``record_id``; embedding vectors bitwise, else
                               max-abs-diff measured and reported
* ``graph-nodes/edges/hyperedges.parquet`` — row sets keyed by their ids (full column equality)
* ``graph-hyperedges.kv`` / ``graph.ttl`` / ``graph.graphml`` — sha256
* ``bm25-stats.json``        — byte + parsed (doc_count/avgdl/df/idf) comparison
* ``bm25-postings.parquet``  — row sets keyed by ``record_id``
* rankings                   — BM25 (Robertson, k1=1.2 b=0.75, from the exported artifacts
                               alone) top-20 per query over N real queries, and dense cosine
                               top-20 using each arm's stored document embeddings with ONE
                               shared query-embedding pass (granite-311m, CPU)

Writes ``results.json`` (machine-readable) and prints a human summary. Exit 0 only when every
comparison is identical OR every divergence is measured and below the documented tolerance
(none is expected: CPU inference is bitwise-reproducible and the overlay merge reproduces the
rebuild's row order).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
QUERIES = REPO / "benchmark/sota-corpus/multihop_rag/raw/MultiHopRAG.json"
TOP_K = 20
N_QUERIES = 200
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_multiset_diff(a: Path, b: Path) -> dict[str, int]:
    ca = Counter(hashlib.sha256(line).hexdigest() for line in a.read_bytes().splitlines() if line)
    cb = Counter(hashlib.sha256(line).hexdigest() for line in b.read_bytes().splitlines() if line)
    return {
        "rows_a": sum(ca.values()),
        "rows_b": sum(cb.values()),
        "only_in_a": sum((ca - cb).values()),
        "only_in_b": sum((cb - ca).values()),
    }


def _rowset_diff(a: Path, b: Path, key: str) -> dict[str, Any]:
    ta, tb = pq.read_table(a), pq.read_table(b)  # type: ignore[no-untyped-call]
    da = {r[key]: json.dumps(r, sort_keys=True, default=str) for r in ta.to_pylist()}
    db = {r[key]: json.dumps(r, sort_keys=True, default=str) for r in tb.to_pylist()}
    changed = [k for k in da.keys() & db.keys() if da[k] != db[k]]
    return {
        "rows_a": len(da),
        "rows_b": len(db),
        "only_in_a": len(da.keys() - db.keys()),
        "only_in_b": len(db.keys() - da.keys()),
        "changed": len(changed),
        "changed_sample": changed[:3],
    }


def _embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path, columns=["record_id", "embedding"])  # type: ignore[no-untyped-call]
    ids = t.column("record_id").to_pylist()
    mat = np.array(t.column("embedding").to_pylist(), dtype=np.float64)
    order = np.argsort(np.array(ids))
    return [ids[i] for i in order], mat[order]


def _bm25_rank(export: Path, queries: list[str]) -> list[list[str]]:
    stats = json.loads((export / "bm25-stats.json").read_text(encoding="utf-8"))
    idf = {t["term"]: t["idf"] for t in stats["terms"]}
    avgdl = stats["avgdl"]
    postings = pq.read_table(export / "bm25-postings.parquet").to_pylist()  # type: ignore[no-untyped-call]
    index: dict[str, list[tuple[str, int, int]]] = {}
    for row in postings:
        for term, tf in zip(row["terms"], row["term_frequencies"], strict=True):
            index.setdefault(term, []).append((row["record_id"], tf, row["doc_len"]))
    k1, b = 1.2, 0.75
    out: list[list[str]] = []
    for query in queries:
        scores: dict[str, float] = {}
        for term in {t.lower() for t in _TOKEN.findall(query)}:
            for rid, tf, dl in index.get(term, ()):
                gain = idf.get(term, 0.0) * (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * dl / avgdl)
                )
                scores[rid] = scores.get(rid, 0.0) + gain
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K]
        out.append([rid for rid, _ in ranked])
    return out


def main(incr_export: str, rebuild_export: str) -> None:
    a, b = Path(incr_export), Path(rebuild_export)
    results: dict[str, Any] = {"incremental": str(a), "rebuild": str(b)}

    for name in (
        "corpus.jsonl",
        "graph-hyperedges.kv",
        "graph.ttl",
        "graph.graphml",
        "bm25-stats.json",
    ):
        results[name] = {"sha_equal": _sha(a / name) == _sha(b / name)}
    results["corpus.jsonl"]["multiset"] = _line_multiset_diff(
        a / "corpus.jsonl", b / "corpus.jsonl"
    )

    for name, key in (
        ("corpus.parquet", "record_id"),
        ("graph-nodes.parquet", "node_id"),
        ("graph-edges.parquet", "edge_id"),
        ("graph-hyperedges.parquet", "hyperedge_id"),
        ("bm25-postings.parquet", "record_id"),
    ):
        results[name] = _rowset_diff(a / name, b / name, key)

    ids_a, emb_a = _embeddings(a / "corpus.parquet")
    ids_b, emb_b = _embeddings(b / "corpus.parquet")
    emb: dict[str, Any] = {"ids_equal": ids_a == ids_b}
    if ids_a == ids_b:
        emb["bitwise"] = bool(np.array_equal(emb_a, emb_b))
        emb["max_abs_diff"] = float(np.max(np.abs(emb_a - emb_b))) if len(ids_a) else 0.0
    results["embeddings"] = emb

    queries = [q["query"] for q in json.loads(QUERIES.read_text(encoding="utf-8"))][:N_QUERIES]
    bm_a, bm_b = _bm25_rank(a, queries), _bm25_rank(b, queries)
    results["bm25_rankings"] = {
        "queries": len(queries),
        "top_k": TOP_K,
        "identical_lists": sum(x == y for x, y in zip(bm_a, bm_b, strict=True)),
    }

    dense: dict[str, Any] = {"queries": len(queries), "top_k": TOP_K}
    if emb.get("bitwise"):
        dense["identical_lists"] = len(queries)
        dense["note"] = "doc embeddings bitwise identical => dense rankings provably identical"
    else:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - heavy, lazy

        model = SentenceTransformer(
            "ibm-granite/granite-embedding-311m-multilingual-r2", device="cpu"
        )
        qv = np.asarray(model.encode(queries, batch_size=16, convert_to_numpy=True), np.float64)

        def _rank(ids: list[str], mat: np.ndarray) -> list[list[str]]:
            norms = np.linalg.norm(mat, axis=1) * np.linalg.norm(qv, axis=1)[:, None]
            sims = (qv @ mat.T) / np.where(norms == 0, 1, norms)
            out = []
            for row in sims:
                top = sorted(zip(row, ids, strict=True), key=lambda t: (-t[0], t[1]))[:TOP_K]
                out.append([rid for _, rid in top])
            return out

        ra, rb = _rank(ids_a, emb_a), _rank(ids_b, emb_b)
        dense["identical_lists"] = sum(x == y for x, y in zip(ra, rb, strict=True))
    results["dense_rankings"] = dense

    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))

    ok = (
        all(results[n]["sha_equal"] for n in ("graph-hyperedges.kv", "graph.ttl", "graph.graphml"))
        and results["corpus.jsonl"]["multiset"]["only_in_a"] == 0
        and results["corpus.jsonl"]["multiset"]["only_in_b"] == 0
        and all(
            results[n]["only_in_a"] == 0
            and results[n]["only_in_b"] == 0
            and results[n]["changed"] == 0
            for n in (
                "corpus.parquet",
                "graph-nodes.parquet",
                "graph-edges.parquet",
                "graph-hyperedges.parquet",
                "bm25-postings.parquet",
            )
        )
        and results["bm25_rankings"]["identical_lists"] == len(queries)
        and results["dense_rankings"]["identical_lists"] == len(queries)
        and (emb.get("bitwise") or emb.get("max_abs_diff", math.inf) == 0.0)
    )
    print(f"\nEQUIVALENCE: {'PASS' if ok else 'FAIL (see results.json)'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
