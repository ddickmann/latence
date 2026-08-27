# Serving with Qdrant — the benched stack on a real ANN engine

> **Status: measured, live.** The full result, with method and per-dataset cells, is
> [Served on Qdrant: recall parity](../evidence/enterprise-serving-02-qdrant-serving-parity.md);
> the graph-latency result behind it is
> [Graph latency at scale](../evidence/enterprise-serving-01-latency.md).
> Every number on this page comes from a live Qdrant v1.19.0
> (Docker, single node) serving real pipeline Exports, scored over the **full** gold query sets
> of the e2e benchmark campaign — no mocks anywhere in the measured path
> (ticket `wayfinder-enterprise-serving/02`; raw results in `benchmark/serving/results/`).

The Export — Parquet/JSONL on your Storage — stays the source of truth (ADR-0017). Serving is a
projection of it: `latence-sink-qdrant` loads an Export into a collection, and
`latence-retrieval`'s `QdrantBackend` reads that exact collection back as the first-stage dense
engine of [`GraphRescueRetrieval`](graph-rescue-retrieval.md). This page is the verified path
from "pipeline finished" to "queries served", with the settings the parity run validated.

## The headline, measured

**Deployed with Qdrant, same recall.** With the export loaded at `distance: dot` and the backend
serving at `search_params: {hnsw_ef: 512}`, the production `GraphRescueRetrieval` composition
returns the brute-force reference's recall to within −0.0004 absolute on every metric over the
full 12,576-query 2WikiMultihopQA set, and exactly (all deltas 0.0) on MultiHop-RAG:

| dataset (full query set) | metric | brute-force reference | served (Qdrant) | delta |
|---|---|---|---|---|
| wiki2multihop (12,576 q) | R@10 | 0.708129 | 0.707851 | −0.000278 |
| wiki2multihop | docR@5 | 0.670165 | 0.669768 | −0.000398 |
| wiki2multihop | docR@10 | 0.718333 | 0.718054 | −0.000278 |
| multihop_rag (2,255 q) | R@10 | 0.375756 | 0.375756 | 0.0 |
| multihop_rag | docR@5 | 0.699076 | 0.699076 | 0.0 |
| multihop_rag | docR@10 | 0.854176 | 0.854176 | 0.0 |

The bare dense lane's ANN recall@10 against the exact top-10 is 0.9995 (wiki2multihop) / 1.0
(multihop_rag); policy-gate decisions agree on 100% of queries on both datasets. Serving latency,
single node over HTTP, one query at a time (whole pipeline = dense + BM25 + gate + expander +
traversal + hypergraph + union): wiki2multihop median 82.6 ms / p95 213.4 ms; multihop_rag median
42.2 ms / p95 83.1 ms. The in-process brute-force reference only *looks* cheap because a
57,812 × 768 matrix is resident in the process — the served path is the one that scales past a
single host's RAM.

## 1 · Stand up the store

The pinned single-node compose (`deploy/qdrant/docker-compose.yml` — the exact engine version the
parity run validated):

```bash
cd deploy/qdrant
docker compose up -d
curl -s http://localhost:6333/readyz    # 200 once all shards are ready
```

Ports: 6333 HTTP (the default transport of both halves), 6334 gRPC (opt-in via `prefer_grpc`).
Collections persist in the `qdrant-storage` volume; `docker compose down -v` wipes them. Bump the
pinned tag deliberately, and re-run the live smoke plus the parity harness when you do (the
comment block in the compose file carries both commands).

## 2 · Load the Export

```bash
latence-sink-qdrant load ./out/export --collection corpus \
    --distance dot --url http://localhost:6333
```

One flag matters for parity: **`--distance dot`**. The pipeline's embedder L2-normalises, but a
GPU embed leaves norms ~±0.4% off exact unit (measured 0.9982–1.0037). Qdrant's `cosine` mode
re-normalises stored vectors at ingest, so ranking near-ties can differ from the raw
inner-product order the Export's scores mean — measured, and **not** ANN error (`exact: true`
reproduces the same mismatch). `dot` stores vectors verbatim and serves the raw inner product:
the score-convention-faithful load for pre-normalised exports, and the one every number above
uses. (The recall impact of `cosine` was itself negligible — it reorders near-ties, it does not
lose gold — but there is no reason to accept it.)

The sink is idempotent and reconciling — re-loads upsert nothing for unchanged records and sweep
stale points, and `verify-absent` is the GDPR erasure audit. The full loader contract (sparse
vectors, `id_namespace` multi-tenancy, failure containment, the offline fake):
[`packages/latence-sink-qdrant/README.md`](https://github.com/ddickmann/latence/blob/main/packages/latence-sink-qdrant/README.md).

## 3 · Serve queries through the production composition

```python
from latence_retrieval import GraphRescueRetrieval, QdrantBackend

dense = QdrantBackend(
    {
        "collection": "corpus",
        "url": "http://localhost:6333",
        "search_params": {"hnsw_ef": 512},   # the validated operating point
        # id_key defaults to "record_id" — the sink stores it in the payload, the
        # backend restores it as Candidate.id, so served candidates merge with the
        # BM25 lane, gold scoring and the graph lanes with no configuration.
    }
)

pipeline = GraphRescueRetrieval(
    {"graph_mode": "auto", "selector": "packed",
     "hyperedge_kv_path": "out/export/graph-hyperedges.kv"},
    dense=dense,
    bm25=my_bm25_backend,                    # BM25 stays local: fast + exact by design
    graph=my_graph_retriever,                # zero-ops DuckDB over the exported parquet
    hyperedge_documents=my_hyperedge_chunk_resolver,
)
```

Two backend contracts the parity work fixed, so you do not have to discover them live:

* **Identity survives the round trip.** Qdrant rejects arbitrary string point ids, so the sink
  derives a UUIDv5 and keeps the content-addressed `record_id` in the payload; the backend's
  `id_key` (default `"record_id"`) restores it as `Candidate.id`. Every downstream consumer —
  RRF dedupe, the rescue lanes' append-and-dedupe, hyperedge→chunk resolution — keys on the
  corpus id. A malformed payload value fails loudly; a foreign collection without the key
  degrades gracefully.
* **The ANN accuracy knob is real and threaded.** `search_params` (`hnsw_ef`, `exact`) reaches
  the live store — the measured ef sweep on 57,812 points: default ef → ANN recall@10 0.9880 /
  R@10 delta −0.0125; `hnsw_ef: 256` → 0.9985 / −0.0025; **`hnsw_ef: 512` → 0.9995 / 0.0**;
  `exact: true` → 1.0 (the rescoring bound, at brute-force cost). 512 is the shipped
  operating point: zero recall delta at ANN latency.

The KG payload columns (`kg_node_ids`, `neighbor_node_ids`, `kg_node_community`,
`kg_node_centrality`) ride through the sink verbatim and come back on `Candidate.metadata`, so
the graph rescue lane works over the served store exactly as over files — on 2Wiki the gate
fires on 99.98% of queries, so those columns are exercised on nearly every request.

## Verifying your own deployment

```bash
# live smoke of the loader + backend against your running store (skipped without the env var):
LATENCE_QDRANT_URL=http://localhost:6333 pytest packages/latence-sink-qdrant -k live
# the resumable parity harness the numbers above came from (brute-force vs served):
PYTHONHASHSEED=0 .venv/bin/python benchmark/serving/parity.py --help
```

BM25 deliberately stays a local exact rescorer rather than moving into Qdrant — parity was
reached on the dense lane alone, and the graph lanes run identically on both sides (their
latency story is the [hypergraph index](graph-rescue-retrieval.md), measured separately).
