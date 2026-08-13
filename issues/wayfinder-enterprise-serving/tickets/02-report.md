# Ticket 02 — Qdrant serving-path parity: report

Date: 2026-08-12/13. All numbers from a **live Qdrant v1.19.0** (Docker, local Mac), real S3
exports, real S5 gold queries. No fake-backed numbers anywhere in this report.

## Verdict

**Deployed with Qdrant, same recall — measured on the full query sets.** With the export loaded
at `distance: dot` and the shipped backend serving at `search_params: {hnsw_ef: 512}`, the
production `GraphRescueRetrieval` composition returns the brute-force bench reference's recall
to within −0.0004 absolute on every metric over the **full 12,576-query wiki2multihop set**
(R@10 −0.000278, docR@5 −0.000398, docR@10 −0.000278 — well inside ANN noise; the bare dense
lane's ANN recall@10 is 0.9995), and exactly (all deltas 0.0) on multihop_rag.

## The deployment

- `deploy/qdrant/docker-compose.yml` — single-node Qdrant, pinned `v1.19.0`, persistent volume,
  ports 6333 (HTTP) / 6334 (gRPC).
- Loaded through the shipped sink CLI (payload columns verbatim, KG metadata included):

  ```
  .venv/bin/latence-sink-qdrant load benchmark/sota-corpus/s3-exports/wiki2multihop \
      --collection wiki2multihop-dot --basename corpus --distance dot --url http://localhost:6333
  ```

- Parity harness: `benchmark/serving/parity.py` (resumable, fixed-seed sampling = the S5
  `SAMPLE_SEED`, per-query JSONL + committed `.summary.json` aggregates under
  `benchmark/serving/results/`).

## Index build stats

| collection | points | dim | distance | HNSW m | ef_construct | segments | on-disk | load wall |
|---|---|---|---|---|---|---|---|---|
| wiki2multihop-dot | 57,812 | 768 | Dot | 16 (default) | 100 (default) | 5 | 337 MB | 65 s |
| multihop_rag-dot | 3,264 | 768 | Dot | 16 (default) | 100 (default) | 5 | 46 MB | 4.5 s |

Load reports (shipped sink, batch 256, HTTP): `upserted == rows read`, 0 repeats, reconcile
sweep converged to 0 stale points on every load. Idempotency verified live by the sink's own
smoke (`re-load ⇒ 0 upserted / N unchanged`, retraction ⇒ `verify_absent == []`).

## Contract fixes made (shipped code)

The round trip broke in two places and mis-served in a third; all three are fixed in shipped
code with offline tests whose fakes reject what the real store rejects, validated against the
live store in this run:

1. **Identity did not survive the trip** (`latence_retrieval.qdrant.QdrantBackend`). Qdrant
   rejects arbitrary string point ids, so the sink derives a UUIDv5 from the content-addressed
   `record_id` and stores the `record_id` in the payload — but the backend returned the UUID as
   `Candidate.id`. Every downstream consumer (RRF dedupe against the BM25 lane, gold scoring,
   `GraphRescueRetrieval`'s append-and-dedupe, hyperedge→chunk resolution) keys on the corpus
   id, so the served lane's candidates could never merge with anything. Fix: `id_key` config
   (default `"record_id"` — the two shipped halves agree with no configuration, mirroring
   `text_key`), graceful-absent for foreign collections, loud on malformed values, consumed key
   dropped from metadata.
2. **No ANN accuracy knobs** (`QdrantBackend`). `query_points` was issued with the store's
   default search params, so the recall/latency trade could not be tuned or even measured. Fix:
   `search_params` config (`hnsw_ef`, `exact`), validated offline at construction, threaded as
   `models.SearchParams`. The offline fake's explicit `query_points` signature carries the new
   kwarg, so a kwarg the real API lacks still fails offline.
3. **Score convention: Qdrant `cosine` re-normalises stored vectors** (documented on
   `QdrantSink`'s `distance` key — measured, not guessed). The export's embedder L2-normalises,
   but the GPU embed leaves norms ~±0.4% off unit (measured range 0.9982–1.0037 on
   multihop_rag). Qdrant's `cosine` divides by the stored norm at ingest; the bench reference
   ranks by raw `matrix @ query`. The two orderings genuinely differ on near-ties — and the
   difference is **not ANN error**: `exact: true` on a cosine collection reproduces the same
   mismatch (ANN-recall@10 0.9522 at defaults **and** exact, byte-identical). `distance: dot`
   stores vectors verbatim and serves the raw inner product — the score-convention-faithful
   load for pre-normalised exports, and the one this validation uses. (Recall impact of the
   cosine convention was itself negligible on the sample: R@10 delta −0.002, docR@10 +0.007 —
   it reorders near-ties, it does not lose gold.)

## Recall parity (brute-force reference vs served Qdrant lane)

Both sides run the **production** `GraphRescueRetrieval` (graph_mode `auto`, selector `none` =
the ranked union; the selectors are deterministic functions of the union, so identical unions ⇒
identical selections). The only variable is the dense engine: exact brute force (bench
`DenseBackend` + the same KG-metadata candidate contract) vs `QdrantBackend` over the loaded
collection. `ann-recall@k` is the bare dense lane against the exact top-k (uncontaminated by
fusion).

### Tuning sweep (wiki2multihop, 57,812 points, fixed-seed 200-query sample, dot collection)

| search_params | ann@10 | ann@100 | R@10 Δ | docR@5 Δ | docR@10 Δ | top-10 identical | served med/p95 ms |
|---|---|---|---|---|---|---|---|
| default | 0.9880 | 0.9738 | −0.0125 | −0.0113 | −0.0125 | 76.0% | 72.9 / 116.0 |
| hnsw_ef 256 | 0.9985 | 0.9960 | −0.0025 | — | −0.0025 | 93.0% | 76.3 / 189.4 |
| **hnsw_ef 512** | **0.9995** | **0.9994** | **0.0** | **0.0** | **0.0** | **98.0%** | 60.1 / 168.8 |
| hnsw_ef 1024 | 1.0 | 0.9999 | 0.0 | 0.0 | 0.0 | 99.0% | 60.3 / 159.1 |
| exact | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 99.5% | 56.4 / 167.6 |

Operating point: **hnsw_ef = 512** (zero recall delta at ANN latency; `exact` recorded as the
rescoring bound). Residual non-identical top-10s at exact are float-noise ties (brute float32
matmul vs the store's own float32 accumulation order), not missing gold — recall deltas are 0.0.

### multihop_rag (3,264 points, 46-query fixed-seed smoke, dot collection)

Exact parity at every setting tried (default ef, 256, exact): ann@10 = ann@100 = 1.0, all
recall deltas 0.0, top-10 identical 46/46.

### Full-set numbers (the gate numbers) — hnsw_ef 512, dot collections

wiki2multihop: **all 12,576 usable queries** (12,576/12,576 resolved). multihop_rag: all 2,255
usable queries (of 2,556; the unusable ones are the S5 gold-resolution exclusions, identical
for both sides).

| dataset | metric | brute-force reference | served (Qdrant) | delta |
|---|---|---|---|---|
| wiki2multihop (12,576 q) | R@10 | 0.708129 | 0.707851 | **−0.000278** |
| wiki2multihop | docR@5 | 0.670165 | 0.669768 | **−0.000398** |
| wiki2multihop | docR@10 | 0.718333 | 0.718054 | **−0.000278** |
| multihop_rag (2,255 q) | R@10 | 0.375756 | 0.375756 | **0.0** |
| multihop_rag | docR@5 | 0.699076 | 0.699076 | **0.0** |
| multihop_rag | docR@10 | 0.854176 | 0.854176 | **0.0** |

Agreement observability (full sets): wiki2multihop — bare-dense ANN recall@10 **0.999491** /
@100 **0.999427**, full-pipeline top-10 lists byte-identical on **98.16%** of queries, gate
decisions identical on **100%** (gate hit rate 0.9998 — 2Wiki questions are almost all
relation-shaped, so the graph lane runs nearly everywhere and the payload's KG columns are
exercised on nearly every query). multihop_rag — ANN recall@10 = @100 = **1.0**, top-10
identical **99.96%**, gate agreement **100%**.

## Serving latency (per query, end-to-end through the production composition)

Local single-node store over HTTP, one query at a time (no batching, no concurrency), Mac
host. `dense lane` is the bare first-stage `QdrantBackend.search` (top-100); the pipeline rows
are the whole `GraphRescueRetrieval.retrieve` (dense + BM25 + gate + expander + traversal +
hypergraph + union).

| dataset | measurement | median ms | p95 ms | mean ms |
|---|---|---|---|---|
| wiki2multihop (12,576 q) | dense lane (Qdrant, top-100) | 28.1 | 57.2 | 32.0 |
| wiki2multihop | pipeline, brute-force reference | 67.2 | 185.1 | 90.6 |
| wiki2multihop | pipeline, served (Qdrant) | 82.6 | 213.4 | 104.9 |
| multihop_rag (2,255 q) | dense lane (Qdrant, top-100) | 23.5 | 47.9 | 26.9 |
| multihop_rag | pipeline, brute-force reference | 22.2 | 39.5 | 26.5 |
| multihop_rag | pipeline, served (Qdrant) | 42.2 | 83.1 | 50.4 |

On wiki2multihop the served pipeline costs +15 ms median over the in-process brute scan (an
HTTP round trip per query at top-100 with full payloads); the brute reference only *looks*
cheap because a 57,812 × 768 matrix is resident in the process — the served path is the one
that scales past a single host's RAM, and its whole-pipeline p95 stays near 0.2 s.

## Fake-vs-real validation

- `packages/latence-sink-qdrant` live smoke (`LATENCE_QDRANT_URL`): **3/3 green** against the
  real v1.19.0 store — load/converge/retraction-absent, and the sparse-vector leg the fake
  cannot stand in for (named sparse index served, term weights in the vector not the payload).
- `packages/latence-retrieval` live backend smoke: green against the loaded 57,812-point
  collection.
- Payload contract round trip (live): top hit's `Candidate.id == record_id`,
  `text == content`, and all four KG columns (`kg_node_ids`, `neighbor_node_ids`,
  `kg_node_community`, `kg_node_centrality`) byte-equal to the export row; `record_id` not
  duplicated into metadata.
- `search_params` demonstrably reach the live store: the ef sweep moves ann-recall@10 from
  0.9880 → 1.0 with everything else fixed.
- Offline suites (the merge gate): `PYTHONHASHSEED=0 pytest packages/latence-retrieval/tests
  packages/latence-sink-qdrant/tests` → **819 passed, 16 skipped** (skips = live-gated +
  optional-dep tests); with `LATENCE_QDRANT_URL` set the live-gated tests pass against the real
  store. `ruff check` clean on both packages + the harness; `mypy --strict
  --explicit-package-bases` clean on every touched/new file (`qdrant.py`, `sink.py`,
  `benchmark/serving/*`).

## Licenses (verified 2026-08-12)

| dependency | version | license | verification |
|---|---|---|---|
| qdrant-client | 1.19.0 | Apache-2.0 | PyPI metadata `License-Expression: Apache-2.0` |
| qdrant (server) | v1.19.0 | Apache-2.0 | github.com/qdrant/qdrant LICENSE |
| grpcio | 1.83.0 | Apache-2.0 | PyPI metadata |
| protobuf | 7.35.1 | BSD-3-Clause | PyPI classifier |
| portalocker | 3.2.0 | BSD-3-Clause | PyPI metadata |
| h2 / hpack / hyperframe | 4.4.1 / 4.2.0 / 6.1.0 | MIT | PyPI metadata |

No model weights involved (a vector store is data infrastructure; ADR-0012 weights-vs-code
split not applicable to the store itself).

## Residuals

- BM25 stays local by design (fast + exact); hybrid-in-Qdrant not needed — parity reached on
  the dense lane alone.
- The graph lanes (traversal + hypergraph) run identically on both sides; their latency is
  ticket 01's surface, not this one's.
