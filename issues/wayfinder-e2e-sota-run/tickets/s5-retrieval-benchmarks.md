# S5 — Retrieval benchmarks (feature matrix × datasets)

Type: task
Status: claimed
Blocked by: s3-pipeline (complete 2026-08-09)

## Spec (Dennis, 2026-08-09)

**Hop depth: 2–4 native only.** No dataset in this corpus exceeds 4 hops (MultiHop-RAG
`evidence_list`, 2Wiki, MuSiQue all cap at 4). Synthesized 5–7-hop chains over the built KG
were considered and REJECTED for this run: they would compare only against our own baselines,
not published ones. The report states plainly that 5–7 hops are untested.

**Legs in the matrix** (SPLADE, MaxSim/MUVERA and rerank/fusion ablations explicitly OUT):

| # | leg | inputs (all present in the S3 exports) |
|---|---|---|
| 1 | dense | `corpus.parquet` 768-d granite vectors |
| 2 | + bm25 | `bm25-postings.parquet` + `bm25-stats.json` |
| 3 | + KG retrieval | `graph-nodes/edges.parquet` (`graph_retrieval.py`) |
| 4 | + hyperedges | `graph-hyperedges.parquet` + `graph-hyperedges.kv` (`hypergraph.py`) |
| 5 | + chain of evidence | `coe.py` marginal-utility selection over gated ρ signals |
| 6 | + knapsack packer | `pack.py` token-budgeted packing |

Cumulative ladder (each row adds to the previous) so every increment is attributable, plus
each leg standalone where meaningful — the ablation is the point, not one blended number.

## Query inventory (verified 2026-08-09)

| dataset | queries | hops | gold evidence | explicit CoE |
|---|---|---|---|---|
| 2wiki | 12,576 dev | 2–4 | `supporting_facts` | `evidences` (s,r,o) triples |
| multihop_rag | 2,556 | 2–4 | `evidence_list` (doc+url), 4 question types | doc chain |
| musique | M6 dev/test | 2–4 | supporting paragraphs | decomposition |
| hotpotqa | M6 dev/test | 2 | supporting facts | — |
| ohr_bench | 4,330 | 1 | `evidence_context` + page + text/table type | — |
| vidoseek | 1,142 | 1 | `reference_page` | — |
| uda | fin + paper_tab | 1 | `evidence` + `program` | — |

Single-hop sets (ohr_bench, vidoseek, uda) are the CONTROL group: the hypergraph/CoE legs
should show little or no gain there, and a large apparent gain would signal a leak, not a win.

## Gaps to close inside this stage

1. **No dense ANN index.** Vectors exist, nothing is indexed (the CLOSEOUT-AUDIT F-VEC
   residual). Brute-force is acceptable at 297,548 rows for evaluation; record the choice.
2. **No feature-matrix harness.** `benchmark/official/official_eval.py` is tier/KG-shaped for
   M6. S5 needs a harness that runs (dataset × leg-combination) and emits one comparable table.
3. **Metric parity.** Use the metrics each dataset's published baseline uses (recall@k /
   nDCG@k / evidence-chain F1 for the multi-hop sets), so SOTA comparison is honest.

## Gate artifacts

Per-dataset × per-leg table (recall@1/5/10, chain metrics where gold chains exist, latency,
tokens packed), the ablation deltas, and a SOTA-comparison section that only compares where
the metric and query set genuinely match. Dennis inspects → approve opens S6 (final report).
