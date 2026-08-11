# S5b — Retrieval architecture fix: apply the graph lane the way it is meant to be applied

Type: task
Status: claimed
Blocked by: s5 equal-weight baseline (complete 2026-08-11, 49/49 cells)

## Why (Dennis, 2026-08-11)

The S5 matrix measured every graph leg REGRESSING recall. Investigation showed the components
are fine and the harness is wrong: dumb equal-weight RRF, an under-budgeted packer, and graph
results applied as a *fused competing list* rather than as what they are. These are
**algorithmic weaknesses in the application, not component failures**. We fix them and measure
at SOTA level.

## The evidence base

**1. `colsearch` (github.com/latenceainew/colsearch) — a working KG retrieval lane we own.**
Its documented flow (`docs/guides/latence-graph-sidecar.md`) is the target architecture:

```
query → dense retrieval (+ BM25S in parallel) → merge/dedupe first-stage candidates
      → graph POLICY check
           off → solver / packing
           on  → resolve_entities(query, FIRST-STAGE DOCS) → expand_local → retrieve_community
                 → linked_evidence → graph_features
      → final solver / context packing
```

Its own words: *"It does not replace the base retrieval engine with graph traversal."* and
*"without turning graph traversal into the default first-stage router."* — which is precisely
what our ladder did.

**2. HyCE-RAG (arXiv:2607.22597)** — the paper hyperedges + CoE derive from. Verified in
`issues/wayfinder-hypergraph-coe/tickets/02`: its evidence budget is a **COUNT cap of 6
segments**, not a token budget, and it reports **no recall@token-budget curves**. Our 4,000-token
packer budget came from neither the paper nor a measurement.

**3. HippoRAG 2 (arXiv:2502.14802)** is the de-facto retrieval comparison line: **passage
recall@5**, MuSiQue 74.7, HotpotQA 96.3. Our musique chunk-level R@10 is 0.341 — but that is a
**metric mismatch, not a 2x gap**: they recall *passages*, we recall *chunks*, and 10 chunks
cover far fewer than 10 distinct documents.

## The four defects, and the fix for each

| # | Defect (measured) | Fix |
|---|---|---|
| 1 | Graph legs fused into RRF at equal weight; a weak alias hit competes with a strong dense hit. R@1 degrades monotonically (ohr_bench 0.228 → 0.122) | Graph becomes **post-retrieval candidate rescue**: base ranking is preserved, graph APPENDS candidates the base missed, and contributes **features** to the packer — never a co-equal fused list |
| 2 | Entity resolution seeds from **query n-grams only** | Seed from **query + first-stage documents** (colsearch step 3). The first-stage docs are the strongest entity signal available and we ignored them |
| 3 | Graph lane runs unconditionally on every query | **Policy gate** (`auto`): entity-heavy queries, relation cues, low dense/BM25 agreement, low first-stage confidence. Known-item/navigational queries stay graph-off |
| 4 | Packer at 4,000 tokens, measured packing only 1,895 mean on wiki2multihop; CoE and packer are **selection** stages scored with a **ranking** metric | Budget to **≥10,000 tokens** (Dennis), and evaluate selection stages on selection metrics: recall@budget curve, chain coverage within budget, context precision per token — never bare recall@10 |

## Metrics (fixes the comparability gap)

- **Document-level recall@k** alongside chunk-level, so HippoRAG-2's passage recall@5 is a
  like-for-like comparison instead of an apples-to-oranges one.
- **Recall as a curve over token budget** (our native protocol; the research ticket's own
  recommendation) plus recall@5 at a standard budget for the published-baseline line.
- Chain coverage and full-chain hit stay first-class — they are where the hypergraph's value
  already showed even under the broken fusion (multihop_rag chain 0.626 → 0.671 with hyperedges
  while R@10 fell).

## Amendments (Dennis, 2026-08-11, after the multihop_rag smoke)

- **Per-selector headline budgets** (e9d66c2): the knapsack saturates ~5.5k tokens (10k and 16k
  selections identical) so its headline stays 10k; CoE keeps buying recall through 16k (spend
  9,944@10k → 15,942@16k, docR@10 0.8380 → 0.8455) so its headline moves to 16k. Ruling: "going
  up to 16k for a retrieval is not a problem at all" — modern LLMs' smart zone is ~80–100k.
  24k/32k curve points added to locate CoE's true saturation.
- **Packer backfill** (6f3b379): the greedy QKP stops at the first non-positive marginal, which
  stranded ~45% of the 10k budget — unspent budget is unrecovered recall on a retrieval bench.
  Shipped `Packer` gains opt-in `fill: "backfill"` (spend the remainder in value order, greedy
  phase untouched; default stays pure-QKP "stop"). The rescue leg opts in. This is the "prove
  the packer does its job at SOTA level" fix.
- **Full-blown run**: all 7 datasets, ALL queries (supersedes the earlier 1,000-query cap on
  wiki2multihop/hotpotqa/uda). Vectorized BM25 (30f9a53) makes this affordable.
- **Production lanes**: the architecture must be usable in production and configurable via the
  campaign wizard — "not just for the benches and README flexing". Productization runs as its
  own work item (latence-retrieval composition + wizard block + docs/guides entry).

## Non-negotiables

- Zero LLM calls in the online path (the framework's differentiator; HyCE-RAG and HippoRAG-2
  both use LLMs where we do not — position on recall-per-token at zero marginal LLM cost).
- The equal-weight baseline stays as the honest "naive fusion" reference; the new architecture
  is reported beside it, never silently replacing it.
- All scoring through the shipped `latence-retrieval` components; the harness supplies engines
  and policy only (ADR-0048).
