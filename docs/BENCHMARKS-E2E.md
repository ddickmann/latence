# End-to-end results — GPU pipeline and retrieval

> **STATUS: UNFILLED — these are the numbers that gate public release.**
>
> Every value below is a placeholder. They are filled from a real GPU run, and this repository is
> **not published publicly until they are**. See
> [`RELEASE-GATES.md`](https://github.com/ddickmann/latence-framework/blob/main/RELEASE-GATES.md).
>
> The placeholders are **machine-checked**. `test_public_release_gate.py` inventories every
> `PENDING:` token on this page; under `LATENCE_RELEASE_CHECK=1` it **fails the build** while any
> remain. Publishing with an unfilled table by forgetting is not possible.

This page is distinct from [`BENCHMARKS.md`](BENCHMARKS.md), which carries **already-measured**
per-Stage suite results (Parse, Redaction, Relation extraction). Those are real. Nothing here is,
yet. The separation is deliberate — mixing measured and pending numbers on one page is how a reader
ends up quoting a placeholder.

Every table states its **method**. A benchmark number without its method is not a result, it is a
claim, and a reader who knows the field will discount it entirely.

---

## 1. Pipeline throughput (staged GPU run)

**Hardware:** `PENDING:gpu-hardware` · **Corpus:** `PENDING:corpus-identity` ·
**Commit:** `PENDING:repro-commit`

**Method:** staged — OCR server up → parse → server down → learned stages. Wall-clock and peak VRAM
read from the run's own `quality-report.json`, not from a stopwatch.

| Stage | Provider | Units | Wall-clock | Peak VRAM | Throughput |
|---|---|---:|---:|---:|---:|
| parse | `parser.lighton_vllm` | `PENDING:parse-count` | `PENDING:parse-time` | `PENDING:parse-vram` | `PENDING:parse-pps` |
| chunk | `chunk.markdown` | `PENDING:chunk-count` | `PENDING:chunk-time` | — | — |
| induce | `label_inducer.lexicon` | `PENDING:induce-count` | `PENDING:induce-time` | — | — |
| extract | `fused_entity_relation.gliner2` | `PENDING:extract-count` | `PENDING:extract-time` | `PENDING:extract-vram` | `PENDING:extract-cps` |
| redact | `redaction.gliner2` | `PENDING:redact-count` | `PENDING:redact-time` | `PENDING:redact-vram` | — |
| disambiguate | `disambiguation.glinker` | `PENDING:disambig-count` | `PENDING:disambig-time` | `PENDING:disambig-vram` | — |
| graph | `graph.canonical` | `PENDING:graph-count` | `PENDING:graph-time` | — | — |
| complete | `graph_completion.ultra` | `PENDING:linkpred-count` | `PENDING:linkpred-time` | `PENDING:linkpred-vram` | — |

**Total:** `PENDING:total-wallclock` for `PENDING:total-docs` documents.

### Device-honesty check

Two throughput remedies in this repo are **Blackwell-only** and are gated on compute capability. On
Ampere both must be inactive — if they are not, every number above was measured under a handicap and
is not comparable to anything.

| Check | Required on sm_86 | Observed |
|---|---|---|
| TorchScript JIT | left **enabled** | `PENDING:jit-observed` |
| `VLLM_USE_FLASHINFER_SAMPLER` | **not forced** | `PENDING:flashinfer-observed` |
| CUDA graphs | **captured** | `PENDING:cudagraph-observed` |
| torch.compile | **active** | `PENDING:compile-observed` |

---

## 2. Retrieval

**Method:** `PENDING:retrieval-method` · **Corpus:** `PENDING:retrieval-corpus` ·
**k =** `PENDING:retrieval-k`

### 2a. Per-signal, then fused

A fusion result without per-signal baselines is uninterpretable — the reader cannot tell whether
fusion contributed anything at all.

| Signal | Recall@k | nDCG@k | MRR |
|---|---:|---:|---:|
| Dense only | `PENDING:dense-recall` | `PENDING:dense-ndcg` | `PENDING:dense-mrr` |
| SPLADE (sparse) only | `PENDING:splade-recall` | `PENDING:splade-ndcg` | `PENDING:splade-mrr` |
| BM25 only | `PENDING:bm25-recall` | `PENDING:bm25-ndcg` | `PENDING:bm25-mrr` |
| Graph traversal only | `PENDING:graph-recall` | `PENDING:graph-ndcg` | `PENDING:graph-mrr` |
| **Fused — RRF (k=60)** | `PENDING:rrf-recall` | `PENDING:rrf-ndcg` | `PENDING:rrf-mrr` |
| **Fused — weighted, normalised** | `PENDING:wsum-recall` | `PENDING:wsum-ndcg` | `PENDING:wsum-mrr` |

Both fusers are reported because RRF's `k=60` is not a principled constant — in the original paper's
own words it was *"fixed during a pilot investigation and not altered during subsequent
validation"*, measured once over 30 homogeneous TREC runs. Bruch et al. (TOIS 2023) show a
normalised convex combination beats it. We report the comparison, not a preferred outcome.

### 2b. Stratified by hop count and shape — never pooled

A 7-wide fan-out and a 3-deep chain are different tasks with different failure modes. Averaging them
into one "multi-hop accuracy" is exactly the move the shortcut literature exists to puncture.

| Hops | Shape | n | Dense | SPLADE | BM25 | Graph | Fused |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | chain | `PENDING:h2-n` | `PENDING:h2-dense` | `PENDING:h2-splade` | `PENDING:h2-bm25` | `PENDING:h2-graph` | `PENDING:h2-fused` |
| 3 | chain | `PENDING:h3-n` | `PENDING:h3-dense` | `PENDING:h3-splade` | `PENDING:h3-bm25` | `PENDING:h3-graph` | `PENDING:h3-fused` |
| 4 | chain | `PENDING:h4-n` | `PENDING:h4-dense` | `PENDING:h4-splade` | `PENDING:h4-bm25` | `PENDING:h4-graph` | `PENDING:h4-fused` |
| 5–7 | fan-out | `PENDING:h57-n` | `PENDING:h57-dense` | `PENDING:h57-splade` | `PENDING:h57-bm25` | `PENDING:h57-graph` | `PENDING:h57-fused` |

> Depth beyond 4 is reported as **breadth**, not as chains. No published, human-validated benchmark
> exceeds 4 hops of serial depth — HotpotQA and 2Wiki are 2-hop, MuSiQue is 2–4, HopWeaver is 2.
> MuSiQue's best model scores 57.9 / 47.9 / **28.1** at 2/3/4 hops, so a 7-deep chain measures
> near-zero and discriminates nothing.

### 2c. Question-set validity — the funnel

A generator that reports only its output count is hiding its selection rate. MuSiQue's per-edge
necessity filter passed **26.5%** of candidates; a comparable rate is expected and published here.

| Stage | Surviving |
|---|---:|
| Candidates enumerated | `PENDING:funnel-enumerated` |
| Passed structural filters | `PENDING:funnel-structural` |
| Passed G3 (per-edge necessity) | `PENDING:funnel-g3` |
| Passed G4 (DiRe) | `PENDING:funnel-g4` |
| Human-audited as genuinely multi-hop | `PENDING:funnel-human` |

| Gate | What it proves | Result |
|---|---|---|
| G1 closed-book | A no-retrieval LLM scores at chance | `PENDING:g1` |
| G2 single-hop baseline | BM25+reader and dense+reader below fused, below oracle | `PENDING:g2` |
| G3 per-edge necessity | Masking a predecessor's answer breaks the sub-question | `PENDING:g3` |
| G4 DiRe | Disconnected reasoning cannot recover the answer | `PENDING:g4` |
| G5 bridge leakage | No intermediate entity named in the question | `PENDING:g5` |
| G7 evidence accessibility | Recall@k **and** Support F1, separately | `PENDING:g7` |
| G8 human audit | % judged genuinely multi-hop | `PENDING:g8` |

### 2d. Knowledge-graph ablation

The independent literature is markedly more sceptical than the GraphRAG marketing: DIGIMON (VLDB)
found no graph method consistently beats VanillaRAG; GraphRAG-Bench found graph methods achieve
higher recall but *lower* relevance. The defensible number is that ~13.6% of MultiHop-RAG queries
are answerable *only* via graph. **If the graph does not earn its weight here, that is the result we
publish.**

| Configuration | Recall@k | Δ |
|---|---:|---:|
| Without graph leg | `PENDING:ablate-nograph` | — |
| With graph leg | `PENDING:ablate-graph` | `PENDING:ablate-delta` |
| With graph + link prediction | `PENDING:ablate-linkpred` | `PENDING:ablate-lp-delta` |

### 2e. External benchmarks

Our own generated question set cannot establish standing on its own. These can.

| Benchmark | Metric | Published reference | Ours |
|---|---|---:|---:|
| BEIR / HotpotQA | nDCG@10 | `PENDING:beir-ref` | `PENDING:beir-ours` |
| StrategyQA | Recall@10 | 0.457 (2022 SOTA) | `PENDING:strategyqa-ours` |

---

## 3. Link prediction

**The honest ceiling, stated before the number:** on automatically-extracted open-schema KGs, SOTA
Hits@1 is **0.26–0.31**. Unfiltered top-1 densification means roughly 7 in 10 added edges are wrong —
and a false edge in a traversal graph does not merely fail to help, it **manufactures a spurious
path** that pulls unrelated documents into the neighbourhood. Edges are therefore admitted only
above a measured precision threshold, calibrated on a held-out sample from **our own** graph, never
on FB15k-237.

| Metric | Value |
|---|---:|
| Relation vocabulary before canonicalization | `PENDING:rel-before` |
| Relation vocabulary after | `PENDING:rel-after` |
| Held-out sample size | `PENDING:lp-sample` |
| Precision at admission threshold | `PENDING:lp-precision` |
| Edges admitted (% of predicted) | `PENDING:lp-admitted` |
| **End-to-end retrieval Δ** | `PENDING:lp-retrieval-delta` |

Link prediction is gated on **end-to-end retrieval**, never on MRR. A densification that improves
MRR and degrades retrieval is a failure.

---

## 4. Reproduction

Reproduced by
[`issues/e2e-extended/RUNBOOK.md`](https://github.com/ddickmann/latence-framework/blob/main/issues/e2e-extended/RUNBOOK.md). Corpus
acquisition: `bash benchmark/corpus/fetch_corpus.sh`.

**Corpus:** `PENDING:repro-corpus` · **Date:** `PENDING:repro-date`
