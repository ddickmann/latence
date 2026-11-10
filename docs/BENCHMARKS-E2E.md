# End-to-end results — GPU pipeline

> **STATUS: UNFILLED — these are the numbers that gate a release.**
>
> Every value below is a placeholder. They are filled from a real GPU run. The source of this
> repository is public as a pre-1.0 preview; a `v*` release is **not cut until they are**
> ([ADR-0066](adr/0066-publication-and-release-are-separate-gates.md)). See
> [`RELEASE-GATES.md`](https://github.com/ddickmann/latence/blob/main/RELEASE-GATES.md).
>
> The placeholders are **machine-checked**. `test_public_release_gate.py` inventories every
> `PENDING:` token on this page; under `LATENCE_RELEASE_CHECK=1` it **fails the build** while any
> remain. Cutting a release with an unfilled table by forgetting is not possible.

!!! note "What this protocol no longer covers"
    This protocol used to include a retrieval section: per-signal and fused results,
    hop-stratified results, the question-set funnel, a knowledge-graph ablation and two external
    retrieval benchmarks, plus a retrieval-delta row under link prediction. Every one of those
    values was still a placeholder. They were taken off the page because
    [ADR-0068](adr/0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)
    (2026-11-08) removed the query-time retrieval tooling they would have measured, so they can
    no longer be run from this repository. No measured number was deleted.

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

**Method:** staged — OCR server up → parse → server down → learned stages. Wall-clock is read
from the run's own `quality-report.json` (`stages[].duration_seconds`), not from a stopwatch. The
Quality Report records no VRAM figure, so the peak-VRAM column needs its own reading; state how
it was taken when the column is filled.

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

## 2. Link prediction

**The honest ceiling, stated before the number:** on automatically-extracted open-schema KGs, SOTA
Hits@1 is **0.26–0.31**. Unfiltered top-1 densification means roughly 7 in 10 added edges are wrong —
and a false edge does not merely fail to help, it **manufactures a spurious path** between
entities the corpus never connected. `graph_completion.ultra` therefore emits an inferred edge
only above a precision-targeted threshold (`target_precision`, default 0.9), calibrated on a
held-out split of **our own** graph's edges, never on FB15k-237. Inferred edges are marked as
such and counted separately in the Quality Report.

| Metric | Value |
|---|---:|
| Relation vocabulary before canonicalization | `PENDING:rel-before` |
| Relation vocabulary after | `PENDING:rel-after` |
| Held-out sample size | `PENDING:lp-sample` |
| Precision at admission threshold | `PENDING:lp-precision` |
| Edges admitted (% of predicted) | `PENDING:lp-admitted` |

This table used to end with an end-to-end retrieval delta, the measure link prediction was gated
on in place of MRR. That row went with the retrieval tooling, and this page defines no replacement
for it yet.

---

## 3. Reproduction

The stack is the stage table in §1, run end to end on the stated hardware. `latence setup
--profile enterprise-sota` generates a stack with the same Providers for chunk, extract, redact,
disambiguate, graph and complete, and with `parser.lighton_vllm` for parse when the corpus holds
scans or PDFs. It differs in one Provider: it wires `label_inducer.llm` for `induce` where the
table names `label_inducer.lexicon`. Its other Stages (source, the two screening Stages,
profiling, type consolidation and the two exports) are not timed by the table. Corpus acquisition
is `bash benchmark/corpus/fetch_corpus.sh`.

**Corpus:** `PENDING:repro-corpus` · **Date:** `PENDING:repro-date`
