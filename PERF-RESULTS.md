# PERF-RESULTS — measured throughput envelopes

Every number in this document was **measured on a named machine on a stated date with a stated
corpus, and can be reproduced with the command printed beside it.** Nothing here is extrapolated,
interpolated, or modelled: where a measurement does not exist, it is named in
[§ Not yet measured](#not-yet-measured) instead of being estimated. The large-scale (100k / 1M
document) envelopes are **parked**, not previewed — see that section.

How to read a row: `docs/hour` is that Stage's own wall clock (the `duration_seconds` the Quality
Report already records) scaled to an hour, for the corpus shape named in the header. A Stage's
throughput is a function of the corpus, not a constant — 13 KiB business documents are not
300-page scanned contracts, and the tables say which was run.

**Contents**

1. [CPU envelope — per-Stage, Apple M1 Pro, default stack](#cpu-envelope--per-stage-default-stack-apple-m1-pro)
2. [GPU envelope — Parse (OCR), NVIDIA RTX 2000 Ada](#gpu-envelope--parse-ocr-nvidia-rtx-2000-ada)
3. [The OCR acceleration ladder that produced the GPU numbers](#the-accel-ladder-content-safe-byte-identical-across-rungs)
4. [Encoder Providers — measured per-call latency](#encoder-providers--the-win-is-bf16-and-library-dependent)
5. [Not yet measured](#not-yet-measured)

---

## CPU envelope — per-Stage, default stack, Apple M1 Pro

**Hardware:** Apple M1 Pro (10 cores: 8 performance + 2 efficiency), 32 GiB unified memory,
macOS 26.3.1 (Darwin 25.3.0), Python 3.11.15. Single process, no GPU, no network.
**Stack profile:** `stacks/default.yaml` — the blessed CPU-first offline default (Source → Intake
Screening → Parse `parser.pdfplumber` → Chunk → Content Screening → Entity `entity.gazetteer` →
Relation `relation.pattern` → Redaction `redaction.hybrid_rule` → Profiling → Disambiguation →
Graph Assembly → KG + corpus Export). **No learned/GPU Provider is in this stack.**
**Corpus shape:** synthetic-but-real-shaped business documents, ~13.0 KiB each, 25 % of them
8-page text PDFs, the rest markdown/plain-text memos, notes and support tickets; entity-rich and
cross-referencing so Extraction/Disambiguation/Graph do real work. 15.5 chunks per document at
`max_tokens: 256`.
**Date measured:** 2026-07-18.
**Reproduce:**

```bash
uv run python scripts/bench_throughput.py --docs 200 --doc-kib 16 --repeat 3
uv run python scripts/bench_throughput.py --docs 1000 --doc-kib 16 --repeat 2
```

### 200 documents (2.54 MiB), median of 3 runs

| Stage | Capability | Provider | records in→out | median s | docs/hour | MiB/hour |
|---|---|---|---|---|---|---|
| source | source | source.local_folder | 0→200 | 0.056 | 12,937,773 | 164,047 |
| intake_screen | intake_screening | screening.intake_signature | 200→200 | 0.002 | 306,122,449 | 3,881,538 |
| parse | parse | parser.pdfplumber | 200→200 | 1.647 | 437,203 | 5,544 |
| chunk | chunk | chunk.markdown | 200→3100 | 2.623 | 274,504 | 3,481 |
| content_screen | content_screening | screening.content_keyword | 3100→3100 | 0.054 | 13,433,337 | 170,331 |
| entities | entity_extraction | entity.gazetteer | 3100→59450 | 2.264 | 318,078 | 4,033 |
| redact | redaction | redaction.hybrid_rule | 3100→3100 | 2.157 | 333,739 | 4,232 |
| relations | relation_extraction | relation.pattern | 62550→13950 | 1.221 | 589,560 | 7,475 |
| profiling | profiling | profiling.statistical | 59650→201 | 0.825 | 873,164 | 11,071 |
| export_corpus | export | export.jsonl_parquet | 3100→3100 | 0.278 | 2,592,801 | 32,876 |
| disambiguate | disambiguation | disambiguation.cascade | 73400→11406 | 0.564 | 1,275,958 | 16,179 |
| graph | graph_assembly | graph.canonical | 11406→11406 | 0.590 | 1,220,945 | 15,481 |
| export_kg | export | export.knowledge_graph | 11406→11406 | 0.396 | 1,819,675 | 23,073 |
| **end-to-end** | — | — | — | **14.275** | **50,438** | **640** |

### 1000 documents (12.67 MiB), median of 2 runs

| Stage | Capability | Provider | records in→out | median s | docs/hour | MiB/hour |
|---|---|---|---|---|---|---|
| source | source | source.local_folder | 0→1000 | 0.432 | 8,324,104 | 105,492 |
| intake_screen | intake_screening | screening.intake_signature | 1000→1000 | 0.011 | 320,184,996 | 4,057,741 |
| parse | parse | parser.pdfplumber | 1000→1000 | 6.341 | 567,715 | 7,195 |
| chunk | chunk | chunk.markdown | 1000→15500 | 10.032 | 358,862 | 4,548 |
| content_screen | content_screening | screening.content_keyword | 15500→15500 | 0.368 | 9,787,782 | 124,042 |
| entities | entity_extraction | entity.gazetteer | 15500→296650 | 7.313 | 492,291 | 6,239 |
| redact | redaction | redaction.hybrid_rule | 15500→15500 | 12.294 | 292,833 | 3,711 |
| relations | relation_extraction | relation.pattern | 312150→69750 | 7.938 | 453,534 | 5,748 |
| profiling | profiling | profiling.statistical | 297650→1001 | 4.370 | 823,774 | 10,440 |
| export_corpus | export | export.jsonl_parquet | 15500→15500 | 1.208 | 2,980,129 | 37,768 |
| disambiguate | disambiguation | disambiguation.cascade | 366400→57006 | 4.182 | 860,733 | 10,908 |
| graph | graph_assembly | graph.canonical | 57006→57006 | 3.358 | 1,071,987 | 13,585 |
| export_kg | export | export.knowledge_graph | 57006→57006 | 3.971 | 906,649 | 11,490 |
| **end-to-end** | — | — | — | **61.818** | **58,235** | **738** |

**What these two rows do and do not say.** They are two measured points, 200 and 1000 documents,
on one laptop-class CPU. End-to-end throughput did not degrade from 200 → 1000 documents (50.4k →
58.2k docs/hour — the per-run fixed cost amortises), which is evidence *at this scale only*. It is
**not** a claim that the curve holds at 100k or 1M documents; that measurement is parked (below).
The dominant Stages on this corpus are Redaction, Chunk and Relation Extraction — a sizing exercise
for a redaction-heavy or PDF-heavy corpus should re-run the harness on *its own* documents, which
is exactly what `scripts/bench_throughput.py --stack <your stack>` is for.

---

## GPU envelope — Parse (OCR), NVIDIA RTX 2000 Ada

**Hardware:** NVIDIA RTX 2000 Ada, driver CUDA 12.4, single GPU, single process.
**Stack profile:** generative OCR Parse via `parser.lighton` (LightOnOCR-2), content-safe
`max_new_tokens=4096`.
**Corpus shape:** 6 real LBBW German banking PDF pages — dense, multi-column scans.
**Date measured:** the W8 acceleration run recorded in ADR-0036 (§ *No roster change; CI green;
pod-measured*).
**Reproduce:** `scripts/validate-lighton-on-gpu.md` (the pod runbook) on a CUDA host.

The rows below are the **same measured seconds-per-page as the ladder in the next section**,
expressed per hour (a unit conversion of a measured number — no modelling, no batching assumption
beyond the batch size named in the rung):

| Parse configuration | s/page (measured) | pages/hour, 1 GPU |
|---|---|---|
| fp32 (old default) | 44.5 | 81 |
| bf16 | 29.9 | 120 |
| bf16 + 1540px longest-side cap | 29.6 | 122 |
| bf16 + cap + page-batch 4 | 25.1 | 143 |
| bf16 + cap + batch 4 + flash_attention_2 | 23.8 | **151** |
| bf16 + cap + batch 4 + FA2 + torch.compile | 24.0 | 150 |

**Sizing note, stated exactly as far as the measurement goes:** ~151 pages/hour per RTX 2000 Ada on
*this* corpus at the content-safe token budget. Pages per document, a faster GPU, tensor/data
parallelism across GPUs, and the served-vLLM path are all **unmeasured** — do not multiply this
number by anything without measuring it.

## Environment (OCR ladder)

- **Model:** LightOnOCR-2 (`parser.lighton`).
- **GPU:** NVIDIA RTX 2000 Ada.
- **Driver:** CUDA 12.4.
- **Corpus:** 6 real LBBW German banking PDF pages (dense, multi-column).
- **Content-safe path:** `max_new_tokens=4096` — the natural-EOS budget where every page's OCR
  output is stable/complete. Across every rung below the output is **byte-identical**
  (hash `c5a570c`, 19905 chars total), so the ladder measures pure throughput at zero content loss.

## The accel ladder (content-safe, byte-identical across rungs)

| Rung | Config | s/page | vs fp32 | output |
|---|---|---|---|---|
| R0 | fp32 (old default) | 44.5 | 1.00× | baseline |
| R1 | +bf16 | 29.9 | 1.49× | ≈identical |
| R2 | +1540px longest-side cap | 29.6 | 1.50× | identical |
| R4safe | +page-batch 4 @ 4096 tok | 25.1 | 1.77× | identical (no loss) |
| R5 | +flash_attention_2 | 23.8 | 1.87× | identical |
| R6 | +torch.compile | 24.0 | 1.85× | identical (no gain) |

## Findings

- **bf16 is the biggest single win (1.49×) AND mandatory.** fp16 produces garbage on LightOn (per
  LightOn's own guidance), so bf16 — not fp16 — is the correct load dtype. This is the default the
  W8 perf seam resolves on GPU (fp32 on CPU).
- **The 1540px longest-side cap measured ~nil on THIS corpus** (1.49× → 1.50×) because the workload
  is **decode-bound at a 4096-token budget** — the vision-encoder cost the cap reduces is dwarfed by
  the autoregressive decode. **Kept anyway:** it has no downside on decode-bound pages and helps
  vision-heavy / short-output pages (where the vision encode is a larger share of the cost).
- **Page-batching (batch 4) is a real 1.77×** and record-for-record identical to per-page (the
  offline equivalence test + this byte-identical pod run both confirm it).
- **flash_attention_2 adds only +6%** (1.77× → 1.87×) on this GPU/model — a modest incremental win,
  taken when `flash_attn` is importable, gracefully degrading to sdpa otherwise.
- **torch.compile gave no gain** (1.87× → 1.85×, within noise / slightly worse from warmup) — which
  **validates the compile-off-by-default decision**; the autotuner turns it on only where it wins.
- **`max_new_tokens=1024` truncated ~25% of dense-page text** (19905 → 14951 chars — the pages
  genuinely emit >1024 tokens), silently dropping document content. Worse than slow for enterprise,
  so the **default was corrected to the content-safe 4096** (fully configurable; tune down only for
  known-light corpora, never below a page's real token length).

## Encoder providers — the win is bf16, and library-dependent

The learned ENCODER providers were validated separately on real weights. **The honest encoder win
is bf16 — NOT flash-attention.** How the dtype is applied depends on what the underlying library
accepts, and that mattered: the first W8 cut crashed on real weights by assuming a uniform kwarg
path.

- **Plain `gliner` library (0.2.27) — `entity.gliner`, `relation.gliner_relex`, `redaction.gliner_pii`.**
  `GLiNER.from_pretrained` **rejects** `torch_dtype` AND `attn_implementation`
  (`TypeError: UniEncoderSpanModel.__init__() got an unexpected keyword argument 'torch_dtype'`), and
  the encoder does **NOT** support `attn_implementation` / FlashDeBERTa (its only `flashdeberta`
  references live in `gliner/serve/`, not the encoder). The correct path is **load plain, then
  `model.to(bf16)` on cuda** — measured **19.1 → 14.0 ms/call = 1.36× faster, byte-identical
  entities** (Alice/person, Acme/organization, Berlin/location). Routed through `load_then_cast`.
  **No FA2 / FlashDeBERTa is claimed for these — the library does not support it.**
- **`gliner2` library — `redaction.gliner2`.** `GLiNER2.from_pretrained(torch_dtype=…,
  attn_implementation=…)` **loads fine** (kept). But mdeberta uses **disentangled attention**, so
  FA2 does not accelerate it — the honest win is **bf16**; `attn_implementation` is passed only where
  the lib accepts it, not claimed as a speedup.
- **`embedding.sentence_transformers` (DenseOn = ModernBERT).**
  `model_kwargs={torch_dtype, attn_implementation}` loads, 768-dim output (kept). ModernBERT
  genuinely supports FA2, so this is the one encoder where `flash_attention_2` is a real option.

No fabricated encoder numbers beyond the measured **gliner 1.36×** (identical output).

**As a throughput envelope** (same RTX 2000 Ada, same date, unit conversion only): the measured
gliner call latency is **19.1 ms/call fp32 → 14.0 ms/call bf16**, i.e. **≈188,000 → ≈257,000
extraction calls/hour on one GPU**. A *call* is one Entity/Relation/Redaction inference, **not** a
document — documents/hour for the learned Providers is deliberately **not** derived here, because
calls-per-document depends on the chunk size and corpus of the deployment, and multiplying through
would be an extrapolation. Measure it on your corpus with the harness above against a GPU stack.

## transformers-vs-vLLM OCR comparison — PENDING (newer-driver host)

**Not measured. Not fabricated.** The vLLM head-to-head could not run on this pod: its driver
(**CUDA 12.4**) is too old for the torch build (**cu126+**) that any LightOnOCR-2-supporting vLLM
requires. This is a documented **infrastructure constraint, not a code result** — it needs a host
with **driver ≥ CUDA 12.6**.

| Rung | Config | pages/s | startup cost | output fidelity |
|---|---|---|---|---|
| vLLM | LightOnOCR-2 served (vLLM) | pending newer-driver host | pending | pending |

The enterprise deploy choice (optimized transformers vs served-vLLM) stays open until this row is
filled on a driver ≥ 12.6 host — no throughput/fidelity numbers are guessed here.

---

## Not yet measured

Everything below is **absent from this document on purpose**. None of it is estimated, and no row
above may be multiplied out to cover it. If you need one of these, the harness and the pod runbooks
are in the repo — measure it and add the row with its hardware, profile, corpus, date and command.

**Scale.**

- **100k / 1M document throughput and peak RSS, and the no-OOM-at-ceiling proof.** Parked for a
  dedicated perf rig and tracked in **issue #64**; the streaming-Runner issue (#63) shipped with
  these numbers explicitly deferred, as ADR-0033 § *Consequences* records. The two CPU points above
  (200 / 1000 docs) are **not** a preview of it and must not be extrapolated toward it.
- **Anything above 1000 documents on any hardware.** The largest corpus ever measured end-to-end is
  the 1000-document run above.
- **Peak memory / backpressure behaviour under sustained load.** The streaming Runner is
  memory-bounded by design (ADR-0033) and asserted at fixture scale; the RSS *number* at scale is
  the perf rig's job.

**Hardware.**

- **Any server-class CPU** (x86_64 Xeon/EPYC, Graviton, or a cloud instance type). The only CPU
  envelope here is one Apple M1 Pro laptop.
- **Any GPU other than the RTX 2000 Ada**, and any multi-GPU configuration. No A100/H100/L4/T4
  numbers exist.
- **Multi-process, multi-node or sharded execution.** The Runner measured here is single-node,
  single-process; the Airflow adapter's scale-out throughput is unmeasured.
- **Object-store (s3://, gs://, abfs://) Storage.** Every number above is local-filesystem Storage;
  the network-Storage penalty is unmeasured.

**Stages and Providers.**

- **Per-Stage docs/hour for the learned GPU Providers** (`entity.gliner`, `relation.gliner_relex`,
  `redaction.gliner_pii`/`gliner2`, `disambiguation.glinker`, the schema inducer, link prediction).
  Only per-call latency exists (above) — see **issue #64**.
- **Embedding throughput** (`embedding.sentence_transformers` / the endpoint embedders). No Embed
  Stage runs in `stacks/default.yaml`, so no embedding row was measured.
- **The endpoint Providers** (`parser.endpoint`, `ner.endpoint`, `embedder.endpoint`) — their
  throughput is a property of the remote service, and none was measured against a live endpoint.
- **The universal/document parsers on real office formats** (`parser.document`, `parser.render`,
  `parser.glm`) and **served-vLLM OCR** — see the PENDING section above.
- **Retrieval / query-time throughput** (`latence-retrieval`, SPLADE, MUVERA, the DuckDB graph
  hops). This document covers index-time pipeline Stages only (ADR-0049 clock split).
- **Delta / retract / purge runs.** Only full runs were measured.

**Corpora.**

- **Scanned, image-only, or non-Latin-script documents on the CPU stack.** The CPU corpus is
  digital-text markdown/text/PDF; OCR-grade scans were measured only on the GPU OCR path, on 6
  German banking pages.
- **Documents materially larger than ~13 KiB** on the CPU stack, and PDFs longer than 8 pages.
