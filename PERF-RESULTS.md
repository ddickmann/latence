# PERF-RESULTS — W8 OCR acceleration ladder (real, pod-measured)

Real measured throughput for the W8 performance seam on the generative OCR path. **All numbers
below are measured on a GPU pod — none are fabricated.** Where an environment constraint blocked a
measurement (the vLLM head-to-head), that is recorded as pending, not estimated.

## Environment

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
