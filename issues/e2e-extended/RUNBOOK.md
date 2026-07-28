# Extended end-to-end GPU run — RUNBOOK

**Audience: a fresh session with no memory of this work.** This document is self-contained. You do
not need the conversation that produced it. Read this file top to bottom before touching anything.

Companion documents, in the same directory:

- [`DECISIONS.md`](DECISIONS.md) — *why* every choice was made, with the citation or `file:line` that
  justified it. Read it when you want to challenge a decision; do not re-litigate from memory.
- [`TICKETS.md`](TICKETS.md) — *what* was built and what is still outstanding, with acceptance bars.

---

## 0. State at handoff

| | |
|---|---|
| Repo | `/Users/dennisdickmann/Developer/latence-framework` |
| Branch | `main`, tree clean |
| Unpushed | **39 commits ahead of `origin/main`** — pushing is the operator's call, not yours |
| `verify-local` | **GREEN**, `REAL_EXIT=0` |
| `mypy --strict` | 33 packages, src + tests |
| Test suite | **4,590 passed / 33 skipped** (~9m30s) |
| Contract schema | `SCHEMA_VERSION = 19` — **v18 checkpoints will NOT load** |

The 33 skips are environment-gated (live Qdrant, GPU providers, s3). **On the pod they become real
tests.** If they still skip there, the environment is wrong — that is a signal, not a pass.

### How to re-verify before you trust any of this

```bash
cd /Users/dennisdickmann/Developer/latence-framework
bash scripts/verify-local.sh > /tmp/verify.log 2>&1; echo "REAL_EXIT=$?"
```

**Capture the exit code by redirect, never by piping to `tail`.** A pipeline returns the *last*
command's status, so `verify-local.sh | tail` reports `0` on a red tree. That mistake was made once
in this workstream and produced a false green.

Test counts need `-o addopts=""` — `pyproject.toml` sets `addopts = "-q"`, so a bare
`uv run pytest -q` resolves to `-qq` and suppresses the summary line entirely:

```bash
PYTHONHASHSEED=0 uv run pytest -q -o addopts=""
```

`PYTHONHASHSEED=0` is mandatory — a conftest guard refuses the run without it.

---

## 1. What is built, and what is not

Be precise about this. An earlier summary called it "half built", which was wrong and unhelpful.

### Built, merged, green — everything that PROCESSES data

| Ticket | What landed |
|---|---|
| **T-DEV-1** | TorchScript-JIT kill is now gated on compute capability, **fail-safe** |
| **T-DEV-2** | Microbatch + thread defaults derived from detected VRAM / core count |
| **T-OCR-1** | vLLM serve surface made Ampere-correct; FlashInfer override now conditional |
| **T-OCR-2** | Page images rounded onto the 28-px patch-merge grid |
| **T-OCR-3** | OCR DPI + longest-dim reachable from the wizard |
| **T-WIZ-1** | Chunk token budget reachable (`--chunk-max-tokens`), model-side window asserted |
| **T-WIZ-3** | GLiNER2 microbatch / thread / dtype knobs reachable |
| **T-LABEL-1** | New package `latence-label-profiler` — non-LLM label derivation |
| **T-RET-1** | Qdrant **named sparse vectors** written (SPLADE becomes a real signal) |
| **T-RET-2** | Graph **retrieval leg** — `Capability.GRAPH`, a peer of dense/sparse/BM25 |
| **T-FUSE-1** | Weighted (min-max normalised) fusion first-class; RRF kept as fallback |
| **T-RET-4** | Three pod-path defects fixed (see §9, they would have killed the run) |

### NOT built — everything that MEASURES it

| Ticket | Why it is deliberately deferred |
|---|---|
| **T-KG-2** | Link-prediction precision gate — needs a **real extracted KG** to calibrate a threshold against. Calibrating on FB15k-237 is exactly the mistake Akrami et al. (SIGMOD 2020) documented. |
| **T-BENCH-1** | Hop-stratified question generation — needs a **real gold graph** |
| **T-BENCH-2** | The eight necessity gates (G1–G8) — needs real questions to gate |
| **T-BENCH-3** | The KG ablation — depends on T-BENCH-1/2 |
| **T-BENCH-4** | StrategyQA / BEIR ingestion — independent, but low value before the above |
| **T-RET-3** | Live-Qdrant **search** test — writable offline, only *runs* on the pod |
| **T-RET-5** | Metadata-filter payload indexes — measurable only against real payloads |
| **T-CORPUS-1** | Corpus acquisition — §4 of this document |

**The reason for deferring is specific, not vague.** Building a question generator against the small
RFC fixture and then meeting the real corpus means rewriting it. On a previous run, a benchmark loop
was trimmed to run against a frozen dataset; when fixes changed the *pipeline*, the loop re-found the
same finding forever — roughly 77 minutes lost to a design error. **Do not build the measurement half
against a stand-in.** Build it against Stage-2 output.

---

## 2. The hardware constraint, and why the run is STAGED

Target: **1× RTX 3090, 24 GB, Ampere sm_86.** Not Blackwell. **8 GB less VRAM than the 32 GB card
every default in this repo was originally tuned on.**

The binding number: GLiNER2 at micro-batch 4 peaked at **21.6 GiB** on a 32 GB card. On 24 GB that
leaves ~2.4 GiB — before the OCR server, before allocator fragmentation.

**Therefore: never co-host the vLLM OCR server and the learned stages.** Serve OCR → parse the whole
corpus → **tear the server down** → run extraction. Attempting one pass is the single most likely way
to lose an afternoon to an OOM that looks like a code bug.

### A second, independent reason the run is multi-pass

`latence-label-profile` (T-LABEL-1) walks a directory of **text** (`--pattern '*.txt'`). It profiles
what the parser produced, not the raw binaries. So the label pool cannot exist before parse. The
sequence is forced:

```
parse  →  profile the parsed text  →  extract with the derived label pool
```

This is not a limitation to work around — it is the correct order. A label set derived from raw PDFs
would be profiling file structure, not content.

---

## 3. Environment bring-up

Adapt paths if the pod differs; the shape is what matters.

```bash
# --- 0. sanity: confirm the card is what we think it is -----------------------------
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
#   expect: NVIDIA GeForce RTX 3090, 24576 MiB, 8.6
#   If compute_cap is 12.x you are on Blackwell and HALF THIS DOCUMENT IS WRONG - stop.

# --- 1. repo ------------------------------------------------------------------------
cd /workspace && git clone <remote> latence-framework && cd latence-framework
git log --oneline -1     # confirm you have the 39-commit head, not origin/main

# --- 2. python ----------------------------------------------------------------------
uv venv --python 3.12 && source .venv/bin/activate
uv sync

# --- 3. the learned stack (heavy, GPU) ----------------------------------------------
uv pip install -e packages/latence-core \
               -e packages/latence-parser-lighton-vllm \
               -e packages/latence-parser-render \
               -e packages/latence-extract-gliner2 \
               -e packages/latence-pii-gliner2 \
               -e packages/latence-disambig-glinker \
               -e packages/latence-linkpred-ultra \
               -e packages/latence-embedder-st \
               -e packages/latence-label-profiler
uv pip install gliner2 gliner glinker flashdeberta peft accelerate \
               sentence-transformers openai pypdfium2 pillow pyarrow fsspec networkx rdflib

# --- 4. retrieval extras ------------------------------------------------------------
uv pip install -e "packages/latence-splade[local]" \
               -e "packages/latence-sink-qdrant[client]" \
               -e "packages/latence-retrieval" \
               -e "packages/latence-retrieval[graph-duckdb]"
```

**Do NOT pass `--no-deps` to the editable installs.** `peft`, `glinker`, `flashdeberta` and `gliner`
are hidden transitive dependencies; omitting them produces a stack that installs cleanly and dies at
import. (`deploy/env/gpu-learned/install.sh` does use `--no-deps`, but only because
`requirements.lock` has already solved the graph — that is a different path.)

### Ampere-specific: things NOT to do

Two environment variables exist in this repo's history as **Blackwell-only** remedies. Both are now
conditional in code, but if you set them by hand you will silently degrade this run:

- **Do not export `VLLM_USE_FLASHINFER_SAMPLER=0`.** It exists because FlashInfer misparses compute
  capability `12.0` as the integer `12`, which compares below its `75` floor. `86` parses fine.
- **Do not export `PYTORCH_JIT=0`.** It exists for `libnvrtc-builtins.so.13.0` missing on CUDA-13.
  On Ampere the nvrtc path is healthy, and the JIT is worth real throughput on the three stages that
  dominate the run.
- **Do not export `HF_HUB_DISABLE_XET=1`.** Blackwell-era workaround, doc-only, unnecessary here.

**Verification that the gate is doing its job** — this is a GPU-ONLY acceptance item from T-DEV-1:

```bash
python -c "
from latence_core.providers.perf import neutralize_torch_jit
import logging; logging.basicConfig(level=logging.INFO)
print('neutralized:', neutralize_torch_jit(provider_name='smoke'))
"
# On sm_86 the JIT should be LEFT ENABLED. If the log says it was neutralized,
# the capability probe is failing and you are running with the Blackwell handicap.
```

### Secrets

```bash
export OPENROUTER_API_KEY='<key>'      # never commit, never put in YAML
```

The key is read at runtime by the induction provider and the benchmark harness. `recipe.py` selects
the env var name from the base URL (`openrouter` in the URL → `OPENROUTER_API_KEY`, else
`OPENAI_API_KEY`). **Store it pod-resident only** (e.g. `/workspace/.or_key`, read via command
substitution). It must never enter a config file, a commit, or a published artifact.

Note: with the non-LLM label path (§5) the pipeline itself makes **zero** LLM calls. The key is
needed only if you later use an LLM for benchmark question generation.

---

## 4. The corpus

Full reasoning, licences and rejected alternatives are in [`DECISIONS.md`](DECISIONS.md) §D4. This
section is the operational summary. **Three axes, chosen because no single source covers all three.**

### 4a. GovDocs1 — file-type diversity, CC0

**986,278 files / 333 GB compressed / ~500 GB uncompressed.** Harvested from `.gov` web servers.
Published type histogram:

| Type | Count | Type | Count |
|---|---|---|---|
| PDF | 231,106 | DOC (legacy OLE) | 76,605 |
| HTML | 202,440 | XLS (legacy OLE) | 62,876 |
| JPEG | 109,283 | PPT (legacy OLE) | 51,292 |
| TXT | 88,856 | CSV | 18,717 |
| PostScript | 20,630 | RTF | 1,067 |
| GIF/PNG/BMP | ~40,000 | DOCX / XLSX / PPTX | 164 / 39 / 220 |

**Licence: CC0.** *"CC0 is in effect because these materials were made by US Government employees as
part of their official duties."* This is the only corpus here that is **redistributable without
argument** — so anything you publish must come from this one.

```bash
# Smoke first: ten pre-built random 1,000-file subsets, 3.09 GB total
aws s3 sync --no-sign-request \
  s3://digitalcorpora/corpora/files/govdocs1/threads/ /workspace/corpus/govdocs1-threads/

# Full pull, 333 GB
aws s3 sync --no-sign-request \
  s3://digitalcorpora/corpora/files/govdocs1/zipfiles/ /workspace/corpus/govdocs1/
```

⚠️ **Two traps:**

1. **The corpus contains live malware, deliberately retained.** *"the malware will not be removed
   from the files because it is legitimately part of the corpus."* We only ever **parse** these
   files, never execute them — but treat the pod as untrusted: no host mounts, no credential
   material beyond what the run needs, and expect AV noise. This must be a deliberate posture, not a
   2 a.m. surprise.
2. **`by_type/` has no `pdf.zip` and no `html.zip`** — the per-type archives omit the two largest
   classes. Take `zipfiles/`.

⚠️ **Honest limit:** GovDocs1 is a 2009 harvest, so office formats are **legacy OLE** (DOC/XLS/PPT),
not modern OOXML. **Modern OOXML at volume is untested in this run** and the manifest must say so.
Do not let "many file types" imply coverage we do not have.

### 4b. SEC VPRR — scanned corporate pages we can publish

**315,037 PDFs / 113 GB / 100% scans** of paper-filed SEC submissions. Index:
`https://www.sec.gov/Archives/edgar/vprr/index.html`; each directory exposes `index.json`.

**Licence:** *"Information presented on sec.gov is considered public information and may be copied or
further distributed by users of the web site without the SEC's permission."* No EULA, no API key.
Carve-outs are narrow: do not use the SEC seal, logos, or the "EDGAR" trademark.

**This is the scanned axis we can show in a public artifact** — which is why it is preferred over
IIT-CDIP / UCSF IDL, whose posture is fair use (access granted, rights not).

⚠️ **Two hard operational requirements:**

```bash
export SEC_UA="YourOrg contact@yourdomain.com"   # a real, reachable contact
curl -H "User-Agent: $SEC_UA" -H "Accept-Encoding: gzip, deflate" ...
```

- **A declared User-Agent is mandatory.** A client without one gets **HTTP 403** at the edge. This is
  enforced, not advisory.
- **10 requests/second**, counted per entity across all machines, with a 10-minute IP cooldown on
  violation. Rate-limit your fetcher.

### 4c. OHR-Bench — the scored control

**1.84 GB, CC-BY-4.0.** Raw PDFs + QA ground truth across seven domains, five of which are
corporate/institutional (Finance, Law, Manual, Administration, Newspaper).

```bash
hf download opendatalab/OHR-Bench --repo-type dataset --include "pdfs.zip" "*.parquet" \
  --local-dir /workspace/corpus/ohr-bench
```

This is the only axis with ground truth, so it is how you claim parse *quality* rather than parse
*throughput*.

### 4d. Held in reserve

- **IIT-CDIP @ NIST** — 1.41 TB, ~7 M docs, 1990s OCR, handwriting, stains. DOI
  `10.18434/mds2-2531`, free, unauthenticated. Use only if 315k VPRR scans prove too few. Its rights
  posture is fair use → **nothing derived from it may enter a public artifact.**
- **DocLayNet-v1.2** — CDLA-Permissive, embeds source PDFs + hand-annotated layout GT. Use if you
  need layout ground truth beyond OHR-Bench.

### 4e. Explicitly rejected — do not re-propose

| Source | Reason |
|---|---|
| OmniDocBench | *"for research purposes only and not for commercial use"* |
| FUNSD | Explicitly non-commercial |
| Kleister NDA/Charity | **No licence at all** — GitHub API returns `license: null` |
| CourtListener RECAP | Highest PII surveyed **and** an FCRA restriction in its ToS that would propagate downstream forever |
| World Bank D&R | Non-commercial, **no derivatives** — not the CC BY people assume |
| UN Digital Library | All rights reserved; harvesting explicitly forbidden |
| `anon-rabbit/annual_reports` | A robots.txt-disallowed scrape, relicensed Apache-2.0 by an uploader with no standing |
| Enron / Podesta | No lawful disclosure event in the chain of custody (Podesta); no format diversity (Enron) |

---

## 5. Stage 1 — parse

**Goal: corpus → parsed text, with the OCR server up and then torn down.**

### 5a. Start the vLLM OCR server

```bash
bash deploy/env/served-ocr-server/serve.sh
```

This now emits an Ampere-correct command line (T-OCR-1). Confirm from the startup log:

- `torch.compile` active — look for `Start compiling function` / `torch_compile_cache`
- CUDA graphs **captured** — the absence of `--enforce-eager` is not enough, verify capture
- FlashAttention-2 selected (it is the sm_86 default; verify rather than assert)
- **`VLLM_USE_FLASHINFER_SAMPLER` is NOT forced to 0**

`--gpu-memory-utilization` starts at **0.80** on 24 GB. **On any OOM, fall back to 0.72.** That is a
measurement, not a constant to copy from the Blackwell run.

### 5b. Generate the stack from the wizard — this is the entry gate

**The wizard is the entry gate. Hand-edited YAML is a failure of the exercise.** Every knob below is
reachable end to end (`SetupAnswers` field → CLI flag → emitted YAML → provider reads it).

```bash
latence setup --profile enterprise-sota \
  --source /workspace/corpus/govdocs1 \
  --storage file:///workspace/latence-out \
  --chunk-max-tokens 768 \
  --chunk-overlap-tokens 96 \
  --ocr-dpi 200 \
  --ocr-max-longest-dim 1540 \
  --microbatch-max-texts 4 \
  --microbatch-token-budget 3072 \
  --intra-op-threads 8
```

**Why these values:**

| Flag | Value | Source |
|---|---|---|
| `--chunk-max-tokens` | **768** | Aligned to the GLiNER2 model window |
| `--ocr-dpi` | **200** | LightOn's published render density |
| `--ocr-max-longest-dim` | **1540** | LightOn's published cap; `processor_config.json` has `size: {longest_edge: 1540}` and no other size key |
| `--microbatch-*` | 4 / 3072 | 32 GB-derived; T-DEV-2 now auto-derives, so **prefer omitting these and letting detection run**, then compare |

⚠️ **There is no minimum-shorter-side rule.** A widely-repeated "longest 1536 → shorter ≥ 768"
constraint is **Qwen2-VL `smart_resize` semantics and does not apply** — LightOnOCR uses a Pixtral
image path. The real constraint, implemented in T-OCR-2, is that **both sides must be multiples of
28** (patch 14 × merge 2), because the patch merger uses `unfold(kernel=2, stride=2)` which silently
drops an incomplete trailing row/column.

⚠️ **Chunk-token trap, already covered by a test but worth knowing:** the chunker tokenizes ~1.2×
*fewer* tokens than mdeberta. A 768 chunk budget previously produced **median 666 / max 921** mdeberta
tokens — silent truncation. The T-WIZ-1 test asserts the **model-side** count, not the YAML value.

### 5c. Run parse

```bash
latence process        # or the stage-scoped invocation if you want to stop after parse
```

⚠️ **Checkpoint-resume trap.** A "re-run" once completed in **1.5 seconds** because every stage
resumed from checkpoint. If you change config and re-run, **clear the run directory** or you will
measure the previous run and believe it.

### 5d. Tear the OCR server down

```bash
pkill -f "vllm serve"     # then confirm with nvidia-smi that VRAM is actually released
nvidia-smi
```

**Do not skip this.** §2 explains why.

### Stage-1 verification checklist

- [ ] Parsed document count vs input file count — what fell out, and why?
- [ ] Per-file-type success rate. GovDocs1 lies about extensions on **33,917 files** by design, so a
      content-detection failure is a finding, not noise.
- [ ] Pages/sec measured. LightOn publishes 5.71 pages/s on an H100; a 3090 estimate of ~1–2 pages/s
      is **unverified** — measure it, do not quote it.
- [ ] Peak VRAM during parse.
- [ ] `quality-report.json` parse section: page-map drift, parse errors, bytes_after_parse.

---

## 6. Stage 2 — derive the label pool (no LLM)

**This is the step that replaces LLM induction, and it must run between parse and extract.**

```bash
latence-label-profile /workspace/latence-out/_latence/runs/<run_id>/parse \
  --pattern '*.txt' \
  --max-types 40 \
  --sample-size 5000 \
  --seed 0 \
  --min-mention-mass 3 \
  --min-document-frequency 2
```

### What it emits, and what you must read

The artifact carries **more than the labels**, deliberately (T-LABEL-1):

- the **marginal-gain curve** — the diminishing-returns shape
- the **per-corpus optimality certificate** — `f(OPT_N) ≤ f(A_i) + Σ of the N largest remaining
  gains`. Usually far tighter than the worst-case (1 − 1/e) ≈ 63.2% guarantee, and it lets you say
  *"these 40 labels are within X% of the best possible 40 **on this corpus**"*
- **empirical coverage `C(N)`**
- the fitted **Zipf α**
- the **Good–Turing** estimate of residual uncovered mention mass

⚠️ **Read the α before you quote the 40.** There is **no published standard** for a "90% coverage" or
"marginal gain < x%" cutoff. If α ≈ 1, coverage grows only logarithmically and **no sharp knee
exists** — in that case N=40 is a *budget decision* and the write-up must say so. Do not let the
artifact imply a discovered natural boundary that the data does not contain.

### Why the pool must be narrowed per chunk

**Do not feed all 40 labels to every chunk.** GLiNER2 is a **uni-encoder** — labels and text share one
transformer sequence, cost O((n+m)²). Published guidance:

- The author recommends **20** labels for vanilla GLiNER
- The docs say performance **degrades past 30**
- Training cap is `max_types = 25`
- Measured: **+60%** CPU latency from 5 → 50 labels; **−98.7%** throughput from 1 → 1024
- Scores are **not permutation-invariant** in the label list (a confirmed positional-encoding effect)

So the profiler's output is a **candidate pool**, and `label_inducer.lexicon` narrows it **per chunk**
(deterministic, CPU-only, zero-network). Its own docstring says there is *deliberately no corpus
mode* — *"a corpus-merged schema is the very thing the redesign removed."*

⚠️ **The trap to avoid:** riding the pool on `predefined_labels` looks correct and is not. With no
`induce` stage, no chunk carries `induced_labels`, so `effective_labels` returns the config labels
unchanged and **every chunk gets all 40** — reintroducing the exact defect the T1–T7 redesign
removed. Wire `label_inducer.lexicon` with the pool as its `candidates`.

### T-LABEL-2 — measure N, do not assume it

**No controlled accuracy-vs-label-count ablation exists in the literature.** Run extraction at
N ∈ {10, 20, 30, 40} on a held-out slice and report quality + throughput per N. Also run one
label-order permutation at fixed N to quantify order-sensitivity on our data. If 40 costs accuracy,
you will have the number that says so — **that is a result, not a failure.**

---

## 7. Stage 3 — extract → KG

With the OCR server down and the label pool wired, run the learned stages: chunk-level induction →
NER + relations → PII redaction + guardrails → enrichment → type consolidation → disambiguation →
graph assembly → link prediction → export.

### Watch these, in this order

1. **Peak VRAM per stage.** The 21.6 GiB figure was measured on 32 GB. If you approach 24 GB, lower
   `--microbatch-max-texts` before anything else.
2. **`disambiguate` is the wall-clock hog** — it was 1,274 s of a 1,743 s run previously. Expect it to
   dominate; that is normal, not a hang.
3. **Silent CPU fallback.** A previous run had `device: cuda` configured, ran at ~11.5 s/chunk with
   the GPU idle, and emitted **no warning** — the plain `gliner` library does not auto-place.
   `place_on_device` now verifies by read-back and raises on mismatch. If you see a device mismatch
   error, that guard is working.
4. **`torch.compile` recompile-thrash.** GLiNER2 with `dynamic=True` once pinned a worker at ~290%
   CPU with the GPU at 0% and no forward progress, because label-count and batch shape both varied
   per chunk. Label-set bucketing + micro-batching fixed it. If throughput collapses with no error,
   suspect this.

### T-KG-1 — two gates before link prediction, in order

1. **Canonicalize the relation vocabulary first.** Open extraction yields |R| ≈ |V| — ReVerb45K has
   21,623 relation phrases against 27,008 noun phrases. ULTRA's relation graph has |R| nodes and up to
   4|R|² edges; at |R| ~ 10⁴ that is prohibitive. `type_consolidation.cascade` exists — **measure |R|
   before and after** and assert a bounded vocabulary.
2. **Run Personalized PageRank over the raw graph and see whether it already closes the gap.**
   HippoRAG reports up to **+20% on multi-hop QA from PPR over an OpenIE graph with zero link
   prediction**, at 10–30× lower cost. If PPR is sufficient, the entire link-prediction workstream is
   unnecessary — and that is the cheapest possible refutation of the premise. `graph_features.pagerank`
   ships; its damping and iteration knobs are now reachable.

### T-KG-2 — the precision gate (unbuilt; build it here, against real data)

`graph_completion.ultra` is already wired with `target_precision: 0.9`. Use `mgalkin/ultra_50g` —
**MIT code and MIT weights**, verified separately. Zero-shot first, as a **re-ranker over candidates**,
never an all-pairs scorer. The paper's own fine-tuning ran on a **single RTX 3090** in 1,000–2,000
steps, so fine-tuning is a sub-hour experiment on exactly this hardware.

⚠️ **The honest ceiling:** on automatically-extracted open-schema KGs, SOTA **Hits@1 is 0.26–0.31**.
Unfiltered top-1 densification means **~7 in 10 added edges are wrong**, and a false edge in a
traversal graph does not merely fail to help — it **manufactures a spurious path** that pulls
unrelated documents into the neighbourhood. Densification failure is *worse* than recall failure.

Therefore:

- Hold out a manually-verified sample **from our own graph**. Never calibrate on FB15k-237.
- Plot precision vs threshold **per relation type** — it varies enormously.
- Admit only the top ~1–5% at **≥0.8 measured precision**.
- **Cap additions per node** so you densify the sparse periphery, not the hubs.
- Tag every added edge with provenance so it is ablatable and reversible.
- **Gate on end-to-end retrieval, not on MRR.** A densification that improves MRR and degrades
  retrieval is a failure.

---

## 8. Stage 4 — index

```bash
QDRANT_VER=v1.12.4
mkdir -p /workspace/qdrant-bin && curl -sL \
  https://github.com/qdrant/qdrant/releases/download/${QDRANT_VER}/qdrant-x86_64-unknown-linux-gnu.tar.gz \
  | tar xz -C /workspace/qdrant-bin
cd /workspace/qdrant-bin && nohup ./qdrant > /workspace/qdrant.log 2>&1 &
sleep 6 && curl -sf localhost:6333/healthz && echo " <- real qdrant up"

export LATENCE_QDRANT_URL=http://localhost:6333
latence-sink-qdrant load /workspace/latence-out/_latence/runs/<run_id>/export \
  --collection govdocs1 --url "$LATENCE_QDRANT_URL" --vector sparse
```

**Real Qdrant binary, no Docker.** `FakeQdrantClient` is offline-CI only — using it here would be
fake green, which is explicitly out of bounds.

**The `--vector sparse` flag is new (T-RET-1).** Before that work, `sink.py` created **dense-only**
`VectorParams`; the Export's `sparse_indices`/`sparse_values` rode into the *payload* as ordinary
lists — stored, never queryable. Meanwhile the query side already built `models.SparseVector` and
issued `query_points(..., using=...)`. **We had a query for an index that was never built.** Verify
the collection actually carries a named sparse vector before trusting any SPLADE number.

⚠️ **SPLADE-v3 weights are CC-BY-NC-SA-4.0.** Acceptable for this test; the flag
`--accept-noncommercial-license` is mandatory and **it must never become a shipped default.**
Exported sparse vectors inherit the NC + share-alike obligation.

⚠️ **No payload indexes are created.** Every metadata filter is currently an unindexed payload scan
(T-RET-5). Measure filtered retrieval before claiming it.

### Stage-4 verification

- [ ] Point count matches chunk count
- [ ] A **sparse-only** query returns documents a dense-only query does not (this is T-RET-1's
      GPU-ONLY acceptance criterion — it cannot be proven offline)
- [ ] A **graph-only** query returns something (T-RET-2's leg is live)
- [ ] A filtered query returns the expected subset

---

## 9. Known traps — read before debugging anything

These are all measured, not theoretical. Each one previously cost real time.

| Trap | Symptom | Resolution |
|---|---|---|
| Piping `verify-local.sh` to `tail` | Reports green on a red tree | Redirect, then `echo $?` |
| `uv run pytest -q` | No summary line (resolves to `-qq`) | Add `-o addopts=""` |
| Missing `PYTHONHASHSEED=0` | conftest refuses the run | Always set it |
| Checkpoint resume | "Re-run" finishes in 1.5 s | Clear the run dir |
| `PYTORCH_JIT=0` set by hand | Silent throughput loss on the 3 heaviest stages | Let the capability gate decide |
| `VLLM_USE_FLASHINFER_SAMPLER=0` by hand | Disables a working fast path on Ampere | Do not set it |
| Silent CPU fallback | `device: cuda`, GPU idle, ~11.5 s/chunk, no warning | `place_on_device` now raises |
| `torch.compile` thrash | ~290% CPU, GPU 0%, no progress, no error | Label bucketing + micro-batching |
| Graph `node_id`s | Seeding traversal with `"RFC9110"` returns nothing | They are 16-hex content hashes; resolve via `canonical_name` first |
| `Bm25Rescorer.from_export()` | Constructor call fails | It is a **classmethod** |
| BM25 at scale | O(corpus) pure-Python tokenisation **per query** | Fine at 3.5k chunks; a hard wall at BEIR scale. It is a **rescorer, not an index** |
| SEC without a User-Agent | HTTP 403 | Declare one; 10 req/s |
| v18 checkpoints | Will not load | `SCHEMA_VERSION = 19` is BREAKING; re-chunk from Parse |

**Three defects fixed in T-RET-4 that would have killed the run** (documented because they show the
*shape* of bug to look for — a contract asserted in one module, violated in another):

1. `run_benchmark.py` queried the edge parquet as `SELECT source, target`; the `GraphEdge` contract
   defines `source_node_id` / `target_node_id`. DuckDB binder error → **the whole multi-hop leg dies
   before the first query.**
2. Both harnesses passed the SPLADE checkpoint as `{"model_id": ...}`; the provider reads
   `cfg.get("model", "")` and does **not reject unknown keys** — so it silently resolved to `""`. The
   leg ran, contributed nothing, and reported success.
3. `build_questions.py` passed a `SparseVector` where `dict[str, float]` was required.

All three sat behind `# pragma: no cover - pod path`. **Assume more of this class exists in any
uncovered pod path you touch.**

---

## 10. Stage 5 — build the measurement half (against real data)

**This is the work that was deliberately deferred. Do it here, not before.**

### T-BENCH-1 — generation, hop-stratified

Generate from the real gold graph:

- **2–4 hops = serial depth chains.**
- **5–7 hops = fan-out / constraint-intersection**, not chains.

**Why the split is not arbitrary:** no published, human-validated benchmark exceeds **4 hops of
serial depth**. HotpotQA and 2Wiki are 2-hop; MuSiQue is 2–4; HopWeaver (the strongest 2025 automatic
generator) does 2. FanOutQA's "7" and FRAMES' "2–15" are **breadth**. Difficulty saturates before 7 —
MuSiQue's best model scores **57.9 / 47.9 / 28.1** at 2/3/4 hops, so a 7-deep chain measures
near-zero and discriminates nothing.

**Report stratified by hop count AND by shape. Never pooled.** Pooling a 7-wide fan-out with a 3-deep
chain into one "multi-hop accuracy" is precisely the move the shortcut literature exists to puncture.

### T-BENCH-2 — the eight necessity gates

| Gate | Condition | Reference |
|---|---|---|
| **G1** closed-book | A no-retrieval LLM scores at/near chance. **Report the number.** | FRAMES reports 0.408 |
| **G2** single-hop baseline | BM25+reader and dense+reader both well below fused and below oracle | FRAMES 0.474 / 0.66 / 0.729 |
| **G3** per-edge necessity | For every edge, a strong reader with the predecessor's answer **masked out of the sub-question** must fail | MuSiQue S3 — only **26.5%** of candidates survive |
| **G4** DiRe | Split supporting chunks, run each half independently, combine trivially | HotpotQA 68.8 / 2Wiki 63.4 / MuSiQue 37.8 |
| **G5** bridge leakage | No intermediate entity's name or alias appears in the question | MuSiQue `a2 ∉ q1` |
| **G6** adversarial distractors | Distractors are gold chunks of *other* paths sharing entity type — not random | MuSiQue "positive distractors" |
| **G7** evidence accessibility | Report Recall@k **and Support F1 separately** | HopWeaver: 0.705 vs 0.222 |
| **G8** human audit | Stratified by hop count; % judged genuinely multi-hop | HopWeaver 96.4–98.9, MuSiQue 91.2 |

**Report the funnel:** enumerated → structural → G3 → G4 → human-audited. MuSiQue's tail-survival was
26.5%; expect similar and publish it. A generator that reports only its output count is hiding its
selection rate.

⚠️ **Do not repeat the previous headline's shape.** The prior proof's gate was
`fused > dense AND fused > bm25` — which **selects** the rows where fusion wins. On that kept set,
"dense 0.000 / bm25 0.000 / fused 1.000" is close to definitional given single-passage gold and
binary recall@10. The strict set was **7 questions of 90**. That was reported honestly at the time,
but it is not the claim the headline implies.

⚠️ The old RFC multi-hop template **named the whole chain in the question text**
(`"RFC {a} obsoletes RFC {b}, which {rel} RFC {c}"`) and defined gold as chunks containing
`"RFC {c}"`. A plain BM25 query finds those. **Nothing was forced to traverse.** G5 exists for this.

### T-BENCH-3 — the KG ablation

Retrieval **with and without** the graph leg, stratified by hop count.

**Be honest about the expected size.** The independent literature is far more sceptical than the
GraphRAG marketing: DIGIMON (VLDB) found no graph method consistently beats VanillaRAG (+1.7 points
on Quality; G-retriever *underperformed ZeroShot by 14.17*). GraphRAG-Bench (ICLR 2026) found graph
methods get **higher recall but lower relevance** → noisier context, lower generation accuracy. Han
et al. found plain RAG beats graph RAG on single-hop (NQ F1 64.78 vs 63.01) and that entity
extraction **loses ~34% of answer entities**. The genuinely real number: **13.6%** of MultiHop-RAG
queries are answered *only* by GraphRAG; routing captures **+1.1%**, dual retrieval **+6.4% at 2×
compute**.

And none of those papers compared against a **strong hybrid with a reranker** — every baseline was
dense-only or BM25-only. **Our stack is stronger than every baseline they beat, so their deltas are
an upper bound, probably a loose one.** If the KG does not earn its weight, report that.

### T-FUSE-1 — fusion comparison

`WeightedFuser` already implements per-list min-max-normalised weighted sum — the family Bruch et al.
(TOIS 2023) recommend. Run it against `RrfFuser(k=60)` on the same candidate lists and sweep weights
on a small labelled slice ("a handful of labeled queries" suffices).

⚠️ **RRF's `k=60` is not principled.** From the original 2-page SIGIR'09 paper: *"k = 60 was fixed
during a pilot investigation and not altered during subsequent validation."* Its own sweep shows MAP
.2139 (k=30) → **.2145 (k=60)** → .2147 (k=80). It is a flat-plateau point measured once over **30
homogeneous TREC runs** — not 4 heterogeneous signals of query-dependent reliability. RRF was
explicitly invented *as a baseline*.

**Report the comparison, not a preferred outcome.**

### T-BENCH-4 — external benchmarks

**StrategyQA** is the best fit: open-domain scale, *implicit* multi-hop (the strategy is not stated in
the question, so KG traversal should shine), an **official Recall@10 scorer**, and a **stale 2022 SOTA
of 0.457** with nobody publishing retrieval numbers since. Cost of entry: index the 2020-05-11
Wikipedia Cirrus dump yourself. Its licence is **UNVERIFIED** (AI2's dataset page is dead) — confirm
before any external claim.

**Dropped:** Bamboogle (125 questions, no corpus, no gold evidence — recall@k is not computable).
**Excluded:** LoCoMo (**CC BY-NC 4.0**).

---

## 11. The iterate loop

The pattern that works here, learned the hard way:

1. **Run one stage. Measure. Commit the measurement.**
2. **When something fails, fix the SHAPE, not the symptom.** Four audit rounds once patched the
   consequences of a design defect (`rehome_page_maps`, an LRU window, then that window's own
   justification) and never the shape. The eventual fix *deleted* the compensating mechanism — that
   is the tell that a fix is right.
3. **Max 2 live validation runs per fix cycle.** Prove the core claim, commit, log residuals to a
   findings file, move on. Stored evidence and component probes beat end-to-end reruns.
4. **Never re-run against a frozen dataset when the pipeline changed.** That is the 77-minute trap.
5. **Re-run `verify-local` before every commit that touches shared code**, capturing the real exit
   code.

### Escalate to the operator rather than deciding, when:

- A licence question is unresolved (do not guess; mark **UNVERIFIED**)
- A measured result contradicts a stated goal (report it; do not quietly lower the bar — a bar was
  once lowered unilaterally in this workstream and that was wrong)
- The corpus behaves so differently that a ticket's premise no longer holds

---

## 12. Deliverables to harvest

Everything below must be **durably copied off the pod** — a pod dies without warning.

| Artifact | Path | Notes |
|---|---|---|
| Corpus manifest | per-source counts, bytes, licence, **measured** file-type histogram | Must state the legacy-OLE/no-modern-OOXML limit |
| Stack config | the wizard-emitted YAML | Proves the entry-gate claim |
| `quality-report.json` / `.md` | run dir | Per-stage counts, durations, drift |
| Stage charts | SVG | Chunk-size distribution, entity/relation funnel, KG degree + centrality |
| Label-profile artifact | profiler output | Curve + certificate + `C(N)` + Zipf α + Good–Turing |
| Export | `records.parquet`, `graph-{nodes,edges}.parquet`, `bm25-stats.json` | |
| Benchmark results | JSON + charts | **Stratified by hop count and shape** |
| Throughput/VRAM log | per stage | The numbers nobody can get offline |
| `PROOF.md` | narrative | Must carry its own caveats, as the previous one did |

⚠️ **Publication constraint:** anything derived from **IIT-CDIP / UCSF IDL** is fair-use and must
**not** enter a public artifact. Publishable material comes from **GovDocs1 (CC0)**, **SEC VPRR (SEC
public information)**, **OHR-Bench (CC-BY-4.0)** or **DocLayNet (CDLA-Permissive)** only.

⚠️ `graph.ttl` was dropped from a previous deliverable set because its sha256 node ids trip the
secret-scan hash detector. Expect the same.

---

## 13. Open decisions for the operator

These are **not** for the incoming session to decide alone:

1. **Push the 39 commits?** `main` is ahead of `origin/main`. Never pushed without an explicit ask.
2. **Staged or single-pass?** §2 argues staged. The 24 GB ceiling and the parse→profile→extract
   ordering both force it.
3. **SSH access** to the 3090 — not requested until the operator is ready to start.
4. **Modern OOXML coverage.** GovDocs1 is legacy-OLE. If modern OOXML at volume matters, the source
   is SAM.gov (measured: PDF 70% / DOCX 20% / XLSX 5.7%), whose **ToU prohibits scraping** — it must
   go through the keyed public API with rate limiting. Currently **out of scope** and the manifest
   must say so.

---

## 14. One-paragraph summary for someone in a hurry

The data-processing pipeline is built, merged and green (4,590 tests, `verify-local` exit 0, mypy
strict on 33 packages, 39 commits unpushed on `main`). The measurement harness — question generation,
the necessity gates, the KG ablation, external benchmarks, link-prediction calibration — is
**designed but deliberately unbuilt**, because it needs a real gold graph and a real extracted KG to
be worth writing. Run the pod in stages: bring up the OCR server → parse GovDocs1 + SEC VPRR +
OHR-Bench → **tear the server down** → profile the parsed text for a ≤40-type label pool (no LLM) →
run the learned stages with `label_inducer.lexicon` narrowing that pool per chunk → index into a real
Qdrant with named sparse vectors → *then* build the benchmark against what is actually there. Do not
set `PYTORCH_JIT=0` or `VLLM_USE_FLASHINFER_SAMPLER=0` — both are Blackwell remedies that silently
cost throughput on this Ampere card. Do not pipe `verify-local.sh` to `tail`. Do not build the
benchmark against the RFC fixture.
