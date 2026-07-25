# Tutorial — fresh GPU pod → AI-ready data (enterprise-SOTA, stage by stage, then end-to-end)

This walkthrough takes you from a **bare GPU pod with a folder of mixed file types** to AI-ready
data, using the **enterprise-SOTA pipeline** — served-vLLM OCR, LLM schema induction, fused GLiNER2
entity+relation extraction, learned financial-PII redaction, the neural **GLinker** entity-linker +
audited resolver, IBM Granite r2 embeddings, and ULTRA graph completion. It follows the same spine
as before:

1. a working install (+ a running vLLM OCR endpoint, the one real prereq),
2. running and **inspecting each Stage in isolation**,
3. running the **whole pipeline end-to-end**, and
4. **discovering and understanding the final output**.

Every command is copy-pasteable. Where a claim comes from a real pod run it is stated as a warning.

> **Honest expectations — read this first.**
>
> - **The SOTA pipeline needs a GPU** (the OCR VLM, GLiNER2, GLinker, and Granite embedder are all
>   neural) **and a running vLLM OCR endpoint** the parser POSTs page-images to (§2). On a CPU-only
>   host the learned Stages are honestly **skip-with-flag** at `latence stack validate` (device
>   honesty) — the config still validates, but a real run needs CUDA. Deployment topologies
>   (Docker Compose, K8s/Helm, air-gapped, the Blackwell note) live in
>   [`docs/deployment.md`](deployment.md).
> - **This is the ingest/AI-ready-data product, not a query engine.** It emits a RAG corpus + a
>   knowledge graph **as files** — the retrieval/query engine that consumes them is a separate
>   product. The exact build contract for that consumer is
>   [`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`](RETRIEVAL-ENGINE-INPUT-SPEC.md).
> - **Full-corpus GLiNER2 + GLinker is validated at small scale.** The composed enterprise stack is
>   pod-validated end-to-end on a small corpus; it is the SOTA half of every seam, not a
>   throughput-at-1M-docs benchmark.

---

## 0. Prereqs — what the pipeline produces

Point the pipeline at a folder of messy enterprise files (PDF, scans, Office docs, text, markdown)
and it emits, per corpus run, three deliverables as **plain files** (JSONL / Parquet / RDF / JSON) on
any fsspec backend — never a live database (ADR-0017):

- **A chunk-level RAG corpus** — `records.jsonl` + `records.parquet`: retrieval text + full
  `Provenance` (source + offsets) + `Classification` + per-chunk `induced_labels` + a KG
  `context_header` + a dense Granite r2 `embedding` vector. With PII ON the text is the
  **masked** (PII-safe) variant.
- **An evidence-linked knowledge graph** — `graph-nodes.parquet`, `graph-edges.parquet`,
  `graph.ttl`, `graph.graphml`: typed entities + relations (asserted **and** ULTRA-completed),
  every edge citing the mentions/documents that justify it.
- **A Quality Report** — `quality-report.json` + `quality-report.md`: per-Stage counts, KG
  statistics, the disambiguation merge audit, contract-completeness, no-PII-leak check.

The full field reference for consumers lives in
[`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`](RETRIEVAL-ENGINE-INPUT-SPEC.md).

---

## 1. Set up the fresh pod

### 1.1 Clone the framework

```bash
cd /workspace
git clone <your-remote-url> latence-framework
cd latence-framework
```

### 1.2 Install the packages the enterprise-SOTA stack uses — editable, **letting deps resolve**

RunPod's system Python is **PEP-668 "externally managed"**, so every `pip install` needs
`--break-system-packages` (or a venv — but the pod's global torch/CUDA is usually why you're on a
pod, so we install globally here).

> **Warning — do NOT use `--no-deps` for the editable installs.** The learned providers declare real
> hidden deps (`peft` for GLiNER2, `glinker` + `flashdeberta` for the GLinker disambiguator, `gliner`
> for the PII redactor). A `--no-deps` editable install silently skips them and the Stage then
> crashes at runtime. Install each package **normally** so its deps resolve:

```bash
cd /workspace/latence-framework

# core + the providers the enterprise-SOTA stack composes (deps resolve normally)
pip install --break-system-packages \
  -e packages/latence-core \
  -e packages/latence-parser-lighton-vllm \
  -e packages/latence-schema-inducer \
  -e packages/latence-extract-gliner2 \
  -e packages/latence-pii-gliner2 \
  -e packages/latence-pii-gliner \
  -e packages/latence-disambig-glinker \
  -e packages/latence-linkpred-ultra \
  -e packages/latence-embedder-st
```

Then the runtime deps the learned/export Stages import (some are declared, but install them
explicitly so nothing is missing at run time):

```bash
pip install --break-system-packages \
  gliner2 gliner glinker flashdeberta peft accelerate \
  sentence-transformers openai \
  pypdfium2 pillow \
  pyarrow fsspec networkx rdflib
```

- `gliner2 peft accelerate` — the GLiNER2 fused entity+relation extractor (mdeberta-v3 backbone).
- The redactor both §3's wizard and §4's hand-written stack wire is `redaction.gliner2` (#197,
  ADR-0041 §7a) — the SAME GLiNER2 engine `extract` runs, so redaction adds no second model library.
  Do not confuse the two names: the `gliner2` line in the pip block above is the third-party **PyPI
  wheel** (the library + mdeberta-v3 backbone), while the `redaction.gliner2` **entry point** is
  registered only by the **`packages/latence-pii-gliner2`** distribution — which is why that package
  is in the editable list.
- `gliner` + **`packages/latence-pii-gliner`** — the *alternative* learned redactor
  `redaction.gliner_pii`. Still fully supported (it drops into §4's `redact` stage unchanged), just
  not what the guided setup blesses. Installed here only so you can swap it in; drop both lines if
  you do not want the second model library.
- `glinker flashdeberta` — the neural GLinker entity-linker (`disambiguation.glinker`): the
  `gliner-linker-large` biencoder + `gliner-linker-rerank` reranker + FlashDeBERTa.
- `sentence-transformers` — the Granite r2 embedder (the GLinker resolver's opt-in STS rung **and**
  the corpus-export vectors; ADR-0045).
- `openai` — the OpenAI-compatible client used by **both** the vLLM OCR parser and the schema-
  induction LLM.
- `pypdfium2 pillow` — the vLLM OCR parser's page rasteriser (PDF→image, license-clean PDFium).
- `pyarrow fsspec` — Parquet export + storage; `networkx rdflib` — the GraphML/TTL KG export.

`latence-linkpred-ultra` (`graph_completion.ultra`) has no extra runtime pip deps here — its ULTRA
checkpoint is not vendored, so the Stage runs `skip_if_unavailable` (a graceful skip, never a failed
run) unless you supply one.

### 1.3 Required env — JIT off, the HF-Hub XET workaround, and the induction key

> **Warning — `nvrtc: failed to open libnvrtc-builtins.so.13.0`.** On recent NVIDIA (SM 12.x /
> Blackwell) with torch 2.11-cu13x, GLiNER2 / DeBERTa can crash inside a JIT-compiled kernel. Disable
> the JIT **before running anything**:

```bash
export PYTORCH_JIT=0
```

> **Warning — Blackwell (SM 12.x) model downloads hang / error on the XET backend.** Getting the
> GLiNER2 / GLinker / Granite weights down on a fresh Blackwell pod, the Hugging Face `hf_xet`
> transfer backend was flaky in validation. Force the classic HTTP download path:

```bash
export HF_HUB_DISABLE_XET=1
```

The schema-induction Stage calls an OpenAI-compatible LLM. Provide the key **as an environment
variable only — never in a file** (an inlined `api_key` would be scrubbed from `manifest.json`, but
don't rely on that; keep keys out of YAML):

```bash
export OPENROUTER_API_KEY="sk-or-..."     # OpenRouter (the default endpoint)  # pragma: allowlist secret
# or, for direct OpenAI:
# export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

Model default is `openai/gpt-4.1-mini` reached via `base_url: https://openrouter.ai/api/v1`. Set the
base URL to `http://<vllm-host>:8000/v1` for an on-prem vLLM (any placeholder key), or leave it
`null` for direct OpenAI with `$OPENAI_API_KEY`.

> **GLiNER2 `compile` is ON by default on GPU again (T7), and it is now safe.** It was forced OFF
> because gliner2 compiles with `torch.compile(dynamic=True)` and, while each forward was one
> varied-length chunk scored against a corpus-wide ~40-label schema, a fresh (label-count, batch)
> shape family per chunk **recompile-thrashed** (pod-measured: worker pinned ~290% CPU, GPU 0%, no
> progress). The extraction redesign removes both varying dimensions: chunks are **bucketed by label
> set** (constant label dimension within a bucket) and each bucket is **micro-batched** (capped batch
> dimension), so dynamo sees one shape family per bucket. Two guards remain, so a backend that cannot
> compile costs throughput only: a compile fault at load degrades to a plain load, and a lazy
> `BackendCompilerFailed` on the first forward re-loads the uncompiled model once. Pin
> `compile: false` if you want it off.

### 1.4 Verify the install

```bash
# the CLI is on PATH once latence-core is installed
latence --help                       # or: python -m latence_core.cli --help

# imports resolve (no output == good)
python -c "import gliner2, gliner, sentence_transformers, peft, pyarrow, networkx, rdflib, openai; print('deps OK')"

# every Stage provider the SOTA stack uses is discoverable
python - <<'PY'
from latence_core.capability import ProviderRegistry
names = set(ProviderRegistry().names())
for n in ("source.local_folder","screening.intake_signature","parser.lighton_vllm",
          "chunk.markdown","screening.content_keyword","label_inducer.llm",
          "fused_entity_relation.gliner2","redaction.gliner_pii","redaction.gliner2",
          "profiling.statistical",
          "disambiguation.glinker","graph.canonical","graph_completion.ultra",
          "context.kg_header","export.knowledge_graph","export.jsonl_parquet"):
    print(("ok " if n in names else "MISSING "), n)
PY
```

> `ProviderRegistry().names()` lists every discovered provider; equivalently
> `python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='latence.providers')])"`.
> A provider printing `MISSING` means its package wasn't installed in §1.2 — `disambiguation.glinker`
> ships in the **separate** `latence-disambig-glinker` heavy package (ADR-0016).

---

## 2. Serve the LightOn-OCR vLLM endpoint (the OCR prereq)

The enterprise parse Stage is `parser.lighton_vllm` — it does **not** run OCR in-process; it POSTs
each rasterised page to a **running vLLM server**. Pod-measured, vLLM at concurrency is
~**2.7× faster than optimised in-process transformers**, which is why production OCR is a warm
server (the full recipe + tuning is [`docs/serving-lighton-vllm.md`](serving-lighton-vllm.md)).

Stand it up on the pod (LightOn's official, optimal-throughput config):

```bash
export HF_HUB_DISABLE_XET=1   # (from §1.3 — the Blackwell download workaround)
vllm serve lightonai/LightOnOCR-2-1B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 10000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --max-num-seqs 32
```

The server exposes an OpenAI-compatible endpoint at `http://localhost:8000/v1`, serving the model id
`lightonai/LightOnOCR-2-1B` (Apache-2.0 weights **and** code). Those two values — the `base_url` and
the `model` — are exactly what the parser (and the `latence setup` wizard, §3) ask for. `api_key`
is any placeholder (`EMPTY`); a local vLLM does not check it.

> **Blackwell / FlashInfer caveat (KNOWN).** On bleeding-edge **Blackwell (SM 12.x)** GPUs, vLLM
> 0.24's bundled FlashInfer refuses a capable `sm_120` device at startup with
> `FlashInfer requires sm75+` (it misparses compute capability 12.0 as `12`). Export
> **`VLLM_USE_FLASHINFER_SAMPLER=0`** before the serve — that disables only the *sampler*, so
> cudagraphs stay ON. `--enforce-eager` also clears the abort but kills cudagraphs and floors your
> throughput; the compatibility matrix records it as deliberately **not** the fix
> (`FM-FLASHINFER-SM120`). The other exit is upgrading `flashinfer` to an SM 12.x build. Mainstream
> GPUs (Hopper / Ada / Ampere, CUDA ≥ 12.6) need none of this.

Sanity-check the server is up before running the pipeline:

```bash
curl -s http://localhost:8000/v1/models | python -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])"
# → ['lightonai/LightOnOCR-2-1B']   (the base_url + model to point the parser at)
```

GPU / driver requirements (a LightOnOCR-2-supporting vLLM needs torch cu126+, i.e. driver ≥ 12.6)
and the container/K8s topologies are in [`docs/deployment.md`](deployment.md).

---

## 3. The guided path: `latence setup --profile enterprise-sota` → `latence process`

**This is the fast path — start here.** A guided, deterministic wizard converges to a clean,
validated enterprise-SOTA config, then one command runs it. §4 hand-authors the same stack so you
understand every knob; you don't need it to get a run going.

```bash
# Interactive: each prompt has a sensible default + a one-line "why it matters".
latence setup --profile enterprise-sota
# → writes ./latence.stack.yaml (validated), then prints "Run `latence process` to start."
latence process
```

**What the enterprise-SOTA wizard asks** (each with a default; press Enter to accept):

| Prompt | Default | Why it matters |
|---|---|---|
| **Source folder** | `file://$PWD/corpus` | The folder Latence reads. Validated to exist; its file types are **auto-detected** and reported. |
| **Profile** | — (you passed `enterprise-sota`) | The tier. `enterprise-sota` emits the pod-validated composed stack; `lite` is the CPU-first laptop path. |
| **Output / storage URI** | `file://$PWD/latence-out` | Where state + the exported RAG corpus and KG are written. |
| **OCR endpoint base URL** | `http://localhost:8000/v1` | The **served vLLM OCR endpoint** from §2 the parser POSTs page-images to. (enterprise-sota forces `parser.lighton_vllm` and **skips** the parser-choice prompt.) |
| **OCR model id** | `lightonocr` | The model your vLLM server registers — for the §2 recipe that is `lightonai/LightOnOCR-2-1B`. |
| **Schema-induction LLM base URL / model** | OpenRouter + `openai/gpt-4.1-mini` | The quality lever — a small LLM discovers your corpus's own `{entity, relation, pii}` schema. The **API key is read from the environment at run** (`$OPENROUTER_API_KEY`, or `$OPENAI_API_KEY` for a direct-OpenAI base URL) — **never stored** in the file. The wizard warns if it is unset. |
| **PII redaction + guardrail screening** | ON | **The one optional toggle.** ON adds the `content_screen` + `redact` stages (learned **financial-PII** detection). OFF omits exactly those two stages. |
| **Mask before extraction** (`--extract-on`) | `unmasked` | **Advanced (W18).** `unmasked` (default) — extraction reads the real text, the KG keeps entities. `masked` — extraction reads the REDACTED chunks so PII never reaches the KG (stricter posture; requires PII ON). |
| **Domain / description hint** | (empty) | Free text, recorded as a header comment for readers. |

Device is **pinned to `cuda`** for enterprise-sota (every learned Stage is neural); on a CPU-only
host they validate as skip-with-flag.

The wizard then **generates the enterprise-SOTA spine**, **dry-validates it** (structural + DAG +
capability wiring — the SAME machinery `latence stack validate` builds on, no pipeline run), and
**saves** `./latence.stack.yaml`. The generated stack is exactly the SOTA stack of §4:

```text
source → intake_screen → parse[lighton_vllm] → chunk[640] → content_screen
  → induce[corpus] → extract[gliner2, thr 0.3] → redact[gliner2, financial-PII]
  → profiling → consolidate_types → disambiguate[glinker] → graph[+pagerank features] → complete[ultra, skip_if_unavailable]
  → enrich(context) → export_kg + export_corpus[Granite r2, context_columns, bm25]
```

**Deterministic + scriptable (CI/repro).** `--non-interactive` takes every answer from flags, and
the **same inputs produce a byte-identical config**:

```bash
latence setup --non-interactive --profile enterprise-sota \
  --source file:///workspace/corpus --storage file:///workspace/latence-out \
  --ocr-base-url http://localhost:8000/v1 --ocr-model lightonai/LightOnOCR-2-1B \
  --llm-base-url https://openrouter.ai/api/v1 --llm-model openai/gpt-4.1-mini \
  --pii --domain "retail banking contracts" \
  --output ./latence.stack.yaml
latence process --config ./latence.stack.yaml --run-id run-0001
```

> **What the enterprise-sota profile wires for you (ADR-0046, `stacks/gpu-sota-glinker.yaml`).**
>
> - **parse → `parser.lighton_vllm`** — the served-vLLM LightOn OCR endpoint (§2). The wizard writes
>   your `base_url` + `model`, with `api_key: EMPTY` and `max_concurrency: 8`.
> - **induce → `label_inducer.llm`** at **chunk** granularity (`dedup: true`, `temperature 0`) —
>   the schema-induction quality lever (its `base_url`/`model`/key are the LLM prompts above). Each
>   chunk keeps only its OWN small, relevant label set; `dedup` induces once per DISTINCT chunk
>   text, so a corpus's boilerplate costs one call, not one per occurrence.
> - **extract → `fused_entity_relation.gliner2`** (`threshold 0.3`, labels floor
>   `[person, organization]`, `compile` on by default on GPU — T7) and **redact → `redaction.gliner2`**
>   (#197, ADR-0041 §7a — the SAME GLiNER2 engine `extract` runs, so redaction adds no second
>   model library; chunk-level, financial-PII floor — the W16 `redact → enrich → export_corpus`
>   dataflow).
> - **disambiguate → `disambiguation.glinker`** — the neural GLinker entity-linker + audited resolver
>   (`context_window 200`, `threshold 0.3`, reranker + FlashDeBERTa on) with the **IBM Granite r2**
>   embedder. The precision-by-default knobs (embedding-merge OFF, fuzzy edit-distance rung +
>   distinguishing-token guard ON) ride the provider defaults — the config stays minimal and the ER
>   is precise out of the box.
> - **graph → `graph.canonical`** with the ADR-0051 **`graph_features: graph_features.pagerank`**
>   block — PageRank `centrality` + label-propagation `community` stamped on every node, promoted to
>   `graph-nodes.parquet` columns and projected onto each chunk. These are what the retrieval
>   Expander's graph bonuses and the Knapsack packer's centrality/cluster inputs read; without the
>   block they are absent/`0.0`.
> - **→ complete → `graph_completion.ultra`** (`skip_if_unavailable: true` — no vendored checkpoint
>   means a graceful skip, never a failed run) → **`export.knowledge_graph`**
>   + **`export.jsonl_parquet`** with the Granite r2 corpus vectors **and the ADR-0049
>   `bm25: tokenizer.regex` block** — the index-time term statistics (`bm25-stats.json` +
>   `bm25-postings.parquet`) the query-side sparse rescorer runs on. Without it the stack ships
>   dense-only and the hybrid fuser has no sparse leg.

> **PII note (W16, ADR-0042).** With **PII ON**, `redact` is a **chunk-level transform**: it consumes
> the induced **chunk** stream and sets each chunk's `masked_content` (each chunk ≤ the model window,
> so no PII is truncated), and `enrich` reads THAT redacted chunk stream
> (`redact → enrich → export_corpus`). The corpus Export materializes `masked_content`, so the
> exported RAG corpus is **both masked (PII-safe) AND context-enriched** — the per-chunk
> `context_header`/`induced_labels` and the masking compose, with no trade-off.

> **Redaction floor + mask-before-extract (W18, ADR-0044).** (1) A **universal financial-PII floor** —
> `credit_card` / `iban` / `ssn` are ALWAYS masked at the shared redaction seam no matter which
> detector backend you configure, so a learned redactor that *misses* one still cannot leak it (the
> floor is non-bypassable). (2) An optional **`--extract-on masked`** mode: extraction reads the
> REDACTED chunk stream, so PII never reaches entity/relation extraction or the KG at all. Default
> `unmasked` keeps the KG's real entities; with `masked` the wizard wires `extract` to depend on
> `redact` for you.

---

## 4. Author the SOTA stack YAML by hand (understand every knob)

The wizard above generates this file for you. Author it by hand when you want to see — and tune —
every knob. Save it as `stacks/my-corpus.yaml`. This is the **exact** enterprise-sota spine (the
`stacks/gpu-sota-glinker.yaml` shape with the full induce → enrich → complete spine).

> **Why this config matters more than the model.** With generic English labels, a low threshold, and
> no schema induction, the output is noisy — pronouns/laws/products get mistyped as entities and the
> relations are garbage. Turn schema induction **on** (the LLM discovers the domain's real labels per
> corpus), use **`granularity: chunk`** (each chunk keeps only its own relevant types — a
> corpus-wide merged schema has to be salience-capped, which drops the rare-but-critical type from
> the one chunk that needed it and scores every other chunk against types that are not in it; type
> consistency is earned downstream in the resolution phase), give redaction real **financial-PII**
> labels, cap chunks at
> **`max_tokens: 640`**, and let the GLinker disambiguator's precision defaults ride. Then entities
> are correctly typed and the relations are meaningful.
>
> **Why 640.** The GLiNER2 extractor/redactor run on mdeberta-v3, whose token window is **768**.
> `chunk.markdown`'s `max_tokens` is counted by the *chunker's* tokenizer, which runs ~1.2× the
> model's (pod-measured: a 768-chunker-token cap hit mdeberta median 666 / max 921). So 640
> chunker-tokens ≈ ≤768 mdeberta-tokens — as big as is safe. The providers also carry a
> `max_len: 768` guard as the hard backstop.

```yaml
# stacks/my-corpus.yaml — the enterprise-SOTA stack:
#   source → intake_screen → parse[lighton_vllm] → chunk[640] → content_screen
#     → induce[corpus] → extract[gliner2, thr 0.3] → redact[gliner2, financial-PII]
#     → profiling → consolidate_types → disambiguate[glinker] → graph[+pagerank features] → complete[ultra]
#     → enrich(context) → export_kg + export_corpus[Granite r2 + bm25]
#
# {CORPUS_DIR} / {STORAGE_URI} are placeholders — fill them in §4.1 before `latence run`.
name: my-corpus-stack
storage_uri: "{STORAGE_URI}"
stages:
  # 1) Read the folder. Only these extensions are admitted; adjust to your corpus.
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: "{CORPUS_DIR}"
      sensitivity: internal          # default Classification.sensitivity for the corpus
      extensions: [txt, md, markdown, pdf, docx, pptx, xlsx, png, jpg, jpeg, zip]

  # 2) Intake screening: reject oversized / malformed / zip-bomb files (quarantined, not crashed).
  - name: intake_screen
    capability: intake_screening
    provider: screening.intake_signature
    depends_on: [source]
    config: {max_bytes: 2097152, max_zip_ratio: 50.0}

  # 3) Parse: the served-vLLM LightOn OCR endpoint (§2). base_url + model point at your vLLM box;
  #    api_key is any placeholder; max_concurrency fans pages out (pair with --max-num-seqs).
  - name: parse
    capability: parse
    provider: parser.lighton_vllm
    depends_on: [intake_screen]
    config:
      base_url: http://localhost:8000/v1
      model: lightonai/LightOnOCR-2-1B
      api_key: EMPTY
      max_concurrency: 8

  # 4) Chunk at 640 (the mdeberta-window-calibrated max — see the callout above).
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
    config: {max_tokens: 640, overlap_tokens: 80, min_tokens: 8}

  # 5) Content screening: flag/quarantine injection & keyword-risk chunks before extraction.
  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]

  # 6) Schema induction (the quality lever): a small LLM discovers each CHUNK's own
  #    {entity, relation, pii} types. granularity: chunk → every chunk carries only the types that
  #    are actually in it (best recall AND best zero-shot precision; the small label set is also
  #    what keeps the GLiNER2 span-scoring tensor small). Corpus-wide type consistency is earned
  #    downstream in the resolution phase, not bought here with a salience cap.
  - name: induce
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [content_screen]
    config:
      model: openai/gpt-4.1-mini             # OpenRouter model id (openai/ prefix). Direct OpenAI: gpt-4.1-mini
      base_url: https://openrouter.ai/api/v1 # null ⇒ OpenAI; http://<vllm-host>:8000/v1 ⇒ on-prem vLLM
      # api_key comes from $OPENROUTER_API_KEY (falls back to $OPENAI_API_KEY). NEVER inline it here.
      granularity: chunk                     # per-chunk induction: each chunk keeps its own types
      dedup: true                            # induce once per DISTINCT chunk text (the cost lever)
      max_tokens: 2048                        # truncated reply retries at double, then fails open
      max_labels: 40
      temperature: 0.0                        # deterministic

  # 7) Fused NER+RE in ONE pass (GLiNER2). config labels are a FLOOR; the induced entity/relation
  #    types are UNIONed on per corpus (config ∪ induced). threshold 0.3 = the SOTA recall floor.
  #    `compile` is left to the provider default (T7: ON on cuda — the forward is bucketed by label
  #    set, so it no longer recompile-thrashes; §1.3).
  - name: extract
    capability: fused_entity_relation
    provider: fused_entity_relation.gliner2
    depends_on: [induce]
    config:
      device: cuda
      labels: [person, organization]         # entity floor; induced entity_types union on
      relation_labels: [works for, partner of]  # relation floor; induced relation_types union on
      threshold: 0.3

  # 8) Learned PII redaction with FINANCIAL PII, not just person/email/phone. The provider is
  #    `redaction.gliner2` — the SAME unified GLiNER2 engine `extract` above already runs (#197,
  #    ADR-0041 §7a), so redaction loads no second model library. (Its twin `redaction.gliner_pii`
  #    remains supported and drops in here unchanged, but it is not what the guided setup blesses.)
  #    W16 (ADR-0042): redact is a CHUNK→CHUNK transform — it consumes the induced CHUNK stream and
  #    sets each chunk's masked_content (each chunk ≤ the model window, so no PII is truncated). It
  #    depends_on induce so the induced pii_types UNION onto this floor; enrich then reads THIS
  #    redacted chunk stream, so export_corpus is masked AND enriched.
  #    T2: the `pii_labels` floor is UNIVERSAL — applied to every chunk, never subsetted by what a
  #    chunk induced (a leaked card in an "irrelevant" chunk must still be masked); only the induced
  #    pii_types are per chunk. `guardrails: true` turns on the GUARDRAILS half of the same
  #    checkpoint under the same rule: every chunk carries the universal guard floor (prompt
  #    injection / harmful content) unioned with its own induced guard_types, and a hit is FLAGGED
  #    as a RiskMarker in the corpus, never masked. The deterministic keyword content_screen Stage
  #    above is a separate, earlier pre-filter and is untouched.
  - name: redact
    capability: redaction
    provider: redaction.gliner2
    depends_on: [induce]
    config:
      device: cuda
      pii_labels: [person, email, phone number, address, iban, credit card number, tax id, bank account number, social security number]
      guardrails: true
      max_len: 768
      threshold: 0.5

  # 9) Corpus profiling (statistics + drift diagnostics) over docs + mentions.
  - name: profiling
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, extract]

  # 9b) Type consolidation: canonicalize the corpus's TYPE vocabulary (T3, ADR-0054). Labels are
  #     induced PER CHUNK above, which is what keeps each chunk's label set small, relevant and
  #     uncapped — and what makes one chunk say `org` where another says `organization` or
  #     `company`. This Stage folds that drift into ONE canonical vocabulary and rewrites every
  #     mention's + relation's label onto it BEFORE the resolver runs (the resolver votes a
  #     cluster's type from its members' labels and gates its low-precision rungs on type
  #     compatibility, so drifted types in = split clusters and mistyped entities out). The
  #     semantic rung rides the SAME Granite embedder over a few dozen type STRINGS — no second
  #     model. The raw induced label stays on each record and the alias map lands on the Quality
  #     Report, so every fold is visible and correctable via `aliases`.
  - name: consolidate_types
    capability: type_consolidation
    provider: type_consolidation.cascade
    depends_on: [extract]
    config:
      use_embedding_merge: true
      embedder:
        provider: embedding.sentence_transformers
        config: {model: ibm-granite/granite-embedding-311m-multilingual-r2, dimension: 768, device: cuda, attn_implementation: flash_attention_2}

  # 10) Disambiguation: the enterprise GLinker entity-linker + audited resolver (ADR-0046).
  #     Neural L2→L3→L4→L0 linking (gliner-linker biencoder + reranker + FlashDeBERTa) LINKS each
  #     mention to its canonical entry, then the audited resolver FUSES by exact/alias/acronym/
  #     substring/fuzzy with a no-silent-over-merge audit log. The precision defaults ride:
  #     embedding-merge OFF (retrieval embedders over-merge short names), fuzzy edit-distance rung +
  #     distinguishing-token guard ON. The Granite embedder is the opt-in STS rung (kept here as
  #     wiring; it still powers export_corpus's vectors).
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.glinker
    depends_on: [consolidate_types]
    config:
      device: cuda
      context_window: 200          # ±chars of chunk context for the neural linker input
      threshold: 0.3               # neural linking (L0) confidence threshold
      use_reranker: true           # the L4 gliner-linker reranker
      use_flash_deberta: true      # FlashDeBERTa when flash_attn present (graceful fallback)
      embedder:
        provider: embedding.sentence_transformers
        config: {model: ibm-granite/granite-embedding-311m-multilingual-r2, dimension: 768, device: cuda, attn_implementation: flash_attention_2}

  # 11) Assemble the canonical, evidence-linked knowledge graph — WITH the ADR-0051 graph-feature
  #     enrichment. `graph_features` resolves the PageRank computer and stamps a `centrality` score
  #     + a label-propagation `community` id onto every node; export_kg promotes them to
  #     graph-nodes.parquet columns and `enrich` projects them onto each chunk
  #     (kg_node_centrality / kg_node_community). Those are the signals the retrieval Expander's
  #     graph bonuses and the Knapsack packer's centrality/cluster inputs read — WITHOUT this block
  #     they are all absent/0.0 and the graph half of the stack is wired but dead.
  #     Pure Python, CPU-only, deterministic, zero extra deps, and additive: omit the block and the
  #     emitted graph is byte-identical to before.
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: my-corpus-stack
      graph_features:
        provider: graph_features.pagerank
        config: {}

  # 12) ULTRA graph completion (zero-shot inductive link prediction). skip_if_unavailable: true —
  #     the checkpoint is not vendored, so on a host without it the Stage skips-with-flag (no
  #     inferred edges) rather than failing the run. Point `checkpoint:` at an ULTRA .pth to enable.
  - name: complete
    capability: graph_completion
    provider: graph_completion.ultra
    depends_on: [graph]
    config:
      corpus_id: my-corpus-stack
      scorer: auto
      device: cuda
      target_precision: 0.9
      skip_if_unavailable: true

  # 13) W13 context enrichment: project the assembled KG back onto each chunk as a compact, capped,
  #     confidence-gated context_header (its canonical entities + top-k neighbor triples).
  #     W16 (ADR-0042): its chunk input is the REDACTED induced stream (redact set masked_content),
  #     so the enriched chunks carry induced_labels + context_header AND masked_content — the corpus
  #     row is masked (PII-safe) AND enriched. (With no redact stage, wire this back to `induce`.)
  - name: enrich
    capability: context_enrichment
    provider: context.kg_header
    depends_on: [redact, graph]
    config:
      max_neighbors_per_entity: 4     # hub-entity guard: per entity keep only its top-k relations
      max_triples: 6                  # overall cap on the header's triples
      min_entity_confidence: 0.0
      min_edge_confidence: 0.0

  # 14a) KG export (nodes/edges Parquet + TTL + GraphML), asserted + ULTRA-completed edges.
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph, complete]
    config: {basename: graph}

  # 14b) RAG-corpus export from the REDACTED+ENRICHED chunk stream: masked_content (materialized as
  #      the corpus text, W16) + provenance + classification + induced_labels + context_header + a
  #      dense Granite r2 embedding. context_columns: true → the two W13 Parquet columns.
  #      `bm25` (ADR-0049) adds the INDEX-TIME sparse signal: it materializes bm25-stats.json +
  #      bm25-postings.parquet next to the corpus — the corpus statistics the query-side BM25
  #      rescorer runs on. WITHOUT this block no such files are written at all and the hybrid
  #      fuser's sparse leg is dead (dense-only retrieval). `tokenizer.regex` is the reference
  #      tokenizer (lowercase Unicode alphanumeric runs) the query side applies, so accented and
  #      non-Latin terms match; it is not a word segmenter — an unspaced script (zh/ja/th) should
  #      plug its own segmenter in behind the Tokenizer seam. Deterministic, CPU-only, additive.
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [enrich]
    config:
      basename: records
      context_columns: true
      embedder:
        provider: embedding.sentence_transformers
        config: {model: ibm-granite/granite-embedding-311m-multilingual-r2, dimension: 768, device: cuda, attn_implementation: flash_attention_2}
      bm25:
        provider: tokenizer.regex
        config: {}
```

> **Masked and enriched in one file (W16, ADR-0042).** `redact` is a **chunk→chunk transform**: it
> sets each chunk's `masked_content`, and `enrich` reads THAT redacted chunk stream
> (`redact → enrich → export_corpus`). The corpus Export materializes `masked_content` as the row's
> text, so the exported RAG corpus carries **masked (PII-safe) content AND** `induced_labels` +
> `context_header` + `embedding` — masking and the per-chunk headers **compose in one file**. The
> structured `pii_spans` ride the same `ChunkRecord` (the raw PII value is never stored). With PII
> OFF there is no `redact` Stage, so wire `enrich` back to `induce`.

### 4.1 Fill the placeholders

`latence run` reads the YAML **verbatim** — it does **not** substitute `{CORPUS_DIR}` / `{STORAGE_URI}`.
Materialise a concrete pipeline with `envsubst` (or just edit the two lines by hand):

```bash
export CORPUS_DIR=/workspace/corpus
export STORAGE_URI=file:///workspace/out

# render the two placeholders into a concrete pipeline file
CORPUS_DIR="$CORPUS_DIR" STORAGE_URI="$STORAGE_URI" \
  envsubst '$CORPUS_DIR $STORAGE_URI' < stacks/my-corpus.yaml > /workspace/pipeline.yaml

grep -E 'path:|storage_uri:' /workspace/pipeline.yaml     # sanity-check they're real paths now
```

`storage_uri` is any fsspec URI; `file:///workspace/out` writes everything under `/workspace/out`.
(The placeholder-carrying `stacks/*.yaml` files are for `latence stack validate`, which fills them
with a temp dir + the bundled corpus. This tutorial uses `latence run` on **your** corpus.)

Put your mixed corpus in the source folder first:

```bash
mkdir -p /workspace/corpus
# ... copy your PDFs, scans, .docx/.xlsx/.pptx, .txt, .md into /workspace/corpus ...
ls -la /workspace/corpus
```

`source.local_folder` walks that folder and only admits the file types you list in `extensions`
(anything else is ignored at source; malformed/oversized files are **quarantined** at intake, never
crash the run).

---

## 5. Run each Stage separately + inspect

**There is no `--until` / `--from` / partial-run flag** on `latence run` (confirmed in
`packages/latence-core/src/latence_core/cli.py` — the only options are `--run-id` and `--verbose`).
Stage isolation instead rides the runner's **checkpoint/resume**:

- Every Stage writes a checkpoint at
  `<storage>/_latence/runs/<run_id>/checkpoints/<stage-name>.jsonl` (JSONL of that Stage's output
  records), confirmed in `runner.py` (`_checkpoint_uri`, `run_dir`). Default `run_id` is `run-0001`.
- On re-run, **a Stage whose checkpoint already exists is skipped** and its output re-read from disk;
  a fully-completed pipeline is a no-op.

That gives you two ways to work Stage-by-Stage:

**Method A — progressive stacks (true isolation).** Run a stack that stops at the Stage you care
about, inspect its checkpoint, then run a larger stack that reuses the earlier checkpoints. Keep the
**same `--run-id` and the same `storage_uri`**, and keep every earlier Stage's **name and config
byte-identical** — resume is keyed by **stage name**, so a renamed-or-reconfigured Stage under an
existing checkpoint is **not** re-run (you'd inspect stale output).

**Method B — run once, inspect every checkpoint.** Run the full pipeline (§6) and then read each
Stage's checkpoint. Same files, less re-rendering. Use this unless you specifically want to gate a
Stage before the next one runs.

Either way, the per-Stage checkpoint files are your inspection surface. Set:

```bash
CK=/workspace/out/_latence/runs/run-0001/checkpoints    # <stage>.jsonl lives here
```

A tiny reusable pretty-printer (first N records of any checkpoint):

```bash
peek () { python - "$1" "${2:-3}" <<'PY'
import json, sys
path, n = sys.argv[1], int(sys.argv[2])
for i, line in enumerate(open(path)):
    if i >= n: break
    print(json.dumps(json.loads(line), indent=2, ensure_ascii=False)[:1600]); print("---")
PY
}
```

Run the progressive stack up to a Stage (Method A), e.g. to stop after `parse`:

```bash
# make a source→parse-only copy, render it, run it (same run-id + storage as the full run)
sed '/- name: chunk/,$d' stacks/my-corpus.yaml > stacks/upto-parse.yaml
CORPUS_DIR=$CORPUS_DIR STORAGE_URI=$STORAGE_URI envsubst '$CORPUS_DIR $STORAGE_URI' \
  < stacks/upto-parse.yaml > /workspace/upto-parse.yaml
latence run /workspace/upto-parse.yaml --run-id run-0001
```

Then inspect and move on. Per-Stage: **what lands, and what to look for.**

### 5.1 `source` → `source.jsonl` (raw `ParserInput`s)
```bash
peek $CK/source.jsonl
# look for: one record per admitted file; correct file_name / file_type; sensitivity=internal.
python -c "print(sum(1 for _ in open('$CK/source.jsonl')),'files admitted')"
```

### 5.2 `intake_screen` → `intake_screen.jsonl` (the PASSED files)
Quarantined files are **not** here — they're in the sidecar. Check both:
```bash
peek $CK/intake_screen.jsonl
ls  /workspace/out/_latence/runs/run-0001/quarantine/    # withheld files (oversized/malformed)
ls  /workspace/out/_latence/runs/run-0001/screening/     # per-Stage screening findings
```

### 5.3 `parse` → `parse.jsonl` (`DocumentRecord`s — OCR'd markdown)
This is the served-vLLM OCR payoff — each page was POSTed to your vLLM endpoint and assembled onto
an exact-offset page map. If this Stage is empty or errors, the endpoint is the first suspect (§9).
```bash
peek $CK/parse.jsonl
# look for: clean, markup-stripped markdown `content`; a `page_map`; provenance file_name/type/size intact.
python - <<PY
import json
for line in open("$CK/parse.jsonl"):
    d=json.loads(line); print(d["provenance"]["file_name"], "→", len(d.get("content","")), "chars")
PY
```

### 5.4 `chunk` → `chunk.jsonl` (`ChunkRecord`s — sizes)
```bash
peek $CK/chunk.jsonl
# look for: token_count clustered at/under 640; index runs 0..k per document; content is coherent.
python - <<PY
import json, statistics as st
tc=[json.loads(l)["token_count"] for l in open("$CK/chunk.jsonl")]
print("chunks",len(tc),"| token_count min/median/max",min(tc),int(st.median(tc)),max(tc))
PY
```

### 5.5 `induce` → `induce.jsonl` (chunks stamped with the DISCOVERED schema)
This is the payoff of schema induction — the domain labels the LLM found for **your** corpus. In
`granularity: chunk` each chunk carries its OWN `induced_labels`, so you should see MANY distinct
schemas (one per distinct chunk text); a single distinct schema would mean corpus mode:
```bash
python - <<PY
import json
seen=set()
for l in open("$CK/induce.jsonl"):
    il=json.loads(l).get("induced_labels")
    if il: seen.add(json.dumps({k:il[k] for k in ("entity_types","relation_types","pii_types")}, sort_keys=True))
print("distinct induced schemas:", len(seen))          # chunk mode → many; corpus mode → 1
for s in list(seen)[:1]: print(json.dumps(json.loads(s), indent=2))
PY
# look for: sensible domain types (e.g. bank, law, payment_product / owns, regulated_by / iban, tax_id),
# NOT generic English. Empty/None everywhere ⇒ the key/endpoint failed OR the model returned nothing
# usable — the loud signal for the latter is induction_empty_after_sanitization in the Quality Report (§6).
```

> **Empty-recovery telemetry — a quality signal to check.** When the LLM answers but every induced
> label sanitises away to nothing, the Stage does NOT silently pretend it induced a schema — it falls
> back to config labels **loudly**, bumping `induction_empty_after_sanitization` (a WARNING in the
> log, a counter in the Quality Report). A non-zero count means the induction call reached the model
> but produced no usable labels for those documents (a prompt/model mismatch), distinct from a
> dead-endpoint fallback. Watch it alongside `induction_fallbacks` — both non-zero and generic
> `induced_labels` everywhere point straight at the LLM endpoint/model.

### 5.6 `extract` → `extract.jsonl` (typed entities + relations, one fused stream)
A **mixed** checkpoint: `EntityMention`s and `RelationMention`s interleaved (a relation line has a
`head_mention_id`). Check the **types** and that relations are meaningful:
```bash
python - <<PY
import json
ents=rels=0
for l in open("$CK/extract.jsonl"):
    d=json.loads(l)
    if "head_mention_id" in d:
        rels+=1
        if rels<=8: print(f"REL  {d['head_text']!r} —{d['label']}→ {d['tail_text']!r}  ({d['confidence']:.2f})")
    else:
        ents+=1
        if ents<=8: print(f"ENT  {d['text']!r}  [{d['label']}]  ({d['confidence']:.2f})")
print("---", ents, "mentions,", rels, "relations")
PY
# look for: entities typed by the DOMAIN schema (bank/law/payment_product), not "pronoun"/"product";
# relations that actually read true (head → label → tail). Noise here ⇒ revisit threshold/induce.
```

### 5.7 `redact` → `redact.jsonl` (`ChunkRecord`s — `masked_content` + `pii_spans`, W16)
```bash
python - <<PY
import json
for l in open("$CK/redact.jsonl"):
    d=json.loads(l)
    spans=d.get("pii_spans",[])
    print(d["provenance"]["file_name"], "→", len(spans), "PII spans:",
          sorted({s["pii_type"] for s in spans}))
PY
# W16 (ADR-0042): redact is chunk→chunk — each ChunkRecord gains masked_content (the spans replaced)
# + pii_spans, so enrich/export carry the masked text forward. look for: financial types firing
# (iban, credit card number, tax id) — not just person/email/phone. The raw PII value is NEVER stored.
```

### 5.8 `disambiguate` → `disambiguate.jsonl` (canonical entities + the merge audit)
This is the GLinker payoff: `ENTITY`-scope records each carry a `CanonicalEntity` (`canonical_text`,
`entity_type`, `member_mention_ids`, `kb_id`) **and** the `merge_decisions` audit trail that built
it. Read the canonical entities and the audited merges:
```bash
python - <<PY
import json
ents=0
for l in open("$CK/disambiguate.jsonl"):
    d=json.loads(l)
    if d.get("scope")!="entity": continue      # skip RELATION-scope records
    e=d["entity"]; ents+=1
    if ents<=12:
        kb = f"  kb={e['kb_id']}" if e.get("kb_id") else ""
        print(f"{e['canonical_text']!r} [{e['entity_type']}]  ×{len(e['member_mention_ids'])} mentions{kb}")
        for m in e.get("merge_decisions", []):
            print(f"    merged {m['source_name']!r}+{m['target_name']!r}  via {m['reason']}  ({m['confidence']:.2f})")
print("---", ents, "canonical entities")
PY
```

**What "good" looks like** for the GLinker ER (the precision-first posture, ADR-0046):

- **Distinct persons stay distinct.** Two different people with similar names are NOT collapsed —
  the audited resolver's embedding-by-label rung is OFF by default precisely because retrieval
  embedders over-merge short names (they put distinct names within 0.85–0.89 cosine).
- **OCR / spelling variants merge.** The conservative fuzzy edit-distance rung folds same-label
  variants the string cascade cannot — e.g. `Kai Kusch` ↔ `Kai Kölsch` (an OCR variant) merges,
  audited as `fuzzy_edit_distance`.
- **The distinguishing-token guard keeps look-alikes apart.** `§3 GwG` ≠ `§11 GwG` (a numeric-token
  difference) and `K² Ventures UG` ≠ `D² Ventures UG` (a distinct leading short-code) are NOT
  merged even though their char-similarity is high — the guard, not the threshold, vetoes them.
- **Neural linking resolves mentions in context.** IBM-in-a-hardware-context vs a different org in a
  similar context are linked to the right canonical entry by the gliner-linker biencoder + reranker,
  not by surface string alone.

Every merge is auditable (`merge_decisions` here; the corpus totals `merges_applied` /
`merges_below_policy` / `merges_by_strategy` in the Quality Report, §6) — **no silent over-merge**.

### 5.9 `graph` → `graph.jsonl` (`GraphRecord`s — nodes + edges)
```bash
python - <<PY
import json
nodes=edges=0
for l in open("$CK/graph.jsonl"):
    d=json.loads(l)
    if d.get("scope")=="node" and d.get("node"):
        nodes+=1
        if nodes<=6: print("NODE", d["node"]["canonical_name"], "[",d["node"]["label"],"]")
    elif d.get("scope")=="edge" and d.get("edge"):
        edges+=1
        if edges<=6: print("EDGE", d["edge"]["source_node_id"][:8],"-",d["edge"]["label"],"->",d["edge"]["target_node_id"][:8])
print("---", nodes, "nodes,", edges, "edges")
PY
```

### 5.10 `complete` → `complete.jsonl` (ULTRA-inferred edges — or a graceful skip)
```bash
# With no vendored ULTRA checkpoint the Stage skips-with-flag → this file is absent/empty (expected).
[ -s $CK/complete.jsonl ] && peek $CK/complete.jsonl || echo "ULTRA skipped (skip_if_unavailable) — no inferred edges; check quality-report.md 'skipped'."
```

### 5.11 `enrich` → `enrich.jsonl` (chunks + `context_header`)
```bash
python - <<PY
import json
withhdr=0
for l in open("$CK/enrich.jsonl"):
    ch=json.loads(l).get("context_header")
    if ch:
        withhdr+=1
        if withhdr<=3:
            print("entities:", ch["entities"])
            print("triples :", ch["neighbor_triples"])
            print("neighbor_node_ids:", ch["neighbor_node_ids"]); print("---")
print(withhdr, "chunks got a context header")
PY
# look for: each header = the chunk's canonical entities + a few "A —label→ B" triples, capped
# (≤ max_triples). A chunk with no gated entity has context_header=null (that's fine).
```

---

## 6. Run end-to-end

One command runs the whole DAG (skipping any Stage whose checkpoint already exists from §5):

```bash
export PYTORCH_JIT=0
export HF_HUB_DISABLE_XET=1
export OPENROUTER_API_KEY="sk-or-..."     # (already set from §1.3)  # pragma: allowlist secret
# vLLM OCR server from §2 must be up (curl http://localhost:8000/v1/models)
latence run /workspace/pipeline.yaml --run-id run-0001 --verbose
```

`latence run` prints the Quality Report as JSON to stdout and writes both renderings to Storage.
Read the human-readable one:

```bash
RUN=/workspace/out/_latence/runs/run-0001
cat $RUN/quality-report.md
```

What to check in the report:

- **`contracts_complete: true`** — every record carried valid Provenance + Classification end to end.
- **Per-Stage `records_in` / `records_out`** — counts flow sensibly (files → docs → chunks →
  mentions → nodes/edges); a Stage with `records_out: 0` that you expected to produce data is the
  first thing to investigate.
- **Disambiguation merge audit** — `merges_applied` / `merges_below_policy` (folded vs recorded-but-
  not-applied, the no-silent-over-merge audit), `merges_by_strategy` (which resolver rung), and
  `entities_linked` / `entities_unlinked` (matched to the external KB vs graceful fallback).
- **KG stats** — `nodes` / `edges` non-zero, `evidence_coverage` near 1.0 (every edge cites evidence).
- **No-PII-leak** — the corpus RAG scan reports clean (with PII ON the exported text is masked).
- **`induction_empty_after_sanitization` / `induction_fallbacks`** — both should be 0 on a healthy
  run; non-zero means the LLM produced no usable labels / the endpoint misbehaved (§5.5, §9).
- **`skipped` / `skip_reason`** — on a CPU-only host the `cuda` Stages are honestly
  **skipped-with-flag** (never faked); `complete` (ULTRA) legitimately skips when no checkpoint is
  vendored. On a real GPU pod the learned Stages should run, not skip.

The machine-readable `quality-report.json` carries the same, per-Stage, for gating.

---

## 7. Discover the final output

Everything lands under `export/`:

```bash
ls -la /workspace/out/_latence/runs/run-0001/export/
#   records.jsonl        records.parquet          ← RAG corpus
#   bm25-stats.json      bm25-postings.parquet    ← sparse (BM25) index-time signal
#   graph-nodes.parquet  graph-edges.parquet
#   graph.ttl            graph.graphml            ← knowledge graph
EX=/workspace/out/_latence/runs/run-0001/export
```

### 7.1 The RAG corpus — `records.jsonl` (full record) / `records.parquet` (flat columns)

The JSONL carries the **full** `ChunkRecord` dump (every field, nested objects intact). Walk one row:

```bash
peek $EX/records.jsonl 1
```

Field-by-field (source: `contracts.py` `ChunkRecord`/`Record`; full table in
`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md` §2.1):

- **`content`** — the retrieval text (markup-stripped). W16 (ADR-0042): with the `redact` Stage on,
  the Export materializes each chunk's **`masked_content`** here, so `content` is the **PII-safe
  (masked)** text — never the raw variant.
- **`provenance`** — `source_uri`, `file_name`, `file_type`, `file_size`, `document_id`, and the
  offsets `page_start/page_end` + `char_start/char_end` locating the chunk in its source. Your audit
  trail back to the exact page/character.
- **`classification`** — `language`, `category`, `sensitivity`.
- **`induced_labels`** — the LLM-discovered `{entity_types, relation_types, pii_types}` for the corpus
  (the schema induction payoff), plus the `model_id` that induced it.
- **`context_header`** — the KG projected onto this chunk: `entities` + aligned `entity_types`,
  `neighbor_triples` (`A —label→ B`, capped), `kg_node_ids` (the chunk's own entities) and
  `neighbor_node_ids` (query-time KG-expansion join keys). `null` for a chunk with no gated entity.
- **`embedding`** — the dense **Granite r2 311m** vector (dimension 768, bf16 + FA2). It was computed
  over `rendered_context_header + "\n" + content`, so the vector carries the KG coherence — and,
  under W16, over the **PII-safe (masked)** corpus text, consistent with the stored `content`.
- **`record_id`** (primary key), `document_record_id` (join key), `index`, `token_count`,
  `page_map`/`offset_map`, `risk_markers`.

The **Parquet** is a flat, retrieval-DB-friendly projection (nested fields as JSON strings). With
`context_columns: true` it has: `record_id, schema_version, provenance, classification, content,
media_type, embedding` **plus** `context_header` (JSON string) and `neighbor_entity_ids`
(`list<string>` — the KG-expansion keys). Read it:

```bash
python - <<PY
import pyarrow.parquet as pq
t = pq.read_table("$EX/records.parquet")
print("rows:", t.num_rows, "| columns:", t.column_names)
row0 = {k: t.column(k)[0].as_py() for k in t.column_names}
print("embedding dim:", len(row0.get("embedding") or []))
print("neighbor_entity_ids[0]:", row0.get("neighbor_entity_ids"))
PY
```

> If `context_header` / `neighbor_entity_ids` columns are missing, you forgot `context_columns: true`
> on `export_corpus` (the JSONL always carries `context_header` via the record dump; the two extra
> **Parquet** columns are opt-in). If `embedding` is missing, no `embedder:` was wired.

### 7.2 The knowledge graph

```bash
# nodes: node_id, type (entity type), canonical_name, confidence, source_document_ids
# + centrality (PageRank) and community (label-propagation) when `graph_features` is wired (§4 §11)
python -c "import pyarrow.parquet as pq; t=pq.read_table('$EX/graph-nodes.parquet'); print(t.column_names); print(t.to_pylist()[:5])"
# edges: edge_id, source_node_id, target_node_id, label, confidence (+ evidence)
python -c "import pyarrow.parquet as pq; t=pq.read_table('$EX/graph-edges.parquet'); print(t.column_names); print(t.to_pylist()[:5])"

head -30 $EX/graph.ttl          # RDF Turtle: each node a subject (rdfs:label, lg:entityType, lg:confidence)
head -20 $EX/graph.graphml      # XML for yEd / Gephi / networkx
```

Load the graph in networkx if you want to explore it:

```bash
python -c "import networkx as nx; g=nx.read_graphml('$EX/graph.graphml'); print(g.number_of_nodes(),'nodes',g.number_of_edges(),'edges')"
```

### 7.3 The Quality Report

`quality-report.md` (human) + `quality-report.json` (machine) — the per-run scorecard from §6.

**Full field reference for downstream consumers:** [`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`](RETRIEVAL-ENGINE-INPUT-SPEC.md).

---

## 8. Incremental updates: add / update / retract / purge

Your source folder changes after the first run — new files arrive, some get edited, some must be
deleted (a customer exercises their right-to-be-forgotten). You do **not** re-run everything and you
do **not** write a Python script. Three CLI commands drive the corpus forward, each committing the
next **Corpus Version** (immutable, inspectable, rollback-able). They take the **same stack YAML**
`latence run` takes.

> **Extraction is incremental too (W17, ADR-0043).** A delta's **extraction compute** is
> O(changed), not O(corpus): each **unchanged** document (matched by content-addressed
> `document_id`) **skips its doc-level Stages** (parse/chunk/extract/redact) and **reuses its
> committed records**, and only new/changed documents are extracted — so only their chunks are
> **induced**, too. The corpus-level Stages (disambiguate/graph/…) then run over the UNION of
> reused + freshly-extracted records. The churn's `extraction: N extracted, M reused` line is the
> witness.
>
> **How a delta stays consistent with the base run (T4).** Labels are induced *per chunk*, so there
> is no single corpus schema to freeze. What the corpus carries forward instead is its **canonical
> type vocabulary**: the delta's freshly-induced types are canonicalized *against* the committed
> one, so a drifted surface (`org` where the corpus says `organization`) **aliases into the existing
> type** and a genuinely new type **appends** a new one. Existing canonical type names never change,
> so nothing already in your knowledge graph is retyped, and the churn's
> `canonical types: +N new, +M aliased into existing` line shows you exactly what grew.
> The one escape hatch is `--reinduce`: the **full rebuild** — re-induce every chunk and
> re-canonicalize the type vocabulary from scratch, re-extracting every document. It is the only
> path that may re-elect a canonical type label, which is why it is not the default.

### 8.1 `latence delta` — add / update (the source folder changed)

Drop new or edited files into the source folder, then:

```bash
# Each delta needs a FRESH run id; omit --run-id and it auto-generates a UTC-timestamped one
# (delta-YYYYMMDDTHHMMSSZ). Pin one with --run-id if you want a stable name.
latence delta stacks/my-corpus.yaml
# or, explicit:
latence delta stacks/my-corpus.yaml --run-id delta-2026-07-13
# full rebuild: re-induce every chunk AND re-canonicalize the type vocabulary from scratch
# (lets the vocabulary move with the corpus — canonical type names may be re-elected):
latence delta stacks/my-corpus.yaml --reinduce
```

It content-hashes the current folder against the last committed Version, detects added / changed /
removed files, and commits Version N+1. The churn it prints:

```text
Corpus Version v3 -> v4 (incremental)
  documents: +2 ~0 -1 (=118 unchanged)
  entities:  +5 created, 1 merged, 0 split
  edges:     +7 -1
  records:   0 retracted, 0 purged (retraction)
  affected set: 6/431 corpus-level records recomputed
  extraction: 2 extracted, 118 reused (unchanged documents' doc-level Stages skipped — issue #42 / ADR-0043)
  drift: 0.024
```

`affected set: 6/431` is the stored-corpus payoff (only 6 of 431 corpus-level records re-resolved);
`extraction: 2 extracted, 118 reused` is the **compute** payoff (only the 2 changed documents were
re-parsed/-chunked/-extracted, the other 118 reused verbatim). (An edit under the content-addressed
local-folder Source shows up as `+1 ~1`-style *add + delete*, not `~updated`: editing a file mints a
new content-id and retires the old one — see ADR-0029 §4.)

Re-running a delta over an **unchanged** folder is a **no-op** — the fingerprint matches head, so no
new Version is minted (idempotent). Deltas are **deterministic**: the same source + config reproduces
the same Corpus Version fingerprint.

### 8.2 Where to get a `document_id`

`retract` and `purge` take content-addressed document ids (`sha256:<64 hex>`). Two ways to get one:

```bash
# (a) From the export — every record carries its source document id:
python -c "import json; print({json.loads(l)['provenance']['document_id'] for l in open('$EX/records.jsonl')})"

# (b) Hash the file directly (the id IS the sha256 of the file bytes):
printf 'sha256:%s\n' "$(shasum -a 256 /workspace/corpus/contract-4471.pdf | cut -d' ' -f1)"
```

### 8.3 `latence retract` — soft delete (auditable, rollback-able — the default)

Exclude a document's derived records from the **live** corpus, but **retain** them in the prior
Corpus Version for audit and rollback. Any entity cluster that was merged *solely* because of the
retracted document splits back apart.

```bash
# One id, or several (--doc is repeatable); or a file of ids, one per line.
latence retract stacks/my-corpus.yaml --doc sha256:1a2b...ef
latence retract stacks/my-corpus.yaml --doc sha256:1a2b...ef --doc sha256:9c8d...01
latence retract stacks/my-corpus.yaml --docs-file to-remove.txt
```

The retracted records still live in the version that preceded the retraction — inspect it any time
(`CorpusStore.read_records(<version>)`), or roll forward by re-adding the file with `latence delta`.

### 8.4 `latence purge` — hard delete (GDPR / right-to-be-forgotten)

The irreversible one. Unlike a retraction, **nothing is retained**: the on-disk source file, every
derived record across **all** Corpus Versions (and any abandoned WAL residue), and the run artifacts
are physically erased. Because it is destructive and non-recoverable, `purge` prompts for
confirmation; pass `--yes` to skip it in a script.

```bash
# Prompts "This cannot be undone. Continue?" unless --yes is given.
latence purge stacks/my-corpus.yaml --doc sha256:1a2b...ef
latence purge stacks/my-corpus.yaml --docs-file gdpr-erasures.txt --yes
```

**Retraction vs Purge in one line:** *Retraction* is a soft, auditable, rollback-able tombstone — the
records survive in prior Versions for compliance; *Purge* is a hard GDPR erasure — the records are
physically removed everywhere and cannot be recovered. Reach for retraction by default; reserve purge
for a genuine right-to-be-forgotten request.

Every command re-persists the run's `quality-report.json` + `quality-report.md` with the delta churn
under `_latence/runs/<run-id>/`, exactly like `latence run`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `parse` produces 0 docs / `openai`-client connection errors | the vLLM OCR endpoint isn't up, or `base_url`/`model` don't match the server | stand up the server (§2); `curl http://localhost:8000/v1/models` and match `parse.config.base_url` + `model` to it; `api_key: EMPTY` is fine. |
| `FlashInfer requires sm75+` when starting vLLM on Blackwell | FlashInfer's capability check misparses `sm_120` as `12` and rejects a capable device | `export VLLM_USE_FLASHINFER_SAMPLER=0` before the serve — disables only the *sampler*, cudagraphs stay ON (§2). **Not** `--enforce-eager`: it clears the abort by killing cudagraphs and flooring throughput. |
| Model downloads hang / error on `hf_xet` (Blackwell pod) | the Hugging Face XET transfer backend is flaky on some SM 12.x pods | `export HF_HUB_DISABLE_XET=1` before downloading (§1.3). |
| `nvrtc: failed to open libnvrtc-builtins.so.13.0` (GLiNER2/DeBERTa crash) | torch JIT on SM 12.x / cu13x | `export PYTORCH_JIT=0` before running (§1.3). |
| Extraction worker pinned ~290% CPU, GPU idle, no progress | `compile: true` on GLiNER2 — `dynamic=True` recompiles per distinct (label-count, batch) shape | T7 removed the cause (label-set bucketing + micro-batching pin both dimensions), so compile is on by default again; if you still see it, pin `compile: false` and report the chunk-length/label-set spread (§1.3). |
| `error: externally-managed-environment` on `pip install` | RunPod PEP-668 system Python | add `--break-system-packages` (§1.2), or use a venv. |
| `ModuleNotFoundError: peft` / `glinker` / `flashdeberta` / `gliner` at run time | a `--no-deps` editable install skipped a hidden dep | reinstall the package **without** `--no-deps` and `pip install ...` the runtime deps (§1.2). |
| `disambiguation.glinker` prints `MISSING` at verify (§1.4) | the separate `latence-disambig-glinker` heavy package wasn't installed | `pip install -e packages/latence-disambig-glinker` + `pip install glinker flashdeberta` (ADR-0016). |
| Every chunk's `induced_labels` is `null`; high `induction_fallbacks` | no/invalid induction key, wrong `base_url`, or unreachable endpoint (the Stage **fails open** to config labels — never crashes) | set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`); check `base_url`; run `--verbose` and read the `schema_induction fell back to config labels: ...` log lines. |
| `induction_empty_after_sanitization` non-zero | the LLM answered but every induced label sanitised away (prompt/model mismatch) | try a more capable induction model; the run still completes on config labels, but the discovered schema is empty (§5.5). |
| GLinker over-merges distinct entities / two people collapse into one | the embedding-by-label rung was turned on with the retrieval embedder | keep `use_embedding_merge` at its **default OFF** — recall is carried by the neural linker + string cascade + the conservative fuzzy rung; re-enable ONLY with a discriminative STS embedder (§5.8). |
| Look-alikes wrongly merged (`§3 GwG` with `§11 GwG`, `K²` with `D²`) | should not happen — the distinguishing-token guard vetoes these | if it does, you overrode the guard; leave the GLinker precision defaults to ride (ADR-0046, §5.8). |
| `complete` (ULTRA) reports `skipped` | no ULTRA checkpoint is vendored; `skip_if_unavailable: true` skips gracefully | expected — supply an ULTRA `.pth` via `complete.config.checkpoint` to enable inferred edges. |
| Tail entities/relations missing; offsets look off on long chunks | chunk tokenised past the mdeberta **768** window and was silently truncated | keep `chunk.markdown max_tokens ≤ 640` (~≤768 mdeberta-tokens); the provider `max_len: 768` guard is the backstop (§4 callout). |
| GPU Stages report `skipped` / `skip_reason` in the report | CPU-only host (device honesty — skipped-with-flag, never faked) | run on a real CUDA pod with a vLLM OCR endpoint up; the CPU-reachable spine still validates. |
| `context_header` / `neighbor_entity_ids` absent from `records.parquet` | `context_columns` not set on `export_corpus`, or the corpus was exported from `redact` not `enrich` | set `context_columns: true` and wire `export_corpus → enrich` (§4). |
| A re-run didn't pick up a config change to an early Stage | resume is keyed by **stage name**; an existing checkpoint under the same name is skipped | delete that Stage's checkpoint (`rm $CK/<stage>.jsonl`) or use a fresh `--run-id`. |
| `latence delta` committed no new Version | the source folder is unchanged — a delta over an identical corpus is an idempotent no-op | change the folder before the delta; nothing to commit is correct behaviour (§8.1). |
| `retract`/`purge` says the id `does not look like a document id` | the id isn't `sha256:<64 hex>` | get a real `document_id` from `records.jsonl` (`provenance.document_id`) or by hashing the file (§8.2). |
