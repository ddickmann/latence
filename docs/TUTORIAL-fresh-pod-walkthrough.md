# Tutorial — fresh RunPod pod → AI-ready data (mixed-file corpus, stage by stage, then end-to-end)

**Branch:** `w13-context-enrichment` (the only branch that carries the full stack — schema induction,
GLiNER2 fused extraction, GLiNER2 PII redaction, KG assembly **and** the W13 `context_enrichment`
Stage). Worktree used while writing this: `/Users/dennisdickmann/Developer/latence-framework-w13`.

This walkthrough takes you from a **bare GPU pod with a folder of mixed file types** to:

1. a working install,
2. running and **inspecting each Stage in isolation**,
3. running the **whole pipeline end-to-end**, and
4. **discovering and understanding the final output**.

Every command is copy-pasteable. Where a claim comes from a real pod run it is stated as a warning.

---

## 0. Prereqs — what the pipeline produces

Point the pipeline at a folder of messy enterprise files (PDF, scans, Office docs, text, markdown)
and it emits, per corpus run, three deliverables as **plain files** (JSONL / Parquet / RDF / JSON) on
any fsspec backend — never a live database (ADR-0017):

- **A chunk-level RAG corpus** — `records.jsonl` + `records.parquet`: retrieval text + full
  `Provenance` (source + offsets) + `Classification` + per-chunk `induced_labels` + a KG
  `context_header` + an optional dense `embedding` vector.
- **An evidence-linked knowledge graph** — `graph-nodes.parquet`, `graph-edges.parquet`,
  `graph.ttl`, `graph.graphml`: typed entities + relations, every edge citing the mentions/documents
  that justify it.
- **A Quality Report** — `quality-report.json` + `quality-report.md`: per-Stage counts, KG
  statistics, contract-completeness, no-PII-leak check.

The full field reference for consumers lives in [`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`](RETRIEVAL-ENGINE-INPUT-SPEC.md).

---

## 1. Set up the fresh pod

### 1.1 Clone the framework (the `w13` branch)

```bash
cd /workspace
git clone <your-remote-url> latence-framework
cd latence-framework
git checkout w13-context-enrichment
```

### 1.2 Install the packages you need — editable, **letting deps resolve**

RunPod's system Python is **PEP-668 "externally managed"**, so every `pip install` needs
`--break-system-packages` (or a venv — but the pod's global torch/CUDA is usually why you're on a
pod, so we install globally here).

> **Warning — do NOT use `--no-deps` for the editable installs.** `latence-pii-gliner2` /
> `latence-extract-gliner2` declare `pdfplumber`, `pyarrow`, and — critically — **`peft`** (a real
> hidden GLiNER2 dependency) transitively. A `--no-deps` editable install silently skips them and the
> Stage then crashes at runtime. Install each package **normally** so its deps resolve:

```bash
cd /workspace/latence-framework

# core + the providers the GOOD stack uses (deps resolve normally)
pip install --break-system-packages \
  -e packages/latence-core \
  -e packages/latence-parser-pdfplumber \
  -e packages/latence-parser-render \
  -e packages/latence-parser-lighton \
  -e packages/latence-schema-inducer \
  -e packages/latence-extract-gliner2 \
  -e packages/latence-pii-gliner2 \
  -e packages/latence-disambig-embedding \
  -e packages/latence-embedder-st
```

Then the runtime deps that the learned/parse/export Stages import lazily (some are declared, but a
mixed corpus needs the whole set — install them explicitly so nothing is missing at run time):

```bash
pip install --break-system-packages \
  gliner gliner2 sentence-transformers peft accelerate \
  pdfplumber pypdfium2 python-docx openpyxl \
  pyarrow fsspec networkx rdflib openai
```

- `gliner gliner2 peft accelerate` — the GLiNER2 fused extractor + PII redactor (mdeberta-v3 backbone).
- `sentence-transformers` — the embedding disambiguator **and** the corpus-export embedder (Granite r2 311m, bf16+FA2).
- `pdfplumber` — fast text-layer PDF parsing; `pypdfium2 python-docx openpyxl` — the universal
  `parser.render` path (PDF→image, Office→text/render).
- `pyarrow fsspec` — Parquet export + storage; `networkx rdflib` — the GraphML/TTL KG export.
- `openai` — the schema-induction LLM client (OpenAI-compatible; used for OpenRouter/vLLM too).

### 1.3 Required env — JIT off (Blackwell/torch-cu13 caveat) and the induction key

> **Warning — `nvrtc: failed to open libnvrtc-builtins.so.13.0`.** On recent NVIDIA (SM 12.x /
> Blackwell) with torch 2.11-cu13x, GLiNER2 / DeBERTa can crash inside a JIT-compiled kernel. Disable
> the JIT **before running anything**:

```bash
export PYTORCH_JIT=0
```

The schema-induction Stage calls an OpenAI-compatible LLM. Provide the key **as an environment
variable only — never in a file** (an inlined `api_key` would be scrubbed from `manifest.json`, but
don't rely on that; keep keys out of YAML):

```bash
export OPENROUTER_API_KEY="sk-or-..."     # OpenRouter (default in the stack below)  # pragma: allowlist secret
# or, for direct OpenAI:
# export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

Model default is `openai/gpt-4.1-mini` reached via `base_url: https://openrouter.ai/api/v1`. Set the
base URL to `http://<vllm-host>:8000/v1` for an on-prem vLLM (any placeholder key), or leave it
`null` for direct OpenAI with `$OPENAI_API_KEY`.

### 1.4 Verify the install

```bash
# the CLI is on PATH once latence-core is installed
latence --help                       # or: python -m latence_core.cli --help

# imports resolve (no output == good)
python -c "import gliner2, sentence_transformers, peft, pdfplumber, pyarrow, networkx, rdflib, openai; print('deps OK')"

# every Stage provider is discoverable (should print all the names the stack uses)
python - <<'PY'
from latence_core.capability import ProviderRegistry
names = set(ProviderRegistry().names())
for n in ("source.local_folder","parser.render","parser.pdfplumber","chunk.markdown",
          "label_inducer.llm","fused_entity_relation.gliner2","redaction.gliner2",
          "disambiguation.embedding","graph.canonical","context.kg_header",
          "export.jsonl_parquet","export.knowledge_graph"):
    print(("ok " if n in names else "MISSING "), n)
PY
```

> `ProviderRegistry().names()` lists every discovered provider; equivalently
> `python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='latence.providers')])"`.

---

## Guided setup: `latence setup` → `latence process`

The rest of this tutorial (§2–§5) hand-authors the GOOD stack so you understand every knob. If you
just want the **same quality-baked stack generated for you**, two commands do it — a guided,
deterministic wizard that converges to a clean, validated config, then one command to run it:

```bash
# Interactive: each prompt has a sensible default + a one-line "why it matters".
latence setup
# → writes ./latence.stack.yaml (validated), then prints "Run `latence process` to start."
latence process
```

**What the wizard asks** (each with a default; press Enter to accept):

| Prompt | Default | Why it matters |
|---|---|---|
| **Source folder** | `file://$PWD/corpus` | The folder Latence reads. Validated to exist; its file types are **auto-detected** and reported. |
| **Profile** | `lite` | **The tier.** `lite` (CPU-first, laptop-friendly — the stack in this table). `enterprise-sota` generates the pod-validated composed stack instead (see the box below). |
| **Output / storage URI** | `file://$PWD/latence-out` | Where state + the exported RAG corpus and KG are written. |
| **Parser** | recommended from the detected types | `parser.render` (universal / mixed), `parser.pdfplumber` (born-digital PDF text), or `parser.lighton` (OCR for scans/images). Confirm or override. *(enterprise-sota forces the served-vLLM `parser.lighton_vllm` and skips this prompt.)* |
| **Device** | `auto` | `auto` picks CUDA when present, else CPU (skip-with-flag). *(enterprise-sota pins `cuda`.)* |
| **Schema-induction LLM** | OpenRouter + `openai/gpt-4.1-mini` | The quality lever — a small LLM discovers your corpus's own `{entity, relation, pii}` schema. The **API key is read from the environment at run** (`$OPENROUTER_API_KEY`, or `$OPENAI_API_KEY` for a direct-OpenAI base URL) — **never stored** in the file. The wizard warns if it is unset. |
| **PII redaction + guardrail screening** | ON | **The one optional toggle.** ON adds the `content_screen` + `redact` stages (learned **financial-PII** detection). OFF omits exactly those two stages. |
| **Mask before extraction** (`--extract-on`) | `unmasked` | **Advanced (W18).** `unmasked` (default) — extraction reads the real text, the KG keeps entities. `masked` — extraction reads the REDACTED chunks so PII never reaches the KG (stricter posture; requires PII ON). |
| **Domain / description hint** | (empty) | Free text, recorded as a header comment for readers. |

The wizard then **generates the quality-baked spine**, **dry-validates it** (structural + DAG +
capability wiring — the SAME machinery `latence stack validate` builds on, no pipeline run), and
**saves** `./latence.stack.yaml`. The generated stack is exactly the GOOD stack of §3
(`chunk[640]` → `induce[corpus]` → `extract[gliner2, thr 0.5]` → optional `redact[gliner2,
financial-PII]` → `disambiguate` → `graph` → `enrich(context)` → `export_corpus[context_columns]` +
`export_kg`).

**Deterministic + scriptable (CI/repro).** `--non-interactive` takes every answer from flags, and
the **same inputs produce a byte-identical config**:

```bash
latence setup --non-interactive \
  --source file:///workspace/corpus \
  --storage file:///workspace/latence-out \
  --parser parser.render --device cuda \
  --llm-base-url https://openrouter.ai/api/v1 --llm-model openai/gpt-4.1-mini \
  --pii --domain "retail banking contracts" \
  --output ./latence.stack.yaml
latence process --config ./latence.stack.yaml --run-id run-0001
```

> **The `enterprise-sota` profile (ADR-0046, `stacks/gpu-sota-glinker.yaml`).** Pass
> `--profile enterprise-sota` (or pick it at the profile prompt) and the wizard emits the
> **pod-validated composed stack** instead of the lite path — the SOTA half of every seam, wired and
> dry-validated for you:
>
> - **parse → `parser.lighton_vllm`** — the served-vLLM LightOn OCR endpoint (the production OCR
>   path, ~2.7× the in-process parser). The wizard prompts for the endpoint **`base_url`**
>   (default `http://localhost:8000/v1`) and **`model`** (default `lightonocr`); `api_key` defaults
>   to `EMPTY` (a local vLLM accepts any placeholder) and `max_concurrency` to `8`.
> - **induce → `label_inducer.llm`** at **corpus** granularity (`max_labels 40`, `temperature 0`) —
>   the same schema-induction quality lever (its `base_url`/`model`/key are the LLM prompts above).
> - **extract → `fused_entity_relation.gliner2`** (`threshold 0.3`, labels floor
>   `[person, organization]`) and **redact → `redaction.gliner_pii`** (chunk-level, financial-PII
>   floor — the same W16 `redact → enrich → export_corpus` dataflow).
> - **disambiguate → `disambiguation.glinker`** — the neural GLinker entity-linker + audited resolver
>   (`context_window 200`, `threshold 0.3`, reranker + FlashDeBERTa on) with the **IBM Granite r2**
>   embedder. The precision-by-default knobs (embedding-merge OFF, fuzzy + distinguishing-token guard
>   ON) ride the provider defaults — the config stays minimal and the ER is precise out of the box.
> - **graph → `graph.canonical` → complete → `graph_completion.ultra`** (`skip_if_unavailable: true`
>   — no vendored checkpoint means a graceful skip, never a failed run) → **`export.knowledge_graph`**
>   + **`export.jsonl_parquet`** with the Granite r2 corpus vectors.
>
> Every learned Stage runs on **CUDA**; on a CPU-only host `latence stack validate` reports them
> skip-with-flag (device honesty) and the generated config still validates. Fully scriptable:
>
> ```bash
> latence setup --non-interactive --profile enterprise-sota \
>   --source file:///workspace/corpus --storage file:///workspace/latence-out \
>   --ocr-base-url http://localhost:8000/v1 --ocr-model lightonocr \
>   --llm-base-url https://openrouter.ai/api/v1 --llm-model openai/gpt-4.1-mini \
>   --pii --output ./latence.stack.yaml
> ```

> **PII note (W16, ADR-0042).** With **PII ON**, `redact` is a **chunk-level transform**: it
> consumes the induced **chunk** stream and sets each chunk's `masked_content` (each chunk ≤ the
> model window, so no PII is truncated), and `enrich` reads THAT redacted chunk stream
> (`redact → enrich → export_corpus`). The corpus Export materializes `masked_content`, so the
> exported RAG corpus is **both masked (PII-safe) AND context-enriched** — the per-chunk
> `context_header`/`induced_labels` and the masking compose, with no trade-off. `latence setup --pii`
> now produces a genuinely PII-safe, context-enriched corpus. (This closes the pre-W16 gap where the
> RAG chunks were enriched but un-masked and a doc-level redactor could truncate PII past the window.)

> **Redaction floor + mask-before-extract (W18, ADR-0044).** Two safety improvements ride on top:
> (1) a **universal financial-PII floor** — `credit_card` / `iban` / `ssn` are ALWAYS masked at the
> shared redaction seam no matter which detector backend you configure, so a learned redactor that
> *misses* one still cannot leak it (the floor is non-bypassable and can't be forgotten per-provider).
> (2) An optional **`--extract-on masked`** mode: extraction reads the REDACTED chunk stream, so PII
> never reaches entity/relation extraction or the KG at all — the stricter posture some enterprises
> require. Default `unmasked` keeps the KG's real entities and is byte-identical to before. With
> `masked`, the wizard wires `extract` to depend on `redact` for you.

---

## 2. Drop in your files + pick the parser

Put your mixed corpus anywhere on the pod, e.g.:

```bash
mkdir -p /workspace/corpus
# ... copy your PDFs, scans, .docx/.xlsx/.pptx, .txt, .md into /workspace/corpus ...
ls -la /workspace/corpus
```

`source.local_folder` walks that folder and only admits the file types you list in `extensions`
(anything else is ignored at source; malformed/oversized files are **quarantined** at intake, never
crash the run). Match `extensions` to what your parser can read.

**Which parser?** One Parse Stage = one parser Provider, so choose by what dominates your corpus:

| Your files | Parser | Notes |
|---|---|---|
| Genuinely mixed (scans, images, Office, PDF) | **`parser.render`** + an `ocr:` sub-provider (`parser.lighton`) | Universal, license-clean (PDFium/Pillow/LibreOffice/python-docx — never AGPL PyMuPDF). PDF/image/soffice pages are rasterised then OCR'd (GPU). |
| PDFs with a real text layer (born-digital) | **`parser.pdfplumber`** | Fast, CPU, no OCR. **PDF only.** |
| Only scanned PDFs / photos | `parser.lighton` (OCR VLM) directly | GPU. |

The GOOD stack below uses **`parser.render` + `parser.lighton`** because the scenario is "many
different file types". If your corpus is born-digital PDFs + text + markdown only, swap the `parse`
Stage to `parser.pdfplumber` (drop the `ocr:` block) for a much lighter, CPU-only, OCR-free run.

---

## 3. Author the stack YAML — the full GOOD stack

This is the whole quality story in one file. Save it as `stacks/my-corpus.yaml`.

> **Why this config matters more than the model.** With generic English labels, a low threshold, and
> no schema induction, the output is noisy — pronouns/laws/products get mistyped as entities and the
> relations are garbage. Turn schema induction **on** (the LLM discovers the domain's real labels per
> corpus), use **`granularity: corpus`** (one shared canonical schema, so entity types stay
> consistent across the whole KG instead of drifting per chunk), set the extraction **`threshold`
> ~0.5**, give redaction real **financial-PII** labels (IBAN, credit card, tax id — not just
> person/email/phone), and cap chunks at **`max_tokens: 640`**. Then entities are correctly typed
> (bank / law / payment-product) and the relations are meaningful.
>
> **Why 640.** The GLiNER2 extractor/redactor run on mdeberta-v3, whose token window is **768**.
> `chunk.markdown`'s `max_tokens` is counted by the *chunker's* tokenizer, which runs ~1.2× the
> model's (pod-measured: a 768-chunker-token cap hit mdeberta median 666 / max 921). So 640
> chunker-tokens ≈ ≤768 mdeberta-tokens — as big as is safe (bigger chunks = better relations + RAG
> context). The providers also carry a `max_len: 768` guard as the hard backstop.

```yaml
# stacks/my-corpus.yaml — the full GOOD stack:
#   source → intake_screen → parse → chunk[640] → content_screen
#     → induce[corpus] → extract[gliner2, thr 0.5] → redact[gliner2, financial-PII]
#     → profiling → disambiguate → graph → enrich(context) → export_corpus + export_kg
#
# {CORPUS_DIR} / {STORAGE_URI} are placeholders — fill them in §3.1 below before `latence run`.
name: my-corpus-stack
storage_uri: "{STORAGE_URI}"
stages:
  # 1) Read the folder. Only these extensions are admitted; adjust to your corpus + parser.
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

  # 3) Parse: the universal renderer + a GPU OCR sub-provider for pixel pages.
  #    For a born-digital-PDF-only corpus, replace this whole Stage with:
  #      provider: parser.pdfplumber   (and delete the config block)
  - name: parse
    capability: parse
    provider: parser.render
    depends_on: [intake_screen]
    config:
      device: cuda                   # OCR sub-provider is GPU; skip-with-flag on a CPU host
      dpi: 200                       # PDF→image rasterisation DPI (higher = sharper, slower)
      soffice_timeout_s: 120         # hard cap on the LibreOffice office-render subprocess
      ocr:
        provider: parser.lighton     # permissive GPU OCR VLM (or parser.glm)
        config: {device: cuda, max_new_tokens: 4096}

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

  # 6) Schema induction (the quality lever): a small LLM discovers the corpus's own
  #    {entity, relation, pii} types. granularity: corpus → ONE shared, canonical, deduped schema
  #    stamped on every chunk (no per-chunk/per-doc type drift across the KG).
  - name: induce
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [content_screen]
    config:
      model: openai/gpt-4.1-mini             # OpenRouter model id (openai/ prefix). Direct OpenAI: gpt-4.1-mini
      base_url: https://openrouter.ai/api/v1 # null ⇒ OpenAI; http://<vllm-host>:8000/v1 ⇒ on-prem vLLM
      # api_key comes from $OPENROUTER_API_KEY (falls back to $OPENAI_API_KEY). NEVER inline it here.
      granularity: corpus                    # one shared canonical schema for the whole corpus
      max_tokens: 2048                        # truncated reply retries at double, then fails open
      max_labels: 40
      temperature: 0.0                        # deterministic

  # 7) Fused NER+RE in ONE pass (GLiNER2). config labels are a FLOOR; the induced entity/relation
  #    types are UNIONed on per corpus (config ∪ induced). threshold ~0.5 = the quality sweet spot.
  - name: extract
    capability: fused_entity_relation
    provider: fused_entity_relation.gliner2
    depends_on: [induce]
    config:
      device: cuda
      labels: [organization, person]         # entity floor; induced entity_types union on
      relation_labels: [works for, partner of]  # relation floor; induced relation_types union on
      threshold: 0.5
      max_len: 768                            # mdeberta window guard (hard backstop)

  # 8) Learned PII redaction (GLiNER2) with FINANCIAL PII, not just person/email/phone.
  #    W16 (ADR-0042): redact is a CHUNK→CHUNK transform — it consumes the induced CHUNK stream and
  #    sets each chunk's masked_content (each chunk ≤ the model window, so no PII is truncated). It
  #    depends_on induce so the induced pii_types UNION onto this floor; enrich then reads THIS
  #    redacted chunk stream, so export_corpus is masked AND enriched.
  - name: redact
    capability: redaction
    provider: redaction.gliner2
    depends_on: [induce]
    config:
      device: cuda
      pii_labels: [person, email, phone number, address, iban, credit card number, tax id]
      threshold: 0.5

  # 9) Corpus profiling (statistics + drift diagnostics) over docs + mentions.
  - name: profiling
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, extract]

  # 10) Disambiguation: collapse surface mentions to canonical entities (learned surface+context).
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.embedding
    depends_on: [extract]
    config:
      threshold: 0.90
      context_window: 160
      embedder:
        provider: embedding.sentence_transformers
        config: {model: ibm-granite/granite-embedding-311m-multilingual-r2, dimension: 768, device: cuda, attn_implementation: flash_attention_2}
      kb:
        # entity_type MUST match the lowercased mention labels — the KB linker's type filter is
        # case-sensitive, so "ORG" would never match an "organization" mention.
        - {kb_id: Q1, name: Acme, entity_type: organization}
        - {kb_id: Q2, name: International Business Machines, entity_type: organization, aliases: [IBM]}
      kb_threshold: 0.5

  # 11) Assemble the canonical, evidence-linked knowledge graph.
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config: {corpus_id: my-corpus-stack}

  # 12) W13 context enrichment: project the assembled KG back onto each chunk as a compact,
  #     capped, confidence-gated context_header (its canonical entities + top-k neighbor triples).
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

  # 13a) KG export (nodes/edges Parquet + TTL + GraphML). basename → graph-*.
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
    config: {basename: graph}

  # 13b) RAG-corpus export from the REDACTED+ENRICHED chunk stream: masked_content (materialized as
  #      the corpus text, W16) + provenance + classification + induced_labels + context_header + a
  #      dense embedding. context_columns: true → the two W13 Parquet columns (context_header,
  #      neighbor_entity_ids).
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
```

> **Redaction AND context headers — masked and enriched in one file (W16, ADR-0042).** With W16,
> `redact` is a **chunk→chunk transform**: it consumes the induced **chunk** stream and sets each
> chunk's `masked_content` (each chunk ≤ the model window, so no PII is truncated), and `enrich`
> reads THAT redacted chunk stream (`redact → enrich → export_corpus`). The corpus Export
> materializes `masked_content` as the row's text, so the exported RAG corpus carries **masked
> (PII-safe) content AND** `induced_labels` + `context_header` + `embedding` — the masking and the
> per-chunk headers **compose in one file**, with no trade-off. The structured `pii_spans` ride the
> same `ChunkRecord` (the raw PII value is never stored). This closes the pre-W16 gap where the RAG
> chunks were enriched but un-masked (and a doc-level redactor could truncate PII past the model
> window). With PII OFF there is no `redact` Stage, so wire `enrich` back to `induce`.

### 3.1 Fill the placeholders

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

---

## 4. Run each Stage separately + inspect

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
about, inspect its checkpoint, then run a larger stack that reuses the earlier checkpoints and only
executes the new Stages. Keep the **same `--run-id` and the same `storage_uri`**, and keep every
earlier Stage's **name and config byte-identical** — resume is keyed by **stage name**, so a
renamed-or-reconfigured Stage under an existing checkpoint is **not** re-run (you'd inspect stale
output). Copy `my-corpus.yaml`, delete the trailing Stages you don't want yet, render, run.

**Method B — run once, inspect every checkpoint.** Run the full pipeline (§5) and then read each
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

### 4.1 `source` → `source.jsonl` (raw `ParserInput`s)
```bash
peek $CK/source.jsonl
# look for: one record per admitted file; correct file_name / file_type; sensitivity=internal.
python -c "print(sum(1 for _ in open('$CK/source.jsonl')),'files admitted')"
```

### 4.2 `intake_screen` → `intake_screen.jsonl` (the PASSED files)
Quarantined files are **not** here — they're in the sidecar. Check both:
```bash
peek $CK/intake_screen.jsonl
ls  /workspace/out/_latence/runs/run-0001/quarantine/    # withheld files (oversized/malformed)
ls  /workspace/out/_latence/runs/run-0001/screening/     # per-Stage screening findings
```

### 4.3 `parse` → `parse.jsonl` (`DocumentRecord`s — clean text)
```bash
peek $CK/parse.jsonl
# look for: clean, markup-stripped `content`; a `page_map`; provenance file_name/type/size intact.
python - <<PY
import json
for line in open("$CK/parse.jsonl"):
    d=json.loads(line); print(d["provenance"]["file_name"], "→", len(d.get("content","")), "chars")
PY
```

### 4.4 `chunk` → `chunk.jsonl` (`ChunkRecord`s — sizes)
```bash
peek $CK/chunk.jsonl
# look for: token_count clustered at/under 640; index runs 0..k per document; content is coherent.
python - <<PY
import json, statistics as st
tc=[json.loads(l)["token_count"] for l in open("$CK/chunk.jsonl")]
print("chunks",len(tc),"| token_count min/median/max",min(tc),int(st.median(tc)),max(tc))
PY
```

### 4.5 `induce` → `induce.jsonl` (chunks stamped with the DISCOVERED schema)
This is the payoff of schema induction — the domain labels the LLM found for **your** corpus. In
`granularity: corpus` every chunk carries the **same** merged `induced_labels`:
```bash
python - <<PY
import json
seen=set()
for l in open("$CK/induce.jsonl"):
    il=json.loads(l).get("induced_labels")
    if il: seen.add(json.dumps({k:il[k] for k in ("entity_types","relation_types","pii_types")}, sort_keys=True))
print("distinct induced schemas:", len(seen))          # corpus mode → 1
for s in list(seen)[:1]: print(json.dumps(json.loads(s), indent=2))
PY
# look for: sensible domain types (e.g. bank, law, payment_product / owns, regulated_by / iban, tax_id),
# NOT generic English. Empty/None everywhere ⇒ the key/endpoint failed — see Troubleshooting.
```

### 4.6 `extract` → `extract.jsonl` (typed entities + relations, one fused stream)
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

### 4.7 `redact` → `redact.jsonl` (`ChunkRecord`s — `masked_content` + `pii_spans`, W16)
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

### 4.8 `disambiguate` → `disambiguate.jsonl` (canonical entities)
```bash
peek $CK/disambiguate.jsonl
# look for: surface variants (IBM / International Business Machines) collapsed to one canonical id;
# KB links (kb_id Q1/Q2) where the type + surface matched.
```

### 4.9 `graph` → `graph.jsonl` (`GraphRecord`s — nodes + edges)
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

### 4.10 `enrich` → `enrich.jsonl` (chunks + `context_header`)
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

## 5. Run end-to-end

One command runs the whole DAG (skipping any Stage whose checkpoint already exists from §4):

```bash
export PYTORCH_JIT=0
export OPENROUTER_API_KEY="sk-or-..."     # (already set from §1.3)  # pragma: allowlist secret
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
- **KG stats** — `nodes` / `edges` non-zero, `evidence_coverage` near 1.0 (every edge cites evidence).
- **No-PII-leak** — the corpus RAG scan reports clean (see the redaction wiring note in §3 for what
  "clean" means given your export choice).
- **`skipped` / `skip_reason`** — on a CPU-only host the `cuda` Stages are honestly
  **skipped-with-flag** (never faked). On a real GPU pod they should run, not skip.
- **`induction_fallbacks` / `induction_documents`** — how many docs fell back to config labels; a
  high fallback count means the induction key/endpoint is misbehaving (Troubleshooting).

The machine-readable `quality-report.json` carries the same, per-Stage, for gating.

---

## 6. Discover the final output

Everything lands under `export/`:

```bash
ls -la /workspace/out/_latence/runs/run-0001/export/
#   records.jsonl        records.parquet          ← RAG corpus
#   graph-nodes.parquet  graph-edges.parquet
#   graph.ttl            graph.graphml            ← knowledge graph
EX=/workspace/out/_latence/runs/run-0001/export
```

### 6.1 The RAG corpus — `records.jsonl` (full record) / `records.parquet` (flat columns)

The JSONL carries the **full** `ChunkRecord` dump (every field, nested objects intact). Walk one row:

```bash
peek $EX/records.jsonl 1
```

Field-by-field (source: `contracts.py` `ChunkRecord`/`Record`; full table in
`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md` §2.1):

- **`content`** — the retrieval text (markup-stripped). Byte-unchanged by W13 (the context header is
  added to the *embedding input only*, never the stored text). W16 (ADR-0042): with the `redact`
  Stage on, the Export materializes each chunk's **`masked_content`** here, so `content` is the
  **PII-safe (masked)** text — never the raw variant (see the §3 redaction note).
- **`provenance`** — `source_uri`, `file_name`, `file_type`, `file_size`, `document_id`, and the
  offsets `page_start/page_end` + `char_start/char_end` locating the chunk in its source. Your audit
  trail back to the exact page/character.
- **`classification`** — `language`, `category`, `sensitivity`.
- **`induced_labels`** — the LLM-discovered `{entity_types, relation_types, pii_types}` for the corpus
  (the schema induction payoff), plus the `model_id` that induced it.
- **`context_header`** — the KG projected onto this chunk: `entities` + aligned `entity_types`,
  `neighbor_triples` (`A —label→ B`, capped), `kg_node_ids` (the chunk's own entities) and
  `neighbor_node_ids` (query-time KG-expansion join keys). `null` for a chunk with no gated entity.
- **`embedding`** — the dense Granite r2 311m vector (dimension 768). It was computed over
  `rendered_context_header + "\n" + content`, so the vector carries the KG coherence — and, under W16,
  over the **PII-safe (masked)** corpus text, consistent with the stored `content`.
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

### 6.2 The knowledge graph

```bash
# nodes: node_id, type (entity type), canonical_name, confidence, source_document_ids
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

### 6.3 The Quality Report

`quality-report.md` (human) + `quality-report.json` (machine) — the per-run scorecard from §5.

**Full field reference for downstream consumers:** [`docs/RETRIEVAL-ENGINE-INPUT-SPEC.md`](RETRIEVAL-ENGINE-INPUT-SPEC.md).

---

## 7. Incremental updates: add / update / retract / purge

Your source folder changes after the first run — new files arrive, some get edited, some must be
deleted (a customer exercises their right-to-be-forgotten). You do **not** re-run everything and you
do **not** write a Python script. Three CLI commands drive the corpus forward, each committing the
next **Corpus Version** (immutable, inspectable, rollback-able). They take the **same stack YAML**
`latence run` takes.

> **Honest boundary (issue #42).** The *committed* result of a delta is **surgical**: only the
> records in the blocking neighborhoods your change touched are re-resolved and re-committed; every
> untouched cluster is carried forward byte-identical, and the churn prints the affected-set size as
> the witness. What is **not** yet incremental is the **extraction pass** — a delta re-parses /
> re-chunks / re-extracts the **whole current source** from scratch (there is no cross-run doc-level
> checkpoint reuse yet). So the *stored corpus* honours "recompute only the affected set", but the
> *compute cost* of a delta is still O(corpus). Don't expect a 1-file delta to run in 1-file time.

### 7.1 `latence delta` — add / update (the source folder changed)

Drop new or edited files into the source folder, then:

```bash
# Each delta needs a FRESH run id; omit --run-id and it auto-generates a UTC-timestamped one
# (delta-YYYYMMDDTHHMMSSZ). Pin one with --run-id if you want a stable name.
latence delta stacks/my-corpus.yaml
# or, explicit:
latence delta stacks/my-corpus.yaml --run-id delta-2026-07-13
```

It content-hashes the current folder against the last committed Version, detects added / changed /
removed files, and commits Version N+1. The churn it prints:

```text
Corpus Version v3 -> v4 (incremental)
  documents: +2 ~0 -1 (=118 unchanged)
  entities:  +5 created, 1 merged, 0 split
  edges:     +7 -1
  records:   0 retracted, 0 purged (retraction)
  affected set: 6/431 corpus-level records recomputed (extraction still re-runs the whole source — issue #42)
  drift: 0.024
```

`affected set: 6/431` is the payoff signal — only 6 of the 431 corpus-level records were re-resolved;
the other 425 were carried forward verbatim. (An edit under the content-addressed local-folder Source
shows up as `+1 ~1`-style *add + delete*, not `~updated`: editing a file mints a new content-id and
retires the old one. That's expected — see ADR-0029 §4.)

Re-running a delta over an **unchanged** folder is a **no-op** — the fingerprint matches head, so no
new Version is minted (idempotent). Deltas are **deterministic**: the same source + config reproduces
the same Corpus Version fingerprint.

### 7.2 Where to get a `document_id`

`retract` and `purge` take content-addressed document ids (`sha256:<64 hex>`). Two ways to get one:

```bash
# (a) From the export — every record carries its source document id:
python -c "import json; print({json.loads(l)['provenance']['document_id'] for l in open('$EX/records.jsonl')})"

# (b) Hash the file directly (the id IS the sha256 of the file bytes):
printf 'sha256:%s\n' "$(shasum -a 256 /workspace/corpus/contract-4471.pdf | cut -d' ' -f1)"
```

### 7.3 `latence retract` — soft delete (auditable, rollback-able — the default)

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

### 7.4 `latence purge` — hard delete (GDPR / right-to-be-forgotten)

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

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `nvrtc: failed to open libnvrtc-builtins.so.13.0` (GLiNER2/DeBERTa crash) | torch JIT on SM 12.x / cu13x | `export PYTORCH_JIT=0` before running (§1.3). |
| `error: externally-managed-environment` on `pip install` | RunPod PEP-668 system Python | add `--break-system-packages` (§1.2), or use a venv. |
| `ModuleNotFoundError: peft` (or `pdfplumber`, `pyarrow`) at run time | a `--no-deps` editable install skipped a hidden GLiNER2 dep | reinstall the package **without** `--no-deps` and `pip install ... peft pdfplumber pyarrow` (§1.2). `peft` is a real, undeclared GLiNER2 dependency. |
| Every chunk's `induced_labels` is `null`; `quality-report` shows high `induction_fallbacks` | no/invalid induction key, wrong `base_url`, or unreachable endpoint (the Stage **fails open** to config labels — never crashes) | set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) as an env var; check `base_url` matches your provider; run `latence run ... --verbose` and read the `schema_induction fell back to config labels: ...` log lines. |
| Tail entities/relations missing; offsets look off on long chunks | chunk tokenised past the mdeberta **768** window and was silently truncated | keep `chunk.markdown max_tokens ≤ 640` (~≤768 mdeberta-tokens); the provider `max_len: 768` guard is the backstop (§3 callout). |
| Output is noisy — pronouns/laws/products mistyped, garbage relations | generic English labels + no schema induction + low threshold | turn `induce` **on** with `granularity: corpus`, raise extraction `threshold` to ~0.5, give redaction the financial-PII labels (§3). This matters more than the model choice. |
| KB linking silently drops every candidate | `kb.entity_type` case mismatch with the (lowercased) mention labels — the linker's type filter is case-sensitive | make `entity_type` match the mention labels exactly (`organization`, not `ORG`). |
| GPU Stages report `skipped` / `skip_reason` in the report | CPU-only host (device honesty — skipped-with-flag, never faked) | run on a real CUDA pod with the heavy packages installed; the CPU-reachable spine still validates. |
| A file didn't make it into the corpus | not in `source.local_folder extensions`, or quarantined at intake, or a `PARSE_ERROR` | check the `extensions` list, the `quarantine/` and `screening/` sidecars, and the parse checkpoint (§4.2, §4.3). |
| `context_header` / `neighbor_entity_ids` absent from `records.parquet` | `context_columns` not set on `export_corpus`, or the corpus was exported from `redact` not `enrich` | set `context_columns: true` and wire `export_corpus → enrich` (§3). |
| A re-run didn't pick up a config change to an early Stage | resume is keyed by **stage name**; an existing checkpoint under the same name is skipped | delete that Stage's checkpoint (`rm $CK/<stage>.jsonl`) or use a fresh `--run-id`. |
| `latence delta` committed no new Version / printed the same version number | the source folder is unchanged — a delta over an identical corpus is an idempotent no-op (the fingerprint matches head) | change the folder (add/edit/remove a file) before the delta; nothing to commit is correct behaviour (§7.1). |
| `retract`/`purge` fails with `[unknown]: Cannot delete from an empty corpus` | you deleted before any `latence run`/`delta` committed a Corpus Version | run the pipeline once first so there is a Version to delete from (§7.3). |
| `retract`/`purge` says the id `does not look like a document id` | the id isn't `sha256:<64 hex>` (e.g. you passed a file path or a truncated id) | get a real `document_id` from `records.jsonl` (`provenance.document_id`) or by hashing the file (§7.2). |
| A `latence delta` ran far slower than the size of the change | expected — extraction is not yet cross-run incremental; a delta re-extracts the whole source (issue #42) | the *committed* corpus is still surgical (see the `affected set: N/total` line); the compute caveat is documented, not a bug (§7). |
