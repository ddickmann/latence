# Quickstart — from a folder of documents to a corpus and a knowledge graph

*Roughly 15 minutes on a laptop. No GPU. Once installed, only step 3 touches the network.*

By the end of this page you will have run the pipeline over your own documents and read the
result. What you get out is **files** — a chunked, PII-masked corpus with vectors, a
knowledge graph whose every edge carries the evidence it was built from,
and a Quality Report — which is the whole point: nothing is locked inside a service.

If you want to keep going after this page, the
[tutorials](tutorials/index.md) pick up exactly where it stops.

**Every command below is real.** Where a step prints something, a `✓ look for` block states what a
correct result looks like, so you can tell "it worked" from "it ran". If your output disagrees with
a `✓ look for` block, stop there rather than continuing — the mismatch is the finding.

---

## 1. Install

The framework is a workspace of small packages: a thin core plus one package per Provider
([ADR-0016](adr/0016-thin-core-per-provider-packages.md)). From a clone:

```bash
git clone https://github.com/ddickmann/latence
cd latence
uv sync
```

`uv sync` installs the CPU-first, offline, deterministic set: `latence-core` (which brings the
`latence` CLI), the plaintext / pdfplumber / document parsers, the benchmark harness, the run
console, and the demo. The heavy learned Providers (torch, transformers, spaCy, Airflow) are
**deliberately not** in that set — you add the ones you want in step 3.

```bash
uv run latence --help
```

✓ **look for** — a command list containing `run`, `setup`, `process`, `stack`, `bake-off`, `tune`,
`delta`, `retract`, `purge`, `contracts` and `env`. If `latence` is not found, `uv sync` did not
finish; re-run it and read its output.

!!! note "`uv run` vs a bare command"
    Every command on this page is written `uv run <cmd>` so it resolves inside the workspace
    environment. If you activated the venv (`source .venv/bin/activate`) you can drop the
    `uv run` prefix everywhere.

---

## 2. Prove the install in one command

Before pointing anything at your own data, run the bundled demo. It stages a deliberately messy
four-document corpus — including a zip bomb, an oversized file, a PDF whose bytes are not a PDF,
and a document carrying a prompt-injection payload — runs the full spine over it CPU-only and
offline, and asserts that each planted hazard was caught:

```bash
uv run latence-demo --storage-uri file://$PWD/demo-out --run-id demo
```

✓ **look for** — a summary block, and a **zero exit code**:

```
  documents: 4 ingested, 3 quarantined, 1 flagged
  graph:     6 nodes, 33 edges, 2 KB-linked, 33 edges with Evidence
  contracts: complete
  screening checks:
    - archive.zip: expected quarantine:zip_bomb, got quarantine:zip_bomb [ok]
    - dump.txt: expected quarantine:oversized, got quarantine:oversized [ok]
    - invoice.pdf: expected quarantine:type_spoof, got quarantine:type_spoof [ok]
    - support-ticket.md: expected flag:prompt_injection, got flag:prompt_injection [ok]
```

Every count in that block is fixed — the corpus is bundled and every Provider in this stack is
deterministic, so a different number means something is wrong, not that your machine differs.
`latence-demo` exits non-zero if a planted hazard slips through or contract-completeness fails, so
this command is a smoke test as well as a display.

The last line prints a `file://` URL for a **self-contained** HTML viewer — one file, inlined CSS,
no CDN, no network. Open it.

✓ **look for** — the six-node graph, and beneath it a table headed **"Edges & per-edge Evidence"**
with a row per edge: head, relation, tail, confidence, the **evidence mentions** the edge was built
from, and the source documents. All 33 edges carry it. Evidence on every edge is a contract, not a
nicety — it is what makes an assembled graph auditable rather than merely plausible.

---

## 3. Point it at your own documents

`latence setup` is the front door. It walks your source folder, detects the file types, recommends
a parse path, and writes **one** validated stack config; `latence process` runs it. There is no
hand-authored YAML in this path.

### 3a. Install the Providers the `lite` profile wires

The `lite` profile is CPU-first but not dependency-free — it wires real learned models
(GLiNER2 for extraction and PII, a Granite embedder) which is what makes its output good. Install
their packages:

```bash
uv pip install -e packages/latence-extract-gliner2 \
               -e packages/latence-pii-gliner2 \
               -e packages/latence-embedder-st
```

!!! warning "This is the step that costs time and bandwidth"
    Those three packages pull the torch/transformers stack, and the first run downloads model
    weights (`fastino/gliner2-multi-v1`, `fastino/GLiNER2-Guardrails-PII-Multi`,
    `ibm-granite/granite-embedding-97m-multilingual-r2`). Budget several minutes and a few GB, once.
    Everything runs on CPU afterwards.

    If you would rather not download anything, **skip to
    [Tutorial 1](tutorials/01-your-first-pipeline.md)**, which runs the same spine end to end with
    the in-core deterministic Providers only — zero downloads, sub-second runs — and comes back to
    the learned Providers later.

### 3b. Give it a schema, or give it an LLM

Extraction needs to know *what* to extract. You have two options and the wizard supports both:

* **Per-chunk schema induction (the default).** A small LLM derives the entity, relation and PII
  types present in each chunk. Set the key it reads (`OPENROUTER_API_KEY` for the default
  OpenRouter endpoint, `OPENAI_API_KEY` for anything else) and let the wizard use its defaults.
* **A fixed schema you supply (`--predefined-labels`).** No `induce` stage is emitted at all, zero
  LLM calls, no key needed. Right when your taxonomy is already known — and the right choice for a
  first run.

Take the second road here. Write your schema to `labels.yaml`:

```yaml
entity_types: [organization, person, location]
relation_types: [works_for, partner_of]
pii_types: [person, email, phone, iban]
```

### 3c. Generate the stack

```bash
uv run latence setup \
  --source ./my-documents \
  --storage "file://$PWD/latence-out" \
  --profile lite \
  --parser parser.pdfplumber \
  --device cpu \
  --predefined-labels ./labels.yaml \
  --non-interactive
```

Drop `--non-interactive` (and most of the other flags) to be walked through it with a prompt and a
one-line rationale per question. Every unset flag falls back to the same default either way, so the
same inputs always produce a byte-identical config.

✓ **look for** — a summary ending in an instruction to run `latence process`, and above it:

```
Detected file types under the source folder: md

Predefined-only mode: schema induction is OFF (no `induce` stage, zero LLM calls); extraction
runs every chunk against your fixed labels.

Saved quality-baked stack to: latence.stack.yaml
  profile: lite
  stages (12): source -> intake_screen -> parse -> chunk -> content_screen -> extract -> redact ->
               profiling -> disambiguate -> graph -> export_kg -> export_corpus
  parser: parser.pdfplumber   device: cpu
  intake size cap: 2097152 bytes — sized from your corpus (largest file 232 bytes)
  PII redaction + guardrail screening: ON (financial-PII; chunk-level masking + report)
```

Two lines are worth reading rather than skimming:

* **`Detected file types under the source folder: ...`** (printed first). If it is missing your
  documents' extension, `--source` is pointing at the wrong place and everything downstream will
  be empty.
* **the intake size cap.** It is derived from *your* corpus (largest observed file × 2, floored at
  2 MiB), not a magic constant. If the wizard warns that your largest file is over the cap, those
  files will be quarantined at intake and never reach the corpus — split them or raise
  `intake_screen.config.max_bytes` in the generated file.

The summary also prints a `schema induction:` line naming an LLM endpoint and its key variable. In
predefined-only mode that is the default it *would* have used; the generated config contains no
`induce` stage, so nothing reads the key. Confirm it: `grep -c 'name: induce' latence.stack.yaml`
prints `0`.

The generated `latence.stack.yaml` is a plain, diffable, heavily-commented config. Read it. Every
stage names a Capability and a Provider, and each one is swappable without touching another.

### 3d. Run it

```bash
uv run latence process --config latence.stack.yaml --run-id run-0001
```

`process` first runs the same offline structural check as `latence stack check` — DAG, phase
boundary, capability wiring, every nested `{provider, config}` block, config keys that go nowhere —
and refuses to start on an error, so a typo costs a second rather than a pipeline. Then it executes
the stages on the local Runner.

✓ **look for**
* the command exits **0** and prints the Quality Report as JSON on stdout (it is pipeable —
  `| jq .` works);
* `contracts_complete` is `true` in that JSON;
* `latence-out/_latence/runs/run-0001/` now exists, holding `checkpoints/`, `export/`,
  `quality-report.json`, `quality-report.md` and `manifest.json`.

If a stage fails, the run stops with a **typed** error naming the provider and the operation — never
a raw library traceback, and never a message containing your record content.

---

## 4. Read the result

The run console is a separate package so the core CLI stays dependency-thin. It is read-only,
loopback-only, and makes no network requests:

```bash
uv run latence-console ./latence-out
```

Point it at the **storage location** — the directory holding `_latence/runs/` — not at a single run
directory.

✓ **look for** — `Latence run console — read-only, this machine only.` and a
`http://127.0.0.1:8787/` URL. Opening it shows the run list with a headline status
(`Passed` / `Needs attention` / `Failed` / `Unfinished`), and selecting the run renders its Quality
Report as **plain sentences with the numbers in them** — traceability, KG node/edge counts and
evidence coverage, redaction counts — before any per-stage table. `Ctrl-C` stops it and leaves
nothing behind.

The same content is on disk as `quality-report.md` if you would rather read it in a terminal.

---

## 5. What you actually got

```bash
ls latence-out/_latence/runs/run-0001/export/
```

✓ **look for** — the corpus and the knowledge graph, as portable files:

| File | What it is |
| --- | --- |
| `records.jsonl` / `records.parquet` | One row per chunk: masked text, provenance, offsets, page span, and the dense `embedding`. |
| `graph-nodes.parquet` | Resolved entity nodes — canonical name, type, confidence, member mentions, source documents. |
| `graph-edges.parquet` | Extracted relations — each with an `evidence` block naming the source documents and the exact mentions it was built from. |
| `graph.ttl` / `graph.graphml` | The same graph as RDF and as GraphML, for tools that speak them. |
| `quality-report.md` / `.json` | The artifact that substantiates "AI-ready" — human- and machine-readable. |

The processed text is in `records.parquet`; everything an auditor needs is in the
graph files and the report. [Tutorial 2](tutorials/02-understanding-the-output.md) reads each one
field by field.

---

## 6. Where to go next

<div class="grid cards" markdown>

- :material-numeric-1-box: **[Your first pipeline](tutorials/01-your-first-pipeline.md)**
  The same run with no downloads at all, stage by stage, with the config explained line by line.

- :material-numeric-2-box: **[Understanding the output](tutorials/02-understanding-the-output.md)**
  Every artifact, every column, and how to trace a claim back to the sentence it came from.

- :material-numeric-3-box: **[Bringing your own Provider](tutorials/04-bringing-your-own-provider.md)**
  Implement a Capability, register the entry point, and have the conformance suite check you.

- :material-numeric-4-box: **[Production](tutorials/05-production.md)**
  GPU serving, staged runs, checkpoint/resume, and incremental corpus deltas.

</div>

Reference material, when you need it rather than in order:
[Concepts & Architecture](concepts/index.md) for the Capability / Provider / Stage / Runner /
Storage vocabulary, the [Provider catalog](providers/catalog.md) for what is pluggable, and
[Writing an adapter](writing-an-adapter.md) for the helpers that make a Provider small.
