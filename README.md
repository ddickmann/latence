# Latence Framework

A model-agnostic, cloud-agnostic framework for turning messy, siloed enterprise data into high-quality, AI-ready data — from source documents through parsing, screening, entity and relation extraction, PII handling, enrichment, and disambiguation, to a corpus-level knowledge graph and a RAG-ready corpus.

The framework defines the **stack and the orchestration contracts**. It does **not** dictate which OCR parser, NER model, embedding model, or PII cleaner you use, or which cloud you run on. Those are plugins and configuration.

> Status: **built — Phase-1 complete and G1-passed.** The full 13-stage spine
> (Source → … → Export) runs end-to-end through the local Runner on CPU, and the
> G1 profound-testing gate is signed off — E2E on multiple messy corpora,
> security review, reliability/chaos (crash-resume, Delta idempotency), a
> performance baseline, and determinism ([`issues/gate/G1-RESULTS.md`](./issues/gate/G1-RESULTS.md)).
> The grilled architecture lives in [`CONTEXT.md`](./CONTEXT.md) (the glossary)
> and [`docs/adr/`](./docs/adr/) (32 architecture decisions), with
> [`docs/PORTING-MAP.md`](./docs/PORTING-MAP.md) recording what was mined from four
> private production repos. **Not yet 1.0:** performance-at-scale and deeper
> structural file validation are flagged as Phase-2 hardening; see the honest
> residuals in the G1 results.

## The idea in one screen

- **Stages** are units of work with typed input/output contracts (Source → Screen → Parse → Chunk → Screen → Entity Extraction → Relation Extraction → Redaction → Profiling → Disambiguation → Graph Assembly → Export).
- **Capabilities** are the narrow interfaces a Stage depends on (Parser, EntityExtractor, Embedder, …). The framework never names a model.
- **Providers** fulfil Capabilities — an in-process model, a remote endpoint, or a cloud AI service. Swapping a model is a one-line config change.
- **Runners** execute a Pipeline. A built-in local Runner runs the whole thing on a laptop (CPU-only) with checkpoint/resume; thin adapters run the same Stages on Databricks or Airflow.
- **Storage** is one fsspec seam: `file://`, `s3://`, `gs://`, `abfs://` — no code change.
- Every record carries **Provenance** (source lineage) and **Classification** (language, category, sensitivity), enforced at every Stage boundary.
- Every run emits a **Quality Report** so "AI-ready" is measured, not asserted.

## Design principles (see `docs/adr/`)

1. Open-source core, Apache-2.0, published before any employment engagement (ADR-0001).
2. Stage library with pluggable Runners — not a control plane (ADR-0003).
3. Capability protocols with Provider plugins as the model-agnostic seam (ADR-0004).
4. v1 is the AI-ready-data spine only; retrieval-serving, ontology induction, link prediction, trace scoring are post-v1 plugins (ADR-0005).
5. One schema, dual serialization (JSONL + Parquet/Arrow) (ADR-0006).
6. CPU-first reference Providers; GPU serving is an upgrade path (ADR-0007).
7. Permissive weights only as defaults; restricted-license models (e.g. EmbeddingGemma) are opt-in (ADR-0012).
8. Python core and Providers; Rust only for proven text hot paths, behind a Python fallback (ADR-0015).
9. Thin core plus per-Provider packages, so you install only what you use (ADR-0016).

## Repository layout

```
latence-framework/
├── CONTEXT.md                 # glossary — the canonical language
├── docs/
│   ├── adr/                   # architecture decisions (0001–0033)
│   └── PORTING-MAP.md         # what was mined from the private repos
├── packages/
│   ├── latence-core/          # contracts, Capabilities, Runner, Storage (pure-Python)
│   ├── latence-parser-*/      # Parser Providers
│   ├── latence-ner-*/         # EntityExtractor Providers
│   ├── ...                    # one package per Provider family
│   ├── latence-sink-qdrant/   # one-shot loader: Export → a Qdrant collection (idempotent)
│   └── latence-demo/          # the CPU demo set: bundled corpus + viewer + `latence-demo`
└── demo/                      # docker compose: messy corpus → KG + Quality Report
```

## Try the demo (one command)

`docker compose up` (or `uv run latence-demo`) ingests a bundled **messy** corpus — mixed
text/markdown + a real PDF, plus deliberately dangerous files — and produces parsed markdown, a
browsable **knowledge graph**, and the **Quality Report**, rendered in a **self-contained web
viewer** (no external CDN/network). CPU-only, no cloud, no GPU, no credentials.

```bash
# From a clean clone, with uv:
uv sync
uv run latence-demo
open /tmp/latence-demo/_latence/runs/demo/export/index.html

# Or with Docker (no local Python):
cd demo && docker compose up
open out/_latence/runs/demo/export/index.html
```

You will see the type-spoofed file, the zip bomb, and the oversized file **Quarantined**; the
prompt-injection document **flagged**; and `IBM` / `International Business Machines` collapsed
to one node with `works_for` / `partner_of` edges, each carrying **Evidence**.

**The model- and cloud-agnostic proof** — change ONE line and rerun the *identical* Pipeline:

```bash
# The learned-NER Provider ships heavy deps (torch), so it is NOT in the CPU demo set —
# install it once, then swap in a single Provider:
uv pip install -e packages/latence-ner-gliner          # brings the GLiNER model Provider
uv run latence-demo --entity-provider entity.gliner    # a local model instead of the gazetteer
```

Without that install the swap exits cleanly (non-zero) with a clear
"FAIL: entity Provider 'entity.gliner' is not installed. Install it first: pip install latence-ner-gliner"
message naming what to install — the default `uv run latence-demo` above stays CPU-only and needs
no extra packages.

`diff examples/demo-pipeline.yaml examples/demo-pipeline.swap.yaml` shows the single-Provider
change. See [`packages/latence-demo/README.md`](./packages/latence-demo/README.md) for details.

## Install

The framework is a thin core plus one installable package per Provider family
(ADR-0016) — you install only what you use, so the CPU-first "runs on a laptop"
promise holds (no torch/cloud client unless you ask for it).

```bash
pip install latence-core            # contracts, Capabilities, local Runner, Storage
pip install latence-demo            # + the CPU demo set (real-PDF parser), one-command onboarding
pip install latence-ner-gliner      # opt-in: a learned NER Provider (pulls torch/transformers)
pip install latence-relation-llm    # opt-in: LLM relation extraction over any OpenAI-compatible endpoint
```

The meta-package pins the CPU demo set (`latence-core` + `latence-parser-*` +
`latence-demo`); the heavy/endpoint Providers stay isolated. Python ≥ 3.11.

## Package map

| Package | What it is |
|---|---|
| `latence-core` | Contracts, Capability protocols, Pipeline, local Runner, Storage seam + all pure-Python in-core reference Providers. Near-zero deps. |
| `latence-parser-plaintext` | Passthrough text/markdown Parser. |
| `latence-parser-document` | CPU PDF+text → markdown + page map Parser (`pypdf`). |
| `latence-parser-endpoint` | Parse performance path (any OpenAI-compatible vision endpoint). |
| `latence-ner-gliner` | Learned zero-shot NER Provider (GLiNER). |
| `latence-ner-endpoint` | NER performance path (endpoint-backed). |
| `latence-relation-gliner` | Fused Entity+Relation Provider (gliner-relex). |
| `latence-relation-llm` | LLM relation extraction (OpenAI-compatible / Ollama). |
| `latence-pii-gliner` | Learned zero-shot PII Redaction Provider. |
| `latence-pii-presidio` | Real Microsoft Presidio PII Redaction Provider (spaCy NER + recognizers). |
| `latence-embedder-endpoint` | Embedder performance path (endpoint-backed). |
| `latence-runner-airflow` | Airflow Runner adapter (`airflow` extra). |
| `latence-demo` | One-command offline demo + self-contained KG/Report viewer. |
| `latence-benchmark` | The reproducible **public** benchmark harness (pinned datasets, pinned stack configs, published results page). Not a pipeline Provider. |

## Documentation & governance

- [`CONTEXT.md`](./CONTEXT.md) — the glossary; the canonical language of the project.
- [`docs/adr/`](./docs/adr/) — the 32 architecture decisions.
- [`docs/TUTORIAL-fresh-pod-walkthrough.md`](./docs/TUTORIAL-fresh-pod-walkthrough.md) — guided setup (`latence setup` → `latence process`) and the full stage-by-stage walkthrough.
- [`docs/running-a-stack.md`](./docs/running-a-stack.md) — run + validate a full-spine stack (`latence stack validate`).
- [`docs/running-a-bakeoff.md`](./docs/running-a-bakeoff.md) — run a per-Stage provider bake-off (`latence bake-off`).
- [`docs/BENCHMARKS.md`](./docs/BENCHMARKS.md) — public benchmark results (our reproducible numbers, kept separate from cited vendor claims), and [`docs/running-the-benchmarks.md`](./docs/running-the-benchmarks.md) to reproduce them yourself.
- [`docs/TUTORIAL-fresh-pod-walkthrough.md#7-incremental-updates-add--update--retract--purge`](./docs/TUTORIAL-fresh-pod-walkthrough.md#7-incremental-updates-add--update--retract--purge) — incremental corpus updates (`latence delta` / `retract` / `purge`).
- [`issues/gate/G1-RESULTS.md`](./issues/gate/G1-RESULTS.md) — the signed profound-testing gate.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — dev setup (uv workspace), the quality gates, the DoD tiers.
- [`SECURITY.md`](./SECURITY.md) — trust model, built-in controls, responsible disclosure.
- [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md) — Contributor Covenant.
- [`THIRD-PARTY-LICENSES.md`](./THIRD-PARTY-LICENSES.md) — runtime dep + model licenses (all permissive).

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
Copyright 2026 Dennis Dickmann (latence.ai). Open-source and personally owned
(ADR-0001).
