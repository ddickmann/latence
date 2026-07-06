# Latence Framework

A model-agnostic, cloud-agnostic framework for turning messy, siloed enterprise data into high-quality, AI-ready data — from source documents through parsing, screening, entity and relation extraction, PII handling, enrichment, and disambiguation, to a corpus-level knowledge graph and a RAG-ready corpus.

The framework defines the **stack and the orchestration contracts**. It does **not** dictate which OCR parser, NER model, embedding model, or PII cleaner you use, or which cloud you run on. Those are plugins and configuration.

> Status: **design phase.** This repository currently holds the grilled architecture — [`CONTEXT.md`](./CONTEXT.md) (the glossary) and [`docs/adr/`](./docs/adr/) (17 architecture decisions) — plus the [`docs/PORTING-MAP.md`](./docs/PORTING-MAP.md) that mines four private production repos into this design. Code follows.

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

## Repository layout (target)

```
latence-framework/
├── CONTEXT.md                 # glossary — the canonical language
├── docs/
│   ├── adr/                   # architecture decisions (0001–0017)
│   └── PORTING-MAP.md         # what to mine from the private repos
├── packages/
│   ├── latence-core/          # contracts, Capabilities, Runner, Storage (pure-Python)
│   ├── latence-parser-*/      # Parser Providers
│   ├── latence-ner-*/         # EntityExtractor Providers
│   ├── ...                    # one package per Provider family
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
