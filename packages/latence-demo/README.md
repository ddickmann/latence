# latence-demo — the showable one-command demo

The reference demo for the Latence framework (ADR-0011): ingest a **bundled messy corpus**
(mixed text/markdown + a real multi-page PDF, plus deliberately dangerous files) and produce
**parsed markdown, a browsable knowledge graph, and the Quality Report** — CPU-only, no cloud,
no GPU, no credentials — rendered in a **thin, self-contained web viewer** (no external CDN or
network).

It composes the S1–S9 spine and adds only three things: the bundled corpus, the
`export.demo_site` viewer Export Provider, and the `DemoBuilder` that stages the corpus, runs
the Pipeline on the built-in local Runner, and assembles the showable page.

## Quickstart (from a clean clone)

```bash
# 1. From the repo root, with uv (https://docs.astral.sh/uv/):
uv sync

# 2. Run the whole demo (stages the corpus, runs the full spine, writes the viewer):
uv run latence-demo

# 3. Open the self-contained viewer printed at the end, e.g.:
open /tmp/latence-demo/_latence/runs/demo/export/index.html
```

Or with Docker (no local Python needed):

```bash
cd demo
docker compose up      # runs the identical demo; artifacts land in ./out
open out/_latence/runs/demo/export/index.html
```

The command exits non-zero (a built-in smoke test) if a planted dangerous file was **not**
caught or if contract-completeness failed.

## What you see

- **Knowledge graph** — nodes (canonical entities), edges (relations), and **per-edge
  Evidence**. `IBM` and `International Business Machines` collapse to one node (cross-document
  Disambiguation); `works_for` / `partner_of` edges connect them, each justified by Evidence.
- **Quality Report** — per-Stage metrics, contract-completeness, and the **Screening** table:
  the type-spoofed file, the zip bomb, and the oversized file are **Quarantined**; the
  prompt-injection document is **flagged** (it survives into the corpus with a risk marker so a
  RAG consumer can exclude it).
- **Bundled corpus** — every ingested file and its Screening disposition.

## The one-line Provider swap (model- and cloud-agnostic proof)

Change exactly one line — the Entity-Extraction Provider — and rerun the **identical**
Pipeline:

```bash
# In-core CPU gazetteer (default):
uv run latence-demo

# A local GLiNER model instead — same Stages, same contracts, same viewer:
uv pip install -e packages/latence-ner-gliner   # brings the model Provider
uv run latence-demo --entity-provider entity.gliner

# Or a cloud / vLLM endpoint (no pipeline change, just the Provider + its base_url):
uv run latence-demo --entity-provider entity.endpoint
```

The committed `examples/demo-pipeline.yaml` and `examples/demo-pipeline.swap.yaml` are the same
Pipeline as YAML; `diff` them to see the single-Provider change.

## Contents

- `latence_demo.corpus` — the deterministic bundled messy corpus (auditable Python, no opaque
  binaries) + the planted dangerous fixtures.
- `latence_demo.pipeline.build_pipeline` — the full-spine demo Pipeline (S1–S9).
- `latence_demo.viewer.DemoSiteExport` — the `export.demo_site` Export Provider: renders the KG
  as one self-contained HTML file on Storage.
- `latence_demo.builder.DemoBuilder` — stage → run → compose the showable artifact.
- `latence_demo.cli` — the `latence-demo` command.

CPU-only, offline, permissively licensed (ADR-0007/0012). `pip install latence-demo` pulls only
`latence-core` + `latence-parser-document` — no torch, no cloud client, no GPU.
