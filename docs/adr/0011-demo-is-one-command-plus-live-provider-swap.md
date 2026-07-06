# Demo: one command plus a live provider swap

The reference demo is `docker compose up`: it ingests a bundled messy sample corpus (mixed PDFs, scans, office docs, and a couple of deliberately dangerous files to exercise Screening) and produces parsed markdown, a browsable knowledge graph, and the Quality Report — CPU-only, no cloud, no GPU. The decisive moment is changing ONE line of pipeline config to swap a Provider (local GLiNER → Azure Comprehend, local embeddings → Bedrock) and rerunning the identical Pipeline, demonstrating model- and cloud-agnosticism concretely rather than by assertion. A thin, self-contained web viewer renders the KG and report. Rejected: a notebook walkthrough (reads as a script, hides the ops story — resume, storage-agnostic, Runners), a cloud-deployed stack on personal infra (undercuts "runs in your environment," not handable to a client), and three vertical demos (3× solo build plus domain-data tuning before the start date).

---

**S13 build note — how the demo is realised.** Shipped as the `latence-demo` package (the CPU
demo set of ADR-0016): a deterministic, code-generated bundled messy corpus (mixed text/markdown
+ a real multi-page PDF + deliberately dangerous fixtures — a type-spoofed file, a zip bomb, an
oversized file, and a prompt-injection document — auditable Python, no opaque binaries); the
full-spine demo Pipeline (`build_pipeline`) composing S1–S9; a `DemoBuilder` that stages the
corpus, runs the Pipeline on the built-in `LocalRunner` (CPU-only, offline), and composes the
showable artifact; and `export.demo_site`, an Export Provider (registered via
`latence.providers`) that renders the knowledge graph — nodes, edges, and per-edge Evidence — as
ONE self-contained HTML file on Storage (all CSS/JS/graph inlined; no external CDN or network),
exactly as ADR-0017 makes the KG a portable file rather than a live DB. The one-line Provider
swap is `build_pipeline(entity_provider=…)` / `latence-demo --entity-provider …` (default the
in-core gazetteer; `entity.gliner` local model or `entity.endpoint` cloud reruns the identical
Pipeline), with `examples/demo-pipeline.yaml` and `examples/demo-pipeline.swap.yaml` the
committed one-line diff. `docker compose up` runs the identical code with no credentials; the
command exits non-zero if a planted dangerous fixture slips through or contract-completeness
fails, so the demo doubles as a smoke test.
