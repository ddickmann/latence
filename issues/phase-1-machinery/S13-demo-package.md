# S13 — Demo package

**Parent**: S9 · **Phase 1 · Baseline bar**

## What to build
The showable artifact (ADR-0011). `docker compose up` ingests a bundled messy sample corpus (mixed PDFs, scans, office docs, and a couple of deliberately dangerous files to exercise Screening) and produces parsed markdown, a browsable knowledge graph, and the Quality Report — CPU-only, no cloud, no GPU. The decisive moment: change ONE line of pipeline config to swap a Provider (local GLiNER → a cloud NER, or local embeddings → a hosted embedder) and rerun the identical Pipeline. A thin, self-contained web viewer renders the KG and the report.

## Ports (algorithmic assets)
- None; composes S1–S9 outputs.

## Acceptance criteria
- [ ] `docker compose up` runs the full spine on the bundled corpus with no external credentials.
- [ ] A dangerous fixture is visibly Quarantined; a prompt-injection fixture is visibly flagged.
- [ ] The web viewer renders the KG (nodes/edges/Evidence) and the Quality Report; no external CDN/network needed.
- [ ] A documented one-line config change swaps a Provider and the rerun reflects it — the model- and cloud-agnostic proof.
- [ ] README quickstart reproduces the demo from a clean clone.
- [ ] Baseline bar met.

## Blocked by
S9. (S10, S11 enrich the demo when available.)
