# Concepts & Architecture

Latence is built from five load-bearing concepts. Learn these and the rest of the
framework — every Stage, Provider, and ADR — reads cleanly.

| Concept | One-line definition | Deepen |
| --- | --- | --- |
| **Capability** | A narrow, model-agnostic interface a Stage depends on (`Parser`, `EntityExtractor`, `PIIDetector`), typed against the framework's data contracts. | [ADR-0004](../adr/0004-capability-protocols-with-provider-plugins.md) · [reference](../reference/capability.md) |
| **Provider** | A concrete implementation of a Capability — an in-process model, a remote endpoint, or a cloud service — registered as a plugin. The framework never names a model. | [ADR-0004](../adr/0004-capability-protocols-with-provider-plugins.md) · [catalog](../providers/catalog.md) |
| **Stage** | A unit of pipeline work with a typed input and output contract. Stages declare what they consume and produce; they do not know how they are scheduled or which model fulfils them. | [ADR-0003](../adr/0003-stage-library-with-pluggable-runners.md) · [stages](../pipeline/index.md) |
| **Runner** | An execution adapter that schedules Stages on a substrate. Two ship: the built-in local Runner and Airflow (`latence-runner-airflow`); other substrates are seam-compatible follow-ons. Runners own scheduling, retries, checkpointing; they never define pipeline semantics. | [ADR-0003](../adr/0003-stage-library-with-pluggable-runners.md) · [reference](../reference/runner.md) |
| **Storage** | The `fsspec`-backed seam that makes IO cloud-agnostic — the same Pipeline reads and writes local disk, S3, GCS, or Azure without code changes. Local Runner state is files-on-Storage, no database. | [ADR-0009](../adr/0009-fsspec-storage-seam.md) · [ADR-0010](../adr/0010-local-runner-state-is-files-on-storage.md) |

## Read next

<div class="grid cards" markdown>

- :material-sitemap: **[Architecture](architecture.md)**
  How Capability, Provider, Stage, Runner and Storage compose into a runnable Pipeline —
  the seams and why they are where they are.

- :material-alphabetical: **[Domain language](domain-language.md)**
  The project's controlled vocabulary (`CONTEXT.md`): the exact term for every concept,
  and the words to avoid.

- :material-scale-balance: **[Decision Log (ADRs)](../adr/index.md)**
  All 72 architecture decisions, in order — the reasoning behind every seam above.

</div>

## Cross-cutting invariants

Four properties every Stage and Provider must preserve — they are what make the output
trustworthy:

- **Provenance.** The immutable chain from any record back to its source (file name,
  type, size, source system, document id, page + character offsets). Every record at
  every Stage carries it; Chunk must preserve it losslessly.
- **Classification.** Descriptive content attributes (language, category, sensitivity)
  attached after Parse and inherited by every downstream record.
- **Evidence.** The record linking a knowledge-graph edge (or a Disambiguation merge) back
  to the specific source mentions and offsets that justify it, with a confidence value.
- **Quality Report.** A first-class per-run artifact: per-Stage metrics, corpus-level KG
  statistics, contract-completeness checks, and optional gold-set precision/recall — the
  substantiation of the "AI-ready" claim ([ADR-0008](../adr/0008-quality-report-as-first-class-artifact.md)).
