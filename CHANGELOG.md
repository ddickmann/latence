# Changelog

All notable changes to the Latence Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project (all publishable `latence-*` distributions) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All packages in the
workspace are versioned and released together under one version number.

## [Unreleased]

### Added

- **`latence-sink-qdrant`** — the first first-party **sink**: one command between "pipeline
  finished" and "my retrieval engine serves the corpus" (#183, the first slice of #156).
  `latence-sink-qdrant load ./out/export --collection corpus --url …` reads the Export's
  `records.parquet` (streamed in bounded batches), creates the collection if absent at the corpus's
  own vector width, and upserts with the `embedding` column as the vector and every remaining column
  as payload.
  - **Idempotent** — point ids are a deterministic UUIDv5 of the content-addressed `record_id` (a
    raw `record_id` string is not a legal Qdrant point id), and a stored content digest means an
    unchanged record is **not** re-upserted, so a re-run or an incremental/Delta run
    ([ADR-0018](docs/adr/0018-incremental-corpus-deltas.md)) writes only what changed.
  - **Retraction/Purge honoured** — records the Export no longer contains are deleted by a
    reconciling sweep, and the load **counts the stale set back to zero before reporting success**;
    `latence-sink-qdrant verify-absent <record_id>…` is the operator-side audit step (exit 1 while
    any named record is still there). `--no-reconcile` covers the incremental-Export case.
  - A store fault mid-load aborts **before** the sweep, so a half-uploaded corpus can never delete
    the records it did not reach.
  - The Export stays the source of truth ([ADR-0017](docs/adr/0017-ai-ready-deliverable-is-rag-corpus-plus-kg-as-files.md));
    the sink holds no index and runs no search ([ADR-0048](docs/adr/0048-retrieval-tooling-is-stateless.md)).
  - `latence-core` takes **no** new dependency: the mapping/idempotency/reconcile core is pure-Python
    over `latence-core`, and `qdrant-client` (Apache-2.0) rides the optional `[client]` extra,
    imported lazily. Ships a faithful `FakeQdrantClient` that rejects what a real Qdrant rejects
    (404/409/403/400 incl. non-UUID point ids and wrong-width vectors), plus an opt-in live smoke
    test (`LATENCE_QDRANT_URL`).

## [0.1.0] - 2026-07-14

First public, coherent release of the workspace. All publishable distributions
(`latence-core` plus the provider packages) are pinned to `0.1.0` and licensed
Apache-2.0.

### Added

- **The pipeline spine** — a model-agnostic, cloud-agnostic set of Stage contracts
  and a local Runner that turn messy, siloed source data into AI-ready outputs:
  Source → Parse/OCR → Chunk → Screening → Entity/Relation Extraction →
  Schema Induction → Redaction (PII) → Profiling → Disambiguation (entity linking /
  GLinker-style resolution) → Graph Assembly → Graph Completion → Context Enrichment →
  Export (Parquet · JSONL · TTL · GraphML). Every Stage declares typed input/output
  contracts and is fulfilled by a swappable Provider discovered via entry points.

- **`latence-core`** — data contracts, Capability protocols, the Pipeline + local
  Runner, the `latence` CLI (`stack validate`, `bake-off`), the Storage seam
  (`file://` and `s3://` via fsspec), and the near-zero-dependency built-in reference
  Providers (local-folder source, markdown/sentence chunkers, gazetteer entities,
  pattern relations, hybrid-rule redaction, statistical/lightweight profiling, cascade
  and exact-surface disambiguation, canonical/weighted graph assembly, reference graph
  completion, KG-header context enrichment, hashing embedder, JSONL/Parquet + KG export).

- **The provider ecosystem** — each heavy/endpoint model isolated behind a Stage seam in
  its own package so `latence-core` stays thin (ADR-0016):
  - _Parse / OCR_ — `latence-parser-plaintext`, `latence-parser-document`,
    `latence-parser-pdfplumber`, `latence-parser-render` (universal any-file, license-clean),
    `latence-parser-lighton`, `latence-parser-lighton-vllm`, `latence-parser-glm`,
    `latence-parser-endpoint`.
  - _Entities / Relations_ — `latence-ner-gliner`, `latence-ner-endpoint`,
    `latence-relation-gliner`, `latence-relation-llm`, `latence-extract-gliner2`,
    `latence-schema-inducer`.
  - _PII / Redaction_ — `latence-pii-gliner`, `latence-pii-gliner2`, `latence-pii-presidio`.
  - _Embedders_ — `latence-embedder-st`, `latence-embedder-endpoint`.
  - _Disambiguation_ — `latence-disambig-embedding`, `latence-disambig-glinker`.
  - _Graph completion_ — `latence-linkpred-ultra`.
  - _Runners_ — `latence-runner-airflow`.
  - _Demo_ — `latence-demo` (end-to-end showcase).

- **Enterprise-readiness gates** — a Provider Conformance suite parametrized over every
  registered Provider (contract, graceful failure, license declaration, determinism),
  the Goal-A end-to-end stack validation on a bundled corpus, and the Goal-B per-Stage
  bake-off matrices (quality/latency/memory/cost/license/determinism/error-rate).

- **Packaging & metadata** — every publishable package carries full `[project]` metadata
  (description, readme, `Apache-2.0` license, authors, keywords, PyPI classifiers marked
  `Development Status :: 4 - Beta` and `Typing :: Typed`, and Homepage/Repository/Issues/
  Changelog URLs).

### Engineering

- **CI** split into a fast required lane (`ci.yml`: secret-scan, lint+type fast gate,
  test matrix on py3.11/3.12 with surfaced coverage, Goal-A stack validation, all-package
  build) and a heavy lane (`ci-heavy.yml`: MinIO `s3://` e2e, real-Airflow DAG tests, and
  the Goal-B bake-offs) that also runs nightly. Actions are pinned by commit SHA with
  least-privilege `permissions`, per-ref `concurrency`, and docs-only `paths-ignore` on the
  heavy lane. `scripts/verify-local.sh` mirrors the gates for offline, deterministic local
  verification.

- **Release engineering** — `release.yml` builds all distributions and cuts a GitHub
  Release with generated notes on a `v*` tag; a PyPI Trusted-Publishing (OIDC) job is
  scaffolded, environment-gated, and disabled pending the maintainer's go-ahead
  (see `docs/RELEASING.md`).

[Unreleased]: https://github.com/ddickmann/latence-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ddickmann/latence-framework/releases/tag/v0.1.0
