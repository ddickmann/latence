# Changelog

All notable changes to the Latence Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project (all publishable `latence-*` distributions) adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). All packages in the
workspace are versioned and released together under one version number.

## [Unreleased]

### Added

- **Reproducible environments (`deploy/env/`)** — a pinned lockfile, a one-command install recipe
  and a one-command smoke test per Provider-set slice, starting with `cpu-text` (GLiNER2 fused
  extraction, learned GLiNER PII redaction, GLinker disambiguation, Granite r2 dense embeddings,
  SPLADE sparse). The pins are **enforced constraints**: `latence env verify` cross-checks the
  generated `requirements.in`/`constraints.txt`, the solved `requirements.lock`, the Provider
  packages' real dependency declarations and the smoke stack, offline and import-free, and
  `scripts/verify-local.sh` runs it for every environment — so dependency drift that would break
  the stand-up turns the build red instead of a pod. New `latence env verify|render` commands and
  the optional `latence-core[envspec]` extra.

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
