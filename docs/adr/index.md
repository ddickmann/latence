# Decision Log (ADRs)

The **Architecture Decision Records** are the reasoning behind every seam in the framework —
each recording the context, the choice, and the alternatives rejected. They
are the authoritative source; the [Concepts](../concepts/index.md) and
[Architecture](../concepts/architecture.md) pages are navigational summaries over them.

!!! note
    ADRs are numbered by decision order. Number 0040 was never issued (no gap in reasoning —
    the sequence simply skipped it), so the log holds 71 records numbered 0001–0072.

## Re-scope: the semantic-resolution and ontology engine

The direction since 2026-11-08: a traceable semantic and ontology layer built from warehouse
metadata and unstructured text, standalone and as a sidecar to Metis. The route is charted as a
wayfinder map in `.scratch/semantic-sidecar/map.md`.

| ADR | Decision |
| --- | --- |
| [0067](0067-latence-is-re-scoped-into-a-semantic-resolution-and-ontology-engine-standalone-and-as-a-metis-sidecar.md) | Latence is re-scoped into a semantic-resolution and ontology engine that runs standalone and as a sidecar to Metis; warehouse metadata is a new input modality; the text path and Graph Completion stay (partly supersedes the scope framing of 0005 and 0017) |
| [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) | Query-time retrieval tooling, hyperedges and Context Enrichment are removed; embedders stay only as a resolution signal (supersedes 0039, 0048–0053, 0056–0059; narrows 0045) |
| [0069](0069-the-semantic-layer-is-a-replay-of-an-append-only-ledger-of-assertions-and-decisions.md) | The semantic layer is a replay of an append-only ledger of Assertions and Decisions; two levels (Entities may be auto-accepted in the calibrated band, Concepts are proposals a human approves); Bindings are keyed on Unity Catalog ids |
| [0070](0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md) | Metis consumes the engine through one reviewable proposal file; everything else reaches it through Unity Catalog. No shared code, one route per source for new measures, and no importer plugin or live tool calls in v1 |
| [0071](0071-warehouse-metadata-enters-the-pipeline-as-sources-and-row-values-are-matched-inside-the-workspace.md) | Warehouse readers are new Source kinds in the one Pipeline and meet the text path in one resolution phase; row values are matched inside the customer's workspace; filters and masks are labelled and masked data excluded; signals come in three tiers with a hosted referee off by default |
| [0072](0072-on-databricks-the-engine-runs-as-jobs-in-the-workspace-and-keeps-its-state-in-delta.md) | On Databricks the engine runs as jobs in the customer's workspace and keeps the ledger, results and review state in Delta tables in Unity Catalog; the review surface is an in-workspace app with a local mode; neural models run on GPU job compute; review explanations are built in, hosted ones opt-in |

## Foundations & scope

| ADR | Decision |
| --- | --- |
| [0001](0001-open-source-core-owned-personally.md) | Open-source core, personally owned, published before the Deloitte start date |
| [0002](0002-clean-monorepo-mined-from-private-repos.md) | Clean-history monorepo; private repos are quarries, not ancestors |
| [0005](0005-v1-scope-the-ai-ready-data-spine.md) | v1 scope: the AI-ready-data spine, nothing else |
| [0015](0015-python-core-declarative-yaml-rust-for-hot-paths.md) | Python core, declarative-YAML Pipeline contract, Rust only for proven hot paths |
| [0055](0055-relicence-to-polyform-noncommercial.md) | ~~Relicence to PolyForm Noncommercial — free for research, commercial by agreement~~ — **SUPERSEDED by [0063](0063-relicence-to-apache-2-0.md)**; kept as the dated record of a decision that was made and reversed |
| [0062](0062-distribution-is-signed-github-releases-only-no-pypi-under-polyform-noncommercial.md) | Distribution is signed GitHub Releases only — no PyPI — **rationale SUPERSEDED by [0063](0063-relicence-to-apache-2-0.md), decision REOPENED**; the `pypi-publish` job stays disabled pending a new decision |
| [0063](0063-relicence-to-apache-2-0.md) | Relicence to Apache-2.0 — the licence the intended adopter is allowed to install (supersedes 0055, restores 0001's grant; reopens 0062) |

## The core seams

| ADR | Decision |
| --- | --- |
| [0003](0003-stage-library-with-pluggable-runners.md) | Stage library with pluggable Runners, not an orchestration control plane |
| [0004](0004-capability-protocols-with-provider-plugins.md) | Capability protocols with provider plugins as the model-agnosticity seam |
| [0006](0006-one-schema-dual-serialization.md) | One schema, dual serialization (JSONL + Parquet/Arrow) |
| [0009](0009-fsspec-storage-seam.md) | fsspec-backed Storage seam for cloud-agnostic IO |
| [0010](0010-local-runner-state-is-files-on-storage.md) | Local Runner state is files-on-Storage, no database |
| [0016](0016-thin-core-per-provider-packages.md) | Thin core plus per-Provider packages, managed as a uv workspace |
| [0017](0017-ai-ready-deliverable-is-rag-corpus-plus-kg-as-files.md) | The AI-ready deliverable is a RAG corpus **and** a knowledge graph, both as files |
| [0033](0033-streaming-memory-bounded-runner.md) | Streaming, memory-bounded local Runner |
| [0034](0034-observability-typed-errors-and-thin-tracing-seam.md) | Observability: typed error taxonomy and a thin, OTel-free tracing seam |
| [0047](0047-observability-export-seam-prometheus-otel-optional.md) | A metrics EXPORT seam (Prometheus / OpenTelemetry) layered on ADR-0034, behind an optional extra |
| [0032](0032-strict-mypy-on-tests-and-the-conftest-module-collision.md) | Strict mypy is enforced on tests too, and the whole-tree conftest module collision is resolved |

## Providers, licensing & the ecosystem

| ADR | Decision |
| --- | --- |
| [0007](0007-cpu-first-reference-providers.md) | CPU-first reference Providers; GPU serving is an upgrade path |
| [0012](0012-permissive-only-reference-providers-gemma-opt-in.md) | Reference Providers ship permissive weights only; restricted models are opt-in |
| [0035](0035-second-cpu-reference-providers-for-single-provider-capabilities.md) | A second CPU reference Provider for each single-provider Capability |
| [0036](0036-provider-ecosystem-and-bakeoff.md) | Provider ecosystem: adapters, device profiles, conformance, and the bake-off harness |
| [0030](0030-performance-path-endpoint-providers-and-second-runner-adapter.md) | Performance path is endpoint Providers; the second Runner reuses the Stage engine |
| [0045](0045-granite-r2-default-embedder-gpu-311m-fa2-bf16-cpu-97m.md) | IBM Granite Embedding r2 is the default learned Embedder |
| [0065](0065-the-h-g1-dependency-cap-is-a-workspace-wide-default-with-recorded-exemptions.md) | The H-G1 dependency cap is a workspace-wide default over every third-party pin (extras included), with machine-read, reason-bearing exemptions |
| [0066](0066-publication-and-release-are-separate-gates.md) | Publication and release are separate gates — the source is published as a pre-1.0 preview while Gate 1 (measured end-to-end numbers) stays open; a `v*` tag still requires every gate (amends `RELEASE-GATES.md`) |

## The Stages (contracts per Stage)

| ADR | Decision |
| --- | --- |
| [0014](0014-source-seam-thin-connectors.md) | Source Capability seam; v1 ships only filesystem/object-store connectors |
| [0019](0019-parser-input-contract.md) | Parse consumes a raw ParserInput, not a pre-decoded DocumentRecord |
| [0020](0020-parse-page-map-and-error-disposition.md) | Parse carries a page map and an error disposition on the record contract |
| [0021](0021-chunk-offsets-and-screening-two-checkpoints.md) | Chunk preserves offsets; Screening is two checkpoints with Quarantine off the DAG |
| [0022](0022-entity-mention-record-and-zero-shot-provider-split.md) | Entity mentions are first-class Records; the zero-shot label set lives in config |
| [0023](0023-relation-mention-record-and-fused-stage-routing.md) | A RelationMention is a first-class Record; a fused Stage emits both carriers |
| [0024](0024-redaction-record-and-metadata-aware-policy.md) | Redaction is one RedactionRecord per document, keyed off Classification.sensitivity |
| [0025](0025-profiling-feature-record-corpus-level-stage-and-fixed-feature-set.md) | Profiling is a corpus-level Stage emitting scoped FeatureRecords over a fixed set |
| [0026](0026-disambiguation-canonical-entity-scoped-record-ported-resolver-cascade-and-cpu-viable-linking.md) | Disambiguation emits scoped records over a CPU-viable resolver cascade |
| [0027](0027-graph-assembly-scoped-graph-record-ported-builder-and-kg-export-as-files.md) | Graph Assembly emits scoped GraphRecords; Export writes the KG as files |
| [0031](0031-entity-extraction-emits-true-original-offsets-via-carried-strip-offset-map.md) | Entity Extraction emits TRUE original offsets via a carried strip offset map |
| [0042](0042-chunk-level-redaction-fixes-the-document-truncation-leak.md) | Redaction is a chunk→chunk transform (fixes the document-truncation PII leak) |
| [0044](0044-universal-redaction-floor-and-optional-mask-before-extract.md) | A universal financial-PII redaction floor, and an optional mask-before-extract mode |
| [0060](0060-page-boundaries-cross-the-ocr-handoff-in-a-sidecar-and-an-absent-page-map-is-a-value-not-a-page-1.md) | Page boundaries cross the OCR handoff in a sidecar; an absent page map is a value, not a `page 1` |
| [0064](0064-gliner-2-5-is-a-new-provider-package-behind-an-enforced-gliner2-major-boundary.md) | GLiNER 2.5 is a NEW Provider package behind an enforced `gliner2` major boundary; `chunk.page` is page-preferred, 1024 otherwise |

## Optional Stages

| ADR | Decision |
| --- | --- |
| [0037](0037-graph-completion-capability-optional-post-assembly-stage-and-inferred-edge-representation.md) | Graph Completion is an OPTIONAL post-Assembly Capability; predicted edges are inferred-marked |
| [0038](0038-schema-induction-optional-llm-stage-induced-labels-contract-and-additive-extractor-seam.md) | Schema Induction is an OPTIONAL LLM Stage; induced labels are additive |
| [0039](0039-context-enrichment-optional-stage-kg-projected-onto-chunks-embedding-input-only.md) | ~~Context Enrichment projects the KG onto chunks into the EMBEDDING input only~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0054](0054-type-vocabulary-canonicalization-is-a-capability-of-the-resolution-phase.md) | Type-vocabulary canonicalization is its own Capability in the resolution phase (per-chunk induction's downstream fix) |

## The Quality Report, deltas & operations

| ADR | Decision |
| --- | --- |
| [0008](0008-quality-report-as-first-class-artifact.md) | Quality Report as a first-class run artifact |
| [0028](0028-quality-report-promoted-gold-set-harness-is-a-runner-side-evaluation-keyed-by-file-name.md) | The Quality Report is promoted; the gold-set harness is a Runner-side evaluation |
| [0011](0011-demo-is-one-command-plus-live-provider-swap.md) | Demo: one command plus a live provider swap |
| [0018](0018-incremental-corpus-deltas.md) | Incremental corpus deltas: affected-set recompute, drift-triggered reconcile, dual-mode deletion |
| [0029](0029-delta-runner-affected-set-via-blocking-keys-and-source-exclusion-deletion.md) | Delta orchestration: DeltaRunner wraps the LocalRunner; affected-set via blocking keys |
| [0041](0041-guided-setup-and-process-cli-opinionated-quality-baked-config-and-one-toggle.md) | Guided setup is a deterministic wizard emitting ONE opinionated, quality-baked config |
| [0043](0043-incremental-extraction-frozen-schema-and-doc-level-cross-run-reuse.md) | Incremental extraction: a frozen induced schema + doc-level cross-run reuse |
| [0061](0061-incremental-ingest-exact-corpus-recompute-over-an-order-preserving-reuse-union.md) | Incremental ingest: exact corpus-level recompute over an order-preserving reuse union (enrichment pipelines included) |

## Empirical choices

| ADR | Decision |
| --- | --- |
| [0013](0013-relation-extraction-provider-chosen-empirically-fused-capability-allowed.md) | Relation Extraction Provider is chosen empirically; a Provider may fulfil fused Capabilities |
| [0046](0046-enterprise-glinker-entity-linking-and-audited-resolver.md) | Enterprise GLinker entity-linking + audited resolver, ported from dataset-intelligence |

## Retrieval-quality tooling (the query-side layer) — superseded by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)

The stateless, downstream layer that fed an adopter's retrieval engine without being one (design;
see [Retrieval-quality tooling](../retrieval-tooling.md)). Every record in this section is
**superseded by ADR-0068** and kept as the dated record.

| ADR | Decision |
| --- | --- |
| [0048](0048-retrieval-tooling-is-strictly-stateless.md) | ~~Retrieval tooling is strictly stateless — it generates signals and transforms candidates, holds no index~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0049](0049-clock-split-index-time-signals-extend-the-pipeline-query-time-is-a-downstream-library.md) | ~~Clock-split: index-time signals extend the pipeline; query-time is a downstream library with its own `latence.retrieval` seam~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0050](0050-multi-vector-and-muvera-fde-ship-as-a-real-but-experimental-off-by-default-stub.md) | ~~Multi-vector / MUVERA-FDE ships real-but-experimental, off by default, behind a stable seam~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0051](0051-zero-ops-multi-hop-graph-retrieval-via-duckdb-over-the-emitted-graph-parquet.md) | ~~Zero-ops multi-hop graph retrieval via DuckDB over the emitted graph parquet; Neo4j optional~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0052](0052-retrieval-mcp-ports-memory-engine-hardening-with-config-driven-stack-adaptive-copy.md) | ~~Retrieval MCP ports memory_engine hardening; bring-your-own backend; config-driven stack-adaptive copy~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0053](0053-splade-sparse-signal-is-a-structured-composition-of-model-ner-and-metadata.md) | ~~SPLADE sparse is a structured composition (model ⊕ confidence-gated NER ⊕ metadata), redacted by default~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0056](0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md) | ~~Hypergraph Evidence sidecar is read per call over a point-lookup KV; no global edge-adjacency matrix~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0057](0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md) | ~~A Hyperedge is a third GraphScope emitted by Graph Assembly; `tau_text` follows the corpus redaction variant~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0058](0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md) | ~~`token_cost` is an approximate index-time weight; the budget guarantee is an exact query-time verification pass~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
| [0059](0059-query-side-entry-is-tiered-zero-model-default-with-gliner2-strictly-opt-in.md) | ~~Query-side entry is tiered: zero-model default over the alias KV; GLiNER2 query NER strictly opt-in~~ — **SUPERSEDED by [0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md)** |
