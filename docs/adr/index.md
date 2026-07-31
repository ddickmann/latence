# Decision Log (ADRs)

The **Architecture Decision Records** are the reasoning behind every seam in the framework —
each recording the context, the choice, and the alternatives rejected. They
are the authoritative source; the [Concepts](../concepts/index.md) and
[Architecture](../concepts/architecture.md) pages are navigational summaries over them.

!!! note
    ADRs are numbered by decision order. Number 0040 was never issued (no gap in reasoning —
    the sequence simply skipped it), so the log holds 57 records numbered 0001–0058.

## Foundations & scope

| ADR | Decision |
| --- | --- |
| [0001](0001-open-source-core-owned-personally.md) | Open-source core, personally owned, published before the Deloitte start date |
| [0002](0002-clean-monorepo-mined-from-private-repos.md) | Clean-history monorepo; private repos are quarries, not ancestors |
| [0005](0005-v1-scope-the-ai-ready-data-spine.md) | v1 scope: the AI-ready-data spine, nothing else |
| [0015](0015-python-core-declarative-yaml-rust-for-hot-paths.md) | Python core, declarative-YAML Pipeline contract, Rust only for proven hot paths |
| [0055](0055-relicence-to-polyform-noncommercial.md) | Relicence to PolyForm Noncommercial — free for research, commercial by agreement (supersedes 0001's Apache-2.0; amends 0012, 0023) |

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

## Optional Stages

| ADR | Decision |
| --- | --- |
| [0037](0037-graph-completion-capability-optional-post-assembly-stage-and-inferred-edge-representation.md) | Graph Completion is an OPTIONAL post-Assembly Capability; predicted edges are inferred-marked |
| [0038](0038-schema-induction-optional-llm-stage-induced-labels-contract-and-additive-extractor-seam.md) | Schema Induction is an OPTIONAL LLM Stage; induced labels are additive |
| [0039](0039-context-enrichment-optional-stage-kg-projected-onto-chunks-embedding-input-only.md) | Context Enrichment projects the KG onto chunks into the EMBEDDING input only |
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

## Empirical choices

| ADR | Decision |
| --- | --- |
| [0013](0013-relation-extraction-provider-chosen-empirically-fused-capability-allowed.md) | Relation Extraction Provider is chosen empirically; a Provider may fulfil fused Capabilities |
| [0046](0046-enterprise-glinker-entity-linking-and-audited-resolver.md) | Enterprise GLinker entity-linking + audited resolver, ported from dataset-intelligence |

## Retrieval-quality tooling (the query-side layer)

The stateless, downstream layer that feeds an adopter's retrieval engine without being one (design;
see [Retrieval-quality tooling](../retrieval-tooling.md)).

| ADR | Decision |
| --- | --- |
| [0048](0048-retrieval-tooling-is-strictly-stateless.md) | Retrieval tooling is strictly stateless — it generates signals and transforms candidates, holds no index |
| [0049](0049-clock-split-index-time-signals-extend-the-pipeline-query-time-is-a-downstream-library.md) | Clock-split: index-time signals extend the pipeline; query-time is a downstream library with its own `latence.retrieval` seam |
| [0050](0050-multi-vector-and-muvera-fde-ship-as-a-real-but-experimental-off-by-default-stub.md) | Multi-vector / MUVERA-FDE ships real-but-experimental, off by default, behind a stable seam |
| [0051](0051-zero-ops-multi-hop-graph-retrieval-via-duckdb-over-the-emitted-graph-parquet.md) | Zero-ops multi-hop graph retrieval via DuckDB over the emitted graph parquet; Neo4j optional |
| [0052](0052-retrieval-mcp-ports-memory-engine-hardening-with-config-driven-stack-adaptive-copy.md) | Retrieval MCP ports memory_engine hardening; bring-your-own backend; config-driven stack-adaptive copy |
| [0053](0053-splade-sparse-signal-is-a-structured-composition-of-model-ner-and-metadata.md) | SPLADE sparse is a structured composition (model ⊕ confidence-gated NER ⊕ metadata), redacted by default |
| [0056](0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md) | Hypergraph Evidence sidecar is read per call over a point-lookup KV; no global edge-adjacency matrix |
| [0057](0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md) | A Hyperedge is a third GraphScope emitted by Graph Assembly; `tau_text` follows the corpus redaction variant |
| [0058](0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md) | `token_cost` is an approximate index-time weight; the budget guarantee is an exact query-time verification pass |
