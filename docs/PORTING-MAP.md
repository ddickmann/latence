# Porting Map — what to mine from the private repos, what to leave

The four private repos (latence-python, superpod, latenceai-dataset-intelligence, pipeline-worker) are **quarries** (ADR-0002). This map records, per source, what is genuinely good and worth porting, what is overengineered or vendor-coupled and should be left behind, and what is missing and must be built new. Nothing is merged with history; good parts are re-implemented deliberately into the thin-core / per-Provider structure (ADR-0016).

Legend: **KEEP** = port the idea/code into the framework · **RECAST** = good idea, but re-home behind a new seam · **DROP** = do not port · **BUILD** = net-new, no source.

## latence-python (SDK, ~14.7k LOC)

| Item | Verdict | Where it lands |
|---|---|---|
| Pydantic data models (Entity, Relation, KnowledgeGraph, DataPackage) | KEEP | Become the versioned `latence-core` contracts (ADR-0006) |
| Two-tier API, fluent PipelineBuilder + topological DAG | RECAST | Builder becomes the Python constructor for the declarative YAML Pipeline (ADR-0015) |
| Exception hierarchy, retry-with-jitter, async/sync parity | KEEP | Core utilities |
| Job resumption / checkpoint handles | KEEP | Informs the local Runner (ADR-0010) |
| Hardcoded `SERVICE_PARENT` DAG duplicated from the worker | DROP | Replaced by the Pipeline declaration as single source of truth |
| Coupling to hosted `api.latence.ai` + B2 presigned URLs | DROP | Framework is not a client of a hosted service; Storage seam (ADR-0009) |
| Deprecated-property shim layer | DROP | New package, no legacy surface |

## superpod (GPU multi-service pod, ~48k LOC)

| Item | Verdict | Where it lands |
|---|---|---|
| The 10 model integrations (PaddleOCR-VL, GLiNER extraction, span-predictor RE, PII, embeddings) | RECAST | Each becomes a Provider of a Capability (ADR-0004); decoupled from the pod |
| GPU phase-serialization (SIGSTOP/SIGCONT), colocated-vLLM startup | DROP | Existed to amortize RunPod serverless pricing — not an adopter problem (ADR-0007) |
| vLLM pooling-plugin patches (extra_kwargs passthrough) | RECAST | Only needed for the GPU "performance" path; custom encoder also runs via plain transformers/ONNX (ADR-0013) |
| BaseService retry / transient-error handling | KEEP | Provider base utilities |
| Result-trimming for RunPod's 20MB payload cap | DROP | Storage-based IO has no payload cap |
| Env-driven config discipline (defaults match Dockerfile) | KEEP | Style carries into Provider config |
| ColBERT/ColPali retrieval, trace/groundedness, compression | DROP (from v1) | Post-v1 Provider plugins (ADR-0005) |
| `text_processing` Rust wheel + Python fallback | KEEP | The template for Rust hot-path acceleration (ADR-0015) |

## latenceai-dataset-intelligence (corpus KG, ~22.7k src LOC)

| Item | Verdict | Where it lands |
|---|---|---|
| PageOffsetIndex / page_mapper (offset→page, fuzzy drift recovery) | KEEP | Core to Provenance integrity; a Rust hot-path candidate |
| 5-strategy entity resolution + GLinker linking pipeline | KEEP | The Disambiguation Stage's reference Provider |
| Graph builder with EvidenceRecord provenance | PORTED (S9) | The Graph Assembly Stage — `latence_core.graph.GraphBuilder` (content-addressed ids, endpoint validation, Evidence links) behind the `graph.canonical` Provider; KG Export as Parquet+TTL+GraphML (ADR-0027) |
| Write-ahead log + atomic state updates | KEEP | Backs the local Runner's files-on-Storage state (ADR-0010) AND transactional Corpus Versions (ADR-0018) |
| delta_detector (content-hash new/updated/unchanged) | KEEP | Delta computation (ADR-0018) |
| graph/patcher (retract edges by source_document_ids) | KEEP | Evidence-based Retraction/Purge (ADR-0018) |
| incremental_linker (link delta vs existing KB) | RECAST | Good for attach-to-known; extend to affected-set merge/split (ADR-0018) |
| ontology/incremental, link_prediction/incremental_trainer (RotatE warm-start) | DROP (from v1) | Ride with the post-v1 link-prediction/ontology plugins (ADR-0005) |
| Feature disk-spilling under memory pressure | KEEP | Profiling Stage |
| Tier-based config routing (Tier 1/2/3) | RECAST | Becomes Pipeline declarations, not bespoke handler logic |
| 9 enrichers as a registry | RECAST | Consolidate; the registry is heavier than a fixed set needs |
| RotatE link prediction (transductive, no quality metrics, unbounded training) | DROP (from v1) | Post-v1 plugin; unvalidated |
| Formal ontology induction (concept_inducer, SHACL/OWL, hierarchy learning) | DROP (from v1) | Least-validated code; possible dead paths; post-v1 plugin |
| Governance + Alignment modules (~700 LOC) | DROP | Speculative enterprise "compliance theater" |
| FacilityRisk-specific pipeline (~800 LOC) | DROP | One-off domain module; belongs in an examples repo if anywhere |
| vLLM-vs-sentence-transformers dual embedding backend | RECAST | Two Providers of one Embedder Capability, not a branch inside one module |

## pipeline-worker (orchestrator, ~7.7k LOC)

| Item | Verdict | Where it lands |
|---|---|---|
| Stage-first batch streaming, DAG waves, topological sort | KEEP | The local Runner's execution model (ADR-0003) |
| Per-stage checkpoint/resume to storage | KEEP | Local Runner (ADR-0010) |
| Tenacity retry strategy, cancellation, boot watchdog | KEEP | Local Runner resilience |
| Markup-stripping / output-transformation contracts | KEEP | Stage IO normalization |
| Hallucination-cleaning heuristics for VLM output | RECAST | A parse post-processing option, Provider-scoped |
| Supabase job-state coupling | DROP | Files-on-Storage state (ADR-0010) |
| Backblaze-B2 boto3 client, hardcoded endpoints | DROP | fsspec Storage seam (ADR-0009) |
| RunPod endpoint lifecycle / ref-counted leases / per-service GPU rates | DROP | RunPod-specific; not in an adopter-agnostic Runner |
| Billing / credits / Discord + portal webhooks | DROP | Product-ops concerns, not framework concerns |
| Legacy BETA_TESTING super-pod path | DROP | Dead code |

## Missing across all repos — BUILD new

- **Screening Stage** (intake + content: malware/zip-bomb/type-spoof, prompt-injection, harmful, sensitivity) — none of the repos have it (ADR: CONTEXT `Screening`).
- **Enforced Provenance + Classification** on every record at every boundary — today it's convention, not contract (ADR-0006).
- **Capability protocols** as explicit, typed, documented seams — today models are wired directly (ADR-0004).
- **fsspec Storage seam** — today it's Backblaze-B2-specific (ADR-0009).
- **Quality Report + gold-set harness** — today quality is asserted via unmeasured thresholds (ADR-0008).
- **CONTEXT.md + ADRs + tests-as-spec** — none of the four repos have CONTEXT.md or ADRs; test coverage is thin and integration-only.
- **Per-Provider packaging with isolated deps** — today it's one fat environment (ADR-0016).
- **Cloud/runner adapters** (Databricks, Airflow) behind the Runner seam (ADR-0003).
