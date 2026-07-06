# Algorithmic Asset Inventory — enterprise IP to preserve, not reinvent

This catalogs the non-trivial **algorithms** in the four private repos that carry real enterprise value (correctness, scale, robustness) and would be costly to rebuild. "Deferred to post-v1" here means **preserved and scheduled**, never discarded. Each asset is assigned to the v1 slice or the post-v1 plugin that will port it. Reinvention estimate for the whole set: ~4–6 months senior ML + 2–3 months infra.

Verdicts: **PORT-AS-IS** (algorithm is sound; re-home behind the new seam) · **PORT-WITH-HARDENING** (validate/benchmark first) · **EXTRACT-ALGORITHM-ONLY** (keep the idea, drop the impl — use a std library or upstream it).

## v1 assets (land in the spine)

| Asset | Source file | LOC | Technique | Ready | Reinvent | Verdict | Lands in |
|---|---|---|---|---|---|---|---|
| PageOffsetIndex | dataset-intelligence `page_mapper.py` | 500 | Cumulative page boundaries + `bisect` O(log n) offset→page, fuzzy drift-recovery fallback, full drift diagnostics, never-crash degradation | 5/5 | MED | PORT-AS-IS | S2/S8 (Provenance integrity) |
| EntityResolver | `resolution/entity_resolver.py` | 350 | 5-strategy cascade (exact→alias→acronym→substring→embedding) + Union-Find merge map + confidence-weighted, audited merge log (no silent over-merge) | 5/5 | MED | PORT-AS-IS | S8 Disambiguation |
| RelationNormalizer | `relations/normalizer.py` | 250 | GLinker-style relation L2 resolution: fuzzy+alias, batched embedding search, type filtering, inverse detection + direction swap | 4.5/5 | MED | PORT-AS-IS | S8 Disambiguation |
| GLinker linking pipeline | `linking/backends.py` | 1233 | L2 candidate-gen → L3 biencoder → L4 rerank → L0 threshold, precomputed embeddings, KB-text fallback | 4.5/5 | HIGH | PORT-AS-IS | S8 Disambiguation |
| Delta Detector | `state/delta_detector.py` | 200 | Deterministic SHA-256 over (doc_id, markdown, entity texts, relation labels); manifest diff → new/updated/unchanged/deleted | 5/5 | MED | PORT-AS-IS | S11 Deltas |
| Write-Ahead Log | `state/write_ahead_log.py` | 200 | Temp-write → manifest-write → atomic rename; crash recovery via MANIFEST.json | 5/5 | MED | PORT-AS-IS | S1 (state) / S11 (Corpus Version txns) |
| Graph patcher | `graph/patcher.py` | 177 | Retract/add edges by `source_document_ids` provenance; vocab expansion on serialized snapshot | 5/5 | MED | PORT-AS-IS | S11 affected-set recompute |
| Streaming orchestrator (algorithm) | pipeline-worker `orchestrator.py` | 2563 | Stage-first concurrent streaming, semaphore backpressure, non-blocking IO overlap, LLM chunk budgeting, pre-flight validation, drift compensation, 18-pattern markup stripping | 4.5/5 | HIGH | PORT-AS-IS (strip RunPod/B2) | S1 local Runner + S3 Chunk |
| Retry w/ jitter + error classing | pipeline-worker `retry_utils.py` | 97 | Exp backoff + jitter; 5xx/429 retryable vs 4xx terminal | 5/5 | LOW | EXTRACT-ALGORITHM-ONLY (use `tenacity`) | S1 Runner |

## Post-v1 assets (PRESERVED — scheduled as plugins, not discarded)

| Asset | Source file | LOC | Technique | Ready | Reinvent | Verdict | Lands in |
|---|---|---|---|---|---|---|---|
| ConceptInducer | dataset-intelligence `ontology/concept_inducer.py` | 620 | 8-step ontology induction: agglomerative embedding clustering, Union-Find concept merge (>0.98 cos) with parent-child fixup, relation signatures, quality scoring | 4.5/5 | HIGH | PORT-AS-IS | P4 Ontology plugin |
| Taxonomy learner | `ontology/taxonomy.py` | 400 | 3-signal hierarchy: distributional inclusion + embedding hypernym + lexical patterns (Hearst'92, Roller'18), seed-augmented | 4/5 | HIGH | PORT-AS-IS | P4 Ontology plugin |
| AxiomInducer | `ontology/axiom_inducer.py` | 600 | OWL2 axioms: disjointness (<5% overlap), functional (1:1), domain/range (distribution), property chains (2-hop), restrictions (participation ratio) | 4/5 | MED-HIGH | PORT-AS-IS | P4 Ontology plugin |
| Link prediction stack | `link_prediction/{scorer,ultra_scorer,candidate_generator,calibrator}.py` | 876 | **ULTRA** (ICLR'24, inductive/zero-shot) + PyKEEN behind one scorer abstraction; type-aware tiered 2-hop candidate gen w/ recall guarantee; temperature scaling + precision-targeted thresholds | 5/5 | HIGH | PORT-AS-IS | P3 Link-prediction plugin |
| ColBERT pooler | superpod `vllm_plugins/colbert_plugin/pooler_v2.py` | 250 | Late-interaction pooling: 768→128 projection, per-token L2, query padding, punctuation skiplist, vLLM API shims | 4/5 | MED-HIGH | PORT-AS-IS | P1 Retrieval plugin |
| Fused CUDA kernels | `colbert_plugin/custom_kernels/*` | 300 | Fused RoPE+local-attn, GLU+matmul, dropout+residual | 4/5 | HIGH | PORT-WITH-HARDENING (validate vs torch.compile) | P1 Retrieval plugin |
| Qwen3 compression | `vllm_plugins/qwen3_compression_plugin/model.py` | 400 | Context compression via learned token merging | 4/5 | MED | PORT-WITH-HARDENING | P2 Compression plugin |

## Do-not-port (replace with standard practice)

- superpod `patches/vllm_pooling_extra_kwargs.py` (2/5, fragile monkey-patch) — upstream to vLLM or use a supported hook.
- latence-python `_client.py` HTTP/HMAC/retry — standard httpx patterns; the framework is not a client of a hosted API.

## Correction to the scope framing

ADR-0005 deferred ontology induction, link prediction, retrieval serving, and compression from v1. The earlier architecture pass described the ontology code as "least-validated / weakest link" — the algorithmic inventory refutes that (4–4.5/5, HIGH reinvention cost, plus an overlooked ULTRA inductive scorer). The v1 scope boundary stands, but the rationale is **sequencing and dependency surface, not asset quality**. These assets are preserved verbatim here and scheduled as post-v1 plugins P1–P4.
