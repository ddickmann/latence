# S5 — Relation Extraction (fused-capable)

**Parent**: S4 · **Phase 1 · Baseline bar**

## What to build
A **Relation Extraction** Capability that finds typed relations between entity mentions within a document. The seam must allow a **Fused Provider** that fulfils both Entity Extraction and Relation Extraction in one pass (ADR-0013) — the Pipeline must not force one Provider per Stage. Ship two candidate reference Providers and let the choice be empirical: (1) an LLM-based extractor over any OpenAI-compatible endpoint (incl. local Ollama), (2) `knowledgator/gliner-relex-multi-v1.0` (Apache-2.0, fused NER+RE).

## Ports (algorithmic assets)
- None to port; the author's custom span-predictor encoder is a later performance Provider (decoupled from superpod's vLLM pooling patch).

## Acceptance criteria
- [x] RelationExtractor Capability protocol defined; both reference Providers implemented (licenses verified). — `RelationExtractor` + `FusedEntityRelationExtractor` protocols in `latence_core.capability`; Providers `relation.llm` (LLM/OpenAI-compatible, incl. Ollama) and `relation.gliner_relex` (fused gliner-relex), plus the in-core deterministic `relation.pattern`. Licenses verified Apache-2.0 both ways (ADR-0012/0023, 2026-07-06).
- [x] A single Fused Provider can satisfy S4+S5 in one pass; the Runner routes correctly without double-extracting. — `CapabilityKind.FUSED_ENTITY_RELATION`; the fused Stage emits `[*mentions, *relations]` and runs *instead of* a separate Entity-Extraction Stage. Proven by `test_fused_provider_satisfies_s4_and_s5_in_one_pass` (asserts no `entity_extraction` Stage ran).
- [x] Relations carry head/tail mention refs, type, confidence, and Provenance. — `RelationMention` contract (`head_mention_id`/`tail_mention_id`/`head_text`/`tail_text`/`label`/`confidence` + a covering-span Provenance enforced by the validator).
- [x] Provider is selectable per Pipeline via one config change; swapping LLM↔gliner-relex requires no pipeline edit. — entry-point `provider:` name in the Stage; the same seam, one-line swap (ADR-0004).
- [x] Relation counts + confidence land in the Quality Report. — `RelationQuality` (per-type counts + `ConfidenceHistogram` + offset-resolution invariant), wired in the Runner's report; asserted in `test_relation_slice_end_to_end`.
- [x] Baseline bar met. — ruff + mypy strict + pytest (217 passed / 1 pre-existing S3-MinIO skip) green; deterministic across hash seeds; wheels build; ADR-0023 added.

## Blocked by
S4.
