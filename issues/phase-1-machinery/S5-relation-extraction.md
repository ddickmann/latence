# S5 — Relation Extraction (fused-capable)

**Parent**: S4 · **Phase 1 · Baseline bar**

## What to build
A **Relation Extraction** Capability that finds typed relations between entity mentions within a document. The seam must allow a **Fused Provider** that fulfils both Entity Extraction and Relation Extraction in one pass (ADR-0013) — the Pipeline must not force one Provider per Stage. Ship two candidate reference Providers and let the choice be empirical: (1) an LLM-based extractor over any OpenAI-compatible endpoint (incl. local Ollama), (2) `knowledgator/gliner-relex-multi-v1.0` (Apache-2.0, fused NER+RE).

## Ports (algorithmic assets)
- None to port; the author's custom span-predictor encoder is a later performance Provider (decoupled from superpod's vLLM pooling patch).

## Acceptance criteria
- [ ] RelationExtractor Capability protocol defined; both reference Providers implemented (licenses verified).
- [ ] A single Fused Provider can satisfy S4+S5 in one pass; the Runner routes correctly without double-extracting.
- [ ] Relations carry head/tail mention refs, type, confidence, and Provenance.
- [ ] Provider is selectable per Pipeline via one config change; swapping LLM↔gliner-relex requires no pipeline edit.
- [ ] Relation counts + confidence land in the Quality Report.
- [ ] Baseline bar met.

## Blocked by
S4.
