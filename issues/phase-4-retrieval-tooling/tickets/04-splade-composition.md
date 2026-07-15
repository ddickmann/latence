## Parent
Spec: retrieval-quality tooling (#158)

## What to build
A real multilingual SPLADE `SparseEmbedder` whose vector is the model output composed with confidence-gated NER entity terms and selected metadata terms, generated over redacted text by default, emitted as the sparse columns.

## Acceptance criteria
- [ ] A license-verified (weights and code, separately) multilingual sparse model is used; the model is config-selectable and none is hardcoded before license clearance.
- [ ] The sparse vector = model term-weights unioned with confidence-gated entity surface terms and selected metadata terms, with configurable boosts (term-injection default; weighted mode optional).
- [ ] Input text follows the corpus redaction policy by default; entity terms below an extraction-confidence threshold are excluded (ADR-0053).
- [ ] Emits the parallel sparse index/value columns; tested with a faithful fake model, with real-model validation done out of band.

## Blocked by
- #160 — index-time signal seam
