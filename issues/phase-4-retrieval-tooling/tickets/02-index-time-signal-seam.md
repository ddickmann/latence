## Parent
Spec: retrieval-quality tooling (#158)

## What to build
The three new index-time signal Capabilities and the additive file-emission plumbing, so a corpus run can attach auxiliary retrieval signals — proven with a deterministic reference emitter; the corpus is byte-identical when nothing is wired.

## Acceptance criteria
- [ ] New `SparseEmbedder`, `MultiVectorEmbedder`, and `FdeConverter` Capabilities are declared with typed contracts (sparse term-weight map; variable-length multi-vector; multi-vector→dense).
- [ ] Export additively emits parallel sparse index/value columns, an FDE dense column, and a raw-multivector sidecar when the corresponding generator is configured; the schema is additive and versioned.
- [ ] With no generator wired, the exported corpus is byte-identical to today (ADR-0017/0049).
- [ ] A deterministic reference emitter demonstrates each emission path and is covered by Provider conformance + a `latence stack validate` run.

## Blocked by
- None — can start immediately
