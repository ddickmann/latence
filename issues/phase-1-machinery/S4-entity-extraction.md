# S4 — Entity Extraction

**Parent**: S3 · **Phase 1 · Baseline bar**

## What to build
An **Entity Extraction** Capability that finds typed entity mentions in chunked text (zero-shot NER), with one CPU-viable reference Provider. Mentions carry offsets and Provenance back to the source page.

## Ports (algorithmic assets)
- None to port (new); reference Provider wraps a permissively-licensed GLiNER checkpoint via plain `transformers`/ONNX (CPU path, no vLLM).

## Acceptance criteria
- [ ] EntityExtractor Capability protocol defined; ≥1 CPU reference Provider (license verified).
- [ ] Emits typed mentions with confidence, char offsets, and Provenance resolving to the correct source page.
- [ ] Label set is configurable per Pipeline (zero-shot); no label set hardcoded in core.
- [ ] Confidence distribution + per-type counts land in the Quality Report.
- [ ] Empty/near-empty chunk handled without error.
- [ ] Baseline bar met.

## Blocked by
S3.
