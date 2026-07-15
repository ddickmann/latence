## Parent
Spec: retrieval-quality tooling (#158)

## What to build
An optional index-time exporter that writes BM25 postings / term-statistics as a sidecar artifact a customer can load into their store's BM25.

## Acceptance criteria
- [ ] An optional exporter emits corpus term statistics / postings as a documented sidecar artifact.
- [ ] Off by default; the corpus is byte-identical when disabled.
- [ ] Tests assert the artifact shape and that term statistics match a reference tokenization.

## Blocked by
- #160 — index-time signal seam
