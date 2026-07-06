# S7 — Profiling

**Parent**: S3 · **Phase 1 · Baseline bar**

## What to build
A corpus-level **Profiling** Capability that computes statistical and quality features over the corpus: per-document (density, readability, Zipf α, compression ratio, structure) and cross-document (entity frequency, co-occurrence, type consensus, source coverage). Consolidate the private stack's 9 enrichers into a lean, fixed feature set (drop the heavier registry indirection).

## Ports (algorithmic assets)
- The 9 enrichers (`dataset-intelligence` enrichment modules) → consolidate into one `document_features` + `corpus_features` module. RECAST (simplify registry).

## Acceptance criteria
- [x] Profiling Capability emits per-document and cross-document `FeatureRecord`s tied to Provenance.
- [x] Features are computed streaming/spill-to-disk friendly (no whole-corpus-in-RAM assumption for the feature pass).
- [x] Feature summary lands in the Quality Report.
- [x] Runs CPU-only within the demo budget on the sample corpus.
- [x] Baseline bar met.

## Blocked by
S3.
