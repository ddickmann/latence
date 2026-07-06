# S10 — Quality Report + gold-set harness

**Parent**: S9 · **Phase 1 · Baseline bar**

## What to build
Promote the Quality Report from skeleton to the full first-class artifact (ADR-0008): per-Stage metrics, corpus-level KG statistics, contract-completeness checks, Screening/Quarantine summary, and Disambiguation merge decisions with Evidence. Add a **gold-set harness**: an adopter drops in 20–50 annotated documents and gets precision/recall per Stage on their own corpus.

## Ports (algorithmic assets)
- None; new. May reuse calibration ideas from the link-prediction calibrator conceptually.

## Acceptance criteria
- [ ] Quality Report is a versioned artifact emitted every run, covering all Stages the pipeline ran.
- [ ] Gold-set harness computes per-Stage precision/recall/F1 against provided annotations and writes them into the report.
- [ ] Report includes contract-completeness (Provenance/Classification present, offsets aligned) and drift diagnostics.
- [ ] Report is machine-readable (JSON) and human-readable (rendered summary).
- [ ] Baseline bar met.

## Blocked by
S9.
