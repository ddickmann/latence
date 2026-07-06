# S11 — Incremental deltas

**Parent**: S9 · **Phase 1 · Baseline bar**

## What to build
The lean-yet-enterprise delta machinery (ADR-0018). A **Delta** is a content-hash diff of the Source manifest against the last committed **Corpus Version**. Doc-level Stages re-run only on added/updated docs; deleted docs' records are retracted. Corpus-level Stages recompute only the **affected set** (blocking neighborhoods touched by changed mentions + neighbors), catching merge-on-add and split-on-delete. A drift metric triggers full **Reconciliation**. Deletion is dual-mode: **Retraction** (soft tombstone, audit-retained) by default and **Purge** (physical GDPR erase) on request. Every delta run is a transaction producing an immutable Version N+1.

## Internal sub-tasks (do in order)
1. **Corpus Version transaction** — WAL-backed read-N → write-N+1, atomic manifest swap, idempotent re-run, rollback/inspection.
2. **Delta detection** — content-hash manifest diff → added/updated/deleted/unchanged.
3. **Affected-set recompute** — blocking-neighborhood recompute for Disambiguation; graph patch by `source_document_ids`.
4. **Drift-triggered reconciliation** — cheap drift metric + threshold → full recompute.
5. **Retraction + Purge** — soft tombstone (excluded from exports/KG/RAG, retained) and hard purge (source + derived + Evidence removed) + reconcile.

## Ports (algorithmic assets — all PORT-AS-IS)
- Delta Detector (`state/delta_detector.py`, 5/5) · Write-Ahead Log (`state/write_ahead_log.py`, 5/5) · Graph patcher (`graph/patcher.py`, 5/5). IncrementalLinker RECAST → extend to affected-set merge/split.

## Acceptance criteria
- [ ] Adding a file updates the corpus touching only the affected set; a bridging doc merges two prior clusters.
- [ ] Deleting a file (Retraction) tombstones its records, excludes them from exports/KG/RAG, retains them for audit, and splits any merge justified solely by it.
- [ ] Purge physically removes source + all derived records + Evidence, then reconciles; verified nothing derived survives.
- [ ] Drift threshold triggers a full Reconciliation; the trigger and outcome are reported.
- [ ] Re-running the same Delta is idempotent; a failed run leaves Version N intact; any prior Version is inspectable.
- [ ] Delta churn (docs ±, entities created/merged/split, edges ±, drift) lands in the Quality Report.
- [ ] Baseline bar met.

## Blocked by
S9.
