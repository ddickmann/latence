# Roadmap — dependency graph and critical path

## Phase 1 — Machinery (Baseline bar), strict build-through S1→S13

```
S1 walking-skeleton
└─ S2 parse
   └─ S3 chunk + screening
      ├─ S4 entity-extraction
      │  ├─ S5 relation-extraction (fused-capable)
      │  │  └─ S8 disambiguation ── S9 graph-assembly + export
      │  │        (S8 subtasks: page-map · entity-resolver · GLinker · relation-normalizer · integration)
      │  └─ S6 redaction (PII)
      └─ S7 profiling
S9 ─┬─ S10 quality-report + gold-set harness
    ├─ S11 incremental-deltas
    │     (S11 subtasks: corpus-version txn · delta-detect · affected-set recompute · reconcile · retraction+purge)
    ├─ S12 performance-path + runner-adapter
    └─ S13 demo-package
```

## Gate G — Profound testing (Gate bar) — HARD STOP

```
S1..S13 all done  ──▶  G1 profound-testing-gate  ──▶  (only if green) Phase 2
```

## Phase 2 — Hardening loop (Hardening bar), per-stage dedicated issues

```
G1 green ──▶ H-core (contracts/Runner/Storage) ──▶ H2 parse ─ H3 chunk+screening ─ H4 NER
             ─ H5 RE ─ H6 redaction ─ H7 profiling ─ H8 disambiguation ─ H9 assembly+export
             ─ H10 quality ─ H11 deltas ─ H12 scale/runners ─ H13 demo
```
Hardening order is by risk/impact, not strict linear; each Hn is blocked only by G1 (and H-core).

## Post-v1 (preserved backlog, unblocked by v1 completion)

P1 retrieval-serving · P2 compression · P3 link-prediction (ULTRA) · P4 formal-ontology · P5 source-connectors · P6 trace-scoring. Each ports catalogued IP (`../docs/ASSET-INVENTORY.md`).

## Critical path to first end-to-end KG

S1 → S2 → S3 → S4 → S5 → S8 → S9. Everything else layers onto that spine.
