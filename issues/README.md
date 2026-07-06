# Issue tracker — how this works

This directory is the framework's execution plan as version-controlled, independently-grabbable issues (vertical "tracer-bullet" slices). It is the source of truth now; it mirrors to GitHub Issues when the repo is published.

## Three phases (do not interleave)

1. **Phase 1 — Machinery** (`phase-1-machinery/`, issues S1–S13). Build the whole spine end-to-end, thin-but-real, to the **Baseline bar**. Goal: prove the machinery works at enterprise-level quality, security, and reliability — not yet perfect, but real.
2. **Gate G — Profound testing** (`gate/`, issue G1). A hard stop. No Phase-2 issue starts until G1's exit criteria are all green.
3. **Phase 2 — Hardening loop** (`phase-2-hardening/`, issues H-core, H2–H13). One dedicated issue per stage; harden each to the **Hardening bar**.

Post-v1 assets are preserved in `backlog-post-v1/` (P1–P6) — scheduled plugins, not discarded (see `../docs/ASSET-INVENTORY.md`).

## The three DoD tiers

Every issue names which bar applies. See `DEFINITION-OF-DONE.md`:
- **Baseline bar** — Phase-1 slices. Works, typed, tested happy-path + key failures, CI-green, secure-by-default.
- **Gate bar** — G1 only. Full validation: E2E on real corpora, security review, reliability/chaos, perf baseline, gold-set quality, determinism.
- **Hardening bar** — Phase-2 issues. Adversarial/fuzz/property tests, scale, full observability, second provider, benchmarks.

## Issue format

Each file: **Parent** · **Phase / DoD bar** · **What to build** (end-to-end behavior, no premature file paths) · **Ports** (algorithmic assets to reuse) · **Acceptance criteria** (checkboxes) · **Blocked by**.

## Workflow

WIP = 1 on the critical path (see `NEXT.md`). An issue is done only when its DoD bar is met and CI is green. Phase-1 order is strictly S1→S13 (build-through); within S8 and S11, do the internal sub-tasks in listed order.
