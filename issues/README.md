# The quality bars — how work is planned here

Planning now happens on **GitHub Issues and Discussions**; the forward-looking picture lives in
[`ROADMAP.md`](../ROADMAP.md) at the repository root. What stays in this directory is the part that
outlives any one ticket: the **bar a change has to meet** before it can merge, and the two plans that
are still forward-looking rather than history.

## The three phases the framework was built in

1. **Phase 1 — Machinery** (issues S1–S13). Build the whole spine end-to-end, thin-but-real, to the
   **Baseline bar**. Goal: prove the machinery works at enterprise-level quality, security, and
   reliability — not yet perfect, but real.
2. **Gate G — Profound testing** (issue G1). A hard stop. No Phase-2 issue started until G1's exit
   criteria were all green. The signed results are published as
   [`docs/evidence/G1-RESULTS.md`](../docs/evidence/G1-RESULTS.md).
3. **Phase 2 — Hardening loop** (issues H-core, H2–H13). One dedicated issue per stage; harden each
   to the **Hardening bar**. The remaining plan is [`PHASE-2-ROADMAP.md`](PHASE-2-ROADMAP.md); the
   provider-ecosystem half is [`PROVIDER-ECOSYSTEM-PLAN.md`](PROVIDER-ECOSYSTEM-PLAN.md).

## The three DoD tiers

Every issue names which bar applies. See `DEFINITION-OF-DONE.md`:
- **Baseline bar** — Phase-1 slices. Works, typed, tested happy-path + key failures, CI-green, secure-by-default.
- **Gate bar** — G1 only. Full validation: E2E on real corpora, security review, reliability/chaos, perf baseline, gold-set quality, determinism.
- **Hardening bar** — Phase-2 issues. Adversarial/fuzz/property tests, scale, full observability, second provider, benchmarks.

## Workflow

WIP = 1 on the critical path. An issue is done only when its DoD bar is met and CI is green — that
rule applies to a contributor's pull request exactly as it applied to the maintainer's, and
[`../CONTRIBUTING.md`](../CONTRIBUTING.md) is where the mechanics live.
