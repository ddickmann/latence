<!--
Thanks for contributing. This repo is held to a deliberately high bar
(issues/DEFINITION-OF-DONE.md) and speaks a canonical language (CONTEXT.md).
Fill in the summary and tick every box that applies. An unchecked box is fine
if you explain why in the summary — a silently-skipped gate is not.
-->

## Summary

<!-- What changes and why. Link the issue(s): "Closes #123". Use CONTEXT.md terms
     (Stage, Runner, Capability, Provider, Provenance, Evidence, Quarantine, …). -->

## Definition of Done

<!-- Declare the bar this change meets (issues/DEFINITION-OF-DONE.md). -->

- [ ] Baseline bar (works, strictly typed, tested happy-path **and** key failure modes, ≥1 end-to-end test, packaged, licensed, secure-by-default, documented)
- [ ] Gate bar (G1 profound testing)
- [ ] Hardening bar (Phase-2: adversarial / property / fuzz / scale)

## Checklist

- [ ] **`bash scripts/verify-local.sh` is GREEN** (mirrors the CI lint · type · test · build · conformance · stack-validate · bake-off gates locally).
- [ ] **Tests added that would FAIL if this change were reverted.** For a security/DoS/ReDoS fix, a pathological adversarial input; for a perf change, an assertion of the new bound.
- [ ] **Anti-false-green:** any fake/stub used in tests faithfully mirrors the real API (shape-matching), so a passing test reflects real behaviour — not a fake that always agrees. Heavy/endpoint Providers are exercised with the real dep monkeypatched by a faithful fake.
- [ ] **Contracts preserved.** Inter-Stage data stays versioned Pydantic; `Provenance`/`Classification`/`Evidence` present and validated at every boundary touched; `SCHEMA_VERSION` bumped on any contract change; offsets, page-provenance, and Delta idempotency intact.
- [ ] **Heavy deps stay out of `latence-core`** (ADR-0016); core stays near-zero-dep, pure-Python; CPU-first/offline/deterministic env intact (ADR-0007).
- [ ] **ADR added/updated** if this makes a surprising, non-obvious, or hard-to-reverse decision (`docs/adr/`, numbered + dated + alternatives-considered). Overclaiming docstrings/ADRs corrected to match the code.
- [ ] **License verified** for any new model or dependency — **weights AND code verified separately** (ADR-0012), recorded (NOTICE / THIRD-PARTY-LICENSES.md). Only permissive weights ship as defaults; restricted-license models are opt-in and surface their license at install time.
- [ ] **Secrets & PII:** no secrets committed (`detect-secrets` gate green); no secret written to manifests/logs; no real PII in fixtures or the PR.
- [ ] **Docs updated** (README/CONTEXT/relevant `docs/`), and the CONTEXT.md glossary respected (no banned synonyms; new concepts get a glossary entry).

## Not run locally (if applicable)

<!-- verify-local.sh does not cover: the MinIO s3:// e2e lane, the Airflow DAG lane,
     or GPU-provider real runs (validated on the CUDA pod). Note anything you could
     not exercise and why. -->
