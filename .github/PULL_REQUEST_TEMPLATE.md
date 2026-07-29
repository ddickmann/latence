<!--
Thanks for contributing. This repo is held to a deliberately high bar
(issues/DEFINITION-OF-DONE.md) and speaks a canonical language (CONTEXT.md).
Read CONTRIBUTING.md if you have not — the CLA requirement and the doc-truth
gates are the two things that surprise people.

Fill in the summary and tick every box that applies. An unchecked box is fine if
you explain why in the summary — a silently-skipped gate is not.
-->

## Summary

<!-- What changes and why. Link the issue(s): "Closes #123". Use CONTEXT.md terms
     (Stage, Runner, Capability, Provider, Provenance, Evidence, Quarantine, …). -->

## Licence

- [ ] **I can sign the Contributor Licence Agreement for this contribution.** Every merged contribution requires one, because the licence reserves commercial rights to the owner and the owner must hold sufficient rights to grant them. If your employer may own this work, say so here.

## Definition of Done

<!-- Declare the bar this change meets (issues/DEFINITION-OF-DONE.md). -->

- [ ] Baseline bar (works, strictly typed, tested happy-path **and** key failure modes, ≥1 end-to-end test, packaged, licensed, secure-by-default, documented)
- [ ] Gate bar (G1 profound testing)
- [ ] Hardening bar (Phase-2: adversarial / property / fuzz / scale)

## The test that was red

<!-- Name it. `packages/<pkg>/tests/test_x.py::test_y` -->

- [ ] **A test that FAILED before this change now passes.** Not merely a test that exercises the new code — one that is red on the base commit. For a security/DoS/ReDoS fix, a pathological adversarial input; for a perf change, an assertion of the new bound or that the faster path is taken.
- [ ] **Anti-false-green:** any fake or stub used in tests mirrors the real API's shape and rejects what the real one rejects, so a pass reflects real behaviour rather than a fake that always agrees. Heavy/endpoint Providers are exercised with the real dep monkeypatched by a faithful fake.

## Gates

```bash
bash scripts/verify-local.sh > /tmp/verify.log 2>&1; echo "REAL_EXIT=$?"
```

- [ ] **`scripts/verify-local.sh` is GREEN**, and I captured the exit code by **redirect**, not by piping to `tail`. (A pipeline returns its *last* command's status and will report green on a red tree.) `REAL_EXIT=` <!-- paste it -->
- [ ] **`PYTHONHASHSEED=0 uv run pytest <paths> -o addopts=""`** — seed pinned at launch (a conftest guard refuses the run otherwise); `addopts` cleared so the summary line is not swallowed by the implicit `-qq`.
- [ ] **`uv run ruff check packages`** clean.
- [ ] **`MYPY_CMD="uv run mypy" bash scripts/typecheck.sh`** clean — `--strict` over `src` **and** `tests`, one pass per package.
- [ ] **No new nondeterminism.** Nothing added depends on dict/set ordering or wall-clock time.

## Contracts, packaging, licences

- [ ] **Contracts preserved.** Inter-Stage data stays versioned Pydantic; `Provenance`/`Classification`/`Evidence` present and validated at every boundary touched; `SCHEMA_VERSION` bumped on any contract change; offsets, page-provenance, and Delta idempotency intact.
- [ ] **Heavy deps stay out of `latence-core`** (ADR-0016); core stays near-zero-dep, pure-Python; the CPU-first / offline / deterministic env is intact (ADR-0007).
- [ ] **New Provider?** It lives in its own package, declares a `ProviderProfile`, registers via `[project.entry-points."latence.providers"]` — **no `latence-core` edit** — and passes C1–C6 conformance.
- [ ] **Licence verified** for any new model or dependency — **weights AND code verified separately** (ADR-0012), recorded in `NOTICE` / `THIRD-PARTY-LICENSES.md`. Only permissive weights ship as defaults; restricted-licence models are opt-in and surface their licence and flow-down obligations.

## Security, secrets, PII

- [ ] **No secrets committed** (`detect-secrets` gate green); no secret written to manifests or logs; **no real PII** in fixtures, tests, or this PR.
- [ ] **Untrusted-input surface?** If this touches a Parser, a Screener, or anything reading ingested bytes, the adversarial input is in the test suite.
- [ ] **Redaction surface?** The universal financial-PII floor still cannot be bypassed by Provider choice or narrowed by config, and its regexes still match the in-core rule Redactor's built-ins.

## Docs and decisions

- [ ] **Doc-truth gates green.** Docs that make claims about code are machine-checked (links, section anchors, counts, shipped-vs-aspirational, licence summaries). If I added a claim to a published page, I added or updated the gate that keeps it true.
- [ ] **ADR added/updated** if this makes a surprising, non-obvious, or hard-to-reverse decision (`docs/adr/`, numbered + dated + **alternatives considered and why rejected**). A superseded ADR is amended in place with a pointer forward, never deleted.
- [ ] **Overclaims corrected.** Any docstring or ADR that claimed more than the code does has been tightened to match reality.
- [ ] **Docs updated** (README / CONTEXT / relevant `docs/`), and the CONTEXT.md glossary respected — no banned synonyms; new concepts get a glossary entry.

## Not run locally (if applicable)

<!-- verify-local.sh does not cover: the MinIO s3:// e2e lane, the Airflow DAG lane,
     GPU-provider real runs (validated on the CUDA pod), or the deploy/env gates that
     need the installed heavy venv. Note anything you could not exercise, and why. -->
