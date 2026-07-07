# Contributing to the Latence Framework

Thanks for your interest. This is a model-agnostic, cloud-agnostic framework for
turning messy enterprise data into AI-ready data. It is built to a deliberately
high bar (see `issues/DEFINITION-OF-DONE.md`); contributions are held to the same
bar. This guide covers dev setup, the quality gates, and the discipline that
keeps the codebase navigable.

## Development setup — a uv workspace

The monorepo is a [uv](https://docs.astral.sh/uv/) workspace: a thin
`latence-core` plus one installable package per Provider family (ADR-0016), so an
adopter installs only what they use and the CPU-first "runs on a laptop" promise
holds (ADR-0007).

```bash
# 1. Install uv (https://docs.astral.sh/uv/getting-started/installation/)
# 2. Sync the dev environment (core + CPU demo set only — no torch, no cloud client):
uv sync

# 3. Run the demo end-to-end (CPU-only, offline):
uv run latence-demo
```

The heavy/optional Provider packages (`latence-ner-gliner`,
`latence-relation-gliner`, `latence-pii-gliner`, the `*-endpoint` packages, and
`latence-runner-airflow`) are workspace members but are **not** installed into the
default dev env: they pull `torch`/`transformers`, the `openai` client, or
`apache-airflow`, which would break the CPU-first, offline, deterministic test env
(ADR-0007/0016). Their *source* is still type-checked, and their tests run offline
by monkeypatching the heavy/endpoint dep with a fake. To work on one, install its
source without its heavy deps:

```bash
uv pip install --no-deps -e packages/latence-ner-gliner
```

## The quality gates — all must be green

CI (`.github/workflows/ci.yml`) runs exactly these. Run them locally before you
push:

```bash
# Lint + import-sort (ruff):
uv run ruff check packages

# Strict type check over BOTH src AND tests, one strict pass per package:
MYPY_CMD="uv run mypy" bash scripts/typecheck.sh

# Tests (deterministic, hash-stable):
uv run pytest

# Build every wheel + sdist:
uv build --package latence-core   # ...repeat per package, or see CI for the full list
```

Notes:

- **`scripts/typecheck.sh` runs mypy per-package, not whole-tree.** The package
  dirs are hyphenated, so a single `mypy --strict packages` maps every
  `tests/conftest.py` to the same bare `conftest` module and aborts with a
  duplicate-module error (issue #24). The per-package loop keeps each conftest
  unique in its process while still strict-checking every `src` **and** `tests`
  file.
- **Determinism is required.** Tests are seeded and hash-stable
  (`PYTHONHASHSEED=0`). A test that depends on dict/set ordering or wall-clock
  time is a bug.
- **Secret scan.** CI runs `detect-secrets` over the tree; do not commit secrets,
  `.env*`, `*.pem`, `*.key`, `.pypirc`, or credentials (all git-ignored).

## Definition of Done — pick your tier

Every change declares which bar it meets (`issues/DEFINITION-OF-DONE.md`):

- **Baseline bar** — works, strictly typed, tested (happy path **and** key
  failure modes) with ≥1 end-to-end test through the local Runner, CI-green,
  correctly packaged (heavy deps isolated), licensed, secure-by-default,
  observable, documented.
- **Gate bar** — reserved for the G1 profound-testing gate (E2E on real messy
  corpora, security review, reliability/chaos, perf baseline, gold-set quality,
  determinism fingerprint).
- **Hardening bar** — Phase-2: adversarial/property/fuzz tests, scale validation,
  full observability, ≥2 providers, benchmarks vs a named baseline.

Concretely, for a typical contribution:

- **Add a test that would FAIL if your change were reverted.** For a
  security/DoS/ReDoS fix, that means a pathological adversarial input. For a perf
  change, at least an assertion of the new bound or the wired path.
- **Preserve the contracts.** Inter-Stage data is versioned Pydantic; `Provenance`
  and `Classification` are present and validated at every boundary you touch;
  bump `SCHEMA_VERSION` on any contract change. Never break offsets,
  page-provenance, Evidence, or Delta idempotency.
- **Keep heavy deps out of `latence-core` (ADR-0016).** Core stays near-zero-dep,
  pure-Python.

## ADR + CONTEXT discipline

- **`CONTEXT.md` is the canonical language.** Use the glossary terms exactly
  (Stage, Runner, Capability, Provider, Provenance, Evidence, Quarantine, …) and
  avoid the banned synonyms it lists. New concepts get a glossary entry.
- **`docs/adr/` records the hard decisions.** If your change makes a surprising,
  non-obvious trade-off, add an ADR (numbered, dated, alternatives-considered).
  If an existing docstring or ADR overclaims relative to what the code actually
  does, correct the wording to match reality as part of your change.
- **Model licenses.** Any new default model Provider requires per-model weights
  **and** code license verification, recorded, before it ships (ADR-0012). Only
  permissive (Apache-2.0 / MIT / BSD) weights ship as defaults; restricted-license
  models are opt-in and must surface their license at install time.

## Verify-before-merge

Work is tracked as version-controlled issues under `issues/` (WIP = 1 on the
critical path; see `issues/NEXT.md`). An issue is done only when its declared DoD
bar is met and CI is green. A change is not "done" because it looks right — it is
done when the gates pass locally and in CI, and a reviewer has verified the claim
against the code, not the description.

## Legal

By contributing you agree your contributions are licensed under the project's
Apache License 2.0 (see `LICENSE`) and that you have the right to submit them.
