#!/usr/bin/env bash
# Local CI-equivalent gate — the merge gate while GitHub Actions is unavailable.
#
# Mirrors .github/workflows/ci.yml's lint·type·test·build lane + the secret-scan gate +
# the conformance/stack-validate/bake-off gates, run LOCALLY so a merge can be gated on
# local-green when Actions minutes are exhausted. Same commands, same env, same order.
#
# NOT covered here (need containers/heavy installs — run separately when relevant):
#   - the S3-emulator s3:// e2e lane (needs a versitygw container)
#   - the Airflow DAG lane (needs apache-airflow installed under its constraints)
#   - GPU-provider real runs (validated on the CUDA pod)
# Those are called out in the summary so nothing is silently skipped.
#
# Usage: bash scripts/verify-local.sh   (from the repo root)
# Exit 0 = all local gates green; non-zero = the first failing gate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0 LATENCE_CUDA=0

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }
ok() { printf '\033[1;32m✓ %s\033[0m\n' "$1"; }

step "secret-scan gate (detect-secrets, volatile-field-immune)"
uv run --with detect-secrets --no-project python scripts/secret_scan_gate.py
ok "secret-scan"

step "sync workspace (dev env)"
uv sync
# Heavy/endpoint provider SOURCES only (no heavy deps) — their tests monkeypatch the heavy
# dep; keeps the CPU-first offline env intact (ADR-0007/0016). Mirrors the CI step exactly.
#
# `--no-deps` is also what lets BOTH gliner2 majors' tests run in ONE venv (ADR-0064):
# latence-extract-gliner2 / latence-pii-gliner2 pin gliner2>=1.3,<2 and latence-gliner25 pins
# >=2.0,<3, which no resolver can satisfy together — and neither is resolved here, because neither
# library is installed at all. Each package's tests inject their own faithful fake. That is also
# why latence-gliner25 is excluded from the uv workspace (see the root pyproject.toml): `uv lock`
# DOES resolve the members together, and correctly refuses the pair.
while read -r p; do
  [ -d "packages/$p" ] && uv pip install --no-deps -e "packages/$p" >/dev/null 2>&1 || true
done < <(grep -vE '^[[:space:]]*(#|$)' scripts/heavy-packages.txt)
ok "workspace synced + heavy sources installed --no-deps"

# Editable installs must resolve into THIS checkout. The venv is shared, and a session that
# ran the install loop from a git worktree left three latence_* packages resolving into
# .claude/worktrees/<name>/ — so tests here imported (and mypy checked) a DIFFERENT tree than
# the one under test, silently. The loop above repairs the paths as a side effect of running
# from this root; this step is what proves the repair instead of trusting it, and catches any
# latence_* module the loop does not cover.
step "editable installs resolve into this checkout"
uv run python - <<'PYEOF'
import importlib.util, pathlib, sys
root = pathlib.Path.cwd().resolve()
stray = {}
for finder in list(pathlib.Path(p) for p in sys.path if "site-packages" in p):
    for pth in finder.glob("_editable_impl_latence_*.pth"):
        name = pth.stem.removeprefix("_editable_impl_")
        spec = importlib.util.find_spec(name)
        if spec and spec.origin and not pathlib.Path(spec.origin).resolve().is_relative_to(root):
            stray[name] = spec.origin
if stray:
    lines = "\n".join(f"  {k} -> {v}" for k, v in sorted(stray.items()))
    sys.exit(f"editable installs resolve OUTSIDE this checkout — tests would validate the "
             f"wrong tree:\n{lines}\nfix: uv pip install --no-deps -e packages/<name>")
PYEOF
ok "all latence_* editables resolve under $(pwd)"

# The lint scope is CI's, verbatim. This script's whole promise is "what CI will say, before
# you push", and a narrower scope here breaks that promise silently: the campaign scripts and
# scripts/ were unlinted locally while CI checked them. test_verify_local_mirrors_ci_lint_scope
# pins the two together so they cannot drift apart again.
step "ruff (lint)"
uv run ruff check packages benchmark/s5 benchmark/serving benchmark/sota-campaign scripts
ok "ruff"

step "mypy --strict (src + tests, per-package)"
MYPY_CMD="uv run mypy" bash scripts/typecheck.sh
ok "mypy strict"

step "pytest (full suite)"
uv run pytest -q
ok "pytest"

step "provider conformance (every registered provider, LATENCE_CUDA=0)"
uv run pytest packages/latence-core/tests/test_conformance.py \
              packages/latence-core/tests/test_e2e_conformance.py -q
ok "conformance"

step "stack validate (Goal A — default + gpu-sota + gpu-sota-pod + gpu-ocr + linkpred-base + schema-induction, end-to-end)"
uv run latence stack validate stacks/default.yaml
uv run latence stack validate stacks/gpu-sota.yaml
uv run latence stack validate stacks/gpu-sota-pod.yaml
uv run latence stack validate stacks/gpu-sota-glinker.yaml  # ADR-0046 enterprise GLinker; disambiguate skip-with-flag on CPU
uv run latence stack validate stacks/gpu-ocr.yaml
uv run latence stack validate stacks/universal-parse.yaml   # W5 any-file parser; Parse skip-with-flag on CPU
uv run latence stack validate stacks/linkpred-base.yaml   # the reference completer RUNS on CPU
uv run latence stack validate stacks/schema-induction.yaml  # induce+extract skip-with-flag; CPU spine validates (W4)
uv run latence stack validate stacks/context-enrichment.yaml  # the KG→chunk enrich Stage RUNS on CPU (W13)
ok "stack validate"

step "reproducible-environment specs (#195 — pins, enforced conflict boundaries, providers)"
# The OFFLINE gate of every deploy/env/<name>/ environment: `smoke.sh --spec-only`. It needs no
# heavy install and no network, and it is what turns dependency drift in a Provider package RED
# on every commit. It also enforces the #198 transformers-split resolution BEFORE any install:
# a package whose declared window is disjoint from an enforced boundary, or an excluded
# in-process OCR package entering the set, fails right here. The remaining gates — standing the
# environment up, PROVING real GPU placement, and the tiny end-to-end slice — need the installed
# venv (and a GPU); see deploy/env/README.md.
for env in deploy/env/*/smoke.sh; do
  [ -f "$env" ] && bash "$env" --spec-only
done
ok "environment specs"

step "Blackwell/CUDA-13 compatibility matrix (#199 — every cell measured or explicitly UNVERIFIED)"
# Offline and import-free, like the env specs. It fails when a cell claims 'verified' without a
# date/hardware/command/observed measurement, when an UNVERIFIED cell has no runnable reproduce
# command, when a learned Provider or serving dependency of a committed environment has no cell at
# all, when a recorded version drifts from that environment's lock, when one of the live run's
# failure modes leaves the matrix, or when the published docs/compatibility-matrix.md is not the
# render of the YAML. The `--require-verified` variant (which also fails while anything is still
# UNVERIFIED) is deliberately NOT run here — it is the maintainer's switch ON the Blackwell host.
uv run latence env matrix deploy/env/compatibility-matrix.yaml
ok "compatibility matrix"

step "bake-off matrices (Goal B — the shipped CPU matrices)"
for m in parse chunk embed redaction disambiguation graph_completion schema_induction; do
  [ -f "matrix/$m.yaml" ] && uv run latence bake-off "matrix/$m.yaml" --out-dir "$(mktemp -d)" >/dev/null
done
ok "bake-off"

printf '\n\033[1;32m========================================\n'
printf 'LOCAL VERIFY GREEN — all mirrored CI gates pass.\n'
printf '========================================\033[0m\n'
printf 'Not run here (separate/infra): S3-emulator s3 e2e, Airflow DAG lane, GPU real runs (pod),\n'
printf '  deploy/env/* gates 2+ (need the installed heavy venv, and a CUDA-13 GPU for the\n  gpu-learned placement gate: bash deploy/env/<name>/smoke.sh).\n'
