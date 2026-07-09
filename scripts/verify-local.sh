#!/usr/bin/env bash
# Local CI-equivalent gate — the merge gate while GitHub Actions is unavailable.
#
# Mirrors .github/workflows/ci.yml's lint·type·test·build lane + the secret-scan gate +
# the conformance/stack-validate/bake-off gates, run LOCALLY so a merge can be gated on
# local-green when Actions minutes are exhausted. Same commands, same env, same order.
#
# NOT covered here (need containers/heavy installs — run separately when relevant):
#   - the MinIO s3:// e2e lane (needs a MinIO container)
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
for p in latence-ner-gliner latence-relation-gliner latence-relation-llm latence-pii-gliner \
         latence-pii-presidio latence-parser-endpoint latence-ner-endpoint \
         latence-embedder-endpoint latence-embedder-st latence-runner-airflow \
         latence-parser-lighton latence-parser-glm latence-linkpred-ultra; do
  [ -d "packages/$p" ] && uv pip install --no-deps -e "packages/$p" >/dev/null 2>&1 || true
done
ok "workspace synced + heavy sources installed --no-deps"

step "ruff (lint)"
uv run ruff check packages
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

step "stack validate (Goal A — default + gpu-sota + gpu-sota-pod + gpu-ocr + linkpred-base, end-to-end)"
uv run latence stack validate stacks/default.yaml
uv run latence stack validate stacks/gpu-sota.yaml
uv run latence stack validate stacks/gpu-sota-pod.yaml
uv run latence stack validate stacks/gpu-ocr.yaml
uv run latence stack validate stacks/linkpred-base.yaml   # the reference completer RUNS on CPU
ok "stack validate"

step "bake-off matrices (Goal B — the shipped CPU matrices)"
for m in parse chunk embed redaction disambiguation graph_completion; do
  [ -f "matrix/$m.yaml" ] && uv run latence bake-off "matrix/$m.yaml" --out-dir "$(mktemp -d)" >/dev/null
done
ok "bake-off"

printf '\n\033[1;32m========================================\n'
printf 'LOCAL VERIFY GREEN — all mirrored CI gates pass.\n'
printf '========================================\033[0m\n'
printf 'Not run here (separate/infra): MinIO s3 e2e, Airflow DAG lane, GPU real runs (pod).\n'
