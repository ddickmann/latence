#!/usr/bin/env bash
# deploy/env/cpu-text/smoke.sh — THE one-command guard for the CPU/text enterprise-sota
# environment (issue #195). This script IS the regression guard: dependency drift that would
# break the stand-up turns it red.
#
#   bash deploy/env/cpu-text/smoke.sh              # full: spec gate + live stand-up + e2e slice
#   bash deploy/env/cpu-text/smoke.sh --spec-only  # gate 1 only — offline, no install needed
#
# THREE GATES, cheapest first:
#
#   1. SPEC (offline, import-free, always runs). `latence env verify` cross-checks the committed
#      requirements.in / constraints.txt / requirements.lock against the Provider packages' real
#      pyproject metadata + the smoke stack. A dependency added, a floor raised, a cap dropped, a
#      Provider renamed, or an enforced conflict boundary that stops holding — each is a finding
#      and a non-zero exit. This gate is what scripts/verify-local.sh and the test suite run, so
#      drift is caught on every commit, on any machine, with no heavy install.
#
#   2. STAND-UP (needs the installed environment). `latence env verify --probe` LOADS every
#      declared Provider through the registry, which imports the heavy stack. A transformers/torch
#      major mismatch raises at module import — this gate catches it in seconds, before a single
#      weight is fetched. It also asserts `disambiguation.glinker` imports, which the CPU end-to-
#      end slice cannot exercise (compute="gpu"; its runtime slice is #198).
#
#   3. END-TO-END SLICE (needs the installed environment + model weights). `latence stack
#      validate stack.yaml` runs the tiny corpus through parse -> chunk -> GLiNER2 fused
#      extraction -> learned PII redaction -> disambiguation -> graph -> KG export + RAG export
#      with BOTH the dense Granite r2 leg and the SPLADE sparse leg, asserting the contract /
#      KG / RAG / report / determinism / resume / graceful-failure checks.
#
# Gates 2 and 3 need the environment install.sh built. If it is absent this script FAILS rather
# than reporting success on gate 1 alone — a smoke test that quietly narrows its own scope is a
# false green. Use --spec-only to ask for gate 1 deliberately.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../.." && pwd)"
VENV="${LATENCE_ENV_VENV:-$ENV_DIR/.venv}"
SPEC="$ENV_DIR/environment.yaml"
SPEC_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --spec-only) SPEC_ONLY=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

# --- Gate 1: the spec (offline) --------------------------------------------------------------
step "gate 1/3 — environment spec (offline: pins, conflict boundaries, providers, stack)"
# Resolution order matters: the environment's own venv first, then the WORKSPACE (`uv run`, so a
# checkout is verified against its own source), and only then a `latence` that happens to be on
# PATH — a stale global install must never decide whether this gate passes.
if [ -x "$VENV/bin/latence" ]; then
  "$VENV/bin/latence" env verify "$SPEC" --repo-root "$REPO_ROOT"
elif [ -f "$REPO_ROOT/pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  ( cd "$REPO_ROOT" && uv run latence env verify "$SPEC" --repo-root "$REPO_ROOT" )
elif command -v latence >/dev/null 2>&1; then
  latence env verify "$SPEC" --repo-root "$REPO_ROOT"
else
  echo "no 'latence' available (install the env venv, or run from the workspace with uv)" >&2
  exit 2
fi

if [ "$SPEC_ONLY" = "1" ]; then
  printf '\n\033[1;32mSMOKE (spec-only) GREEN — gate 1 passed; gates 2+3 not requested.\033[0m\n'
  exit 0
fi

if [ ! -x "$VENV/bin/latence" ]; then
  cat >&2 <<EOF

The cpu-text environment is not installed at: $VENV
Gates 2 and 3 need it. Either install it:

    bash $ENV_DIR/install.sh

or ask for the offline gate deliberately:

    bash $ENV_DIR/smoke.sh --spec-only
EOF
  exit 2
fi

# --- Gate 2: live stand-up -------------------------------------------------------------------
step "gate 2/3 — live stand-up (load every declared Provider: imports torch/transformers/glinker)"
"$VENV/bin/latence" env verify "$SPEC" --repo-root "$REPO_ROOT" --probe

# --- Gate 3: the end-to-end slice ------------------------------------------------------------
step "gate 3/3 — tiny end-to-end slice through the installed stack"
# LATENCE_CUDA=0 pins the device seam deterministically: this environment ships CPU-only torch,
# so a probe that guessed otherwise would fabricate a GPU decision.
LATENCE_CUDA=0 PYTHONHASHSEED=0 "$VENV/bin/latence" stack validate "$ENV_DIR/stack.yaml"

printf '\n\033[1;32m========================================\n'
printf 'CPU/TEXT ENVIRONMENT SMOKE GREEN — all 3 gates passed.\n'
printf '========================================\033[0m\n'
