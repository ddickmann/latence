#!/usr/bin/env bash
# deploy/env/gpu-learned/smoke.sh — THE one-command guard for the GPU learned-Provider environment
# (issue #198). This script IS the regression guard: dependency drift that would break the
# stand-up, and any regression to a silent CPU fallback (#191), turns it red.
#
#   bash deploy/env/gpu-learned/smoke.sh              # full: spec + stand-up + PLACEMENT + e2e
#   bash deploy/env/gpu-learned/smoke.sh --spec-only  # gate 1 only — offline, no GPU, no install
#
# FOUR GATES, cheapest first:
#
#   1. SPEC (offline, import-free, no GPU, always runs). `latence env verify` cross-checks the
#      committed requirements.in / constraints.txt / requirements.lock against the Provider
#      packages' real pyproject metadata + this environment's stack. It also enforces the
#      transformers-split resolution AHEAD of any install: a package whose declared version window
#      is disjoint from an enforced boundary is E-CONSTRAINT-CONFLICT, and an excluded package
#      (the in-process transformers>=5 OCR VLMs) named directly OR dragged in through another
#      package's workspace dependencies is E-PACKAGE-EXCLUDED. That is what "fails at dependency
#      resolution, not at runtime" means here — this gate runs BEFORE install.sh.
#
#   2. STAND-UP (needs the installed environment). `latence env verify --probe` LOADS every
#      declared Provider through the registry, which imports the heavy stack. A transformers/torch
#      major mismatch raises at module import — in seconds, before a single weight is fetched.
#
#   3. GPU PLACEMENT (needs the installed environment + a CUDA-13 GPU + weights). `latence env
#      placement` drives ONE real inference through every learned Provider and reads torch's CUDA
#      allocator around the call. A model that silently stayed on the CPU allocates ZERO CUDA bytes
#      however correct its output is — which is exactly what #191 looked like on the pod, and
#      exactly what this gate makes impossible to ship again. No CUDA device is a FAILURE, not a
#      skip.
#
#   4. END-TO-END SLICE (needs all of the above). `latence stack validate stack.yaml` runs the tiny
#      corpus through parse -> chunk -> GLiNER2 fused extraction -> BOTH learned redactors ->
#      GLinker disambiguation -> graph -> KG + RAG export with the Granite r2 GPU-tier dense leg,
#      asserting the contract / KG / RAG / report / determinism / resume checks.
#
# Gates 2-4 need the installed environment. If it is absent this script FAILS rather than reporting
# success on gate 1 alone — a smoke test that quietly narrows its own scope is a false green. Use
# --spec-only to ask for gate 1 deliberately.
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
    -h|--help) sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

# --- Gate 1: the spec (offline) --------------------------------------------------------------
step "gate 1/4 — environment spec (offline: pins, conflict boundaries, exclusions, providers, stack)"
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
  printf '\n\033[1;32mSMOKE (spec-only) GREEN — gate 1 passed; gates 2-4 not requested.\033[0m\n'
  exit 0
fi

if [ ! -x "$VENV/bin/latence" ]; then
  cat >&2 <<EOF

The gpu-learned environment is not installed at: $VENV
Gates 2-4 need it (and a CUDA-13 GPU). Either install it:

    bash $ENV_DIR/install.sh

or ask for the offline gate deliberately:

    bash $ENV_DIR/smoke.sh --spec-only
EOF
  exit 2
fi

# --- Gate 2: live stand-up -------------------------------------------------------------------
step "gate 2/4 — live stand-up (load every declared Provider: imports torch/transformers/glinker)"
"$VENV/bin/latence" env verify "$SPEC" --repo-root "$REPO_ROOT" --probe

# --- Gate 3: real GPU placement ---------------------------------------------------------------
step "gate 3/4 — GPU PLACEMENT (every learned Provider must allocate CUDA memory: the #191 guard)"
# No LATENCE_CUDA override here, deliberately. This gate must read the REAL device situation: an
# override that forced the seam's answer would be the environment telling itself what it wants to
# hear, which is the whole failure mode being guarded against.
"$VENV/bin/latence" env placement "$SPEC"

# --- Gate 4: the end-to-end slice ------------------------------------------------------------
step "gate 4/4 — tiny end-to-end slice through the installed GPU stack"
LATENCE_CUDA=1 PYTHONHASHSEED=0 "$VENV/bin/latence" stack validate "$ENV_DIR/stack.yaml"

printf '\n\033[1;32m========================================\n'
printf 'GPU LEARNED-PROVIDER ENVIRONMENT SMOKE GREEN — all 4 gates passed.\n'
printf '========================================\033[0m\n'
