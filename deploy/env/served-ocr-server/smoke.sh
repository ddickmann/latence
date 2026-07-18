#!/usr/bin/env bash
# deploy/env/served-ocr-server/smoke.sh — THE one-command guard for the vLLM LightOn-OCR serving
# environment (#196). This script IS the regression guard: drift that would break the stand-up,
# or an exclusion that has quietly stopped being enforced, turns it red.
#
#   bash deploy/env/served-ocr-server/smoke.sh              # all three gates
#   bash deploy/env/served-ocr-server/smoke.sh --spec-only  # gate 1 only — offline, no install
#
# THREE GATES, cheapest first:
#
#   1. SPEC (offline, import-free, always runs). `latence env verify` cross-checks the committed
#      requirements.in / constraints.txt / requirements.lock against this environment's declared
#      requirements, its enforced boundaries and its EXCLUSIONS. An excluded distribution that
#      appears in the lock is E-EXCLUSION-VIOLATED; a boundary that stopped holding is
#      E-CONSTRAINT-VIOLATED. scripts/verify-local.sh runs this gate on every commit.
#
#   2. EXCLUSION ENFORCEMENT (needs network, no install). Asks the REAL resolver for each excluded
#      distribution under the committed constraints and requires it to REFUSE, then re-solves the
#      real requirements and requires it to SUCCEED. This is the gate that proves the exclusion is
#      a pin and not a wish — and that it is not so blunt it breaks the environment it protects.
#
#   3. STAND-UP (needs the installed environment). Imports the serving stack, asserts torch is the
#      CUDA-13 build the lock pins, and runs verify-exclusions.py against the live interpreter —
#      absence proved by import, not by reading a manifest.
#
# Gates 2 and 3 need network / the installed venv. If either is absent this script FAILS rather
# than reporting success on gate 1 alone — a smoke test that quietly narrows its own scope is a
# false green. Use --spec-only to ask for gate 1 deliberately.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../.." && pwd)"
VENV="${LATENCE_ENV_VENV:-$ENV_DIR/.venv}"
SPEC="$ENV_DIR/environment.yaml"
SPEC_ONLY=0
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"

while [ $# -gt 0 ]; do
  case "$1" in
    --spec-only) SPEC_ONLY=1; shift ;;
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

# --- Gate 1: the spec (offline) --------------------------------------------------------------
step "gate 1/3 — environment spec (offline: pins, boundaries, exclusions)"
# Resolution order matters: the WORKSPACE first (`uv run`, so a checkout is verified against its
# own source), then a `latence` that happens to be on PATH. This serving venv holds no latence
# package at all — that is the point of a service environment — so it is not consulted.
if [ -f "$REPO_ROOT/pyproject.toml" ] && command -v uv >/dev/null 2>&1; then
  ( cd "$REPO_ROOT" && uv run latence env verify "$SPEC" --repo-root "$REPO_ROOT" )
elif command -v latence >/dev/null 2>&1; then
  latence env verify "$SPEC" --repo-root "$REPO_ROOT"
else
  echo "no 'latence' available (run from the workspace with uv, or install latence-core)" >&2
  exit 2
fi

if [ "$SPEC_ONLY" = "1" ]; then
  printf '\n\033[1;32mSMOKE (spec-only) GREEN — gate 1 passed; gates 2+3 not requested.\033[0m\n'
  exit 0
fi

command -v uv >/dev/null 2>&1 || { echo "uv is required for gates 2+3." >&2; exit 2; }

# --- Gate 2: the exclusions are pins, not wishes ----------------------------------------------
step "gate 2/3 — exclusion enforcement (the real resolver must REFUSE each excluded wheel)"
EXCLUDED="$(sed -n 's/^\([A-Za-z0-9._-]*\)<0$/\1/p' "$ENV_DIR/constraints.txt")"
[ -n "$EXCLUDED" ] || { echo "no exclusions found in constraints.txt — refusing a vacuous gate" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
for dist in $EXCLUDED; do
  printf '%s\n' "$dist" > "$WORK/one.in"
  if uv pip compile "$WORK/one.in" -c "$ENV_DIR/constraints.txt" \
        --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
        -o "$WORK/one.lock" >"$WORK/one.log" 2>&1; then
    echo "FAIL: '$dist' is EXCLUDED, yet the resolver installed it under constraints.txt." >&2
    echo "      The exclusion is not being enforced. Re-render constraints.txt from the spec." >&2
    exit 1
  fi
  echo "  refused: $dist  ($(grep -m1 'we can conclude' "$WORK/one.log" | sed 's/^ *//' || echo 'no solution found'))"
done

# The other half of the claim, and the reason this gate is not just `grep`. An exclusion that is
# too BLUNT is as broken as one that is too weak: it would make the environment itself
# unresolvable. So the committed requirements must still solve under exactly these constraints.
# (Deliberately NOT asserted byte-identical to requirements.lock: transitive upstreams publish new
# releases continuously, so that comparison would paint this environment red for other people's
# releases. The lock is what gets INSTALLED; this gate proves the constraints stay satisfiable.)
echo "  and the environment itself still resolves under exactly those constraints:"
uv pip compile "$ENV_DIR/requirements.in" -c "$ENV_DIR/constraints.txt" \
  --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --extra-index-url "$TORCH_CUDA_INDEX" --index-strategy unsafe-best-match \
  -o "$WORK/env.lock" >/dev/null
grep -q '^vllm==0.24.0$' "$WORK/env.lock" || {
  echo "FAIL: a fresh solve under constraints.txt no longer yields the pinned vllm==0.24.0." >&2
  exit 1
}
echo "  resolved (vllm==0.24.0 reachable under the exclusions)."

# --- Gate 3: live stand-up --------------------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  cat >&2 <<EOF

The served-ocr-server environment is not installed at: $VENV
Gate 3 needs it. Either install it:

    bash $ENV_DIR/install.sh

(or build the image: docker build -f $ENV_DIR/Dockerfile -t latence/ocr-server:0.24.0 $ENV_DIR)
or ask for the offline gate deliberately:

    bash $ENV_DIR/smoke.sh --spec-only
EOF
  exit 2
fi

step "gate 3/3 — live stand-up (import the serving stack; prove the excluded wheels are absent)"
"$VENV/bin/python" - <<'PY'
import sys
import torch
import vllm

cuda = torch.version.cuda or ""
print(f"  torch {torch.__version__} (cuda {cuda}) / vllm {vllm.__version__}")
if not cuda.startswith("13."):
    sys.exit(
        f"FAIL: torch reports CUDA {cuda!r}; this environment pins the cu130 build "
        "(torch==2.11.0+cu130). The install resolved against the wrong index."
    )
PY
"$VENV/bin/python" "$ENV_DIR/verify-exclusions.py" "$ENV_DIR/constraints.txt"

printf '\n\033[1;32m========================================\n'
printf 'SERVED-OCR SERVER ENVIRONMENT SMOKE GREEN — all 3 gates passed.\n'
printf 'Serve it:  bash %s/serve.sh\n' "$ENV_DIR"
printf '========================================\033[0m\n'
