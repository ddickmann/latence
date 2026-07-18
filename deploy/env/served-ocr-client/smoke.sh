#!/usr/bin/env bash
# deploy/env/served-ocr-client/smoke.sh — THE one-command guard for the served-OCR CLIENT
# environment (issue #196), and the ticket's end-to-end acceptance criterion: a small document
# parses end to end through the served endpoint, from the pinned environment.
#
#   bash deploy/env/served-ocr-client/smoke.sh --spec-only    # gate 1 — offline, no install
#   LATENCE_OCR_BASE_URL=http://gpu-host:8000/v1 \
#     bash deploy/env/served-ocr-client/smoke.sh              # all three gates
#
# THREE GATES, cheapest first:
#
#   1. SPEC (offline, import-free, always runs). `latence env verify` cross-checks the committed
#      requirements.in / constraints.txt / requirements.lock against the Provider packages' real
#      pyproject metadata, the enforced boundaries, the EXCLUSIONS and the smoke stack.
#      scripts/verify-local.sh runs this gate on every commit.
#
#   2. STAND-UP (needs the installed environment). Loads every declared Provider through the
#      registry, which imports the module behind it — where a torch/transformers drift detonates,
#      in seconds, before any network call.
#
#   3. END-TO-END THROUGH THE SEAM (needs the installed environment AND a reachable vLLM
#      LightOn-OCR server). Renders a one-page document with THIS environment's pinned Pillow,
#      runs stack.yaml over it, and asserts the page's sentinel tokens came back out in the
#      exported RAG corpus. The stack accepts image/PDF extensions only and has exactly one Parse
#      Provider — the served one — so there is no path by which this can pass without the endpoint
#      having done the work.
#
# NO DEFAULT ENDPOINT. Gate 3 requires $LATENCE_OCR_BASE_URL and fails without it, rather than
# trying localhost: a default would let this smoke go green against whatever happened to be
# listening on port 8000. Likewise, a missing venv FAILS instead of silently degrading to gate 1 —
# a smoke test that quietly narrows its own scope is a false green. Use --spec-only to ask for
# gate 1 deliberately.
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
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

# --- Gate 1: the spec (offline) --------------------------------------------------------------
step "gate 1/3 — environment spec (offline: pins, boundaries, exclusions, providers, stack)"
# Resolution order: the environment's own venv first, then the WORKSPACE (`uv run`, so a checkout
# is verified against its own source), and only then a `latence` that happens to be on PATH.
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

The served-ocr-client environment is not installed at: $VENV
Gates 2 and 3 need it. Either install it:

    bash $ENV_DIR/install.sh

or ask for the offline gate deliberately:

    bash $ENV_DIR/smoke.sh --spec-only
EOF
  exit 2
fi

# --- Gate 2: live stand-up -------------------------------------------------------------------
step "gate 2/3 — live stand-up (load every declared Provider: imports torch/transformers/openai)"
"$VENV/bin/latence" env verify "$SPEC" --repo-root "$REPO_ROOT" --probe

# --- Gate 3: end to end across the endpoint seam ----------------------------------------------
if [ -z "${LATENCE_OCR_BASE_URL:-}" ]; then
  cat >&2 <<EOF

Gate 3 needs a reachable vLLM LightOn-OCR endpoint, and there is deliberately no default:

    export LATENCE_OCR_BASE_URL=http://<gpu-host>:8000/v1
    bash $ENV_DIR/smoke.sh

Stand the server up from its own (separate, by design) environment:

    bash $REPO_ROOT/deploy/env/served-ocr-server/install.sh
    bash $REPO_ROOT/deploy/env/served-ocr-server/serve.sh
EOF
  exit 2
fi

step "gate 3/3 — one page end to end: rasterise -> POST -> vLLM LightOn-OCR -> PageMap -> export"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/corpus" "$WORK/out"

echo "  rendering the fixture page with this environment's pinned Pillow"
"$VENV/bin/python" "$ENV_DIR/make_fixture.py" "$WORK/corpus/served-ocr-smoke.pdf"

# Fill the stack's three placeholders. `|` as the sed delimiter because every value is a path/URL.
sed -e "s|{CORPUS_DIR}|$WORK/corpus|g" \
    -e "s|{STORAGE_URI}|file://$WORK/out|g" \
    -e "s|{OCR_BASE_URL}|$LATENCE_OCR_BASE_URL|g" \
    "$ENV_DIR/stack.yaml" > "$WORK/stack.yaml"

echo "  running the served-OCR stack against $LATENCE_OCR_BASE_URL"
# A local vLLM accepts any placeholder key; the Provider reads OPENAI_API_KEY when no api_key is
# configured. Nothing secret is embedded in the stack (ADR-0012).
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" PYTHONHASHSEED=0 \
  "$VENV/bin/latence" run "$WORK/stack.yaml" --run-id served-ocr-env-smoke

echo "  asserting the page's text survived the round trip into the exported corpus"
"$VENV/bin/python" "$ENV_DIR/assert_parsed.py" "$WORK/out"

printf '\n\033[1;32m========================================\n'
printf 'SERVED-OCR CLIENT ENVIRONMENT SMOKE GREEN — all 3 gates passed.\n'
printf '========================================\033[0m\n'
