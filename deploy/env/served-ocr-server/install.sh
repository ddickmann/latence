#!/usr/bin/env bash
# deploy/env/served-ocr-server/install.sh — THE documented install path for the vLLM LightOn-OCR
# serving environment (#196), for a bare CUDA-13 / Blackwell machine where Docker is not the unit
# of deployment. The container path is deploy/env/served-ocr-server/Dockerfile; both install the
# SAME committed lock with the SAME constraints, so they are the same environment.
#
#   bash deploy/env/served-ocr-server/install.sh                 # -> deploy/env/served-ocr-server/.venv
#   bash deploy/env/served-ocr-server/install.sh --venv /opt/ocr # elsewhere
#   bash deploy/env/served-ocr-server/install.sh --relock        # RE-SOLVE the lock, then install
#
# Then:
#   bash deploy/env/served-ocr-server/smoke.sh   # the guard
#   bash deploy/env/served-ocr-server/serve.sh   # the endpoint
#
# PEP 668: everything happens inside a dedicated virtualenv, so an externally-managed system
# Python never blocks the install. No `--break-system-packages`, ever.
#
# WHY EACH FLAG IS THERE (these are the trial-and-error steps this file exists to delete):
#   -c constraints.txt       the ENFORCED boundaries AND the exclusions. `flash-attn` and
#                            `torchcodec` are pinned to `<0`, which nothing satisfies, so the
#                            resolver REFUSES to install them however they are asked for. This is
#                            the enforcement — there is no "uninstall them afterwards" step here
#                            or anywhere, because such a step only runs when somebody remembers.
#   --extra-index-url .../cu130
#                            the CUDA-13 torch wheels. The lock pins `torch==2.11.0+cu130`; that
#                            local version does not exist on PyPI, so without this index the
#                            install FAILS LOUDLY instead of silently substituting the cu12 build
#                            (which is the wrong CUDA line for the target host).
#   --index-strategy unsafe-best-match
#                            lets the resolver consider BOTH indexes for the same distribution,
#                            which is what makes the `+cu130` wheels reachable at all.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ENV_DIR/.venv"
RELOCK=0
PYTHON_VERSION="3.12"
PLATFORM="x86_64-unknown-linux-gnu"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --relock) RELOCK=1; shift ;;
    -h|--help) sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null 2>&1 || {
  echo "uv is required (https://docs.astral.sh/uv/). Install it, then re-run." >&2
  exit 2
}

if [ "$RELOCK" = "1" ]; then
  echo "== re-solving requirements.lock (network) =="
  # The EXACT command the committed lock was produced with. Kept here, not in a comment somewhere:
  # a lockfile whose regeneration command is folklore is not reproducible.
  ( cd "$ENV_DIR" && uv pip compile requirements.in \
      -c constraints.txt \
      --python-version "$PYTHON_VERSION" \
      --python-platform "$PLATFORM" \
      --extra-index-url "$TORCH_CUDA_INDEX" \
      --index-strategy unsafe-best-match \
      -o requirements.lock )
fi

echo "== creating venv: $VENV (python $PYTHON_VERSION) =="
uv venv --python "$PYTHON_VERSION" "$VENV"

echo "== installing the solved serving set from requirements.lock =="
VIRTUAL_ENV="$VENV" uv pip install \
  --requirement "$ENV_DIR/requirements.lock" \
  --constraint "$ENV_DIR/constraints.txt" \
  --extra-index-url "$TORCH_CUDA_INDEX" \
  --index-strategy unsafe-best-match

echo
echo "== installed. verify + serve: =="
echo "   bash $ENV_DIR/smoke.sh"
echo "   bash $ENV_DIR/serve.sh"
