#!/usr/bin/env bash
# deploy/env/served-ocr-client/install.sh — THE documented install path for the served-OCR CLIENT
# environment (issue #196): the CPU box that runs the pipeline and calls the vLLM LightOn-OCR
# endpoint over HTTP. The SERVER half is deploy/env/served-ocr-server/ and is installed
# separately, on the GPU host — that separation is the architecture, not an inconvenience.
#
#   bash deploy/env/served-ocr-client/install.sh                 # -> deploy/env/served-ocr-client/.venv
#   bash deploy/env/served-ocr-client/install.sh --venv /opt/lat # elsewhere
#   bash deploy/env/served-ocr-client/install.sh --relock        # RE-SOLVE the lock, then install
#
# Then, with the server reachable:
#   LATENCE_OCR_BASE_URL=http://<gpu-host>:8000/v1 bash deploy/env/served-ocr-client/smoke.sh
#
# PEP 668: everything happens inside a dedicated virtualenv. No `--break-system-packages`, ever.
#
# WHY EACH FLAG IS THERE:
#   -c constraints.txt       the ENFORCED boundaries AND the exclusions. `transformers<5` is the
#                            client side of the split (the serving environment is on 5.x and the
#                            two can never merge); `flash-attn` / `torchcodec` are pinned to `<0`,
#                            which nothing satisfies, so the resolver REFUSES them however they
#                            are asked for. No "uninstall afterwards" step exists here.
#   --extra-index-url .../cpu
#                            CPU-only torch. This box runs no model — the OCR is on the far side
#                            of the socket — so the ~2.5 GiB nvidia-* payload the default PyPI
#                            torch drags in is pure download weight.
#   --index-strategy unsafe-best-match
#                            lets the resolver see both indexes for one distribution, which is
#                            what makes the `+cpu` wheels reachable.
#   --no-deps (workspace pkgs)
#                            the latence-* packages are installed from source; their third-party
#                            surface is already solved in requirements.lock, and `latence env
#                            verify` asserts the lock satisfies everything they declare — so
#                            re-resolving here could only UNPIN the environment.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../.." && pwd)"
VENV="$ENV_DIR/.venv"
RELOCK=0
PYTHON_VERSION="3.12"
PLATFORM="x86_64-unknown-linux-gnu"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --relock) RELOCK=1; shift ;;
    -h|--help) sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null 2>&1 || {
  echo "uv is required (https://docs.astral.sh/uv/). Install it, then re-run." >&2
  exit 2
}

if [ "$RELOCK" = "1" ]; then
  echo "== re-solving requirements.lock (network) =="
  # The EXACT command the committed lock was produced with.
  ( cd "$ENV_DIR" && uv pip compile requirements.in \
      -c constraints.txt \
      --python-version "$PYTHON_VERSION" \
      --python-platform "$PLATFORM" \
      --extra-index-url "$TORCH_CPU_INDEX" \
      --index-strategy unsafe-best-match \
      -o requirements.lock )
fi

echo "== creating venv: $VENV (python $PYTHON_VERSION) =="
uv venv --python "$PYTHON_VERSION" "$VENV"

echo "== installing the solved third-party set from requirements.lock =="
VIRTUAL_ENV="$VENV" uv pip install \
  --requirement "$ENV_DIR/requirements.lock" \
  --constraint "$ENV_DIR/constraints.txt" \
  --extra-index-url "$TORCH_CPU_INDEX" \
  --index-strategy unsafe-best-match

echo "== installing the workspace Provider packages from source (--no-deps) =="
# Read the install order straight out of the spec, so adding a package to the environment is a
# one-line spec edit and never a forgotten line in this script.
WORKSPACE_PACKAGES="$("$VENV/bin/python" - "$ENV_DIR/environment.yaml" <<'PY'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(spec["workspace_packages"]))
PY
)"
for pkg in $WORKSPACE_PACKAGES; do
  VIRTUAL_ENV="$VENV" uv pip install --no-deps "$REPO_ROOT/$pkg"
done

echo
echo "== installed. verify + smoke (the server must be reachable for gate 3): =="
echo "   bash $ENV_DIR/smoke.sh --spec-only"
echo "   LATENCE_OCR_BASE_URL=http://<gpu-host>:8000/v1 bash $ENV_DIR/smoke.sh"
