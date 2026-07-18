#!/usr/bin/env bash
# deploy/env/cpu-text/install.sh — THE documented install path for the CPU/text enterprise-sota
# environment (issue #195). There is no second path and no manual step: a clean Linux/x86-64
# machine with a Python 3.11/3.12 interpreter runs this and gets the stack.
#
#   bash deploy/env/cpu-text/install.sh                 # install into deploy/env/cpu-text/.venv
#   bash deploy/env/cpu-text/install.sh --venv /opt/lat # install elsewhere
#   bash deploy/env/cpu-text/install.sh --relock        # RE-SOLVE requirements.lock, then install
#
# Then, from the repo root:
#   bash deploy/env/cpu-text/smoke.sh                   # the one-command guard
#
# PEP 668. Every step happens inside a dedicated virtualenv, so an externally-managed system
# Python (Debian/Ubuntu/Fedora `error: externally-managed-environment`) never blocks the install
# and nothing is ever written to the system site-packages. No `--break-system-packages`, ever.
#
# WHY EACH FLAG IS THERE (these are the trial-and-error steps this file exists to delete):
#   -c constraints.txt          the ENFORCED conflict boundaries; without them the resolver takes
#                               transformers 5.x / huggingface-hub 1.x / tokenizers 0.23.1 and the
#                               GLinker + transformers-4.x stack breaks at import.
#   --extra-index-url .../cpu   CPU-only torch wheels (`+cpu` local version). Without it the
#                               default PyPI torch drags ~2.5 GiB of nvidia-* CUDA payload into a
#                               CPU environment.
#   --index-strategy unsafe-best-match
#                               lets the resolver consider BOTH indexes for the same distribution,
#                               which is what makes the `+cpu` wheels reachable at all.
#   --no-deps (workspace pkgs)  the latence-* packages are installed from source; their
#                               third-party surface is already solved in requirements.lock, and
#                               `latence env verify` asserts the lock satisfies every dependency
#                               they declare — so re-resolving here could only *unpin* the env.
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
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null 2>&1 || {
  echo "uv is required (https://docs.astral.sh/uv/). Install it, then re-run." >&2
  exit 2
}

if [ "$RELOCK" = "1" ]; then
  echo "== re-solving requirements.lock (network) =="
  # The EXACT command the committed lock was produced with. Keep it here, not in a comment
  # somewhere: a lockfile whose regeneration command is folklore is not reproducible.
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
echo "== installed. verify + smoke: =="
echo "   bash $ENV_DIR/smoke.sh"
