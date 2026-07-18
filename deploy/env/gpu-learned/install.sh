#!/usr/bin/env bash
# deploy/env/gpu-learned/install.sh — THE documented install path for the GPU learned-Provider
# environment on CUDA 13 / Blackwell (issue #198). There is no second path and no manual step: a
# clean Linux/x86-64 pod with an sm_120 GPU, a CUDA-13 driver and a Python 3.11/3.12 interpreter
# runs this and gets the stack.
#
#   bash deploy/env/gpu-learned/install.sh                 # into deploy/env/gpu-learned/.venv
#   bash deploy/env/gpu-learned/install.sh --venv /opt/lat # elsewhere
#   bash deploy/env/gpu-learned/install.sh --relock        # RE-SOLVE requirements.lock, then install
#
# Then, from the repo root:
#   bash deploy/env/gpu-learned/smoke.sh                   # the one-command guard (4 gates)
#
# PEP 668. Every step happens inside a dedicated virtualenv, so an externally-managed system Python
# (`error: externally-managed-environment` — the field condition on the pod, which had no venvs at
# all) never blocks the install and nothing is written to the system site-packages. No
# `--break-system-packages`, ever.
#
# WHY EACH FLAG IS THERE (these are the trial-and-error steps this file exists to delete):
#   -c constraints.txt          the ENFORCED conflict boundaries. Without them the resolver takes
#                               transformers 5.x / huggingface-hub 1.x / tokenizers 0.23.1 and the
#                               whole GLiNER + transformers-4.x stack breaks at import.
#   --extra-index-url .../cu130 the CUDA-13 torch wheels. These are the builds carrying Blackwell
#                               (sm_120) kernels; the default PyPI torch is a cu12 build whose
#                               kernels this GPU cannot execute ("no kernel image is available").
#                               torch publishes +cu130 wheels only from 2.9.0 — hence the >=2.9
#                               floor in constraints.txt.
#   --index-strategy unsafe-best-match
#                               lets the resolver consider BOTH indexes for the same distribution,
#                               which is what makes the `+cu130` local-version wheels reachable.
#   --no-deps (workspace pkgs)  the latence-* packages are installed from source; their third-party
#                               surface is already solved in requirements.lock, and
#                               `latence env verify` asserts the lock satisfies every dependency
#                               they declare — so re-resolving here could only *unpin* the env.
#
# NOT INSTALLED HERE, ON PURPOSE: the in-process OCR VLMs (latence-parser-lighton and friends).
# They sit on the transformers>=5 side of the split and are reached over HTTP through
# `parser.endpoint`, across the ADR-0030 endpoint boundary. `environment.yaml`'s
# `excluded_packages` makes adding them a gate-1 failure rather than a load-time crash.
set -euo pipefail

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ENV_DIR/../../.." && pwd)"
VENV="$ENV_DIR/.venv"
RELOCK=0
PYTHON_VERSION="3.12"
PLATFORM="x86_64-unknown-linux-gnu"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --relock) RELOCK=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
  # a lockfile whose regeneration command is folklore is not reproducible. (uv does not echo the
  # --extra-index-url back into the lock header, which is precisely why it must live in this file.)
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

echo "== installing the solved CUDA-13 third-party set from requirements.lock =="
VIRTUAL_ENV="$VENV" uv pip install \
  --requirement "$ENV_DIR/requirements.lock" \
  --constraint "$ENV_DIR/constraints.txt" \
  --extra-index-url "$TORCH_CUDA_INDEX" \
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
