#!/usr/bin/env bash
# deploy/env/full-stack/verify-full-stack.sh — THE one documented full-stack verification (#199).
#
# One command, on a Blackwell (sm_120) / CUDA-13 host, from the PINNED environments: a small
# corpus of scanned pages goes in; a Quality Report and the Exports come out; and every claim the
# compatibility matrix makes about the composed stack is measured rather than inferred.
#
#   deploy/env/served-ocr-server/   vLLM + transformers 5.x   ==HTTP==>  the OCR boundary
#   deploy/env/gpu-learned/         transformers 4.x, in-process GPU learned Providers + exports
#
#   bash deploy/env/full-stack/verify-full-stack.sh --corpus <dir>
#   bash deploy/env/full-stack/verify-full-stack.sh --corpus <dir> --record   # emit the YAML
#                                                                            # evidence block
#
# Required: $LATENCE_OCR_BASE_URL pointing at the RUNNING served-OCR environment. There is no
# localhost default, deliberately — a default lets this "pass" against whatever is listening.
#
# FOUR gates, cheapest first. Gate 0 runs anywhere; gates 1-3 need the hardware.
#
#   0. The two environments' own offline spec gates (`smoke.sh --spec-only` for each). If the pins
#      the run is supposed to verify have drifted, nothing measured afterwards describes them.
#   1. The GPU environment's own gates (`smoke.sh`), including the #191 placement probe. Composing
#      a stack out of a leg that was never proven green is how a full-stack run produces an
#      unattributable failure.
#   2. The composed run: `latence run` over stack.yaml with the served OCR endpoint wired in.
#   3. The artifacts: `assert_full_stack.py` (skipped Stage, cpu-resolved Stage, empty export and
#      document_count=0 all FAIL here — see its module docstring).
#
# `--record` prints the `full_stack_run:` YAML block, with the observed numbers filled in from the
# artifacts, to paste into deploy/env/compatibility-matrix.yaml. The evidence is transcribed by the
# machine that ran it, never typed from memory.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
GPU_ENV="$REPO_ROOT/deploy/env/gpu-learned"
OCR_ENV="$REPO_ROOT/deploy/env/served-ocr-server"
VENV="${LATENCE_ENV_VENV:-$GPU_ENV/.venv}"
RUN_ID="${LATENCE_RUN_ID:-full-stack-0001}"
CORPUS=""
WORKDIR=""
RECORD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) CORPUS="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --record) RECORD=1; shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }

# --- Gate 0: the pinned environments still are what they say they are ------------------------
step "gate 0/3 — the two pinned environments' offline spec gates"
bash "$OCR_ENV/smoke.sh" --spec-only
bash "$GPU_ENV/smoke.sh" --spec-only

if [ -z "$CORPUS" ]; then
  cat >&2 <<'EOF'

--corpus <dir> is required: a small corpus of SCANNED pages (pdf/png/jpg/tiff/webp/bmp).
Text files are not accepted by this stack's Source on purpose — a .txt would be parsed by a
text Provider and the verification would go green without the OCR server ever being called.

Gate 0 (offline) has passed. Gates 1-3 need a Blackwell/CUDA-13 host, the installed
gpu-learned environment and a running served-OCR endpoint.
EOF
  exit 2
fi
if [ ! -d "$CORPUS" ]; then
  echo "corpus directory not found: $CORPUS" >&2; exit 2
fi
if [ -z "${LATENCE_OCR_BASE_URL:-}" ]; then
  echo "LATENCE_OCR_BASE_URL is unset — point it at the RUNNING served-OCR endpoint" >&2
  echo "  (start it with: bash $OCR_ENV/serve.sh)" >&2
  exit 2
fi
if [ ! -x "$VENV/bin/latence" ]; then
  echo "the gpu-learned environment is not installed at $VENV" >&2
  echo "  install it with: bash $GPU_ENV/install.sh" >&2
  exit 2
fi

# --- Gate 1: the GPU leg is green on its own, placement probe included ------------------------
step "gate 1/3 — the gpu-learned environment's own gates (stand-up + GPU PLACEMENT + slice)"
bash "$GPU_ENV/smoke.sh" --venv "$VENV"

# --- Gate 2: the composed run -----------------------------------------------------------------
WORKDIR="${WORKDIR:-$(mktemp -d)}"
STORAGE="$WORKDIR/storage"
STACK="$WORKDIR/stack.yaml"
mkdir -p "$STORAGE"
sed -e "s#{CORPUS_DIR}#$(cd "$CORPUS" && pwd)#g" \
    -e "s#{STORAGE_URI}#$STORAGE#g" \
    -e "s#{OCR_BASE_URL}#$LATENCE_OCR_BASE_URL#g" \
    "$HERE/stack.yaml" > "$STACK"

step "gate 2/3 — the composed run (served OCR -> GPU learned Providers -> exports)"
echo "corpus:  $CORPUS"
echo "storage: $STORAGE"
echo "ocr:     $LATENCE_OCR_BASE_URL"
# No LATENCE_CUDA override, deliberately: this run must read the REAL device situation. Forcing
# the seam's answer would be the verification telling itself what it wants to hear.
PYTHONHASHSEED=0 "$VENV/bin/latence" run "$STACK" --run-id "$RUN_ID" >"$WORKDIR/report.json"

# --- Gate 3: the artifacts --------------------------------------------------------------------
step "gate 3/3 — the artifacts (Quality Report + Exports, with no Stage skipped or CPU-resolved)"
"$VENV/bin/python" "$HERE/assert_full_stack.py" "$STORAGE" "$RUN_ID"

printf '\n\033[1;32m========================================\n'
printf 'FULL-STACK VERIFICATION GREEN — all 3 gates passed.\n'
printf '========================================\033[0m\n'
echo "artifacts: $STORAGE/_latence/runs/$RUN_ID"

if [ "$RECORD" = "1" ]; then
  step "the evidence block for deploy/env/compatibility-matrix.yaml"
  "$VENV/bin/python" - "$STORAGE" "$RUN_ID" <<'PY'
import json, subprocess, sys, datetime
from pathlib import Path

storage, run_id = Path(sys.argv[1]), sys.argv[2]
report = json.loads((storage / "_latence/runs" / run_id / "quality-report.json").read_text())
try:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
except Exception:                                     # noqa: BLE001 - best effort provenance
    gpu = "UNKNOWN (nvidia-smi unavailable) — fill this in by hand"
docs = report.get("document_count", 0)
print("full_stack_run:")
print("  status: verified")
print("  command: >-")
print("    LATENCE_OCR_BASE_URL=... bash deploy/env/full-stack/verify-full-stack.sh --corpus <dir>")
print("  script: deploy/env/full-stack/verify-full-stack.sh")
print("  stack: deploy/env/full-stack/stack.yaml")
print("  artifacts:")
for artifact in (
    f"_latence/runs/{run_id}/quality-report.json",
    f"_latence/runs/{run_id}/quality-report.md",
    f"_latence/runs/{run_id}/export/corpus.jsonl",
    f"_latence/runs/{run_id}/export/graph-nodes.parquet",
    f"_latence/runs/{run_id}/export/graph-edges.parquet",
):
    print(f"    - {artifact}")
print("  evidence:")
print(f"    date: \"{datetime.date.today().isoformat()}\"")
print(f"    hardware: \"{gpu}\"")
print("    command: bash deploy/env/full-stack/verify-full-stack.sh --corpus <dir>")
print(
    f"    observed: \"{docs} document(s); contracts_complete="
    f"{report.get('contracts_complete')}; every learned Stage device_selected=cuda, none "
    "skipped; Quality Report + KG + RAG corpus written\""
)
PY
fi
