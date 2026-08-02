#!/usr/bin/env bash
# serve_ocr.sh — start LightOnOCR-2-1B under vLLM as a daemon on the shared 16GB pod.
#
# Contract (s2-ocr stage runs this via ssh before launching ocr_worker.py):
#   * Refuses to start if the GPU already has >2GiB in use — the pod is SHARED; stomping
#     another job's VRAM would kill both runs.
#   * Idempotent: an already-running, ready daemon (pid file alive + /v1/models answering)
#     exits 0 immediately.
#   * Installs vllm>=0.11.1 (+ the worker's deps: pypdfium2, pillow, openai) into a
#     dedicated venv at /workspace/venvs/ocr only if missing.
#   * Serving flags are the S0-diligence card flags verbatim (ocr-diligence.md).
#   * Daemonized with nohup + pid file; readiness = polling /v1/models; logs to vllm.log.
#
# No secrets live here: model + flags only. Exit codes: 0 ready, 3 GPU busy, 4 not ready
# within the readiness budget, other non-zero = setup failure.
set -euo pipefail

ROOT="${S2_ROOT:-/workspace/latence-sota/s2}"
VENV="${S2_VENV:-/workspace/venvs/ocr}"
MODEL="lightonai/LightOnOCR-2-1B"
PORT=8044
LOG="$ROOT/vllm.log"
PID_FILE="$ROOT/vllm.pid"
GPU_BUSY_MIB=2048
READINESS_TRIES=180   # x 5s = 15 min budget (first run downloads ~2.5GB of weights)

mkdir -p "$ROOT"

ready() { curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

# Idempotent: adopt an already-ready server (e.g. from a manual test-serve) instead of
# refusing on the busy GPU it itself occupies.
if ready; then
  echo "vllm already ready on :$PORT — adopting it."
  exit 0
fi

# Already running and answering? Nothing to do.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && ready; then
  echo "vllm already serving $MODEL on :$PORT (pid $(cat "$PID_FILE"))"
  exit 0
fi

# Shared pod guard: refuse to serve on a GPU somebody else is using.
USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
if [ "$USED_MIB" -gt "$GPU_BUSY_MIB" ]; then
  echo "refusing to start: GPU already has ${USED_MIB}MiB in use (>${GPU_BUSY_MIB}MiB — shared pod)."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
  exit 3
fi

# Install-if-missing into the dedicated venv (never the system python).
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c 'import vllm' 2>/dev/null; then
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "vllm==0.11.1" pypdfium2 pillow openai
fi

# The card flags (ocr-diligence.md / s2 ticket item 2) + Dennis's serving spec
# (2026-08-02): 0.8 GPU util, CUDA graphs on (no --enforce-eager), full optimization
# (-O3 = torch.compile + piecewise CUDA graphs).
nohup "$VENV/bin/vllm" serve "$MODEL" \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  -O3 \
  --max-num-seqs 16 \
  --port "$PORT" \
  >>"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "vllm launched (pid $(cat "$PID_FILE")); waiting for readiness on :$PORT ..."

for _ in $(seq 1 "$READINESS_TRIES"); do
  if ready; then
    echo "vllm ready: $MODEL on :$PORT"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "vllm died during startup; last log lines:"
    tail -n 30 "$LOG" || true
    exit 4
  fi
  sleep 5
done

echo "vllm not ready within the readiness budget; last log lines:"
tail -n 30 "$LOG" || true
exit 4
