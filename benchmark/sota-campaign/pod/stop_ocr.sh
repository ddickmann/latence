#!/usr/bin/env bash
# stop_ocr.sh — stop the vLLM OCR daemon and verify the shared GPU's VRAM is freed.
#
# SIGINT first (vLLM shuts down cleanly on it), SIGTERM after a grace period, then verify
# via nvidia-smi that memory use is back under the shared-pod threshold — S3's GLiNER2
# needs the card. Exit 0 = stopped + VRAM freed; 5 = a process survived or VRAM still held.
set -euo pipefail

ROOT="${S2_ROOT:-/workspace/latence-sota/s2}"
PID_FILE="$ROOT/vllm.pid"
GPU_BUSY_MIB=2048

if [ ! -f "$PID_FILE" ]; then
  echo "no pid file at $PID_FILE — nothing to stop."
else
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "stopping vllm (pid $PID) with SIGINT ..."
    kill -INT "$PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "still alive; escalating to SIGTERM ..."
      kill -TERM "$PID" 2>/dev/null || true
      for _ in $(seq 1 15); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 1
      done
    fi
    if kill -0 "$PID" 2>/dev/null; then
      echo "vllm (pid $PID) survived SIGINT+SIGTERM — intervene manually."
      exit 5
    fi
  else
    echo "pid $PID not running (stale pid file)."
  fi
  rm -f "$PID_FILE"
fi

# Verify the VRAM actually came back (model unload can lag process exit briefly).
for _ in $(seq 1 12); do
  USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [ "$USED_MIB" -le "$GPU_BUSY_MIB" ]; then
    echo "GPU freed: ${USED_MIB}MiB in use."
    exit 0
  fi
  sleep 5
done

echo "GPU still holds ${USED_MIB}MiB after stop — check nvidia-smi."
exit 5
