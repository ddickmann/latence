#!/usr/bin/env bash
# podcheck_s2.sh — one-shot S2 status print for a human ssh'd into the pod.
#
# Shows: the worker's state.json, the tails of the worker + vLLM logs, the failure count,
# and one nvidia-smi line. Read-only; safe to run at any time.
set -uo pipefail

ROOT="${S2_ROOT:-/workspace/latence-sota/s2}"

echo "== state.json =="
if [ -f "$ROOT/state.json" ]; then
  cat "$ROOT/state.json"
else
  echo "(absent — job not launched, or launched with a different root)"
fi

echo
echo "== supervisor.log (last 8 lines) =="
tail -n 8 "$ROOT/supervisor.log" 2>/dev/null || echo "(no supervisor.log)"

echo
echo "== worker.log (last 15 lines) =="
tail -n 15 "$ROOT/worker.log" 2>/dev/null || echo "(no worker.log)"

echo
echo "== vllm.log (last 10 lines) =="
tail -n 10 "$ROOT/vllm.log" 2>/dev/null || echo "(no vllm.log)"

echo
echo "== failures =="
if [ -f "$ROOT/failures.jsonl" ]; then
  echo "$(wc -l <"$ROOT/failures.jsonl" | tr -d ' ') logged failure(s); last 3:"
  tail -n 3 "$ROOT/failures.jsonl"
else
  echo "(no failures.jsonl)"
fi

echo
echo "== gpu =="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader \
  || echo "(nvidia-smi unavailable)"
