#!/usr/bin/env bash
# run_worker.sh — the pod-side supervisor around ocr_worker.py.
#
# Field lesson (2026-08-02, real S2 run): the render phase was OOM-killed silently at
# 36,430 pages — pid dead, log empty, state.json stale. The worker is now crash-isolated
# internally (render children) AND supervised here: every pass is skip-if-exists, so a
# relaunch simply continues from the file manifest. Up to MAX_LAUNCHES attempts; if the
# worker still cannot finish, a failed state.json is stamped so the campaign client
# blocks promptly with the reason instead of waiting out its stall budget.
#
# Launched by the s2-ocr stage as:  nohup bash pod/run_worker.sh <ROOT> & echo $! > worker.pid
# Python runs with -u so worker.log always flushes (a silent death leaves its traceback).
set -uo pipefail

ROOT="${1:-/workspace/latence-sota/s2}"
PYTHON="${S2_PYTHON:-/workspace/venvs/ocr/bin/python}"
POD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_LAUNCHES=5
RELAUNCH_DELAY_S=10

rc=1
for attempt in $(seq 1 "$MAX_LAUNCHES"); do
  echo "[supervisor] launch $attempt/$MAX_LAUNCHES at $(date -u +%FT%TZ)"
  "$PYTHON" -u "$POD_DIR/ocr_worker.py" --root "$ROOT" >>"$ROOT/worker.log" 2>&1
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] worker finished cleanly on launch $attempt"
    exit 0
  fi
  echo "[supervisor] worker exited rc=$rc (launch $attempt); last worker.log lines:"
  tail -n 10 "$ROOT/worker.log" 2>/dev/null || true
  if [ "$attempt" -lt "$MAX_LAUNCHES" ]; then
    echo "[supervisor] relaunching in ${RELAUNCH_DELAY_S}s (skip-if-exists resumes the manifest)"
    sleep "$RELAUNCH_DELAY_S"
  fi
done

echo "[supervisor] giving up after $MAX_LAUNCHES launches (last rc=$rc)"
# Stamp a failed state (atomic temp+mv) so the client blocks promptly with the reason
# instead of discovering a frozen counter after the full stall budget.
TMP="$ROOT/state.json.tmp"
printf '{"phase": "failed", "failed_phase": "supervisor", "progress": 0, "error": "worker died %s times; last rc=%s - see worker.log + supervisor.log on the pod"}\n' \
  "$MAX_LAUNCHES" "$rc" >"$TMP"
mv "$TMP" "$ROOT/state.json"
exit 1
