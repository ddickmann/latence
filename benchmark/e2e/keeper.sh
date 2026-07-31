#!/usr/bin/env bash
# Keep the e2e supervisor alive across everything that has actually killed it.
#
# The supervisor survives stage failures — that is its job. What it does not survive is dying
# itself, and it has now died repeatedly for reasons that had nothing to do with the run: an SSH
# session going away, a `pkill -f` pattern matching the launching shell's own command line, a
# detached launch that did not actually detach. Each time, orphaned children (curl, python) kept
# running with ppid 1, so the work looked alive in `ps` while nothing was driving it.
#
# One process supervises the pipeline; this one supervises the supervisor. It is deliberately dumb:
# a loop, a liveness check, and a relaunch. The intelligence lives one level down.
#
# It does NOT relaunch on `blocked` — that is an escalation the operator must see, and restarting
# through it would turn a considered stop into a crash loop.
#
# Usage (from the repo root, on the pod):
#   setsid nohup bash benchmark/e2e/keeper.sh >> /workspace/run/keeper.log 2>&1 < /dev/null &

set -u

# Same reason as supervise.sh: a detached shell does not read the rc files that put user-local
# binaries on PATH, and the keeper relaunches the supervisor, so it must hand down a usable one.
export PATH="${HOME:-/root}/.local/bin:${HOME:-/root}/.cargo/bin:/usr/local/bin:$PATH"

REPO="${REPO:-/workspace/latence-framework}"
RUN_DIR="${RUN_DIR:-/workspace/run}"
STATE="$RUN_DIR/STATE.json"
CHECK_SECS="${CHECK_SECS:-60}"

PIDFILE="$RUN_DIR/supervisor.pid"

# Liveness by PID FILE, never by `pgrep -f`. A pattern matches any command line that merely MENTIONS
# the script — including an `scp`+`mv` one-liner that happens to carry the path in its argv. That
# phantom match kept this keeper idle for seven hours while the supervisor was dead: it saw a hit,
# concluded all was well, and never relaunched.
#
# Checking the pid also confirms the process is really the supervisor, so a recycled pid belonging
# to something else cannot impersonate it either.
supervisor_alive() {
  local pid
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "supervi""se.sh" || return 1
  return 0
}

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { printf '[%s] keeper: %s\n' "$(now)" "$*"; }

overall() {
  python3 -c "
import json
try:
    print(json.load(open('$STATE'))['overall'])
except Exception:
    print('unknown')
" 2>/dev/null || echo unknown
}

say "up (pid $$) — checking every ${CHECK_SECS}s"

restarts=0
while true; do
  state=$(overall)

  case "$state" in
    complete)
      say "run reported COMPLETE — keeper exiting"
      exit 0
      ;;
    blocked)
      # Deliberate stop, not a crash. Leave it stopped and visible.
      say "run is BLOCKED — not relaunching (an escalation is for the operator, not a retry loop)"
      exit 0
      ;;
  esac

  if ! supervisor_alive; then
    restarts=$(( restarts + 1 ))
    say "supervisor not running (state=$state) — relaunch #$restarts"
    # Reap orphans first. A dead supervisor leaves its stage children with ppid 1, still holding
    # locks and still writing to the files the next attempt is about to use.
    bash "$REPO/benchmark/e2e/reap.sh" 2>/dev/null || true
    sleep 3
    ( cd "$REPO" && setsid nohup bash benchmark/e2e/supervise.sh \
        >> "$RUN_DIR/supervisor.log" 2>&1 < /dev/null & )
    sleep 20
    if supervisor_alive; then
      say "supervisor is up"
    else
      say "RELAUNCH FAILED — supervisor did not come up; see $RUN_DIR/supervisor.log"
    fi
  fi

  sleep "$CHECK_SECS"
done
