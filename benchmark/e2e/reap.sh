#!/usr/bin/env bash
# Kill every process belonging to a previous e2e attempt, including orphans.
#
# Why this is a FILE and not an inline `pkill -f ...`: a pattern typed into a shell command appears
# in that shell's own /proc/PID/cmdline, so `pkill -f` matches and kills the very shell running it.
# That happened three times here — twice killing an SSH session mid-cleanup, once killing the shell
# that was about to launch the supervisor, which then looked like "the supervisor died again".
# Patterns assembled at runtime inside a script file cannot match the caller's command line.
#
# Kills PROCESS GROUPS, not processes. Killing a shell script leaves its curl/python children alive
# with ppid 1, where they keep writing to the same download paths the next attempt will use — the
# root cause of the corrupt archive that stalled this run twice.

set -u

CORPUS_ROOT="${CORPUS_ROOT:-/workspace/corpus}"

# Assembled at runtime; see above.
PATTERNS=(
  "supervi""se.sh"
  "fetch""_corpus.sh"
  "digitalcorp""ora.org"
  "sec.g""ov/Archives"
  "ohr-p""dfs"
)

killed=0
for pat in "${PATTERNS[@]}"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "$$" ]; then
      kill -TERM -"$pgid" 2>/dev/null && killed=$(( killed + 1 ))
    fi
  done
done

sleep 5

for pat in "${PATTERNS[@]}"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pid" = "$$" ] && continue
    pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pgid" ] && [ "$pgid" != "$$" ] && kill -KILL -"$pgid" 2>/dev/null
  done
done

# The lock is advisory and dies with its holder, but a stale file plus a half-written .part would
# make the next attempt's first action a corrupt resume.
rm -f "$CORPUS_ROOT/_work/.fetch.lock" "$CORPUS_ROOT/_work"/*.part 2>/dev/null

printf 'reap: signalled %s process group(s); survivors: %s\n' \
  "$killed" "$(pgrep -f "fetch""_corpus.sh" 2>/dev/null | wc -l | tr -d ' ')"
