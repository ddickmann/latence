#!/usr/bin/env bash
# Report whether the `pypi-publish` job is enabled AT THE REF THIS RUN IS EXECUTING.
#
# Why this exists at all: GitHub Actions runs the workflow DEFINITION from the ref the run is
# on. A tag cut before the PyPI enable was committed therefore executes the *disabled*
# definition forever — `pypi-publish` evaluates to false and is skipped. A skipped job exits 0,
# emits no annotation, and leaves no trace in the summary, so "green release run, nothing on
# PyPI" is indistinguishable from a successful publish. Committing the enable on `main`
# afterwards changes nothing for that tag, and dispatching from `main` instead is refused by
# release-integrity's non-tag signing-identity guard. See docs/GO-LIVE-RUNBOOK.md → Step 4.
#
# This script makes that state VISIBLE in every release run. Both states are legitimate, so
# neither fails the run; only a `pypi-publish` job with no executable `if:` at all is an error,
# because that gate is missing rather than closed.
#
# Usage: bash scripts/pypi-gate-status.sh [path-to-release.yml]
set -euo pipefail

workflow="${1:-.github/workflows/release.yml}"

if [ ! -r "$workflow" ]; then
  echo "::error::pypi-gate-status: cannot read '${workflow}'" >&2
  exit 2
fi

# The job's EXECUTABLE `if:` — comment lines are skipped, so the ENABLE CHECKLIST prose (which
# quotes `if: ${{ false && … }}` verbatim) can never be mistaken for the condition itself.
condition="$(
  awk '
    /^  pypi-publish:[[:space:]]*$/ { in_job = 1; next }
    in_job && /^  [^[:space:]#]/    { in_job = 0 }
    in_job && /^[[:space:]]*#/      { next }
    in_job && /^    if:[[:space:]]/ {
      sub(/^    if:[[:space:]]*/, "")
      print
      exit
    }
  ' "$workflow"
)"

if [ -z "$condition" ]; then
  echo "::error::pypi-gate-status: the pypi-publish job in '${workflow}' has no executable \`if:\` — the PyPI gate is MISSING, not closed; a tag push could publish without review"
  exit 1
fi

case "$condition" in
  '${{ false'*)
    echo "**PyPI publish gate: DISABLED at this ref.** \`pypi-publish\` is short-circuited by \`${condition}\` in the \`release.yml\` this run is executing, so it will be SKIPPED and nothing will reach PyPI — a green run here is NOT a publish. The enable must be merged to \`main\` BEFORE the tag is cut; committing it afterwards cannot affect an existing tag. See docs/GO-LIVE-RUNBOOK.md → Step 4."
    ;;
  *)
    echo "**PyPI publish gate: ENABLED at this ref.** \`pypi-publish\` is gated only by \`${condition}\`, so a non-dry run on this ref WILL upload to PyPI once the protected \`pypi\` environment is approved. PyPI uploads cannot be deleted, only yanked."
    ;;
esac
