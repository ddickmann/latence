#!/usr/bin/env bash
# Strict, whole-tree type check over BOTH src and tests for every workspace package.
#
# Why a per-package loop and not a single `mypy --strict packages` invocation:
# the package directories are hyphenated (packages/latence-core, ...), which mypy
# cannot turn into dotted module names, so every package's tests/conftest.py maps
# to the SAME bare top-level module `conftest`. A single whole-tree invocation
# therefore aborts with `Duplicate module named "conftest"` before checking
# anything (see issue #24). Checking one package at a time keeps each package's
# conftest the only `conftest` in that mypy process, so there is no collision, and
# every src AND tests file is still strictly checked. This script is the single
# command CI runs; a non-zero exit means at least one package failed strict mypy.
#
# Determinism: PYTHONHASHSEED is pinned by CI/pytest; mypy itself is deterministic.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional runner prefix (e.g. `uv run`). Default to bare mypy so the script works
# both under `uv run scripts/typecheck.sh` and a plain venv.
MYPY=(${MYPY_CMD:-mypy})

failed=()
checked=0

for pkg in packages/*/; do
  pkg="${pkg%/}"
  src="$pkg/src"
  tests="$pkg/tests"
  [ -d "$src" ] || continue

  roots=("$src")
  # Put the package's own tests dir on MYPYPATH so its conftest fixtures resolve,
  # and add tests as a checked root when present.
  mypypath="$src"
  if [ -d "$tests" ]; then
    roots+=("$tests")
    mypypath="$src:$tests"
  fi

  echo "== strict typecheck: $pkg =="
  if ! MYPYPATH="$mypypath" "${MYPY[@]}" --strict --namespace-packages "${roots[@]}"; then
    failed+=("$pkg")
  fi
  checked=$((checked + 1))
done

echo
if [ ${#failed[@]} -ne 0 ]; then
  echo "STRICT TYPECHECK FAILED for ${#failed[@]}/${checked} package(s): ${failed[*]}"
  exit 1
fi
echo "STRICT TYPECHECK PASSED for all $checked package(s) (src + tests)."
