# SPDX-License-Identifier: Apache-2.0
"""No source module grows past 1,000 lines, and the ones already past it do not grow at all.

A module past a thousand lines is one a reviewer can no longer hold in mind, and where an
unrelated change hides. The modules listed below were past that line when this gate was added;
each is held at its size then. Splitting one lowers its entry (or removes it); adding to one
fails here, so new code goes into a new module. Vendored trees (``_vendor``) are third-party
code kept verbatim and are not counted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIMIT = 1000
# The modules past the limit when the gate was added, held at that size (lines).
_HELD = {
    "packages/latence-core/src/latence_core/contracts.py": 2570,
    "packages/latence-core/src/latence_core/providers/perf.py": 2139,
    "packages/latence-core/src/latence_core/runner.py": 2057,
    "packages/latence-core/src/latence_core/cli.py": 1836,
    "packages/latence-core/src/latence_core/conformance/cases.py": 1790,
    "packages/latence-core/src/latence_core/setup/recipe.py": 1789,
    "packages/latence-core/src/latence_core/quality_report_builder.py": 1605,
    "packages/latence-core/src/latence_core/envspec.py": 1323,
    "packages/latence-core/src/latence_core/conformance/stub.py": 1245,
    "packages/latence-schema-inducer/src/latence_schema_inducer/provider.py": 1110,
    "packages/latence-core/src/latence_core/delta_runner.py": 1089,
    "packages/latence-core/src/latence_core/compat.py": 1069,
    "packages/latence-extract-gliner2/src/latence_extract_gliner2/provider.py": 1056,
    "packages/latence-core/src/latence_core/observability.py": 1024,
    "packages/latence-disambig-glinker/src/latence_disambig_glinker/entity_resolver.py": 1023,
}


def _sizes() -> dict[str, int]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "packages/*/src/*.py"],
        cwd=_REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode()
    return {
        name: len((_REPO_ROOT / name).read_text(encoding="utf-8").splitlines())
        for name in filter(None, listed.split("\0"))
        if "/_vendor/" not in name
    }


def test_no_source_module_grows_past_its_ceiling() -> None:
    grown = [
        f"{name}: {lines} lines (ceiling {_HELD.get(name, _LIMIT)})"
        for name, lines in sorted(_sizes().items())
        if lines > _HELD.get(name, _LIMIT)
    ]
    assert not grown, "split the module rather than raise its ceiling:\n" + "\n".join(grown)


def test_every_held_module_still_exists_and_is_still_past_the_limit() -> None:
    sizes = _sizes()
    stale = [n for n in _HELD if n not in sizes or sizes[n] <= _LIMIT]
    assert not stale, "remove these entries, the modules are gone or back under the limit: " + (
        ", ".join(stale)
    )
