#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Proprietary
"""Known-vulnerability gate over every committed lock file.

Exports each lock (the workspace's ``uv.lock`` and ``packages/latence-gliner25/uv.lock``, which
sits outside the workspace, ADR-0064) to a pinned requirements list and audits it with
``pip-audit`` against the PyPI/OSV advisory database. The run fails when:

- an advisory is reported that ``scripts/vuln-accepted.toml`` does not list with a reason, or
- an entry in that file is no longer reported (a stale acceptance is removed, not kept).

It needs the network (the advisory database). Run from the repository root:

    uv run --no-project --with pip-audit python scripts/vuln_audit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKS = (ROOT, ROOT / "packages" / "latence-gliner25")
ACCEPTED = ROOT / "scripts" / "vuln-accepted.toml"


def _reported(project: Path, scratch: Path) -> set[tuple[str, str]]:
    requirements = scratch / f"{project.name}.txt"
    subprocess.run(
        ["uv", "export", "--frozen", "--all-packages", "--all-extras", "--no-hashes",
         "--no-emit-workspace", "--no-emit-project", "-q", "-o", str(requirements)],
        cwd=project, check=True,
    )
    report = scratch / f"{project.name}.json"
    done = subprocess.run(
        [sys.executable, "-m", "pip_audit", "-r", str(requirements), "--no-deps",
         "--disable-pip", "--progress-spinner", "off", "-f", "json", "-o", str(report)],
        cwd=project,
    )
    if done.returncode not in (0, 1) or not report.exists():  # 1 means "found some"
        sys.exit(f"pip-audit could not audit {project.relative_to(ROOT) or '.'} "
                 f"(exit {done.returncode}); the gate does not pass on a failed audit")
    found: set[tuple[str, str]] = set()
    for dep in json.loads(report.read_text(encoding="utf-8"))["dependencies"]:
        for vuln in dep.get("vulns", []):
            found.add((dep["name"].lower(), vuln["id"]))
    return found


def main() -> int:
    accepted = {
        (entry["package"].lower(), vid)
        for entry in tomllib.loads(ACCEPTED.read_text(encoding="utf-8"))["accepted"]
        for vid in entry["ids"]
    }
    with tempfile.TemporaryDirectory() as tmp:
        reported: set[tuple[str, str]] = set()
        for project in LOCKS:
            reported |= _reported(project, Path(tmp))
    new = sorted(reported - accepted)
    stale = sorted(accepted - reported)
    for name, vid in new:
        print(f"ADVISORY  {name}  {vid}  (not accepted in scripts/vuln-accepted.toml)")
    for name, vid in stale:
        print(f"STALE     {name}  {vid}  (accepted, but no longer reported: remove the entry)")
    print(f"{len(reported)} advisories reported, {len(reported & accepted)} accepted with a "
          f"reason, {len(new)} new, {len(stale)} stale")
    return 1 if new or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
