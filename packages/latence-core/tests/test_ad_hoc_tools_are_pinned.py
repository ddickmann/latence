# SPDX-License-Identifier: LicenseRef-Proprietary
"""Every tool a gate fetches on the fly is pinned to an exact version.

The workspace's own tools (ruff, mypy, pytest) are pinned by ``uv.lock``. The ones the gates
fetch beside it — ``uv run --with <tool>``, ``uvx <tool>``, ``uv pip install <tool>`` in the CI
workflows, the scripts, the Makefile and the pre-commit configuration — are not, unless the
command names a version. Unpinned, the secret scan, the vulnerability audit or the release
signature can change behaviour between two runs of the same commit. The Airflow lane installs with
``python -m pip`` under Airflow's own constraint file, and is not in this gate's scope.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCOPE = (".github", "scripts", "Makefile", ".pre-commit-config.yaml")
_FETCH = re.compile(
    r"(?:uv run\b[^\n]*?--with|uvx|uv tool run|uv pip install)((?:\s+(?:--?[\w-]+(?:=\S+)?))*)"
    r"\s+([A-Za-z][\w.\-\[\]]*(?:[=<>!~]=?[^\s'\"]*)?)"
)


def _files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *_SCOPE], cwd=_REPO_ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [n for n in listed.split("\0") if n and n.endswith((".yml", ".yaml", ".sh", "Makefile"))]


def test_every_fetched_tool_names_an_exact_version() -> None:
    unpinned: list[str] = []
    for name in _files():
        for number, line in enumerate(
            (_REPO_ROOT / name).read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.lstrip().startswith("#"):
                continue
            for match in _FETCH.finditer(line):
                flags, spec = match.group(1), match.group(2)
                if "--no-deps" in flags or "-e" in flags.split():
                    continue  # a workspace package installed from its source
                if "==" not in spec:
                    unpinned.append(f"{name}:{number}: {spec}")
    assert not unpinned, "pin these to an exact version (tool==X.Y.Z):\n" + "\n".join(unpinned)
