"""Every tracked test file runs in a gate (ticket 24: no gate ran ``benchmark/corpus`` or
``benchmark/rfc``, and two tests there had been failing unseen).

``pytest`` collects ``packages/`` (``testpaths``). A test file anywhere else runs only if the merge
gate (``scripts/verify-local.sh``) and CI (``.github/workflows/ci.yml``) name it. The prototypes
under ``.scratch/`` are exempt by name: they carry their own environments and runners, and no
repository gate covers them (the owner's brief of 14 Nov says so).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXEMPT_PREFIXES = (".scratch/",)
_TEST_FILE = re.compile(r"(^|/)test_[^/]*\.py$")


def _tracked_tests() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if _TEST_FILE.search(p) and not p.startswith(("packages/", *_EXEMPT_PREFIXES))]


def test_every_test_file_outside_the_packages_is_named_by_the_merge_gate_and_by_ci() -> None:
    files = _tracked_tests()
    assert files, "no test file outside packages/: the check has nothing to hold (the scan is broken)"
    gate = (_REPO_ROOT / "scripts" / "verify-local.sh").read_text(encoding="utf-8")
    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    missing = [f"{f} ({where})" for f in files for where, text in (("verify-local.sh", gate), ("ci.yml", ci)) if f not in text]
    assert missing == [], f"test files no gate runs: {missing}"
