# SPDX-License-Identifier: LicenseRef-Proprietary
"""Third-party material in the tracked tree whose licence asks for attribution is named in NOTICE.

``NOTICE`` is the file Apache-2.0 §4(d) has every redistributor carry, so it is where a copy of
this repository carries the attribution a Creative Commons licence asks for. An outside
vocabulary file states its own licence (``licence``) and attribution (``attribution``); one under
a CC BY licence (CC BY-SA included) that NOTICE does not name would travel in every clone with
its attribution only inside the file itself.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _vocabularies() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "*.vocabulary.*.json"],
        cwd=_REPO_ROOT, capture_output=True, check=True,
    ).stdout.decode()
    return [name for name in listed.split("\0") if name]


def test_there_is_a_vocabulary_to_check() -> None:
    assert _vocabularies(), "no outside vocabulary file is tracked; this gate checks nothing"


def test_every_attribution_licensed_vocabulary_is_named_in_notice() -> None:
    notice = (_REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    missing: list[str] = []
    for name in _vocabularies():
        vocabulary = json.loads((_REPO_ROOT / name).read_text(encoding="utf-8"))
        if not str(vocabulary.get("licence", "")).upper().startswith("CC-BY"):
            continue
        if name not in notice or vocabulary["licence"] not in notice:
            missing.append(f"{name} ({vocabulary['licence']})")
    assert not missing, "NOTICE does not name, with its licence: " + ", ".join(missing)
