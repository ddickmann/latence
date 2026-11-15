# SPDX-License-Identifier: Apache-2.0
"""Every e-mail address in the tracked tree is one no one can receive mail at.

Fixtures, examples and synthetic corpora are full of addresses, because PII is what this
framework redacts. An address at a registrable domain (``acme.com``, ``corp.com``, ``b.com``)
belongs to someone: printed in a public repository it is a real person's or company's address
in a "synthetic" record, and a test that ever sent to it would reach them. RFC 2606 and RFC 6761
reserve names that can never be registered (``example.com/.net/.org`` and the ``.example``,
``.test``, ``.invalid`` and ``.localhost`` top-level domains); synthetic data uses those.

The only other addresses allowed are the project's own contact (read from ``CITATION.cff``, the
file that states it) and the ASF's, quoted in ``CLA.md``. ``example.de`` is held by DENIC, the
``.de`` registry, as its documentation domain, and is kept for the German synthetic records,
whose recorded character offsets would move with any other name.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADDRESS = re.compile(r"[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})")
_RESERVED = re.compile(
    r"(?:^|\.)(?:example|test|invalid|localhost)$|(?:^|\.)example\.(?:com|net|org)$", re.I
)
_DOCUMENTATION_DOMAINS = frozenset({"example.de"})  # DENIC's documentation domain


def _contact() -> set[str]:
    text = (_REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^\s*email:\s*(\S+)\s*$", text, re.M)}


def test_every_address_in_the_tracked_tree_is_at_a_reserved_domain() -> None:
    allowed = _contact() | {"secretary@apache.org"}
    files = subprocess.run(
        ["git", "ls-files", "-z"], cwd=_REPO_ROOT, capture_output=True, check=True
    ).stdout.decode().split("\0")
    offending: list[str] = []
    for name in filter(None, files):
        path = _REPO_ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue
        for match in _ADDRESS.finditer(text):
            domain = match.group(1).lower()
            if match.group(0) in allowed or _RESERVED.search(domain):
                continue
            if domain in _DOCUMENTATION_DOMAINS:
                continue
            line = text.count("\n", 0, match.start()) + 1
            offending.append(f"{name}:{line}: {match.group(0)}")
    assert not offending, (
        "addresses at registrable domains (use example.com/.org/.net or a .example/.test/.invalid "
        "name):\n" + "\n".join(offending)
    )
