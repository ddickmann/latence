# SPDX-License-Identifier: Apache-2.0
"""Every package in ``packages/`` is named on a page a newcomer reads first.

A distribution no front-door page names is one a reader of this public repository cannot find:
the ontology build and its proposal writer shipped while the README still said that work was not
built. The front door is the README, the roadmap, the install map, the agent briefing and the
Provider catalog; a package named on none of them fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FRONT_DOOR = (
    "README.md",
    "ROADMAP.md",
    "docs/getting-started/index.md",
    "docs/agent-briefing.md",
    "docs/providers/catalog.md",
)


def test_every_package_is_named_on_a_front_door_page() -> None:
    text = "\n".join((_REPO_ROOT / page).read_text(encoding="utf-8") for page in _FRONT_DOOR)
    packages = sorted(p.parent.name for p in (_REPO_ROOT / "packages").glob("*/pyproject.toml"))
    assert packages, "no package found; this gate checks nothing"
    unnamed = [p for p in packages if not re.search(rf"(?<![\w-]){re.escape(p)}(?![\w-])", text)]
    assert not unnamed, "named on no front-door page (" + ", ".join(_FRONT_DOOR) + "): " + (
        ", ".join(unnamed)
    )
