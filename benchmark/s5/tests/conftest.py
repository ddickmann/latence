# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Put the repo root and the two source packages the harness reads on the path."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for path in (
    _ROOT,
    _ROOT / "packages" / "latence-benchmark" / "src",
    _ROOT / "packages" / "latence-retrieval" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
