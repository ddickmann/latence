"""Make this package importable in the workspace dev env without pinning it as a dev dependency.

``latence-label-profiler`` is a workspace member (``packages/*``), but it is deliberately NOT added
to the root dev meta-project's ``dependencies``: doing so would rewrite ``uv.lock`` for every lane
working in parallel on this branch set. The package is pure-stdlib and dependency-free, so putting
its ``src`` on ``sys.path`` is enough to run the REAL code — no stub, no mock, no skip.

The integrator pinning this package for CI should add it to the root ``[project].dependencies`` +
``[tool.uv.sources]`` (as ``latence-demo`` is) and delete this file; nothing else changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
