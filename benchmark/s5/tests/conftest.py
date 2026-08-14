# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Put the repo root and the two source packages the harness reads on the path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
for path in (
    _ROOT,
    _ROOT / "packages" / "latence-benchmark" / "src",
    _ROOT / "packages" / "latence-retrieval" / "src",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(autouse=True)
def _pinned_thread_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test the determinism contract a real run has.

    ``run_matrix.require_deterministic`` refuses an unpinned BLAS thread count, and the suite is
    launched with only ``PYTHONHASHSEED`` pinned. Setting the thread variables here mirrors what
    an operator's shell provides, rather than weakening the refusal so the tests can pass — the
    tests that assert the refusal override this fixture on purpose.
    """
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
