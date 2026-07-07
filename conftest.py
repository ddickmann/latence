"""Repo-root pytest configuration: verify the string-hash seed is pinned.

Some code paths iterate dicts/sets keyed by strings, whose order depends on the
per-process hash seed. To make those reproducible the suite runs with
``PYTHONHASHSEED=0``.

The seed is read *once* at interpreter startup, so it must be set in the
environment *before* the process launches — CI does this via a job-level env var
(``.github/workflows/ci.yml``) and locally ``make test`` does it. Setting it from
a fixture (``os.environ[...] = "0"``) is a no-op for the running interpreter and
only misleads; that is why there is no such fixture.

This hook is the *guard*: it fails fast, with one clear message, if the seed was
not actually pinned — instead of silently running non-deterministically.
"""

from __future__ import annotations

import sys

import pytest


def pytest_configure(config: pytest.Config) -> None:
    if sys.flags.hash_randomization:
        raise pytest.UsageError(
            "PYTHONHASHSEED is not pinned, so string-hash ordering is randomized "
            "and hash-order-dependent tests are non-deterministic. Run the suite "
            "with the seed pinned at launch, e.g. `make test` or "
            "`PYTHONHASHSEED=0 uv run pytest`."
        )
