"""Console test fixtures — a real Storage carrying real run files.

The builders live in :mod:`reports` (a uniquely-named module, because every package's
``tests/conftest.py`` maps to the same bare ``conftest`` module — ADR-0032).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.storage import Storage
from reports import delta_report, failed_report, gapped_report, healthy_report, write_run


@pytest.fixture
def storage_uri(tmp_path: Path) -> str:
    return f"file://{tmp_path}/corpus"


@pytest.fixture
def storage(storage_uri: str) -> Storage:
    return Storage.for_uri(storage_uri)


@pytest.fixture
def populated(storage: Storage, storage_uri: str) -> str:
    """A Storage carrying one run of each outcome, plus an unfinished one."""
    write_run(storage, storage_uri, "run-0001", healthy_report("run-0001"))
    write_run(storage, storage_uri, "run-0002", gapped_report("run-0002"))
    write_run(storage, storage_uri, "run-0003", failed_report("run-0003"))
    write_run(storage, storage_uri, "run-0004", delta_report("run-0004"))
    write_run(storage, storage_uri, "run-0005", None)
    return storage_uri
