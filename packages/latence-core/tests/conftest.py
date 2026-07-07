"""Shared fixtures. Determinism (a pinned PYTHONHASHSEED) is enforced and verified
by the repo-root conftest, not here."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """A folder of two deterministic text documents."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "hello.txt").write_text(
        "the framework reads the folder and exports the records\n", encoding="utf-8"
    )
    (d / "notes.md").write_text("# Notes\n\nthe and of to is in that it\n", encoding="utf-8")
    # A non-text file that Source must skip.
    (d / "ignore.bin").write_bytes(b"\x00\x01\x02")
    return d
