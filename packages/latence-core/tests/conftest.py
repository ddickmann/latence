"""Shared fixtures. Determinism is enforced via PYTHONHASHSEED in CI and pytest addopts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _deterministic_env() -> None:
    # Belt-and-suspenders: keep any hash-dependent ordering stable within a test.
    os.environ.setdefault("PYTHONHASHSEED", "0")


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
