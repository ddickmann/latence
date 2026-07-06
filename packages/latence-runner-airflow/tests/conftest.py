"""Shared fixtures for the Airflow adapter tests (deterministic, offline)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _deterministic_env() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture()
def docs_dir(tmp_path: Path) -> Path:
    """A folder of two deterministic text documents (mirrors the core suite's fixture)."""
    d = tmp_path / "docs"
    d.mkdir()
    (d / "hello.txt").write_text(
        "the framework reads the folder and exports the records\n", encoding="utf-8"
    )
    (d / "notes.md").write_text("# Notes\n\nthe and of to is in that it\n", encoding="utf-8")
    return d
