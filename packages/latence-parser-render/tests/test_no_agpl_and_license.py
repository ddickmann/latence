"""The license discipline gate: NO fitz/PyMuPDF (AGPL) anywhere, permissive deps cited.

The whole reason for this clean port (ADR-0012 / research-diligence): colsearch renders PDF with
PyMuPDF (``fitz``) — AGPL-3.0, copyleft. This slice ports the converter WITHOUT it. These tests
enforce it structurally (a grep over the installed package source) and assert the evidence-bearing
permissive license the profile declares.
"""

from __future__ import annotations

from pathlib import Path

import latence_parser_render
from latence_core.providers.profile import REPO_LICENSE_CODE
from latence_parser_render.provider import UniversalRenderParser

# The forbidden AGPL PDF library (and its import spellings) — NEVER present in this package.
_FORBIDDEN = ("fitz", "pymupdf", "PyMuPDF")


def _package_sources() -> list[Path]:
    root = Path(latence_parser_render.__file__).parent
    return list(root.rglob("*.py"))


def test_no_fitz_or_pymupdf_imported_anywhere() -> None:
    """A grep over the installed package: no ``import fitz`` / PyMuPDF (AGPL) — pypdfium2 only."""
    offenders: list[str] = []
    for path in _package_sources():
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            # Only flag genuine import statements, not prose in docstrings/comments naming the ban.
            if stripped.startswith(("import ", "from ")) and any(
                token in stripped for token in ("fitz", "pymupdf", "PyMuPDF")
            ):
                offenders.append(f"{path.name}: {stripped}")
    assert not offenders, f"AGPL PDF lib imported (forbidden): {offenders}"


def test_pyproject_declares_no_agpl_dep() -> None:
    """The pyproject dependency list names no fitz/PyMuPDF — PDF is pypdfium2 (Apache/BSD)."""
    pyproject = Path(latence_parser_render.__file__).parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    # Extract the [project].dependencies array crudely and assert no forbidden token in a dep line.
    in_deps = False
    for line in text.splitlines():
        if line.strip().startswith("dependencies = ["):
            in_deps = True
            continue
        if in_deps:
            if line.strip() == "]":
                break
            dep = line.strip().strip('",')
            assert not any(f.lower() in dep.lower() for f in _FORBIDDEN), f"AGPL dep: {dep}"
    assert "pypdfium2" in text  # the permissive PDF renderer IS declared


def test_profile_license_is_permissive_and_evidence_bearing() -> None:
    """The declared license is verified, permissive, and cites where a human checked (H-B1)."""
    profile = UniversalRenderParser.profile()
    assert profile.license == REPO_LICENSE_CODE
    assert profile.license_verified is True
    assert profile.license_verified_on == "2026-07-09"
    assert profile.license_source is not None
    source = profile.license_source
    # The citation names the permissive deps AND the AGPL-free discipline (evidence-bearing).
    for token in ("pypdfium2", "Pillow", "MIT", "LibreOffice", "NO fitz"):
        assert token in source
