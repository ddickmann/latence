"""Regression guards for issue #24: strict mypy over src + tests, and the conftest collision.

These are deterministic, offline, dependency-free repo-hygiene tests. They fail if the
before-G1 debt fix is reverted:

* every workspace package that ships a source distribution advertises inline types with a
  ``py.typed`` marker (without it, a package's OWN tests import it as an untyped module and
  strict mypy silently degrades to ``Any``, hiding real findings);
* the duplicate-``conftest`` collision that blocks a single ``mypy --strict packages`` run
  still exists structurally (≥2 ``tests/conftest.py`` under hyphenated package dirs), so the
  per-package ``scripts/typecheck.sh`` loop is the sanctioned enforcement — and both that
  script and its CI wiring are present.
"""

from __future__ import annotations

from pathlib import Path

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"


def _source_packages() -> list[Path]:
    """Every ``packages/<pkg>`` that has a ``src/<module>`` layout."""
    out: list[Path] = []
    for pkg in sorted(_PACKAGES.iterdir()):
        src = pkg / "src"
        if src.is_dir() and any(p.is_dir() for p in src.iterdir()):
            out.append(pkg)
    return out


def test_every_source_package_ships_py_typed() -> None:
    """A package whose src is strictly type-checked must advertise it (PEP 561)."""
    missing: list[str] = []
    for pkg in _source_packages():
        (module_dir,) = [p for p in (pkg / "src").iterdir() if p.is_dir()]
        if not (module_dir / "py.typed").is_file():
            missing.append(f"{pkg.name} ({module_dir.name})")
    assert not missing, f"packages missing a py.typed marker: {missing}"


def test_conftest_collision_is_real_so_per_package_loop_is_required() -> None:
    """Document WHY the enforcement is a per-package loop, not one whole-tree invocation.

    mypy maps a file to a module by its path; the package dirs are hyphenated, so it cannot
    build a dotted name and every ``tests/conftest.py`` becomes the bare module ``conftest``.
    Two or more of them make a single ``mypy --strict packages`` abort with
    ``Duplicate module named "conftest"``. If this guard ever sees <2 conftests, the collision
    rationale is stale and the single-invocation shortcut should be reconsidered.
    """
    conftests = sorted(_PACKAGES.glob("*/tests/conftest.py"))
    # All of them map to the SAME top-level module name under a naive whole-tree run.
    derived_modules = {c.stem for c in conftests}
    assert derived_modules == {"conftest"}
    assert len(conftests) >= 2, (
        "expected the duplicate-conftest collision the per-package typecheck loop works around"
    )


def test_strict_typecheck_script_and_ci_wiring_present() -> None:
    """The sanctioned strict-over-src+tests entrypoint exists and CI invokes it."""
    script = _REPO_ROOT / "scripts" / "typecheck.sh"
    assert script.is_file(), "scripts/typecheck.sh (the strict src+tests entrypoint) is missing"
    body = script.read_text(encoding="utf-8")
    # It must check tests, not just src, and run strict.
    assert "--strict" in body
    assert "/tests" in body

    ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/typecheck.sh" in ci, "CI no longer runs the src+tests strict typecheck script"
