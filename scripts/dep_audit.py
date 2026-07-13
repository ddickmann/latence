#!/usr/bin/env python3
"""Dependency-completeness audit for the latence-framework monorepo.

For every package under ``packages/<pkg>/src/``, parse the top-level third-party
modules it actually imports, map import-name -> distribution-name, and check each
against that package's declared ``[project].dependencies`` /
``[project.optional-dependencies]`` (plus ``latence-core``-transitive deps). Reports
any imported third-party module whose distribution is NOT covered — the class of bug
that makes a clean ``pip install <package>`` (no --no-deps) fail with
``ModuleNotFoundError`` on a fresh machine (e.g. the ``peft`` gap).

Nested (inside-a-function) imports are marked non-module-level: those are legitimately
covered by a lazy/optional extra with a graceful ImportError. Vendored ``_vendor/``
trees and relative imports are excluded.

Run from the repo root:  python3 scripts/dep_audit.py
Exit code is non-zero if a genuine (module-level, uncovered) gap is found.
"""
from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKGS = ROOT / "packages"
STDLIB = set(sys.stdlib_module_names) | {"__future__"}

# import top-level name -> PyPI distribution name (only where they differ)
DIST = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "sentence_transformers": "sentence-transformers",
    "presidio_analyzer": "presidio-analyzer",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "bs4": "beautifulsoup4",
    "torch_geometric": "torch-geometric",
    "torch_scatter": "torch-scatter",
    "airflow": "apache-airflow",
}
# transitive deps a package inherits by declaring latence-core
CORE_TRANSITIVE = {"pydantic", "fsspec", "pyarrow", "pyyaml", "typer"}


def _imports(src_dir: Path) -> dict[str, bool]:
    """module top-level name -> module_level (True if any use is a module-level import)."""
    out: dict[str, bool] = {}
    for py in src_dir.rglob("*.py"):
        if "/_vendor/" in str(py):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [n.name.split(".")[0] for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    names = [node.module.split(".")[0]]
            if not names:
                continue
            module_level = node.col_offset == 0
            for m in names:
                out[m] = out.get(m, False) or module_level
    return out


def _own(pkg: Path) -> set[str]:
    src = pkg / "src"
    return {c.name for c in src.iterdir() if c.is_dir()} if src.exists() else set()


def _declared(pkg: Path):
    data = tomllib.loads((pkg / "pyproject.toml").read_text())
    proj = data.get("project", {})
    deps = list(proj.get("dependencies", []))
    extras = {k: list(v) for k, v in proj.get("optional-dependencies", {}).items()}
    return deps, extras


def _dist(dep: str) -> str:
    s = dep.split(";")[0].strip()
    for sep in ("[", ">", "<", "=", "!", "~", " "):
        if sep in s:
            s = s.split(sep)[0]
    return s.strip().lower()


def main() -> int:
    rows: list[tuple[str, str, str, bool, str]] = []
    gaps: list[tuple[str, str, str]] = []
    for pkg in sorted(p for p in PKGS.iterdir() if (p / "src").exists()):
        imports = _imports(pkg / "src")
        own = _own(pkg)
        deps, extras = _declared(pkg)
        dep_dists = {_dist(d) for d in deps}
        has_core = "latence-core" in dep_dists
        extra_dists: dict[str, list[str]] = {}
        for ek, evs in extras.items():
            for ev in evs:
                extra_dists.setdefault(_dist(ev), []).append(ek)
        for m, module_level in sorted(imports.items()):
            if m in STDLIB or m in own or m.startswith("latence"):
                continue
            dist = DIST.get(m, m).lower()
            if dist in dep_dists:
                covered = "declared"
            elif has_core and dist in CORE_TRANSITIVE:
                covered = "core"
            elif dist in extra_dists:
                covered = f"extra[{','.join(extra_dists[dist])}]"
            else:
                covered = "*** NOT COVERED ***"
                if module_level:  # only a hard, module-level import is a real gap
                    gaps.append((pkg.name, m, dist))
            rows.append((pkg.name, m, dist, module_level, covered))

    w = max((len(r[0]) for r in rows), default=10)
    print(f"{'PKG':<{w}}  {'IMPORT':<22} {'DIST':<20} {'MOD-LVL':<8} COVERED-BY")
    print("-" * (w + 62))
    for name, m, dist, ml, cov in rows:
        print(f"{name:<{w}}  {m:<22} {dist:<20} {str(ml):<8} {cov}")

    print("\n=== GENUINE GAPS (module-level import, not covered) ===")
    if gaps:
        for g in gaps:
            print(f"  {g[0]}: '{g[1]}' -> {g[2]}")
        return 1
    print("  (none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
