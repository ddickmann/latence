"""Fail the build if an EXCLUDED distribution made it into the image anyway (#196).

The exclusions are enforced by pins (``constraints.txt`` pins ``flash-attn`` / ``torchcodec`` to
``<0``, which no release satisfies) and that pinning is what actually prevents the install. This
script is the independent *proof*: it asserts the finished image cannot import them, so a base
image that shipped one pre-installed — a case constraints cannot touch, because nothing resolved
it — still fails the build instead of shipping.

It derives the list from the committed ``constraints.txt`` rather than repeating it, so an
exclusion added to ``environment.yaml`` is checked here automatically and the two cannot drift.

Stdlib only. It runs inside the image (no argument — the baked-in constraints path) and equally
against a bare-metal install (``python verify-exclusions.py path/to/constraints.txt``).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

DEFAULT_CONSTRAINTS = Path("/etc/latence/ocr-constraints.txt")

# `flash-attn` the distribution installs `flash_attn` the module. PEP 503 normalisation maps the
# other way for every name here, so the mechanical rule is right; anything exotic would need an
# explicit entry, and an unknown name is checked by distribution metadata regardless.
def _module_name(distribution: str) -> str:
    return distribution.replace("-", "_").replace(".", "_")


def excluded_distributions(text: str) -> list[str]:
    """The distributions pinned to the unsatisfiable ``<0`` specifier, in file order."""
    names = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.endswith("<0"):
            names.append(line[: -len("<0")].strip())
    return names


def main(argv: list[str]) -> int:
    constraints = Path(argv[1]) if len(argv) > 1 else DEFAULT_CONSTRAINTS
    if not constraints.is_file():
        print(f"FAIL: {constraints} is missing — the image was built without its boundaries.")
        return 2
    excluded = excluded_distributions(constraints.read_text(encoding="utf-8"))
    if not excluded:
        # An empty check that reports success is exactly the false green this repo refuses.
        print(f"FAIL: no exclusions in {constraints}; refusing to pass a vacuous check.")
        return 2

    failures = []
    for distribution in excluded:
        try:
            version = importlib.metadata.distribution(distribution).version
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            failures.append(f"{distribution}=={version} is INSTALLED (distribution metadata)")
        module = _module_name(distribution)
        if importlib.util.find_spec(module) is not None:
            failures.append(f"{distribution}: module '{module}' is importable")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        print(
            "\nThese distributions are excluded from the served-OCR serving environment: their "
            "prebuilt wheels were ABI-broken by the torch 2.8+cu128 -> 2.11+cu130 move and vLLM "
            "does not need them. See deploy/env/served-ocr-server/environment.yaml."
        )
        return 1
    print(f"OK: none of {', '.join(excluded)} is installed or importable.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
