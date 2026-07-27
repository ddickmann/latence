"""The Provider roster this workspace DECLARES, read off the package manifests.

Every Provider is already declared exactly once, in its own package's
``[project.entry-points."latence.providers"]`` block — that declaration is what
:class:`~latence_core.capability.ProviderRegistry` discovers at runtime. The conformance suite
used to restate all of it a second time as a hand-maintained ``frozenset`` of names plus a
hard-coded count, so shipping a Provider meant editing its ``pyproject.toml`` *and* a test in
another package, and the restatement drifted exactly as you would expect (its own comment claimed
"50 Providers" while the set held 59). Deriving the expectation from the manifests makes adding a
Provider ONE edit, and turns the assertion into something with real content: *the environment under
test contains every Provider this workspace ships*.

That is a stronger gate than the hand-list, not a weaker one. A Provider declared in a package the
dev/CI environment forgets to install used to be invisible on both sides of the comparison —
missing from the registry *and* missing from the hand-list — and the suite passed while the
Provider went entirely unconformed. Derived from the manifests it is present on the expected side
and absent from the registry, so it FAILs. Nothing here can silently skip: a workspace root that
cannot be found, a manifest that cannot be parsed, or a scan that finds no Providers at all is an
error, never an empty expectation that passes vacuously.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

#: The entry-point group every Provider registers under (mirrors
#: :data:`latence_core.capability.ENTRY_POINT_GROUP`; restated as a literal because this module
#: reads static manifests, not the installed environment).
ENTRY_POINT_GROUP = "latence.providers"


def workspace_root() -> Path:
    """The repo root: the nearest ancestor whose ``pyproject.toml`` declares the uv workspace."""
    for candidate in Path(__file__).resolve().parents:
        manifest = candidate / "pyproject.toml"
        if not manifest.is_file():
            continue
        if "[tool.uv.workspace]" in manifest.read_text(encoding="utf-8"):
            return candidate
    msg = (
        f"no uv workspace root above {Path(__file__).resolve()} — the declared-Provider roster "
        f"cannot be derived, and an empty roster must never be mistaken for 'no Providers'."
    )
    raise RuntimeError(msg)


def providers_declared_in(root: Path) -> dict[str, str]:
    """Map every declared Provider name to the package directory that declares it.

    Raises when two packages declare the same name (the installed registry would silently keep one
    of them, so a collision has to fail here) or when the scan finds nothing at all.
    """
    manifests = sorted((root / "packages").glob("*/pyproject.toml"))
    if not manifests:
        msg = f"no package manifests under {root / 'packages'} — nothing to derive a roster from."
        raise RuntimeError(msg)
    declared: dict[str, str] = {}
    for manifest in manifests:
        package = manifest.parent.name
        project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
        entry_points = project.get("entry-points", {}).get(ENTRY_POINT_GROUP, {})
        for name in entry_points:
            if name in declared:
                msg = (
                    f"Provider {name!r} is declared twice: in {declared[name]!r} and in "
                    f"{package!r}. Entry-point names are global, so one of them would silently "
                    f"shadow the other in the installed environment."
                )
                raise RuntimeError(msg)
            declared[name] = package
    if not declared:
        msg = (
            f"no '{ENTRY_POINT_GROUP}' entry points found under {root / 'packages'} — refusing to "
            f"report an empty roster, which would let the conformance gate pass vacuously."
        )
        raise RuntimeError(msg)
    return declared


def declared_providers() -> dict[str, str]:
    """Every Provider this workspace declares, mapped to its declaring package directory."""
    return providers_declared_in(workspace_root())
