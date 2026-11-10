"""Every workspace package is installed by the documented setup, and the list lives in one place.

The heavy/endpoint Provider packages are workspace members that are deliberately NOT dependencies
of the dev environment (ADR-0007/0016), so their source is installed separately with ``--no-deps``.
That list used to exist as three hand-synchronised copies — ``Makefile``, ``scripts/verify-local.sh``
and ``.github/actions/setup-workspace/action.yml`` — with a comment in the CI action promising it was
"kept byte-identical" and nothing enforcing it.

It was not identical. The ``Makefile`` copy had drifted to **9 of 21** entries, so a contributor who
followed ``AGENTS.md`` — ``make sync`` then ``make test`` — got a red suite for environment reasons:
twelve Providers never registered their entry points, which surfaces as *registry* failures
(``providers_total`` short of what the manifests declare) rather than as an import error, so it does
not read as a setup problem.

This module gates the repair two ways: the list is single-sourced in
``scripts/heavy-packages.txt`` and no consumer re-hardcodes it, and — the invariant that actually
matters — **every** package under ``packages/`` is accounted for by one of the three mechanisms that
can make it importable.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIST = _REPO_ROOT / "scripts" / "heavy-packages.txt"

#: Packages that are neither a root dependency nor in the heavy list, with the reason each is
#: nonetheless importable under pytest. An entry here is a claim that must stay checkable, so the
#: test below verifies the stated mechanism still exists rather than trusting the comment.
_EXEMPT = {
    # Pure-stdlib and dependency-free. Its own packages/latence-label-profiler/conftest.py puts
    # src/ on sys.path, running the REAL code with no stub and no skip, deliberately avoiding a
    # uv.lock rewrite across parallel lanes. That file names its own exit condition: an integrator
    # pinning it for CI adds it to [project].dependencies + [tool.uv.sources] and deletes it.
    "latence-label-profiler": "conftest.py",
}


def _heavy_packages() -> list[str]:
    """The single-source list, with blank lines and ``#`` comments stripped."""
    return [
        line.strip()
        for line in _LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _root_dependencies() -> set[str]:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {re.split(r"[<>=!\[ ]", dep)[0] for dep in data["project"]["dependencies"]}


def _workspace_packages() -> set[str]:
    return {p.name for p in (_REPO_ROOT / "packages").iterdir() if (p / "pyproject.toml").is_file()}


def test_the_list_is_non_vacuous_and_every_entry_is_a_real_package() -> None:
    heavy = _heavy_packages()
    assert heavy, f"{_LIST} parsed to an empty list — the strip logic or the file is broken"
    assert len(heavy) == len(set(heavy)), f"duplicate entries in {_LIST}: {heavy}"

    missing = [p for p in heavy if not (_REPO_ROOT / "packages" / p / "pyproject.toml").is_file()]
    assert not missing, (
        f"{_LIST} names package(s) that do not exist: {missing}. The install loops skip a missing "
        "directory silently, so a typo here removes a package from the environment without any "
        "visible failure until its tests fail for unrelated-looking reasons."
    )


def test_every_workspace_package_is_installed_by_the_documented_setup() -> None:
    """The invariant the three drifted copies violated."""
    uncovered = _workspace_packages() - _root_dependencies() - set(_heavy_packages()) - set(_EXEMPT)
    assert not uncovered, (
        f"workspace package(s) reachable by no documented install path: {sorted(uncovered)}. "
        f"Each must be a root dependency in pyproject.toml, an entry in {_LIST.name}, or carry a "
        "reasoned entry in _EXEMPT. Otherwise `make sync` leaves it unregistered and its Provider "
        "entry points are missing from the registry at test time."
    )


def test_each_exemption_still_has_the_mechanism_it_claims() -> None:
    """An exemption is a claim about the tree, not a permanent waiver."""
    for package, mechanism in _EXEMPT.items():
        pkg_dir = _REPO_ROOT / "packages" / package
        assert pkg_dir.is_dir(), f"_EXEMPT names {package}, which no longer exists — drop the entry"
        assert (pkg_dir / mechanism).is_file(), (
            f"_EXEMPT claims {package} is importable via {mechanism}, but that file is gone. "
            "Either the package was pinned as a real dependency (drop the exemption) or it is now "
            "silently untested."
        )


def test_no_consumer_re_hardcodes_the_list() -> None:
    """Single-sourcing only holds while every consumer actually reads the file."""
    consumers = (
        _REPO_ROOT / "Makefile",
        _REPO_ROOT / "scripts" / "verify-local.sh",
        _REPO_ROOT / ".github" / "actions" / "setup-workspace" / "action.yml",
    )
    # Two entries that appear in no other list, so a re-hardcoded copy is unmistakable.
    canaries = ("latence-ner-gliner", "latence-disambig-glinker")
    for consumer in consumers:
        text = consumer.read_text(encoding="utf-8")
        assert "heavy-packages.txt" in text, (
            f"{consumer.relative_to(_REPO_ROOT)} no longer reads {_LIST.name}. The list is "
            "single-sourced precisely because three hand-synchronised copies drifted."
        )
        inlined = [c for c in canaries if c in text]
        assert not inlined, (
            f"{consumer.relative_to(_REPO_ROOT)} inlines {inlined} instead of reading "
            f"{_LIST.name} — that is the drift this gate exists to prevent."
        )
