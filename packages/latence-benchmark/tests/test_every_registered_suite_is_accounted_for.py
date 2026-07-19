"""Every suite the *registry* knows about must be accounted for in the published docs.

``test_benchmarks_page_truthfulness.py`` already checks that the guide does not describe numbers
the results page lacks. But it derives the set of suites that exist from **prose** — the suite
table in ``docs/running-the-benchmarks.md``. That makes the documentation its own source of truth,
which is exactly the shape of the defect it was written to prevent.

The consequence is a blind spot with teeth: a suite that is registered but *absent from the guide*
is claimed nowhere, published nowhere and parked nowhere, so no gate has anything to compare. That
is the mechanism by which #187 shipped four suites whose numbers were published nowhere while the
build stayed green — the docs were never the authority, so consistency-with-the-docs could not
notice a suite the docs had never heard of. It also means the cheapest way to turn the existing
guard green is to *delete a row* rather than publish a number.

This module closes that by taking :func:`~latence_benchmark.suite.suite_names` as the claim set —
the same registry ``--suite all`` iterates, so it cannot drift from what the harness actually runs.
Each registered suite must be either **published** (a section on ``docs/BENCHMARKS.md`` *and* a
measurement in the sidecar) or **explicitly parked** under *Not yet measured* with the command that
would produce it. Unpublished is fine; undisclosed is not.

Like the freshness gate, this asserts nothing about any pass rate, precision or recall. It needs no
dataset, no network and no Provider — only the registry and two committed files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import latence_benchmark  # noqa: F401  -- import registers every built-in suite
import pytest
from latence_benchmark.suite import suite_names

# tests/  ->  latence-benchmark/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GUIDE = _REPO_ROOT / "docs" / "running-the-benchmarks.md"
_PAGE = _REPO_ROOT / "docs" / "BENCHMARKS.md"
_SIDECAR = _REPO_ROOT / "docs" / "benchmarks" / "results.json"

#: ``## Redaction — `redaction` suite`` on the published page.
_PAGE_SECTION = re.compile(r"^## .*—\s*`([a-z_]+)`\s+suite\s*$", re.MULTILINE)

_REGENERATE = "uv run latence-benchmark run --suite all"


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(_REPO_ROOT)} must exist"
    return path.read_text(encoding="utf-8")


def _registered() -> set[str]:
    names = set(suite_names())
    assert names, "no benchmark suite is registered — this guard has gone blind"
    return names


def _published() -> set[str]:
    page = set(_PAGE_SECTION.findall(_text(_PAGE)))
    sidecar = {m["suite"] for m in json.loads(_text(_SIDECAR))["measurements"]}
    assert page == sidecar, (
        "the markdown page and the JSON sidecar disagree about which suites are published: "
        f"page={sorted(page)} sidecar={sorted(sidecar)}"
    )
    return page


def _parked() -> set[str]:
    """Suites the guide explicitly declares unmeasured, under its own heading."""
    for chunk in re.split(r"^## ", _text(_GUIDE), flags=re.MULTILINE)[1:]:
        if "not yet measured" in chunk.splitlines()[0].lower():
            return set(re.findall(r"`([a-z_]+)`", chunk)) & _registered()
    return set()


def test_every_registered_suite_is_published_or_explicitly_parked() -> None:
    """A suite the harness runs may not be missing from the docs without an admission."""
    unaccounted = _registered() - _published() - _parked()
    assert not unaccounted, (
        "these suites are registered — `--suite all` runs them — but they have no section on "
        "docs/BENCHMARKS.md, no measurement in docs/benchmarks/results.json, and no entry under "
        f"'Not yet measured' in docs/running-the-benchmarks.md: {sorted(unaccounted)}. "
        f"Publish them (`{_REGENERATE}`) or declare them unmeasured with the repro command. "
        "Removing them from the guide's table is not a fix: the registry is the claim set."
    )


def test_the_results_page_publishes_no_suite_that_no_longer_exists() -> None:
    """A renamed or deleted suite must not leave an orphan section asserting stale numbers."""
    orphaned = _published() - _registered()
    assert not orphaned, (
        f"docs/BENCHMARKS.md publishes measurements for suites that are not registered: "
        f"{sorted(orphaned)}. A suite that no longer exists cannot be reproduced, so its numbers "
        f"may not stay on the page — re-run `{_REGENERATE}`."
    )


def test_a_parked_suite_is_a_real_suite_stated_with_its_repro_command() -> None:
    """'Not yet measured' must name registered suites and say how to measure them."""
    parked = _parked()
    if not parked:
        return
    assert not (parked & _published()), (
        f"suites are both published and declared unmeasured: {sorted(parked & _published())}"
    )
    section = next(
        chunk
        for chunk in re.split(r"^## ", _text(_GUIDE), flags=re.MULTILINE)[1:]
        if "not yet measured" in chunk.splitlines()[0].lower()
    )
    assert "latence-benchmark run" in section, (
        "the 'Not yet measured' section must state the command that produces the missing numbers"
    )


@pytest.mark.parametrize("suite", sorted(suite_names()))
def test_each_registered_suite_is_named_in_the_guide(suite: str) -> None:
    """The guide is the reader's index of what exists; a registered suite may not be omitted."""
    assert f"`{suite}`" in _text(_GUIDE), (
        f"suite {suite!r} is registered and runnable but is never mentioned in "
        "docs/running-the-benchmarks.md, so a reader has no way to learn it exists"
    )
