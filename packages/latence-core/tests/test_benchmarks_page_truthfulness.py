"""The benchmark guide may not describe numbers the results page does not carry.

``docs/running-the-benchmarks.md`` is the page that tells an outsider the published numbers are
"not something you have to take on faith" — and then tables the suites and says what each one
*publishes*. Before the R1 audit fix it described five suites in that present tense while
``docs/BENCHMARKS.md`` carried exactly one (Parse) and ``docs/benchmarks/results.json`` carried a
one-element ``measurements`` array, with no *Not yet measured* section anywhere: the reader
following ``README.md``'s "public benchmark results" link found no redaction number at all, and
no admission that there wasn't one.

These tests therefore compare two sets derived from the repository itself, never a flag that
records intent:

* the **claim set** — the suites ``docs/running-the-benchmarks.md`` tables as reporting numbers;
* the **published set** — the suites that actually have a section on ``docs/BENCHMARKS.md`` and a
  measurement in the JSON sidecar.

Every claimed suite must be published *or* declared under an explicit **Not yet measured**
heading with the command that would produce it. Redaction is checked by name on top of that: its
corpus ships inside ``latence-benchmark`` and is pinned by content digest, so it runs offline
from a clean clone and cannot honestly be parked as unmeasured.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_GUIDE = _REPO_ROOT / "docs" / "running-the-benchmarks.md"
_PAGE = _REPO_ROOT / "docs" / "BENCHMARKS.md"
_SIDECAR = _REPO_ROOT / "docs" / "benchmarks" / "results.json"

#: ``## Redaction — `redaction` suite`` on the published page.
_PAGE_SECTION = re.compile(r"^## .*—\s*`([a-z_]+)`\s+suite\s*$", re.MULTILINE)
#: A leading ``| `redaction` | ...`` cell in the guide's suite table.
_GUIDE_TABLE_SUITE = re.compile(r"^\|\s*`([a-z_]+)`\s*\|", re.MULTILINE)


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(_REPO_ROOT)} must exist"
    return path.read_text(encoding="utf-8")


def _section(text: str, title_fragment: str) -> str | None:
    """Body of the first ``##`` section whose heading contains ``title_fragment``."""
    for chunk in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        if title_fragment.lower() in chunk.splitlines()[0].lower():
            return chunk
    return None


def _claimed_suites() -> set[str]:
    """Suites the guide's overview table says report numbers."""
    guide = _text(_GUIDE)
    head = guide.split("## What the harness refuses to do")[0]
    claimed = set(_GUIDE_TABLE_SUITE.findall(head))
    assert claimed, "the guide no longer tables any suite — this guard has gone blind"
    return claimed


def _published_suites() -> set[str]:
    page = set(_PAGE_SECTION.findall(_text(_PAGE)))
    sidecar = {m["suite"] for m in json.loads(_text(_SIDECAR))["measurements"]}
    assert page == sidecar, (
        "the markdown page and the JSON sidecar disagree about which suites are published: "
        f"page={sorted(page)} sidecar={sorted(sidecar)}"
    )
    return page


def _parked_suites() -> set[str]:
    """Suites the guide explicitly declares unmeasured, PERF-RESULTS.md style."""
    section = _section(_text(_GUIDE), "Not yet measured")
    if section is None:
        return set()
    return set(re.findall(r"`([a-z_]+)`", section))


def test_every_suite_the_guide_describes_is_published_or_declared_unmeasured() -> None:
    """The claim set minus the published set must be explicitly parked, never silent."""
    undeclared = _claimed_suites() - _published_suites() - _parked_suites()
    assert not undeclared, (
        "docs/running-the-benchmarks.md describes suites that docs/BENCHMARKS.md does not "
        f"publish and that no 'Not yet measured' section declares: {sorted(undeclared)}"
    )


def test_the_offline_redaction_number_is_actually_published() -> None:
    """The privacy claim's number ships in-package; parking it as unmeasured is not honest."""
    assert "redaction" in _published_suites(), (
        "the redaction suite runs offline from a clean clone (its synthetic corpus ships inside "
        "latence-benchmark, pinned by content digest) — its number must be on the results page"
    )
    measurement = next(
        m for m in json.loads(_text(_SIDECAR))["measurements"] if m["suite"] == "redaction"
    )
    dimensions = {slice_["dimension"] for slice_ in measurement["metrics"]}
    # The guide promises per-language, per-PII-type and per-criterion slices. Publishing only a
    # headline would leave those sentences false in a subtler way.
    assert {"language", "pii_type", "match"} <= dimensions, (
        f"the published redaction measurement is missing slices the guide promises: {dimensions}"
    )
    assert measurement["headline"]["recall"] is not None, "no published redaction recall"


def test_a_parked_suite_states_how_to_produce_its_number() -> None:
    """'Not yet measured' is a commitment, not an excuse — it carries the repro command."""
    parked = _parked_suites()
    if not parked:
        return
    section = _section(_text(_GUIDE), "Not yet measured")
    assert section is not None
    assert "latence-benchmark run" in section, (
        "the 'Not yet measured' section must state the command that produces the missing numbers"
    )


@pytest.mark.parametrize("suite", sorted(_published_suites()))
def test_every_published_measurement_states_when_it_was_measured(suite: str) -> None:
    """A page assembled from several runs must not imply every number is from the last one."""
    measurement = next(
        m for m in json.loads(_text(_SIDECAR))["measurements"] if m["suite"] == suite
    )
    measured_on = measurement.get("measured_on")
    assert measured_on and re.fullmatch(r"\d{4}-\d{2}-\d{2}", measured_on), (
        f"published measurement for {suite!r} carries no measurement date: {measured_on!r}"
    )
    assert f"Measured {measured_on}" in _text(_PAGE), (
        f"the {suite!r} section does not render its measurement date"
    )
