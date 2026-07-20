"""Regression guards for the published throughput envelopes (issue #180).

``PERF-RESULTS.md`` is a *buyer-facing* document: an operator sizes a cluster from it. The
failure mode it must never have is an unprovenanced or unreproducible number, so these tests
assert the properties that make a published number trustworthy, not that a particular string
is present:

* a **CPU** envelope and a **GPU** envelope both exist, each covering per-Stage throughput;
* the CPU envelope covers **every Stage of the stack it names** — a table that quietly omits
  the slow Stage would flatter the engine;
* every envelope states **hardware, stack profile, corpus shape, date and a repro command**;
* the large-scale (100k/1M) numbers stay **parked and referenced**, never previewed;
* the document is **reachable** from the README and from the docs site;
* the documented reproduction command **actually runs and measures every Stage** — the one
  check that keeps the table from drifting into fiction as the code changes.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERF = _REPO_ROOT / "PERF-RESULTS.md"
_BENCH = _REPO_ROOT / "scripts" / "bench_throughput.py"
_DEFAULT_STACK = _REPO_ROOT / "stacks" / "default.yaml"


def _perf_text() -> str:
    assert _PERF.is_file(), "PERF-RESULTS.md must exist at the repo root"
    return _PERF.read_text(encoding="utf-8")


def _section(title_fragment: str) -> str:
    """The body of the first ``##`` section whose heading contains ``title_fragment``."""
    text = _perf_text()
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    matches = [s for s in sections if title_fragment.lower() in s.splitlines()[0].lower()]
    assert matches, f"PERF-RESULTS.md has no '## ...{title_fragment}...' section"
    return matches[0]


def _stack_stage_names() -> list[str]:
    spec = yaml.safe_load(_DEFAULT_STACK.read_text(encoding="utf-8"))
    return [stage["name"] for stage in spec["stages"]]


def test_cpu_envelope_covers_every_stage_of_the_stack_it_names() -> None:
    """A per-Stage CPU table that omits a Stage would hide the slow one."""
    section = _section("CPU envelope")
    assert "stacks/default.yaml" in section, "the CPU envelope must name the stack profile it ran"
    missing = [name for name in _stack_stage_names() if f"| {name} |" not in section]
    assert not missing, f"CPU throughput table omits Stages: {missing}"
    # Throughput, not just latency: a buyer sizes a cluster in documents per hour.
    assert "docs/hour" in section
    assert re.search(r"\|\s*\*\*end-to-end\*\*\s*\|", section), "no end-to-end row"


def test_gpu_envelope_publishes_named_gpu_throughput() -> None:
    section = _section("GPU envelope")
    assert "NVIDIA RTX 2000 Ada" in section, "the GPU envelope must name the GPU"
    assert "pages/hour" in section, "the GPU envelope must publish throughput, not only s/page"
    assert "parser.lighton" in section, "the GPU envelope must name the Stage's Provider"


@pytest.mark.parametrize("envelope", ["CPU envelope", "GPU envelope"])
def test_every_envelope_states_hardware_profile_corpus_date_and_repro(envelope: str) -> None:
    """A throughput number without its provenance is not a result."""
    section = _section(envelope)
    for required in ("**Hardware:**", "**Stack profile:**", "**Corpus shape:**", "**Date measured"):
        assert required in section, f"{envelope} is missing {required}"
    assert "**Reproduce:**" in section, f"{envelope} has no reproduction steps"
    # A date the reader can check, not "recently" — and not an ADR cross-reference either
    # (an ADR id is not a date, and the ADR it points at may carry no date at all).
    assert re.search(r"\d{4}-\d{2}-\d{2}", section), f"{envelope} has no concrete date"


def test_every_date_measured_line_carries_an_iso_date_not_a_crossreference() -> None:
    """`**Date measured:**` must resolve on its own line, with no indirection.

    Regression guard: the GPU envelope once read "**Date measured:** the W8 acceleration run
    recorded in ADR-0036" — a cross-reference to an ADR section that itself carried no date, so
    the published GPU sizing numbers had no measurement date anywhere in the tree. An operator
    sizing a cluster cannot tell whether a driver/torch combination is still the one shipped by
    ``deploy/env/`` if the run has no date. A pointer is not a date: this asserts the ISO date is
    on the line itself, and that no ADR/issue/PR token can stand in for it.
    """
    iso = re.compile(r"\d{4}-\d{2}-\d{2}")
    lines = [line for line in _perf_text().splitlines() if "**Date measured" in line]
    assert lines, "PERF-RESULTS.md states no measurement date at all"
    undated = [line for line in lines if not iso.search(line)]
    assert not undated, f"'Date measured' lines without an ISO date on the line: {undated}"
    # And the substitute that used to satisfy the guard must not be the only date-shaped token.
    for line in lines:
        stripped = iso.sub("", line)
        assert not re.search(r"\b(ADR|#)-?\d+\b", stripped) or iso.search(line), line


@pytest.mark.parametrize("envelope", ["CPU envelope", "GPU envelope"])
def test_envelope_date_survives_removal_of_adr_crossreferences(envelope: str) -> None:
    """The date check must not be satisfiable by an ``ADR-\\d{4}`` token.

    Deleting every ADR reference from the section must leave the measurement date standing.
    """
    section = re.sub(r"ADR-\d{4}", "", _section(envelope))
    assert re.search(r"\d{4}-\d{2}-\d{2}", section), (
        f"{envelope}'s only date-shaped token is an ADR cross-reference, not a measurement date"
    )


def test_gpu_envelope_crossreferences_resolve_to_a_dated_adr_section() -> None:
    """If the GPU envelope points at an ADR for its provenance, that ADR must be dated too.

    The GPU envelope cites ADR-0036's W8 acceleration status note. A reader who follows that
    pointer must land on a section that states when the run happened — otherwise the citation
    launders an undated number through a second document.
    """
    section = _section("GPU envelope")
    adrs = set(re.findall(r"ADR-(\d{4})", section))
    assert adrs, "the GPU envelope no longer cites its ADR — update this guard deliberately"
    for number in sorted(adrs):
        matches = sorted((_REPO_ROOT / "docs" / "adr").glob(f"{number}-*.md"))
        assert matches, f"the GPU envelope cites ADR-{number}, which does not exist"
        body = matches[0].read_text(encoding="utf-8")
        note = [
            block
            for block in re.split(r"^## ", body, flags=re.MULTILINE)
            if "W8-perf-accel" in block.splitlines()[0]
        ]
        assert note, f"ADR-{number} has no W8-perf-accel status note to cite"
        assert re.search(r"\d{4}-\d{2}-\d{2}", note[0]), (
            f"ADR-{number}'s W8-perf-accel note states no measurement date, so the GPU "
            "envelope's cross-reference resolves to nothing checkable"
        )


def test_large_scale_numbers_stay_parked_and_referenced() -> None:
    """#64's parked 100k/1M envelopes are referenced as parked — never previewed here."""
    section = _section("Not yet measured")
    assert "100k" in section and "1M" in section
    assert "#64" in section, "the parked large-scale numbers must reference their issue"
    # No table row anywhere may claim a measured 100k/1M corpus.
    for line in _perf_text().splitlines():
        if line.startswith("|") and re.search(r"\b(100[,_]?000|1[,_]?000[,_]?000)\b", line):
            pytest.fail(f"PERF-RESULTS.md previews an unmeasured large-scale row: {line}")


def test_not_measured_list_covers_the_known_blind_spots() -> None:
    """The honest gap list, not a token sentence."""
    section = _section("Not yet measured")
    for blind_spot in ("Embedding", "endpoint", "multi-GPU", "Object-store", "issue #64"):
        assert blind_spot.lower() in section.lower(), f"'{blind_spot}' is not declared unmeasured"


def test_perf_results_is_linked_from_readme_and_the_docs_site() -> None:
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "PERF-RESULTS.md" in readme, "README does not link the throughput envelopes"

    page = _REPO_ROOT / "docs" / "perf-results.md"
    assert page.is_file(), "the docs site has no throughput-envelopes page"
    assert '--8<-- "PERF-RESULTS.md"' in page.read_text(encoding="utf-8"), (
        "the docs page must embed the root document, not fork a second copy of the numbers"
    )
    nav = (_REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "perf-results.md" in nav, "the throughput page is not in the mkdocs nav"


def test_documented_repro_command_measures_every_stage() -> None:
    """The published repro command runs, and reports a real duration for every Stage.

    This is the anti-drift guard: if the harness or the stack changes so the documented command
    no longer produces a per-Stage table, the published numbers become unreproducible — and this
    test fails before a buyer discovers it.
    """
    assert _BENCH.is_file(), "the documented reproduction harness is missing"
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(_BENCH), "--docs", "4", "--doc-kib", "1", "--repeat", "1"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    for name in _stack_stage_names():
        assert f"| {name} |" in out, f"the harness reported no timing for Stage {name}"
    assert "docs/hour" in out
