"""The public-release gate — this repository cannot be published with unfilled results.

The failure this prevents is specific: shipping a README whose headline claims a sophisticated
evaluator can falsify in an afternoon. Two of the four capabilities most likely to be quoted from
this repo's docs were, when these gates were written, *structurally unable to work* — the SPLADE
sparse index was never built despite a complete query path, and the graph could only reorder a
candidate pool it had not chosen. Both are implemented now. Neither is yet proven on real data.

So the numbers live in exactly one place (``docs/BENCHMARKS-E2E.md``), every one of them is a
``PENDING:`` token until measured, and this module makes the tokens **load-bearing**:

* **always** — the gate documents must exist, be linked to each other, and every placeholder must be
  well-formed. An inventory is printed so the outstanding set is visible on every run.
* **under ``LATENCE_RELEASE_CHECK=1``** — every placeholder must be GONE. This is the check the
  release runbook calls, and it is why publishing with an unfilled table by forgetting is not
  possible.

The two modes matter. A gate that fails during ordinary development gets disabled within a week; a
gate that only ever passes proves nothing. This one is quiet while the work is in progress and
absolute at the moment it counts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_GATES = _REPO_ROOT / "RELEASE-GATES.md"
_RESULTS = _REPO_ROOT / "docs" / "BENCHMARKS-E2E.md"

#: A placeholder awaiting a measured value. Deliberately a distinctive token: `TODO` and `TBD` occur
#: naturally in prose and would make this gate fire on unrelated text.
_PENDING = re.compile(r"`PENDING:([a-z0-9\-]+)`", re.I)

#: A malformed placeholder — the token without its backticks, or with unexpected characters. These
#: would slip past `_PENDING` and therefore past the release check, which is the one way this gate
#: could silently fail to do its job.
_MALFORMED = re.compile(r"(?<!`)PENDING:(?!\S*`)|`PENDING:[^`]*[^a-z0-9\-`][^`]*`", re.I)

_RELEASE_MODE = os.environ.get("LATENCE_RELEASE_CHECK", "").strip().lower() in {"1", "true", "yes"}


def _text(path: Path) -> str:
    assert path.exists(), (
        f"{path.relative_to(_REPO_ROOT)} is missing. It is a release gate: the numbers that block "
        "publication live there, and deleting it would silently remove the block."
    )
    return path.read_text(encoding="utf-8")


def _placeholders(text: str) -> list[str]:
    return _PENDING.findall(text)


# --- always-on structural checks -------------------------------------------------------------


def test_the_gate_documents_exist() -> None:
    """Both halves must be present: the checklist, and the results page it gates."""
    assert _GATES.exists(), "RELEASE-GATES.md is missing"
    assert _RESULTS.exists(), "docs/BENCHMARKS-E2E.md is missing"


def test_the_gates_point_at_the_results_page_and_back() -> None:
    """A checklist that does not reference its evidence is decoration."""
    gates = _text(_GATES)
    results = _text(_RESULTS)

    assert "BENCHMARKS-E2E.md" in gates, (
        "RELEASE-GATES.md does not reference docs/BENCHMARKS-E2E.md, so nothing ties the checklist "
        "to the numbers it is gating."
    )
    assert "RELEASE-GATES.md" in results, (
        "docs/BENCHMARKS-E2E.md does not reference RELEASE-GATES.md, so a reader who lands on the "
        "results page has no way to learn the numbers are placeholders."
    )


def test_the_results_page_is_not_confused_with_the_measured_suite_page() -> None:
    """``docs/BENCHMARKS.md`` carries REAL measured per-Stage numbers and must stay separate.

    Mixing measured and pending values on one page is how a reader ends up quoting a placeholder as
    a result. The pending page must say so explicitly.
    """
    results = _text(_RESULTS)
    assert "BENCHMARKS.md" in results, (
        "docs/BENCHMARKS-E2E.md must distinguish itself from docs/BENCHMARKS.md, which carries "
        "already-measured per-Stage suite results."
    )


def test_every_placeholder_is_well_formed() -> None:
    """A malformed token would be invisible to the release check — the one silent failure mode."""
    results = _text(_RESULTS)
    malformed = _MALFORMED.findall(results)
    assert not malformed, (
        f"malformed PENDING token(s) in docs/BENCHMARKS-E2E.md: {malformed}. "
        "Use exactly `PENDING:lowercase-hyphenated`. A malformed token is not matched by the "
        "release check, so it would pass silently while the value is still missing."
    )


def test_placeholders_are_greppable_and_the_page_gates_something_substantial() -> None:
    """Each placeholder names the value it awaits, so filling them in is mechanical.

    An earlier version of this test asserted a minimum name length, which was simply wrong: ``g1``
    through ``g8`` are the standard identifiers for the necessity gates in the multi-hop literature
    (closed-book, single-hop baseline, per-edge necessity, DiRe, …). Renaming them to satisfy an
    arbitrary rule would have made the document worse to serve the test. The property actually
    worth asserting is that each token is greppable and that the page gates a substantial set —
    a results page with three placeholders is not gating a release.
    """
    names = _placeholders(_text(_RESULTS))
    assert names, "no PENDING placeholders found — if the results are filled, delete this gate"

    bad = [n for n in names if not re.fullmatch(r"[a-z0-9][a-z0-9\-]*", n)]
    assert not bad, f"placeholder names must be lowercase-hyphenated and greppable; got {bad}"

    assert len(names) >= 20, (
        f"only {len(names)} placeholders on the results page. The end-to-end run produces "
        "throughput, per-signal retrieval, hop-stratified results, the question funnel, the KG "
        "ablation and link-prediction numbers — a page gating fewer than 20 values has probably "
        "lost a section."
    )


def test_the_results_page_declares_its_own_status() -> None:
    """A reader must learn on the first screen that nothing here is measured yet."""
    head = _text(_RESULTS)[:1200]
    assert "UNFILLED" in head.upper(), (
        "docs/BENCHMARKS-E2E.md must declare UNFILLED status near the top. A results page that "
        "does not say its numbers are placeholders is the exact defect this gate exists to prevent."
    )


def test_the_gates_cover_the_capabilities_most_likely_to_be_overclaimed() -> None:
    """The four an evaluator probes first. Each was, or is, partial — none may be assumed."""
    gates = _text(_GATES).lower()
    for capability in ("splade", "graph leg", "link prediction", "verify-local"):
        assert capability in gates, (
            f"RELEASE-GATES.md does not gate {capability!r}. It is one of the capabilities an "
            "evaluator probes first, and it was partial when these gates were written."
        )


def test_the_outstanding_set_is_inventoried(capsys: pytest.CaptureFixture[str]) -> None:
    """Print what is still missing, so every test run surfaces the remaining work."""
    names = _placeholders(_text(_RESULTS))
    unchecked = _text(_GATES).count("- [ ]")
    with capsys.disabled():
        print(
            f"\n  RELEASE GATE: {len(names)} unfilled result(s), "
            f"{unchecked} unchecked gate item(s). "
            f"{'RELEASE MODE — these are hard failures.' if _RELEASE_MODE else ''}"
        )
    assert isinstance(names, list)


# --- the release check ------------------------------------------------------------------------


@pytest.mark.skipif(not _RELEASE_MODE, reason="set LATENCE_RELEASE_CHECK=1 to run the release gate")
def test_release_no_result_is_still_a_placeholder() -> None:
    """RELEASE MODE: every number must be measured before this repository is published."""
    names = _placeholders(_text(_RESULTS))
    assert not names, (
        f"{len(names)} result(s) in docs/BENCHMARKS-E2E.md are still placeholders:\n  "
        + "\n  ".join(sorted(set(names)))
        + "\n\nThese are filled from a real GPU run (issues/e2e-extended/RUNBOOK.md). Do NOT fill "
        "one with a number from a different corpus, commit, or card — each table states its own "
        "method, and a number that does not match its stated method is worse than a missing one, "
        "because it looks checked."
    )


@pytest.mark.skipif(not _RELEASE_MODE, reason="set LATENCE_RELEASE_CHECK=1 to run the release gate")
def test_release_every_gate_item_is_checked() -> None:
    """RELEASE MODE: the checklist must be complete, including the items no test can verify."""
    gates = _text(_GATES)
    unchecked = [line.strip() for line in gates.splitlines() if line.strip().startswith("- [ ]")]
    assert not unchecked, (
        f"{len(unchecked)} release gate item(s) are unchecked:\n  "
        + "\n  ".join(unchecked)
        + "\n\n"
        "Note that legal review and the environment-gated test lanes are NOT machine-verifiable — "
        "they are checked by a human who did the work, not by this assertion."
    )


@pytest.mark.skipif(not _RELEASE_MODE, reason="set LATENCE_RELEASE_CHECK=1 to run the release gate")
def test_release_the_licence_is_declared_consistently_across_every_package() -> None:
    """RELEASE MODE: one package still on the old licence undermines the whole relicense."""
    declared: dict[str, set[str]] = {}
    for pyproject in sorted((_REPO_ROOT / "packages").glob("*/pyproject.toml")):
        text = pyproject.read_text(encoding="utf-8")
        found = set(re.findall(r'license\s*=\s*[{"\']?[^\n]*?([A-Za-z0-9.\-]+-\d\.\d)', text))
        found |= set(re.findall(r'license\s*=\s*["\']([^"\']+)["\']', text))
        if found:
            declared[pyproject.parent.name] = found

    assert declared, "no licence metadata found in any package — the gate cannot verify coherence"
    identifiers = {ident for idents in declared.values() for ident in idents}
    assert len(identifiers) == 1, (
        f"packages declare {len(identifiers)} different licences: {sorted(identifiers)}.\n"
        + "\n".join(f"  {pkg}: {sorted(v)}" for pkg, v in sorted(declared.items()))
        + "\nEvery package must agree; a stray declaration is what a licence scanner will surface."
    )
