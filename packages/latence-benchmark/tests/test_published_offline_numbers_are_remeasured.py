"""A published number the harness can re-run must equal what re-running it measures.

``test_published_artifacts_are_fresh`` proves the committed page is what today's renderer emits
*for the numbers in the committed sidecar* — freshness, explicitly not quality. It rebuilds those
numbers from the sidecar and compares them to a re-render of themselves, so it is true of any
sidecar, including one whose numbers are fiction. ``test_benchmarks_page_truthfulness`` asserts a
redaction headline exists and is non-None; ``test_every_registered_suite_is_accounted_for``
compares suite-name sets. No gate has ever compared a published number to a live run.

For most suites there is a reason: RED^FM needs a download, Parse needs olmOCR-bench, and a merge
gate cannot depend on the network. The one anti-restamping mechanism that works offline,
:class:`~latence_benchmark.scoring.ScorerPin`, is scoped to the unit-test scorer and is ``None``
for every labelled-set suite — the labelled-set scoring rules (``metrics.confusion``/``micro``/
``total``, ``redaction_suite.coverage``) carry no pin at all. So
:func:`~latence_benchmark.results.carry_forward` republishes a labelled-set measurement
indefinitely, validating its dataset revision and nothing else about the number.

The redaction suite is the case where that is indefensible. Its corpus ships in-package, its
Provider is CPU-only and deterministic, and the whole run takes about a third of a second — and the
number it publishes, ``precision 100.0% · recall 77.1%``, is the framework's central privacy
evidence (ADR-0044). The published run note even names the regression this is supposed to catch:
"Narrowing that default would show up here as a recall drop." Nothing made it show up. Narrow the
default recogniser set and ``test_redaction_suite`` still passes (it asserts only
``recall > 0.7``), the freshness gate re-renders the stale sidecar byte-identically, and the page
keeps publishing 77.1%.

So this module re-runs it. For every published measurement whose suite
:func:`~latence_benchmark.results.reproduces_offline` — the registry's own criterion for "ships
in-package, nothing to download" — the suite is run live against the shipped corpus and the whole
published record is compared to what the live run measured: every metric slice, the headline, the
counts, the stack pin, the notes. Not a bound; the record. Any change to the scoring rules, the
Provider defaults or the corpus that moves the number turns this red, which is what a labelled-set
scorer pin would have been an approximation of.

The excluded field is ``measured_on``, which is a fact about when the run happened and is
deliberately allowed to be older than today (:func:`carry_forward` exists so an assembled page can
carry each measurement's own date). Chained with the freshness gate — sidecar equals live run,
page equals a render of the sidecar — the published page is true, not merely self-consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import latence_benchmark  # noqa: F401  -- import registers every built-in suite
import pytest
from latence_benchmark.datasets import ShippedCorpusSource
from latence_benchmark.results import carry_forward, reproduces_offline
from latence_benchmark.suite import SuiteResult, get_suite

# tests/  ->  latence-benchmark/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIDECAR = _REPO_ROOT / "docs" / "benchmarks" / "results.json"

_REGENERATE = "uv run latence-benchmark run --suite all"

#: When the measurement was produced. A carried measurement is older than today's date on purpose.
_NOT_A_CLAIM_ABOUT_THE_NUMBER = "measured_on"


def _published() -> tuple[SuiteResult, ...]:
    """Every measurement on the committed sidecar, rebuilt and validated by ``carry_forward``."""
    assert _SIDECAR.is_file(), (
        f"{_SIDECAR.relative_to(_REPO_ROOT)} is missing. It is a committed, generated artifact — "
        f"regenerate it with `{_REGENERATE}`."
    )
    return carry_forward([], json_path=_SIDECAR)


def _offline_published() -> tuple[str, ...]:
    """Published suites whose numbers this process can reproduce, with no network."""
    return tuple(sorted(r.suite for r in _published() if reproduces_offline(r.suite)))


def _record(result: SuiteResult) -> dict[str, Any]:
    """The measurement as a claim: everything ``as_dict`` states except *when* it was measured."""
    return {k: v for k, v in result.as_dict().items() if k != _NOT_A_CLAIM_ABOUT_THE_NUMBER}


def test_the_gate_is_not_vacuous() -> None:
    """If nothing published reproduces offline, this module proves nothing and must say so."""
    published = _published()
    assert published, (
        "docs/benchmarks/results.json carries no measurements, which would make every assertion "
        "below trivially true"
    )
    assert _offline_published(), (
        "no published suite reproduces offline any more, so nothing below re-runs anything. The "
        "redaction suite ships its corpus in-package and used to be checked here; if it was "
        "re-pinned to a downloaded dataset, this gate needs replacing, not deleting — the "
        "framework's central privacy number would otherwise be republished on trust forever."
    )


@pytest.mark.parametrize("suite", _offline_published())
def test_a_published_number_the_harness_can_re_run_equals_what_re_running_it_measures(
    suite: str,
) -> None:
    """Re-run the suite and demand the published record, not a bound around it.

    This is the only gate in the repository that compares a published benchmark number against a
    live measurement. It needs no network: the corpus ships in this package and the pinned
    Provider is CPU-only.
    """
    published = next(r for r in _published() if r.suite == suite)
    registered = get_suite(suite)
    root = ShippedCorpusSource().materialise(registered.dataset)

    live = registered.run(root)

    assert _record(live) == _record(published), (
        f"the published measurement for suite {suite!r} is NOT what the harness measures today. "
        "Its corpus ships in this package, so this ran the real Stage over the real committed "
        "bytes in a fraction of a second — the published number is stale or false, and "
        "docs/BENCHMARKS.md is currently asserting it to the outside world.\n"
        f"  published: {json.dumps(_record(published)['headline'], sort_keys=True)}\n"
        f"  live run:  {json.dumps(_record(live)['headline'], sort_keys=True)}\n"
        f"If the change to the Stage is intended, re-publish: `{_REGENERATE}`. Editing the "
        "sidecar to match is not a fix — the sidecar is generated, not authored."
    )


def test_a_falsified_published_number_is_caught_rather_than_re_rendered(tmp_path: Path) -> None:
    """The mutation the freshness gate cannot see: fiction that re-renders byte-identically.

    A sidecar whose redaction slices claim every gold value was caught satisfies every assertion
    the three existing gate modules make — freshness compares the page to a re-render of these
    same numbers, and the truthfulness page-gate only asks that a headline exists. It must not
    survive a live re-run.
    """
    suites = _offline_published()
    assert suites, "no offline-reproducible suite is published, so this proves nothing"
    suite = suites[0]

    payload = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    measurement = next(m for m in payload["measurements"] if m["suite"] == suite)
    headline_dimension = measurement["headline_dimension"]
    falsified = [m for m in measurement["metrics"] if m["dimension"] == headline_dimension]
    assert falsified, f"suite {suite!r} publishes no {headline_dimension!r} slice to falsify"
    for slice_ in falsified:
        # A perfect score: every miss re-labelled as a catch.
        slice_["true_positives"] += slice_["false_negatives"]
        slice_["false_negatives"] = 0
    sidecar = tmp_path / "results.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    # carry_forward accepts it — the dataset revision and the scorer pin are both untouched.
    tampered = next(r for r in carry_forward([], json_path=sidecar) if r.suite == suite)
    headline = tampered.headline
    assert headline is not None and headline.recall == 1.0, (
        "the falsified sidecar did not actually produce a flattering number, so the assertion "
        "below would pass for the wrong reason"
    )

    registered = get_suite(suite)
    live = registered.run(ShippedCorpusSource().materialise(registered.dataset))
    assert _record(live) != _record(tampered), (
        f"a sidecar claiming perfect recall for suite {suite!r} is indistinguishable from the "
        "live run, so re-measuring cannot catch a falsified published number"
    )
