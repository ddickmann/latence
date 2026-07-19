"""The one command, and the published page's separation of measurements from claims."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from latence_benchmark import cli
from latence_benchmark.datasets import OMNIDOCBENCH
from latence_benchmark.metrics import MetricScore
from latence_benchmark.parse_suite import ParseSuite
from latence_benchmark.results import VENDOR_CLAIMS, render_markdown
from latence_benchmark.suite import StackPin, SuiteResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

PDF = "old_scans/1.pdf"
BODY = ["a passage that the parser will certainly recover from this page."]


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pdf": PDF,
        "page": 1,
        "id": "row",
        "type": "present",
        "max_diffs": 0,
        "text": BODY[0],
        "case_sensitive": True,
        "checked": "verified",
    }
    row.update(overrides)
    return row


@pytest.fixture
def dataset(
    tmp_path: Path,
    write_dataset: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
    write_pdf: Callable[[Path, str, Sequence[str]], None],
) -> Path:
    root = write_dataset(tmp_path / "data", {"old_scans": [_row(id="a"), _row(id="b")]})
    write_pdf(root, PDF, BODY)
    return root


@pytest.fixture(autouse=True)
def _single_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the registered suite at the one split the offline fixture ships."""
    monkeypatch.setattr(cli, "get_suite", lambda _name: ParseSuite(splits=("old_scans",)))


def test_one_command_runs_the_benchmark_and_publishes_both_artifacts(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    markdown = tmp_path / "out" / "BENCHMARKS.md"
    sidecar = tmp_path / "out" / "results.json"

    code = cli.main(
        [
            "run",
            "--suite",
            "parse",
            "--dataset-root",
            str(dataset),
            "--markdown-out",
            str(markdown),
            "--json-out",
            str(sidecar),
        ]
    )

    assert code == 0
    assert "2/2 scored unit tests passed (100.0%)" in capsys.readouterr().out

    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["measurements"][0]["overall"]["passed"] == 2
    assert payload["measurements"][0]["dataset"]["revision"].startswith("54a96a6")
    # Claims live in their own top-level key — never mixed into the measurements.
    assert {c["kind"] for c in payload["vendor_claims"]} == {"vendor-claim"}

    page = markdown.read_text(encoding="utf-8")
    assert "### Our measurements" in page
    assert "### Vendor claims — NOT measured by us" in page


def test_a_non_commercial_dataset_is_refused_before_anything_is_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gated = replace(ParseSuite(splits=("old_scans",)), dataset=OMNIDOCBENCH)
    monkeypatch.setattr(cli, "get_suite", lambda _name: gated)

    def _explode(*_args: object, **_kwargs: object) -> object:  # pragma: no cover - must not run
        raise AssertionError("acquisition ran despite a non-commercial license")

    monkeypatch.setattr(cli, "_source", _explode)

    code = cli.main(["run", "--suite", "parse", "--markdown-out", str(tmp_path / "m.md")])
    assert code == 2
    assert "does NOT permit commercial use" in capsys.readouterr().err


def test_an_unusable_dataset_root_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["run", "--dataset-root", str(tmp_path / "absent")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_list_names_the_registered_suites(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list"]) == 0
    assert "parse" in capsys.readouterr().out


def test_the_results_page_never_presents_a_claim_as_a_measurement(dataset: Path) -> None:
    """A vendor number and one of ours must not be able to share a table.

    The types enforce it — VendorClaim has no computed pass rate, SuiteResult has no citation —
    and the page keeps them in separate sections with the caveat attached to the claims.
    """
    result = ParseSuite(splits=("old_scans",)).run(dataset)
    page = render_markdown([result], generated_on="2026-07-18")

    measurements = page.index("### Our measurements")
    claims = page.index("### Vendor claims — NOT measured by us")
    assert measurements < claims
    assert "We did" in page and "not run these systems" in page
    for claim in VENDOR_CLAIMS:
        assert claim.source_url in page
    assert "directly comparable to" in page


def test_the_results_page_records_the_verified_licence_and_the_reproduce_command(
    dataset: Path,
) -> None:
    page = render_markdown([ParseSuite(splits=("old_scans",)).run(dataset)], generated_on="d")
    assert "| Corpus license | **ODC-BY-1.0** |" in page
    assert "| Evaluation-toolkit license | **Apache-2.0** |" in page
    assert "latence-benchmark run --suite parse --provider parser.pdfplumber" in page
    assert "54a96a6fb6a2bd3b297e59869491db4d3625b711" in page


def _criteria_result() -> SuiteResult:
    """A result shaped like the redaction suite's: a language PARTITION plus match CRITERIA.

    Both dimensions cover the same 35 gold items — the language slices by carving them up, the
    match slices by re-scoring all of them under three different rules.
    """
    return SuiteResult(
        suite="parse",
        stage="redaction",
        dataset=OMNIDOCBENCH,
        stack_pin=StackPin(stage="redaction", provider="redaction.hybrid_rule"),
        splits=(),
        documents_scored=20,
        metrics=(
            MetricScore("language", "de", true_positives=14, false_positives=0, false_negatives=4),
            MetricScore("language", "en", true_positives=13, false_positives=0, false_negatives=4),
            MetricScore("match", "coverage", true_positives=27, false_positives=0, false_negatives=8),
            MetricScore("match", "exact span", true_positives=25, false_positives=2, false_negatives=10),
            MetricScore("match", "span+type", true_positives=22, false_positives=5, false_negatives=13),
        ),
        headline_dimension="language",
        measured_on="2026-07-18",
    )


def test_the_page_publishes_no_aggregate_over_a_criteria_dimension() -> None:
    """A `match` table must carry no bold MICRO row.

    Its slices are alternative scoring criteria over ONE gold set, so summing their confusion
    counts multiply-counts every gold item: a 35-span corpus rendered with a Gold column of 105,
    published as the only bold row in the section and therefore read as its authoritative
    summary. It corresponds to no measurable quantity, so it must not be rendered at all.
    """
    page = render_markdown([_criteria_result()], generated_on="2026-07-18")
    section = page[page.index("#### By match criterion") :]
    section = section[: section.index("Documents scored:")]

    assert "**MICRO**" not in section
    assert "| 105 |" not in section  # the multiply-counted gold denominator
    assert "alternative scoring criteria over the same gold items" in section
    # The measurements themselves are untouched — only the fabricated aggregate is gone.
    assert "| `coverage` | 100.0% | 77.1% | 87.1% | 27 | 0 | 8 | 35 |" in section


def test_the_page_still_totals_a_partition_dimension() -> None:
    """Removing the fabricated aggregate must not remove the real ones."""
    page = render_markdown([_criteria_result()], generated_on="2026-07-18")
    section = page[page.index("#### By language") : page.index("#### By match criterion")]
    assert "| **MICRO** |" in section
    assert "| 27 | 0 | 8 | 35 |" in section  # the language slices DO partition the 35 gold spans
    assert "**Headline (micro over language):**" in page
