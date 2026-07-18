"""The one command, and the published page's separation of measurements from claims."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
from latence_benchmark import cli
from latence_benchmark.datasets import OMNIDOCBENCH
from latence_benchmark.parse_suite import ParseSuite
from latence_benchmark.results import VENDOR_CLAIMS, render_markdown

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
