"""The Parse tracer slice end-to-end, against the REAL ``parser.pdfplumber`` Provider.

Nothing is stubbed here: the dataset root is the real on-disk layout, the PDFs are real PDF
bytes, and the Parse Stage is resolved from the real ``latence.providers`` entry-point registry
and instantiated the way the Runner instantiates it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from latence_benchmark.parse_suite import ParseSuite, ProviderUnavailableError, parse_suite
from latence_benchmark.suite import SuiteResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

PDF = "old_scans/1.pdf"
BODY = [
    "as we are deprived of Voting for whoever we Desire at the Primaries.",
    "one Law Here must apply to all citizens equally.",
    "the closing paragraph of this page.",
]


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pdf": PDF,
        "page": 1,
        "id": "row",
        "type": "present",
        "max_diffs": 0,
        "text": BODY[0],
        "case_sensitive": True,
        "first_n": None,
        "last_n": None,
        "checked": "verified",
        "url": "https://crowd.loc.gov/",
    }
    row.update(overrides)
    return row


@pytest.fixture
def dataset(
    tmp_path: Path,
    write_dataset: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
    write_pdf: Callable[[Path, str, Sequence[str]], None],
) -> Path:
    root = write_dataset(
        tmp_path,
        {
            "old_scans": [
                _row(id="present-hit"),
                _row(id="present-miss", text="a passage this page never contained"),
                _row(id="absent-hit", type="absent", text="Page 3 of 9"),
                _row(id="order-hit", type="order", text=None, before=BODY[0], after=BODY[2]),
                _row(id="order-miss", type="order", text=None, before=BODY[2], after=BODY[0]),
                _row(id="math-unsupported", type="math", text=None, math="x^2"),
            ]
        },
    )
    write_pdf(root, PDF, BODY)
    return root


def _suite(dataset_root: Path) -> ParseSuite:
    return ParseSuite(splits=("old_scans",))


def test_the_real_parse_stage_is_scored_end_to_end(dataset: Path) -> None:
    result = _suite(dataset).run(dataset)

    assert isinstance(result, SuiteResult)
    assert result.stage == "parse"
    assert result.documents_scored == 1
    assert result.document_failures == 0

    (split,) = result.splits
    assert (split.passed, split.failed, split.unsupported) == (3, 2, 1)
    assert split.scored == 5
    assert split.pass_rate == pytest.approx(0.6)


def test_unsupported_tests_stay_out_of_the_pass_rate_denominator(dataset: Path) -> None:
    """Six tests exist; five are scorable. A pass rate of 3/6 would understate the parser."""
    overall = _suite(dataset).run(dataset).overall
    assert overall.total == 6
    assert overall.scored == 5


def test_a_missing_pdf_fails_its_tests_rather_than_disappearing(
    tmp_path: Path,
    write_dataset: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    root = write_dataset(tmp_path, {"old_scans": [_row(id="a"), _row(id="b")]})
    result = ParseSuite(splits=("old_scans",)).run(root)
    (split,) = result.splits
    assert (split.passed, split.failed) == (0, 2)
    assert any("referenced PDF" in note for note in result.notes)


def test_a_document_that_cannot_be_parsed_fails_its_tests(
    tmp_path: Path,
    write_dataset: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> None:
    """A parser must not be able to raise its way to a better score."""
    root = write_dataset(tmp_path, {"old_scans": [_row(id="a"), _row(id="b")]})
    corrupt = root / "bench_data" / "pdfs" / PDF
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"%PDF-1.4\nthis is not a pdf at all")

    result = ParseSuite(splits=("old_scans",)).run(root)
    (split,) = result.splits
    assert result.document_failures == 1
    assert (split.passed, split.failed) == (0, 2)


def test_two_runs_of_the_same_pin_are_identical(dataset: Path) -> None:
    first = _suite(dataset).run(dataset)
    second = _suite(dataset).run(dataset)
    assert first.as_dict() == second.as_dict()


def test_a_capped_run_is_labelled_as_a_subset(dataset: Path) -> None:
    result = _suite(dataset).run(dataset, limit=1)
    assert any(note.startswith("SUBSET RUN") for note in result.notes)
    assert any("NOT comparable" in note for note in result.notes)


def test_the_stack_pin_is_recorded_and_fingerprinted(dataset: Path) -> None:
    result = _suite(dataset).run(dataset)
    assert result.stack_pin.provider == "parser.pdfplumber"
    assert len(result.stack_pin.fingerprint) == 16
    assert result.as_dict()["stack_pin"]["fingerprint"] == result.stack_pin.fingerprint


def test_repinning_the_provider_changes_the_fingerprint() -> None:
    assert parse_suite(provider="parser.document").stack_pin.fingerprint != (
        ParseSuite().stack_pin.fingerprint
    )


def test_an_uninstalled_provider_is_named_never_silently_substituted(dataset: Path) -> None:
    suite = ParseSuite(stack_pin=parse_suite(provider="parser.nonexistent").stack_pin)
    with pytest.raises(ProviderUnavailableError, match="parser.nonexistent"):
        suite.run(dataset)


def test_the_dataset_licence_travels_with_the_result(dataset: Path) -> None:
    payload = _suite(dataset).run(dataset).as_dict()
    licence = payload["dataset"]["license"]
    assert licence["dataset_license"] == "ODC-BY-1.0"
    assert licence["commercial_use"] is True
    assert payload["dataset"]["revision"] == "54a96a6fb6a2bd3b297e59869491db4d3625b711"
