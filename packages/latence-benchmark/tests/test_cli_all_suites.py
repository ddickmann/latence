"""One command, every suite, one results page — the #187 acceptance criterion.

Runs the real CLI over real offline dataset roots (olmOCR-bench and RED^FM fixtures in their
real on-disk layouts, plus the shipped synthetic PII corpus) and checks that all five suites
land on the SAME published page and the SAME machine-readable sidecar.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from latence_benchmark import cli
from latence_benchmark.suite import suite_names
from redfm_fixtures import redfm_entity, redfm_row

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

PDF = "old_scans/1.pdf"
BODY = ["a passage that the parser will certainly recover from this page."]
TEXT = "Ada Lovelace worked in London in England."


@pytest.fixture
def roots(
    tmp_path: Path,
    write_dataset: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
    write_pdf: Callable[[Path, str, Sequence[str]], None],
    write_redfm: Callable[[Path, dict[str, list[dict[str, Any]]]], Path],
) -> tuple[Path, Path]:
    olmocr = write_dataset(
        tmp_path / "olmocr",
        {
            "old_scans": [
                {
                    "pdf": PDF,
                    "page": 1,
                    "id": "a",
                    "type": "present",
                    "max_diffs": 0,
                    "text": BODY[0],
                    "case_sensitive": True,
                    "checked": "verified",
                }
            ]
        },
    )
    write_pdf(olmocr, PDF, BODY)
    ada = redfm_entity(TEXT, "Ada Lovelace", "PER", "Q7259")
    london = redfm_entity(TEXT, "London", "LOC", "Q84")
    england = redfm_entity(TEXT, "England", "LOC", "Q21")
    redfm = write_redfm(
        tmp_path / "redfm",
        {"en": [redfm_row("doc-a", TEXT, [ada, london, england], [(london, "location", england)])]},
    )
    return olmocr, redfm


def test_every_suite_runs_from_one_command_onto_one_page(
    roots: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    olmocr, redfm = roots
    markdown = tmp_path / "out" / "BENCHMARKS.md"
    sidecar = tmp_path / "out" / "results.json"

    code = cli.main(
        [
            "run",
            "--suite",
            "all",
            "--dataset-root",
            f"olmocr-bench={olmocr}",
            "--dataset-root",
            f"redfm={redfm}",
            "--markdown-out",
            str(markdown),
            "--json-out",
            str(sidecar),
        ]
    )

    assert code == 0
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert [m["suite"] for m in payload["measurements"]] == list(suite_names())

    page = markdown.read_text(encoding="utf-8")
    for suite in suite_names():
        assert f"`{suite}` suite" in page
    # The unit-test suite keeps its pass-rate table; the labelled-set suites publish
    # precision/recall/F1 on the same page.
    assert "| Split | Pass rate | Passed | Failed | Scored | Unsupported |" in page
    assert "| Slice | Precision | Recall | F1 | TP | FP | FN | Gold |" in page
    printed = capsys.readouterr().out
    assert "redaction: precision" in printed
    assert "parse: 1/1 scored unit tests passed" in printed


def test_the_shipped_corpus_needs_no_dataset_root(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    """The redaction corpus travels in the package; requiring a path for it would make the
    privacy number unreproducible for anyone who just pip-installed the harness."""
    code = cli.main(
        [
            "run",
            "--suite",
            "redaction",
            "--markdown-out",
            str(tmp_path / "m.md"),
            "--json-out",
            str(tmp_path / "r.json"),
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert payload["measurements"][0]["dataset"]["name"] == "synthetic-pii"


def test_repinning_a_provider_is_refused_for_a_multi_suite_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One Provider name cannot be the right pin for four different Stages."""
    code = cli.main(["run", "--suite", "all", "--provider", "parser.plaintext"])
    assert code == 2
    assert "cannot be combined with --suite all" in capsys.readouterr().err


def test_repinning_is_recorded_in_the_fingerprint(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    """Re-pinning works for every suite, not just Parse, and the new pin lands in the result."""
    _, redfm = roots
    sidecar = tmp_path / "r.json"
    code = cli.main(
        [
            "run",
            "--suite",
            "entity_extraction",
            "--provider",
            "entity.gazetteer",
            "--dataset-root",
            str(redfm),
            "--markdown-out",
            str(tmp_path / "m.md"),
            "--json-out",
            str(sidecar),
        ]
    )
    assert code == 0
    pin = json.loads(sidecar.read_text(encoding="utf-8"))["measurements"][0]["stack_pin"]
    assert pin["provider"] == "entity.gazetteer"
    assert len(pin["fingerprint"]) == 16
