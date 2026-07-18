"""The unit-test reader must REJECT what the real dataset would never contain.

A lenient reader is the false-green machine here: dropping an unparseable row shrinks the
numerator and the denominator together, so the pass rate barely moves and the harness reports
a benchmark it did not actually run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from latence_benchmark.unittests import (
    BenchmarkDataError,
    UnitTestType,
    load_split,
    parse_unit_tests,
)

# Rows copied from the real bench_data JSONL at the pinned revision (values shortened).
REAL_PRESENT: dict[str, Any] = {
    "pdf": "long_tiny_text/14a_pg1.pdf",
    "page": 1,
    "id": "14a_pg1_text_01",
    "type": "present",
    "max_diffs": 1,
    "checked": "verified",
    "text": "afterlife, a new or continued or transformed existence after death",
    "case_sensitive": True,
    "first_n": None,
    "last_n": None,
    "url": "https://archive.org/details/newencyclopaedia01chic",
}
REAL_ORDER: dict[str, Any] = {
    "pdf": "multi_column/0a2b_page_8_pg1.pdf",
    "page": 1,
    "id": "0a2b_page_8.pdf_order_9257021d",
    "type": "order",
    "before": "48: 105-9.",
    "after": "TGF-beta signaling: positive and negative effects on tumorigenesis.",
    "max_diffs": 1,
    "checked": "verified",
    "url": "http://koreanjpathol.org/upload/journal/4404346.pdf",
}
REAL_TABLE: dict[str, Any] = {
    "pdf": "tables/b5c5_pg4.pdf",
    "page": 1,
    "id": "b5c5_pg4_table_02",
    "type": "table",
    "max_diffs": 0,
    "checked": "verified",
    "cell": "0.569",
    "up": None,
    "down": None,
    "left": None,
    "right": None,
    "top_heading": None,
    "left_heading": "BO",
    "url": "http://www.aensiweb.com/old/jasa/rjfh/2014/995-1001.pdf",
}
REAL_MATH: dict[str, Any] = {
    "pdf": "arxiv_math/2503.04048_pg46.pdf",
    "url": "https://arxiv.org/pdf/2503.04048",
    "page": 1,
    "id": "2503.04048_pg46_math_000",
    "type": "math",
    "max_diffs": 0,
    "checked": None,
    "math": "{\\mathcal{V}}(\\psi_m)\\rightarrow +\\infty",
}


def _lines(*rows: dict[str, Any]) -> list[str]:
    return [json.dumps(row) for row in rows]


def test_real_rows_of_every_class_parse() -> None:
    tests = parse_unit_tests(
        _lines(REAL_PRESENT, REAL_ORDER, REAL_TABLE, REAL_MATH), origin="fixture"
    )
    assert [t.type for t in tests] == [
        UnitTestType.PRESENT,
        UnitTestType.ORDER,
        UnitTestType.TABLE,
        UnitTestType.MATH,
    ]
    assert tests[0].max_diffs == 1
    assert tests[1].before == "48: 105-9."


def test_unknown_metadata_columns_do_not_break_a_pinned_run() -> None:
    row = {**REAL_PRESENT, "some_future_column": {"nested": 1}}
    assert parse_unit_tests(_lines(row), origin="fixture")[0].id == REAL_PRESENT["id"]


def test_blank_lines_are_skipped() -> None:
    assert len(parse_unit_tests([json.dumps(REAL_PRESENT), "", "  "], origin="fixture")) == 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"type": "handwriting"}, "not a valid olmOCR-bench unit test"),
        ({"text": None}, "missing required non-empty"),
        ({"text": ""}, "missing required non-empty"),
        ({"max_diffs": -1}, "not a valid olmOCR-bench unit test"),
        ({"id": ""}, "not a valid olmOCR-bench unit test"),
        ({"pdf": ""}, "not a valid olmOCR-bench unit test"),
        ({"first_n": 0}, "not a valid olmOCR-bench unit test"),
        ({"page": 0}, "not a valid olmOCR-bench unit test"),
    ],
)
def test_malformed_rows_are_refused_not_dropped(mutation: dict[str, Any], match: str) -> None:
    row = {**REAL_PRESENT, **mutation}
    with pytest.raises(BenchmarkDataError, match=match):
        parse_unit_tests(_lines(row), origin="fixture")


def test_an_order_row_needs_both_sides() -> None:
    with pytest.raises(BenchmarkDataError, match="missing required non-empty field\\(s\\)"):
        parse_unit_tests(_lines({**REAL_ORDER, "after": None}), origin="fixture")


def test_a_table_row_needs_a_cell() -> None:
    with pytest.raises(BenchmarkDataError, match="missing required non-empty"):
        parse_unit_tests(_lines({**REAL_TABLE, "cell": None}), origin="fixture")


def test_a_math_row_needs_an_expression() -> None:
    with pytest.raises(BenchmarkDataError, match="missing required non-empty"):
        parse_unit_tests(_lines({**REAL_MATH, "math": None}), origin="fixture")


def test_the_failing_line_number_is_reported() -> None:
    lines = [json.dumps(REAL_PRESENT), "{not json"]
    with pytest.raises(BenchmarkDataError, match="fixture:2:"):
        parse_unit_tests(lines, origin="fixture")


def test_a_json_array_row_is_refused() -> None:
    with pytest.raises(BenchmarkDataError, match="expected a JSON object"):
        parse_unit_tests(["[1, 2, 3]"], origin="fixture")


def test_a_missing_split_is_named(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkDataError, match="old_scans"):
        load_split(tmp_path, "old_scans")
