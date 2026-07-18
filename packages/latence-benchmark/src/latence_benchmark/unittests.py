"""The olmOCR-bench unit-test schema — parsed strictly, so a malformed row is never scored.

olmOCR-bench does not score a parser by comparing its markdown against a gold transcription;
it ships **unit tests over the output**, one JSONL row per test, each asserting a property a
good parse must have (a passage is present, a header/footer is absent, two passages appear in
this reading order, a table cell sits below this heading, this LaTeX expression survived).

Every row is validated here rather than duck-typed downstream, and the validation is
deliberately *as strict as the real data*: a ``present`` row without ``text``, an ``order`` row
without both ``before`` and ``after``, a negative ``max_diffs`` or an unknown ``type`` is a hard
:class:`BenchmarkDataError` naming the file and line. That is the anti-false-green rule applied
to the harness's own input — a lenient reader would silently drop rows and quietly inflate the
pass rate, because the denominator would shrink along with the numerator.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from pathlib import Path


class BenchmarkDataError(ValueError):
    """A benchmark data file could not be read as the schema it claims to be."""


class UnitTestType(StrEnum):
    """The five olmOCR-bench test classes."""

    PRESENT = "present"
    ABSENT = "absent"
    ORDER = "order"
    TABLE = "table"
    MATH = "math"


class UnitTest(BaseModel):
    """One olmOCR-bench unit test.

    Unknown keys are ignored (upstream adds metadata columns over time and that must not break
    a pinned run), but every key this harness *acts on* is validated, and the per-type required
    fields are enforced in :meth:`_require_fields_for_type`.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(..., min_length=1)
    pdf: str = Field(..., min_length=1, description="Repo-relative PDF path under bench_data/pdfs.")
    page: int = Field(default=1, ge=1)
    type: UnitTestType
    #: Levenshtein budget: the assertion holds if the text matches within this many edits.
    max_diffs: int = Field(default=0, ge=0)

    # present / absent
    text: str | None = None
    case_sensitive: bool = True
    first_n: int | None = Field(default=None, ge=1)
    last_n: int | None = Field(default=None, ge=1)

    # order
    before: str | None = None
    after: str | None = None

    # table
    cell: str | None = None
    up: str | None = None
    down: str | None = None
    left: str | None = None
    right: str | None = None
    top_heading: str | None = None
    left_heading: str | None = None

    # math
    math: str | None = None

    @model_validator(mode="after")
    def _require_fields_for_type(self) -> UnitTest:
        required: tuple[str, ...]
        if self.type in (UnitTestType.PRESENT, UnitTestType.ABSENT):
            required = ("text",)
        elif self.type is UnitTestType.ORDER:
            required = ("before", "after")
        elif self.type is UnitTestType.TABLE:
            required = ("cell",)
        else:
            required = ("math",)
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            msg = (
                f"{self.type.value!r} unit test {self.id!r} is missing required non-empty "
                f"field(s) {missing}; it cannot be scored and is not silently dropped."
            )
            raise ValueError(msg)
        return self


def parse_unit_tests(lines: list[str], *, origin: str) -> list[UnitTest]:
    """Parse JSONL ``lines`` into :class:`UnitTest`s, failing loudly on the first bad row."""
    tests: list[UnitTest] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"{origin}:{number}: not valid JSON ({exc.msg})."
            raise BenchmarkDataError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"{origin}:{number}: expected a JSON object, got {type(payload).__name__}."
            raise BenchmarkDataError(msg)
        try:
            tests.append(UnitTest.model_validate(payload))
        except ValueError as exc:
            msg = f"{origin}:{number}: not a valid olmOCR-bench unit test — {exc}"
            raise BenchmarkDataError(msg) from exc
    return tests


def load_split(root: Path, split: str) -> list[UnitTest]:
    """Load one ``bench_data/<split>.jsonl`` split from a materialised dataset ``root``."""
    path = root / "bench_data" / f"{split}.jsonl"
    if not path.is_file():
        msg = f"Benchmark split {split!r} not found at {path}."
        raise BenchmarkDataError(msg)
    return parse_unit_tests(path.read_text(encoding="utf-8").splitlines(), origin=str(path))
