"""Scoring one parser output against one olmOCR-bench unit test.

The metric olmOCR-bench defines is a **pass rate over unit tests**, not an edit distance
against a gold transcription, and the tests are fuzzy by design: each row carries a
``max_diffs`` Levenshtein budget, because a parse that renders ``fi`` as a ligature or drops a
soft hyphen is not wrong in any way a reader would care about.

So the primitive here is approximate substring search — "does ``needle`` occur anywhere in
``haystack`` within ``max_diffs`` edits, and if so where does the leftmost such occurrence
end". :func:`fuzzy_find` is Sellers' algorithm (Levenshtein DP with a free start position),
short-circuited by an exact ``str.find`` first because the overwhelming majority of tests match
exactly and the DP is O(len(haystack) × len(needle)).

Four of the six test classes are scored here — ``present``, ``absent``, ``order`` and
``baseline`` — which is every class that is a pure property of *the text, its reading order, and
whether a plausible parse happened at all*. ``table`` and
``math`` are structural assertions needing a markdown-table model and LaTeX normalisation
respectively; they are reported as :attr:`Outcome.UNSUPPORTED` with an explicit count rather
than being counted as passes or quietly dropped, and issue #187 adds them through the same
:class:`~latence_benchmark.suite.BenchmarkSuite` seam. An unsupported test is excluded from the
pass-rate denominator *and* surfaced in the results page, so a partial implementation can never
read as a better score than a complete one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from latence_benchmark.unittests import BenchmarkDataError, UnitTestType

if TYPE_CHECKING:
    from latence_benchmark.unittests import UnitTest

_WHITESPACE = re.compile(r"\s+")

#: Scoring classes that are implemented; the rest are reported UNSUPPORTED (#187).
SUPPORTED_TYPES: frozenset[UnitTestType] = frozenset(
    {UnitTestType.PRESENT, UnitTestType.ABSENT, UnitTestType.ORDER, UnitTestType.BASELINE}
)

#: A ``baseline`` row asserts nothing specific — it asserts the page produced a *plausible*
#: parse. The predicate implemented here is stated explicitly so a reader can see exactly what
#: our number means: the normalised output must be non-empty, and must not be a degenerate
#: repetition of a single character (the classic OCR/VLM collapse). When the row sets
#: ``check_disallowed_characters`` the output must additionally contain no Unicode replacement
#: character, which is what a mis-decoded text layer produces.
BASELINE_MIN_CHARS = 1
_REPLACEMENT_CHAR = "�"


def normalise(text: str) -> str:
    """Collapse every whitespace run to a single space and strip.

    Layout whitespace (column gutters, line wraps, page breaks) is a rendering artifact, not
    content, and every parser emits it differently. Comparing on normalised text is what makes
    a fuzzy budget of 0–4 edits meaningful rather than a measure of line-wrapping style.
    """
    return _WHITESPACE.sub(" ", text).strip()


def fuzzy_find(haystack: str, needle: str, max_diffs: int) -> int | None:
    """Return the end index of the leftmost occurrence of ``needle`` within ``max_diffs`` edits.

    Returns ``None`` when no such occurrence exists. An empty ``needle`` matches at 0.
    """
    if not needle:
        return 0
    exact = haystack.find(needle)
    if exact != -1:
        return exact + len(needle)
    if max_diffs <= 0 or not haystack:
        return None

    m = len(needle)
    # Sellers: column 0 is the "matched i characters, consumed nothing" prefix cost; the free
    # start is expressed by resetting the top row to 0 at every haystack position.
    previous = list(range(m + 1))
    best_end: int | None = None
    for j, hay_char in enumerate(haystack, start=1):
        current = [0] * (m + 1)
        for i in range(1, m + 1):
            substitution = previous[i - 1] + (0 if needle[i - 1] == hay_char else 1)
            current[i] = min(substitution, previous[i] + 1, current[i - 1] + 1)
        if current[m] <= max_diffs:
            best_end = j
            break
        previous = current
    return best_end


def fuzzy_contains(haystack: str, needle: str, max_diffs: int) -> bool:
    """Whether ``needle`` occurs in ``haystack`` within ``max_diffs`` edits."""
    return fuzzy_find(haystack, needle, max_diffs) is not None


class Outcome(StrEnum):
    """What happened to one unit test."""

    PASSED = "passed"
    FAILED = "failed"
    #: The harness cannot score this test class yet — never counted as a pass or a failure.
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class TestResult:
    """The outcome of one unit test, with a human-readable reason."""

    test_id: str
    split: str
    type: UnitTestType
    outcome: Outcome
    reason: str = ""


def _required(value: str | None, field: str, test_id: str) -> str:
    """Read a field the :class:`UnitTest` validator already guarantees, or fail loudly.

    Reaching the ``None`` branch means the schema and the scorer disagree, which is a bug, not
    a test failure — so it raises rather than scoring the test as failed.
    """
    if not value:
        msg = f"unit test {test_id!r} reached scoring without its required {field!r} field"
        raise BenchmarkDataError(msg)
    return value


def _window(text: str, test: UnitTest) -> str:
    """Apply a test's ``first_n``/``last_n`` character window to normalised ``text``."""
    if test.first_n is not None:
        return text[: test.first_n]
    if test.last_n is not None:
        return text[-test.last_n :]
    return text


def _score_baseline(test: UnitTest, text: str, split: str) -> TestResult:
    """Score a ``baseline`` sanity check. See :data:`BASELINE_MIN_CHARS` for the predicate."""
    if len(text) < BASELINE_MIN_CHARS:
        return TestResult(
            test.id, split, test.type, Outcome.FAILED, "the parse produced no text at all"
        )
    if len(set(text.replace(" ", ""))) <= 1:
        return TestResult(
            test.id,
            split,
            test.type,
            Outcome.FAILED,
            "the parse output is a degenerate repetition of a single character",
        )
    if test.check_disallowed_characters and _REPLACEMENT_CHAR in text:
        return TestResult(
            test.id,
            split,
            test.type,
            Outcome.FAILED,
            "the parse output contains Unicode replacement characters (mis-decoded text)",
        )
    return TestResult(test.id, split, test.type, Outcome.PASSED)


def score(test: UnitTest, output: str, *, split: str) -> TestResult:
    """Score one parser ``output`` against one ``test``."""
    if test.type not in SUPPORTED_TYPES:
        return TestResult(
            test.id,
            split,
            test.type,
            Outcome.UNSUPPORTED,
            f"{test.type.value!r} scoring is not implemented in this harness slice (see #187)",
        )

    text = normalise(output)

    if test.type is UnitTestType.BASELINE:
        return _score_baseline(test, text, split)

    if not test.case_sensitive:
        text = text.casefold()

    def prepared(value: str) -> str:
        needle = normalise(value)
        return needle if test.case_sensitive else needle.casefold()

    if test.type in (UnitTestType.PRESENT, UnitTestType.ABSENT):
        haystack = _window(text, test)
        needle = prepared(_required(test.text, "text", test.id))
        found = fuzzy_contains(haystack, needle, test.max_diffs)
        want_present = test.type is UnitTestType.PRESENT
        if found is want_present:
            return TestResult(test.id, split, test.type, Outcome.PASSED)
        verb = "not found in" if want_present else "found in"
        return TestResult(
            test.id,
            split,
            test.type,
            Outcome.FAILED,
            f"expected text was {verb} the parse output (max_diffs={test.max_diffs})",
        )

    before = prepared(_required(test.before, "before", test.id))
    after = prepared(_required(test.after, "after", test.id))
    before_end = fuzzy_find(text, before, test.max_diffs)
    after_end = fuzzy_find(text, after, test.max_diffs)
    if before_end is None or after_end is None:
        which = "before" if before_end is None else "after"
        return TestResult(
            test.id,
            split,
            test.type,
            Outcome.FAILED,
            f"the {which!r} passage does not appear in the parse output, so reading order "
            "cannot hold",
        )
    if before_end < after_end:
        return TestResult(test.id, split, test.type, Outcome.PASSED)
    return TestResult(
        test.id,
        split,
        test.type,
        Outcome.FAILED,
        "the two passages appear in the wrong reading order",
    )
