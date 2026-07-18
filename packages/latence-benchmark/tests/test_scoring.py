"""Scoring semantics: the fuzzy budget, the four assertions, and honest UNSUPPORTED."""

from __future__ import annotations

from typing import Any

import pytest

from latence_benchmark.scoring import Outcome, fuzzy_contains, fuzzy_find, normalise, score
from latence_benchmark.unittests import UnitTest, UnitTestType


def _test(**overrides: Any) -> UnitTest:
    payload: dict[str, Any] = {
        "id": "t1",
        "pdf": "x/y.pdf",
        "type": "present",
        "max_diffs": 0,
        "text": "hello world",
        "case_sensitive": True,
    }
    payload.update(overrides)
    return UnitTest.model_validate(payload)


# -- the fuzzy primitive -------------------------------------------------------------------


def test_exact_matches_need_no_budget() -> None:
    assert fuzzy_find("the quick brown fox", "quick", 0) == 9


def test_a_substitution_is_within_a_one_edit_budget() -> None:
    assert fuzzy_contains("the quick brown fox", "quiek", 1)
    assert not fuzzy_contains("the quick brown fox", "quiek", 0)


def test_an_insertion_and_a_deletion_are_both_edits() -> None:
    assert fuzzy_contains("brown fox", "browwn", 1)
    assert fuzzy_contains("brown fox", "brwn", 1)
    assert not fuzzy_contains("brown fox", "brwn", 0)


def test_the_budget_is_a_ceiling_not_a_suggestion() -> None:
    assert not fuzzy_contains("brown fox", "xyzzy plugh", 3)


def test_an_empty_needle_matches_at_the_start() -> None:
    assert fuzzy_find("anything", "", 0) == 0


def test_layout_whitespace_is_normalised_away() -> None:
    assert normalise("a\n\n  b\tc  ") == "a b c"


# -- present / absent ----------------------------------------------------------------------


def test_present_passes_when_the_passage_survived_the_parse() -> None:
    result = score(_test(), "prelude. Hello\nworld. postlude", split="s")
    assert result.outcome is Outcome.FAILED  # case-sensitive: "Hello" != "hello"

    result = score(_test(case_sensitive=False), "prelude. Hello\nworld. postlude", split="s")
    assert result.outcome is Outcome.PASSED


def test_present_fails_when_the_passage_is_gone() -> None:
    result = score(_test(), "nothing of the sort", split="s")
    assert result.outcome is Outcome.FAILED
    assert "not found" in result.reason


def test_absent_inverts_present() -> None:
    header = _test(type="absent", text="Page 3 of 9")
    assert score(header, "body text only", split="s").outcome is Outcome.PASSED
    assert score(header, "Page 3 of 9\nbody text", split="s").outcome is Outcome.FAILED


def test_first_n_restricts_the_absent_assertion_to_the_top_of_the_page() -> None:
    """`first_n` is why a page number in the FOOTER does not fail a HEADER assertion."""
    header = _test(type="absent", text="31", first_n=10, case_sensitive=False)
    assert score(header, "31 is the header line", split="s").outcome is Outcome.FAILED
    assert score(header, "clean body text ......... 31", split="s").outcome is Outcome.PASSED


def test_last_n_restricts_to_the_bottom_of_the_page() -> None:
    footer = _test(type="absent", text="confidential", last_n=20, case_sensitive=False)
    assert score(footer, "confidential" + " x" * 40, split="s").outcome is Outcome.PASSED


# -- order ---------------------------------------------------------------------------------


def test_order_passes_when_reading_order_is_preserved() -> None:
    order = _test(type="order", text=None, before="alpha", after="omega")
    assert score(order, "alpha ... middle ... omega", split="s").outcome is Outcome.PASSED


def test_order_fails_when_the_columns_were_interleaved() -> None:
    """The multi-column failure mode: both passages present, wrong sequence."""
    order = _test(type="order", text=None, before="alpha", after="omega")
    result = score(order, "omega ... middle ... alpha", split="s")
    assert result.outcome is Outcome.FAILED
    assert "wrong reading order" in result.reason


def test_order_fails_when_a_passage_is_missing_entirely() -> None:
    order = _test(type="order", text=None, before="alpha", after="omega")
    result = score(order, "alpha only", split="s")
    assert result.outcome is Outcome.FAILED
    assert "'after'" in result.reason


# -- honest coverage -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "table", "text": None, "cell": "0.569"},
        {"type": "math", "text": None, "math": "x^2"},
    ],
)
def test_unimplemented_classes_are_unsupported_never_a_free_pass(payload: dict[str, Any]) -> None:
    """A class we cannot score must not count as a pass — that would inflate the number.

    It must not count as a failure either, which would understate a parser we never tested.
    UNSUPPORTED keeps it out of the pass-rate denominator AND visible in the results page.
    """
    result = score(_test(**payload), "anything at all", split="s")
    assert result.outcome is Outcome.UNSUPPORTED
    assert "#187" in result.reason
    assert result.type in (UnitTestType.TABLE, UnitTestType.MATH)
