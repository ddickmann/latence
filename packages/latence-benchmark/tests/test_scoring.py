"""Scoring semantics: the fuzzy budget, the four assertions, and honest UNSUPPORTED."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest
from latence_benchmark import scoring
from latence_benchmark.scoring import (
    Outcome,
    _semantic_source,
    fuzzy_contains,
    fuzzy_find,
    normalise,
    score,
    scorer_pin,
)
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


def test_a_later_exact_hit_never_hides_the_leftmost_within_budget_hit() -> None:
    """The exact `str.find` is a BOUND on the Sellers scan, never an answer to it.

    `fuzzy_find` promises "the end index of the leftmost occurrence within `max_diffs` edits".
    Short-circuiting on `str.find` returned the *exact* occurrence's end even when an
    approximate one ended earlier, so the answer depended on whether a later exact hit existed
    at all — which is what let a noisier parse score better on `order` (audit R7).
    """
    # "aXc" is one substitution from "abc" and ends at 12; the exact "abc" ends at 24.
    with_a_later_exact_hit = "zzzz aXc zzzzzzzzzzzzzzz abc"
    assert with_a_later_exact_hit.find("abc") == 25
    assert fuzzy_find(with_a_later_exact_hit, "abc", 1) == 8

    # Deleting the later exact occurrence must not change where the leftmost hit is.
    assert fuzzy_find("zzzz aXc zzzzzzzzzzzzzzz ---", "abc", 1) == 8


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


def test_adding_an_ocr_error_can_never_flip_an_order_verdict() -> None:
    """A parser must never improve its number by degrading its output (audit R7).

    A running header repeats "Introduction" further down the page, so the parse contains an
    OCR-damaged first occurrence *and* a clean later one. Scoring the clean one made the
    `before` passage look like it came after `after`; corrupting it too — strictly more OCR
    damage — exposed the earlier within-budget occurrence and flipped the same test to PASSED.
    """
    order = _test(type="order", text=None, before="Introduction", after="Conclusion", max_diffs=1)
    damaged = "Intrpduction ... Conclusion ... Introduction ..."
    more_damaged = "Intrpduction ... Conclusion ... Intreduction ..."

    assert (
        score(order, damaged, split="s").outcome
        is score(order, more_damaged, split="s").outcome
    )


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


# -- baseline ------------------------------------------------------------------------------


def test_baseline_passes_on_a_plausible_parse() -> None:
    row = _test(type="baseline", text=None)
    assert score(row, "A page with genuine content on it.", split="s").outcome is Outcome.PASSED


def test_baseline_fails_on_an_empty_parse() -> None:
    """The failure a `present`/`absent` pair cannot express: the page yielded nothing."""
    row = _test(type="baseline", text=None)
    result = score(row, "   \n\n  ", split="s")
    assert result.outcome is Outcome.FAILED
    assert "no text at all" in result.reason


def test_baseline_fails_on_a_degenerate_repetition() -> None:
    row = _test(type="baseline", text=None)
    result = score(row, "aaaa aaaa aaaaaa aaa", split="s")
    assert result.outcome is Outcome.FAILED
    assert "degenerate" in result.reason


def test_baseline_flags_mis_decoded_text_only_when_the_row_asks() -> None:
    lenient = _test(type="baseline", text=None)
    strict = _test(type="baseline", text=None, check_disallowed_characters=True)
    garbled = "some text with a � replacement character"
    assert score(lenient, garbled, split="s").outcome is Outcome.PASSED
    assert score(strict, garbled, split="s").outcome is Outcome.FAILED


def test_a_baseline_row_needs_no_assertion_payload() -> None:
    """The real dataset's baseline rows carry only metadata — the schema must accept that."""
    row = UnitTest.model_validate(
        {
            "pdf": "tables/9e6a_pg12.pdf",
            "page": 1,
            "id": "9e6a_pg12_table_baseline",
            "type": "baseline",
            "max_diffs": 0,
            "checked": "verified",
            "check_disallowed_characters": False,
        }
    )
    assert row.type is UnitTestType.BASELINE


# -- the scorer pin ------------------------------------------------------------------------


def test_the_pin_records_which_classes_this_scorer_actually_scores() -> None:
    pin = scorer_pin()
    assert pin.supported_types == ("absent", "baseline", "order", "present")
    assert pin.unsupported_types == ("table", "math")
    assert set(pin.supported_types) | set(pin.unsupported_types) == {t.value for t in UnitTestType}


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("SUPPORTED_TYPES", frozenset(UnitTestType)), ("BASELINE_MIN_CHARS", 25)],
)
def test_changing_what_the_scorer_does_moves_its_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: Any
) -> None:
    """Otherwise a published pass rate could be re-described by rules that never produced it."""
    before = scorer_pin().fingerprint
    monkeypatch.setattr(scoring, attribute, value)
    assert scorer_pin().fingerprint != before


def _pre_r5_score_document_failure(test: UnitTest, *, split: str, reason: str) -> Any:
    """The document-failure rule as it stood before ``fa35f06``: every test FAILED.

    Under this rule a `table`/`math` test of a `PARSE_ERROR` or missing-PDF document lands inside
    the pass-rate denominator; under today's rule it stays UNSUPPORTED, outside it. A synthetic
    split with one `present`, one `table` and one `math` test scores 0/1/2 (scored 1) today and
    0/3/0 (scored 3) under this one — a different published number.
    """
    return scoring.TestResult(test.id, split, test.type, Outcome.FAILED, reason)


def _no_window(text: str, test: UnitTest) -> str:
    """``_window`` with the ``first_n``/``last_n`` rule removed — a real change to the metric."""
    return text


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        # The rule audit R5 changed. It decides whether an unscoreable test of a failed document
        # is FAILED (in the denominator) or UNSUPPORTED (out of it) — a verdict of its own.
        ("score_document_failure", _pre_r5_score_document_failure),
        # A helper `score` delegates a scoring rule to.
        ("_window", _no_window),
        # Constants read inside fingerprinted functions but written outside them: what `normalise`
        # collapses, and what a `baseline` row rejects as mis-decoded text.
        ("_WHITESPACE", re.compile(r"[ ]+")),
        ("_REPLACEMENT_CHAR", "?"),
    ],
)
def test_every_rule_the_metric_reaches_moves_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: Any
) -> None:
    """The pin is only a guard if it covers *every* rule, not the five functions someone listed.

    `_SCORING_BEHAVIOUR` used to be the fingerprinted set itself, so helpers and module constants
    were opted out by default: swapping any of these left the digest — and therefore the
    fingerprint stamped on every published measurement — bit-for-bit identical, and
    `carry_forward` republished a pass rate the changed rules would not have produced.
    """
    before = scorer_pin().fingerprint
    monkeypatch.setattr(scoring, attribute, value)
    assert scorer_pin().fingerprint != before


def test_the_fingerprint_ignores_docstrings_comments_and_formatting() -> None:
    """A prose rewrite must not invalidate a measurement; only changed behaviour may."""

    def original() -> Callable[[str], str]:
        def probe(text: str) -> str:
            """Original wording."""
            # a comment
            return text.strip()

        return probe

    def reworded() -> Callable[[str], str]:
        def probe(text: str) -> str:
            """Completely different wording, several sentences — same behaviour."""
            return text.strip()

        return probe

    def rebehaved() -> Callable[[str], str]:
        def probe(text: str) -> str:
            """Original wording."""
            return text.strip().casefold()

        return probe

    assert _semantic_source(original()) == _semantic_source(reworded())
    assert _semantic_source(original()) != _semantic_source(rebehaved())
