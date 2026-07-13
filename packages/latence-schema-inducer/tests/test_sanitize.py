"""Unit tests for the induced-label sanitizer (W4 safety net #5) — deterministic, offline."""

from __future__ import annotations

from latence_schema_inducer.sanitize import (
    MAX_LABEL_LEN,
    MAX_LABELS,
    sanitize_label,
    sanitize_labels,
)


def test_strips_control_chars_and_trims() -> None:
    assert sanitize_label("  per\x00son \n") == "person"
    assert sanitize_label("\t\torg\x07") == "org"


def test_collapses_internal_whitespace() -> None:
    assert sanitize_label("credit   card\tnumber") == "credit card number"


def test_empty_and_whitespace_only_dropped() -> None:
    assert sanitize_label("") is None
    assert sanitize_label("   \t\n ") is None
    assert sanitize_label("\x00\x07") is None


def test_length_capped() -> None:
    out = sanitize_label("x" * 5000)
    assert out is not None
    assert len(out) == MAX_LABEL_LEN


def test_non_str_coerced() -> None:
    assert sanitize_label(123) == "123"
    # str(None) is a non-empty label; upstream schema rejects it. Canonicalized (casefolded).
    assert sanitize_label(None) == "none"


def test_canonical_normal_form_casefold_and_separator_collapse() -> None:
    # W12: casefold + treat '_'/'-'/whitespace as space-equivalent → ONE normal form per label.
    assert sanitize_label("Date") == "date"
    assert sanitize_label("date") == "date"
    assert sanitize_label("payment_product") == "payment product"
    assert sanitize_label("payment product") == "payment product"
    assert sanitize_label("payment-product") == "payment product"
    assert sanitize_label("Website") == "website"


def test_canonical_form_collapses_case_and_format_variants_to_two() -> None:
    # The spec's headline case: Date/date + payment_product/payment product → 2 canonical labels.
    result = sanitize_labels(["Date", "date", "payment_product", "payment product"])
    assert result.labels == ["date", "payment product"]
    assert result.dropped_for_count == 0


def test_list_dedupes_case_and_format_insensitively() -> None:
    result = sanitize_labels(["Person", "person", "PERSON", "Org"])
    assert result.labels == ["person", "org"]
    assert result.dropped_for_count == 0


def test_list_caps_count_and_reports_drops() -> None:
    result = sanitize_labels([f"type_{i}" for i in range(MAX_LABELS + 15)])
    assert len(result.labels) == MAX_LABELS
    assert result.dropped_for_count == 15  # no silent truncation — the overflow is reported


def test_list_order_stable() -> None:
    result = sanitize_labels(["gamma", "alpha", "beta"])
    assert result.labels == ["gamma", "alpha", "beta"]


def test_non_list_yields_empty() -> None:
    assert sanitize_labels("person").labels == []
    assert sanitize_labels({"a": 1}).labels == []
    assert sanitize_labels(None).labels == []


def test_mixed_junk_and_valid() -> None:
    # 'also_valid' canonicalizes to 'also valid' ('_' is space-equivalent); 'valid' dedupes.
    result = sanitize_labels(["  valid ", "", "\x00", "also_valid", "valid"])
    assert result.labels == ["valid", "also valid"]
