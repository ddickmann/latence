"""Unit tests for the induced-label sanitizer (W4 safety net #5) — deterministic, offline."""

from __future__ import annotations

from latence_schema_inducer.sanitize import (
    MAX_LABEL_LEN,
    MAX_LABELS,
    is_value_shaped,
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


# ---------------------------------------------------------------------------
# The S3 value-shape guard — one test per rule, with the EXACT junk strings the
# live ohr_bench run stamped as types (each observed with 1 mention) as fixtures.
# ---------------------------------------------------------------------------


def _guarded(raw: list[str]) -> list[str]:
    """Run the full sanitize path WITH the value-shape guard on (the induced-label path)."""
    return sanitize_labels(raw, reject_value_shapes=True).labels


def test_rule1_purely_numeric_decimal_percentage_rejected() -> None:
    # Observed junk: "10.3", "10.6" — bare metric values stamped as entity types.
    for junk in ("10.3", "10.6", "42", "1,000", "15%"):
        assert is_value_shaped(junk), junk
    assert _guarded(["10.3", "10.6", "42", "15%"]) == []


def test_rule2_digit_ratio_and_fiscal_shapes_rejected() -> None:
    # Observed junk: "1q22" (a fiscal quarter, digit ratio 0.75). Fiscal/quarter shapes reject
    # even where the ratio alone would not ("fy2023" is also caught by ratio; "q3 2024" by both).
    for junk in ("1q22", "q3 2024", "fy2023", "fy 2023"):
        assert is_value_shaped(junk), junk
    assert _guarded(["1q22", "q3 2024", "fy2023"]) == []


def test_rule3_date_shapes_rejected() -> None:
    # Observed junk: "october 27, 1986"-style dates and the "15 to 24 years" range. ISO dates
    # arrive canonicalized ("2024-01-15" → "2024 01 15") and are rejected in either spelling.
    for junk in ("october 27, 1986", "15 to 24 years", "2024 01 15", "2024/01/15", "may 2024"):
        assert is_value_shaped(junk), junk
    assert _guarded(["october 27, 1986", "15 to 24 years", "2024-01-15"]) == []


def test_rule4_address_shapes_rejected() -> None:
    # Observed junk: "1024 fulton street", "10 chestnut st." — a digit + a street suffix token.
    for junk in ("1024 fulton street", "10 chestnut st.", "7 elm avenue", "22 baker rd"):
        assert is_value_shaped(junk), junk
    assert _guarded(["1024 fulton street", "10 chestnut st."]) == []
    # A street suffix WITHOUT a digit is not address evidence — "main street" survives.
    assert not is_value_shaped("main street")


def test_rule5_artifact_fragments_and_measurement_heads_rejected() -> None:
    # Observed junk: ".net" (leading dot), "1.2 mb floppy disc", "20 mb hard disc" (number + unit
    # head names one artifact/spec, not a type). Single characters are fragments too.
    for junk in (".net", "x", "1.2 mb floppy disc", "20 mb hard disc"):
        assert is_value_shaped(junk), junk
    assert _guarded([".net", "1.2 mb floppy disc", "20 mb hard disc"]) == []
    # A dot INSIDE a longer label is legitimate — only the leading-dot fragment shape rejects.
    assert not is_value_shaped("asp.net framework")
    assert not is_value_shaped("u.s. government")


def test_value_shape_guard_keeps_legitimate_types() -> None:
    # The keeper set: ordinary type names, including ones that merely CONTAIN a digit.
    keepers = ["person", "financial_metric", "works_for", "central bank", "2008 farm bill"]
    result = sanitize_labels(keepers, reject_value_shapes=True)
    assert result.labels == [
        "person", "financial metric", "works for", "central bank", "2008 farm bill",
    ]
    assert result.dropped_for_shape == 0


def test_borderline_rulings_decided_by_digit_ratio() -> None:
    # The two documented borderline rulings (module docstring): the digit-ratio rule decides.
    # "2008 farm bill": 4 digits / 12 non-space chars ≈ 0.33 ≤ 0.4 → KEPT.
    assert not is_value_shaped("2008 farm bill")
    # "1933 act": 4 digits / 7 non-space chars ≈ 0.57 > 0.4 → REJECTED.
    assert is_value_shaped("1933 act")
    assert _guarded(["2008 farm bill", "1933 act"]) == ["2008 farm bill"]


def test_value_shape_guard_is_opt_in_config_labels_untouched() -> None:
    # The guard is OFF by default: a hand-authored config label set (the lexicon path) is trusted
    # and must pass through unchanged — even when a label happens to look like a value.
    result = sanitize_labels(["10.3", ".net", "1q22", "person"])
    assert result.labels == ["10.3", ".net", "1q22", "person"]
    assert result.dropped_for_shape == 0


def test_shape_drops_ride_the_auditable_accounting() -> None:
    # No new silent path: every distinct shape-reject is tallied on dropped_for_shape, separate
    # from (and not corrupting) the count cap's dropped_for_count. Duplicates dedup first, so a
    # repeated junk shape counts once.
    result = sanitize_labels(
        ["person", "10.3", "1q22", "10.3", "central bank"], reject_value_shapes=True
    )
    assert result.labels == ["person", "central bank"]
    assert result.dropped_for_shape == 2  # "10.3" (once) + "1q22"
    assert result.dropped_for_count == 0
    # Shape-rejected junk never consumes a ceiling slot: 3 keepers fit a ceiling of 3 even when
    # junk precedes them in the list.
    capped = sanitize_labels(
        ["10.3", "person", "org", "central bank"], max_labels=3, reject_value_shapes=True
    )
    assert capped.labels == ["person", "org", "central bank"]
    assert capped.dropped_for_shape == 1
    assert capped.dropped_for_count == 0


def test_value_shape_guard_deterministic() -> None:
    # Pure + deterministic: repeated runs over the same input are byte-identical.
    mixed = ["person", "1q22", "10.3", ".net", "1024 fulton street", "2008 farm bill"]
    first = sanitize_labels(mixed, reject_value_shapes=True)
    for _ in range(5):
        again = sanitize_labels(mixed, reject_value_shapes=True)
        assert again == first
    assert first.labels == ["person", "2008 farm bill"]


def test_value_shape_guard_unicode_safe() -> None:
    # Unicode digits count as digits (ratio rule); non-ASCII type names are untouched keepers.
    assert is_value_shaped("١٢٣")  # Arabic-Indic digits: purely numeric by Unicode digit semantics
    assert not is_value_shaped("rechnungsnummer")
    assert not is_value_shaped("zolltarifnummer der ware")
