# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: Apache-2.0
"""The shared S2 OCR logic: table normalization, repetition flag, manifest, fin cap.

These functions run in two places — the campaign stage locally and ``ocr_worker.py`` on
the pod (the stage pushes this exact module there) — so this suite is the single proof
both sides rely on.
"""

from __future__ import annotations

from latence_benchmark.campaign.ocr_lib import (
    assemble_document,
    has_repetition,
    missing_pages,
    normalize_html_tables,
    page_filename,
    provenance_header,
    qa_coverage,
    safe_zip_entry,
    sample_spot_docs,
    select_capped_docs,
    spot_page_for,
)


class TestFinCapSelection:
    def test_top_docs_by_count_with_ascending_name_tie_break(self) -> None:
        counts = {"delta": 3, "alpha": 3, "carol": 5, "bob": 1}
        assert select_capped_docs(counts, 3) == ("carol", "alpha", "delta")

    def test_cap_larger_than_population_selects_everything(self) -> None:
        counts = {"a": 1, "b": 2}
        assert select_capped_docs(counts, 150) == ("b", "a")

    def test_selection_is_deterministic_across_calls(self) -> None:
        counts = {f"doc{i:03d}": (i * 7) % 13 for i in range(300)}
        first = select_capped_docs(counts, 150)
        assert first == select_capped_docs(dict(reversed(list(counts.items()))), 150)

    def test_qa_coverage_counts_selected_over_total(self) -> None:
        counts = {"a": 10, "b": 5, "c": 1}
        assert qa_coverage(counts, ("a", "b")) == (15, 16)


class TestZipSlipGuard:
    def test_plain_nested_entries_are_safe(self) -> None:
        assert safe_zip_entry("fin_docs/AAL_2018.pdf")
        assert safe_zip_entry("academic/deep/nested/file.pdf")

    def test_parent_traversal_is_rejected(self) -> None:
        assert not safe_zip_entry("../evil.pdf")
        assert not safe_zip_entry("fine/../../evil.pdf")

    def test_absolute_and_drive_paths_are_rejected(self) -> None:
        assert not safe_zip_entry("/etc/passwd")
        assert not safe_zip_entry("C:evil.pdf")
        assert not safe_zip_entry("\\\\server\\share.pdf")

    def test_backslash_separators_and_empty_names_are_rejected(self) -> None:
        assert not safe_zip_entry("dir\\file.pdf")
        assert not safe_zip_entry("")
        assert not safe_zip_entry("dir//file.pdf")  # empty component


class TestRepetitionDetector:
    def test_exactly_four_repeats_is_legitimate_text(self) -> None:
        phrase = "one two three four five six seven eight "
        assert has_repetition(phrase * 4) is False

    def test_five_repeats_flags_the_loop(self) -> None:
        phrase = "one two three four five six seven eight "
        assert has_repetition(phrase * 5) is True

    def test_short_text_never_flags(self) -> None:
        assert has_repetition("too short to hold an eight gram") is False
        assert has_repetition("") is False

    def test_varied_long_text_does_not_flag(self) -> None:
        text = " ".join(f"word{i}" for i in range(500))
        assert has_repetition(text) is False


class TestTableNormalization:
    def test_basic_table_becomes_github_markdown(self) -> None:
        page = (
            "before\n"
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
            "\nafter"
        )
        result = normalize_html_tables(page)
        assert result.converted == 1
        assert result.kept_raw == 0
        assert result.text == (
            "before\n| A | B |\n| --- | --- |\n| 1 | 2 |\nafter"
        )

    def test_colspan_flattens_to_repeated_cells(self) -> None:
        page = '<table><tr><td colspan="2">X</td><td>Y</td></tr></table>'
        result = normalize_html_tables(page)
        assert result.converted == 1
        assert result.text == "| X | X | Y |\n| --- | --- | --- |"

    def test_invalid_colspan_defaults_to_one(self) -> None:
        page = '<table><tr><td colspan="abc">X</td></tr></table>'
        assert normalize_html_tables(page).text == "| X |\n| --- |"

    def test_nested_table_is_linearized_into_the_outer_cell(self) -> None:
        page = (
            "<table><tr><td>out<table><tr><td>in</td></tr></table></td>"
            "<td>plain</td></tr></table>"
        )
        result = normalize_html_tables(page)
        assert result.converted == 1
        assert result.kept_raw == 0
        assert result.text == "| out in | plain |\n| --- | --- |"

    def test_ragged_rows_are_padded_to_the_widest(self) -> None:
        page = "<table><tr><td>a</td><td>b</td></tr><tr><td>only</td></tr></table>"
        assert normalize_html_tables(page).text == (
            "| a | b |\n| --- | --- |\n| only |  |"
        )

    def test_pipes_inside_cells_are_escaped(self) -> None:
        page = "<table><tr><td>a|b</td></tr></table>"
        assert normalize_html_tables(page).text == "| a\\|b |\n| --- |"

    def test_unclosed_table_is_kept_raw_and_counted(self) -> None:
        page = "text <table><tr><td>a</td></tr> more text"
        result = normalize_html_tables(page)
        assert result.converted == 0
        assert result.kept_raw == 1
        assert result.text == page  # byte-identical: nothing lost

    def test_rowless_table_is_kept_raw(self) -> None:
        page = "x <table></table> y"
        result = normalize_html_tables(page)
        assert result.converted == 0
        assert result.kept_raw == 1
        assert result.text == page

    def test_multiple_tables_are_converted_independently(self) -> None:
        page = (
            "<table><tr><td>1</td></tr></table> gap <table></table>"
        )
        result = normalize_html_tables(page)
        assert result.converted == 1
        assert result.kept_raw == 1
        assert result.text == "| 1 |\n| --- | gap <table></table>"

    def test_page_without_tables_is_untouched(self) -> None:
        page = "# Heading\n\nplain markdown, a < b and b > a, no tables."
        result = normalize_html_tables(page)
        assert (result.text, result.converted, result.kept_raw) == (page, 0, 0)

    def test_implied_closes_are_repaired_like_a_browser(self) -> None:
        # No </td> / </tr> anywhere — new cells/rows imply the closes.
        page = "<table><tr><td>a<td>b<tr><td>c<td>d</table>"
        assert normalize_html_tables(page).text == (
            "| a | b |\n| --- | --- |\n| c | d |"
        )


class TestManifestPlanning:
    def test_page_filename_is_four_digit_markdown(self) -> None:
        assert page_filename(1) == "0001.md"
        assert page_filename(1234) == "1234.md"

    def test_missing_pages_skips_existing_artifacts(self) -> None:
        existing = {"0001.md", "0003.md"}
        assert missing_pages(4, existing) == (2, 4)

    def test_no_existing_artifacts_plans_everything(self) -> None:
        assert missing_pages(3, set()) == (1, 2, 3)

    def test_complete_manifest_plans_nothing(self) -> None:
        assert missing_pages(2, {"0001.md", "0002.md"}) == ()


class TestAssembly:
    def test_provenance_header_carries_the_five_fields(self) -> None:
        header = provenance_header(
            doc_id="AAL_2018", dataset="uda", page_count=12,
            model="lightonai/LightOnOCR-2-1B", revision="main",
        )
        assert header.startswith("<!--") and header.endswith("-->")
        for fragment in (
            "doc_id=AAL_2018", "dataset=uda", "pages=12",
            "model=lightonai/LightOnOCR-2-1B", "revision=main",
        ):
            assert fragment in header

    def test_assembled_document_joins_pages_with_blank_lines(self) -> None:
        text = assemble_document(
            ["page one", "page two"],
            doc_id="d", dataset="ohr_bench", model="m", revision="r",
        )
        header, body = text.split("\n\n", 1)
        assert "pages=2" in header
        assert body == "page one\n\npage two\n"


class TestSpotCheckSampling:
    # Doc-first by necessity: the streaming worker deletes a doc's images at cleanup,
    # so the sample must be decidable per doc — 12 docs upfront, one stable page each.

    def test_small_doc_population_selects_every_doc(self) -> None:
        assert sample_spot_docs(["b", "a", "c"]) == ("a", "b", "c")

    def test_large_population_yields_twelve_sorted_unique_docs(self) -> None:
        keys = [f"ds/doc{i:03d}" for i in range(200)]
        docs = sample_spot_docs(keys)
        assert len(docs) == 12
        assert list(docs) == sorted(docs)
        assert len(set(docs)) == 12

    def test_doc_sampling_ignores_input_order(self) -> None:
        keys = [f"ds/doc{i:03d}" for i in range(50)]
        assert sample_spot_docs(keys) == sample_spot_docs(list(reversed(keys)))

    def test_spot_page_is_within_the_doc_and_stable(self) -> None:
        page = spot_page_for("ds/x", 17)
        assert 1 <= page <= 17
        assert page == spot_page_for("ds/x", 17)  # relaunches retain the same evidence

    def test_spot_page_varies_across_docs(self) -> None:
        pages = {spot_page_for(f"ds/doc{i}", 1000) for i in range(20)}
        assert len(pages) > 1  # keyed per doc, not one global page number

    def test_tiny_docs_pin_page_one(self) -> None:
        assert spot_page_for("ds/one-pager", 1) == 1
        assert spot_page_for("ds/empty", 0) == 1
