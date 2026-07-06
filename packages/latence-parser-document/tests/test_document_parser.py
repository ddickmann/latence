"""DocumentParser: PDF (per-page) + text -> markdown + page map + Classification.

Happy paths and the key failure modes: a multi-page PDF yields a real page map
that resolves offsets to the right source page; plain text is a single page;
Classification is attached at Parse from the decoded text; a corrupt/spoofed/
encrypted file degrades to a PARSE_ERROR record instead of crashing.
"""

from __future__ import annotations

from collections.abc import Callable

from latence_core.contracts import Classification, Disposition, ParserInput, Provenance
from latence_core.pagemap import PageOffsetIndex
from latence_parser_document import DocumentParser, size_after_parse


def _input(
    content: bytes | str,
    *,
    file_type: str,
    file_name: str = "doc",
    language: str = "und",
    sensitivity: str = "internal",
    category: str | None = None,
    config: dict[str, str] | None = None,
) -> ParserInput:
    size = len(content) if isinstance(content, (bytes, bytearray)) else len(content.encode())
    return ParserInput(
        provenance=Provenance(
            source_uri=f"file:///c/{file_name}.{file_type}",
            file_name=f"{file_name}.{file_type}",
            file_type=file_type,
            file_size=size,
            document_id="sha256:x",
        ),
        classification=Classification(
            language=language, sensitivity=sensitivity, category=category
        ),
        content=content,
        config=config or {},
    )


# -- PDF happy path --------------------------------------------------------


def test_pdf_yields_per_page_map_and_offsets_resolve(
    make_text_pdf: Callable[[list[str]], bytes],
) -> None:
    pdf = make_text_pdf(
        [
            "Hello from page one of the report.",
            "Second page and the end of it here.",
            "The third and final page of text.",
        ]
    )
    out = list(DocumentParser().parse([_input(pdf, file_type="pdf")]))
    assert len(out) == 1
    rec = out[0]
    assert rec.disposition is Disposition.PARSED
    assert rec.page_map is not None
    assert len(rec.page_map.pages) == 3
    # Provenance page span reflects the real page count.
    assert rec.provenance.page_start == 1
    assert rec.provenance.page_end == 3
    assert rec.provenance.char_start == 0
    assert rec.provenance.char_end == len(rec.content)
    # Text from each page is present, in order.
    assert "page one" in rec.content
    assert rec.content.index("Second page") > rec.content.index("page one")
    # Any offset resolves to the correct source page via the ported index.
    idx = PageOffsetIndex(rec.page_map)
    for span in rec.page_map.pages:
        assert idx.page_for_offset(span.char_start) == span.page_number
    assert idx.diagnostics.recovered == 0  # no drift on a well-formed doc


def test_size_after_parse_is_assembled_markdown_bytes(
    make_text_pdf: Callable[[list[str]], bytes],
) -> None:
    pdf = make_text_pdf(["Alpha beta.", "Gamma delta."])
    rec = next(iter(DocumentParser().parse([_input(pdf, file_type="pdf")])))
    assert size_after_parse(rec) == len(rec.content.encode("utf-8"))


# -- text happy path -------------------------------------------------------


def test_text_is_single_page_with_classification_at_parse() -> None:
    raw = _input(
        b"the framework reads the folder and exports the records",
        file_type="txt",
        language="und",  # Source left it undetermined; Parse must detect it
    )
    rec = next(iter(DocumentParser().parse([raw])))
    assert rec.disposition is Disposition.PARSED
    assert rec.page_map is not None
    assert len(rec.page_map.pages) == 1
    # Classification attached at Parse: language auto-detected from decoded text.
    assert rec.classification.language == "en"
    assert rec.classification.sensitivity == "internal"  # inherited from the input


def test_markdown_extension_and_str_content() -> None:
    rec = next(iter(DocumentParser().parse([_input("# Title\n\nBody.", file_type="md")])))
    assert rec.content == "# Title\n\nBody."
    assert rec.media_type == "text/markdown"


def test_category_override_and_inheritance() -> None:
    # Config category overrides the inherited one.
    parser = DocumentParser({"category": "contract"})
    rec = next(iter(parser.parse([_input(b"hello the and of", file_type="txt", category="memo")])))
    assert rec.classification.category == "contract"
    # With no override, the inherited category is preserved.
    rec2 = next(
        iter(DocumentParser().parse([_input(b"hello the and of", file_type="txt", category="memo")]))
    )
    assert rec2.classification.category == "memo"


def test_per_input_encoding_hint() -> None:
    raw = _input("café".encode("latin-1"), file_type="txt", config={"encoding": "latin-1"})
    rec = next(iter(DocumentParser().parse([raw])))
    assert rec.content == "café"


# -- failure modes: graceful degradation, never crash ----------------------


def test_corrupt_pdf_becomes_parse_error_record() -> None:
    # Valid %PDF- header but garbage body: pypdf fails, we record it.
    corrupt = b"%PDF-1.4\nthis is not a real pdf body at all\n%%EOF"
    rec = next(iter(DocumentParser().parse([_input(corrupt, file_type="pdf")])))
    assert rec.disposition is Disposition.PARSE_ERROR
    assert rec.error is not None
    assert rec.content == ""
    assert rec.page_map is None
    # Provenance is still intact for audit.
    assert rec.provenance.file_type == "pdf"


def test_type_spoofed_pdf_is_rejected_without_crash() -> None:
    # A .pdf whose bytes are not a PDF (no %PDF- signature).
    spoof = b"MZ\x90\x00this is actually an exe pretending to be a pdf"
    rec = next(iter(DocumentParser().parse([_input(spoof, file_type="pdf")])))
    assert rec.disposition is Disposition.PARSE_ERROR
    assert "signature" in rec.error.lower()  # type: ignore[union-attr]


def test_unsupported_file_type_is_parse_error() -> None:
    rec = next(iter(DocumentParser().parse([_input(b"\x00\x01\x02", file_type="xlsx")])))
    assert rec.disposition is Disposition.PARSE_ERROR
    assert "unsupported" in rec.error.lower()  # type: ignore[union-attr]


def test_one_bad_document_does_not_stop_the_others() -> None:
    good = _input(b"the and of to is in", file_type="txt", file_name="good")
    bad = _input(b"%PDF-1.4\nbroken\n", file_type="pdf", file_name="bad")
    good2 = _input(b"the and of to is in", file_type="md", file_name="good2")
    out = list(DocumentParser().parse([good, bad, good2]))
    assert [r.disposition for r in out] == [
        Disposition.PARSED,
        Disposition.PARSE_ERROR,
        Disposition.PARSED,
    ]


def test_empty_input_yields_nothing() -> None:
    assert list(DocumentParser().parse([])) == []
