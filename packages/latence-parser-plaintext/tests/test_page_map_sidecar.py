"""PlaintextParser page provenance across the out-of-band OCR handoff (ADR-0060).

The defect this pins: markdown that was paginated somewhere else and re-ingested used to
parse into ONE page span, so every downstream record cited ``page 1`` — populated, plausible,
and wrong on 236k chunks of the S3 corpus. These tests fix the three outcomes the parser is
now allowed to have: a real map from the sidecar, an ADMITTED assumption when there is none,
and a refusal when the operator declared the corpus paginated or the sidecar is stale.
"""

from __future__ import annotations

import json

from latence_core.contracts import (
    Classification,
    Disposition,
    PageMap,
    PageMapOrigin,
    ParserInput,
    Provenance,
)
from latence_parser_plaintext import PlaintextParser

PAGES = ["page one text", "page two text", "page three text"]


def _assembly() -> tuple[str, bytes]:
    """A three-page document plus the sidecar bytes describing it."""
    assembled = PageMap.from_page_texts(list(PAGES))
    sidecar = json.dumps(
        {
            "pages": [span.model_dump() for span in assembled.page_map.pages],
            "total_chars": assembled.page_map.total_chars,
        }
    ).encode("utf-8")
    return assembled.text, sidecar


def _raw(
    content: bytes | str,
    *,
    sidecar: bytes | None = None,
) -> ParserInput:
    return ParserInput(
        provenance=Provenance(
            source_uri="file:///c/doc.md",
            file_name="doc.md",
            file_type="md",
            file_size=len(content),
            document_id="sha256:doc",
        ),
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        page_map_sidecar=sidecar,
    )


def test_sidecar_restores_the_real_page_boundaries() -> None:
    text, sidecar = _assembly()
    parsed = next(iter(PlaintextParser().parse([_raw(text, sidecar=sidecar)])))

    assert parsed.disposition is Disposition.PARSED
    assert parsed.content == text  # the document itself is untouched by the handoff
    assert parsed.page_map is not None
    assert len(parsed.page_map.pages) == 3
    assert parsed.page_map.origin is PageMapOrigin.SIDECAR
    # The whole point: an offset in the third page resolves to page 3, not page 1.
    third = text.index(PAGES[2])
    assert parsed.page_map.pages[2].char_start <= third < parsed.page_map.pages[2].char_end
    # And the document's Provenance spans the real page range.
    assert parsed.provenance.page_start == 1
    assert parsed.provenance.page_end == 3


def test_no_sidecar_admits_the_assumption_instead_of_claiming_page_one() -> None:
    parsed = next(iter(PlaintextParser().parse([_raw("a page-less note")])))

    assert parsed.page_map is not None
    # Resolution is byte-identical to the pre-ADR behaviour...
    assert len(parsed.page_map.pages) == 1
    assert parsed.provenance.page_start == 1
    assert parsed.provenance.page_end == 1
    # ...but the record no longer CLAIMS the document has one page, which is the difference
    # between a detectable degradation and the silent one that shipped.
    assert parsed.page_map.origin is PageMapOrigin.ASSUMED_SINGLE_PAGE


def test_require_page_map_refuses_a_document_with_no_page_information() -> None:
    parser = PlaintextParser({"require_page_map": True})
    parsed = next(iter(parser.parse([_raw("a page-less note")])))

    assert parsed.disposition is Disposition.PARSE_ERROR
    assert parsed.page_map is None
    assert parsed.content == ""
    assert parsed.error is not None
    assert ".pagemap.json" in parsed.error
    # Provenance survives so the refusal is auditable per document (ADR-0020).
    assert parsed.provenance.document_id == "sha256:doc"


def test_require_page_map_is_satisfied_by_a_sidecar() -> None:
    text, sidecar = _assembly()
    parser = PlaintextParser({"require_page_map": True})
    parsed = next(iter(parser.parse([_raw(text, sidecar=sidecar)])))

    assert parsed.disposition is Disposition.PARSED
    assert parsed.page_map is not None
    assert len(parsed.page_map.pages) == 3


def test_a_stale_sidecar_is_a_parse_error_not_a_wrong_page_map() -> None:
    """A drifted sidecar still validates and still resolves — to the WRONG pages."""
    text, sidecar = _assembly()
    parsed = next(iter(PlaintextParser().parse([_raw(text + " edited", sidecar=sidecar)])))

    assert parsed.disposition is Disposition.PARSE_ERROR
    assert parsed.page_map is None
    assert parsed.error is not None and "stale" in parsed.error


def test_a_malformed_sidecar_is_a_parse_error_and_does_not_abort_the_run() -> None:
    text, sidecar = _assembly()
    stream = [
        _raw(text, sidecar=b"{not json"),
        _raw(text, sidecar=sidecar),
    ]
    out = list(PlaintextParser().parse(stream))

    assert [record.disposition for record in out] == [
        Disposition.PARSE_ERROR,
        Disposition.PARSED,
    ]
    # The bad document did not take the good one down with it.
    assert out[1].page_map is not None and len(out[1].page_map.pages) == 3


def test_default_behaviour_is_unchanged_for_a_plain_text_corpus() -> None:
    """No sidecar anywhere ⇒ identical content, offsets and page numbers to pre-ADR-0060."""
    parsed = next(iter(PlaintextParser().parse([_raw(b"hello world")])))

    assert parsed.content == "hello world"
    assert parsed.provenance.char_start == 0
    assert parsed.provenance.char_end == len("hello world")
    assert parsed.provenance.page_start == 1
    assert parsed.provenance.page_end == 1
