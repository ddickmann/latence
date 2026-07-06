"""PlaintextParser: decodes raw ParserInput bytes into a markdown DocumentRecord.

Provenance/Classification are inherited losslessly; the character span is filled
in from the decoded text; decoding (bytes -> markdown) is the Parser's job now
(ADR-0019), so the seam matches a real bytes-consuming parser.
"""

from __future__ import annotations

from latence_core.contracts import Classification, ParserInput, Provenance
from latence_parser_plaintext import PlaintextParser


def _raw(content: bytes | str = b"hello world", config: dict[str, str] | None = None) -> ParserInput:
    return ParserInput(
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=len(content),
            document_id="sha256:x",
        ),
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        config=config or {},
    )


def test_decodes_bytes_and_preserves_contract() -> None:
    raw = _raw(b"hello world")
    out = list(PlaintextParser().parse([raw]))
    assert len(out) == 1
    parsed = out[0]
    # Classification carried through losslessly.
    assert parsed.classification == raw.classification
    # The record id is the source document id.
    assert parsed.record_id == raw.provenance.document_id
    # Raw bytes decoded to text; media type restated as markdown.
    assert parsed.content == "hello world"
    assert parsed.media_type == "text/markdown"
    # The character span is filled in from the decoded length; the rest of
    # Provenance is inherited unchanged.
    assert parsed.provenance.char_start == 0
    assert parsed.provenance.char_end == len("hello world")
    assert parsed.provenance.document_id == raw.provenance.document_id


def test_accepts_already_decoded_str_content() -> None:
    out = list(PlaintextParser().parse([_raw("already text")]))
    assert out[0].content == "already text"


def test_per_input_encoding_hint_overrides_default() -> None:
    # latin-1 'café' is not valid utf-8; the per-input hint must win.
    raw = _raw("café".encode("latin-1"), config={"encoding": "latin-1"})
    out = list(PlaintextParser().parse([raw]))
    assert out[0].content == "café"


def test_empty_input() -> None:
    assert list(PlaintextParser().parse([])) == []
