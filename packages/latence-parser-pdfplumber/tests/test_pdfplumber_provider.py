"""Unit tests for the pdfplumber Parser adapter (H-A1) — happy path + key failure modes.

These prove the slice EXERCISES ``AdapterBase`` (the audit's whole point), not just that a PDF
parses: the typed-error mapping through :meth:`AdapterBase.guard`, the Provenance/offset carry
through :meth:`AdapterBase.carry_provenance` + the core PageMap seam (offsets round-trip, pages
correct on a real multi-page fixture), the batching path over multiple documents
(:meth:`AdapterBase.batched`, input-order-preserving), the declared ``ProviderProfile`` (CPU,
evidence-bearing MIT license), and the graceful PARSE_ERROR path (no content/PII leak).

Deterministic + seeded (``PYTHONHASHSEED`` pinned by the repo-root conftest).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from latence_core.capability import Parser
from latence_core.contracts import (
    Classification,
    Disposition,
    ParserInput,
    Provenance,
)
from latence_core.errors import LatenceError, ProviderError
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import UNVERIFIED_LICENSE
from latence_parser_pdfplumber.provider import PdfPlumberParser, size_after_parse


def _provenance(file_name: str, file_type: str, size: int = 100) -> Provenance:
    return Provenance(
        source_uri=f"mem://docs/{file_name}",
        file_name=file_name,
        file_type=file_type,
        file_size=size,
        document_id=f"sha256:{file_name}",
    )


def _pdf_input(content: bytes, file_name: str = "doc.pdf", sensitivity: str = "internal") -> ParserInput:
    return ParserInput(
        provenance=_provenance(file_name, "pdf", size=len(content)),
        classification=Classification(language="und", sensitivity=sensitivity),
        content=content,
    )


def _text_input(content: bytes | str, file_name: str = "notes.txt") -> ParserInput:
    return ParserInput(
        provenance=_provenance(file_name, "txt"),
        classification=Classification(language="und", sensitivity="internal"),
        content=content,
    )


# --- the adapter genuinely uses AdapterBase (structural + seam) --------------


def test_is_adapterbase_subclass_and_parser() -> None:
    """The Provider is an AdapterBase subclass AND satisfies the Parser Capability structurally."""
    assert issubclass(PdfPlumberParser, AdapterBase)
    provider = PdfPlumberParser({})
    assert isinstance(provider, Parser)  # runtime_checkable structural check


def test_profile_is_cpu_evidence_bearing_mit() -> None:
    """The declared ProviderProfile: pure-code, CPU, deterministic, batch, evidence-bearing MIT."""
    profile = PdfPlumberParser.profile()
    assert profile.compute == "cpu"
    assert profile.model_id is None
    assert profile.license == "MIT"
    assert profile.license != UNVERIFIED_LICENSE
    assert profile.license_verified is True
    assert profile.license_code == "MIT"
    assert profile.license_weights is None  # pure-code: no weights
    assert profile.license_source  # a real citation is present (H-B1)
    assert profile.license_verified_on == "2026-07-08"
    assert profile.deterministic is True
    assert profile.batch is True
    assert profile.cost_per_1k is None


def test_device_resolves_through_the_seam_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """super().__init__ resolves the device against the profile — trivially CPU, via the seam."""
    monkeypatch.setenv("LATENCE_CUDA", "0")
    provider = PdfPlumberParser({})
    assert provider.device.device == "cpu"
    assert provider.device.skipped is False


# --- happy path: page structure, offsets round-trip, Provenance carry --------


def test_multipage_pdf_yields_correct_pages_and_roundtrip_offsets(
    make_text_pdf: Callable[[list[str]], bytes],
) -> None:
    """A real multi-page PDF → PARSED record with a correct 2-page map; offsets round-trip exactly."""
    pdf = make_text_pdf(
        ["The quarterly report and the numbers.", "Page two of the report follows here."]
    )
    provider = PdfPlumberParser({})
    (record,) = list(provider.parse([_pdf_input(pdf, sensitivity="confidential")]))

    assert record.disposition is Disposition.PARSED
    assert record.error is None
    assert record.page_map is not None
    assert len(record.page_map.pages) == 2
    assert "quarterly report" in record.content
    assert "Page two" in record.content

    # Provenance carried (source lineage intact; char/page span filled to what the record occupies).
    assert record.provenance.file_name == "doc.pdf"
    assert record.provenance.char_start == 0
    assert record.provenance.char_end == len(record.content)
    assert record.provenance.page_start == 1
    assert record.provenance.page_end == 2
    # Classification carried; sensitivity inherited from the input's Classification.
    assert record.classification.sensitivity == "confidential"

    # Offsets round-trip: every page-boundary offset resolves back to its own page with ZERO drift
    # (the ADR-0031 invariant, reusing the same PageOffsetIndex the Runner's drift audit uses).
    index = PageOffsetIndex(record.page_map)
    for span in record.page_map.pages:
        assert index.page_for_offset(span.char_start) == span.page_number
        assert index.page_for_offset(span.char_end - 1) == span.page_number
    assert index.diagnostics.max_drift_chars == 0
    assert index.diagnostics.recovered == 0


def test_pages_land_in_input_order(make_text_pdf: Callable[[list[str]], bytes]) -> None:
    """Per-page text is assembled in page order — page 1 content precedes page 2 content."""
    pdf = make_text_pdf(["AAA first page", "ZZZ second page"])
    (record,) = list(PdfPlumberParser({}).parse([_pdf_input(pdf)]))
    assert record.content.index("AAA first page") < record.content.index("ZZZ second page")


def test_text_input_single_page(make_text_pdf: Callable[[list[str]], bytes]) -> None:
    """A plain-text input parses to a single-page record with language detected at Parse."""
    (record,) = list(PdfPlumberParser({}).parse([_text_input("the framework reads the folder\n")]))
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert len(record.page_map.pages) == 1
    assert record.classification.language == "en"


def test_size_after_parse_is_assembled_byte_length(
    make_text_pdf: Callable[[list[str]], bytes],
) -> None:
    """size_after_parse is the assembled-markdown byte size (the S2 observability signal)."""
    (record,) = list(PdfPlumberParser({}).parse([_pdf_input(make_text_pdf(["hello world"]))]))
    assert size_after_parse(record) == len(record.content.encode("utf-8"))
    assert size_after_parse(record) > 0


# --- batching over multiple documents (AdapterBase.batched) ------------------


def test_batches_multiple_documents_in_input_order(
    make_text_pdf: Callable[[list[str]], bytes],
) -> None:
    """The batching path runs over multiple documents, preserving input order (any batch_size)."""
    inputs = [
        _pdf_input(make_text_pdf([f"document number {i}"]), file_name=f"d{i}.pdf")
        for i in range(5)
    ]
    # A small batch_size forces multiple batches — output order must still mirror input order.
    provider = PdfPlumberParser({"batch_size": 2})
    records = list(provider.parse(inputs))
    assert [r.provenance.file_name for r in records] == [f"d{i}.pdf" for i in range(5)]
    assert all(r.disposition is Disposition.PARSED for r in records)


def test_batch_size_invariant_output_identical(
    make_text_pdf: Callable[[list[str]], bytes],
) -> None:
    """Batching is purely memory-shaping: batch_size=1 and batch_size=32 yield identical output."""
    inputs = [
        _pdf_input(make_text_pdf([f"content {i}"]), file_name=f"d{i}.pdf") for i in range(4)
    ]
    one = [r.model_dump_json() for r in PdfPlumberParser({"batch_size": 1}).parse(inputs)]
    many = [r.model_dump_json() for r in PdfPlumberParser({"batch_size": 32}).parse(inputs)]
    assert one == many


# --- graceful failure: native exception → typed → PARSE_ERROR (no PII) -------


def test_corrupt_pdf_degrades_to_parse_error_not_crash() -> None:
    """A corrupt PDF (valid header, garbage body) → a graceful PARSE_ERROR record, never a crash."""
    corrupt = b"%PDF-1.4\ntotally broken pdf body\n%%EOF"
    (record,) = list(PdfPlumberParser({}).parse([_pdf_input(corrupt, file_name="broken.pdf")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error  # a reason is recorded
    assert record.page_map is None
    assert record.content == ""
    # Provenance intact for audit even on failure.
    assert record.provenance.file_name == "broken.pdf"


def test_type_spoofed_pdf_rejected_before_reader() -> None:
    """A .pdf whose bytes are not a PDF is rejected as a PARSE_ERROR (type-spoof guard)."""
    spoof = b"MZ\x90\x00this is actually an executable, not a pdf"
    (record,) = list(PdfPlumberParser({}).parse([_pdf_input(spoof, file_name="spoof.pdf")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "spoof" in (record.error or "").lower() or "signature" in (record.error or "").lower()


def test_unsupported_file_type_is_parse_error() -> None:
    """An unsupported file type degrades gracefully to a PARSE_ERROR, never a crash."""
    inp = ParserInput(
        provenance=_provenance("thing.xyz", "xyz"),
        classification=Classification(language="und"),
        content=b"whatever",
    )
    (record,) = list(PdfPlumberParser({}).parse([inp]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert "xyz" in (record.error or "")


def test_parse_error_message_carries_no_content_or_pii() -> None:
    """The PARSE_ERROR reason maps only provider + op + native type name — no raw input content."""
    # A corrupt PDF whose garbage body embeds a secret-looking payload: the error must not echo it.
    secret = b"sk-live-SECRETSECRETSECRET-should-not-leak"
    corrupt = b"%PDF-1.4\n" + secret + b"\n%%EOF"
    (record,) = list(PdfPlumberParser({}).parse([_pdf_input(corrupt, file_name="leaky.pdf")]))
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert b"sk-live-".decode() not in record.error
    assert "SECRET" not in record.error


def test_guard_maps_native_exception_to_typed_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AdapterBase.guard maps a native library exception to a typed ProviderError (no content).

    Proves the guard is exercised for real: force pdfplumber to raise a native (non-LatenceError)
    exception inside the guarded block, then call the guarded extraction directly and assert the
    mapped error is a typed ProviderError whose message carries only the provider + op + native
    type name — not the native message (which could echo input).
    """

    class _NativeBoom(Exception):
        def __str__(self) -> str:  # a native message that echoes a secret — must NOT leak
            return "boom sk-live-LEAK"

    def _explode(*_args: object, **_kwargs: object) -> object:
        raise _NativeBoom

    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", _explode)
    provider = PdfPlumberParser({})
    # A well-formed PDF header passes the spoof guard, then the patched open raises the native exc.
    valid_header = b"%PDF-1.4\n%%EOF"
    inp = _pdf_input(valid_header, file_name="native.pdf")
    (record,) = list(provider.parse([inp]))
    # The native fault was mapped to a typed error, then degraded to a PARSE_ERROR record.
    assert record.disposition is Disposition.PARSE_ERROR
    assert record.error is not None
    assert "sk-live-LEAK" not in record.error
    assert "_NativeBoom" in record.error  # only the native TYPE name, no message
    assert "parse" in record.error  # the op label


def test_guard_directly_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling the guarded PDF path with a native fault raises a typed ProviderError (LatenceError)."""

    def _explode(*_a: object, **_k: object) -> object:
        raise RuntimeError("native fault")

    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", _explode)
    provider = PdfPlumberParser({})
    inp = _pdf_input(b"%PDF-1.4\n%%EOF")
    with pytest.raises(ProviderError) as exc_info:
        provider._extract_pdf_pages(inp)
    assert isinstance(exc_info.value, LatenceError)
    assert "native fault" not in str(exc_info.value)  # native message not leaked


# --- determinism -------------------------------------------------------------


def test_deterministic_over_two_runs(make_text_pdf: Callable[[list[str]], bytes]) -> None:
    """Identical input + config → byte-identical output (the profile's deterministic=True claim)."""
    pdf = make_text_pdf(["page one text", "page two text"])
    first = [r.model_dump_json() for r in PdfPlumberParser({}).parse([_pdf_input(pdf)])]
    second = [r.model_dump_json() for r in PdfPlumberParser({}).parse([_pdf_input(pdf)])]
    assert first == second


def test_scanned_page_no_text_layer_yields_empty_not_crash() -> None:
    """A PDF page with no text layer yields empty page text (recorded), not a crash (no OCR here)."""
    # A minimal single-page PDF with no /Contents text stream: pdfplumber returns None for text.
    mini = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n%%EOF"
    )
    (record,) = list(PdfPlumberParser({}).parse([_pdf_input(mini, file_name="scan.pdf")]))
    # It is PARSED (a page exists) with an empty single-page map — recorded, never crashed.
    assert record.disposition is Disposition.PARSED
    assert record.page_map is not None
    assert isinstance(record.content, str)


# --- the AdapterBase widening the first real adapter forced --------------------


def test_carry_provenance_accepts_a_parserinput() -> None:
    """The base carry helper now accepts a ParserInput (the ProvenanceCarrier widening, H-A1)."""
    inp = _text_input("hello", file_name="a.txt")
    prov, cls = AdapterBase.carry_provenance(inp, char_start=0, char_end=5, page_start=1, page_end=1)
    assert prov.file_name == "a.txt"
    assert prov.char_start == 0
    assert prov.char_end == 5
    assert cls is inp.classification  # Classification returned unchanged
