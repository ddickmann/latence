"""CPU reference document Parser Provider — PDF + text → markdown + page map.

The S2 Parse reference Provider (ADR-0004, ADR-0007 CPU-first, ADR-0012 permissive
weights/deps). Consumes a raw :class:`ParserInput` (undecoded source bytes plus
Provenance/Classification, ADR-0019) and produces a markdown
:class:`DocumentRecord` that carries:

* a **page map** — real per-page boundaries for PDFs (extracted page by page),
  a single span for plain text — so any downstream character offset resolves back
  to its original source page via the ported ``PageOffsetIndex`` (Provenance
  integrity, S2);
* a first-class **Classification** attached *at Parse*: language auto-detected
  from the decoded text (the shared core heuristic — not forked), a content
  category (from config, else inherited from the ParserInput), and a sensitivity
  default (inherited, else config, else ``"unknown"``);
* complete **Provenance**: the ParserInput's source lineage plus the character
  span (0..len) and page span (1..n) filled in once decoded. The per-record
  assembled-markdown byte size — the Provenance 'size-after-parse', exposed by
  the :func:`size_after_parse` helper — is summed by the Runner into the Quality
  Report's ``ParseQuality.bytes_after_parse`` (S2 observability).

A corrupt or unsupported/spoofed file does **not** crash the run: the Parser
catches the failure and emits a ``PARSE_ERROR`` :class:`DocumentRecord` carrying
its Provenance and the failure reason, so the run continues and the failure is
auditable (S2 AC: "handles a corrupt/unparseable file gracefully").

OCR is a Provider *technique*, never the Capability name (CONTEXT glossary): this
reference does not OCR scanned images — a scanned PDF page with no text layer
yields empty page text (recorded, not crashed), and an image-first Provider is a
drop-in second Parser behind the same seam. That keeps the shipped default lean
and CPU-viable (pypdf is pure-Python, BSD-3-Clause) rather than pulling a heavy
OCR/vision stack into the default install (ADR-0016).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
)
from latence_core.language import detect_language
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ParserInput

# File types this reference Provider decodes as page-structured PDFs.
_PDF_EXTENSIONS = frozenset({"pdf"})
# File types decoded as a single-page plain-text/markdown document.
_TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "text"})

# PDF magic bytes: a real PDF starts with "%PDF-" at byte 0. pypdf itself reads the
# header at offset 0, so we require it there — allowing only a tiny, bounded leading
# BOM/whitespace window — rather than accepting a ``%PDF-`` substring anywhere in the
# first 1 KB (which a spoofed file could smuggle to slip past this guard; see the
# intake Screener's matching hardening, issue #62). Used to catch a type-spoofed file
# (a ``.pdf`` that is not a PDF) before handing it to the PDF reader.
_PDF_MAGIC = b"%PDF-"
# Bounded leading window of BOM / ASCII whitespace tolerated before the header.
_PDF_LEADING_SLACK = 8
_PDF_LEADING_SKIP = b"\x00\r\n\t \xef\xbb\xbf\xff\xfe"


class DocumentParser:
    """Parser Provider: PDF (per-page) and text (single-page) → markdown records.

    Config:
        encoding: str  — text-decode encoding (default 'utf-8'). A per-input
            ``config['encoding']`` on the ParserInput overrides this default.
        category: str | None  — content category stamped onto Classification,
            overriding any category inherited from the ParserInput.
        sensitivity: str | None  — sensitivity stamped onto Classification. If
            unset, the ParserInput's inherited sensitivity is preserved.
        page_separator: str  — markdown joined between PDF pages (default '\\n\\n').
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, permissive.

        Ships no model weights (``model_id=None``): decoding is deterministic pure-Python
        (this Provider is Apache-2.0 framework code; its one dep ``pypdf`` is
        pure-Python, BSD-3-Clause — both weights-and-code permissive, verified per ADR-0012). CPU-
        only, does not batch (a per-document transform), and has no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=64,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._encoding = str(config.get("encoding", "utf-8"))
        self._category = config.get("category")
        self._sensitivity = config.get("sensitivity")
        self._page_separator = str(config.get("page_separator", "\n\n"))

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        for raw in inputs:
            yield self._parse_one(raw)

    # -- per-document parse --------------------------------------------------

    def _parse_one(self, raw: ParserInput) -> DocumentRecord:
        file_type = raw.provenance.file_type.lower()
        try:
            if file_type in _PDF_EXTENSIONS:
                page_texts = self._extract_pdf_pages(raw)
            elif file_type in _TEXT_EXTENSIONS:
                page_texts = [self._decode_text(raw)]
            else:
                msg = f"Unsupported file type {file_type!r} for the document Parser."
                raise _UnsupportedType(msg)
        except _ParseFailure as exc:
            # Graceful degradation: record the failure, do not crash the run.
            return self._error_record(raw, str(exc))

        assembly = PageMap.from_page_texts(page_texts, separator=self._page_separator)
        text = assembly.text
        n_pages = len(assembly.page_map.pages)

        provenance = raw.provenance.model_copy(
            update={
                "char_start": 0,
                "char_end": len(text),
                "page_start": 1,
                "page_end": n_pages,
            }
        )
        classification = self._classify(raw, text)
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=provenance,
            classification=classification,
            content=text,
            media_type="text/markdown",
            page_map=assembly.page_map,
            disposition=Disposition.PARSED,
        )

    def _error_record(self, raw: ParserInput, reason: str) -> DocumentRecord:
        """A ``PARSE_ERROR`` record: Provenance kept, no content, reason recorded."""
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=raw.provenance,
            classification=self._classify(raw, ""),
            content="",
            media_type="text/markdown",
            page_map=None,
            disposition=Disposition.PARSE_ERROR,
            error=reason,
        )

    # -- format decoders -----------------------------------------------------

    def _decode_text(self, raw: ParserInput) -> str:
        if isinstance(raw.content, str):
            return raw.content
        encoding = str(raw.config.get("encoding", self._encoding))
        try:
            return raw.content.decode(encoding, errors="replace")
        except LookupError as exc:  # unknown codec name in the hint
            msg = f"Unknown text encoding {encoding!r}: {exc}"
            raise _ParseFailure(msg) from exc

    def _extract_pdf_pages(self, raw: ParserInput) -> list[str]:
        """Extract per-page text from a PDF; one string per page, in order.

        Type-spoof guard first: a ``.pdf`` whose bytes are not a PDF is rejected
        as a parse failure rather than fed to the reader. Then pypdf (pure-Python,
        BSD-3-Clause, CPU) extracts text page by page — the per-page split is what
        makes the page map real, not a single blob with a guessed boundary. Any
        reader error (encrypted, truncated, malformed) degrades to a parse failure.
        """
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        offset = 0
        limit = min(len(data), _PDF_LEADING_SLACK)
        while offset < limit and data[offset] in _PDF_LEADING_SKIP:
            offset += 1
        if data[offset : offset + len(_PDF_MAGIC)] != _PDF_MAGIC:
            msg = (
                "File declared as PDF but the %PDF- signature is not at the start of the "
                "file (type spoof?)."
            )
            raise _ParseFailure(msg)

        # Import inside the method so a malformed dependency never breaks discovery,
        # and so ``latence-core`` stays free of the pypdf dep (this is a plugin).
        try:
            import io

            from pypdf import PdfReader
            from pypdf.errors import PyPdfError
        except ImportError as exc:  # pragma: no cover - dependency is declared
            msg = f"pypdf is required to parse PDFs: {exc}"
            raise _ParseFailure(msg) from exc

        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                # Try the empty-password path once; many PDFs are "encrypted" with
                # a blank owner password. A real password-protected file fails here.
                try:
                    reader.decrypt("")
                except Exception as exc:  # noqa: BLE001 - normalise to a parse failure
                    msg = f"Encrypted PDF could not be decrypted: {exc}"
                    raise _ParseFailure(msg) from exc
            pages: list[str] = []
            for page in reader.pages:
                # A page with no text layer (scanned image) yields "" — recorded,
                # not crashed. OCR is a separate Provider technique (CONTEXT).
                pages.append(page.extract_text() or "")
        except _ParseFailure:
            raise
        except (PyPdfError, ValueError, OSError, EOFError) as exc:
            msg = f"Corrupt or unreadable PDF: {exc}"
            raise _ParseFailure(msg) from exc
        except Exception as exc:  # noqa: BLE001 - never let a parser bug crash the run
            msg = f"Unexpected error parsing PDF: {type(exc).__name__}: {exc}"
            raise _ParseFailure(msg) from exc

        return pages or [""]

    # -- classification ------------------------------------------------------

    def _classify(self, raw: ParserInput, text: str) -> Classification:
        """Attach Classification at Parse: language from the decoded text.

        Language is auto-detected from the *decoded* text via the shared core
        heuristic (Parse sees the real content, not just a byte view). Category
        and sensitivity are the config override, else the value inherited from the
        ParserInput's Classification, so Source's stamp is not silently dropped.
        """
        inherited = raw.classification
        language = detect_language(text) if text else inherited.language
        category = self._category if self._category is not None else inherited.category
        sensitivity = (
            str(self._sensitivity) if self._sensitivity is not None else inherited.sensitivity
        )
        return Classification(language=language, category=category, sensitivity=sensitivity)


def size_after_parse(record: DocumentRecord) -> int:
    """Assembled-markdown size in bytes — the Provenance 'size-after-parse' (S2 observability).

    A one-line delegation to :attr:`~latence_core.contracts.DocumentRecord.size_after_parse`,
    which is where the rule lives: the size is a property of the record, not of the Parser that
    produced it. Kept as a module-level function because this package publishes the name in its
    ``__all__``; every Parser package delegates to the same implementation, so the signal the
    Runner sums into ``ParseQuality.bytes_after_parse`` cannot drift between Providers.
    """
    return record.size_after_parse


class _ParseFailure(Exception):
    """Internal: a recoverable parse failure, turned into a PARSE_ERROR record."""


class _UnsupportedType(_ParseFailure):
    """Internal: the file type is not one this reference Provider decodes."""
