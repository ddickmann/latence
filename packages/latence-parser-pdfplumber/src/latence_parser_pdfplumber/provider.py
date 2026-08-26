"""``parser.pdfplumber`` — the first REAL Wave-1 adapter built THROUGH ``AdapterBase`` (H-A1).

The Phase-3 provider seam — :class:`~latence_core.providers.adapter.AdapterBase` (P3-F2) +
:class:`~latence_core.providers.profile.ProviderProfile` (P3-F1) — was validated only by tests
that fabricate fake subclasses; it had **zero real production consumers**. This Provider is the
first real one: a CPU, page-aware PDF (and plain-text) Parser over the
`pdfplumber <https://github.com/jsvine/pdfplumber>`_ library (MIT), implemented as an
``AdapterBase`` **subclass** — turning the hypothetical seam into a real two-adapter seam (with
the fabricated-subclass tests) and, more importantly, into a real *second-real-consumer* seam
alongside the in-core reference Providers. It is the **template** every subsequent Wave-1 adapter
(docling, GLiNER variants, embedders, PII) copies, so the ergonomics are the point.

Library choice + fallback (H-A1 "verify early"): the issue **preferred** ``kreuzberg``, but an
early install-and-run verification (recorded in the package README's seam-report) found it
unsuitable on both counts the issue names — its current major (4.x) is **Elastic-2.0**, a
non-permissive source-available license (ADR-0012 requires permissive-or-opt-in), and its last
permissive (MIT) 3.x line **fails to import without OCR system binaries** (the "requires system
packages" failure mode). So, exactly as the issue's fallback clause directs, this substitutes a
**permissive, pip-installable PDF lib** — ``pdfplumber`` (MIT), which is page-aware (so the
PageMap/offset path is genuinely exercised, not a single blob), CPU-only, and deterministic on the
text layer. It needs no OS-level system binaries (no pandoc/tesseract to apt-install, unlike
kreuzberg), though its wheel closure does include native code — ``pillow`` (C-ext) and
``pypdfium2`` (bundled PDFium), both permissive and latent (the used path is pdfminer.six-only);
see the ``profile()`` license note. The package is named for the library it actually uses.

What it EXERCISES on ``AdapterBase`` (the audit's whole point — use the facilities, don't bypass):

* **native-exception → typed taxonomy** (:meth:`AdapterBase.guard`): a corrupt / password-
  protected / type-spoofed PDF makes pdfminer raise its native ``PdfminerException``; the guard
  maps it to a typed :class:`~latence_core.errors.ProviderError` carrying only the provider + op +
  native *type name* — **no raw content / PII**. That typed error is then caught at the
  per-document boundary and turned into a graceful ``PARSE_ERROR`` :class:`DocumentRecord` (ADR-
  0020), so one bad file never aborts the run (the Parse C2 graceful-failure the conformance suite
  asserts). The guard is used for its real purpose (map the native fault to the typed category);
  the PARSE_ERROR disposition is Parse's own graceful-degradation contract layered on top.
* **offset / Provenance + page preservation** (:meth:`AdapterBase.carry_provenance` +
  :func:`PageMap.from_page_texts`): the per-page text pdfplumber yields is assembled with the
  existing core PageMap seam (boundaries exact by construction), and the produced record's
  Provenance is carried from the input via the base helper (source lineage intact, only the
  char/page span narrowed) — **not** a hand-rolled weaker mapper (the audit found the in-core
  GLiNER provider did that; this does not repeat it). Offsets round-trip; pages are correct.
* **device + batching** (:meth:`AdapterBase.batched`): a CPU parser declares ``compute="cpu"``;
  :meth:`parse` runs the batching path over multiple input documents (input-order-preserving,
  one batch at a time — streaming-friendly, ADR-0033). Device routing is trivially CPU here but
  flows through the seam (``self.device`` is resolved in ``super().__init__``).

OCR is a Provider *technique*, never the Capability name (CONTEXT glossary): this Provider reads
the PDF **text layer** only — a scanned page with no text layer yields empty page text (recorded,
not crashed), and an OCR-first Provider is a drop-in second Parser behind the same seam. That
keeps the install lean and CPU-viable (no tesseract / no vision stack — ADR-0016).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
)
from latence_core.errors import LatenceError
from latence_core.language import detect_language
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ParserInput

# File types this Provider decodes as page-structured PDFs (via pdfplumber's page iterator).
_PDF_EXTENSIONS = frozenset({"pdf"})
# File types decoded as a single-page plain-text/markdown document (no PDF reader involved).
_TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "text"})

# PDF magic bytes: a real PDF starts with "%PDF-" at byte 0. A type-spoof guard (a ``.pdf`` whose
# bytes are not a PDF) is defense-in-depth ahead of the reader — matching the in-core
# ``parser.document`` Provider and the intake Screener (issue #62). We require the signature at
# offset 0 (allowing only a tiny, bounded leading BOM/whitespace window), not a substring anywhere
# in the first KB (which a spoofed file could smuggle to slip past this guard).
_PDF_MAGIC = b"%PDF-"
_PDF_LEADING_SLACK = 8
_PDF_LEADING_SKIP = b"\x00\r\n\t \xef\xbb\xbf\xff\xfe"


class PdfPlumberParser(AdapterBase):
    """Parser Provider over pdfplumber: PDF (per-page) + text (single-page) → markdown records.

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass (H-A1): it declares a
    :class:`~latence_core.providers.profile.ProviderProfile`, resolves its device through the base
    (``super().__init__``), batches over multiple input documents via :meth:`AdapterBase.batched`,
    maps the native pdfminer exception through :meth:`AdapterBase.guard`, and carries Provenance
    via :meth:`AdapterBase.carry_provenance`. It satisfies the
    :class:`~latence_core.capability.Parser` Protocol structurally (it implements ``parse``) — the
    base adds no protocol, only the reusable glue.

    Config:
        encoding: str — text-decode encoding for text inputs (default 'utf-8'). A per-input
            ``config['encoding']`` on the ParserInput overrides this default.
        category: str | None — content category stamped onto Classification, overriding any
            category inherited from the ParserInput.
        sensitivity: str | None — sensitivity stamped onto Classification. If unset, the
            ParserInput's inherited sensitivity is preserved.
        page_separator: str — markdown joined between PDF pages (default '\\n\\n').
        batch_size: int — how many input documents the batching path groups per call (default
            32, validated ``>= 1``). Purely memory-shaping — outputs are input-order identical
            for any positive batch size (ADR-0033).
        device: str — 'auto'|'cpu'|'cuda' (resolved by the base; trivially CPU here).

    ``config_exceptions`` is left empty (the base default): pdfplumber/pdfminer raise no
    "operator handed a bad config" native exception in this Provider's code path — an invalid
    ``encoding`` hint is caught explicitly below as a per-document parse failure, and a corrupt
    PDF is a runtime :class:`~latence_core.errors.ProviderError`, not a ConfigError. Declaring
    nothing here means the guard maps every native fault to ProviderError, which is correct.
    """

    #: No native exception in this Provider's path means "bad operator config" — see the class
    #: docstring. Empty (the base default) so every native fault maps to ProviderError, not
    #: ConfigError; nothing is silently mis-classified.
    config_exceptions: ClassVar[tuple[type[Exception], ...]] = ()

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, permissive (MIT).

        Ships no model weights (``model_id=None``): text-layer extraction is deterministic.
        This Provider's own code is the repo's Apache-2.0; its direct dep
        ``pdfplumber`` is **MIT** (``license_code="MIT"``), verified per ADR-0012 from the installed
        version's PyPI metadata (License classifier ``OSI Approved :: MIT License``) + the upstream
        GitHub LICENSE. **Honest dependency footprint (NOT pure-Python):** pdfplumber's install
        closure pulls native code — ``pillow`` (C-extension image decoder, HPND/MIT-CMU) and
        ``pypdfium2`` (bundles Google's PDFium C++, Apache-2.0/BSD-3) — and ``pdfminer.six`` (MIT)
        pulls ``cryptography`` (compiled, Apache-2.0/BSD-3) for encrypted PDFs. **All permissive**
        (no ADR-0012 violation), and this Provider's used path calls only ``page.extract_text()``
        (stays inside pdfminer.six), so pillow/pypdfium2 are latent, not on the reachable code path
        — but the native footprint + its CVE surface (pillow especially) is real and is an accepted
        risk gated by the IntakeScreener (which quarantines malformed/oversized/spoofed input before
        Parse). CPU-only (``compute="cpu"``). ``batch=True`` — it batches over multiple input
        *documents* (:meth:`AdapterBase.batched`), input-order-preserving and streaming-friendly. No
        per-call cost (a local library, not an endpoint).

        A subtlety H-A1 asks the seam-report to answer honestly: ``memory_mb`` and ``cost_per_1k``
        are **inert** for this adapter — ``cost_per_1k=None`` (no endpoint) and nothing consumes
        ``memory_mb`` (the device seam only routes on ``compute``; there is no memory-gated skip).
        They are declared truthfully (a small honest ``memory_mb=96`` for the pdfplumber page
        buffers; ``cost_per_1k=None``) but nothing reads them here — recorded in the README's
        trim-vs-keep note.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=96,
            model_id=None,
            license="MIT",
            license_verified=True,
            license_code="MIT",
            license_weights=None,
            # Cited where a human actually checked the license for the installed version:
            # the pdfplumber PyPI metadata (License classifier) + the upstream GitHub LICENSE.
            license_source="https://github.com/jsvine/pdfplumber/blob/stable/LICENSE.txt",
            license_verified_on="2026-07-08",
            deterministic=True,
            batch=True,
            cost_per_1k=None,
        )

    @property
    def provider_name(self) -> str:
        """The registered entry-point name — the stable tag a mapped error carries (no content)."""
        return "parser.pdfplumber"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Resolve the device against the declared profile via the base (trivially CPU here, but
        # the routing flows through the seam — H-A1 "device routing must flow through the seam").
        super().__init__(config)
        cfg = self.config
        self._encoding = str(cfg.get("encoding", "utf-8"))
        self._category = cfg.get("category")
        self._sensitivity = cfg.get("sensitivity")
        self._page_separator = str(cfg.get("page_separator", "\n\n"))

    # -- Parser Capability method (structural; the base adds no protocol) ---

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        """Consume raw :class:`ParserInput`s and yield parsed markdown records, in input order.

        Runs the base's batching path (:meth:`AdapterBase.batched`) over multiple input documents
        so the seam's streaming-friendly, input-order-preserving batching is genuinely exercised
        (H-A1 "exercise AdapterBase's batching path over multiple input documents"). Each batch is
        parsed one document at a time; ordering is preserved by construction.
        """
        yield from self.batched(list(inputs), self._parse_batch)

    def _parse_batch(self, batch: list[ParserInput]) -> list[DocumentRecord]:
        """Parse one batch of raw inputs → one :class:`DocumentRecord` each, in order."""
        return [self._parse_one(raw) for raw in batch]

    # -- per-document parse -------------------------------------------------

    def _parse_one(self, raw: ParserInput) -> DocumentRecord:
        """Parse a single raw input into a PARSED (or, on failure, PARSE_ERROR) record.

        The native pdfminer exception a corrupt/encrypted/spoofed PDF raises is mapped by
        :meth:`AdapterBase.guard` to a typed :class:`~latence_core.errors.ProviderError` (no
        content/PII), which is then caught here and turned into a graceful ``PARSE_ERROR`` record
        (ADR-0020) — so one bad document never aborts the run. A framework
        :class:`~latence_core.errors.LatenceError` (the guarded fault, or an explicit parse
        failure) is the only thing caught: a genuinely unexpected bug still propagates untyped.
        """
        file_type = raw.provenance.file_type.lower()
        try:
            if file_type in _PDF_EXTENSIONS:
                page_texts = self._extract_pdf_pages(raw)
            elif file_type in _TEXT_EXTENSIONS:
                page_texts = [self._decode_text(raw)]
            else:
                # Unsupported type is a graceful per-document failure, not a crash (ADR-0020).
                msg = f"Unsupported file type {file_type!r} for the pdfplumber Parser."
                raise _ParseFailure(msg)
        except LatenceError as exc:
            # Both the guarded native-fault (ProviderError) and an explicit _ParseFailure
            # (a LatenceError) land here → a graceful PARSE_ERROR record. The message is the
            # framework-typed one (provider + op + native type name), carrying no content/PII.
            return self._error_record(raw, str(exc))

        assembly = PageMap.from_page_texts(page_texts, separator=self._page_separator)
        text = assembly.text
        n_pages = len(assembly.page_map.pages)

        # Carry Provenance via the base helper (source lineage intact; only the char/page span
        # narrowed to what this record occupies) — never a hand-rolled mapper (H-A1). The base
        # helper accepts any ProvenanceCarrier, and a ParserInput is one (ADR-0019), so the raw
        # input is passed directly — the misfit that forced widening carry_provenance is now gone.
        provenance, _ = self.carry_provenance(
            raw,
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=n_pages,
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
        """A ``PARSE_ERROR`` record: Provenance kept, no content, typed reason (ADR-0020)."""
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

    # -- format decoders ----------------------------------------------------

    def _decode_text(self, raw: ParserInput) -> str:
        """Decode an already-textual input (str passthrough, or bytes via the encoding hint)."""
        if isinstance(raw.content, str):
            return raw.content
        encoding = str(raw.config.get("encoding", self._encoding))
        try:
            return raw.content.decode(encoding, errors="replace")
        except LookupError as exc:  # unknown codec name in the hint → graceful parse failure
            msg = f"Unknown text encoding {encoding!r}."
            raise _ParseFailure(msg) from exc

    def _extract_pdf_pages(self, raw: ParserInput) -> list[str]:
        """Extract per-page text from a PDF via pdfplumber; one string per page, in order.

        Type-spoof guard first (a ``.pdf`` whose bytes are not a PDF is rejected as a parse
        failure, defense-in-depth). Then pdfplumber (MIT, CPU) opens the bytes and
        extracts text page by page — the per-page split is what makes the page map real, not a
        single blob with a guessed boundary. The native pdfminer exception a corrupt/encrypted PDF
        raises is caught by :meth:`AdapterBase.guard` (op ``"parse"``) and re-raised as a typed
        :class:`~latence_core.errors.ProviderError` carrying no content — :meth:`_parse_one` then
        degrades it to a PARSE_ERROR record.
        """
        data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
        offset = 0
        limit = min(len(data), _PDF_LEADING_SLACK)
        while offset < limit and data[offset] in _PDF_LEADING_SKIP:
            offset += 1
        if data[offset : offset + len(_PDF_MAGIC)] != _PDF_MAGIC:
            msg = (
                "File declared as PDF but the %PDF- signature is not at the start of the file "
                "(type spoof?)."
            )
            raise _ParseFailure(msg)

        # Import inside the method so a missing/malformed dependency never breaks Provider
        # discovery, and so ``latence-core`` stays free of the pdfplumber dep (ADR-0016 — this is
        # a plugin package). A missing dep is a graceful parse failure, not an import-time crash.
        try:
            import io

            import pdfplumber
        except ImportError as exc:  # pragma: no cover - the dep is declared for this package
            msg = "pdfplumber is required to parse PDFs."
            raise _ParseFailure(msg) from exc

        # The native call under the base guard: a corrupt/encrypted/malformed PDF makes pdfminer
        # raise its native exception, which the guard maps to a typed ProviderError (provider + op
        # + native type name only — NO raw content / PII in the message).
        with self.guard("parse"):
            pages: list[str] = []
            with pdfplumber.open(io.BytesIO(data)) as doc:
                for page in doc.pages:
                    # A page with no text layer (scanned image) yields "" — recorded, not crashed.
                    # OCR is a separate Provider technique (CONTEXT), not this Provider's job.
                    pages.append(page.extract_text() or "")
        return pages or [""]

    # -- classification -----------------------------------------------------

    def _classify(self, raw: ParserInput, text: str) -> Classification:
        """Attach Classification at Parse: language from the decoded text (shared core heuristic).

        Language is auto-detected from the *decoded* text via the shared core heuristic (Parse
        sees the real content). Category and sensitivity are the config override, else the value
        inherited from the ParserInput's Classification, so the Source's stamp is not silently
        dropped — matching the in-core reference Parser exactly.
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


class _ParseFailure(LatenceError):
    """A recoverable, framework-typed parse failure → a PARSE_ERROR record (ADR-0020).

    A :class:`~latence_core.errors.LatenceError` subclass so it flows through the same
    typed-error boundary the guarded native fault does: :meth:`PdfPlumberParser._parse_one`
    catches ``LatenceError`` and turns either into a graceful PARSE_ERROR record. It carries only
    a static reason (unsupported type, spoofed signature, unknown encoding) — never input content.
    """

    category = "provider"
