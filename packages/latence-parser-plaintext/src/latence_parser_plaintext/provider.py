"""Plain-text/markdown Parser Provider.

Consumes a raw :class:`ParserInput` (undecoded source bytes plus Provenance and
Classification) and turns a ``.txt``/``.md`` document into a markdown
:class:`DocumentRecord`. Decoding lives HERE — the Parser owns the bytes→text
turn (ADR-0019) — so this Provider has the same input shape as a real PDF/OCR
parser would. For plain text the transform is a near-identity (text is valid
markdown); Provenance and Classification are inherited losslessly from the
ParserInput produced by Source, with the character span filled in once decoded.
This lives in its OWN package to prove dynamic Provider discovery across package
boundaries (ADR-0004).

**Pages (ADR-0060).** Markdown carries no page structure, so for a long time this Provider
answered every offset→page question with "page 1" — correct for a ``.txt`` note, and a
silent lie for the 3,470 OCR'd documents whose per-page markdown had been assembled
elsewhere and handed back as one blob (measured: ``page_start == page_end == 1`` on all
236k exported chunks). It now reads the page-map sidecar the Source paired with the
document, when there is one, and otherwise says out loud that it *assumed* a single page —
:attr:`~latence_core.contracts.PageMapOrigin.ASSUMED_SINGLE_PAGE`. A corpus that must be
page-accurate sets ``require_page_map`` and gets a ``PARSE_ERROR`` per unpaginated document
instead of a plausible number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    PAGE_MAP_SIDECAR_SUFFIX,
    Disposition,
    DocumentRecord,
    PageMap,
    PageMapOrigin,
)
from latence_core.errors import PageMapSidecarError
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ParserInput


class PlaintextParser:
    """Parser Provider that decodes raw text bytes into a markdown record.

    Config:
        encoding: str  — decode encoding (default 'utf-8'). A per-input
            ``config['encoding']`` on the ParserInput overrides this default. An
            unknown codec name in that (untrusted) hint yields a ``PARSE_ERROR``
            record, never an exception that aborts the run (ADR-0020).
        require_page_map: bool  — whether this corpus is DECLARED paginated
            (default ``False``). Off, a document with no sidecar parses exactly as
            before, as one assumed page. On, it is a ``PARSE_ERROR`` — because
            nothing in an assembled markdown blob distinguishes "a one-page note"
            from "a 40-page report whose page breaks were dropped upstream", only
            the operator knows which corpus this is, and the whole point of
            ADR-0060 is that the wrong answer must be loud rather than plausible.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        PolyForm-Noncommercial-1.0.0.

        A byte-decoding plain-text Parser ships no model weights (``model_id=None``) and is
        framework code (PolyForm-Noncommercial-1.0.0, verified — weights-and-code trivially the same
        code). The decode is deterministic, CPU-only, does not batch (a per-document transform), and
        has no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
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
        self._config = config or {}
        self._encoding = str(self._config.get("encoding", "utf-8"))
        self._require_page_map = bool(self._config.get("require_page_map", False))

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        for raw in inputs:
            # Decode the raw source content into text — this is the Parser's job,
            # not the Source's (ADR-0019). A per-input encoding hint wins over the
            # Provider default; str content is already decoded (passed through).
            encoding = str(raw.config.get("encoding", self._encoding))
            if isinstance(raw.content, str):
                text = raw.content
            else:
                try:
                    text = raw.content.decode(encoding, errors="replace")
                except LookupError as exc:
                    # An unknown codec NAME in the (untrusted) per-input encoding hint
                    # raises LookupError from bytes.decode BEFORE errors="replace" can
                    # act — that only tolerates undecodable BYTES, not a bad codec name.
                    # A Parser must degrade a bad input to a PARSE_ERROR record, never
                    # abort the whole run (ADR-0020, Disposition.PARSE_ERROR); mirror the
                    # document Parser's graceful handling of the same hint (issue #74).
                    yield self._parse_error(raw, f"Unknown text encoding {encoding!r}: {exc}")
                    continue
            try:
                page_map = self._page_map_for(raw, text)
            except PageMapSidecarError as exc:
                # A sidecar that does not match its document is WORSE than none: its spans
                # validate and resolve, just to the wrong pages. Refuse the document rather
                # than emit citations nobody can tell are wrong (ADR-0060).
                yield self._parse_error(raw, str(exc))
                continue
            # Fill in the character span now that we know the decoded length;
            # Provenance is otherwise inherited losslessly from the ParserInput.
            provenance = raw.provenance.model_copy(
                update={
                    "char_start": 0,
                    "char_end": page_map.total_chars,
                    "page_start": page_map.pages[0].page_number,
                    "page_end": page_map.pages[-1].page_number,
                }
            )
            yield DocumentRecord(
                record_id=raw.provenance.document_id,
                provenance=provenance,
                classification=raw.classification,
                content=text,
                media_type="text/markdown",
                page_map=page_map,
            )

    def _page_map_for(self, raw: ParserInput, text: str) -> PageMap:
        """This document's page boundaries: from its sidecar, or an ADMITTED assumption.

        Three outcomes, in the order they are tried:

        * a sidecar the Source paired with the document ⇒ a real multi-page map, cross-checked
          against ``len(text)`` so a stale one raises instead of resolving to wrong pages;
        * no sidecar, ``require_page_map`` set ⇒
          :class:`~latence_core.errors.PageMapSidecarError`, which becomes a ``PARSE_ERROR``
          record: the operator declared this corpus paginated and this document is not;
        * no sidecar, the default ⇒ the historical single span, now labelled
          ``ASSUMED_SINGLE_PAGE``. Byte-identical page resolution to before; the difference is
          that the record no longer *claims* the document has one page.
        """
        sidecar = raw.page_map_sidecar
        if sidecar is not None:
            return PageMap.from_sidecar_json(sidecar, text_length=len(text))
        if self._require_page_map:
            raise PageMapSidecarError(
                f"require_page_map is set but no {PAGE_MAP_SIDECAR_SUFFIX} sidecar was found "
                f"beside {raw.provenance.file_name!r}. Markdown carries no page structure, so "
                f"parsing it anyway would report every offset as page 1 — refusing instead."
            )
        # A page-less blob is ONE span covering everything, so offset→page resolution keeps a
        # single code path — but the map now records that this is a convention, not a
        # measurement, so a consumer that needs true page citation can tell the difference.
        return PageMap.from_page_texts(
            [text], origin=PageMapOrigin.ASSUMED_SINGLE_PAGE
        ).page_map

    @staticmethod
    def _parse_error(raw: ParserInput, reason: str) -> DocumentRecord:
        """A PARSE_ERROR record: Provenance kept, no content/page map, reason recorded.

        Emitted when the input cannot be decoded (an unknown codec hint) so the run
        continues and the failure is auditable in the Quality Report — the downstream
        Chunk stage skips a PARSE_ERROR record, so it never reaches the corpus.
        """
        return DocumentRecord(
            record_id=raw.provenance.document_id,
            provenance=raw.provenance,
            classification=raw.classification,
            content="",
            media_type="text/markdown",
            page_map=None,
            disposition=Disposition.PARSE_ERROR,
            error=reason,
        )
