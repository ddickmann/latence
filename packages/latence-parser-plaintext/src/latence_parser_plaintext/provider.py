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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, DocumentRecord, PageMap
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

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
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        A byte-decoding plain-text Parser ships no model weights (``model_id=None``) and is
        framework code (Apache-2.0, verified — weights-and-code trivially the same code). The
        decode is deterministic, CPU-only, does not batch (a per-document transform), and has
        no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._encoding = str(self._config.get("encoding", "utf-8"))

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
            # A plain-text document is a single source page: one PageMap span
            # covering the whole decoded text, so any downstream offset resolves to
            # page 1 (Provenance integrity, S2). Assembling text and map together
            # keeps the char span exact by construction.
            assembly = PageMap.from_page_texts([text])
            # Fill in the character span now that we know the decoded length;
            # Provenance is otherwise inherited losslessly from the ParserInput.
            provenance = raw.provenance.model_copy(
                update={
                    "char_start": 0,
                    "char_end": assembly.page_map.total_chars,
                    "page_start": 1,
                    "page_end": 1,
                }
            )
            yield DocumentRecord(
                record_id=raw.provenance.document_id,
                provenance=provenance,
                classification=raw.classification,
                content=assembly.text,
                media_type="text/markdown",
                page_map=assembly.page_map,
            )

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
