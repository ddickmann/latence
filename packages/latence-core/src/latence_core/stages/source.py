"""Local-folder Source Provider (ADR-0014, S1 spine).

Lists and reads every ``.txt``/``.md`` file under a folder, stamping each with
initial Provenance (file name/type/size, content-addressed document id) and a
naive-but-honest Classification (language auto-detect), and hands the *raw*
bytes downstream as a :class:`ParserInput` — decoding and the turn into markdown
belong to the Parser (ADR-0019). The entry point that makes siloed data
addressable.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from latence_core.contracts import Classification, ParserInput, Provenance
from latence_core.language import UNKNOWN_LANGUAGE, detect_language

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.storage import Storage

# Text file types whose bytes can be decoded for the naive language hint. For any
# other accepted type (a PDF, an office file) Source hands the raw bytes on with
# language left ``und`` — authoritative Classification is attached at Parse, from
# the *decoded* text, which Source cannot produce for a binary format (ADR-0019).
_TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "text"})

# Default set of file extensions the Source reads. S1's spine is text-only; S2's
# Parse can consume PDFs, so the accepted set is config-widenable (see below)
# without forking a second Source Provider. Anything not in the set is skipped.
_DEFAULT_EXTENSIONS = frozenset({"txt", "md", "markdown"})


def _document_id(data: bytes) -> str:
    """Stable content-addressed id: sha256 keeps Provenance reproducible."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class LocalFolderSource:
    """Source Provider that reads a folder of text documents.

    Config:
        path: str  — Storage URI (or bare path) of the folder to read.
        sensitivity: str  — stamped onto every record's Classification.
        category: str | None  — optional content category.
        follow_symlinks: bool  — whether to follow symlinks out of the source
            folder. Default ``False``: a symlink whose target escapes the source
            root is NOT read, so the Source cannot be tricked into exfiltrating
            files outside the folder it was pointed at (ADR-0014, secure default).
        extensions: list[str] | None  — file extensions (no dot, lowercased) the
            Source reads. Defaults to text/markdown (the S1 spine). Widen it to
            e.g. ``["txt", "md", "pdf"]`` to feed a PDF-capable Parser (S2) with no
            second Source Provider. Binary types are handed on as raw bytes with
            language ``und``; Parse attaches the authoritative Classification.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        path = config.get("path")
        if not path or not isinstance(path, str):
            msg = "LocalFolderSource requires a non-empty string 'path' in its config."
            raise ValueError(msg)
        self._path = path
        self._sensitivity = str(config.get("sensitivity", "unknown"))
        self._category = config.get("category")
        self._follow_symlinks = bool(config.get("follow_symlinks", False))
        raw_exts = config.get("extensions")
        if raw_exts is None:
            self._extensions = _DEFAULT_EXTENSIONS
        else:
            if not isinstance(raw_exts, list) or not all(isinstance(e, str) for e in raw_exts):
                msg = "LocalFolderSource 'extensions' must be a list of strings."
                raise ValueError(msg)
            self._extensions = frozenset(e.lstrip(".").lower() for e in raw_exts)
        # S11: documents to logically exclude from this run by content-addressed id. A
        # Retraction/Purge (ADR-0018) re-runs the pipeline with the deleted documents excluded
        # HERE, at Source, so doc-level Stages simply never see them and the corpus-level Stages
        # re-resolve the survivors — a genuine split-on-delete through the real pipeline, not a
        # bolted-on record edit. Empty (the default) means read everything.
        raw_exclude = config.get("exclude_document_ids")
        if raw_exclude is None:
            self._excluded: frozenset[str] = frozenset()
        else:
            if not isinstance(raw_exclude, list) or not all(
                isinstance(e, str) for e in raw_exclude
            ):
                msg = "LocalFolderSource 'exclude_document_ids' must be a list of strings."
                raise ValueError(msg)
            self._excluded = frozenset(raw_exclude)

    def produce(self, storage: Storage) -> Iterator[ParserInput]:
        for uri in storage.walk_files(self._path, follow_symlinks=self._follow_symlinks):
            file_name = uri.rsplit("/", 1)[-1]
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            if ext not in self._extensions:
                continue
            data = storage.read_bytes(uri)
            document_id = _document_id(data)
            # Logically-deleted (retracted/purged) documents are skipped at Source.
            if document_id in self._excluded:
                continue
            provenance = Provenance(
                source_uri=uri,
                file_name=file_name,
                file_type=ext,
                file_size=len(data),
                document_id=document_id,
            )
            # Naive language detect is Source's own concern, and only meaningful
            # for a text type: it decodes a *view* of the bytes for the heuristic
            # but hands the RAW bytes downstream. For a binary type (PDF/office) it
            # cannot decode text, so it leaves language ``und`` — authoritative
            # decoding-to-markdown and Classification are the Parser's job (ADR-0019).
            if ext in _TEXT_EXTENSIONS:
                language = detect_language(data.decode("utf-8", errors="replace"))
            else:
                language = UNKNOWN_LANGUAGE
            classification = Classification(
                language=language,
                category=self._category,
                sensitivity=self._sensitivity,
            )
            yield ParserInput(
                provenance=provenance,
                classification=classification,
                content=data,
            )
