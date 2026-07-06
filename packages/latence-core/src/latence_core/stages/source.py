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

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.storage import Storage

# Text file types S1 knows how to read. Anything else under the folder is skipped.
_TEXT_EXTENSIONS = {"txt", "md", "markdown"}

# Tiny stop-word signatures for a deliberately naive language auto-detect. The
# glossary permits Source's language detection to be naive in S1; this is honest
# about being a heuristic rather than pretending to be a model.
_LANGUAGE_HINTS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "and", "of", "to", "is", "in", "that", "it"}),
    "de": frozenset({"der", "die", "und", "das", "ist", "nicht", "mit", "ein"}),
    "fr": frozenset({"le", "la", "et", "les", "des", "un", "une", "est"}),
    "es": frozenset({"el", "la", "y", "los", "las", "un", "una", "es"}),
}


def _detect_language(text: str) -> str:
    tokens = [t for t in text.lower().split() if t.isalpha()]
    if not tokens:
        return "und"
    token_set = set(tokens)
    best_lang = "und"
    best_score = 0
    for lang, hints in _LANGUAGE_HINTS.items():
        score = len(token_set & hints)
        if score > best_score:
            best_score, best_lang = score, lang
    return best_lang


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

    def produce(self, storage: Storage) -> Iterator[ParserInput]:
        for uri in storage.walk_files(self._path, follow_symlinks=self._follow_symlinks):
            file_name = uri.rsplit("/", 1)[-1]
            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            if ext not in _TEXT_EXTENSIONS:
                continue
            data = storage.read_bytes(uri)
            document_id = _document_id(data)
            provenance = Provenance(
                source_uri=uri,
                file_name=file_name,
                file_type=ext,
                file_size=len(data),
                document_id=document_id,
            )
            # Naive language detect is Source's own concern; it decodes a *view*
            # of the bytes for the heuristic but hands the RAW bytes downstream —
            # authoritative decoding-to-markdown is the Parser's job (ADR-0019).
            classification = Classification(
                language=_detect_language(data.decode("utf-8", errors="replace")),
                category=self._category,
                sensitivity=self._sensitivity,
            )
            yield ParserInput(
                provenance=provenance,
                classification=classification,
                content=data,
            )
