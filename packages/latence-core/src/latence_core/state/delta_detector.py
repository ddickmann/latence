"""Delta Detector — the ported content-hash manifest diff (PORT-AS-IS, S11).

Ported faithfully from ``latenceai-dataset-intelligence/state/delta_detector.py``
(ASSET-INVENTORY.md, 5/5, PORT-AS-IS). The algorithmic value is preserved verbatim: a
**deterministic SHA-256** over ``(doc_id, markdown, sorted entity texts, sorted relation
labels)`` produces one content hash per document, and a **manifest diff** against the last
committed manifest classifies every document as *added* / *updated* (same identity, new
hash) / *deleted* / *unchanged* — the four :class:`~latence_core.contracts.DeltaKind`s.

This is a pure engine over contract-free views (:class:`ManifestEntry`), mirroring the
:mod:`latence_core.disambiguation` engine's IO-independence: the delta orchestration builds
the manifest from the run's records and hands it here; the detector never touches Storage or
a Pydantic contract. Hashing the *derived* content (entity texts + relation labels), not just
the raw markdown, is the ported subtlety: a document whose text is byte-identical but whose
extraction changed (a new label set, a Provider swap) is correctly flagged *updated*, so the
corpus is not left stale after a re-extraction.

Determinism: the hash inputs are sorted before hashing and the diff iterates sorted keys, so
the same corpus state always yields the same manifest and the same delta (the Baseline bar).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ManifestEntry:
    """One document's content-addressed manifest row (contract-free view).

    ``document_id`` is the stable source identity (the content-addressed id Source stamps);
    ``content_hash`` is the deterministic SHA-256 over the document's markdown plus its
    derived entity texts and relation labels — so two entries with the same ``document_id``
    but different ``content_hash`` are an *update*, not a coincidence. ``file_name`` is carried
    for the audit/report (never part of the identity or the hash).
    """

    document_id: str
    content_hash: str
    file_name: str = ""


@dataclass
class ManifestDiff:
    """The classified diff of a new manifest against the previous one (the ported result).

    Each list holds the ``document_id``s in that :class:`~latence_core.contracts.DeltaKind`
    class. ``updated`` are same-identity, new-hash documents; ``unchanged`` are same-identity,
    same-hash. Every list is sorted for determinism.
    """

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[str]:
        """The documents a delta must re-process (added + updated), sorted."""
        return sorted(self.added + self.updated)

    @property
    def is_empty(self) -> bool:
        """True if nothing changed (no add/update/delete) — an idempotent re-run."""
        return not (self.added or self.updated or self.deleted)


def content_hash(
    document_id: str,
    markdown: str,
    entity_texts: list[str] | None = None,
    relation_labels: list[str] | None = None,
) -> str:
    """Deterministic SHA-256 over ``(doc_id, markdown, sorted entity texts, sorted labels)``.

    The ported hash: the raw markdown plus the *derived* content (entity surface texts and
    relation labels), so a re-extraction that changes the derived records — even over
    byte-identical text — flips the hash and the document is flagged *updated*. Inputs are
    sorted and length-prefixed so the concatenation is unambiguous (two different splits can
    never collide) and order-independent (a re-ordered mention set hashes the same).
    """
    hasher = hashlib.sha256()

    def _feed(part: str) -> None:
        raw = part.encode("utf-8")
        hasher.update(str(len(raw)).encode("ascii"))
        hasher.update(b":")
        hasher.update(raw)

    _feed(document_id)
    _feed(markdown)
    for text in sorted(entity_texts or []):
        _feed(text)
    hasher.update(b"|rel|")
    for label in sorted(relation_labels or []):
        _feed(label)
    return "sha256:" + hasher.hexdigest()


class DeltaDetector:
    """The ported manifest-diff engine: classify each document add/update/delete/unchanged.

    Stateless and pure: :meth:`diff` takes the previous committed manifest and the current
    manifest (both keyed by ``document_id``) and returns the classified
    :class:`ManifestDiff`. The current manifest is built by the orchestration from the run's
    records via :func:`content_hash`; the previous one is read from the last committed Corpus
    Version.
    """

    @staticmethod
    def diff(
        previous: Mapping[str, ManifestEntry],
        current: Mapping[str, ManifestEntry],
    ) -> ManifestDiff:
        """Classify every document by comparing ``current`` against ``previous`` (ported)."""
        result = ManifestDiff()
        for doc_id in sorted(current):
            if doc_id not in previous:
                result.added.append(doc_id)
            elif current[doc_id].content_hash != previous[doc_id].content_hash:
                result.updated.append(doc_id)
            else:
                result.unchanged.append(doc_id)
        for doc_id in sorted(previous):
            if doc_id not in current:
                result.deleted.append(doc_id)
        return result
