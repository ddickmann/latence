"""The induced-label seam: UNION a Stage's config labels with a record's induced labels (W4).

The OPTIONAL ``schema_induction`` Stage (a :class:`~latence_core.capability.LabelInducer` Provider)
annotates each record with an :class:`~latence_core.contracts.InducedLabels` carrier — the entity /
relation / PII **types** it induced for the document. The four extraction Providers
(``entity.gliner``, ``relation.gliner_relex``, ``relation.llm``, ``redaction.gliner_pii``) gain ONE
additive, backward-compatible line that computes their **effective** label set:

    effective = config_labels ∪ record.induced_labels.<slot>   (when the record carries induced
                labels) else config_labels

so config labels are the **mandatory floor** ("presets") and induced labels **extend** them —
deduped, order-stable. A Stage with NO config labels + induced present becomes fully unsupervised;
with config labels it is supervised-plus-augmented. The extraction LOGIC is untouched — only where
the per-record label list comes from.

This is a pure function over the two lists — no import of a Provider, no model, no I/O — so every
Provider package leans on the SAME seam (never a forked, weaker per-package union) and a record with
no induced labels yields *exactly* its config labels back (the backward-compat proof). The union is
**order-stable**: config labels first (in their given order), then any induced label not already
present (in its given order). Dedup is **case-insensitive** (on ``casefold()``, W12 defense-in-
depth): the inducing Provider emits its labels in a canonical casefolded normal form, but config
labels keep their given surface form, so a config ``Bank`` and an induced ``bank`` collapse to one
(``Bank``, the config surface form kept) rather than prompting the extractor with both.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, DocumentRecord, InducedLabels


class LabelSlot(StrEnum):
    """Which slot of an :class:`~latence_core.contracts.InducedLabels` an extractor reads.

    An Entity extractor unions the induced ``entity_types``; a Relation extractor the
    ``relation_types``; a Redaction (PII) extractor the ``pii_types`` — named so a caller picks its
    slot explicitly rather than passing a raw attribute name.
    """

    ENTITY = "entity_types"
    RELATION = "relation_types"
    PII = "pii_types"


def induced_for_slot(induced: InducedLabels | None, slot: LabelSlot) -> list[str]:
    """The induced labels for ``slot`` on ``induced``, or ``[]`` when there are none.

    ``None`` induced labels (the default on every record until a ``schema_induction`` Stage runs)
    yields an empty list, so :func:`effective_labels` returns the config labels unchanged — the
    backward-compatible path.
    """
    if induced is None:
        return []
    value = getattr(induced, str(slot))
    return list(value) if isinstance(value, list) else []


def effective_labels(
    config_labels: list[str], induced: InducedLabels | None, slot: LabelSlot
) -> list[str]:
    """``config_labels ∪ induced[slot]`` — config floor first, then new induced, deduped, ordered.

    The order-stable union an extractor prompts with: every config label in its given order, then
    each induced label of ``slot`` not already present, in its given order. Dedup is
    CASE-INSENSITIVE (on ``casefold()``, W12): a config ``Bank`` and an induced ``bank`` collapse to
    one entry (the config surface form is kept, since config labels come first). A record with no
    induced labels (``induced is None`` or an empty slot) returns the config labels **unchanged**
    (same list contents + order) — so an existing stack with no ``schema_induction`` Stage is
    byte-identical (a config list is only ever deduped against itself, case-insensitively).
    """
    seen: set[str] = set()
    out: list[str] = []
    for label in config_labels:
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            out.append(label)
    for label in induced_for_slot(induced, slot):
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            out.append(label)
    return out


def induced_labels_by_document(
    chunks: Iterable[ChunkRecord],
) -> dict[str, InducedLabels]:
    """Map each ``document_record_id`` to the :class:`InducedLabels` its chunks carry (W4).

    The ``schema_induction`` Stage is chunk→chunk: it broadcasts a document's induced schema to
    **every** chunk of that document (doc granularity) or induces per chunk (chunk granularity), so
    a chunk's ``document_record_id`` is the DocumentRecord id its induced schema belongs to. This is
    the chunk→document bridge the per-document Redaction Stage needs: Redaction reads a
    :class:`DocumentRecord`, not chunks, so the runner projects the chunk-carried induced labels
    onto the DocumentRecords (:func:`project_induced_onto_documents`) before Redaction reads them.

    The **first** chunk with a populated ``induced_labels`` wins for a given document (doc
    granularity makes them identical; chunk granularity may differ, and first-seen is the stable,
    reproducible choice — the same discipline ``relation.llm`` uses). A chunk whose induction failed
    (fail-open, ``induced_labels is None``) contributes nothing, so a document with NO successfully
    induced chunk is simply absent from the map and its DocumentRecord keeps ``induced_labels =
    None`` (Redaction falls back to its config PII floor — the byte-identical backward-compat path).
    """
    out: dict[str, InducedLabels] = {}
    for chunk in chunks:
        if chunk.document_record_id not in out and chunk.induced_labels is not None:
            out[chunk.document_record_id] = chunk.induced_labels
    return out


def project_induced_onto_documents(
    documents: Iterable[DocumentRecord], chunks: Iterable[ChunkRecord]
) -> list[DocumentRecord]:
    """Stamp each DocumentRecord with its chunks' induced labels (the chunk→document bridge, W4).

    Builds the ``document_record_id → InducedLabels`` map from ``chunks``
    (:func:`induced_labels_by_document`) and returns the documents with ``induced_labels`` set on
    each one that has an induced schema. A document with no induced chunk (induction absent or
    fail-open) is returned **unchanged** — ``induced_labels`` stays ``None`` and Redaction uses its
    config PII floor exactly as before, so a stack with no ``schema_induction`` Stage is
    byte-identical. Order-preserving. A document that ALREADY carries ``induced_labels`` (e.g. a
    doc-level inducer stamped it directly) is left untouched.
    """
    by_doc = induced_labels_by_document(chunks)
    out: list[DocumentRecord] = []
    for document in documents:
        induced = by_doc.get(document.record_id)
        if induced is None or document.induced_labels is not None:
            out.append(document)
        else:
            out.append(document.model_copy(update={"induced_labels": induced}))
    return out
