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

**T2 (per-chunk extraction).** With per-chunk induction, unioning a global config floor onto every
chunk re-inflates exactly the label count the redesign shrank. So the union is now a POLICY the
caller states, :class:`LabelFloor`: ``UNIVERSAL`` (the default — config labels are a non-subsettable
SAFETY floor: a Redactor's PII floor, the universal guard floor) or ``PER_CHUNK`` (config labels are
a RELEVANCE floor: a recall net used only when the chunk induced nothing, so an inducing chunk is
scored against its OWN types alone). Relevance labels are scoped per chunk; safety floors are
universal — that asymmetry is the whole rule, and it lives here rather than in each Provider.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, DocumentRecord, InducedLabels


class LabelSlot(StrEnum):
    """Which slot of an :class:`~latence_core.contracts.InducedLabels` an extractor reads.

    An Entity extractor unions the induced ``entity_types``; a Relation extractor the
    ``relation_types``; a Redaction (PII) extractor the ``pii_types``; a guardrails-capable
    Redaction Provider the ``guard_types`` — named so a caller picks its slot explicitly rather
    than passing a raw attribute name.
    """

    ENTITY = "entity_types"
    RELATION = "relation_types"
    PII = "pii_types"
    GUARD = "guard_types"


class LabelFloor(StrEnum):
    """Whether a Stage's CONFIG labels are a UNIVERSAL safety floor or a PER-CHUNK relevance floor.

    This is the one decision the per-chunk extraction redesign turns on, and it is deliberately
    NOT a global setting: the two kinds of label are governed by opposite rules.

    * ``UNIVERSAL`` — the config labels are a **safety floor**. They are applied to EVERY record no
      matter what that record induced, because they encode a security invariant rather than a
      relevance judgement: a leaked card number in a chunk "about" shipping logistics must still be
      masked, and a prompt injection hidden in an "irrelevant" chunk must still be flagged. A
      Redaction Provider's PII floor (ADR-0044) and the universal guard floor
      (:data:`~latence_core.redaction_policy.GUARD_SAFETY_FLOOR`) are UNIVERSAL, and a Provider
      that owns one must not expose a knob that subsets it. This is the historical behaviour of
      :func:`effective_labels` and stays the DEFAULT, so every existing caller is unchanged.

    * ``PER_CHUNK`` — the config labels are a **relevance floor**: a recall safety net for the case
      where induction produced nothing for this record. When the record DID induce labels for the
      slot, the record's own induced set is the whole effective set and the config floor is not
      unioned on. This is the per-chunk extraction path: zero-shot NER/RE precision degrades as the
      label count grows, so scoring a chunk against types it has no business containing is a pure
      tax. The trade is explicit — ``UNIVERSAL`` buys recall insurance at the cost of label-count
      purity; ``PER_CHUNK`` buys precision (and a far smaller span-scoring tensor) at the cost of
      that insurance. For ENTITY/RELATION labels, which are relevance labels by nature, precision
      is the SOTA choice; for PII/guardrail labels it would be a security regression.

    Under ``PER_CHUNK`` a record that induced NOTHING for the slot still falls back to the config
    labels: an extractor prompted with an EMPTY label set would silently extract nothing, which is
    a worse failure than an over-broad one. That fallback is also what keeps every stack WITHOUT a
    ``schema_induction`` Stage byte-identical under either policy.
    """

    UNIVERSAL = "universal"
    PER_CHUNK = "per_chunk"


#: The config key every Provider exposes the :class:`LabelFloor` policy under, so an operator
#: writes the same knob whichever extraction Provider a Pipeline runs.
LABEL_FLOOR_CONFIG_KEY = "label_floor"


def resolve_label_floor(config: dict[str, Any], *, default: LabelFloor) -> LabelFloor:
    """Read the ``label_floor`` knob out of a Provider config → a :class:`LabelFloor`.

    Absent ⇒ ``default`` (each Provider states its own, because the right default differs: a
    relevance-label extractor defaults ``PER_CHUNK``, a safety-floor Redactor is pinned
    ``UNIVERSAL``). Any value that is not one of the enum's is a ``ValueError`` naming the accepted
    values — a typo must be a loud operator error, never a silently ignored key that leaves the
    stack running the other policy (ADR-0034).
    """
    if LABEL_FLOOR_CONFIG_KEY not in config:
        return default
    raw = config[LABEL_FLOOR_CONFIG_KEY]
    for candidate in LabelFloor:
        if raw == candidate.value:
            return candidate
    accepted = ", ".join(repr(candidate.value) for candidate in LabelFloor)
    msg = f"{LABEL_FLOOR_CONFIG_KEY} must be one of {accepted} (got {raw!r})"
    raise ValueError(msg)


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


def _dedup_ordered(labels: Iterable[str]) -> list[str]:
    """``labels`` with case-insensitive duplicates removed, first surface form + order kept."""
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        key = label.casefold()
        if key not in seen:
            seen.add(key)
            out.append(label)
    return out


def effective_labels(
    config_labels: list[str],
    induced: InducedLabels | None,
    slot: LabelSlot,
    *,
    floor: LabelFloor = LabelFloor.UNIVERSAL,
) -> list[str]:
    """The label set an extractor prompts this record with, deduped and order-stable.

    ``floor`` decides what the config labels MEAN (see :class:`LabelFloor`), and it is the only
    knob:

    * ``LabelFloor.UNIVERSAL`` (the default, unchanged behaviour) — ``config_labels ∪
      induced[slot]``: every config label in its given order, then each induced label of ``slot``
      not already present, in its given order. The config labels are a SAFETY floor applied to
      every record.
    * ``LabelFloor.PER_CHUNK`` — the record's OWN ``induced[slot]`` when it induced any, else the
      config labels. The config labels are a RELEVANCE floor: a recall net for a record induction
      said nothing about, not a set stamped onto every record.

    Dedup is CASE-INSENSITIVE (on ``casefold()``, W12): under ``UNIVERSAL`` a config ``Bank`` and an
    induced ``bank`` collapse to one entry and the CONFIG surface form is kept (config labels come
    first); under ``PER_CHUNK`` the induced surface form is kept, since the induced set is the whole
    set. A record with no induced labels (``induced is None`` or an empty slot) returns the config
    labels **unchanged** (same contents + order) under EITHER policy — so an existing stack with no
    ``schema_induction`` Stage is byte-identical whichever floor a Provider chooses (a config list
    is only ever deduped against itself, case-insensitively).
    """
    induced_slot = induced_for_slot(induced, slot)
    if floor is LabelFloor.PER_CHUNK and induced_slot:
        # The chunk's own set IS the effective set — no global floor re-inflates it. Empty-induced
        # falls through to the config labels below (never an empty prompt: see LabelFloor).
        return _dedup_ordered(induced_slot)
    return _dedup_ordered([*config_labels, *induced_slot])


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
