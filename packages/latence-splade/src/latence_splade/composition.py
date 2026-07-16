"""The ADR-0053 SPLADE composition — pure, dependency-free term-weight algebra.

This module holds the *policy* of ADR-0053: a chunk's sparse vector is not plain ``SPLADE(text)``
but the **structured composition**

    sparse = model_weight · SPLADE(text)  ⊕  entity_boost · confidence-gated entity_terms
                                          ⊕  metadata_boost · selected metadata_terms

where ⊕ is a term-id union that **sums** colliding weights. It is deliberately free of any model,
tokenizer, torch, or I/O: it operates only on already-resolved ``{vocab_id: weight}`` maps and
lists of vocab ids, so the whole composition is unit-testable with plain numbers and is identical
whether it runs at index time (over a chunk) or at query time (over a query string) — the ADR-0053
one-composition-two-clocks guarantee.

The two ADR-0053 modes differ only in ``model_weight``: **term-injection** (the default) keeps the
model's own weights untouched (``model_weight == 1.0``) and adds the boosted structured terms on
top; **weighted** scales the model leg by a tunable ``model_weight`` so an operator can dial the
model's contribution up or down against the structured legs. Both are the same summed union — one
honest formula, not two code paths.

The **confidence gate** is applied here (an entity below the extraction-confidence threshold is
dropped so weak NER never pollutes the sparse signal). The **redaction gate** — dropping an entity
whose surface did not survive the corpus's redaction, so a masked PII surface is never re-injected
— lives in the Provider (:mod:`~latence_splade.provider`), because it needs the redacted text; this
module only sees the entities the Provider chose to pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntityTerm:
    """One NER entity mention offered to the composition: its surface string + extraction
    confidence.

    ``surface`` is the exact surface form the pipeline's :class:`EntityMention` carried (ADR-0053
    injects *surface forms*, not canonical ids, so the injected terms land in the model's own
    subword vocabulary); ``confidence`` is the extractor's 0..1 score the confidence gate reads.
    """

    surface: str
    confidence: float


@dataclass(frozen=True)
class SpladeInput:
    """One index-time item to compose: the (redacted) chunk text + its structured signals.

    The deep input the composition Provider consumes per record. ``text`` is the corpus text the
    model encodes (already redacted per the corpus policy — the Provider never reads raw
    ``content``); ``entities`` are the chunk's confidence-bearing :class:`EntityTerm`s;
    ``metadata`` is the chunk's selected ``{field: value}`` map (classification, provenance) from
    which the operator-chosen fields are injected. A bare ``SpladeInput(text)`` (no entities, no
    metadata) composes to exactly the model leg — the shape ``encode_sparse`` builds for the
    text-only Export seam.
    """

    text: str
    entities: Sequence[EntityTerm] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)


def compose(
    *,
    model_terms: Mapping[int, float],
    entity_terms: Sequence[tuple[Sequence[int], float]],
    metadata_terms: Sequence[Sequence[int]],
    confidence_threshold: float,
    entity_boost: float,
    metadata_boost: float,
    model_weight: float,
) -> dict[int, float]:
    """Union the model leg with confidence-gated entity + metadata legs into one term-weight map.

    All three legs are expressed in the **same** vocabulary-id space (the model's), so the union is
    a
    plain per-id summation — the ADR-0053 ⊕. The result is a raw ``{vocab_id: weight}`` map
    (unsorted, possibly containing an incidental zero if boosts cancel a negative model weight);
    the caller materialises it into a canonical :class:`~latence_core.capability.SparseVector` via
    ``SparseVector.from_terms`` (which sorts ascending and drops zeros — the single place ordering
    and zero-pruning are imposed).

    Args:
        model_terms: the SPLADE model's ``{vocab_id: weight}`` map for the text.
        entity_terms: one ``(vocab_ids, confidence)`` per candidate entity — the entity surface's
            subword vocab ids and the extractor confidence the gate reads.
        metadata_terms: one ``vocab_ids`` list per selected metadata value.
        confidence_threshold: an entity with ``confidence < threshold`` is dropped (weak NER never
            pollutes the signal — ADR-0053). ``0.0`` admits every entity.
        entity_boost / metadata_boost: the weight added per injected entity/metadata vocab id.
        model_weight: the multiplier on the model leg — ``1.0`` for term-injection (default),
        tunable in weighted mode.

    Returns: The composed ``{vocab_id: weight}`` map (summed union).
    """
    terms: dict[int, float] = {}
    for idx, weight in model_terms.items():
        terms[idx] = terms.get(idx, 0.0) + model_weight * weight
    for ids, confidence in entity_terms:
        if confidence < confidence_threshold:
            continue
        for idx in ids:
            terms[idx] = terms.get(idx, 0.0) + entity_boost
    for ids in metadata_terms:
        for idx in ids:
            terms[idx] = terms.get(idx, 0.0) + metadata_boost
    return terms


__all__ = ["EntityTerm", "SpladeInput", "compose"]
