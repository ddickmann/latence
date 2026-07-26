"""ADR-0033's own required accumulator audit, actually run (and pinned).

ADR-0033 §"Blast radius / risk" lists as a guard: *"Accumulators must be confirmed to hold only
bounded state (audit Disambiguation/Graph for any hidden full-corpus list)."* That audit was never
performed. It is performed here, and the answer is **no** for two of the accumulators:

* ``CascadeDisambiguator.disambiguate`` does ``mention_list = list(mentions)`` **plus** a
  ``by_id`` dict over the same records;
* ``CascadeTypeConsolidator.consolidate`` does the same for mentions **and** relations.

Neither is a defect — a canonical entity cannot be elected from a prefix of the mentions, and a
canonical *type* vocabulary cannot be elected from a prefix of the labels. What was a defect is
the ADR claiming a bound the code does not hold while nothing tested either claim. These tests pin
the **real** contract so it cannot drift back into folklore:

1. the retained state of those two accumulators is ``O(mentions)`` — they consume the WHOLE stream
   before emitting anything (asserted directly, not measured, so it cannot flake);
2. they consume each input stream **exactly once**, which is the property the Runner's streamed
   read-back actually depends on: ``_stream_inputs_of_type`` hands each carrier a fresh one-shot
   generator, and a provider that iterated one of them twice would silently see an empty second
   pass and drop half the corpus.

ADR-0033's "Honest memory boundary" section was corrected alongside these tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from latence_core.contracts import (
    Classification,
    EntityMention,
    Provenance,
    RelationMention,
)
from latence_core.stages.disambiguation import CascadeDisambiguator
from latence_core.stages.type_consolidation import CascadeTypeConsolidator


def _mention(mid: str, doc: str, text: str, label: str = "ORG") -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///c/{doc}.txt",
        file_name=f"{doc}.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc}",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=mid,
        document_record_id=doc,
        chunk_record_id=f"{doc}#chunk-0",
        index=0,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
    )


def _relation(rid: str, head: EntityMention, tail: EntityMention) -> RelationMention:
    return RelationMention(
        record_id=rid,
        document_record_id=head.document_record_id,
        index=0,
        label="works_at",
        head_mention_id=head.record_id,
        tail_mention_id=tail.record_id,
        head_text=head.text,
        tail_text=tail.text,
        confidence=0.8,
        provenance=head.provenance,
        classification=Classification(language="en", sensitivity="public"),
    )


def _corpus(n: int) -> list[EntityMention]:
    """``n`` mentions over only two distinct surfaces — so the KG state is O(2), the input O(n)."""
    return [
        _mention(f"m{i}", f"d{i}", "Acme" if i % 2 == 0 else "Globex") for i in range(n)
    ]


class _OneShot:
    """An iterable that yields once and raises if anything tries to iterate it a second time.

    Mirrors what the Runner actually hands an accumulator: ``_stream_inputs_of_type`` returns a
    generator reading a checkpoint. A generator silently yields NOTHING on a second pass, which is
    the worst failure mode there is — half the corpus vanishes with no error. This makes that
    failure loud.
    """

    def __init__(self, records: list[object]) -> None:
        self._records = records
        self.pulled: list[object] = []
        self._passes = 0

    def __iter__(self) -> Iterator[object]:
        self._passes += 1
        if self._passes > 1:
            msg = (
                "the input stream was iterated more than once; the Runner hands accumulators "
                "one-shot generators, so a second pass silently reads an EMPTY corpus"
            )
            raise AssertionError(msg)
        for record in self._records:
            self.pulled.append(record)
            yield record


def test_disambiguation_retains_the_whole_mention_set() -> None:
    """The audited truth: Disambiguation's retained state is O(mentions), NOT O(entities).

    It pulls every mention off the stream before it can emit its first record — a canonical entity
    cannot be elected from a prefix. Two distinct surfaces over 200 mentions means the *output* is
    bounded by the KG (2 entities), but the *input it holds* is not.
    """
    mentions = _corpus(200)
    stream = _OneShot(list(mentions))
    provider = CascadeDisambiguator({})

    records = provider.disambiguate(stream, iter(()), ())  # type: ignore[arg-type]
    first = next(iter(records), None)

    assert first is not None
    assert len(stream.pulled) == len(mentions), (
        "Disambiguation emitted a record before consuming the whole mention stream. If it has "
        "genuinely become incremental that is an improvement — but ADR-0033's 'Honest memory "
        "boundary' section now states the bound is O(mentions) for this accumulator, so update "
        "the ADR in the same change rather than letting doc and code disagree again."
    )


def test_type_consolidation_retains_the_whole_mention_set() -> None:
    """Same audited truth for Type Consolidation (ADR-0054): O(mentions), not O(canonical types)."""
    mentions = _corpus(200)
    stream = _OneShot(list(mentions))
    provider = CascadeTypeConsolidator({})

    consolidated = provider.consolidate(stream, iter(()))  # type: ignore[arg-type]

    assert len(stream.pulled) == len(mentions)
    assert len(consolidated.mentions) == len(mentions)  # 1:1 relabel — the bound IS the input


@pytest.mark.parametrize("consolidate_relations", [True, False])
def test_type_consolidation_consumes_each_stream_exactly_once(
    consolidate_relations: bool,
) -> None:
    """Both carriers are single-pass — the property the Runner's streamed read-back depends on.

    ``_consolidate_relations`` picks between the relabel branch and the pass-through branch, and
    the pass-through branch reads the relation stream from a *different* place in the method. A
    branch that touched it twice would drop every relation on the floor without an error, so both
    branches are exercised.
    """
    mentions = _corpus(20)
    relations = [_relation("r0", mentions[0], mentions[1])]
    provider = CascadeTypeConsolidator({"consolidate_relations": consolidate_relations})

    consolidated = provider.consolidate(
        _OneShot(list(mentions)),  # type: ignore[arg-type]
        _OneShot(list(relations)),  # type: ignore[arg-type]
    )

    assert len(consolidated.mentions) == len(mentions)
    assert len(consolidated.relations) == len(relations)


def test_disambiguation_consumes_each_stream_exactly_once() -> None:
    """Mentions, relations and chunks are each read once — one-shot generators stay safe."""
    mentions = _corpus(20)
    relations = [_relation("r0", mentions[0], mentions[1])]
    provider = CascadeDisambiguator({})

    records = list(
        provider.disambiguate(
            _OneShot(list(mentions)),  # type: ignore[arg-type]
            _OneShot(list(relations)),  # type: ignore[arg-type]
            _OneShot([]),  # type: ignore[arg-type]
        )
    )
    assert records
