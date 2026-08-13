"""Direct unit tests for the ADR-0061 §5 fold rule's owned module (``reuse_overlay``).

The fold used to be six hand-synchronised re-derivations inside ``runner.py``, and its merge —
the mechanism that makes an incremental run byte-identical to a rebuild — was covered only by
the heavy e2e delta suites. These tests pin the module directly, without a pipeline run:

* :func:`~latence_core.reuse_overlay.merge_overlay_doc_ordered` — the document-interleaved
  arrival order every ADR-0061-named order-dependence reduces to (the resolver's first-seen
  display spelling, the hyperedge γ-tie-breaks and ``merged_sources`` order, the profiler's
  last-document fields, the export row order): the merged stream must equal, record for record
  and in order, the stream a full rebuild of the whole corpus would deliver.
* the NATURAL-key indexing (commit 4249620's regression, pinned at unit level): the reuse
  store is lexicographically id-sorted, emission order is numeric — ``…#chunk-10`` must fold
  back AFTER ``…#chunk-2``. The e2e churn-equivalence test keeps covering the full path
  (``test_e2e_delta_enrichment_incremental`` runs a >10-chunk document through reuse); this is
  the direct pin on the sort itself.
* :meth:`~latence_core.reuse_overlay.ReuseOverlay.fold_into`'s fold-target decision (first
  DOC-level producer only) and the three named trailer policies — the REAL behavioral
  differences between the Runner's six consumer sites, preserved as
  :class:`~latence_core.reuse_overlay.OverlayTrailer` values.
* the counting twin (:meth:`~latence_core.reuse_overlay.ReuseOverlay.count_for_stage`) and the
  Export/wiring-check faces (``records_for_carrier`` / ``has_carrier``), including the
  mixed-carrier (``Record`` sentinel) expansion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    Provenance,
    Record,
    RelationMention,
)
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.reuse_overlay import (
    OverlayTrailer,
    ReuseOverlay,
    is_doc_level_stage,
    merge_overlay_doc_ordered,
    natural_record_order,
    overlay_carriers,
    stage_produces,
)

# -- fixtures -----------------------------------------------------------------------------------


def _classification() -> Classification:
    return Classification(language="en", sensitivity="public")


def _provenance(doc: str) -> Provenance:
    return Provenance(
        source_uri=f"file:///corpus/{doc}.md",
        file_name=f"{doc}.md",
        file_type="md",
        file_size=100,
        document_id=f"sha256:{doc}",
        char_start=0,
        char_end=10,
        page_start=1,
        page_end=1,
    )


def _chunk(doc: str, index: int) -> ChunkRecord:
    return ChunkRecord(
        record_id=f"sha256:{doc}#chunk-{index}",
        document_record_id=f"sha256:{doc}",
        index=index,
        content=f"{doc} chunk {index}",
        token_count=3,
        provenance=_provenance(doc),
        classification=_classification(),
    )


def _mention(doc: str, index: int, text: str = "Acme") -> EntityMention:
    return EntityMention(
        record_id=f"sha256:{doc}#mention-{index}",
        document_record_id=f"sha256:{doc}",
        chunk_record_id=f"sha256:{doc}#chunk-0",
        index=index,
        label="ORG",
        text=text,
        confidence=0.9,
        provenance=_provenance(doc),
        classification=_classification(),
    )


def _relation(doc: str, index: int) -> RelationMention:
    return RelationMention(
        record_id=f"sha256:{doc}#relation-{index}",
        document_record_id=f"sha256:{doc}",
        index=index,
        label="works_for",
        head_mention_id=f"sha256:{doc}#mention-0",
        tail_mention_id=f"sha256:{doc}#mention-1",
        head_text="Acme",
        tail_text="Globex",
        confidence=0.8,
        provenance=_provenance(doc),
        classification=_classification(),
    )


def _stage(kind: CapabilityKind, name: str, *depends_on: str, **config: Any) -> dict[str, Any]:
    return {
        "name": name,
        "capability": kind.value,
        "provider": f"{kind.value}.fake",
        "depends_on": list(depends_on),
        "config": config,
    }


def _pipeline(*stages: dict[str, Any]) -> Pipeline:
    return Pipeline.model_validate(
        {"name": "p", "storage_uri": "file:///tmp/x", "stages": list(stages)}
    )


#: The reference DAG the fold tests enumerate over: two doc-level ChunkRecord producers
#: (chunk, then redact — chunk→chunk transforms share the carrier), a doc-level mixed-carrier
#: producer (the fused extractor), and a corpus-level mixed-carrier re-emitter (Type
#: Consolidation).
_PIPELINE = _pipeline(
    _stage(CapabilityKind.SOURCE, "source"),
    _stage(CapabilityKind.PARSE, "parse", "source"),
    _stage(CapabilityKind.CHUNK, "chunk", "parse"),
    _stage(CapabilityKind.REDACTION, "redact", "chunk"),
    _stage(CapabilityKind.FUSED_ENTITY_RELATION, "extract", "chunk"),
    _stage(CapabilityKind.TYPE_CONSOLIDATION, "consolidate_types", "extract"),
)
_BY_NAME = {s.name: s for s in _PIPELINE.stages}


def _ids(records: Sequence[Record]) -> list[str]:
    return [r.record_id for r in records]


# -- natural_record_order: the emission-order key (commit 4249620) ------------------------------


def test_natural_order_compares_counters_by_number() -> None:
    """``#chunk-2`` orders before ``#chunk-10`` — the lexicographic store's one lie, undone."""
    ids = [f"sha256:d#chunk-{i}" for i in (0, 2, 10, 11)]
    assert sorted(ids) != ids  # the store's lexicographic order really is wrong past 9 chunks
    assert sorted(ids, key=natural_record_order) == ids


def test_overlay_indexing_reconstructs_emission_order_from_an_id_sorted_store() -> None:
    """ReuseOverlay sorts each carrier by the natural key at construction (read-side fix).

    Fed in the reuse store's canonical LEXICOGRAPHIC id order (``#chunk-10`` before
    ``#chunk-2``), ``of_type`` must return the document's original emission order — the
    regression the ticket-03 real-data smoke caught on a >10-chunk document.
    """
    chunks = [_chunk("d", i) for i in range(12)]
    store_ordered = sorted(chunks, key=lambda c: c.record_id)
    assert [c.index for c in store_ordered] != list(range(12))
    overlay = ReuseOverlay(list(store_ordered))
    assert [c.index for c in overlay.of_type(ChunkRecord)] == list(range(12))


# -- merge_overlay_doc_ordered: the rebuild's arrival order -------------------------------------


def test_empty_overlay_yields_the_fresh_stream_untouched() -> None:
    """The non-delta no-op: same objects, same order, nothing buffered."""
    fresh = [_chunk("a", 0), _chunk("b", 0)]
    assert list(merge_overlay_doc_ordered(iter(fresh), [])) == fresh


def test_merge_reproduces_the_full_rebuild_arrival_order() -> None:
    """Reused documents interleave by (source_uri, document_id) — leading, middle, trailing.

    A rebuild of docs a…e delivers each document's records contiguously in globally-sorted
    URI order. With b/d fresh and a/c/e reused, the merged stream must be byte-order equal to
    that rebuild stream — the property the resolver's first-seen display spelling, the
    hyperedge tie-breaks/``merged_sources`` order, the profiler's last-document fields and
    the export row order all reduce to.
    """
    per_doc = {doc: [_chunk(doc, i) for i in range(2)] for doc in "abcde"}
    rebuild = [record for doc in "abcde" for record in per_doc[doc]]
    fresh = [record for doc in "bd" for record in per_doc[doc]]
    overlay = [record for doc in "ace" for record in per_doc[doc]]
    merged = list(merge_overlay_doc_ordered(iter(fresh), overlay))
    assert _ids(merged) == _ids(rebuild)


def test_merge_with_no_fresh_records_yields_the_whole_overlay_in_doc_order() -> None:
    """The all-reused delta: the stream is exactly the overlay, document-grouped and ordered."""
    overlay = [_chunk("b", 0), _chunk("b", 1), _chunk("a", 0)]
    merged = list(merge_overlay_doc_ordered(iter([]), sorted(overlay, key=lambda c: c.record_id)))
    assert _ids(merged) == _ids([_chunk("a", 0), _chunk("b", 0), _chunk("b", 1)])


def test_merge_keeps_within_document_order_of_both_sides() -> None:
    """Within one document, records stay contiguous and in their given sequence."""
    fresh = [_chunk("b", 0), _chunk("b", 1)]
    overlay = [_chunk("a", 0), _chunk("a", 1), _chunk("a", 2)]
    merged = list(merge_overlay_doc_ordered(iter(fresh), overlay))
    assert _ids(merged) == _ids(overlay) + _ids(fresh)


# -- the fold-target predicates -----------------------------------------------------------------


def test_fold_target_facts_read_off_the_descriptors() -> None:
    """Level + produces are the descriptor's declarations, incl. the mixed-carrier sentinel."""
    facts = {
        "chunk": (
            is_doc_level_stage(_BY_NAME["chunk"]),
            stage_produces(_BY_NAME["chunk"], _PIPELINE, ChunkRecord),
            stage_produces(_BY_NAME["chunk"], _PIPELINE, EntityMention),
        ),
        "extract": (
            is_doc_level_stage(_BY_NAME["extract"]),
            stage_produces(_BY_NAME["extract"], _PIPELINE, EntityMention),
            stage_produces(_BY_NAME["extract"], _PIPELINE, RelationMention),
        ),
        "consolidate_types": (
            is_doc_level_stage(_BY_NAME["consolidate_types"]),
            stage_produces(_BY_NAME["consolidate_types"], _PIPELINE, EntityMention),
            stage_produces(_BY_NAME["consolidate_types"], _PIPELINE, ChunkRecord),
        ),
    }
    assert facts == {
        "chunk": (True, True, False),
        "extract": (True, True, True),
        "consolidate_types": (False, True, False),
    }


def test_overlay_carriers_expands_the_mixed_sentinel_only() -> None:
    """``Record`` delivers both mention sides; a Record carrier itself; raw carriers nothing."""
    assert overlay_carriers(Record) == (EntityMention, RelationMention)
    assert overlay_carriers(ChunkRecord) == (ChunkRecord,)
    assert overlay_carriers(str) == ()


# -- fold_into: first doc-level producer, once — and the three trailer policies -----------------


def test_fold_into_merges_at_the_first_doc_level_producer_only() -> None:
    """Two doc-level ChunkRecord producers: the overlay folds into the first stream, once."""
    overlay = ReuseOverlay([_chunk("a", 0)])
    chunk_fresh, redact_fresh = [_chunk("b", 0)], [_chunk("b", 0)]
    folded = list(
        overlay.fold_into(
            ChunkRecord,
            [
                (_BY_NAME["chunk"], iter(chunk_fresh)),
                (_BY_NAME["redact"], iter(redact_fresh)),
            ],
            _PIPELINE,
            trailer=OverlayTrailer.NEVER,
        )
    )
    assert _ids(folded) == _ids([_chunk("a", 0), _chunk("b", 0), _chunk("b", 0)])


def test_fold_into_never_folds_into_a_corpus_level_producer() -> None:
    """A corpus-level re-emitter already carries the union — folding again would double-count."""
    overlay = ReuseOverlay([_mention("a", 0)])
    union = [_mention("a", 0), _mention("b", 0)]  # the union its checkpoint already carries
    folded = list(
        overlay.fold_into(
            EntityMention,
            [(_BY_NAME["consolidate_types"], iter(union))],
            _PIPELINE,
            trailer=OverlayTrailer.APPEND_IF_CARRIER_UNSEEN,
        )
    )
    # Pass-through AND no trailer: the corpus-level parent counts as "carrier seen".
    assert _ids(folded) == _ids(union)


def test_trailer_append_if_carrier_unseen_appends_only_when_no_producer_enumerated() -> None:
    """The accumulator gather's union semantics for a degenerate DAG (no producing parent)."""
    overlay = ReuseOverlay([_mention("a", 0)])
    folded = list(
        overlay.fold_into(
            EntityMention,
            [(_BY_NAME["parse"], iter([]))],
            _PIPELINE,
            trailer=OverlayTrailer.APPEND_IF_CARRIER_UNSEEN,
        )
    )
    assert _ids(folded) == [_mention("a", 0).record_id]


def test_trailer_never_drops_an_unclaimed_overlay() -> None:
    """A doc-level Stage's input is its parents' records — never records no parent produced."""
    overlay = ReuseOverlay([_mention("a", 0)])
    folded = list(
        overlay.fold_into(
            EntityMention,
            [(_BY_NAME["parse"], iter([]))],
            _PIPELINE,
            trailer=OverlayTrailer.NEVER,
        )
    )
    assert folded == []


def test_trailer_always_appends_an_unclaimed_overlay_unconditionally() -> None:
    """The corpus-wide gather: an unclaimed overlay IS the corpus's records of the carrier."""
    overlay = ReuseOverlay([_mention("a", 0)])
    folded = list(
        overlay.fold_into(
            EntityMention,
            [(_BY_NAME["consolidate_types"], iter([]))],
            _PIPELINE,
            trailer=OverlayTrailer.ALWAYS,
        )
    )
    assert _ids(folded) == [_mention("a", 0).record_id]


def test_fold_into_with_an_unresolvable_stage_passes_through() -> None:
    """A ``None`` enumeration slot (defensive: name not in the validated DAG) never folds."""
    overlay = ReuseOverlay([_chunk("a", 0)])
    fresh = [_chunk("b", 0)]
    folded = list(
        overlay.fold_into(
            ChunkRecord, [(None, iter(fresh))], _PIPELINE, trailer=OverlayTrailer.NEVER
        )
    )
    assert _ids(folded) == _ids(fresh)


def test_fold_into_with_an_empty_overlay_is_a_pure_chain() -> None:
    """Non-delta byte-identity: the enumerated streams, same objects, nothing appended."""
    empty = ReuseOverlay()
    fresh = [_chunk("a", 0), _chunk("b", 0)]
    folded = list(
        empty.fold_into(
            ChunkRecord,
            [(_BY_NAME["chunk"], iter(fresh)), (_BY_NAME["redact"], iter([]))],
            _PIPELINE,
            trailer=OverlayTrailer.ALWAYS,
        )
    )
    assert folded == fresh


def test_fold_into_natural_order_interleave_end_to_end() -> None:
    """The 4249620 shape through the fold itself: a reused >10-chunk doc merges numerically."""
    reused = [_chunk("a", i) for i in range(11)]
    overlay = ReuseOverlay(sorted(reused, key=lambda c: c.record_id))
    fresh = [_chunk("b", 0)]
    folded = list(
        overlay.fold_into(
            ChunkRecord,
            [(_BY_NAME["chunk"], iter(fresh))],
            _PIPELINE,
            trailer=OverlayTrailer.NEVER,
        )
    )
    assert [c.index for c in folded[:11]] == list(range(11))
    assert _ids(folded[11:]) == _ids(fresh)


# -- count_for_stage: the accumulator's records_in twin -----------------------------------------


def test_count_for_stage_counts_each_carrier_once_at_its_first_doc_level_producer() -> None:
    """Mixed-carrier parent counts both mention overlays; a repeat carrier counts zero."""
    overlay = ReuseOverlay([_mention("a", 0), _mention("a", 1), _relation("a", 0)])
    counted: set[type[Record]] = set()
    assert (
        overlay.count_for_stage(_BY_NAME["extract"], _PIPELINE, already_counted=counted) == 3
    )
    # A second producer of the same carriers (the same Stage again is the simplest) adds 0.
    assert (
        overlay.count_for_stage(_BY_NAME["extract"], _PIPELINE, already_counted=counted) == 0
    )


def test_count_for_stage_is_zero_for_a_corpus_level_parent() -> None:
    """A corpus-level parent's checkpoint line count already contains the union — no top-up."""
    overlay = ReuseOverlay([_mention("a", 0)])
    counted: set[type[Record]] = set()
    count = overlay.count_for_stage(
        _BY_NAME["consolidate_types"], _PIPELINE, already_counted=counted
    )
    assert (count, counted) == (0, set())


# -- the Export and wiring-check faces ----------------------------------------------------------


def test_records_for_carrier_expands_the_mixed_sentinel_in_carrier_order() -> None:
    """A fused ancestor's pass-through carrier delivers mentions then relations, doc-ordered."""
    overlay = ReuseOverlay([_relation("a", 0), _mention("a", 0)])
    assert _ids(overlay.records_for_carrier(Record)) == [
        _mention("a", 0).record_id,
        _relation("a", 0).record_id,
    ]
    assert overlay.records_for_carrier(ChunkRecord) == []


def test_has_carrier_is_an_exact_type_lookup() -> None:
    """The wiring-check top-up asks for the parent's DECLARED carrier, concretely."""
    overlay = ReuseOverlay([_mention("a", 0)])
    checks = {
        "mention": overlay.has_carrier(EntityMention),
        "sentinel": overlay.has_carrier(Record),  # a declaration, not a concrete overlay carrier
        "chunk": overlay.has_carrier(ChunkRecord),
        "empty": ReuseOverlay().has_carrier(EntityMention),
    }
    assert checks == {"mention": True, "sentinel": False, "chunk": False, "empty": False}
