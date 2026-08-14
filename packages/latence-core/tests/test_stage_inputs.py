"""The Stage input plan is DATA, and ``StageInputs`` is the one thing that reads it (Wave 4-I).

"What does this Stage receive?" used to be answered in four places: the executor's four-way branch
(which picked the shape AND the ``records_in`` counting rule), two frozensets in ``runner.py`` that
the :class:`~latence_core.capability_descriptor.CapabilityDescriptor` did not derive, the dispatch's
SECOND decision of the same shape, and ~10 hand-written ``_require_*`` validator methods (three of
them dead). These tests hold the two lines that keep it one place:

1. **the plan is the partition** — the derived kind sets reproduce the executor's historical
   four-way split exactly, and the shape cannot disagree with ``accumulates`` (import-time gate);
2. **``StageInputs`` obeys the plan** — shape, count, overlay fold and every wiring message,
   driven against a FAKE record source, so a dispatch branch's input semantics are testable without
   executing a Pipeline (the thing the old form made impossible).

The wiring messages are pinned byte-for-byte here: they are what an operator reads when a stack is
mis-wired, and re-homing the table must not have reworded one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from latence_core.capability_descriptor import (
    ACCUMULATOR_KINDS,
    DESCRIPTORS,
    PER_CARRIER_INPUT_KINDS,
    STREAMED_INPUT_KINDS,
    StageInputPlan,
    StageInputShape,
    index_descriptors,
    input_plan_for,
)
from latence_core.contracts import (
    CanonicalEntity,
    ChunkRecord,
    Classification,
    DisambiguationRecord,
    DisambiguationScope,
    DocumentRecord,
    EntityMention,
    Evidence,
    GraphNode,
    GraphRecord,
    GraphScope,
    Provenance,
    Record,
    RelationMention,
)
from latence_core.errors import ContractError
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.reuse_overlay import ReuseOverlay
from latence_core.stage_inputs import StageInputs

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.runner import StageOutput

# -- fixtures: records, a Pipeline, and a FAKE record source -----------------------------------

_PROVENANCE = Provenance(
    source_uri="file:///c/a.txt",
    file_name="a.txt",
    file_type="txt",
    file_size=10,
    document_id="sha256:d1",
    char_start=0,
    char_end=10,
    page_start=1,
    page_end=1,
)
_CLASSIFICATION = Classification(language="en", sensitivity="public")


def _document(record_id: str = "d1") -> DocumentRecord:
    return DocumentRecord(
        record_id=record_id, provenance=_PROVENANCE, classification=_CLASSIFICATION
    )


def _chunk(record_id: str = "c1") -> ChunkRecord:
    return ChunkRecord(
        record_id=record_id,
        provenance=_PROVENANCE,
        classification=_CLASSIFICATION,
        document_record_id="d1",
        index=0,
        content="Acme sells",
        token_count=2,
    )


def _mention(record_id: str = "m1", *, document: str = "a") -> EntityMention:
    return EntityMention(
        record_id=record_id,
        provenance=_PROVENANCE.model_copy(
            update={"source_uri": f"file:///c/{document}.txt", "document_id": f"sha256:{document}"}
        ),
        classification=_CLASSIFICATION,
        document_record_id="d1",
        chunk_record_id="c1",
        index=0,
        label="ORG",
        text="Acme",
        confidence=1.0,
    )


def _relation(record_id: str = "r1") -> RelationMention:
    return RelationMention(
        record_id=record_id,
        provenance=_PROVENANCE,
        classification=_CLASSIFICATION,
        document_record_id="d1",
        index=0,
        label="sells",
        head_mention_id="m1",
        tail_mention_id="m2",
        head_text="Acme",
        tail_text="Widgets",
        confidence=1.0,
    )


def _graph(record_id: str = "g1") -> GraphRecord:
    node = GraphNode(
        record_id=f"node:{record_id}",
        provenance=_PROVENANCE,
        classification=_CLASSIFICATION,
        node_id=record_id,
        entity_id=f"ent:{record_id}",
        label="ORG",
        canonical_name="Acme",
        confidence=1.0,
        evidence=Evidence(mention_ids=["m1"], confidence=1.0),
    )
    return GraphRecord(
        record_id=record_id,
        provenance=_PROVENANCE,
        classification=_CLASSIFICATION,
        scope=GraphScope.NODE,
        node=node,
    )


def _entity(record_id: str = "e1") -> DisambiguationRecord:
    entity = CanonicalEntity(
        record_id=f"ent:{record_id}",
        provenance=_PROVENANCE,
        classification=_CLASSIFICATION,
        canonical_text="Acme",
        entity_type="ORG",
        member_mention_ids=["m1"],
        evidence=Evidence(mention_ids=["m1"], confidence=1.0),
    )
    return DisambiguationRecord(
        record_id=record_id,
        provenance=_PROVENANCE,
        classification=_CLASSIFICATION,
        scope=DisambiguationScope.ENTITY,
        entity=entity,
    )


def _stage(kind: CapabilityKind, name: str, *parents: str) -> dict[str, Any]:
    return {
        "name": name,
        "capability": kind.value,
        "provider": f"{kind.value}.fake",
        "depends_on": list(parents),
    }


def _pipeline(*stages: dict[str, Any]) -> Pipeline:
    return Pipeline.model_validate(
        {"name": "p", "storage_uri": "file:///tmp/x", "stages": list(stages)}
    )


class _FakeSource:
    """A :class:`~latence_core.stage_inputs.StageRecordSource` over an in-memory record map.

    Records every read so the tests can assert not just the verdict but HOW MUCH of the input was
    read to reach it — which is the whole point of the streamed shapes and of the scan's
    short-circuit.
    """

    def __init__(self, records: dict[str, list[StageOutput]], pipeline: Pipeline) -> None:
        self._records = records
        self._pipeline = pipeline
        self.reads: list[str] = []
        self.counts: list[str] = []

    def records(self, stage_name: str) -> Iterator[StageOutput]:
        for record in self._records.get(stage_name, []):
            self.reads.append(stage_name)
            yield record

    def count(self, stage_name: str) -> int:
        self.counts.append(stage_name)
        return len(self._records.get(stage_name, []))

    def carrier(self, stage: Stage) -> type[StageOutput] | type[Record]:
        from latence_core.run_store import stage_output_type

        return stage_output_type(stage, self._pipeline)


def _inputs(
    pipeline: Pipeline,
    stage_name: str,
    records: dict[str, list[StageOutput]],
    *,
    reuse: list[Record] | None = None,
) -> tuple[StageInputs, _FakeSource]:
    stage = next(s for s in pipeline.stages if s.name == stage_name)
    source = _FakeSource(records, pipeline)
    return (
        StageInputs(stage, pipeline, source, overlay=ReuseOverlay(reuse)),
        source,
    )


# -- 1. the plan IS the executor's four-way partition -------------------------------------------


def test_the_derived_kind_sets_reproduce_the_executors_historical_partition() -> None:
    """The two hand-maintained frozensets, now derived, name exactly the same Capabilities.

    This is the substitution's regression line: ``runner._STREAMED_INPUT_KINDS`` and
    ``runner._PER_CARRIER_INPUT_KINDS`` were literal sets the descriptor did not derive, so a new
    Capability could declare a shape and still be delivered a different one. The literals below are
    those sets as they stood before the move.
    """
    assert frozenset(
        {
            CapabilityKind.PARSE,
            CapabilityKind.CHUNK,
            CapabilityKind.SCHEMA_INDUCTION,
            CapabilityKind.ENTITY_EXTRACTION,
            CapabilityKind.FUSED_ENTITY_RELATION,
            CapabilityKind.REDACTION,
        }
    ) == STREAMED_INPUT_KINDS
    assert frozenset({CapabilityKind.CONTEXT_ENRICHMENT}) == PER_CARRIER_INPUT_KINDS


def test_every_capability_declares_exactly_one_shape_and_the_four_partition_the_kinds() -> None:
    """Shape is total and disjoint — the ``if/elif/elif/else`` it replaced could not overlap."""
    by_shape: dict[StageInputShape, set[CapabilityKind]] = {
        shape: {d.kind for d in DESCRIPTORS if d.input_plan.shape is shape}
        for shape in StageInputShape
    }
    assert set().union(*by_shape.values()) == set(CapabilityKind)
    assert sum(len(kinds) for kinds in by_shape.values()) == len(list(CapabilityKind))
    assert by_shape[StageInputShape.ACCUMULATOR] == set(ACCUMULATOR_KINDS)


def test_a_shape_that_disagrees_with_accumulates_is_refused_at_import() -> None:
    """The executor shapes memory off ``accumulates`` and delivers input off the shape.

    They name one fact. Declaring an accumulator that is not delivered as one (or the reverse) is
    exactly the drift the descriptor exists to make unrepresentable, so it raises where every other
    inconsistency does — at import, typed, naming the offender.
    """
    broken = tuple(
        d.__class__(**{**{f: getattr(d, f) for f in d.__slots__}, "accumulates": False})
        if d.kind is CapabilityKind.PROFILING
        else d
        for d in DESCRIPTORS
    )
    with pytest.raises(ContractError, match="input_plan shape disagrees with 'accumulates'"):
        index_descriptors(broken, CapabilityKind)


# -- 2. the shapes ------------------------------------------------------------------------------


def test_gathered_concatenates_parents_in_declared_order_and_is_idempotent() -> None:
    """A converging DAG sees a's records then b's — and a second read is not an empty stream.

    The dispatch used to guard a one-shot ``incoming`` generator with its own shape decision. There
    is nothing to guard now: ``gathered()`` re-reads through the source and caches, so asking twice
    is the same list rather than a silently emptied one.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.CHUNK, "a"),
        _stage(CapabilityKind.ENTITY_EXTRACTION, "b"),
        _stage(CapabilityKind.EXPORT, "export", "a", "b"),
    )
    inputs, _ = _inputs(
        pipeline, "export", {"a": [_chunk("c1"), _chunk("c2")], "b": [_mention("m1")]}
    )
    assert [r.record_id for r in inputs.gathered()] == ["c1", "c2", "m1"]
    assert [r.record_id for r in inputs.gathered()] == ["c1", "c2", "m1"]


def test_materialised_reports_the_transient_and_is_none_until_one_exists() -> None:
    """The ADR-0033 bound, readable: a streamed Stage never reaches an ``int``."""
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.CHUNK, "chunk", "parse"),
        _stage(CapabilityKind.EXPORT, "export", "chunk"),
    )
    streamed, _ = _inputs(pipeline, "chunk", {"parse": [_document(), _document("d2")]})
    assert streamed.materialised is None
    assert len(list(streamed.stream())) == 2
    assert streamed.materialised is None

    gathered, _ = _inputs(pipeline, "export", {"chunk": [_chunk()]})
    assert gathered.materialised is None
    assert gathered.count() == 1
    assert gathered.materialised == 1


def test_stream_is_pulled_one_record_at_a_time() -> None:
    """Pulling the k-th record has read exactly k — a list in a wrapper would read all of them."""
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.CHUNK, "chunk", "parse"),
    )
    inputs, source = _inputs(
        pipeline, "chunk", {"parse": [_document(f"d{i}") for i in range(5)]}
    )
    pulls = []
    for k, _record in enumerate(inputs.stream(), start=1):
        pulls.append((k, len(source.reads)))
    assert pulls == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]


def test_stream_of_narrows_one_carrier_out_of_the_parents_in_gathered_order() -> None:
    """The streamed form of ``[r for r in gathered if isinstance(r, X)]`` — same records, no list."""
    pipeline = _pipeline(
        _stage(CapabilityKind.FUSED_ENTITY_RELATION, "extract"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "extract"),
    )
    inputs, _ = _inputs(
        pipeline,
        "disambiguate",
        {"extract": [_mention("m1"), _relation("r1"), _mention("m2")]},
    )
    assert [r.record_id for r in inputs.stream_of(EntityMention)] == ["m1", "m2"]
    assert [r.record_id for r in inputs.stream_of(RelationMention)] == ["r1"]


def test_corpus_records_of_spans_the_whole_pipeline_not_just_the_parents() -> None:
    """A Disambiguator's chunks come from the Chunk Stage — a mention ANCESTOR, not a parent."""
    pipeline = _pipeline(
        _stage(CapabilityKind.CHUNK, "chunk"),
        _stage(CapabilityKind.ENTITY_EXTRACTION, "entities", "chunk"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "entities"),
    )
    inputs, _ = _inputs(
        pipeline,
        "disambiguate",
        {"chunk": [_chunk("c1"), _chunk("c2")], "entities": [_mention()]},
    )
    assert [r.record_id for r in inputs.stream_of(ChunkRecord)] == []
    assert [r.record_id for r in inputs.corpus_records_of(ChunkRecord)] == ["c1", "c2"]


# -- 3. the count -------------------------------------------------------------------------------


def test_count_is_the_line_count_for_a_streamed_stage_and_decodes_nothing() -> None:
    """``records_in`` for a streamed Stage costs no decode — the count is checkpoint LINES."""
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.CHUNK, "chunk", "parse"),
    )
    inputs, source = _inputs(pipeline, "chunk", {"parse": [_document(), _document("d2")]})
    assert inputs.count() == 2
    assert source.counts == ["parse"]
    assert source.reads == []


def test_count_for_an_accumulator_adds_the_overlays_reused_records_once_per_carrier() -> None:
    """W17/ADR-0061: the reused documents are counted where the fold rule delivers them."""
    pipeline = _pipeline(
        _stage(CapabilityKind.ENTITY_EXTRACTION, "entities"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "entities"),
    )
    inputs, _ = _inputs(
        pipeline,
        "disambiguate",
        {"entities": [_mention("m1")]},
        reuse=[_mention("m9"), _mention("m8")],
    )
    assert inputs.count() == 3


# -- 4. the wiring rules, and their exact messages ----------------------------------------------


def test_a_plan_with_no_rules_reads_nothing_at_all() -> None:
    """Calling ``validate`` uniformly from every branch is free for the Capabilities with no rule.

    That is what lets the dispatch stop knowing WHICH Capabilities validate — the branch that has
    no rule pays nothing for asking.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.CHUNK, "chunk", "parse"),
    )
    inputs, source = _inputs(pipeline, "chunk", {"parse": [_document()]})
    inputs.validate()
    assert source.reads == [] and source.counts == []


def test_an_empty_input_is_legitimately_empty_for_every_rule() -> None:
    """An empty run has no documents, therefore no chunks and no mentions — not a wiring mistake."""
    for kind, parents in (
        (CapabilityKind.PROFILING, ("parse",)),
        (CapabilityKind.DISAMBIGUATION, ("entities",)),
        (CapabilityKind.GRAPH_ASSEMBLY, ("resolve",)),
        (CapabilityKind.RELATION_EXTRACTION, ("parse",)),
        (CapabilityKind.CONTEXT_ENRICHMENT, ("parse",)),
    ):
        pipeline = _pipeline(
            _stage(CapabilityKind.PARSE, "parse"),
            _stage(CapabilityKind.ENTITY_EXTRACTION, "entities", "parse"),
            _stage(CapabilityKind.DISAMBIGUATION, "resolve", "entities"),
            _stage(kind, "under-test", *parents),
        )
        inputs, _ = _inputs(pipeline, "under-test", {})
        inputs.validate()


@pytest.mark.parametrize(
    ("kind", "name", "parents", "records", "message"),
    [
        (
            CapabilityKind.PROFILING,
            "profile",
            ("entities",),
            {"entities": [_mention()]},
            "Stage 'profile' (profiling) received no DocumentRecords (got: EntityMention). "
            "A Profiling Stage must depend on the Parse Stage (the documents it profiles). "
            "Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.DISAMBIGUATION,
            "disambiguate",
            ("parse",),
            {"parse": [_document()]},
            "Stage 'disambiguate' (disambiguation) received no EntityMentions "
            "(got: DocumentRecord). A Disambiguation Stage must depend on an Entity-Extraction "
            "(or fused) Stage (the mentions it resolves). Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.TYPE_CONSOLIDATION,
            "consolidate",
            ("parse",),
            {"parse": [_document()]},
            "Stage 'consolidate' (type_consolidation) received no EntityMentions "
            "(got: DocumentRecord). A Type-Consolidation Stage must depend on an "
            "Entity-Extraction (or fused) Stage (the mentions whose types it canonicalizes). "
            "Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.GRAPH_ASSEMBLY,
            "graph",
            ("entities",),
            {"entities": [_mention()]},
            "Stage 'graph' (graph_assembly) requires DisambiguationRecord inputs but received: "
            "EntityMention. Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.GRAPH_COMPLETION,
            "complete",
            ("entities",),
            {"entities": [_mention()]},
            "Stage 'complete' (graph_completion) requires GraphRecord inputs but received: "
            "EntityMention. A Graph Completion Stage must depend on the Graph Assembly Stage "
            "(the assembled nodes + edges it completes). Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.RELATION_EXTRACTION,
            "relations",
            ("chunk",),
            {"chunk": [_chunk()]},
            "Stage 'relations' (relation_extraction) received no EntityMentions. "
            "A Relation-Extraction Stage must depend on an Entity-Extraction Stage "
            "(the mentions it relates). Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.RELATION_EXTRACTION,
            "relations",
            ("entities",),
            {"entities": [_mention()]},
            "Stage 'relations' (relation_extraction) received no ChunkRecords. "
            "A Relation-Extraction Stage must also depend on the Chunk (or Content-Screening) "
            "Stage for the text + page map. Check depends_on.",
        ),
        (
            CapabilityKind.CONTEXT_ENRICHMENT,
            "enrich",
            ("assemble",),
            {"assemble": [_graph()]},
            "Stage 'enrich' (context_enrichment) received no ChunkRecords (got: GraphRecord). "
            "A Context-Enrichment Stage must depend on the Chunk (or Content-Screening / "
            "Schema-Induction) Stage — the chunks it enriches. "
            "Check the Pipeline's depends_on wiring.",
        ),
        (
            CapabilityKind.CONTEXT_ENRICHMENT,
            "enrich",
            ("chunk",),
            {"chunk": [_chunk()]},
            "Stage 'enrich' (context_enrichment) received no GraphRecords (got: ChunkRecord). "
            "A Context-Enrichment Stage must also depend on the Graph Assembly Stage — the "
            "assembled KG it projects onto each chunk. Check the Pipeline's depends_on wiring.",
        ),
    ],
)
def test_every_wiring_message_is_byte_identical_to_the_one_the_runner_raised(
    kind: CapabilityKind,
    name: str,
    parents: tuple[str, ...],
    records: dict[str, list[StageOutput]],
    message: str,
) -> None:
    """The messages ARE the interface — an operator reads them when a stack is mis-wired.

    Ten validator methods each transcribed their own; there is now one copy per rule, declared on
    the Capability. Pinned exactly, including the ``got:`` list and the two rules (Relation's) that
    deliberately name no carriers at all.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.CHUNK, "chunk", "parse"),
        _stage(CapabilityKind.ENTITY_EXTRACTION, "entities", "chunk"),
        _stage(CapabilityKind.GRAPH_ASSEMBLY, "assemble", "entities"),
        _stage(kind, name, *parents),
    )
    inputs, _ = _inputs(pipeline, name, records)
    with pytest.raises(ContractError) as excinfo:
        inputs.validate()
    assert str(excinfo.value) == message


def test_required_carriers_are_checked_in_declaration_order() -> None:
    """A Stage missing BOTH sides names the same one it always did (mentions before chunks)."""
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.RELATION_EXTRACTION, "relations", "parse"),
    )
    inputs, _ = _inputs(pipeline, "relations", {"parse": [_document()]})
    with pytest.raises(ContractError, match="received no EntityMentions"):
        inputs.validate()


def test_the_exclusive_rule_names_every_forbidden_carrier_sorted() -> None:
    """Graph Assembly's verdict depends on the whole stream, so all offenders are reported."""
    pipeline = _pipeline(
        _stage(CapabilityKind.FUSED_ENTITY_RELATION, "extract"),
        _stage(CapabilityKind.GRAPH_ASSEMBLY, "graph", "extract"),
    )
    inputs, _ = _inputs(pipeline, "graph", {"extract": [_relation(), _mention()]})
    with pytest.raises(ContractError) as excinfo:
        inputs.validate()
    assert "received: EntityMention, RelationMention." in str(excinfo.value)


def test_a_correctly_wired_exclusive_stage_passes_with_its_own_carrier() -> None:
    pipeline = _pipeline(
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate"),
        _stage(CapabilityKind.GRAPH_ASSEMBLY, "graph", "disambiguate"),
    )
    inputs, _ = _inputs(pipeline, "graph", {"disambiguate": [_entity(), _entity("e2")]})
    inputs.validate()


def test_the_scan_short_circuits_only_where_the_plan_declares_it_sound() -> None:
    """Context Enrichment stops at the second carrier; an accumulator scan still runs to the end.

    The short-circuit is declared, not inferred, precisely because it changes how much is READ.
    A plan with an EXCLUSIVE rule could never declare it — that verdict depends on every type in
    the stream — and the accumulator checks that predate it still pay the full pass.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.CHUNK, "chunk"),
        _stage(CapabilityKind.GRAPH_ASSEMBLY, "graph"),
        _stage(CapabilityKind.CONTEXT_ENRICHMENT, "enrich", "chunk", "graph"),
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.PROFILING, "profile", "parse"),
    )
    enrich, enrich_source = _inputs(
        pipeline,
        "enrich",
        {
            "chunk": [_chunk(f"c{i}") for i in range(5)],
            "graph": [_graph(f"g{i}") for i in range(5)],
        },
    )
    enrich.validate()
    assert len(enrich_source.reads) == 6, "the scan read past the first record of the 2nd carrier"

    profile, profile_source = _inputs(
        pipeline, "profile", {"parse": [_document(f"d{i}") for i in range(5)]}
    )
    profile.validate()
    assert len(profile_source.reads) == 5

    assert input_plan_for(CapabilityKind.CONTEXT_ENRICHMENT).scan_short_circuit
    assert not input_plan_for(CapabilityKind.PROFILING).scan_short_circuit
    assert not any(
        d.input_plan.scan_short_circuit and d.input_plan.exclusive is not None
        for d in DESCRIPTORS
    ), "a short-circuited scan cannot decide an exclusive rule"


def test_an_all_reused_parent_still_counts_as_producing_its_carrier(
) -> None:
    """W17: a delta where every document was reused is not a mis-wired empty parent.

    The parent's FRESH output is empty because its documents were excluded at Source, but the
    overlay carries their records — so the wiring check must see the carrier, or an incremental
    run of a correctly-wired stack fails on the wiring rule.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.ENTITY_EXTRACTION, "entities"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "entities"),
    )
    blind, _ = _inputs(pipeline, "disambiguate", {"entities": [_document()]})
    with pytest.raises(ContractError, match="received no EntityMentions"):
        blind.validate()

    reused, _ = _inputs(
        pipeline,
        "disambiguate",
        {"entities": [_document()]},
        reuse=[_mention("m9")],
    )
    reused.validate()


# -- 5. the trailer policies each shape carries -------------------------------------------------


def test_the_accumulator_and_per_carrier_shapes_carry_different_trailer_policies() -> None:
    """The one honest behavioral difference between the two per-carrier reads (ADR-0061 §5).

    An accumulator's contract is union semantics, so an overlay carrier NO parent produces is still
    delivered; a doc-level per-carrier Stage's input is its parents' records and nothing else. Both
    Stages below have a parent that produces neither carrier, so only the trailer distinguishes
    them.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "parse"),
        _stage(CapabilityKind.CONTEXT_ENRICHMENT, "enrich", "parse"),
    )
    accumulator, _ = _inputs(
        pipeline, "disambiguate", {"parse": [_document()]}, reuse=[_mention("m9")]
    )
    assert [r.record_id for r in accumulator.stream_of(EntityMention)] == ["m9"]

    per_carrier, _ = _inputs(
        pipeline, "enrich", {"parse": [_document()]}, reuse=[_chunk("c9")]
    )
    assert [r.record_id for r in per_carrier.stream_of(ChunkRecord)] == []


def test_the_corpus_wide_read_always_delivers_an_unclaimed_overlay() -> None:
    """Its contract is "every record of this carrier in the run" — so nothing is dropped."""
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "parse"),
    )
    inputs, _ = _inputs(
        pipeline, "disambiguate", {"parse": [_document()]}, reuse=[_chunk("c9")]
    )
    assert [r.record_id for r in inputs.corpus_records_of(ChunkRecord)] == ["c9"]


def test_the_overlay_folds_into_the_first_doc_level_producer_not_the_end() -> None:
    """The fold rule's position, through this seam (ADR-0061 §5).

    Two parents produce the carrier. The reused document's records belong where a full rebuild
    would have delivered them — document-interleaved into the FIRST doc-level producer's stream,
    not appended after the last parent, which is what "same stream, not just same set" means.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.ENTITY_EXTRACTION, "first"),
        _stage(CapabilityKind.ENTITY_EXTRACTION, "second"),
        _stage(CapabilityKind.DISAMBIGUATION, "disambiguate", "first", "second"),
    )
    inputs, _ = _inputs(
        pipeline,
        "disambiguate",
        {
            "first": [_mention("m2", document="b")],
            "second": [_mention("m3", document="c")],
        },
        reuse=[_mention("m1", document="a")],
    )
    assert [r.record_id for r in inputs.stream_of(EntityMention)] == ["m1", "m2", "m3"]


# -- 6. the plan can be supplied, which is what makes a dispatch branch testable ----------------


def test_a_supplied_plan_overrides_the_capabilitys_own() -> None:
    """A test (or a probe) can drive any shape through the same object, with no Pipeline run."""
    pipeline = _pipeline(
        _stage(CapabilityKind.PARSE, "parse"),
        _stage(CapabilityKind.CHUNK, "chunk", "parse"),
    )
    stage = next(s for s in pipeline.stages if s.name == "chunk")
    source = _FakeSource({"parse": [_document(), _document("d2")]}, pipeline)
    as_gathered = StageInputs(
        stage, pipeline, source, plan=StageInputPlan(shape=StageInputShape.GATHERED)
    )
    assert as_gathered.count() == 2
    assert as_gathered.materialised == 2
    assert source.counts == [], "a gathered count must not go through the line counter"
