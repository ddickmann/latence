"""The CapabilityDescriptor is the ONE place a Capability's facts live — and it is enforced.

Eight tables in six modules each encoded one fact about a Capability (Protocol, checkpoint carrier,
Export pass-through carrier, corpus-level for the executor / for the phase boundary / for delta
reuse, entry-point prefix, Quality section). Nothing related them, so adding a Capability meant
finding eight tables by memory — and one was missed: ``TYPE_CONSOLIDATION`` landed in the Runner's
and the DeltaRunner's corpus-level tuples but in NEITHER set in ``stacks.phase_boundary``, so the
invariant whose whole job is "no corpus-level operation runs before redaction" waved a front-half
``consolidate_types`` through.

These tests hold the two lines that make that unrepresentable:

1. **Exhaustiveness at import** — a CapabilityKind with no descriptor raises before any code runs.
2. **Derivation, not duplication** — the sets the other modules consume are computed from the
   descriptors, and are asserted here to agree with what those modules actually do today (the
   Runner's carrier resolution and accumulator tuple included, so the remaining wiring is a
   substitution rather than a rewrite).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.capability_descriptor import (
    ACCUMULATOR_KINDS,
    CORPUS_LEVEL_KINDS,
    DESCRIPTORS,
    DOCUMENT_LEVEL_KINDS,
    CapabilityLevel,
    dag_protocol_for,
    descriptor_for,
    export_carrier_of,
    index_descriptors,
    is_corpus_level,
    kind_for_entry_point,
)
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    ContextHeader,
    DocumentRecord,
    Provenance,
    Record,
)
from latence_core.delta_runner import DeltaRunner
from latence_core.errors import ContractError
from latence_core.pipeline import CapabilityKind, Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import _ACCUMULATOR_KINDS, LocalRunner
from latence_core.stacks.phase_boundary import (
    CHUNK_LEVEL_KINDS,
    check_phase_boundary,
)
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

# Kinds that are not top-level DAG Stages, so a single-Stage Pipeline of that kind is meaningless.
_NOT_A_STAGE = {CapabilityKind.EMBEDDING}

_CONTEXT_HEADER = ContextHeader(
    entities=["Acme"],
    entity_types=["ORG"],
    neighbor_triples=["Acme —sells→ Widgets"],
)


def _chunk_record(*, context_header: ContextHeader | None) -> ChunkRecord:
    """One chunk, optionally carrying the KG-derived header the enrichment Stage stamps on."""
    return ChunkRecord(
        record_id="c1",
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=10,
            document_id="sha256:d1",
            char_start=0,
            char_end=10,
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=0,
        content="Acme sells",
        token_count=2,
        context_header=context_header,
    )


def _stage(kind: CapabilityKind, name: str = "s", **config: Any) -> dict[str, Any]:
    return {
        "name": name,
        "capability": kind.value,
        "provider": f"{kind.value}.fake",
        "config": config,
    }


def _pipeline(*stages: dict[str, Any]) -> Pipeline:
    return Pipeline.model_validate(
        {"name": "p", "storage_uri": "file:///tmp/x", "stages": list(stages)}
    )


# -- 1. exhaustiveness: a Capability cannot exist without declaring its facts -------------------


def test_a_capability_kind_with_no_descriptor_is_rejected() -> None:
    """Dropping one descriptor is exactly "a new Capability nobody declared" — and it must raise."""
    trimmed = tuple(d for d in DESCRIPTORS if d.kind is not CapabilityKind.TYPE_CONSOLIDATION)
    with pytest.raises(ContractError) as excinfo:
        index_descriptors(trimmed, CapabilityKind)
    message = str(excinfo.value)
    assert "no CapabilityDescriptor for type_consolidation" in message
    # The message must tell the author what to declare, not just that something is missing.
    assert "carrier" in message
    assert "level" in message


def test_a_new_capability_kind_without_a_descriptor_fails_at_import() -> None:
    """The ACCEPTANCE guard, run for real: add a CapabilityKind, and the module refuses to import.

    Re-imports ``latence_core.capability_descriptor`` in a fresh interpreter against a
    ``CapabilityKind`` extended with an undeclared member. The module body's
    :func:`index_descriptors` call is what fails, so the failure lands at IMPORT — before any
    ``is_corpus_level`` / ``capability_protocol`` / carrier lookup can silently answer wrongly for
    the new Capability. A subprocess keeps the doctored enum out of this session.
    """
    script = textwrap.dedent(
        """
        import enum, importlib, sys
        import latence_core.pipeline as pipeline

        extended = enum.StrEnum(
            "CapabilityKind",
            [(m.name, m.value) for m in pipeline.CapabilityKind]
            + [("VIBE_EXTRACTION", "vibe_extraction")],
        )
        pipeline.CapabilityKind = extended
        sys.modules.pop("latence_core.capability_descriptor", None)
        try:
            importlib.import_module("latence_core.capability_descriptor")
        except Exception as exc:  # noqa: BLE001 - the point is to report the type + message
            print(type(exc).__name__, exc)
            raise SystemExit(7) from exc
        raise SystemExit(0)
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7, f"import unexpectedly succeeded: {result.stdout}{result.stderr}"
    assert "ContractError" in result.stdout
    assert "vibe_extraction" in result.stdout


def test_a_descriptor_for_an_unknown_kind_and_a_duplicate_are_both_rejected() -> None:
    """The cover must be EXACT: a stale descriptor and a duplicate are as wrong as a missing one."""
    with pytest.raises(ContractError, match="unknown CapabilityKind"):
        index_descriptors(
            DESCRIPTORS, [k for k in CapabilityKind if k is not CapabilityKind.EMBEDDING]
        )
    with pytest.raises(ContractError, match="duplicate CapabilityDescriptor for source"):
        index_descriptors((*DESCRIPTORS, descriptor_for(CapabilityKind.SOURCE)), CapabilityKind)


def test_every_capability_kind_resolves_to_a_descriptor() -> None:
    """``descriptor_for`` is total — the postcondition the import-time check buys us."""
    for kind in CapabilityKind:
        assert descriptor_for(kind).kind is kind


def test_levels_partition_the_dag_kinds() -> None:
    """Corpus- and document-level are COMPLEMENTS, so no kind can land in neither.

    The old pair of hand-written sets in ``phase_boundary`` had exactly that hole: a kind in
    neither set silently answered "chunk-level" to ``is_corpus_level``.
    """
    dag_kinds = {d.kind for d in DESCRIPTORS if d.dag_node}
    assert dag_kinds == CORPUS_LEVEL_KINDS | DOCUMENT_LEVEL_KINDS
    assert set() == CORPUS_LEVEL_KINDS & DOCUMENT_LEVEL_KINDS
    assert CHUNK_LEVEL_KINDS is DOCUMENT_LEVEL_KINDS


# -- 2. the defect the drift caused: the phase boundary now sees the corpus-collapsing Stage ----


def test_type_consolidation_is_corpus_level() -> None:
    """A canonical type vocabulary cannot be elected from a prefix of the labels (ADR-0054)."""
    stage = _pipeline(_stage(CapabilityKind.TYPE_CONSOLIDATION)).stages[0]
    assert is_corpus_level(stage) is True
    assert descriptor_for(CapabilityKind.TYPE_CONSOLIDATION).level is CapabilityLevel.CORPUS


def test_type_consolidation_before_redaction_violates_the_phase_boundary() -> None:
    """THE regression: a hand-edited config that puts ``consolidate_types`` in the front half.

    Before the descriptor, ``is_corpus_level`` answered False for ``type_consolidation`` (it was in
    neither of ``phase_boundary``'s two hand-written sets), so this DAG — a corpus-collapsing Stage
    feeding the last chunk-level Stage — passed the invariant unflagged. The shipped stacks happen
    to sit on the safe side of the line, which is exactly why nothing caught it.
    """
    pipeline = _pipeline(
        _stage(CapabilityKind.SOURCE, "source"),
        {**_stage(CapabilityKind.PARSE, "parse"), "depends_on": ["source"]},
        {**_stage(CapabilityKind.CHUNK, "chunk"), "depends_on": ["parse"]},
        {**_stage(CapabilityKind.ENTITY_EXTRACTION, "extract"), "depends_on": ["chunk"]},
        {
            **_stage(CapabilityKind.TYPE_CONSOLIDATION, "consolidate_types"),
            "depends_on": ["extract"],
        },
        {**_stage(CapabilityKind.REDACTION, "redact"), "depends_on": ["consolidate_types"]},
    )
    violations = check_phase_boundary(pipeline)
    assert [(v.stage, v.corpus_stage) for v in violations] == [("redact", "consolidate_types")]
    assert "corpus-level stage 'consolidate_types'" in violations[0].describe()


def test_type_consolidation_after_redaction_is_still_clean() -> None:
    """The negative control: the placement ADR-0054 prescribes must NOT become a false alarm."""
    pipeline = _pipeline(
        _stage(CapabilityKind.SOURCE, "source"),
        {**_stage(CapabilityKind.PARSE, "parse"), "depends_on": ["source"]},
        {**_stage(CapabilityKind.CHUNK, "chunk"), "depends_on": ["parse"]},
        {**_stage(CapabilityKind.ENTITY_EXTRACTION, "extract"), "depends_on": ["chunk"]},
        {**_stage(CapabilityKind.REDACTION, "redact"), "depends_on": ["chunk"]},
        {
            **_stage(CapabilityKind.TYPE_CONSOLIDATION, "consolidate_types"),
            "depends_on": ["extract", "redact"],
        },
        {
            **_stage(CapabilityKind.DISAMBIGUATION, "disambiguate"),
            "depends_on": ["consolidate_types"],
        },
    )
    assert check_phase_boundary(pipeline) == []


def test_schema_induction_level_still_follows_its_granularity() -> None:
    """The one config-dependent level survives the move onto the descriptor."""
    levels = {}
    for key, granularity in {
        "chunk": "chunk",
        "doc": "doc",
        "corpus": "  CORPUS  ",  # case/whitespace tolerant
    }.items():
        pipeline = _pipeline(_stage(CapabilityKind.SCHEMA_INDUCTION, granularity=granularity))
        levels[key] = is_corpus_level(pipeline.stages[0])
    levels["absent"] = is_corpus_level(_pipeline(_stage(CapabilityKind.SCHEMA_INDUCTION)).stages[0])
    assert levels == {"chunk": False, "doc": False, "corpus": True, "absent": False}


# -- 3. the descriptor's facts agree with what the Runner actually does today -------------------
# The Runner's four tables are wired to the descriptor separately (it is a hot file). These pin the
# descriptor's declarations to the Runner's live behaviour, so that wiring is a substitution.


def test_declared_carriers_match_the_runners_checkpoint_carrier() -> None:
    """Every Stage kind's declared ``carrier`` is what ``LocalRunner._output_type`` returns."""
    for kind in CapabilityKind:
        if kind in _NOT_A_STAGE or kind is CapabilityKind.EXPORT:
            continue  # EXPORT is a pass-through sink; covered by the Export-carrier test below
        pipeline = _pipeline(_stage(kind))
        actual = LocalRunner._output_type(pipeline.stages[0], pipeline)  # noqa: SLF001
        assert descriptor_for(kind).carrier is actual, kind


def test_export_is_declared_as_a_pass_through_sink() -> None:
    """EXPORT has no carrier of its own; the Runner resolves it from the nearest producer."""
    assert descriptor_for(CapabilityKind.EXPORT).carrier is None
    assert export_carrier_of(CapabilityKind.EXPORT) is None


def test_declared_export_carriers_match_the_runners_export_resolution() -> None:
    """``export_carrier_of`` reproduces ``LocalRunner._export_carrier`` for every ancestor kind.

    ``None`` means "not the producer an Export re-emits — keep walking", which bottoms out at
    ``DocumentRecord``; that is precisely how a Source/Intake-Screening ancestor resolves today.
    """
    for kind in CapabilityKind:
        if kind in _NOT_A_STAGE:
            continue
        pipeline = _pipeline(
            _stage(kind, "producer"),
            {**_stage(CapabilityKind.EXPORT, "export"), "depends_on": ["producer"]},
        )
        actual = LocalRunner._export_carrier(pipeline.stages[1], pipeline)  # noqa: SLF001
        assert (export_carrier_of(kind) or DocumentRecord) is actual, kind


def test_declared_accumulators_match_the_runners_accumulator_tuple() -> None:
    """``accumulates`` is the executor's memory-shaping fact (ADR-0033), declared once."""
    assert frozenset(_ACCUMULATOR_KINDS) == ACCUMULATOR_KINDS
    # It is strictly narrower than "corpus-level": a sink and a chunk-stream re-emitter are
    # corpus-level but must NOT be materialized by the executor.
    assert ACCUMULATOR_KINDS < CORPUS_LEVEL_KINDS
    assert not descriptor_for(CapabilityKind.EXPORT).accumulates
    assert not descriptor_for(CapabilityKind.CONTEXT_ENRICHMENT).accumulates


def test_mixed_carrier_kinds_are_the_ones_read_polymorphically() -> None:
    """A fused extractor and Type Consolidation share one MIXED checkpoint (signalled by Record)."""
    mixed = {d.kind for d in DESCRIPTORS if d.mixed_carrier}
    assert mixed == {CapabilityKind.FUSED_ENTITY_RELATION, CapabilityKind.TYPE_CONSOLIDATION}
    assert descriptor_for(CapabilityKind.FUSED_ENTITY_RELATION).carrier is Record


# -- 4. the remaining facts point at things that really exist -----------------------------------


def test_every_declared_quality_section_is_a_real_quality_report_field() -> None:
    """A Quality section nobody can fill is documentation posing as a guard — the sin we removed."""
    fields = set(QualityReport.model_fields)
    declared = {d.quality_section for d in DESCRIPTORS if d.quality_section is not None}
    assert declared <= fields, sorted(declared - fields)


def test_every_declared_conformance_case_is_a_real_conformance_capability() -> None:
    from latence_core.conformance.cases import Capability

    for descriptor in DESCRIPTORS:
        assert Capability(descriptor.conformance_case)


def test_entry_point_prefixes_classify_the_shipped_provider_names() -> None:
    """Including the two Screening kinds, whose split moved out of an if-chain onto the table."""
    assert kind_for_entry_point("screening.intake_pdf") is CapabilityKind.INTAKE_SCREENING
    assert kind_for_entry_point("screening.content_toxicity") is CapabilityKind.CONTENT_SCREENING
    assert kind_for_entry_point("parser.plaintext") is CapabilityKind.PARSE
    assert kind_for_entry_point("parse.document") is CapabilityKind.PARSE
    assert kind_for_entry_point("graph.canonical") is CapabilityKind.GRAPH_ASSEMBLY
    assert kind_for_entry_point("graph_completion.rescal") is CapabilityKind.GRAPH_COMPLETION
    with pytest.raises(KeyError, match="none of: intake\\*, content\\*"):
        kind_for_entry_point("screening.sideways")
    with pytest.raises(KeyError, match="unknown Capability prefix"):
        kind_for_entry_point("mystery.provider")


def test_every_dag_kind_declares_a_protocol_and_embedding_refuses() -> None:
    for kind in CapabilityKind:
        if kind in _NOT_A_STAGE:
            with pytest.raises(KeyError):
                dag_protocol_for(kind)
            continue
        assert dag_protocol_for(kind) is descriptor_for(kind).protocol


def test_the_delta_reuse_skip_set_is_the_descriptor_corpus_set() -> None:
    """The DeltaRunner and the phase boundary ask the SAME question, so they read the SAME answer.

    They used to be two hand-written tuples and they disagreed (``TYPE_CONSOLIDATION`` in one,
    ``CONTEXT_ENRICHMENT`` in the other). Deterministic order is asserted too: the reuse store's
    Stage walk must be reproducible.
    """
    from latence_core.delta_runner import _CORPUS_LEVEL_KINDS

    assert frozenset(_CORPUS_LEVEL_KINDS) == CORPUS_LEVEL_KINDS
    assert list(_CORPUS_LEVEL_KINDS) == [k for k in CapabilityKind if k in CORPUS_LEVEL_KINDS]


def test_the_reuse_store_keeps_the_per_document_chunk_not_the_kg_derived_one(
    tmp_path: Path,
) -> None:
    """The BEHAVIOURAL consequence of unifying the delta set with the phase-boundary set (W17).

    ``context_enrichment`` is corpus-level — a chunk's ``context_header`` summarizes the mutable
    assembled graph — which is exactly why ``DeltaRunner._reuse_supported`` refuses cross-run reuse
    for such a pipeline. But the DeltaRunner's own skip tuple did not list it, so the reuse store
    collected the ENRICHED chunk (topologically last, last-writer-wins) and persisted a
    KG-derived artifact as if it were a per-document one — the stale-header hazard, written down.

    Reading the level off the Capability descriptor makes the two agree: the store now keeps the
    redacted chunk (reproducible from the document's own bytes) and drops the enriched variant.
    """
    storage_uri = tmp_path.as_uri()
    pipeline = _pipeline(
        _stage(CapabilityKind.SOURCE, "source"),
        {**_stage(CapabilityKind.PARSE, "parse"), "depends_on": ["source"]},
        {**_stage(CapabilityKind.CHUNK, "chunk"), "depends_on": ["parse"]},
        {**_stage(CapabilityKind.REDACTION, "redact"), "depends_on": ["chunk"]},
        {**_stage(CapabilityKind.GRAPH_ASSEMBLY, "graph"), "depends_on": ["redact"]},
        {**_stage(CapabilityKind.CONTEXT_ENRICHMENT, "enrich"), "depends_on": ["graph"]},
    )
    pipeline = pipeline.model_copy(update={"storage_uri": storage_uri})

    storage = Storage.for_uri(storage_uri)
    for stage_name, header in (("redact", None), ("enrich", _CONTEXT_HEADER)):
        record = _chunk_record(context_header=header)
        storage.write_atomic(
            f"{storage_uri.rstrip('/')}/_latence/runs/r1/checkpoints/{stage_name}.jsonl",
            (record.model_dump_json() + "\n").encode("utf-8"),
        )

    collected = DeltaRunner()._doc_level_records(pipeline, "r1")  # noqa: SLF001
    assert [type(r).__name__ for r in collected] == ["ChunkRecord"]
    kept = collected[0]
    assert isinstance(kept, ChunkRecord)
    assert kept.context_header is None, "the KG-derived chunk variant must not enter the reuse store"


def test_the_descriptor_table_is_declared_in_capability_kind_order() -> None:
    """Determinism + readability: the table reads in the order the pipeline runs."""
    assert [d.kind for d in DESCRIPTORS] == list(CapabilityKind)


def test_a_descriptor_is_frozen() -> None:
    """Facts about a Capability are not runtime-mutable — no module may 'fix' one locally."""
    descriptor = descriptor_for(CapabilityKind.REDACTION)
    with pytest.raises((AttributeError, TypeError)):
        descriptor.level = CapabilityLevel.CORPUS  # type: ignore[misc]


def test_capability_level_is_a_str_enum_for_stable_serialization() -> None:
    assert isinstance(CapabilityLevel.CORPUS, StrEnum)
    assert str(CapabilityLevel.CORPUS) == "corpus"
    assert sorted(str(level) for level in CapabilityLevel) == ["corpus", "document"]
