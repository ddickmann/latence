"""One frozen :class:`CapabilityDescriptor` per Capability — every fact about a Capability, once.

A Capability is not just a Protocol. Adding one means declaring, somewhere, **nine** separate
facts about it: the Protocol its Provider must satisfy, the record carrier it checkpoints, the
carrier an Export downstream of it re-emits, whether it is a corpus-level accumulator for the
executor's memory shaping, whether it is corpus-level for the PHASE-BOUNDARY invariant, whether it
is corpus-level for delta reuse, its Provider entry-point name prefix, which Quality Report
section it fills, and what its Stage RECEIVES (the input shape the executor delivers plus the
wiring its input must satisfy). Before this module each of those lived in its own hand-ordered
table in a different file, and *nothing* related them — so adding a Capability meant finding nine
tables by memory, and forgetting one was silent.

It had already happened. ``TYPE_CONSOLIDATION`` (T3, ADR-0054) was added to the Runner's
accumulator tuple and to the DeltaRunner's corpus-level tuple but to **neither** set in
``stacks/phase_boundary``. ``is_corpus_level`` therefore answered *False* for a Stage that
collapses the whole run's mentions into one canonical type vocabulary, and
``check_phase_boundary`` — the invariant whose entire job is "no corpus-level operation runs before
redaction" — waved a hand-written config that put ``consolidate_types`` in the front half straight
through. (``phase_boundary.CHUNK_LEVEL_KINDS`` was worse than stale: it was referenced by no code
at all, documentation posing as a guard.) ``capability_guard`` meanwhile *claimed* its map was
"kept in ONE place so the two cannot drift" while three other places held the same shape.

The fix is not a better comment, it is making the drift **unrepresentable**:

* every fact about a Capability is a FIELD of one frozen descriptor, declared next to its siblings;
* the descriptor table is checked for EXHAUSTIVENESS at import (:func:`index_descriptors`), so a
  new :class:`~latence_core.pipeline.CapabilityKind` that does not declare its facts raises a typed
  :class:`~latence_core.errors.ContractError` the first time anything imports ``latence_core`` —
  there is no path where the new Capability merely *behaves* wrongly;
* the derived sets other modules used to hand-maintain (:data:`CORPUS_LEVEL_KINDS`,
  :data:`DOCUMENT_LEVEL_KINDS`, :data:`ACCUMULATOR_KINDS`, the entry-point prefix table, the
  capability→Protocol map) are COMPUTED here from the descriptors, so they cannot disagree with
  each other or go stale. ``DOCUMENT_LEVEL_KINDS`` in particular is now the exact complement of
  ``CORPUS_LEVEL_KINDS`` over the DAG kinds — a partition, not two lists that hopefully agree.

This is ADR-0004's line held properly: the Capability owns the facts about itself, rather than the
Runner (and the guard, and the delta engine, and the conformance suite) each owning a copy.

**Scope.** The descriptor is a pure declaration: no I/O, no Provider instantiation, no heavy
imports (ADR-0016 — this module pulls only ``capability``/``contracts``/``pipeline``, all
torch-free). Reading a descriptor never runs Provider work.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from latence_core.capability import (
    Chunker,
    ContentScreener,
    ContextEnricher,
    Disambiguator,
    Embedder,
    EntityExtractor,
    Export,
    FusedEntityRelationExtractor,
    GraphAssembler,
    GraphCompleter,
    IntakeScreener,
    LabelInducer,
    Parser,
    PIIDetector,
    Profiler,
    RelationExtractor,
    Source,
    TypeConsolidator,
)
from latence_core.contracts import (
    ChunkRecord,
    DisambiguationRecord,
    DocumentRecord,
    EntityMention,
    FeatureRecord,
    GraphRecord,
    ParserInput,
    Record,
    RelationMention,
)
from latence_core.errors import ContractError
from latence_core.pipeline import CapabilityKind

if TYPE_CHECKING:
    from latence_core.pipeline import Stage


class CapabilityLevel(StrEnum):
    """Does a Stage of this Capability see one unit at a time, or the whole corpus?

    ``DOCUMENT`` — the front half. It sees one chunk or one document and never aggregates, so it
    can run per-document, be resumed per-document, and its output can be REUSED across a delta for
    an unchanged document (W17/ADR-0043).

    ``CORPUS`` — the back half. It consumes the whole corpus (or an artifact built from it) to
    produce its output, so it cannot run before the front half has finished, its checkpoint spans
    documents that were never re-extracted, and the PHASE-BOUNDARY invariant forbids a chunk-level
    Stage from depending on it.

    One Capability's level depends on its *config* rather than its kind — ``SCHEMA_INDUCTION`` at
    ``granularity: corpus`` — which is why callers ask :meth:`CapabilityDescriptor.level_of` with
    the Stage rather than reading :attr:`CapabilityDescriptor.level` directly.
    """

    DOCUMENT = "document"
    CORPUS = "corpus"


class StageInputShape(StrEnum):
    """HOW the Runner delivers a Stage's input — the executor's four-way input decision, declared.

    The Runner used to decide this in a four-way ``if`` chain over two hand-maintained frozensets,
    and the dispatch then re-decided it a second time to know whether ``incoming`` was safe to
    materialize. It is a fact about the Capability, so it is declared here with its siblings:

    ``GATHERED``
        The parents' records are concatenated into ONE list (``depends_on`` order). The Capabilities
        whose Provider materializes the input anyway — the screening outcomes, Relation Extraction's
        two-carrier partition, Export's file write — so a lazy stream would move no memory and only
        cost a second pass (ADR-0033's *Honest memory boundary*).
    ``STREAMED``
        The Protocol is ``Iterable``-in and the Provider consumes it exactly ONCE, so the parents'
        checkpoints are read record-by-record into the Provider: the input transient is O(1)
        records rather than O(corpus-slice) (audit R2).
    ``PER_CARRIER``
        The Stage reads MORE than one carrier out of one ``depends_on`` set (Context Enrichment:
        chunks + graph), and its Provider consumes each exactly once in its own order — so one
        ``incoming`` stream cannot serve it and partitioning one would buffer a side whole-corpus.
        It receives one INDEPENDENT lazy stream per carrier instead (dogfood-3).
    ``ACCUMULATOR``
        A corpus-level accumulator (ADR-0033) whose input IS the whole doc-level output: it streams
        each carrier from the parents' checkpoints, so nothing corpus-sized is ever a list. Exactly
        the kinds that declare :attr:`CapabilityDescriptor.accumulates` — a mismatch between the two
        is refused at import (:func:`index_descriptors`).
    """

    GATHERED = "gathered"
    STREAMED = "streamed"
    PER_CARRIER = "per_carrier"
    ACCUMULATOR = "accumulator"


@dataclass(frozen=True, slots=True)
class CarrierRequirement:
    """One "this Stage must be wired to a parent producing X" rule, with the message it raises.

    The wiring checks were ~10 near-identical methods on the Runner, each transcribing its own
    ``ContractError`` text. The rule is (carrier, message) — data — so it lives here beside the
    Capability it constrains, and the message stays byte-identical because there is now exactly one
    copy of it.

    ``message`` is a :meth:`str.format` template over ``{stage}`` (the Stage's ``repr``'d name) and
    ``{got}`` (the comma-joined sorted carrier names actually seen). A template that names no
    ``{got}`` simply does not show one — the eager Relation-Extraction check never did.

    ``subclass_ok`` picks the match: ``False`` tests the carrier is present EXACTLY (the accumulator
    checks' ``X not in present``), ``True`` accepts a subclass (the Context-Enrichment checks'
    ``issubclass`` form, and the ``isinstance`` partition of a gathered list). Preserved per-rule
    rather than unified, because unifying it would silently change which inputs are accepted.
    """

    carrier: type
    message: str
    subclass_ok: bool = False

    def satisfied_by(self, present: frozenset[type]) -> bool:
        """Is this requirement met by the carrier types a Stage's input actually carries?"""
        if self.subclass_ok:
            return any(issubclass(carrier, self.carrier) for carrier in present)
        return self.carrier in present

    def error(self, stage: Stage, got: Iterable[str]) -> ContractError:
        """THE typed wiring error for this unmet requirement — one copy, byte-preserved."""
        return ContractError(
            self.message.format(stage=repr(stage.name), got=", ".join(sorted(got)))
        )


@dataclass(frozen=True, slots=True)
class ExclusiveCarrier:
    """The "this Stage's input must be NOTHING BUT X" rule (Graph Assembly / Graph Completion).

    The mirror image of :class:`CarrierRequirement`: it rejects a carrier rather than requiring one,
    which is why its verdict depends on EVERY type in the input and its carrier scan can never
    short-circuit (see :attr:`StageInputPlan.scan_short_circuit`).

    ``message`` is a :meth:`str.format` template over ``{stage}``, ``{capability}`` and ``{wrong}``
    (the comma-joined sorted names of the carriers that must not be there).
    """

    carrier: type
    message: str

    def violations(self, present: frozenset[type]) -> tuple[str, ...]:
        """The carrier names present that this rule forbids, sorted (empty ⇒ satisfied)."""
        return tuple(sorted(t.__name__ for t in present if t is not self.carrier))

    def error(self, stage: Stage, wrong: Iterable[str]) -> ContractError:
        """THE typed wiring error for a forbidden carrier — one copy, byte-preserved."""
        return ContractError(
            self.message.format(
                stage=repr(stage.name),
                capability=stage.capability.value,
                wrong=", ".join(wrong),
            )
        )


@dataclass(frozen=True, slots=True)
class StageInputPlan:
    """What a Stage of this Capability RECEIVES — shape + wiring rules, as data.

    Answering "what does Graph Assembly receive?" used to mean reading four places: the executor's
    four-way branch, two hand-maintained frozensets the descriptor did not derive, the dispatch's
    second decision of the same shape, and a family of ~10 hand-written validator methods. All four
    are this one field.

    ``scan_short_circuit`` says the carrier scan may stop the moment every required carrier has been
    seen — nothing later in the stream can change a presence verdict. It is DECLARED rather than
    derived from "has no :attr:`exclusive` rule" (which is the soundness condition) because it
    changes how much of the input is READ: the accumulator checks that predate dogfood-3 still pay
    the full pass, and turning that into an early exit is a measurable change to make deliberately,
    not a side effect of moving the table.
    """

    shape: StageInputShape
    requires: tuple[CarrierRequirement, ...] = ()
    exclusive: ExclusiveCarrier | None = None
    scan_short_circuit: bool = False

    @property
    def validates(self) -> bool:
        """Does this plan constrain its input at all? (False ⇒ the wiring check reads nothing.)"""
        return bool(self.requires) or self.exclusive is not None

    @property
    def short_circuit_carriers(self) -> tuple[type, ...]:
        """The carriers a scan may stop at once all are seen — empty unless declared sound."""
        if not self.scan_short_circuit:
            return ()
        return tuple(requirement.carrier for requirement in self.requires)


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Everything the framework must know about one Capability, declared once.

    Adding a Capability = adding one of these. The exhaustiveness check in
    :func:`index_descriptors` makes that mandatory rather than customary.

    Fields, and who consumes each:

    ``kind``
        The :class:`~latence_core.pipeline.CapabilityKind` this describes.
    ``protocol``
        The Capability Protocol a Provider bound to this kind must satisfy — the same check the
        Runner's dispatch enforces at run time and ``stacks.capability_guard`` enforces statically
        on a CPU host (ADR-0036).
    ``carrier``
        The record type a Stage of this kind checkpoints and threads downstream. ``Record`` means a
        MIXED carrier (a fused extractor emits entity *and* relation mentions; Type Consolidation
        re-emits both, relabelled) read back polymorphically. ``None`` means the kind has no carrier
        of its own: ``EXPORT`` is a pass-through sink whose carrier is its nearest producing
        ancestor's (see :func:`export_carrier_of`), and ``EMBEDDING`` is not a DAG node at all.
    ``level``
        :class:`CapabilityLevel` — corpus-level vs document-level *by kind*. Consumed by the
        PHASE-BOUNDARY invariant and by delta reuse.
    ``input_plan``
        :class:`StageInputPlan` — what a Stage of this kind RECEIVES: the shape the executor
        delivers its input in, and the wiring rules its input must satisfy (with the exact typed
        message each raises). Consumed by the Runner's input resolution
        (:class:`~latence_core.stage_inputs.StageInputs`) and by the offline stack check.
    ``accumulates``
        Corpus-level AND its own output is bounded by the knowledge graph rather than by the corpus,
        so the executor materializes its whole input and holds the result (ADR-0033). This is a
        strictly narrower fact than ``level is CORPUS``: an ``EXPORT`` is a sink (nothing downstream
        holds its output) and ``CONTEXT_ENRICHMENT`` re-emits the entire chunk stream (its output is
        corpus-sized, so treating it as an accumulator would defeat the streaming release).
    ``entry_point_prefixes``
        The ``latence.providers`` entry-point name prefixes that name this Capability
        (``entity.gliner`` → ``entity``). A dotted prefix discriminates on the second segment:
        ``screening.intake`` vs ``screening.content`` share a first segment.
    ``conformance_case``
        The value of the conformance suite's ``Capability`` member this kind is checked under.
        Several kinds share one case (a fused extractor is driven through the RELATION case).
    ``quality_section``
        The :class:`~latence_core.quality.QualityReport` field this Capability fills, or ``None``
        when it contributes only per-Stage metrics. ``GRAPH_COMPLETION`` shares ``graph`` — its
        predicted edges are counted separately *inside* ``GraphQuality`` and never merged into the
        asserted-edge statistics (ADR-0037).
    ``dag_node``
        False for ``EMBEDDING``, the one Capability that is not a top-level Stage: it is a nested
        Export sub-config (``config['embedder']``, ADR-0017). A caller that mis-treats it as a Stage
        must fail loudly rather than silently skip a check, so :func:`dag_protocol_for` raises.
    ``corpus_when_config``
        The ``(key, value)`` in a Stage's config that promotes it to corpus level regardless of
        kind. Only ``SCHEMA_INDUCTION`` declares one (``granularity: corpus`` merges every
        document's labels into one salience-capped schema and stamps it onto every chunk — an
        aggregation wherever it sits in the DAG).
    ``materializes_for_export``
        Whether an Export wired (transitively) to a Stage of this kind re-emits THIS kind's carrier.
        False for the kinds that carry records an Export never re-emits: ``SOURCE`` and
        ``INTAKE_SCREENING`` thread raw :class:`~latence_core.contracts.ParserInput`\\ s, so an
        Export above them re-emits what the Parse they must feed produces, and an ``EXPORT`` above
        an ``EXPORT`` resolves through to the real producer.
    """

    kind: CapabilityKind
    protocol: type
    carrier: type | None
    level: CapabilityLevel
    input_plan: StageInputPlan
    entry_point_prefixes: tuple[str, ...]
    conformance_case: str
    quality_section: str | None = None
    dag_node: bool = True
    accumulates: bool = False
    corpus_when_config: tuple[str, str] | None = None
    materializes_for_export: bool = True

    def level_of(self, stage: Stage) -> CapabilityLevel:
        """This Stage's level, honouring the one config-dependent promotion.

        Decided by :attr:`level`, except where :attr:`corpus_when_config` names a config key whose
        value promotes the Stage to :attr:`CapabilityLevel.CORPUS`. Matching is
        case- and whitespace-insensitive; an absent or unrecognised value is NOT treated as corpus
        (the Provider's own default is per-document), so a missing key never fabricates a violation.
        """
        if self.level is CapabilityLevel.CORPUS or self.corpus_when_config is None:
            return self.level
        key, promoting = self.corpus_when_config
        if str(stage.config.get(key, "")).strip().lower() == promoting:
            return CapabilityLevel.CORPUS
        return self.level

    @property
    def mixed_carrier(self) -> bool:
        """Does this kind checkpoint BOTH mention carriers in one file (read polymorphically)?"""
        return self.carrier is Record


# The declaration. One entry per CapabilityKind, in CapabilityKind declaration order so the table
# reads as the pipeline reads. Every rationale that used to be a comment beside one of the eight
# scattered tables is now a comment beside the single field it justifies.
DESCRIPTORS: tuple[CapabilityDescriptor, ...] = (
    CapabilityDescriptor(
        kind=CapabilityKind.SOURCE,
        protocol=Source,
        # Source and Intake Screening thread RAW bytes+metadata, not parsed records.
        carrier=ParserInput,
        level=CapabilityLevel.DOCUMENT,
        # A root: it reads Storage, not upstream records, so the shape is whatever costs nothing.
        input_plan=StageInputPlan(shape=StageInputShape.GATHERED),
        entry_point_prefixes=("source",),
        conformance_case="source",
        # An Export can never re-emit a ParserInput: walk through to the Parse this must feed.
        materializes_for_export=False,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.INTAKE_SCREENING,
        protocol=IntakeScreener,
        carrier=ParserInput,
        level=CapabilityLevel.DOCUMENT,
        # Its Provider returns a ScreeningOutcome holding passed + quarantined, so the input is
        # materialized by the Provider whatever the Runner does.
        input_plan=StageInputPlan(shape=StageInputShape.GATHERED),
        # Both Screening kinds share the ``screening`` first segment, so the prefix is dotted and
        # discriminates on the second (``screening.intake_pdf`` → Intake).
        entry_point_prefixes=("screening.intake",),
        conformance_case="intake_screening",
        quality_section="screening",
        materializes_for_export=False,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.PARSE,
        protocol=Parser,
        carrier=DocumentRecord,
        level=CapabilityLevel.DOCUMENT,
        input_plan=StageInputPlan(shape=StageInputShape.STREAMED),
        # ``parser.*`` and ``parse.*`` both name the Parse Capability (historical drift in the
        # shipped entry-point names, kept for back-compat).
        entry_point_prefixes=("parse", "parser"),
        conformance_case="parse",
        quality_section="parse",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.CHUNK,
        protocol=Chunker,
        carrier=ChunkRecord,
        level=CapabilityLevel.DOCUMENT,
        input_plan=StageInputPlan(shape=StageInputShape.STREAMED),
        entry_point_prefixes=("chunk",),
        conformance_case="chunk",
        quality_section="chunk",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.CONTENT_SCREENING,
        protocol=ContentScreener,
        carrier=ChunkRecord,
        level=CapabilityLevel.DOCUMENT,
        # Like Intake Screening: a ScreeningOutcome (passed + quarantined) the Provider holds.
        input_plan=StageInputPlan(shape=StageInputShape.GATHERED),
        entry_point_prefixes=("screening.content",),
        conformance_case="content_screening",
        quality_section="screening",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.SCHEMA_INDUCTION,
        protocol=LabelInducer,
        # A chunk→chunk transform: the SAME ChunkRecords with ``induced_labels`` populated.
        carrier=ChunkRecord,
        level=CapabilityLevel.DOCUMENT,
        input_plan=StageInputPlan(shape=StageInputShape.STREAMED),
        entry_point_prefixes=("label_inducer",),
        conformance_case="schema_induction",
        # The ONE config-dependent level: ``granularity: corpus`` merges every document's labels
        # into one salience-capped schema and stamps it onto every chunk. Selectable by hand, but
        # it is a corpus-level operation and the phase-boundary check says so.
        corpus_when_config=("granularity", "corpus"),
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.ENTITY_EXTRACTION,
        protocol=EntityExtractor,
        carrier=EntityMention,
        level=CapabilityLevel.DOCUMENT,
        input_plan=StageInputPlan(shape=StageInputShape.STREAMED),
        entry_point_prefixes=("entity",),
        conformance_case="entity",
        quality_section="entities",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.RELATION_EXTRACTION,
        protocol=RelationExtractor,
        carrier=RelationMention,
        level=CapabilityLevel.DOCUMENT,
        # It partitions ONE input into two carriers (chunks for text/page-map context + the
        # mentions it relates) and its Provider indexes the mentions per chunk, so one side must be
        # buffered whatever the Runner does. BOTH sides are required wiring: a missing edge is a
        # typed error, not an empty result. An empty run (no documents) is legitimately empty.
        input_plan=StageInputPlan(
            shape=StageInputShape.GATHERED,
            requires=(
                CarrierRequirement(
                    carrier=EntityMention,
                    subclass_ok=True,
                    message=(
                        "Stage {stage} (relation_extraction) received no EntityMentions. "
                        "A Relation-Extraction Stage must depend on an Entity-Extraction Stage "
                        "(the mentions it relates). Check the Pipeline's depends_on wiring."
                    ),
                ),
                CarrierRequirement(
                    carrier=ChunkRecord,
                    subclass_ok=True,
                    message=(
                        "Stage {stage} (relation_extraction) received no ChunkRecords. "
                        "A Relation-Extraction Stage must also depend on the Chunk (or "
                        "Content-Screening) Stage for the text + page map. Check depends_on."
                    ),
                ),
            ),
        ),
        entry_point_prefixes=("relation",),
        conformance_case="relation",
        quality_section="relations",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.FUSED_ENTITY_RELATION,
        protocol=FusedEntityRelationExtractor,
        # MIXED carrier: one Provider emits entity AND relation mentions in one pass (ADR-0013),
        # so the checkpoint is read polymorphically.
        carrier=Record,
        level=CapabilityLevel.DOCUMENT,
        input_plan=StageInputPlan(shape=StageInputShape.STREAMED),
        entry_point_prefixes=("fused_entity_relation",),
        # Driven through the RELATION conformance case (the fused branch drives extract_fused);
        # ``relation.gliner_relex`` keeps the ``relation`` prefix for back-compat.
        conformance_case="relation",
        quality_section="relations",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.REDACTION,
        protocol=PIIDetector,
        # W16/ADR-0042: a chunk→chunk transform emitting the SAME ChunkRecords with
        # ``masked_content`` + ``pii_spans`` populated — so an Export wired to ``redact`` re-emits
        # ChunkRecords and materializes ``masked_content`` as the PII-safe corpus text.
        carrier=ChunkRecord,
        level=CapabilityLevel.DOCUMENT,
        input_plan=StageInputPlan(shape=StageInputShape.STREAMED),
        entry_point_prefixes=("redaction",),
        conformance_case="redaction",
        quality_section="redaction",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.PROFILING,
        protocol=Profiler,
        carrier=FeatureRecord,
        # Computes corpus statistics — the first operation that needs every document.
        level=CapabilityLevel.CORPUS,
        # Multi-input: a Parse parent for the DocumentRecords (REQUIRED — they are what it
        # profiles) and OPTIONALLY an Entity-Extraction parent for the cross-document entity
        # features. The mention side being optional is why only one rule is declared.
        input_plan=StageInputPlan(
            shape=StageInputShape.ACCUMULATOR,
            requires=(
                CarrierRequirement(
                    carrier=DocumentRecord,
                    message=(
                        "Stage {stage} (profiling) received no DocumentRecords "
                        "(got: {got}). A Profiling Stage must depend on the Parse "
                        "Stage (the documents it profiles). Check the Pipeline's depends_on wiring."
                    ),
                ),
            ),
        ),
        entry_point_prefixes=("profiling",),
        conformance_case="profiling",
        quality_section="profiling",
        accumulates=True,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.TYPE_CONSOLIDATION,
        protocol=TypeConsolidator,
        # T3/ADR-0054: a mention→mention transform re-emitting BOTH carriers relabelled, so it
        # checkpoints the same MIXED carrier a fused extractor does.
        carrier=Record,
        # Corpus-level BY CONSTRUCTION: a canonical type vocabulary cannot be elected from a prefix
        # of the labels, only from all of them. THIS is the fact that was missing from
        # phase_boundary — declaring it here is what closes the hole for every consumer at once.
        level=CapabilityLevel.CORPUS,
        # The mentions' labels ARE the vocabulary it canonicalizes, so the mention side is
        # REQUIRED; the relation side is optional (an entity-only stack is valid). A silent
        # pass-through would leave the KG's types drifted while the report claimed consolidation
        # ran (ADR-0034).
        input_plan=StageInputPlan(
            shape=StageInputShape.ACCUMULATOR,
            requires=(
                CarrierRequirement(
                    carrier=EntityMention,
                    message=(
                        "Stage {stage} (type_consolidation) received no EntityMentions "
                        "(got: {got}). A Type-Consolidation Stage must depend on an "
                        "Entity-Extraction (or fused) Stage (the mentions whose types it "
                        "canonicalizes). Check the Pipeline's depends_on wiring."
                    ),
                ),
            ),
        ),
        entry_point_prefixes=("type_consolidation",),
        conformance_case="type_consolidation",
        quality_section="type_consolidation",
        # Its own output is bounded by the mention set it re-emits 1:1, not by corpus text.
        accumulates=True,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.DISAMBIGUATION,
        protocol=Disambiguator,
        carrier=DisambiguationRecord,
        # Resolves mentions ACROSS documents.
        level=CapabilityLevel.CORPUS,
        # The mentions it resolves are REQUIRED (unlike Profiling's optional mentions: with no
        # mentions there is nothing to disambiguate); the relation side is optional.
        input_plan=StageInputPlan(
            shape=StageInputShape.ACCUMULATOR,
            requires=(
                CarrierRequirement(
                    carrier=EntityMention,
                    message=(
                        "Stage {stage} (disambiguation) received no EntityMentions "
                        "(got: {got}). A Disambiguation Stage must depend on an "
                        "Entity-Extraction (or fused) Stage (the mentions it resolves). Check "
                        "the Pipeline's depends_on wiring."
                    ),
                ),
            ),
        ),
        entry_point_prefixes=("disambiguation",),
        conformance_case="disambiguation",
        quality_section="disambiguation",
        accumulates=True,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.GRAPH_ASSEMBLY,
        protocol=GraphAssembler,
        carrier=GraphRecord,
        level=CapabilityLevel.CORPUS,
        # An EXCLUSIVE rule, not a required one: its input must be entirely DisambiguationRecords
        # (the canonical entities + normalized relations it turns into nodes + edges), so ANY other
        # carrier is the wiring error — which is why its scan can never short-circuit. The message
        # is the one the eager ``_as_type`` narrowing raised, preserved exactly.
        input_plan=StageInputPlan(
            shape=StageInputShape.ACCUMULATOR,
            exclusive=ExclusiveCarrier(
                carrier=DisambiguationRecord,
                message=(
                    "Stage {stage} ({capability}) requires "
                    "DisambiguationRecord inputs but received: {wrong}. "
                    "Check the Pipeline's depends_on wiring."
                ),
            ),
        ),
        entry_point_prefixes=("graph",),
        conformance_case="graph",
        quality_section="graph",
        accumulates=True,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.GRAPH_COMPLETION,
        protocol=GraphCompleter,
        # W2-linkpred: PREDICTED EDGE-scope GraphRecords marked ``properties.inferred = True`` —
        # the same carrier Graph Assembly emits, never merged into the asserted edges.
        carrier=GraphRecord,
        level=CapabilityLevel.CORPUS,
        # Exclusive like Graph Assembly: it consumes the assembled graph's GraphRecords (NODE +
        # EDGE), so any other carrier means it was wired to the wrong parent.
        input_plan=StageInputPlan(
            shape=StageInputShape.ACCUMULATOR,
            exclusive=ExclusiveCarrier(
                carrier=GraphRecord,
                message=(
                    "Stage {stage} (graph_completion) requires GraphRecord inputs "
                    "but received: {wrong}. A Graph Completion Stage must depend on "
                    "the Graph Assembly Stage (the assembled nodes + edges it completes). "
                    "Check the Pipeline's depends_on wiring."
                ),
            ),
        ),
        entry_point_prefixes=("graph_completion",),
        conformance_case="graph_completion",
        # Counted SEPARATELY inside GraphQuality (ADR-0037), not in a section of its own.
        quality_section="graph",
        accumulates=True,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.CONTEXT_ENRICHMENT,
        protocol=ContextEnricher,
        # W13/ADR-0039: a chunk→chunk transform emitting the SAME ChunkRecords with
        # ``context_header`` populated; the stored chunk text stays byte-unchanged.
        carrier=ChunkRecord,
        # Corpus-level: a chunk's context header summarizes the assembled KG, so it cannot be
        # computed before the graph exists — and a reused document's header can go stale.
        level=CapabilityLevel.CORPUS,
        # The MULTI-CARRIER Stage: it depends on the chunk stream (Chunk / Content-Screening /
        # Schema-Induction) AND Graph Assembly, and its Provider consumes each of its inputs
        # exactly once in its own order — so it gets one independent lazy stream per carrier
        # rather than one gathered list to partition. Both parent-side carriers are required
        # wiring; the run's EntityMentions come corpus-wide (not from a parent) and so are not
        # checked here. The presence scan may stop as soon as both are seen — the correct case
        # never reads past the first record of the second carrier.
        input_plan=StageInputPlan(
            shape=StageInputShape.PER_CARRIER,
            requires=(
                CarrierRequirement(
                    carrier=ChunkRecord,
                    subclass_ok=True,
                    message=(
                        "Stage {stage} (context_enrichment) received no ChunkRecords "
                        "(got: {got}). A Context-Enrichment Stage must depend on the Chunk "
                        "(or Content-Screening / Schema-Induction) Stage — the chunks it enriches. "
                        "Check the Pipeline's depends_on wiring."
                    ),
                ),
                CarrierRequirement(
                    carrier=GraphRecord,
                    subclass_ok=True,
                    message=(
                        "Stage {stage} (context_enrichment) received no GraphRecords "
                        "(got: {got}). A Context-Enrichment Stage must also depend on the "
                        "Graph Assembly Stage — the assembled KG it projects onto each chunk. "
                        "Check the Pipeline's depends_on wiring."
                    ),
                ),
            ),
            scan_short_circuit=True,
        ),
        entry_point_prefixes=("context",),
        conformance_case="context_enrichment",
        quality_section="context_enrichment",
        # NOT an accumulator: its output is the whole chunk stream (corpus-sized), so the executor
        # streams and releases it like any other chunk→chunk transform (ADR-0033).
        accumulates=False,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.EMBEDDING,
        protocol=Embedder,
        # Not a DAG node: an Export names its Embedder in ``config['embedder']`` (ADR-0017), so
        # there is no Stage, no checkpoint and no carrier of its own.
        carrier=None,
        level=CapabilityLevel.DOCUMENT,
        # Not a DAG node, so nothing ever delivers it an input; a Stage that declares it anyway is
        # refused by the dispatch. Declared GATHERED because that is the shape the executor's
        # (uniform) input resolution takes for it before the refusal.
        input_plan=StageInputPlan(shape=StageInputShape.GATHERED),
        entry_point_prefixes=("embedding",),
        conformance_case="embedding",
        dag_node=False,
        materializes_for_export=False,
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.EXPORT,
        protocol=Export,
        # A pass-through sink: its carrier is whatever its nearest producing ancestor emits, so it
        # has none of its own (see :func:`export_carrier_of`).
        carrier=None,
        # Materializes the whole corpus (and, with BM25, its term statistics).
        level=CapabilityLevel.CORPUS,
        # It writes a file and holds what it materializes (streamed internally in row groups), so
        # a gathered list is what it needs — and the W17 reuse overlay is merged into that list.
        input_plan=StageInputPlan(shape=StageInputShape.GATHERED),
        entry_point_prefixes=("export",),
        conformance_case="export",
        # A sink: nothing downstream holds its output, so the executor need not materialize it.
        accumulates=False,
        materializes_for_export=False,
    ),
)


def index_descriptors(
    descriptors: Sequence[CapabilityDescriptor],
    kinds: Iterable[CapabilityKind],
) -> Mapping[CapabilityKind, CapabilityDescriptor]:
    """Index the descriptors by kind, refusing anything but an EXACT cover of ``kinds``.

    This is the mechanism that makes the drift unrepresentable, and it runs at **import** time:
    add a :class:`~latence_core.pipeline.CapabilityKind` without declaring its facts and the first
    ``import latence_core`` raises, rather than the new Capability silently answering ``False`` to
    ``is_corpus_level`` (which is exactly how ``TYPE_CONSOLIDATION`` slipped past the
    PHASE-BOUNDARY invariant).

    Raises :class:`~latence_core.errors.ContractError` — typed, never a bare assert (ADR-0034) —
    for a kind with no descriptor, a descriptor for an unknown kind, a duplicate, or an
    :attr:`CapabilityDescriptor.input_plan` whose shape disagrees with
    :attr:`CapabilityDescriptor.accumulates` (the executor shapes memory off the second and
    delivers the input off the first — they name the same fact and cannot be allowed to drift).
    Errors name every offending kind at once and in declaration order, so one run tells the whole
    story.
    """
    by_kind: dict[CapabilityKind, CapabilityDescriptor] = {}
    duplicates: list[str] = []
    for descriptor in descriptors:
        if descriptor.kind in by_kind:
            duplicates.append(str(descriptor.kind))
            continue
        by_kind[descriptor.kind] = descriptor
    declared = tuple(kinds)
    missing = [str(kind) for kind in declared if kind not in by_kind]
    unknown = sorted(str(kind) for kind in by_kind if kind not in set(declared))
    problems: list[str] = []
    if missing:
        problems.append(
            f"no CapabilityDescriptor for {', '.join(missing)} — every CapabilityKind must "
            f"declare its Protocol, carrier, level, entry-point prefix, conformance case and "
            f"Quality section in latence_core.capability_descriptor.DESCRIPTORS"
        )
    if unknown:
        problems.append(f"CapabilityDescriptor for unknown CapabilityKind(s) {', '.join(unknown)}")
    if duplicates:
        problems.append(f"duplicate CapabilityDescriptor for {', '.join(duplicates)}")
    inconsistent = [
        str(kind)
        for kind, descriptor in by_kind.items()
        if (descriptor.input_plan.shape is StageInputShape.ACCUMULATOR) != descriptor.accumulates
    ]
    if inconsistent:
        problems.append(
            f"input_plan shape disagrees with 'accumulates' for {', '.join(inconsistent)} — a "
            f"Capability whose executor materializes its whole input (accumulates) is exactly one "
            f"whose input_plan shape is ACCUMULATOR"
        )
    if problems:
        raise ContractError("; ".join(problems))
    return by_kind


#: The exhaustiveness gate, evaluated at import. See :func:`index_descriptors`.
_BY_KIND: Mapping[CapabilityKind, CapabilityDescriptor] = index_descriptors(
    DESCRIPTORS, CapabilityKind
)


def descriptor_for(kind: CapabilityKind) -> CapabilityDescriptor:
    """The descriptor for ``kind``. Total over :class:`~latence_core.pipeline.CapabilityKind`."""
    return _BY_KIND[kind]


def descriptors() -> tuple[CapabilityDescriptor, ...]:
    """Every descriptor, in :class:`~latence_core.pipeline.CapabilityKind` declaration order."""
    return DESCRIPTORS


def input_plan_for(kind: CapabilityKind) -> StageInputPlan:
    """The :class:`StageInputPlan` for ``kind`` — what a Stage of it receives, and what it requires.

    Total over :class:`~latence_core.pipeline.CapabilityKind`. This is the whole answer to "what
    does this Stage get?": the Runner's input resolution
    (:class:`~latence_core.stage_inputs.StageInputs`) reads the shape, the wiring check reads the
    rules, and the offline stack check reads the same rules statically.
    """
    return _BY_KIND[kind].input_plan


def dag_protocol_for(kind: CapabilityKind) -> type:
    """The Protocol a DAG **Stage** of ``kind`` requires its Provider to satisfy.

    Raises ``KeyError`` for a kind that is not a top-level DAG node (``EMBEDDING`` — a nested Export
    sub-config, not a Stage), so a caller that mis-treats it as a Stage fails loudly rather than
    silently skipping the check.
    """
    descriptor = _BY_KIND[kind]
    if not descriptor.dag_node:
        msg = (
            f"{kind} is not a top-level DAG Stage (it is a nested Export sub-config); "
            f"check its Provider against its Protocol on the nested path instead"
        )
        raise KeyError(msg)
    return descriptor.protocol


def is_corpus_level(stage: Stage) -> bool:
    """Whether this Stage performs a CORPUS-LEVEL operation (aggregates across documents)."""
    return descriptor_for(stage.capability).level_of(stage) is CapabilityLevel.CORPUS


def export_carrier_of(kind: CapabilityKind) -> type | None:
    """The carrier an Export re-emits when ``kind`` is its nearest ancestor, or ``None`` to walk on.

    ``None`` does not mean "no carrier" — it means *this kind is not the producer an Export
    re-emits*, so the resolution continues up ``depends_on`` (a ``SOURCE``/``INTAKE_SCREENING``
    ancestor threads raw ParserInputs, and an ``EXPORT`` above an ``EXPORT`` resolves through to the
    real producer). A walk that reaches the top with no producer falls back to ``DocumentRecord``.
    """
    descriptor = _BY_KIND[kind]
    return descriptor.carrier if descriptor.materializes_for_export else None


def kind_for_entry_point(provider_name: str) -> CapabilityKind:
    """The Capability a Provider's registered ``<capability>.<variant>`` entry-point name declares.

    The prefix before the first ``.`` names the Capability, except where a dotted prefix is declared
    to discriminate on the second segment (``screening.intake_*`` → Intake Screening,
    ``screening.content_*`` → Content Screening — both share the ``screening`` first segment).

    Raises ``KeyError`` for a name whose prefix names no Capability, and for a name whose first
    segment IS declared but whose second segment matches no discriminator: a Provider that cannot be
    classified is a loud failure, never a silent skip.
    """
    prefix, _, rest = provider_name.partition(".")
    kind = _SIMPLE_PREFIXES.get(prefix)
    if kind is not None:
        return kind
    discriminators = _DOTTED_PREFIXES.get(prefix)
    if discriminators is not None:
        for second, dotted_kind in discriminators:
            if rest.startswith(second):
                return dotted_kind
        expected = ", ".join(f"{second}*" for second, _ in discriminators)
        msg = f"Provider {provider_name!r} has prefix {prefix!r} but is none of: {expected}."
        raise KeyError(msg)
    msg = (
        f"Provider {provider_name!r} has an unknown Capability prefix {prefix!r}; declare it in "
        f"latence_core.capability_descriptor.DESCRIPTORS"
    )
    raise KeyError(msg)


def _split_prefixes() -> tuple[
    dict[str, CapabilityKind], dict[str, tuple[tuple[str, CapabilityKind], ...]]
]:
    """Partition the declared entry-point prefixes into plain and second-segment-discriminated."""
    simple: dict[str, CapabilityKind] = {}
    dotted: dict[str, list[tuple[str, CapabilityKind]]] = {}
    for descriptor in DESCRIPTORS:
        for prefix in descriptor.entry_point_prefixes:
            head, _, second = prefix.partition(".")
            if second:
                dotted.setdefault(head, []).append((second, descriptor.kind))
            else:
                simple[head] = descriptor.kind
    return simple, {head: tuple(pairs) for head, pairs in dotted.items()}


_SIMPLE_PREFIXES, _DOTTED_PREFIXES = _split_prefixes()

#: Every first segment any Capability claims — the set a caller tests before deciding a Provider
#: name belongs to some other (non-Stage) seam entirely.
ENTRY_POINT_PREFIX_HEADS: frozenset[str] = frozenset(_SIMPLE_PREFIXES) | frozenset(_DOTTED_PREFIXES)

#: Corpus-level BY KIND — the back half. Derived, so it cannot disagree with the descriptors.
#: (``SCHEMA_INDUCTION`` can join it per-Stage via config; use :func:`is_corpus_level` for a Stage.)
CORPUS_LEVEL_KINDS: frozenset[CapabilityKind] = frozenset(
    d.kind for d in DESCRIPTORS if d.dag_node and d.level is CapabilityLevel.CORPUS
)

#: Document/chunk-level BY KIND — the front half. The exact COMPLEMENT of
#: :data:`CORPUS_LEVEL_KINDS` over the DAG kinds, so the two can never drift apart or leave a kind
#: in neither (the failure mode that let a corpus-collapsing Stage pass the phase-boundary check).
DOCUMENT_LEVEL_KINDS: frozenset[CapabilityKind] = frozenset(
    d.kind for d in DESCRIPTORS if d.dag_node and d.level is CapabilityLevel.DOCUMENT
)

#: The corpus-level ACCUMULATORS (ADR-0033): they materialize their whole input, but their own
#: output is bounded by the knowledge graph rather than by the corpus.
ACCUMULATOR_KINDS: frozenset[CapabilityKind] = frozenset(
    d.kind for d in DESCRIPTORS if d.accumulates
)

#: The doc-level Capabilities whose INPUT the Runner streams (ADR-0033, audit R2): the Protocol is
#: ``Iterable``-in and the Provider consumes it exactly once, so the input transient is O(1)
#: records rather than O(corpus-slice). Derived from the declared :class:`StageInputPlan`, so it
#: cannot drift from the shape the executor actually delivers (it was a hand-maintained frozenset
#: in ``runner.py`` that the descriptor did not derive — the last of the scattered tables).
STREAMED_INPUT_KINDS: frozenset[CapabilityKind] = frozenset(
    d.kind for d in DESCRIPTORS if d.input_plan.shape is StageInputShape.STREAMED
)

#: The MULTI-CARRIER Capabilities whose input the Runner streams PER CARRIER (ADR-0033,
#: dogfood-3): they receive nothing through one ``incoming`` stream and open one independent lazy
#: stream per carrier instead. Derived, like :data:`STREAMED_INPUT_KINDS`.
PER_CARRIER_INPUT_KINDS: frozenset[CapabilityKind] = frozenset(
    d.kind for d in DESCRIPTORS if d.input_plan.shape is StageInputShape.PER_CARRIER
)

__all__ = [
    "ACCUMULATOR_KINDS",
    "CORPUS_LEVEL_KINDS",
    "DESCRIPTORS",
    "DOCUMENT_LEVEL_KINDS",
    "ENTRY_POINT_PREFIX_HEADS",
    "PER_CARRIER_INPUT_KINDS",
    "STREAMED_INPUT_KINDS",
    "CapabilityDescriptor",
    "CapabilityLevel",
    "CarrierRequirement",
    "ExclusiveCarrier",
    "StageInputPlan",
    "StageInputShape",
    "dag_protocol_for",
    "descriptor_for",
    "descriptors",
    "export_carrier_of",
    "index_descriptors",
    "input_plan_for",
    "is_corpus_level",
    "kind_for_entry_point",
]
