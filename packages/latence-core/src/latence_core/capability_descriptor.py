"""One frozen :class:`CapabilityDescriptor` per Capability — every fact about a Capability, once.

A Capability is not just a Protocol. Adding one means declaring, somewhere, **eight** separate
facts about it: the Protocol its Provider must satisfy, the record carrier it checkpoints, the
carrier an Export downstream of it re-emits, whether it is a corpus-level accumulator for the
executor's memory shaping, whether it is corpus-level for the PHASE-BOUNDARY invariant, whether it
is corpus-level for delta reuse, its Provider entry-point name prefix, and which Quality Report
section it fills. Before this module each of those lived in its own hand-ordered table in a
different file, and *nothing* related them — so adding a Capability meant finding eight tables by
memory, and forgetting one was silent.

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
        entry_point_prefixes=("chunk",),
        conformance_case="chunk",
        quality_section="chunk",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.CONTENT_SCREENING,
        protocol=ContentScreener,
        carrier=ChunkRecord,
        level=CapabilityLevel.DOCUMENT,
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
        entry_point_prefixes=("entity",),
        conformance_case="entity",
        quality_section="entities",
    ),
    CapabilityDescriptor(
        kind=CapabilityKind.RELATION_EXTRACTION,
        protocol=RelationExtractor,
        carrier=RelationMention,
        level=CapabilityLevel.DOCUMENT,
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
    for a kind with no descriptor, a descriptor for an unknown kind, or a duplicate. Errors name
    every offending kind at once and in declaration order, so one run tells the whole story.
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

__all__ = [
    "ACCUMULATOR_KINDS",
    "CORPUS_LEVEL_KINDS",
    "DESCRIPTORS",
    "DOCUMENT_LEVEL_KINDS",
    "ENTRY_POINT_PREFIX_HEADS",
    "CapabilityDescriptor",
    "CapabilityLevel",
    "dag_protocol_for",
    "descriptor_for",
    "descriptors",
    "export_carrier_of",
    "index_descriptors",
    "is_corpus_level",
    "kind_for_entry_point",
]
