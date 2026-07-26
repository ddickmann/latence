"""Capability protocols and the entry-point Provider registry (ADR-0004).

A Capability is a narrow, model-agnostic interface a Stage depends on. A Provider
is a concrete implementation, registered as a plugin under the ``latence.providers``
entry-point group and discovered dynamically via ``importlib.metadata`` — the
framework never names a model, only Capabilities.

For S1 the Capabilities are the three spine Stages (Source, Parser, Export). Each
is a ``typing.Protocol`` so any object with the right shape satisfies it — no
base class or import of core required for a third-party Provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from latence_core.providers.profile import ProviderProfile, profile_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from latence_core.contracts import (
        ChunkRecord,
        Delta,
        DisambiguationRecord,
        DocumentRecord,
        EntityMention,
        FeatureRecord,
        GraphEdge,
        GraphNode,
        GraphRecord,
        ParserInput,
        QuarantineRecord,
        Record,
        RelationMention,
        ScreeningFinding,
        TypeVocabulary,
    )
    from latence_core.storage import Storage

ENTRY_POINT_GROUP = "latence.providers"

# The carrier type a Screening checkpoint operates on: ParserInput at intake,
# ChunkRecord at content. ``ScreeningOutcome`` is generic over it.
_Screened = TypeVar("_Screened")


@dataclass(frozen=True)
class ScreeningOutcome(Generic[_Screened]):
    """The result of a Screening checkpoint: what passed, what was Quarantined.

    ``passed`` are the records that proceed downstream — for Content Screening
    these may carry newly-attached risk markers. ``quarantined`` are the
    :class:`~latence_core.contracts.QuarantineRecord`s removed from the pipeline
    (retained for audit, never exported to the corpus). ``findings`` is the full
    per-record decision log Screening emits into the Quality Report — every
    Quarantine reason and every flag, so the disposition is auditable (S3 AC).
    """

    passed: list[_Screened] = field(default_factory=list)
    quarantined: list[QuarantineRecord] = field(default_factory=list)
    findings: list[ScreeningFinding] = field(default_factory=list)


@runtime_checkable
class Source(Protocol):
    """Lists and fetches documents from where they live, stamping Provenance."""

    def produce(self, storage: Storage) -> Iterator[ParserInput]:
        """Yield one raw document per source document, with initial Provenance.

        The yielded :class:`~latence_core.contracts.ParserInput` carries the
        undecoded source ``content`` (bytes) — decoding and the turn into
        markdown belong to the Parser (ADR-0019), not the Source.
        """
        ...


@runtime_checkable
class DocumentEnumerator(Protocol):
    """OPTIONAL Source refinement: enumerate document ids WITHOUT fetching content.

    A delta run has to know which documents the source currently holds *before* it can decide
    which of them are unchanged and can therefore be skipped (ADR-0018/0043). Deriving that set
    from :meth:`Source.produce` is correct but pays the full fetch cost for every document — for
    a remote Source (a SharePoint library, an object store) that means the whole corpus crosses
    the network on every delta, which is precisely the cost the delta exists to avoid, and it
    happens *before* the exclusions are known, so no ``exclude_document_ids`` can prevent it.

    A Source that can name its documents more cheaply than it can fetch them implements this:
    SharePoint reads the content hash Graph reports in the *listing*, so a delta run over an
    unchanged library transfers zero bytes. The ids MUST be the same content-addressed ids
    :meth:`Source.produce` stamps on ``Provenance.document_id`` — an id that disagreed would
    silently mis-classify documents as new. A Source that cannot do better simply does not
    implement it, and the runner falls back to ``produce`` (correct, just not free).

    The enumeration MAY be **partial**, and a Source that can only name *some* of its documents
    cheaply MUST leave the rest out rather than fetch them: an omitted id cannot intersect the
    parent Version, so its document is treated as new and fetched exactly once by ``produce``,
    whereas fetching it here transfers it unconditionally *in addition to* that — which is how a
    no-op delta ends up costing twice a full run (audit R3-connectors_ingestion-8). Omitting can
    only lose a reuse; it can never invent one, and it never changes a document's records.
    """

    def document_ids(self, storage: Storage) -> Iterator[str]:
        """Yield the ``Provenance.document_id`` of every document currently in the source."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Turns raw source documents into parsed markdown records."""

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[DocumentRecord]:
        """Consume raw :class:`ParserInput`s and yield parsed markdown records."""
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits parsed documents into retrieval-sized ChunkRecords (CONTEXT ``Chunk``).

    Preserves offsets, page alignment, Provenance and Classification on every
    emitted chunk. A ``PARSE_ERROR`` DocumentRecord (no usable content) yields no
    chunks.
    """

    def chunk(self, documents: Iterable[DocumentRecord]) -> Iterator[ChunkRecord]:
        """Consume parsed :class:`DocumentRecord`s and yield :class:`ChunkRecord`s."""
        ...


@runtime_checkable
class IntakeScreener(Protocol):
    """Screens raw documents BEFORE Parse (CONTEXT ``Screening`` — intake checkpoint).

    Catches malware, zip bombs, file-type spoofing, oversized/corrupt files. A
    dangerous document is Quarantined (removed from every downstream Stage); a
    clean one passes through untouched.
    """

    def screen_intake(self, inputs: Iterable[ParserInput]) -> ScreeningOutcome[ParserInput]:
        """Partition raw inputs into passed vs quarantined, with findings."""
        ...


@runtime_checkable
class ContentScreener(Protocol):
    """Screens chunked content AFTER Chunk (CONTEXT ``Screening`` — content checkpoint).

    Catches prompt injection, harmful content, sensitivity escalation. A flagged
    chunk keeps flowing but carries a :class:`~latence_core.contracts.RiskMarker`
    that survives into the corpus; a chunk deemed dangerous is Quarantined.
    """

    def screen_content(self, chunks: Iterable[ChunkRecord]) -> ScreeningOutcome[ChunkRecord]:
        """Return chunks (some marked) plus any quarantined chunks and findings."""
        ...


@runtime_checkable
class LabelInducer(Protocol):
    """Induces a document's own label schema and annotates its chunks (CONTEXT ``Classification``).

    The W4 **schema-induction** Capability (ADR-0038): an OPTIONAL Stage inserted right after
    Content Screening and before the extraction Stages. It restores the "point it at a folder and
    it discovers its own schema" property — a small OpenAI-compatible LLM with structured JSON
    output reads the document text and induces the entity / relation / PII **types** actually
    evidenced in it, then broadcasts that :class:`~latence_core.contracts.InducedLabels` schema
    onto every :class:`~latence_core.contracts.ChunkRecord` of the document. The extraction Stages
    then UNION the induced types onto their config labels (the ``latence_core.induced`` seam), so a
    Stage with no config labels becomes fully unsupervised and one with config labels is
    supervised-plus-augmented — without any extractor rewrite.

    It is a **chunk→chunk transform** (the Content-Screening shape): it consumes the run's chunks
    and yields the SAME chunks with ``induced_labels`` populated (a doc-level Provider groups chunks
    by ``document_record_id`` and makes one LLM call per document by default, broadcasting the doc's
    induced labels to all its chunks). The Capability names no LLM and no label set — the Provider
    supplies both. It is **fail-open (G1 posture)**: any inducer failure (network, malformed output,
    truncation, schema-invalid, empty) leaves a chunk's ``induced_labels`` ``None`` so the extractor
    falls back to its config labels and the run never crashes — the untrusted-LLM discipline lives
    entirely in the Provider. A chunk it could not annotate simply flows through unchanged.
    """

    def induce(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Consume the run's chunks; yield the same chunks with ``induced_labels`` set.

        The stream is threaded through 1:1 (never dropped or reordered): a chunk the Provider could
        induce labels for carries a populated :class:`~latence_core.contracts.InducedLabels`; a
        chunk whose induction failed (fail-open) flows through with ``induced_labels`` still
        ``None``. Deterministic + seeded (temperature 0, stable label order) so a run is
        byte-reproducible (Baseline bar).
        """
        ...


@runtime_checkable
class EntityExtractor(Protocol):
    """Finds typed entity mentions in chunked text (CONTEXT ``Entity Extraction``).

    Zero-shot NER: the label set is configured per Pipeline (Stage ``config``), not
    hardcoded — the Capability names no types. Each emitted
    :class:`~latence_core.contracts.EntityMention` carries its label, confidence,
    and the char span it occupies **in the parent document's assembled markdown**,
    so a mention resolves back through the parent's page map to its source page(s).
    An empty or near-empty chunk yields no mentions (never an error).
    """

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        """Consume :class:`ChunkRecord`s and yield typed :class:`EntityMention`s."""
        ...


@runtime_checkable
class RelationExtractor(Protocol):
    """Finds typed relations between entity mentions (CONTEXT ``Relation Extraction``).

    Relation Extraction relates *mentions within a document* (ADR-0022): given the
    document's :class:`~latence_core.contracts.EntityMention`s plus the chunk text
    that supplies the surrounding context, it yields directed, typed
    :class:`~latence_core.contracts.RelationMention`s (head → tail). The relation
    label set is configured per Pipeline (Stage ``config``), not hardcoded — the
    Capability names no relation types (zero-shot / prompt-driven), mirroring
    Entity Extraction. Each emitted relation carries the head/tail mention refs, its
    label, confidence, and a Provenance span covering both endpoints, so it resolves
    back through the parent's page map to its source page(s). A document with fewer
    than two mentions, or no relatable pair, yields no relations (never an error).
    """

    def relate(
        self, chunks: Iterable[ChunkRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[RelationMention]:
        """Consume a run's chunks + mentions and yield typed :class:`RelationMention`s.

        Both carriers are supplied: ``mentions`` are what a relation connects, and
        ``chunks`` carry the text (and the parent page map) a text-based Provider
        needs for context and offset→page resolution. The Provider groups both by
        ``document_record_id`` internally — a relation only ever links two mentions
        of the *same* document.
        """
        ...


@dataclass(frozen=True)
class FusedExtraction:
    """The joint output of a :class:`FusedEntityRelationExtractor` — one pass, both Stages.

    A Fused Provider (ADR-0013: gliner-relex, the custom span-predictor) fulfils
    Entity Extraction AND Relation Extraction in a single pass over the chunks, so
    it returns both carriers together: the ``mentions`` it found and the
    ``relations`` between them. The Runner threads both downstream from the one
    fused Stage — S4's mentions and S5's relations — without invoking a separate
    Entity-Extraction Stage (no double-extraction).
    """

    mentions: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)


@runtime_checkable
class FusedEntityRelationExtractor(Protocol):
    """One Provider that fulfils BOTH Entity + Relation Extraction in one pass (ADR-0013).

    The Pipeline does not force one Provider per Stage: a fused model (gliner-relex,
    the custom span-predictor) does joint NER + RE, so a single fused Stage produces
    both the :class:`~latence_core.contracts.EntityMention`s and the
    :class:`~latence_core.contracts.RelationMention`s. The Runner routes a fused
    Stage's :class:`FusedExtraction` to both downstream consumers without running a
    separate Entity-Extraction Stage — the S5 seam requirement.
    """

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Consume :class:`ChunkRecord`s and return mentions + relations together."""
        ...


@runtime_checkable
class PIIDetector(Protocol):
    """Detects PII per chunk and produces a masked variant (CONTEXT ``Redaction``; W16, ADR-0042).

    The S6 Redaction Capability, made **chunk-level** in W16 to fix the document-level truncation
    leak. Before W16 a Provider scanned the WHOLE parsed
    :class:`~latence_core.contracts.DocumentRecord`, feeding it into a fixed PII model window (768
    tokens for the gliner-family) — so any PII past that window was SILENTLY truncated and leaked
    unmasked into the corpus. Redaction is now a **chunk→chunk transform** (the Content-Screening /
    Context-Enrichment shape): a Provider consumes the run's
    :class:`~latence_core.contracts.ChunkRecord`s and yields the SAME chunks with
    :attr:`~latence_core.contracts.ChunkRecord.masked_content` +
    :attr:`~latence_core.contracts.ChunkRecord.pii_spans` populated. Each chunk's text (≤ the model
    window) is PII-scanned + masked in full — nothing is truncated, so PII past the old 768-token
    document-truncation point is now caught (the ADR-0042 fix).

    The chunk's ``content`` stays UNMASKED (so a later re-read / audit sees the full entities); the
    additive ``masked_content`` is the PII-safe variant the Export materializes for the RAG corpus,
    and ``pii_spans`` are the chunk-local, counts-only detected spans (offsets into the chunk
    ``content``, source pages resolved through the chunk's offset_map+page_map — never the raw
    value, S6 AC). Redaction is metadata-aware: the Provider keys its policy off each chunk's
    ``Classification.sensitivity`` (S6 AC). The PII type set is Provider/config-supplied — the
    Capability names no PII types. A chunk with no detected PII yields a chunk whose
    ``masked_content`` equals its ``content`` and whose ``pii_spans`` is empty (never an error).

    The shared policy/masking/no-op-floor/page-resolution machinery lives in
    :mod:`latence_core.redaction_policy` (``plan_chunk_redaction`` + ``finalize_redacted_chunk``),
    so a Provider owns ONLY detection. Because Redaction reads the chunk's markup-stripped
    ``content``, a span's chunk-local offset resolves to its ORIGINAL source page through the
    chunk's offset_map + the document-scoped
    :class:`~latence_core.page_index.PageIndexResolver` — the coordinate-correctness ADR-0031/0042
    guarantee. A Provider builds ONE resolver per ``redact`` call and hands it to every
    ``plan_chunk_redaction`` (it is a required argument): since schema v18 a document's page map
    rides its FIRST chunk only. Redaction therefore depends on the Chunk Stage (not Parse).

    The document-level :meth:`redact` shape (``Iterable[DocumentRecord] ->
    Iterator[RedactionRecord]``) is DEPRECATED; a Provider MAY retain it as a secondary
    ``redact_documents`` method for the superseded path, but the blessed Capability is the chunk
    seam below.
    """

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Consume the run's :class:`ChunkRecord`s; yield the SAME chunks with PII masked.

        The chunk stream is threaded 1:1 and order-stable (never dropped or reordered): each yielded
        chunk carries ``masked_content`` (its text with every non-``TAG`` PII span replaced) +
        ``pii_spans`` (the chunk-local detected spans) + ``redaction_disabled_for_sensitive`` (the
        H-C1 §4 no-op floor). ``content`` is byte-UNCHANGED. Chunks are emitted in first-seen order
        for determinism; a seeded run is byte-reproducible (Baseline bar).
        """
        ...


@runtime_checkable
class Profiler(Protocol):
    """Computes corpus-level statistical + quality features (CONTEXT ``Profiling``).

    The S7 Profiling Capability. Profiling is a **corpus-level** Stage: it consumes the
    whole run's parsed :class:`~latence_core.contracts.DocumentRecord`s (and, when an
    Entity-Extraction Stage is upstream, the run's
    :class:`~latence_core.contracts.EntityMention`s) and emits
    :class:`~latence_core.contracts.FeatureRecord`s — one ``DOCUMENT``-scope record per
    parsed document carrying its per-document features (density, readability, Zipf α,
    compression ratio, structure) and exactly one ``CORPUS``-scope record carrying the
    cross-document aggregate features (entity frequency, co-occurrence, type consensus,
    source coverage).

    Both carriers are supplied so a fused corpus profile can relate document text to the
    entities extracted from it; a Provider that only wants document text ignores the
    mentions. The feature pass must be **streaming / spill-to-disk friendly** — it must
    not assume the whole corpus fits in RAM (S7 AC) — so a Provider folds documents in
    one at a time rather than materialising the corpus. A ``PARSE_ERROR`` document (no
    usable content) is skipped, mirroring the Chunk Stage. Records are emitted
    deterministically (document-scope in first-seen order, then the single corpus-scope
    record last), so a seeded run is byte-identical.
    """

    def profile(
        self, documents: Iterable[DocumentRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[FeatureRecord]:
        """Consume the run's documents + mentions; yield per-document + corpus FeatureRecords."""
        ...


@dataclass(frozen=True)
class TypeConsolidation:
    """The output of a :class:`TypeConsolidator` — remapped records + the vocabulary that did it.

    Three things travel together because they are one decision: the ``mentions`` and
    ``relations`` with their ``label`` rewritten to canonical form (and their raw induced label
    preserved on ``raw_label``), and the :class:`~latence_core.contracts.TypeVocabulary` that
    defines the mapping. The Runner threads the records downstream and persists the vocabulary
    beside the Stage's checkpoint — so the consolidation is inspectable as data, and an
    incremental run can merge into it (ADR-0017: files are the source of truth).
    """

    mentions: list[EntityMention] = field(default_factory=list)
    relations: list[RelationMention] = field(default_factory=list)
    vocabulary: TypeVocabulary | None = None


@runtime_checkable
class TypeConsolidator(Protocol):
    """Canonicalizes the corpus's TYPE vocabulary (CONTEXT ``Resolution``; T3).

    The counterpart to :class:`Disambiguator` in the resolution phase: where a Disambiguator
    resolves entity **mentions** to canonical entities, a TypeConsolidator resolves the
    **type vocabulary** those mentions are labelled with. Both are corpus-level, and both exist
    because the front half of the pipeline now runs per chunk: each chunk induces its own
    labels, which is what gives it a small, relevant, uncapped label set — and what makes chunk
    A say ``org`` where chunk B says ``organization``. Consistency is earned HERE, downstream,
    instead of being imposed upstream by freezing and salience-capping one corpus schema before
    extraction (which silently dropped rare-but-critical labels).

    It is a **corpus-level mention→mention transform**: it consumes the whole run's
    :class:`~latence_core.contracts.EntityMention`s (required — the labels it canonicalizes) and,
    optionally, its :class:`~latence_core.contracts.RelationMention`s, and returns the SAME
    records with each ``label`` rewritten to its canonical form and the raw induced label kept on
    ``raw_label`` for audit — plus the :class:`~latence_core.contracts.TypeVocabulary` it built.

    It runs BEFORE Disambiguation, not after: the entity resolver votes a cluster's type from its
    members' labels and gates its low-precision rungs on type compatibility, so feeding it a
    drifted vocabulary would both split clusters that belong together and mistype the ones that
    survive. Canonical types in, consistent entities and consistent graph out.

    Determinism is part of the contract: identical inputs must yield an identical vocabulary and
    identical records (stable clustering, stable canonical election), so a seeded run is
    byte-identical (Baseline bar).
    """

    def consolidate(
        self,
        mentions: Iterable[EntityMention],
        relations: Iterable[RelationMention] = (),
    ) -> TypeConsolidation:
        """Consume the run's mentions (+ optional relations); return them remapped + the vocab."""
        ...


@runtime_checkable
class Disambiguator(Protocol):
    """Links mentions to canonical entities + merges duplicates (CONTEXT ``Disambiguation``).

    The S8 Disambiguation Capability — the algorithmic crown. Disambiguation is a
    **corpus-level** Stage: it consumes the whole run's
    :class:`~latence_core.contracts.EntityMention`s (required — the mentions it resolves and
    links) and, OPTIONALLY, the run's :class:`~latence_core.contracts.RelationMention`s (to
    normalise onto the canonical entities), and emits
    :class:`~latence_core.contracts.DisambiguationRecord`s at two scopes:

    * one ``ENTITY``-scope record per resolved
      :class:`~latence_core.contracts.CanonicalEntity` — a cross-document merged cluster with
      its member mentions, an audited, confidence-weighted merge log (**no silent
      over-merge** — a below-policy merge is logged not applied), an optional external-KB
      link (with a **graceful fallback** for unlinked mentions), and page-accurate
      Provenance + :class:`~latence_core.contracts.Evidence` back to the source mentions;
    * one ``RELATION``-scope record per :class:`~latence_core.contracts.NormalizedRelation` —
      a relation whose endpoints are remapped to canonical entities and whose label is L2
      normalised (fuzzy/alias, type-filtered, inverse-detected + direction-swapped).

    Both carriers are supplied so relations can be normalised onto the entities resolved in
    the same pass; a Provider that only disambiguates entities ignores the relations. The
    mention side is REQUIRED (unlike Profiling's optional mentions) — Disambiguation with no
    mentions has nothing to resolve. Records are emitted deterministically (all ENTITY
    records in canonical-text order, then RELATION records), so a seeded run is
    byte-identical (Baseline bar). The CPU reference path must work with no embedder/KB; a
    FAISS/GPU blocking Provider is a pluggable option, not required (S8 AC).

    ``chunks`` is a **third, additive, defaulted** carrier: the run's
    :class:`~latence_core.contracts.ChunkRecord`s, so a Provider that resolves mentions by
    their *surrounding context* (the learned ``disambiguation.embedding`` Provider — the pod
    proved a bare-surface embedding over-merges distinct short surfaces like ``IBM`` and
    ``Apple``, but surface+context separates them cleanly) can locate a mention's parent chunk
    (by ``EntityMention.chunk_record_id``) and read a bounded window of its ``content`` around
    the mention's offset. It defaults to an empty tuple so the signature is backward-compatible:
    the algorithmic cascade ignores it (its behaviour is unchanged), a mentions-only run passes
    no chunks, and any existing Provider that only accepts ``(mentions, relations)`` still
    satisfies the Protocol — the Runner threads chunks only when the parameter exists (W1
    rework, #94).
    """

    def disambiguate(
        self,
        mentions: Iterable[EntityMention],
        relations: Iterable[RelationMention],
        chunks: Iterable[ChunkRecord] = (),
    ) -> Iterator[DisambiguationRecord]:
        """Consume the run's mentions (+ optional relations + optional chunks); yield records."""
        ...


@runtime_checkable
class GraphAssembler(Protocol):
    """Builds the canonical knowledge graph from disambiguated records (CONTEXT ``Graph Assembly``).

    The S9 Graph Assembly Capability. Graph Assembly is a **corpus-level** Stage: it consumes
    the whole run's :class:`~latence_core.contracts.DisambiguationRecord`s (the canonical
    entities and normalized relations the S8 Disambiguation Stage produced) and emits
    :class:`~latence_core.contracts.GraphRecord`s at two scopes:

    * one ``NODE``-scope record per :class:`~latence_core.contracts.CanonicalEntity` — a graph
      node with a **deterministic** content-addressed id, its member mentions, source-document
      reach, an optional external-KB link, and :class:`~latence_core.contracts.Evidence` back
      to the source mentions;
    * one ``EDGE``-scope record per :class:`~latence_core.contracts.NormalizedRelation` — a
      graph edge with a deterministic id, its head/tail node ids, the normalized label, and
      **per-edge Evidence** back to the source relation's mentions/documents (every edge
      carries Evidence, the S9 acceptance criterion).

    An edge is emitted only when *both* its endpoints resolved to a node (endpoint validation
    — no dangling edges). Records are emitted deterministically (all NODE records in node-id
    order, then EDGE records in edge-id order), so a seeded run is byte-identical (Baseline
    bar) and two runs over the same corpus produce the same graph. The reference Provider is
    pure-Python, no embedder/KB required; a graph-DB-writer or embedding-augmented Provider is
    a pluggable option behind this same seam (ADR-0017: Export writes files, never a live DB).
    """

    def assemble(self, records: Iterable[DisambiguationRecord]) -> Iterator[GraphRecord]:
        """Consume the run's DisambiguationRecords; yield GraphRecords (nodes then edges)."""
        ...


@runtime_checkable
class GraphCompleter(Protocol):
    """Predicts missing KG edges over the assembled graph (CONTEXT ``Graph Assembly``; W2-linkpred).

    The **post-Assembly link-prediction** Capability (ADR-0037): an OPTIONAL corpus-level Stage
    that consumes the whole run's assembled :class:`~latence_core.contracts.GraphRecord`s (NODE +
    EDGE) and yields ADDITIONAL EDGE-scope :class:`~latence_core.contracts.GraphRecord`s for
    **predicted** edges — the ``§19`` predicted-vs-extracted split GraphAssembler carried as
    ``is_predicted`` (ADR-0027) turned into its own Stage. It never re-emits the input nodes/edges
    and never mutates an asserted edge; it only appends new predicted edges.

    A predicted edge is a :class:`~latence_core.contracts.GraphEdge` at ``scope=EDGE``,
    UNMISTAKABLY marked ``properties["inferred"] = True`` (plus ``scorer``/``score``/``rank``/
    ``calibrated``) so it is never silently merged into the asserted stream — the enterprise-honesty
    crux (ADR-0037). Its Evidence is honest and model-derived, NOT fabricated mention offsets:
    ``mention_ids = []`` (there is no textual mention to cite), ``document_ids`` = the union of the
    head + tail nodes' ``source_document_ids``, ``snippet`` a human justification naming the
    scorer + score, and ``confidence`` the calibrated score. A predicted edge that COLLIDES
    (same endpoints +
    label) with an asserted edge is DROPPED — it is already asserted — logged, not emitted.

    Mirroring :class:`GraphAssembler`: corpus-level, ``Iterable`` in / ``Iterator`` out,
    deterministic + seeded → byte-identical (predicted edges yielded in edge-id order), the CPU
    reference (``graph_completion.reference``) pure-Python, no torch. A learned ULTRA/PyKEEN scorer
    is a pluggable Provider behind this same seam (Slice 2). The Stage is genuinely OPTIONAL: a
    stack with no ``graph_completion`` Stage behaves exactly as today.
    """

    def complete(self, records: Iterable[GraphRecord]) -> Iterator[GraphRecord]:
        """Consume the assembled graph (NODE + EDGE); yield ADDITIONAL predicted EDGE records.

        Never re-emits the input, never mutates asserted edges — only appends predicted
        EDGE-scope :class:`~latence_core.contracts.GraphRecord`s, each marked
        ``properties["inferred"] = True`` with honest model-derived Evidence, in edge-id order
        (deterministic). A predicted edge colliding with an asserted edge (same endpoints +
        label) is dropped, not emitted.
        """
        ...


@runtime_checkable
class ContextEnricher(Protocol):
    """Projects the assembled KG back onto each chunk as a compact header (W13 context enrichment).

    The W13 **context-enrichment** Capability (ADR-0039): an OPTIONAL Stage placed AFTER Graph
    Assembly that stamps each :class:`~latence_core.contracts.ChunkRecord` with a compact
    :class:`~latence_core.contracts.ContextHeader` — its canonical entities + top-k KG-neighbor
    triples — which the Export prepends to the EMBEDDING input ONLY. The connections a chunk needs
    for retrieval relevance are ALREADY computed as the KG; this surfaces them into the vector + a
    metadata column, without inflating the stored chunk text (the LLM window stays the clean chunk).

    Shaped like :class:`LabelInducer` (a chunk→chunk transform that yields the SAME chunks with an
    additive field populated via ``chunk.model_copy(update=...)``) but corpus-level and multi-input,
    like :class:`Disambiguator`: it consumes the run's :class:`~latence_core.contracts.ChunkRecord`s
    (what it enriches + passes through), :class:`~latence_core.contracts.EntityMention`s (a chunk's
    canonical entities, via ``EntityMention.chunk_record_id``) AND the assembled
    :class:`~latence_core.contracts.GraphRecord`s (NODE + EDGE — the entities' KG neighbors). It
    yields the SAME chunks, 1:1 and order-stable, with ``context_header`` populated (or left
    ``None``
    for a chunk with no gated canonical entity — nothing to add). Genuinely optional +
    contract-preserving: offsets/provenance/content are untouched, so a stack without the Stage is
    byte-identical (the field stays ``None`` and the Export prepends nothing).

    Deterministic + seeded (the CPU reference ``context.kg_header`` is pure-Python, no model, no
    key)
    → byte-identical: every collection in the header is sorted, and the neighbor triples are
    confidence-sorted then CAPPED (``max_neighbors_per_entity`` top-k per entity + an overall
    ``max_triples``) so a corpus-wide hub entity contributes only its TOP relations — the mandatory
    anti-bloat + hub-entity guard (ADR-0039). The optional LLM situating-sentence mode is a
    documented follow-on (deterministic=False), not the default.
    """

    def enrich(
        self,
        chunks: Iterable[ChunkRecord],
        mentions: Iterable[EntityMention],
        graph_records: Iterable[GraphRecord],
    ) -> Iterator[ChunkRecord]:
        """Consume the run's chunks + mentions + assembled graph; yield the SAME chunks enriched.

        The chunk stream is threaded 1:1 and order-stable (never dropped or reordered): a chunk with
        at least one confidence-gated canonical entity carries a populated
        :class:`~latence_core.contracts.ContextHeader`; a chunk with none flows through with
        ``context_header`` still ``None`` (byte-identical to no-stage for that chunk). Deterministic
        +
        seeded → byte-reproducible (sorted + capped + content-addressed, the Baseline bar).
        """
        ...


@dataclass(frozen=True)
class GraphFeatures:
    """Per-node structural features over the assembled KG: centrality + community (ADR-0051).

    The result a :class:`GraphFeatureComputer` returns: for **every** node in the assembled graph,
    a ``centrality`` score and a ``community`` id. It is a *value* the Graph-Assembly Stage stamps
    onto each :class:`~latence_core.contracts.GraphNode`'s ``properties`` bag (``centrality`` /
    ``community``) — the computer holds nothing and mutates nothing (ADR-0048). Invariants (enforced
    at construction so a Provider cannot emit a half-populated result, so the downstream stamp is
    total):

    * ``centrality`` and ``community`` cover the **same** node-id set — every node gets both a
      centrality and a community, never one without the other (the anti-false-green contract: a
      computer that skips nodes fails here, not silently downstream);
    * each ``centrality`` value is a finite float (the normalisation is the Provider's business —
      degree-centrality lands in ``[0, 1]``, PageRank sums to ~1 — but every value is a real number,
      never NaN/inf);
    * each ``community`` id is a non-empty string — the deterministic, content-addressed id of the
      node's community (the reference Providers use the community's lexicographically-smallest
      ``node_id``), so two runs over the same corpus agree byte-for-byte (the Baseline bar).

    An empty graph yields the empty :class:`GraphFeatures` (both maps empty), never an error.
    """

    centrality: dict[str, float] = field(default_factory=dict)
    community: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math

        if self.centrality.keys() != self.community.keys():
            missing_c = sorted(self.community.keys() - self.centrality.keys())
            missing_k = sorted(self.centrality.keys() - self.community.keys())
            msg = (
                "GraphFeatures must cover the same node-id set for centrality and community "
                f"(nodes missing centrality: {missing_c}; missing community: {missing_k})."
            )
            raise ValueError(msg)
        for node_id, value in self.centrality.items():
            if not math.isfinite(value):
                msg = (
                    f"GraphFeatures centrality for node {node_id!r} must be finite, got {value!r}."
                )
                raise ValueError(msg)
        for node_id, community_id in self.community.items():
            if not community_id:
                msg = f"GraphFeatures community id for node {node_id!r} must be a non-empty string."
                raise ValueError(msg)


@runtime_checkable
class GraphFeatureComputer(Protocol):
    """Computes centrality + community per node over the assembled KG (CONTEXT ``Graph Assembly``).

    The index-time **graph-feature enrichment** Capability (ADR-0051): a pure transform that, given
    the assembled graph's :class:`~latence_core.contracts.GraphNode`s and
    :class:`~latence_core.contracts.GraphEdge`s, returns one :class:`GraphFeatures` — a
    **centrality** measure (degree / PageRank) and a **community** id per node. It holds no index,
    runs no search (ADR-0048), names no model. The Graph-Assembly Stage OPTIONALLY resolves a
    computer from its ``config['graph_features']`` (the same ``{"provider", "config"}`` composition
    Export uses for its Embedder) and stamps the returned features onto each node's ``properties``
    bag, so they flow into ``graph-nodes.parquet`` columns and — via Context Enrichment — onto each
    chunk's :class:`~latence_core.contracts.ContextHeader` (the two consumers ADR-0051 names:
    graph-augmented retrieval and the Knapsack packer's ``centrality`` / ``cluster_ids`` inputs).
    Genuinely optional: a Stage that names no computer emits the graph byte-identically to today.

    Behaviour VARIES behind the seam, so it ships **two** in-core reference adapters
    (:mod:`latence_core.stages.graph_features`): ``graph_features.degree`` (degree centrality +
    connected-components community) and ``graph_features.pagerank`` (PageRank centrality +
    label-propagation community). Both are pure-Python, zero-dependency and DETERMINISTIC (fixed
    iteration order, rounded scores, content-addressed community ids), so a seeded run is
    byte-reproducible. They are *references*, not the blessed Louvain/Leiden community detection
    (ADR-0051), which — like SPLADE/MUVERA — ships as a pluggable, license-verified package.
    """

    def compute_features(
        self, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
    ) -> GraphFeatures:
        """Compute one :class:`GraphFeatures` (centrality + community per node) over the KG.

        Consumes the whole assembled graph (all nodes + all edges); returns a
        :class:`GraphFeatures` covering **every** input node (never a partial map). An edge whose
        endpoint is not among ``nodes`` is ignored (belt-and-braces — the assembler validates
        endpoints). An empty graph yields the empty :class:`GraphFeatures`. Deterministic → a seeded
        run is byte-reproducible (the Baseline bar).
        """
        ...


@dataclass(frozen=True)
class DeltaOutcome:
    """The result of applying a :class:`Delta` to the parent corpus (S11 affected-set recompute).

    A :class:`DeltaProcessor` takes the last committed Corpus Version's corpus-level derived
    records, the current run's freshly-extracted corpus-level records, and the classified
    :class:`~latence_core.contracts.Delta`, and returns:

    * ``records`` — the new **live** derived set for Version N+1: the parent records the delta did
      not touch, spliced with the affected-set recompute's freshly-resolved records (the
      merge-on-add / split-on-delete result). This is what the Corpus Version commits.
    * ``retracted`` — the derived records tombstoned by this delta (soft-retained on a
      Retraction, discarded on a Purge), so the audit knows exactly what left the live set.
    * ``entities_created`` / ``entities_merged`` / ``entities_split`` — the affected-set
      recompute's effect on canonical entities vs the parent (a bridging doc that MERGED two
      clusters, a deleted doc that SPLIT one) — the S11 churn the Quality Report reports.
    * ``edges_added`` / ``edges_retracted`` — the graph patch counts.
    * ``affected_set_size`` / ``corpus_record_count`` — the perf-baseline witness: how many
      corpus-level records the affected-set recompute actually re-resolved
      (``affected_set_size``) out of the whole current corpus-level set
      (``corpus_record_count``). On an incremental (non-reconciliation) delta
      ``affected_set_size`` is the blocking-neighborhood closure the recompute touched, which is
      ``<= corpus_record_count`` — the measurable "recompute only the affected set" bound
      (ADR-0018). On a full Reconciliation the two are equal (the whole corpus is the affected
      region).
    """

    records: list[Record] = field(default_factory=list)
    retracted: list[Record] = field(default_factory=list)
    entities_created: int = 0
    entities_merged: int = 0
    entities_split: int = 0
    edges_added: int = 0
    edges_retracted: int = 0
    affected_set_size: int = 0
    corpus_record_count: int = 0


@runtime_checkable
class DeltaProcessor(Protocol):
    """Applies a Delta to the parent corpus with affected-set recompute (CONTEXT ``Delta``).

    The S11 Delta Capability (ADR-0018). A DeltaProcessor is model-agnostic and pure over
    records: given the parent Corpus Version's corpus-level derived records
    (:class:`~latence_core.contracts.DisambiguationRecord`s +
    :class:`~latence_core.contracts.GraphRecord`s), the current run's freshly-extracted
    corpus-level records, and the classified :class:`~latence_core.contracts.Delta`, it returns
    a :class:`DeltaOutcome`: the new live derived set for Version N+1 (parent records the delta
    did not touch, spliced with the affected-set recompute), the retracted set, and the
    entity/edge churn counts.

    The DeltaProcessor owns the **corpus-level** recompute of the *committed* record set — the
    blocking-neighborhood affected set for Disambiguation and the graph patch by
    ``source_document_ids`` (ports ``graph/patcher``). (Doc-level Stages, Parse … Redaction, run
    only on the NEW/CHANGED documents since W17/ADR-0043 — an unchanged document's records are
    reused across runs and the processor consumes the union of reused + freshly-extracted
    corpus-level records exactly as before, so the seam here is unchanged.) When the Delta signals
    a full Reconciliation (drift crossed the threshold), the processor re-resolves ALL blocks
    rather than only the affected set; the seam is identical, only the affected set differs. The
    reference Provider is pure-Python, deterministic and CPU-viable (the blocking-neighborhood
    :func:`~latence_core.state.incremental.compute_affected_set` closure + GraphPatcher); a
    distributed blocking Provider is a pluggable upgrade behind this same seam.
    """

    def apply_delta(
        self,
        parent_records: Iterable[Record],
        current_records: Iterable[Record],
        delta: Delta,
    ) -> DeltaOutcome:
        """Apply ``delta`` to the parent corpus; return the patched live set + churn."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Optionally embeds RAG-corpus text into vectors (CONTEXT — an opt-in Export augmentation).

    The S9 **opt-in** Embedder Capability (ADR-0017: "embeddings are optional (an opt-in
    Embedder Provider) so any vector DB can ingest it"). It is **not required for Export** —
    the RAG-ready corpus exports its cleaned/chunked/PII-handled text with full
    Provenance/Classification whether or not an embedder is wired. When a Pipeline opts in, an
    Embedder maps each text to a fixed-width vector so the exported corpus carries embeddings a
    vector DB can ingest directly.

    The reference embedder in core is a deterministic, dependency-free hashing embedder (no
    model, no weights, no license to verify — ADR-0012/0016), so the seam is exercised end to
    end offline; the default learned Embedder (IBM Granite Embedding r2, Apache-2.0; ADR-0045)
    is a pluggable Provider package, not a core dependency. The Capability names no model — only
    the seam.
    """

    @property
    def dimension(self) -> int:
        """The fixed embedding width every returned vector has."""
        ...

    def embed(self, texts: Iterable[str]) -> Iterator[list[float]]:
        """Map each input text to a ``dimension``-wide vector, in input order."""
        ...


@dataclass(frozen=True)
class SparseVector:
    """A sparse term-weight vector — the index-time SPLADE-family signal (ADR-0050/0053).

    The shape every vector DB's sparse index expects: two parallel columns, ``indices`` (the
    term ids that have non-zero weight) and ``values`` (their weights), materialised at Export as
    ``sparse_indices: list<int>`` + ``sparse_values: list<float>`` (ADR-0050). Invariants (enforced
    at construction so a Provider cannot emit a malformed vector, and so the emission is
    byte-deterministic regardless of a Provider's internal dict ordering):

    * ``indices`` and ``values`` are the same length;
    * ``indices`` are **strictly ascending** (hence unique) and non-negative — the canonical order
      a sparse index ingests, and the order two runs must agree on for a byte-identical export;
    * a zero vector (empty text, no surviving terms) is the empty ``SparseVector()`` — never a
      dense run of zeros.

    A :class:`SparseEmbedder` Provider builds one via :meth:`from_terms` from its natural
    ``{term_id: weight}`` map, which sorts and validates in one place.
    """

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            msg = (
                f"SparseVector indices/values length mismatch: "
                f"{len(self.indices)} indices vs {len(self.values)} values."
            )
            raise ValueError(msg)
        prev = -1
        for idx in self.indices:
            if idx < 0:
                msg = f"SparseVector index must be non-negative, got {idx}."
                raise ValueError(msg)
            if idx <= prev:
                msg = (
                    "SparseVector indices must be strictly ascending (unique) — the canonical "
                    f"sparse-index order; got {idx} after {prev}."
                )
                raise ValueError(msg)
            prev = idx

    @classmethod
    def from_terms(cls, terms: dict[int, float]) -> SparseVector:
        """Build a canonical :class:`SparseVector` from a ``{term_id: weight}`` map.

        Sorts the terms by id (the ascending-index invariant) so the emitted vector is
        byte-identical no matter what order the Provider accumulated the terms in — the single
        place ordering is imposed, so no Provider re-implements it. Zero-weight terms are dropped
        (a sparse vector carries only non-zero weights).
        """
        kept = sorted((idx, w) for idx, w in terms.items() if w != 0.0)
        return cls(indices=[idx for idx, _ in kept], values=[w for _, w in kept])


@runtime_checkable
class SparseEmbedder(Protocol):
    """Encodes corpus text into a sparse term-weight vector (a ``Signal generator``; ADR-0053).

    The index-time SPLADE-family sibling of :class:`Embedder` for the shape a dense vector cannot
    express: a variable-support term-weight map over a fixed vocabulary, emitted at Export as the
    parallel ``sparse_indices`` / ``sparse_values`` columns a store's sparse index ingests directly
    (ADR-0050). Like every signal generator it is a **pure transform** that holds nothing (ADR-0048)
    and names no model — the framework names the seam.

    The blessed real Provider (ADR-0053) composes a multilingual SPLADE model's term-weights with
    the pipeline's confidence-gated NER entity terms + selected metadata, over the *redacted* corpus
    text by default; that composition + its license-verified model ship as a pluggable package
    (``latence-splade``), NOT in core. The in-core ``sparse.hashing`` reference is a deterministic,
    dependency-free hashing sparse encoder that exercises the seam + emission path offline (no
    model, no weights, no license to verify — ADR-0012/0016), exactly as ``embedding.hashing`` does
    for the dense seam. It is a reference, not SPLADE.
    """

    @property
    def vocab_size(self) -> int:
        """The size of the term-index space: every emitted index lies in ``[0, vocab_size)``."""
        ...

    def encode_sparse(self, texts: Iterable[str]) -> Iterator[SparseVector]:
        """Map each text to a :class:`SparseVector` (canonical ascending indices), in input order.

        One vector per input text, in order (like :meth:`Embedder.embed`). Empty/degenerate text
        yields the empty ``SparseVector()`` (never a crash). A deterministic Provider is
        byte-reproducible (the Baseline bar).
        """
        ...


@dataclass(frozen=True)
class SparseSignals:
    """The per-record structured signals a composing ``SparseEmbedder`` injects at index time.

    ADR-0053's sparse vector is not plain ``SPLADE(text)`` but the structured composition
    ``SPLADE(text) ⊕ boost·entity_terms ⊕ boost·metadata_terms``. The text-only
    :meth:`SparseEmbedder.encode_sparse` seam can only build the model leg — no structured signal
    crosses it — so a :class:`ComposingSparseEmbedder` receives this parallel per-record carrier and
    folds the chunk's own structured signals into the emitted vector. That is what makes the
    index-time and query-time sparse spaces **match** (the ADR-0053 one-composition-two-clocks
    guarantee): without it the query encoder injects boosted entity/metadata terms the corpus never
    carried, and a hybrid search on an injected term matches zero documents. A record with no
    structured signals is :data:`EMPTY_SPARSE_SIGNALS` and composes to exactly the model leg —
    identical to the text-only path (so the additive/byte-identical guarantee is untouched when no
    signals exist).

    * ``entities`` — the chunk's ``(surface, confidence)`` entity mentions offered to the
      composition. The composition confidence-gates them; the embedder's redaction gate additionally
      drops any surface that no longer occurs in the (redacted) corpus text, so a masked PII surface
      is never re-injected as a term (ADR-0053, REDACTION-BY-DEFAULT).
    * ``metadata`` — the chunk's ``{field: value}`` map from which the embedder's operator-selected
      ``metadata_fields`` are injected (e.g. its classification category).
    """

    entities: tuple[tuple[str, float], ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


#: The signals of a record that carries no structured entity/metadata signal — composes to exactly
#: the model leg (identical to the text-only :meth:`SparseEmbedder.encode_sparse` path).
EMPTY_SPARSE_SIGNALS = SparseSignals()


@runtime_checkable
class ComposingSparseEmbedder(Protocol):
    """A :class:`SparseEmbedder` that also composes structured per-record signals (ADR-0053).

    The index-time seam for the ADR-0053 **structured composition**. A plain :class:`SparseEmbedder`
    exposes only the text-only :meth:`~SparseEmbedder.encode_sparse` (the model leg); a *composing*
    embedder additionally accepts, per record, the :class:`SparseSignals` (confidence-bearing entity
    surfaces + selected metadata) and folds them into the emitted vector — so the corpus's sparse
    space carries the SAME boosted entity/metadata terms the query-side encoder injects (the
    one-composition-two-clocks guarantee). Export detects this Capability and passes each row's
    signals; an embedder that does NOT implement it (e.g. the deterministic ``sparse.hashing``
    reference) is driven text-only, byte-identically to today.

    An implementer is also a :class:`SparseEmbedder` (it keeps the text-only ``encode_sparse`` for
    the no-signal path); this Protocol declares only the additional composing method so an
    ``isinstance`` check cleanly discriminates the two.
    """

    def encode_sparse_composed(
        self, texts: Iterable[str], signals: Sequence[SparseSignals]
    ) -> Iterator[SparseVector]:
        """Map each ``(text, signals)`` pair to its composed :class:`SparseVector`, in input order.

        ``signals`` is **parallel** to ``texts`` (one :class:`SparseSignals` per text, same length +
        order). One vector per input, in order; a record whose signals are
        :data:`EMPTY_SPARSE_SIGNALS` composes to exactly the model leg (identical to
        :meth:`SparseEmbedder.encode_sparse` on that text — the additive guarantee).
        """
        ...


@runtime_checkable
class MultiVectorEmbedder(Protocol):
    """Encodes text into a variable-length multi-vector — one vector per token (ADR-0050).

    The index-time generator for the ColBERT-style late-interaction shape: a text becomes a
    *sequence* of fixed-width token vectors (not one pooled vector). Raw multi-vectors are heavy and
    have a different access pattern, so Export writes them to a **sidecar** ``multivectors.parquet``
    (``list<list<double>>`` keyed by ``record_id``), never into ``records.parquet`` (ADR-0050). They
    also feed an :class:`FdeConverter` to produce the dense ``fde_embedding`` column.

    Experimental and off by default (ADR-0050). The in-core ``multivector.hashing`` reference is a
    deterministic, dependency-free per-token hashing encoder that exercises the seam offline; a real
    ColBERT-family encoder is a pluggable package.
    """

    @property
    def dimension(self) -> int:
        """The fixed width of every token vector in every returned multi-vector."""
        ...

    def encode_multivector(self, texts: Iterable[str]) -> Iterator[list[list[float]]]:
        """Map each text to a list of ``dimension``-wide token vectors, in input order.

        One multi-vector per input text, in order. Empty/degenerate text yields an empty list (no
        tokens, never a crash). Every token vector has width :attr:`dimension`. A deterministic
        Provider is byte-reproducible.
        """
        ...


@runtime_checkable
class FdeConverter(Protocol):
    """Converts a multi-vector into one fixed-width dense vector (MUVERA FDE; ADR-0050).

    A **data-oblivious, training-free** reduction that maps a variable-length multi-vector to a
    single ``dimension``-wide dense vector, so late-interaction quality can be approximated inside a
    customer's ordinary **dense** index with no special store — emitted at Export as the
    ``fde_embedding`` column (ADR-0050). Pure transform, holds nothing, names no model.

    Experimental and off by default (ADR-0050): the *production* converter decision (canonical
    MUVERA vs a regularizer-refined variant) is deferred, and the quality-blessed canonical MUVERA
    reference is a pluggable Provider (``fde.muvera``). The in-core ``fde.reference`` is a
    deterministic, dependency-free SimHash space-partition-and-sum converter that exercises the seam
    + emission path offline — a genuine data-oblivious FDE (NOT a passthrough that lies, the
    anti-false-green bar), without claiming a MaxSim-approximation quality guarantee on any corpus.
    """

    @property
    def dimension(self) -> int:
        """The fixed width of the produced dense vector, independent of the multi-vector length."""
        ...

    def convert(self, multivectors: Iterable[list[list[float]]]) -> Iterator[list[float]]:
        """Map each multi-vector to one ``dimension``-wide dense vector, in input order.

        One dense vector per input multi-vector, in order. An empty multi-vector yields the
        zero vector of width :attr:`dimension`. A token vector whose width does not match the
        converter's declared input width is rejected with a typed error (the faithful contract a
        real converter enforces — anti-false-green). A deterministic Provider is byte-reproducible.
        """
        ...


@runtime_checkable
class Tokenizer(Protocol):
    """Splits text into lexical terms — the varying seam of the BM25 term-stats export (ADR-0048).

    BM25 is a *lexical* signal, so its corpus statistics (document frequency, term frequency,
    document length, average document length) are a pure function of one thing that genuinely
    **varies**: how text is split into terms. This Capability is that seam. Everything else in the
    :class:`~latence_core.bm25.Bm25Accumulator` — the df/tf fold, the Robertson idf, the ascending
    canonical order — is fixed math; the tokenisation is the knob a corpus tunes. Like every
    retrieval-tooling piece it is a **pure transform that holds nothing** (ADR-0048) and names no
    model — the framework names the seam.

    Two in-core reference adapters ship behind it (:mod:`latence_core.bm25`), and they are
    genuinely different tokenisations (not a clone + a rename — the anti-false-green bar for a real
    seam): ``tokenizer.regex`` folds to lowercase *Unicode* alphanumeric runs (punctuation and the
    underscore are boundaries and are dropped, matching the dense/sparse hashing references — so
    ``Müller`` and ``Таможенный`` are whole terms, though an unspaced script like Chinese yields one
    term per whitespace run: the reference is Unicode-aware, not a word segmenter, and a corpus that
    needs segmentation plugs it in HERE), while ``tokenizer.whitespace`` splits
    only on whitespace and keeps punctuation attached to the token (so ``"foo,"`` and ``"foo"`` are
    distinct terms). A corpus that wants a store's exact analyzer plugs its own Tokenizer in behind
    this same seam. Both references are deterministic, so the emitted BM25 artifact is
    byte-reproducible (the Baseline bar).
    """

    def tokenize(self, texts: Iterable[str]) -> Iterator[list[str]]:
        """Map each text to its list of terms, in input order (one token list per input text).

        One token list per input text, in order (like :meth:`Embedder.embed`). The terms appear in
        their order of occurrence in the text (a BM25 term frequency counts occurrences, so order is
        immaterial to the statistic but the contract is occurrence-order, not sorted). Empty or
        term-free text yields the empty list ``[]`` (never a crash). A deterministic Tokenizer is
        byte-reproducible (the Baseline bar).
        """
        ...


@runtime_checkable
class Export(Protocol):
    """Materializes AI-ready outputs (JSONL + Parquet in S1).

    A corpus Export Provider may **optionally** attach embeddings (ADR-0017: an opt-in
    Embedder augmentation). It opts in by naming an :class:`Embedder` Provider in its own
    Stage ``config['embedder']`` (a ``{"provider": ..., "config": {...}}`` block); the Provider
    resolves that Embedder and threads each exported record's text through it, adding a vector
    per row. A Provider that names no embedder exports its text unchanged — embeddings are never
    required for Export. The seam is one signature so a non-embedding Export (the KG / demo-site
    Providers) satisfies it identically.
    """

    def export(self, records: Iterable[Record], storage: Storage, out_dir: str) -> list[str]:
        """Write records to ``out_dir`` on ``storage``; return the output URIs."""
        ...


_CapabilityT = TypeVar("_CapabilityT")


def protocol_members(protocol: type) -> frozenset[str]:
    """The member names a ``runtime_checkable`` Protocol requires, without touching an instance.

    ``__protocol_attrs__`` is the authoritative set on 3.12+; on 3.11 (this project's floor) typing
    does not expose it publicly, so we fall back to the Protocol's own public ``dir()`` — which for
    every Capability here yields exactly its declared members.
    """
    members = getattr(protocol, "__protocol_attrs__", None)
    if members is None:
        members = {name for name in dir(protocol) if not name.startswith("_")}
    return frozenset(members)


def satisfies_capability(
    obj: object, protocol: type[_CapabilityT] | Any
) -> TypeGuard[_CapabilityT]:
    """``isinstance(obj, protocol)`` for a Capability Protocol — WITHOUT invoking any member.

    Why this exists rather than plain ``isinstance``: CPython implements a ``runtime_checkable``
    Protocol check with ``hasattr(instance, name)`` over the Protocol's members, and several
    Capabilities declare a **property** (:attr:`Embedder.dimension`, :attr:`SparseEmbedder.
    vocab_size`, :attr:`MultiVectorEmbedder.dimension`). ``hasattr`` therefore *calls* that property
    — so an ``isinstance`` check performed at **wiring time** executes Provider work that ADR-0016
    says is deferred to first use. Concretely (audit R6): resolving ``sparse.splade`` behind an
    Export's ``sparse_embedder`` spec made ``isinstance`` read ``vocab_size``, which loaded a
    Hugging Face checkpoint from the network during config validation, and made the R5
    "every committed environment-stack component constructs" guard silently skip the whole Export.

    The check here is structurally identical (every Protocol member must be present) but resolves
    each member against the provider's **class** first: a property, a method and a class attribute
    are all present on the class as objects, and looking them up there never runs the descriptor's
    getter. Only a member absent from the class falls back to the instance — that is a plain
    ``__init__``-assigned attribute, whose lookup is a dict hit, not Provider work.

    A Provider is still rejected exactly as before when it does not carry a member, so the loud
    ``ContractError`` at wiring time (ADR-0034) is unchanged.
    """
    cls = type(obj)
    return all(
        hasattr(cls, name) or hasattr(obj, name) for name in protocol_members(protocol)
    )


class ProviderRegistry:
    """Discovers Providers from ``latence.providers`` entry points.

    Discovery is lazy and cached: entry points are enumerated once, then each is
    loaded (imported) only when its name is first requested. This keeps the
    registry cheap even when many heavy Provider packages are installed.
    """

    def __init__(self, group: str = ENTRY_POINT_GROUP) -> None:
        self._group = group
        self._entry_points: dict[str, metadata.EntryPoint] | None = None
        self._loaded: dict[str, type] = {}
        # P3-F1: a provider's declared ProviderProfile, cached alongside ``_loaded``.
        # A ``None`` value is a *cached* "declares no profile" (distinct from "not yet
        # read", which is the key being absent), so ``profile()`` never re-reads a
        # provider that legitimately declares none and never fabricates one.
        self._profiles: dict[str, ProviderProfile | None] = {}

    def _discover(self) -> dict[str, metadata.EntryPoint]:
        if self._entry_points is None:
            eps = metadata.entry_points(group=self._group)
            self._entry_points = {ep.name: ep for ep in eps}
        return self._entry_points

    def names(self) -> list[str]:
        return sorted(self._discover())

    def load(self, name: str) -> type:
        """Load and return the Provider class registered under ``name``."""
        if name in self._loaded:
            return self._loaded[name]
        eps = self._discover()
        if name not in eps:
            available = ", ".join(self.names()) or "<none>"
            msg = (
                f"No Provider registered as {name!r} in group "
                f"{self._group!r}. Available: {available}"
            )
            raise KeyError(msg)
        obj = eps[name].load()
        if not isinstance(obj, type):
            msg = f"Provider {name!r} did not resolve to a class (got {type(obj)!r})."
            raise TypeError(msg)
        self._loaded[name] = obj
        return obj

    def profile(self, name: str) -> ProviderProfile | None:
        """Return the Provider's declared :class:`ProviderProfile`, or ``None`` if none (P3-F1).

        Loads the Provider class (reading a profile requires the class object) and reads
        whichever declaration form it uses — a ``profile`` classmethod or a ``PROFILE``
        ClassVar (:func:`~latence_core.providers.profile.profile_of`). A Provider that
        declares no profile returns ``None``; the registry **never fabricates** one
        (ADR-0036 §2). The result is cached alongside ``_loaded`` so a second call is free,
        with a cached ``None`` meaning "declares none" (not "not yet read").

        Raises ``KeyError`` for an unknown name (same as :meth:`load`) and ``TypeError``
        for a malformed declaration (e.g. a ``profile`` classmethod that returns a non-
        ``ProviderProfile``) — a typo surfaces rather than reading as "no profile".
        """
        if name in self._profiles:
            return self._profiles[name]
        provider_cls = self.load(name)
        prof = profile_of(provider_cls)
        self._profiles[name] = prof
        return prof

    def profiles(self) -> dict[str, ProviderProfile | None]:
        """Map every registered Provider name to its declared profile (or ``None``) (P3-F1).

        Iterates :meth:`names` and reads each Provider's profile. Reading a profile
        **requires loading** the Provider (importing it), which is acceptable for tooling
        and the bake-off — the lazy single-name :meth:`profile` path stays lazy; only this
        map forces the loads. A Provider whose **import raises** (a missing heavy/optional
        dep, ADR-0016) is skipped-and-flagged — its name is simply omitted from the map,
        never crashing the whole sweep — so a partially-installed workspace still yields
        the profiles of every importable Provider. A Provider that imports cleanly but
        declares no profile is present with a ``None`` value (a declared absence, distinct
        from an import failure's omission).
        """
        result: dict[str, ProviderProfile | None] = {}
        for name in self.names():
            try:
                result[name] = self.profile(name)
            except Exception:  # noqa: BLE001 - skip-and-flag: one bad import never crashes the map
                # An import error (missing torch/openai/… for an uninstalled heavy Provider,
                # ADR-0016) or a malformed declaration is skipped, not fatal — omit the name.
                continue
        return result
