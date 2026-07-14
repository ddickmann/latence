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
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from latence_core.providers.profile import ProviderProfile, profile_of

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import (
        ChunkRecord,
        Delta,
        DisambiguationRecord,
        DocumentRecord,
        EntityMention,
        FeatureRecord,
        GraphRecord,
        ParserInput,
        QuarantineRecord,
        Record,
        RelationMention,
        ScreeningFinding,
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
    chunk's offset_map + page_map — the coordinate-correctness ADR-0031/0042 guarantee. Redaction
    therefore depends on the Chunk Stage (not Parse).

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
