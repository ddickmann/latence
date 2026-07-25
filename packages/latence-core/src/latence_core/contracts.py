"""Versioned data contracts shared across every Stage boundary.

One schema, dual serialization (ADR-0006): these Pydantic v2 models are the
single canonical definition of inter-Stage data. Every :class:`Record` carries
both :class:`Provenance` (immutable source lineage) and :class:`Classification`
(descriptive content attributes); validation fails if either is missing, so a
record can never silently lose its source or attributes at a Stage boundary.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bump on any breaking change to a contract below. Serialized with every record
# so downstream consumers can detect and reject an incompatible schema.
#
# v2 (S2 Parse): DocumentRecord gains ``page_map`` (source page boundaries for
# offset→page Provenance integrity) and ``disposition`` (an error disposition so
# a corrupt/unparseable file records a failure instead of crashing the run);
# PageSpan/PageMap contracts added.
# v3 (S3 Chunk + Screening): ChunkRecord (retrieval-sized piece with preserved
# offsets, page alignment, Provenance and Classification, plus propagating
# RiskMarkers); RiskMarker/ScreeningVerdict/ScreeningFinding contracts; a
# QuarantineRecord for records Screening removes from the pipeline (retained and
# inspectable with the reason recorded).
# v4 (S4 Entity Extraction): EntityMention (a typed entity mention found in a
# chunk, carrying its label, confidence, the char span it occupies in the parent
# document's assembled markdown, and the source page(s) that span resolves to via
# PageOffsetIndex) — an inter-Stage Record like ChunkRecord, so it carries
# Provenance and Classification and is exportable.
# v5 (S5 Relation Extraction): RelationMention (a typed relation between two
# EntityMentions within one document, carrying head/tail mention refs, its label,
# confidence, and a Provenance span covering the head..tail extent) — an inter-Stage
# Record like EntityMention, carrying Provenance and Classification and exportable.
# v6 (S6 Redaction): RedactionRecord (a per-document Redaction result carrying the
# masked/replaced text variant and the structured list of detected PIISpans — each
# with its type, its document-offset char span, the action taken, and a stable
# placeholder — so the action is auditable) plus PIISpan and the RedactionAction
# enum. A RedactionRecord is an inter-Stage Record like every other, carrying
# Provenance and Classification, so the redacted text threads through the DAG and
# exports to the RAG corpus as the PII-handled variant (ADR-0017/0024).
# v7 (S7 Profiling): FeatureRecord (a corpus-level Profiling result carrying a typed,
# fixed FeatureSet of statistical + quality features) at two scopes (FeatureScope:
# DOCUMENT — one per parsed document, carrying that document's per-document features;
# CORPUS — exactly one, carrying the cross-document aggregate features). A FeatureRecord
# is an inter-Stage Record like every other, carrying Provenance and Classification, so
# it threads through the DAG and exports; the CORPUS record's Provenance is the synthetic
# corpus-scope lineage (ADR-0025). Metric names are a fixed, closed set — no dynamic
# registry (S7 RECAST) — so a seeded run is byte-identical and the schema is auditable.
# v8 (S8 Disambiguation): the corpus-level Stage that links entity mentions to canonical
# entities and merges duplicates across documents, with auditable, confidence-weighted
# merge decisions (no silent over-merge) and Evidence back to source mentions (ADR-0026).
# Adds Evidence (the record linking a merge decision — or, later, a KG edge — back to the
# specific source mentions/offsets that justify it, with a confidence value: CONTEXT
# ``Evidence``); MergeStrategy (the 5-strategy resolver cascade EXACT→ALIAS→ACRONYM→
# SUBSTRING→EMBEDDING plus KB_LINK, the ported EntityResolver technique); MergeDecision (one
# audited, confidence-weighted merge or link decision — its strategy, confidence, the
# Evidence, and whether the confidence policy APPLIED it, so a below-threshold decision is
# recorded not silently applied); CanonicalEntity (a merged cluster: canonical text + type,
# its member mention record_ids, its Evidence, an optional KB link, and — via
# PageOffsetIndex — page-accurate Provenance); NormalizedRelation (a RelationMention whose
# head/tail are remapped to canonical entity ids, its label L2-normalized, inverse detected
# + direction swapped); DisambiguationScope (ENTITY / RELATION) + DisambiguationRecord (the
# scoped inter-Stage carrier, one CanonicalEntity or one NormalizedRelation per record, like
# FeatureRecord's scope split). Every one is a versioned Pydantic contract carrying
# Provenance + Classification at the boundary; a seeded run is byte-identical (merge/link
# order is deterministic and every collection is sorted).
# v9 (S9 Graph Assembly + Export): the corpus-level Stage that builds the canonical knowledge
# graph — nodes, edges, and per-edge Evidence — from disambiguated entities and relations
# (CONTEXT ``Graph Assembly``), plus the Export that materializes the two AI-ready
# deliverables on Storage (ADR-0017). Adds GraphNode (one per CanonicalEntity: a deterministic
# sha256 node id, canonical text/type, its member mentions, source-document reach, optional KB
# link, and Evidence back to its source mentions); GraphEdge (one per NormalizedRelation: a
# deterministic edge id, head/tail node ids, the normalized label, confidence, and per-edge
# Evidence back to the source relation's mentions/documents — the CONTEXT ``Evidence`` term
# reused verbatim for edge justification); GraphScope (NODE / EDGE) + GraphRecord (the scoped
# inter-Stage carrier, one GraphNode or one GraphEdge per record, like DisambiguationRecord's
# scope split). Every one is a versioned Pydantic contract carrying Provenance + Classification
# at the boundary; node/edge ids are content-addressed (sha256 over a stable key) so a seeded
# run is byte-identical and two runs over the same corpus produce the same graph.
# v10 (S11 Incremental deltas): the lean-yet-enterprise delta machinery (ADR-0018). A Delta is
# a content-hash diff of the Source manifest against the last committed Corpus Version, and a
# delta run transactionally produces an immutable Version N+1 (CONTEXT ``Corpus Version`` /
# ``Delta`` / ``Affected set`` / ``Reconciliation`` / ``Retraction`` / ``Purge``). Adds
# DeltaKind (ADDED / UPDATED / DELETED / UNCHANGED — the four content-hash classes a document
# falls into); DeletionMode (RETRACTION — the default soft tombstone, retained for audit;
# PURGE — the physical GDPR erasure of source + derived + Evidence); DeltaEntry (one document's
# classification: its DeltaKind, document id, file name, and content hash); Delta (the whole
# classified change set against a parent Corpus Version, plus the drift metric that decided
# whether a full Reconciliation was triggered); CorpusVersion (an immutable, numbered committed
# state of the corpus — its version number, parent, the delta that produced it, the record
# count, a content fingerprint, and whether it was a full reconciliation); DeltaChurn (the S11
# Quality-Report roll-up: documents ±, entities created/merged/split, edges ±, records
# retracted/purged, and the drift that triggered — or did not trigger — a reconciliation).
# Every one is a versioned Pydantic contract; a delta run is deterministic (a seeded corpus
# state yields the same manifest, the same Delta, and the same Version fingerprint — the
# Baseline bar and the G1 determinism criterion).
#
# v11 (S4 offset-map coordinate fix): ChunkRecord gains an optional ``offset_map`` — the
# run-length map from a chunk's markup-stripped ``content`` offsets back to the ORIGINAL
# parsed-markdown offsets (OffsetSegment/OffsetMap contracts). Chunk content is
# de-marked-up (chunking.strip_markup deletes 18 markup constructs) while the chunk's
# outer ``[char_start, char_end)`` stays in original coordinates, so a *sub-chunk* offset
# (an entity mention) cannot be recovered by ``char_start + local_offset`` once markup was
# stripped before it — that shift is a lower bound, not the true offset (ADR-0022 corrected,
# ADR-0031). Carrying the map lets Entity Extraction (and the fused NER+RE Provider) emit a
# mention's TRUE original offset and page at source, so every downstream consumer inherits
# correct offsets with no per-Stage relocation. Optional and additive: a chunk without it
# (pre-v11, or a document with no map) degrades to the ``char_start + local`` shift.
#
# v12: DeltaChurn gained ``affected_set_size`` / ``corpus_record_count`` — the perf-baseline
# witness that the incremental delta recompute is affected-set-scoped, not O(corpus) (issue #42;
# ADR-0018). Additive and defaulted, so a v11 DeltaChurn JSON still validates.
#
# v13 (H-C1 §4 redaction no-op floor): RedactionRecord gains
# ``redaction_disabled_for_sensitive`` — a first-class flag that a document whose
# Classification sensitivity is in the sensitive set (confidential/restricted/…) had its
# redaction control run NO detectors (skip / empty types / empty detector set), so an
# empty ``pii_spans`` there means "control OFF", never "clean". Additive and defaulted
# False, so a v12 RedactionRecord JSON still validates (ADR-0024 amended).
#
# v14 (W4 schema induction): the new OPTIONAL ``schema_induction`` Stage (a small
# OpenAI-compatible LLM with structured JSON output) induces, per document, the entity /
# relation / PII types actually evidenced in the data and annotates each record with an
# :class:`InducedLabels` carrier so the extraction Stages can UNION those induced types onto
# their config label floor (the "point it at a folder and it discovers its own schema"
# property). ChunkRecord AND DocumentRecord each gain an optional ``induced_labels:
# InducedLabels | None`` (additive field, mirrors the ``risk_markers`` annotation precedent —
# a downstream Stage reads it off the record, not out-of-band). Default ``None`` ⇒ every
# existing stack + record is byte-identical (a run with no ``schema_induction`` Stage never
# populates it, and an extractor with a ``None`` induced field uses its config labels exactly
# as before). See ADR-0038.
#
# v15 (W13 context enrichment): the new OPTIONAL ``context_enrichment`` Stage projects the
# already-computed knowledge graph back onto each chunk as a compact :class:`ContextHeader` (its
# canonical entities + top-k KG-neighbor triples), which the Export prepends to the EMBEDDING input
# ONLY — coherence goes into the vector + a metadata column, the stored chunk text stays the clean,
# byte-unchanged chunk (ADR-0039). ChunkRecord gains an optional ``context_header: ContextHeader |
# None`` (additive field, mirrors the ``induced_labels`` / ``risk_markers`` annotation precedent — a
# downstream consumer reads it off the record, not out-of-band). Default ``None`` ⇒ every existing
# stack + record is byte-identical (a run with no ``context_enrichment`` Stage never populates it,
# and the Export prepends nothing when it is absent). The ContextHeader is built deterministically
# (sorted, capped, content-addressed ids) from the run's existing EntityMention + GraphRecord
# streams; the optional LLM situating-sentence mode is a documented follow-on
# (``situating_sentence``
# stays a defaulted-None field). See ADR-0039.
#
# v16 (W16 chunk-level redaction): Redaction becomes a **chunk→chunk transform** — the fix for the
# document-level truncation leak (ADR-0042). Before W16, Redaction scanned the WHOLE parsed
# document, feeding it into a 768-token PII model window, so any PII past token 768 was SILENTLY
# truncated and leaked unmasked into the corpus. Redaction now runs per chunk (each ≤ the model
# window), so nothing is truncated. ChunkRecord gains three additive, defaulted fields the Redaction
# Stage populates: ``masked_content: str | None`` (the PII-handled variant of the chunk text — the
# clean chunk ``content`` stays UNMASKED so extraction still sees full entities, while the Export
# materializes ``masked_content`` for the PII-safe RAG corpus), ``pii_spans: list[PIISpan]`` (the
# chunk-local, counts-only detected spans — offsets into the chunk ``content``, source pages
# resolved through the chunk's offset_map+page_map, never the raw value), and
# ``redaction_disabled_for_sensitive: bool`` (the H-C1 §4 no-op floor carried per chunk). Default
# ``None`` / ``[]`` / ``False`` ⇒ every existing stack + record is byte-identical for its corpus
# text (a run with no Redaction Stage never populates them, the Export falls back to ``content``),
# mirroring the risk_markers/induced_labels/context_header annotation precedent. The document-level
# :class:`RedactionRecord` contract is RETAINED but DEPRECATED (the superseded doc-level path). See
# ADR-0042.
#
# v17 (the extraction redesign) carries TWO additive, defaulted contract additions that ship in the
# SAME revision — they land together, so no record ever exists with one and not the other:
#
# v17a (T2 per-chunk extraction): :class:`InducedLabels` gains a fourth slot, ``guard_types`` — the
# GUARDRAIL categories induced for a chunk, the learned-guardrail twin of ``pii_types``. The
# blessed Redaction checkpoint (``fastino/GLiNER2-Guardrails-PII-Multi``) is a guardrails+PII
# model, so its guard categories follow the SAME per-chunk rule the redesign applies to every
# LEARNED label set: induced per chunk, UNIONed onto a small UNIVERSAL guard safety floor that is
# never subsetted (a security invariant, not a relevance label — see
# :data:`latence_core.redaction_policy.GUARD_SAFETY_FLOOR`). Additive and defaulted to ``[]``, so a
# record produced without guard induction carries an empty slot and every extractor behaves exactly
# as before; the deterministic keyword ``screening.content_keyword`` pre-filter is a SEPARATE stage
# and is untouched.
#
# v17b (T3 type canonicalization): per-chunk induction makes the corpus's TYPE vocabulary drift
# (chunk A induces ``org``, chunk B ``organization``, chunk C ``company``), so a corpus-level
# Type-Consolidation Stage canonicalizes the vocabulary DOWNSTREAM — consistency is earned in the
# resolution phase, never imposed on induction by capping. :class:`EntityMention` and
# :class:`RelationMention` each gain an additive, defaulted ``raw_label: str | None``: when the
# Stage rewrites ``label`` to its canonical form it keeps the extractor's raw induced type here, so
# every remap is auditable on the record itself. ``None`` ⇒ never remapped (a stack with no
# Type-Consolidation Stage never populates it). The canonical vocabulary itself is a
# :class:`TypeVocabulary` — persistable (the Runner writes it beside the checkpoint) and MERGEABLE
# (an incremental delta run extends it without churning existing canonical entries).
SCHEMA_VERSION = 17


class Provenance(BaseModel):
    """The immutable chain from a record back to its source.

    Required fields identify the source document unambiguously; the optional
    offset fields locate a derived record within it (populated by Parse/Chunk).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    source_uri: str = Field(..., min_length=1, description="Storage URI read from.")
    file_name: str = Field(..., min_length=1)
    file_type: str = Field(..., min_length=1, description="Lowercased extension, no dot.")
    file_size: int = Field(..., ge=0, description="Source size in bytes.")
    document_id: str = Field(..., min_length=1, description="Content-addressed source doc id.")
    source_system: str | None = Field(default=None, description="Originating system.")
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)


class Classification(BaseModel):
    """Descriptive attributes of a document's content.

    Attached at ingest / after Parse and inherited by every downstream record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    language: str = Field(..., min_length=1, description="BCP-47-ish tag; 'und' if unknown.")
    category: str | None = Field(default=None, description="Content category, e.g. 'contract'.")
    sensitivity: str = Field(default="unknown", description="'public'/'internal'/'confidential'.")


class Record(BaseModel):
    """Base inter-Stage record: always carries Provenance and Classification.

    Both are required with no default, so constructing a record without them
    raises ``pydantic.ValidationError`` — the enforced Stage-boundary check.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    record_id: str = Field(..., min_length=1)
    provenance: Provenance
    classification: Classification


class PageSpan(BaseModel):
    """One page's half-open ``[char_start, char_end)`` span in the assembled text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(..., ge=1, description="1-based source page number.")
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> PageSpan:
        if self.char_end < self.char_start:
            msg = (
                f"Page {self.page_number} span is inverted: "
                f"char_end {self.char_end} < char_start {self.char_start}"
            )
            raise ValueError(msg)
        return self

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


class PageMap(BaseModel):
    """The page-boundary map carried on a parsed :class:`DocumentRecord`.

    ``pages`` are contiguous, non-overlapping, ordered spans covering the whole
    assembled text; ``total_chars`` is the assembled length (the end of the last
    page). A single-page document (plain text) has exactly one span — the map is
    always present after a successful Parse, never optional, so downstream
    offset→page resolution has one code path. The live resolver over this map is
    :class:`latence_core.pagemap.PageOffsetIndex`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    pages: list[PageSpan] = Field(..., min_length=1)
    total_chars: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _contiguous(self) -> PageMap:
        prev_end = 0
        for i, span in enumerate(self.pages):
            expected_page = i + 1
            if span.page_number != expected_page:
                msg = f"Page numbers must be 1..n in order; got {span.page_number} at index {i}"
                raise ValueError(msg)
            if span.char_start != prev_end:
                msg = (
                    f"Page {span.page_number} starts at {span.char_start}, "
                    f"expected {prev_end} (spans must be contiguous)"
                )
                raise ValueError(msg)
            prev_end = span.char_end
        if self.total_chars != prev_end:
            msg = f"total_chars {self.total_chars} != last page end {prev_end}"
            raise ValueError(msg)
        return self

    @classmethod
    def from_page_texts(cls, page_texts: list[str], separator: str = "\n\n") -> PageMapAssembly:
        """Assemble per-page text into one markdown string plus its page map.

        The Parser-side constructor: hand it the per-page decoded text in page
        order and it returns both the concatenated markdown and the
        :class:`PageMap` recording where each page landed — the two are built
        together so the boundaries are exact by construction, not reverse-engineered.

        A single-page (or zero-page, coerced to one empty page) document is the
        plain-text case: one span covering everything.
        """
        if not page_texts:
            page_texts = [""]
        spans: list[PageSpan] = []
        parts: list[str] = []
        cursor = 0
        sep_len = len(separator)
        last = len(page_texts) - 1
        for i, text in enumerate(page_texts):
            start = cursor
            parts.append(text)
            cursor += len(text)
            # The inter-page separator counts toward the *preceding* page's span so
            # the assembled offsets stay contiguous with no un-attributed gap.
            if i != last:
                parts.append(separator)
                cursor += sep_len
            spans.append(PageSpan(page_number=i + 1, char_start=start, char_end=cursor))
        assembled = "".join(parts)
        page_map = cls(pages=spans, total_chars=len(assembled))
        return PageMapAssembly(text=assembled, page_map=page_map)


class PageMapAssembly(BaseModel):
    """The assembled markdown plus its :class:`PageMap`, built together by Parse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    page_map: PageMap


class Disposition(StrEnum):
    """A parsed record's outcome disposition.

    ``PARSED`` is the healthy path. ``PARSE_ERROR`` marks a document the Parser
    could not decode (a corrupt PDF, an unsupported/spoofed format): the record is
    still emitted — carrying its Provenance and the failure reason — so the run
    does not crash and the failure is auditable in the Quality Report, but it
    carries no usable ``content`` and downstream Stages skip it.
    """

    PARSED = "parsed"
    PARSE_ERROR = "parse_error"


class DocumentRecord(Record):
    """A whole-document record: the parsed Parse output.

    ``content`` is the assembled markdown text; ``media_type`` records how to
    interpret it. ``page_map`` (present when ``disposition == PARSED``) records
    the source page boundaries so any downstream offset resolves back to its
    original page. ``disposition``/``error`` capture a graceful parse failure — a
    corrupt file yields a ``PARSE_ERROR`` record, never an exception that aborts
    the run.
    """

    content: str = Field(default="")
    media_type: str = Field(default="text/markdown")
    page_map: PageMap | None = Field(default=None)
    disposition: Disposition = Field(default=Disposition.PARSED)
    error: str | None = Field(
        default=None, description="Human-readable failure reason when disposition is PARSE_ERROR."
    )
    induced_labels: InducedLabels | None = Field(
        default=None,
        description=(
            "Per-document induced label schema (schema_induction Stage), so the per-document "
            "Redaction Stage UNIONs the induced pii_types onto its config labels. None (default) "
            "⇒ the document is unchanged and Redaction uses its config labels exactly as before."
        ),
    )

    @model_validator(mode="after")
    def _disposition_consistent(self) -> DocumentRecord:
        if self.disposition is Disposition.PARSE_ERROR:
            if self.error is None:
                msg = "A PARSE_ERROR DocumentRecord must carry an 'error' reason."
                raise ValueError(msg)
        elif self.error is not None:
            msg = "A PARSED DocumentRecord must not carry an 'error' reason."
            raise ValueError(msg)
        return self


class ParserInput(BaseModel):
    """A raw document handed to a Parser: source lineage plus undecoded content.

    This is the input side of the Parse seam (ADR-0019). Source produces a
    ``ParserInput`` per source document — Provenance and Classification stamped,
    but ``content`` still the raw bytes (or already-decoded text) exactly as read
    from Storage, NOT yet turned into markdown. The Parser owns decoding and the
    turn into a :class:`DocumentRecord`, so the seam fits real parsers (which
    consume PDF/image/office BYTES), not just the S1 plain-text passthrough.

    ``config`` carries per-input Parser hints (e.g. a forced encoding or an OCR
    language) so a Source or the Runner can steer a Parser without a new contract.

    ``content`` bytes are serialized to JSON as **base64** (``ser_json_bytes`` /
    ``val_json_bytes``), not as a UTF-8 string. This is a serialization-durability
    invariant, not a field change: a raw source document can be *arbitrary binary*
    (a PDF, an office file, a ZIP archive with a deflate stream), and Pydantic's
    default bytes-as-UTF-8 JSON encoding raises on any byte sequence that is not
    valid UTF-8 — which would tear the Runner's checkpoint the moment a genuinely
    binary file (e.g. a Screened ZIP) is threaded through the Source/Intake seam.
    base64 round-trips *any* bytes losslessly (``model_validate_json`` yields the
    identical bytes), so checkpoint/resume survives the messy, dangerous corpus the
    demo is built to ingest. The model fields are unchanged, so the inter-Stage
    contract (and its ``schema_version``) is unchanged — only the on-disk JSON
    encoding of already-binary content is made lossless.
    """

    model_config = ConfigDict(
        extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    schema_version: int = Field(default=SCHEMA_VERSION)
    provenance: Provenance
    classification: Classification
    content: bytes | str = Field(
        ..., description="Raw, undecoded source content — bytes for binary formats, str if text."
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Per-input Parser hints (encoding, OCR lang, …)."
    )


# -- S3: Screening -----------------------------------------------------------


class ScreeningCheckpoint(StrEnum):
    """Which of Screening's two checkpoints produced a finding (CONTEXT glossary).

    ``INTAKE`` runs before Parse over the raw document (malware, zip bombs,
    file-type spoofing, oversized/corrupt files). ``CONTENT`` runs after Chunk
    over the retrieval-sized text (prompt injection, harmful content, sensitivity
    escalation).
    """

    INTAKE = "intake"
    CONTENT = "content"


class ScreeningVerdict(StrEnum):
    """What Screening decided about a record.

    ``PASS`` — nothing found; the record proceeds untouched. ``FLAG`` — a risk was
    found but the record still proceeds, carrying a :class:`RiskMarker` so a RAG
    consumer can exclude it. ``QUARANTINE`` — the record is dangerous and is
    removed from every downstream Stage and Export (retained and inspectable with
    the reason, CONTEXT ``Quarantine``).
    """

    PASS = "pass"
    FLAG = "flag"
    QUARANTINE = "quarantine"


class RiskMarker(BaseModel):
    """A risk flag Content Screening attaches to a record so it propagates.

    The marker survives into the corpus (it is a field on the record contract, not
    an out-of-band note) so a downstream RAG consumer can filter on it — the S3
    acceptance criterion "a risk marker that survives into the corpus so RAG
    consumers can exclude it".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    category: str = Field(..., min_length=1, description="e.g. 'prompt_injection', 'harmful'.")
    detector: str = Field(..., min_length=1, description="Provider that raised the marker.")
    score: float = Field(..., ge=0.0, le=1.0, description="Detector confidence 0..1.")
    reason: str = Field(..., min_length=1, description="Human-readable justification.")


class InducedLabels(BaseModel):
    """Per-document label schema the ``schema_induction`` Stage induced (CONTEXT
    ``Classification``).
    W4 restores the "point it at a folder and it discovers its own schema" property: an OPTIONAL
    :class:`~latence_core.capability.LabelInducer` Stage (a small OpenAI-compatible LLM with
    structured JSON output) reads a document and induces the entity / relation / PII **types**
    actually evidenced in it, then annotates every record it produced with this carrier so the
    downstream extraction Stages can UNION these induced types onto their config label floor
    (``config_labels ∪ induced``, deduped, order-stable). Like :class:`RiskMarker`, it is a
    **field on the record contract** (not an out-of-band note), so it survives into the corpus and
    a downstream Stage reads it off the record it already holds — the additive extractor seam.

    Every list is already **sanitized** by the inducing Provider (control chars stripped, length
    ≤ 64, count ≤ 40, deduped, normalized) — an :class:`InducedLabels` is trusted, bounded data a
    zero-shot extractor prompt can consume, never executed. ``model_id`` records which LLM induced
    it (audit); ``confidences`` maps a label to the model's 0..1 score for it (auditable, optional
    — a label with no score is simply absent from the map). A slot the model evidenced no types for
    is an empty list, not a fabricated one.

    v17 adds the fourth slot ``guard_types`` (guardrail categories), so every LEARNED label set a
    downstream Provider prompts with — entity, relation, PII, guardrail — has a per-record slot.
    Whether a slot's config labels are a droppable RELEVANCE floor or a non-subsettable UNIVERSAL
    SAFETY floor is the consumer's decision, expressed at the
    :func:`latence_core.induced.effective_labels` seam via
    :class:`~latence_core.induced.LabelFloor`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    entity_types: list[str] = Field(
        default_factory=list, description="Sanitized entity types induced for the document."
    )
    relation_types: list[str] = Field(
        default_factory=list, description="Sanitized relation types induced for the document."
    )
    pii_types: list[str] = Field(
        default_factory=list, description="Sanitized PII types induced for the document."
    )
    guard_types: list[str] = Field(
        default_factory=list,
        description=(
            "Sanitized GUARDRAIL categories induced for the record (v17) — the learned-guardrail "
            "twin of ``pii_types``, consumed by a guardrails-capable Redaction Provider. Like "
            "every other slot these are RELEVANCE labels scoped to the record; the UNIVERSAL guard "
            "safety floor (latence_core.redaction_policy.GUARD_SAFETY_FLOOR) is unioned on top and "
            "is never subsetted. Empty (default) ⇒ no guard induction ran."
        ),
    )
    model_id: str = Field(
        ..., min_length=1, description="The LLM that induced these labels (audit)."
    )
    confidences: dict[str, float] = Field(
        default_factory=dict, description="Optional per-label 0..1 confidence (auditable)."
    )


class ScreeningFinding(BaseModel):
    """One Screening decision about one record, recorded for the Quality Report.

    Emitted at either checkpoint. A ``QUARANTINE`` verdict names the removed
    document; a ``FLAG`` carries the :class:`RiskMarker` that was propagated. The
    finding is the audit trail: every Quarantine disposition and reason surfaces
    in the Quality Report (S3 AC), and a flag records what was marked and why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    checkpoint: ScreeningCheckpoint
    verdict: ScreeningVerdict
    document_id: str = Field(..., min_length=1)
    detector: str = Field(..., min_length=1, description="Screener Provider name.")
    category: str = Field(..., min_length=1, description="Risk category checked.")
    reason: str = Field(..., min_length=1)
    marker: RiskMarker | None = Field(
        default=None, description="Present when verdict == FLAG: the propagated marker."
    )

    @model_validator(mode="after")
    def _verdict_consistent(self) -> ScreeningFinding:
        if self.verdict is ScreeningVerdict.FLAG and self.marker is None:
            msg = "A FLAG ScreeningFinding must carry the RiskMarker it propagated."
            raise ValueError(msg)
        if self.verdict is not ScreeningVerdict.FLAG and self.marker is not None:
            msg = "Only a FLAG ScreeningFinding may carry a RiskMarker."
            raise ValueError(msg)
        return self


class QuarantineRecord(Record):
    """A record Screening removed from the pipeline (CONTEXT ``Quarantine``).

    Excluded from every downstream Stage and Export, but **retained and
    inspectable**: Provenance is intact, the offending checkpoint/detector/reason
    are recorded, and the raw byte size is kept so the audit knows what was held
    back. It never reaches Parse (intake) or the corpus (content). A
    ``QuarantineRecord`` carries no usable ``content`` — the point of Quarantine
    is that the data does not flow, only its audit trail.
    """

    checkpoint: ScreeningCheckpoint
    detector: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    quarantined_bytes: int = Field(..., ge=0, description="Byte size of the withheld source.")


# -- S3: Chunk ---------------------------------------------------------------


class OffsetSegment(BaseModel):
    """One contiguous run mapping stripped-``content`` offsets to original offsets.

    Chunk ``content`` is the markup-stripped text; markup stripping only ever
    **deletes** characters (``chunking.strip_markup``), so a chunk's surviving
    characters form contiguous runs of the ORIGINAL parsed markdown. Within a run
    the map is a constant shift: original offset ``= doc_start + (local -
    local_start)``. A new segment begins at each stripped gap, so the whole
    stripped→original map is this short list of run breakpoints (typically a handful
    per chunk), not a per-character table.

    ``local_start`` is the offset into the chunk's ``content``; ``doc_start`` is the
    original-markdown offset of that same character. The first segment always starts
    at ``local_start == 0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    local_start: int = Field(..., ge=0, description="Offset into the chunk's stripped content.")
    doc_start: int = Field(..., ge=0, description="Original-markdown offset of that character.")


class OffsetMap(BaseModel):
    """Run-length map from a chunk's stripped ``content`` to original offsets.

    Carried on a :class:`ChunkRecord` so a downstream Stage that finds a *sub-chunk*
    offset (an entity mention's chunk-local match) can recover that offset's TRUE
    position in the parent document's assembled markdown — not merely the lower bound
    ``char_start + local_offset`` gives once markup was stripped before the mention
    within its chunk (ADR-0022 corrected, ADR-0031). The live resolver over this map
    is :class:`latence_core.offsetmap.OffsetIndex`.

    ``segments`` are ordered by ``local_start`` (strictly increasing, first at 0);
    ``length`` is ``len(content)`` — the exclusive upper bound of the local domain, so
    a half-open local span's original end is resolvable. A markup-free chunk has
    exactly one segment ``(0, char_start)``: the map is then the identity shift and a
    mention resolves to ``char_start + local`` exactly as before.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    segments: list[OffsetSegment] = Field(..., min_length=1)
    length: int = Field(..., ge=0, description="Length of the chunk content this map covers.")

    @model_validator(mode="after")
    def _ordered_within_bounds(self) -> OffsetMap:
        first = self.segments[0].local_start
        if first != 0:
            msg = f"First OffsetSegment must start at local_start 0, got {first}"
            raise ValueError(msg)
        prev = -1
        for seg in self.segments:
            if seg.local_start <= prev:
                msg = (
                    "OffsetSegments must have strictly increasing local_start; "
                    f"got {seg.local_start} after {prev}"
                )
                raise ValueError(msg)
            if self.length and seg.local_start >= self.length:
                msg = (
                    f"OffsetSegment local_start {seg.local_start} is outside "
                    f"content length {self.length}"
                )
                raise ValueError(msg)
            prev = seg.local_start
        return self


class ContextHeader(BaseModel):
    """The KG projected back onto a chunk for retrieval coherence (CONTEXT ``Context Enrichment``).

    W13 (ADR-0039) restores the connections a chunk needs for retrieval relevance — its canonical
    entities and their top-k knowledge-graph neighbor triples — that the RAG export otherwise
    discards by embedding each chunk in isolation. The OPTIONAL ``context_enrichment`` Stage builds
    this compact header per chunk from the run's already-computed :class:`EntityMention` +
    :class:`GraphRecord` streams and the Export prepends it to the **EMBEDDING input only**: the
    coherence inflates the vector + a metadata column, never the stored chunk text (the LLM window
    stays the clean chunk, ``masked_content``/``content`` byte-unchanged).

    Like :class:`RiskMarker` / :class:`InducedLabels` it is a **field on the record contract** (not
    an out-of-band note), so it survives into the corpus and a downstream retriever reads it off the
    chunk it already holds. It is deliberately COMPACT (the bloat discipline, ADR-0039): compact
    rendered triples not neighbor prose, a top-k neighbor + confidence cap (the hub-entity guard),
    and
    join keys not raw KG text (query-time KG expansion is the retriever's job — ADR-0017, files not
    a
    live DB; W13 only emits the keys).

    * ``entities`` — the chunk's canonical entity names (sorted, deduped), and ``entity_types`` the
      aligned entity type per name (same length + order as ``entities``). ``entity_confidences`` is
      the aligned per-entity 0..1 admission confidence (ADR-0053) — the value the enrichment gate
      admitted the entity at, carried so a downstream composing SparseEmbedder's
      ``confidence_threshold`` gates weak NER at index time by the identical rule; ``[]`` for a
      header built without it.
    * ``neighbor_triples`` — the chunk's top KG-neighbor relations rendered ``A —label→ B``,
      confidence-sorted and CAPPED (``max_neighbors_per_entity`` per entity, ``max_triples``
      overall)
      — the anti-bloat + hub-entity guard: a corpus-wide hub entity contributes only its TOP
      relations, never all of them.
    * ``kg_node_ids`` — the deterministic content-addressed node ids of the chunk's own entities
    (the
      join keys a downstream retriever hybridises/reranks on).
    * ``neighbor_node_ids`` — the node ids of the neighbor entities the triples reach (the
    query-time
      KG-expansion keys), sorted + deduped.
    * ``kg_node_centrality`` / ``kg_node_community`` — the OPTIONAL ADR-0051 graph-feature
      projection: each entity's index-time **centrality** score and **community** id, aligned 1:1
      with ``kg_node_ids`` (same length + order), so a downstream ranker reads a chunk's own
      entities' structural importance/cluster off the header it already holds. Both are ``[]`` when
      the ``graph_features`` enrichment is OFF (the nodes carry no centrality) — byte-identical to a
      run without it — and both full-length (aligned with ``kg_node_ids``) when it is ON.
    * ``situating_sentence`` — OPTIONAL LLM-generated Contextual-Retrieval-style sentence
      (deterministic=False, a documented follow-on, ADR-0039); ``None`` in the deterministic default
      reference build. ``model_id`` records the LLM when that mode is used (audit), ``None`` for the
      deterministic build.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    entities: list[str] = Field(
        default_factory=list, description="The chunk's canonical entity names (sorted, deduped)."
    )
    entity_types: list[str] = Field(
        default_factory=list, description="Entity type per name (aligned with ``entities``)."
    )
    entity_confidences: list[float] = Field(
        default_factory=list,
        description=(
            "Per-entity 0..1 admission confidence aligned with ``entities`` (ADR-0053): the value "
            "at which the enrichment gate admitted the entity, so a downstream composing "
            "SparseEmbedder's ``confidence_threshold`` applies the identical rule at index time. "
            "``[]`` for a header built without it (byte-identical to a run without the field)."
        ),
    )
    neighbor_triples: list[str] = Field(
        default_factory=list,
        description="Top-k KG-neighbor relations 'A —label→ B' (confidence-sorted, capped).",
    )
    kg_node_ids: list[str] = Field(
        default_factory=list, description="Node ids of the chunk's own entities (join keys)."
    )
    neighbor_node_ids: list[str] = Field(
        default_factory=list,
        description="Node ids the triples reach (query-time KG-expansion keys; sorted, deduped).",
    )
    kg_node_centrality: list[float] = Field(
        default_factory=list,
        description="Per-entity centrality, aligned with ``kg_node_ids`` (ADR-0051); [] if off.",
    )
    kg_node_community: list[str] = Field(
        default_factory=list,
        description="Per-entity community id, aligned with ``kg_node_ids`` (ADR-0051); [] if off.",
    )
    situating_sentence: str | None = Field(
        default=None,
        description="OPTIONAL LLM situating sentence (deterministic=False follow-on); None here.",
    )
    model_id: str | None = Field(
        default=None, description="The LLM that wrote the situating sentence (audit); None if none."
    )

    @model_validator(mode="after")
    def _entities_aligned(self) -> ContextHeader:
        if len(self.entities) != len(self.entity_types):
            msg = (
                "A ContextHeader's entities and entity_types must be the same length "
                f"(got {len(self.entities)} names, {len(self.entity_types)} types)."
            )
            raise ValueError(msg)
        # ADR-0051: the graph-feature projection is either OFF (empty) or aligned 1:1 with the
        # chunk's own entity node ids — never a partial per-entity map (anti-false-green).
        for name, values in (
            ("kg_node_centrality", self.kg_node_centrality),
            ("kg_node_community", self.kg_node_community),
        ):
            if values and len(values) != len(self.kg_node_ids):
                msg = (
                    f"A ContextHeader's {name} must be empty or aligned with kg_node_ids "
                    f"(got {len(values)} values for {len(self.kg_node_ids)} node ids)."
                )
                raise ValueError(msg)
        return self

    @property
    def is_empty(self) -> bool:
        """True if the header carries no entities and no triples (nothing to prepend)."""
        return not self.entities and not self.neighbor_triples


class ChunkRecord(Record):
    """One retrieval-sized piece of a parsed document (CONTEXT ``Chunk``).

    Chunk splits a :class:`DocumentRecord`'s markdown into retrieval-sized pieces
    while preserving offsets, page alignment, Provenance and Classification
    losslessly. Every chunk carries:

    * ``char_start``/``char_end`` on its **Provenance** — the half-open span the
      chunk occupies in the parent document's assembled markdown, so any chunk
      resolves back to its exact source offsets (round-trip: chunk → offset →
      original page via :class:`~latence_core.pagemap.PageOffsetIndex`);
    * ``page_start``/``page_end`` on its **Provenance** — the source page(s) the
      chunk spans, resolved from the parent's page map;
    * the parent's **Classification**, inherited unchanged;
    * ``risk_markers`` — any :class:`RiskMarker` Content Screening attached, which
      survive into the corpus so RAG consumers can exclude the chunk.

    ``document_record_id`` links the chunk back to its parent document; ``index``
    is the chunk's ordinal within that document (0-based, contiguous).

    ``page_map`` carries the parent document's page-boundary map forward so a
    *downstream* Stage that finds a sub-chunk offset (an entity mention) can
    resolve **that offset's own** source page through
    :class:`~latence_core.pagemap.PageOffsetIndex` — not merely inherit the
    chunk's whole ``[page_start, page_end]`` range. Without it, a mention on a
    later page of a multi-page chunk would cite the chunk's first page. Optional
    (a chunk produced before S4, or from a document with no page map, has ``None``
    and downstream falls back to the inherited chunk page range).

    ``offset_map`` carries the stripped-``content``→original-markdown offset map
    (:class:`OffsetMap`) so a downstream Stage can recover a sub-chunk offset's TRUE
    original position. ``content`` is markup-stripped, so ``char_start + local`` is
    only a lower bound on a mention's real offset once markup was stripped before it;
    the map corrects that (ADR-0031). Optional and additive — absent (pre-v11 chunk,
    or no map), downstream degrades to the ``char_start + local`` shift.
    """

    document_record_id: str = Field(..., min_length=1, description="Parent DocumentRecord id.")
    index: int = Field(..., ge=0, description="0-based chunk ordinal within the document.")
    content: str = Field(default="")
    media_type: str = Field(default="text/markdown")
    token_count: int = Field(..., ge=0, description="Budgeted token count for the chunk text.")
    page_map: PageMap | None = Field(
        default=None,
        description="Parent document's page map, so a sub-chunk offset resolves to its own page.",
    )
    offset_map: OffsetMap | None = Field(
        default=None,
        description=(
            "Stripped-content→original-markdown offset map, so a sub-chunk offset (an entity "
            "mention) resolves to its TRUE original offset, not the lower bound char_start+local."
        ),
    )
    risk_markers: list[RiskMarker] = Field(
        default_factory=list, description="Content-Screening flags that propagate into the corpus."
    )
    induced_labels: InducedLabels | None = Field(
        default=None,
        description=(
            "Per-document induced label schema the OPTIONAL schema_induction Stage attached, so a "
            "downstream extractor UNIONs these types onto its config labels (like risk_markers). "
            "None (default) ⇒ the chunk is unchanged and every extractor uses its config labels."
        ),
    )
    context_header: ContextHeader | None = Field(
        default=None,
        description=(
            "The KG projected back onto this chunk (canonical entities + top-k neighbor triples) "
            "the OPTIONAL context_enrichment Stage attached, which the Export prepends to the "
            "EMBEDDING input ONLY — the stored content is byte-unchanged (W13, ADR-0039). None "
            "(default) ⇒ the chunk is unchanged and the Export prepends nothing (byte-identical to "
            "a run without the Stage), mirroring the risk_markers/induced_labels precedent."
        ),
    )
    masked_content: str | None = Field(
        default=None,
        description=(
            "The PII-handled variant of this chunk's text the OPTIONAL Redaction Stage produced "
            "(W16, ADR-0042): the chunk text with every non-TAG PII span replaced by its "
            "placeholder. The clean ``content`` stays UNMASKED (so extraction still sees full "
            "entities); the Export materializes ``masked_content`` for the PII-safe RAG corpus. "
            "None (default) ⇒ no Redaction Stage ran, and the Export falls back to ``content`` "
            "(byte-identical to a run without the Stage), mirroring the context_header precedent. "
            "Redaction runs per CHUNK (each ≤ the model window) so no PII is truncated — the fix "
            "for the document-level 768-token truncation leak (ADR-0042)."
        ),
    )
    pii_spans: list[PIISpan] = Field(
        default_factory=list,
        description=(
            "The chunk-local PII spans the Redaction Stage detected (W16, ADR-0042): each span's "
            "``char_start``/``char_end`` are offsets into THIS chunk's ``content`` (the coordinate "
            "system ``masked_content`` is masked in), while ``page_start``/``page_end`` resolve to "
            "the original source page through the chunk's offset_map+page_map. Counts-only: a span "
            "never stores the raw PII value (S6 AC). Empty (default) ⇒ no Redaction Stage ran, "
            "or a clean chunk."
        ),
    )
    redaction_disabled_for_sensitive: bool = Field(
        default=False,
        description=(
            "True when this chunk's Classification.sensitivity is in the Redactor's sensitive set "
            "(confidential/restricted/…) yet the applied policy ran NO detectors (skip / empty "
            "types / empty detector set) — so an empty ``pii_spans`` here means 'the control was "
            "OFF', NOT 'no PII found' (the H-C1 §4 no-op floor carried per chunk, W16/ADR-0042)."
        ),
    )

    @property
    def flagged(self) -> bool:
        """True if Content Screening attached any risk marker to this chunk."""
        return bool(self.risk_markers)


# -- S4: Entity Extraction ---------------------------------------------------


class EntityMention(Record):
    """One typed entity mention found in a chunk's text (CONTEXT ``Entity Extraction``).

    Entity Extraction finds typed entity mentions in chunked text (zero-shot NER):
    every mention is an inter-Stage :class:`Record` — it carries Provenance and
    Classification like any other — so it can be threaded through the DAG and
    exported. A mention records:

    * ``label`` — the entity type it was tagged as. The label set is supplied per
      Pipeline (zero-shot); no label set is hardcoded in core.
    * ``text`` — the exact surface string of the mention as it appears in the
      parent document's assembled markdown.
    * ``confidence`` — the extractor's 0..1 score for the mention.
    * ``char_start``/``char_end`` on its **Provenance** — the half-open span the
      mention occupies **in the parent document's assembled markdown** (not in the
      chunk-local text), so a mention resolves back to its exact source offsets and,
      through the parent's page map, to its original source page(s) (the S4
      round-trip: mention → offset → original page, via
      :class:`~latence_core.pagemap.PageOffsetIndex`).
    * ``page_start``/``page_end`` on its **Provenance** — the source page(s) the
      mention span falls on, resolved from the parent's page map.
    * the parent chunk's **Classification**, inherited unchanged.

    ``chunk_record_id``/``document_record_id`` link the mention back to the chunk it
    was found in and to the document that chunk came from; ``index`` is the
    mention's ordinal within its chunk (0-based).
    """

    document_record_id: str = Field(..., min_length=1, description="Parent DocumentRecord id.")
    chunk_record_id: str = Field(..., min_length=1, description="Parent ChunkRecord id.")
    index: int = Field(..., ge=0, description="0-based mention ordinal within the chunk.")
    label: str = Field(..., min_length=1, description="Zero-shot entity type (per-Pipeline label).")
    text: str = Field(..., min_length=1, description="The mention's exact surface string.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extractor confidence 0..1.")
    # T3 (type canonicalization): the label the EXTRACTOR produced, kept verbatim when a
    # corpus-level Type-Consolidation Stage rewrites ``label`` to its canonical form. The audit
    # trail for the remap — an operator reading a mention can always see which raw induced type it
    # came from, so a consolidation decision is inspectable and correctable, never silent.
    # ``None`` means "never remapped" (``label`` IS the raw label), so a stack with no
    # Type-Consolidation Stage is byte-identical to before the field existed apart from the
    # explicit ``null``.
    raw_label: str | None = Field(
        default=None,
        min_length=1,
        description="The pre-canonicalization induced type, when a Type-Consolidation Stage ran.",
    )

    @model_validator(mode="after")
    def _span_present(self) -> EntityMention:
        prov = self.provenance
        if prov.char_start is None or prov.char_end is None:
            msg = "An EntityMention's Provenance must carry a char_start and char_end span."
            raise ValueError(msg)
        if prov.char_end < prov.char_start:
            msg = (
                f"EntityMention span is inverted: char_end {prov.char_end} "
                f"< char_start {prov.char_start}"
            )
            raise ValueError(msg)
        if prov.page_start is None or prov.page_end is None:
            msg = "An EntityMention's Provenance must carry a page_start and page_end."
            raise ValueError(msg)
        return self


# -- S5: Relation Extraction -------------------------------------------------


class RelationMention(Record):
    """One typed relation between two entity mentions (CONTEXT ``Relation Extraction``).

    Relation Extraction finds typed relations between entity mentions *within a
    document*. A relation is an inter-Stage :class:`Record` — it carries Provenance
    and Classification like :class:`EntityMention` — so it threads through the DAG,
    is checkpointed/resumed uniformly, and is exported to the corpus directly. A
    relation records:

    * ``label`` — the relation type (e.g. ``works_for``, ``located_in``). The
      relation label set is supplied per Pipeline (zero-shot / prompt-driven); no
      label set is hardcoded in core, mirroring Entity Extraction.
    * ``head_mention_id``/``tail_mention_id`` — the ``record_id`` of the head (subject)
      and tail (object) :class:`EntityMention` the relation connects. The relation is
      directed head → tail.
    * ``head_text``/``tail_text`` — the head/tail surface strings, carried denormalised
      so a consumer (and Graph Assembly's Evidence) reads the relation without
      re-joining to the mention set.
    * ``confidence`` — the extractor's 0..1 score for the relation.
    * ``char_start``/``char_end`` on its **Provenance** — the half-open span in the
      parent document's assembled markdown that *covers both* endpoints (from the
      earlier mention's start to the later mention's end), so the relation resolves
      back to source offsets and, through the parent's page map, to its source
      page(s). The page span is resolved the same way (mention → offset → page).
    * the parent document's **Classification**, inherited unchanged.

    ``document_record_id`` links the relation back to the document its two mentions
    belong to; ``index`` is the relation's ordinal within that document (0-based).

    The head and tail must be distinct mentions (no self-relation), and the char
    span must be present and well-ordered — a relation with no span or an inverted
    span fails validation, so a Provider that forgets to resolve the endpoint offsets
    cannot silently emit an un-resolvable relation.
    """

    document_record_id: str = Field(..., min_length=1, description="Parent DocumentRecord id.")
    index: int = Field(..., ge=0, description="0-based relation ordinal within the document.")
    label: str = Field(
        ..., min_length=1, description="Zero-shot relation type (per-Pipeline label)."
    )
    head_mention_id: str = Field(..., min_length=1, description="record_id of the head mention.")
    tail_mention_id: str = Field(..., min_length=1, description="record_id of the tail mention.")
    head_text: str = Field(..., min_length=1, description="Head mention surface string.")
    tail_text: str = Field(..., min_length=1, description="Tail mention surface string.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extractor confidence 0..1.")
    # T3: the raw induced relation type, kept for audit when a Type-Consolidation Stage rewrote
    # ``label`` to its canonical form (the EntityMention.raw_label rationale, relation side).
    raw_label: str | None = Field(
        default=None,
        min_length=1,
        description="The pre-canonicalization induced type, when a Type-Consolidation Stage ran.",
    )

    @model_validator(mode="after")
    def _relation_valid(self) -> RelationMention:
        if self.head_mention_id == self.tail_mention_id:
            msg = (
                "A RelationMention's head and tail must be distinct mentions "
                f"(both are {self.head_mention_id!r})."
            )
            raise ValueError(msg)
        prov = self.provenance
        if prov.char_start is None or prov.char_end is None:
            msg = "A RelationMention's Provenance must carry a char_start and char_end span."
            raise ValueError(msg)
        if prov.char_end < prov.char_start:
            msg = (
                f"RelationMention span is inverted: char_end {prov.char_end} "
                f"< char_start {prov.char_start}"
            )
            raise ValueError(msg)
        if prov.page_start is None or prov.page_end is None:
            msg = "A RelationMention's Provenance must carry a page_start and page_end."
            raise ValueError(msg)
        return self


# -- T3: corpus-wide TYPE canonicalization -----------------------------------


class TypeKind(StrEnum):
    """Which label vocabulary a :class:`CanonicalType` belongs to.

    Entity types and relation types are canonicalized in the SAME pass but never mixed: an
    entity type ``organization`` and a relation type ``organization of`` are different
    vocabularies, and folding one into the other would corrupt the graph. Every lookup is
    therefore keyed by ``(kind, label)``.
    """

    ENTITY = "entity"
    RELATION = "relation"


class CanonicalType(BaseModel):
    """One canonical type and every raw induced label folded into it (T3).

    Per-chunk induction produces type drift — ``org`` here, ``organization`` there,
    ``company`` in a third chunk — because each chunk induces its own labels. The
    Type-Consolidation Stage clusters those synonymous surfaces and elects ONE canonical
    label per cluster; this record is the cluster.

    ``aliases`` holds every OTHER raw label in the cluster (sorted, canonical excluded), so the
    alias→canonical map an operator reads on the Quality Report is exactly
    ``{alias: canonical for alias in aliases}``. ``strategies`` names the clustering rungs that
    produced the folds (``normalized``/``alias``/``acronym``/``prefix``/``head_noun``/
    ``embedding``), so a suspicious merge is traceable to the rule that made it — the audit the
    entity resolver's ``merge_decisions`` provides on the entity side. ``mention_count`` is how
    many mentions/relations carried a member label, the salience signal an operator sorts by.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    kind: TypeKind
    canonical: str = Field(..., min_length=1, description="The elected human-facing type label.")
    aliases: list[str] = Field(
        default_factory=list, description="Every other raw label folded here (sorted)."
    )
    mention_count: int = Field(
        default=0, ge=0, description="Mentions/relations that carried a member label."
    )
    strategies: list[str] = Field(
        default_factory=list, description="Clustering rungs that produced the folds (sorted)."
    )

    @model_validator(mode="after")
    def _canonical_not_aliased(self) -> CanonicalType:
        if self.canonical in self.aliases:
            msg = (
                f"CanonicalType {self.canonical!r} lists itself as an alias — the alias set is "
                "the OTHER labels folded into the canonical one."
            )
            raise ValueError(msg)
        if len(set(self.aliases)) != len(self.aliases):
            msg = f"CanonicalType {self.canonical!r} has duplicate aliases."
            raise ValueError(msg)
        return self

    @property
    def labels(self) -> list[str]:
        """Every raw label this cluster covers — the canonical one plus its aliases."""
        return [self.canonical, *self.aliases]


class TypeVocabulary(BaseModel):
    """The corpus's canonical type vocabulary + its alias map (T3) — persistable and mergeable.

    The artifact the Type-Consolidation Stage emits: the ``M`` canonical types the run's ``N``
    raw induced types collapsed to, each with the aliases folded into it. It is a plain
    serializable model (not a :class:`Record`) because it describes the corpus's SCHEMA, not a
    span of a document — it has no Provenance to carry. The Runner persists it beside the
    Stage's checkpoint, which is what makes it survive the run (ADR-0017: files are the source
    of truth).

    **Mergeability is a first-class property** (designed for the T4 incremental-delta lane): a
    later run merges its freshly-induced vocabulary INTO the persisted one with
    :meth:`merge`, which is deliberately *append-don't-churn* — an existing canonical label is
    NEVER renamed by a later run (that would rewrite the type of every node already in the KG);
    new aliases attach to the existing cluster and genuinely new types are appended.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    types: list[CanonicalType] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_cluster_per_label(self) -> TypeVocabulary:
        """A raw label may belong to exactly ONE cluster per kind (the map must be a function)."""
        seen: set[tuple[TypeKind, str]] = set()
        for entry in self.types:
            for label in entry.labels:
                key = (entry.kind, label)
                if key in seen:
                    msg = (
                        f"Type label {label!r} ({entry.kind.value}) appears in more than one "
                        "cluster — the alias→canonical map would be ambiguous."
                    )
                    raise ValueError(msg)
                seen.add(key)
        return self

    def of_kind(self, kind: TypeKind) -> list[CanonicalType]:
        """The clusters of one kind, in emission order (canonical-label order)."""
        return [entry for entry in self.types if entry.kind is kind]

    def alias_map(self, kind: TypeKind) -> dict[str, str]:
        """``{raw label: canonical label}`` for one kind — the operator-facing consolidation map.

        Only genuine ALIASES appear (a type that was already canonical maps to itself and is
        omitted), so the map an operator reads on the Quality Report is exactly the set of
        rewrites the run performed — an empty map means nothing was consolidated.
        """
        return {
            alias: entry.canonical
            for entry in self.types
            if entry.kind is kind
            for alias in entry.aliases
        }

    def canonical_for(self, kind: TypeKind, label: str) -> str:
        """The canonical label for a raw ``label``; the label itself when it is unknown.

        Unknown-is-identity keeps the remap TOTAL: a label the vocabulary never saw (a delta
        run's brand-new type, a hand-written relation label) passes through unchanged rather
        than being dropped or mapped to a wrong cluster.
        """
        for entry in self.types:
            if entry.kind is kind and (label == entry.canonical or label in entry.aliases):
                return entry.canonical
        return label

    def merge(self, other: TypeVocabulary) -> TypeVocabulary:
        """Fold ``other`` into this vocabulary — append-don't-churn (the T4 delta contract).

        For every cluster in ``other``: if any of its labels already belongs to a cluster of the
        same kind here, that EXISTING cluster absorbs it — the existing canonical label is kept
        (renaming it would silently retype every node already in the KG), its aliases are unioned
        and strategies unioned. Otherwise the cluster is appended verbatim. When a cluster in
        ``other`` bridges two existing clusters, the first match (in this vocabulary's order) wins
        and the bridged labels join it — deterministic, and it never splits an existing cluster.

        **``mention_count`` is the LATEST observation, not a running sum** (T4). ``self`` is the
        vocabulary persisted by an earlier run and ``other`` the one a later run built over the
        whole CURRENT corpus, so the two count the same mentions: summing them would report a
        multiple of the true corpus salience on the Quality Report, and would grow without bound
        as deltas accumulate — the count would also differ between two applications of the same
        delta, breaking idempotency. So a cluster ``other`` observed takes ``other``'s count
        (several ``other`` clusters folding into ONE existing cluster do sum, because those are
        disjoint observations *within* one run), and a cluster ``other`` never observed keeps the
        count it was persisted with.

        The result is ordered by ``(kind, canonical)`` and every alias list sorted, so merging
        is associative in outcome and byte-stable: the same pair of vocabularies always merges
        to the identical document.
        """
        merged: list[dict[str, Any]] = [
            {
                "kind": e.kind,
                "canonical": e.canonical,
                "aliases": set(e.aliases),
                "mention_count": e.mention_count,
                "strategies": set(e.strategies),
            }
            for e in self.types
        ]
        # Which slots ``other`` has already contributed a count to in THIS merge — the first
        # contribution REPLACES the persisted count (same corpus, re-counted), later ones add to
        # it (two of ``other``'s clusters folding into one existing cluster are disjoint counts).
        recounted: set[int] = set()
        for entry in other.types:
            labels = set(entry.labels)
            target = next(
                (
                    (index, slot)
                    for index, slot in enumerate(merged)
                    if slot["kind"] is entry.kind
                    and (labels & ({str(slot["canonical"])} | set(slot["aliases"])))
                ),
                None,
            )
            if target is None:
                recounted.add(len(merged))
                merged.append(
                    {
                        "kind": entry.kind,
                        "canonical": entry.canonical,
                        "aliases": set(entry.aliases),
                        "mention_count": entry.mention_count,
                        "strategies": set(entry.strategies),
                    }
                )
                continue
            index, slot = target
            canonical = str(slot["canonical"])
            slot["aliases"] = (set(slot["aliases"]) | labels) - {canonical}
            slot["mention_count"] = (
                int(slot["mention_count"]) + entry.mention_count
                if index in recounted
                else entry.mention_count
            )
            recounted.add(index)
            slot["strategies"] = set(slot["strategies"]) | set(entry.strategies)
        return TypeVocabulary(
            types=[
                CanonicalType(
                    kind=slot["kind"],
                    canonical=str(slot["canonical"]),
                    aliases=sorted(set(slot["aliases"])),
                    mention_count=int(slot["mention_count"]),
                    strategies=sorted(set(slot["strategies"])),
                )
                for slot in sorted(
                    merged, key=lambda s: (str(TypeKind(s["kind"]).value), str(s["canonical"]))
                )
            ]
        )


# -- S6: Redaction (PII) -----------------------------------------------------


class RedactionAction(StrEnum):
    """What a Redaction Provider did to a detected PII span (CONTEXT ``Redaction``).

    ``MASK`` replaces the span with a fixed mask token (e.g. ``[EMAIL]``) — the
    default, lossy, irreversible. ``REPLACE`` substitutes a deterministic surrogate
    of the same type (a pseudonym) so the text stays readable while the real value
    is gone. ``HASH`` substitutes a stable salted hash so equal PII values remain
    joinable across the corpus without exposing the value. ``TAG`` leaves the surface
    text intact but records the span (detect-only) — used when policy for a
    Classification sensitivity is to flag PII for review rather than remove it.
    """

    MASK = "mask"
    REPLACE = "replace"
    HASH = "hash"
    TAG = "tag"


class PIISpan(BaseModel):
    """One detected PII span in a document, with the action taken (CONTEXT ``Redaction``).

    Since W16 (ADR-0042) a PIISpan is carried on the :class:`ChunkRecord` the chunk-level
    Redactor produced: ``char_start``/``char_end`` are then offsets into that chunk's
    ``content`` (the coordinate system its ``masked_content`` is masked in — chunk-LOCAL, not
    document-global), while ``page_start``/``page_end`` still resolve to the ORIGINAL source
    page through the chunk's offset_map+page_map. On the DEPRECATED document-level
    :class:`RedactionRecord` path the offsets are document-global (the historical shape below).

    The structured, auditable record of a single redaction: *what* type of PII was
    found (``pii_type``, from the Provider's configured type set — the Capability
    names none), *where* it sat in the parent text
    (``char_start``/``char_end`` — chunk-local content offsets on a ChunkRecord, or the
    document-offset coordinate system on the deprecated RedactionRecord — the coordinate every
    other Stage uses, so the span resolves back through the parent's page map to its
    source page), *what was done* (``action``), and the ``placeholder`` that replaced
    it in the masked variant (empty for a ``TAG`` action, which leaves the text
    intact). ``detector`` names the recogniser that fired (``regex:email``,
    ``gliner-pii``) so a hybrid Provider's rule vs model provenance is auditable.

    The **raw PII value is deliberately NOT stored** — a PIISpan carries the offset,
    the type, and the placeholder, never the sensitive surface string, so the audit
    trail itself never becomes a PII leak (S6 AC: no raw PII in logs/records).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    pii_type: str = Field(..., min_length=1, description="PII type, e.g. 'email' (per-Provider).")
    detector: str = Field(..., min_length=1, description="Recogniser that fired, e.g. regex:email.")
    char_start: int = Field(..., ge=0, description="Span start in the parent document markdown.")
    char_end: int = Field(..., ge=0, description="Span end (half-open) in the parent document.")
    action: RedactionAction = Field(..., description="What was done to the span.")
    placeholder: str = Field(
        default="",
        description="Replacement text in the masked variant ('' for a TAG/detect-only span).",
    )
    page_start: int | None = Field(default=None, ge=0, description="Source page the span starts.")
    page_end: int | None = Field(default=None, ge=0, description="Source page the span ends.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detector confidence 0..1.")

    @model_validator(mode="after")
    def _span_ordered(self) -> PIISpan:
        if self.char_end < self.char_start:
            msg = (
                f"PIISpan is inverted: char_end {self.char_end} "
                f"< char_start {self.char_start}"
            )
            raise ValueError(msg)
        if self.action is not RedactionAction.TAG and not self.placeholder:
            msg = (
                f"A {self.action.value} PIISpan must carry the placeholder it wrote into "
                "the masked variant (only a TAG/detect-only span may have an empty placeholder)."
            )
            raise ValueError(msg)
        return self

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


# W16 (ADR-0042): ``ChunkRecord.pii_spans`` forward-references :class:`PIISpan`, which is defined
# here (after ChunkRecord). Rebuild the model now that PIISpan exists so pydantic resolves the
# annotation — the idiomatic v2 deferred-forward-ref resolution (``from __future__ import
# annotations`` keeps the field a string reference until this call).
ChunkRecord.model_rebuild()


class RedactionRecord(Record):
    """A per-document Redaction result — DEPRECATED, superseded by chunk-level redaction (W16).

    .. deprecated:: W16 (ADR-0042)
        Document-level Redaction fed the WHOLE parsed document into a fixed PII model window
        (768 tokens for the gliner-family), so any PII past that point was SILENTLY truncated
        and leaked unmasked into the corpus. Redaction is now a **chunk→chunk transform** — the
        :class:`~latence_core.capability.PIIDetector` seam consumes :class:`ChunkRecord`s and
        yields the same chunks with :attr:`ChunkRecord.masked_content` +
        :attr:`ChunkRecord.pii_spans` populated (each chunk ≤ the model window, so nothing is
        truncated). This RedactionRecord
        contract is RETAINED for the deprecated doc-level path (the provider ``redact_documents``
        secondary method) and for prior-version JSON compatibility, but the blessed stacks and the
        Runner's ``redaction`` Stage now flow chunks.

    A per-document Redaction result: the masked text variant + the PII spans (S6).

    Redaction detects PII and produces masked/replaced variants of the text (CONTEXT
    ``Redaction``). Like every inter-Stage carrier it is a :class:`Record`, so it
    carries Provenance and Classification and threads through the DAG / exports to the
    corpus — the ``masked_content`` is the **PII-handled RAG-corpus text** an Export
    materializes (ADR-0017). A RedactionRecord records:

    * ``masked_content`` — the document's assembled markdown with every non-``TAG``
      PII span replaced by its placeholder (a ``TAG``-only run leaves the text
      identical to the source but still records the spans).
    * ``pii_spans`` — the structured, auditable list of every detected span (type,
      document offset, action, placeholder, page) — the S6 "structured record of
      detected PII spans" requirement. The raw PII value is never stored on a span.
    * ``policy`` — the redaction policy name that was applied (keyed off the
      document's ``Classification.sensitivity``, S6 AC), recorded so the run is
      auditable end to end.
    * Provenance ``char_start``/``char_end`` — the covered document extent ``[0,
      len)``; the parent document's Classification, inherited unchanged.

    ``document_record_id`` links the record back to the document it redacted;
    ``media_type`` mirrors the source (``text/markdown``). A document with no detected
    PII yields a RedactionRecord whose ``masked_content`` equals the source and whose
    ``pii_spans`` is empty (never an error) — so the corpus always has one PII-handled
    variant per document.
    """

    document_record_id: str = Field(..., min_length=1, description="Parent DocumentRecord id.")
    masked_content: str = Field(default="", description="Text with PII spans masked/replaced.")
    media_type: str = Field(default="text/markdown")
    policy: str = Field(default="default", description="Redaction policy applied (sensitivity).")
    pii_spans: list[PIISpan] = Field(
        default_factory=list, description="Every detected PII span (type/offset/action/placeholder)"
    )
    redaction_disabled_for_sensitive: bool = Field(
        default=False,
        description=(
            "True when this document's Classification.sensitivity is in the sensitive "
            "set (confidential/restricted/…) yet the applied policy ran NO detectors "
            "(skip:true, empty types, or a detectors list that resolved to nothing) — "
            "so an empty pii_spans here means 'the control was OFF', NOT 'no PII found'. "
            "A first-class, auditable signal (H-C1 §4): a disabled control can never "
            "masquerade as a clean bill of health."
        ),
    )

    @model_validator(mode="after")
    def _spans_ordered_and_in_bounds(self) -> RedactionRecord:
        prov = self.provenance
        if prov.char_start is None or prov.char_end is None:
            msg = "A RedactionRecord's Provenance must carry a char_start and char_end span."
            raise ValueError(msg)
        # Spans must be ordered and non-overlapping so the masked variant is
        # reconstructable and the audit is unambiguous.
        last_end = -1
        for span in self.pii_spans:
            if span.char_start < last_end:
                msg = (
                    "RedactionRecord pii_spans must be ordered and non-overlapping; "
                    f"span at {span.char_start} overlaps a prior span ending at {last_end}."
                )
                raise ValueError(msg)
            last_end = span.char_end
        return self

    @property
    def pii_count(self) -> int:
        """Number of detected PII spans (0 for a clean document)."""
        return len(self.pii_spans)


# -- S7: Profiling -----------------------------------------------------------


class FeatureScope(StrEnum):
    """The scope a :class:`FeatureRecord` describes (CONTEXT ``Profiling``).

    Profiling is a *corpus-level* Stage, so it emits at two scopes. ``DOCUMENT`` is
    one record per parsed document, carrying that document's per-document features
    (density, readability, Zipf α, compression ratio, structure). ``CORPUS`` is
    exactly one record per run, carrying the cross-document aggregate features
    (entity frequency, co-occurrence, type consensus, source coverage). Both are
    inter-Stage :class:`Record`s — the split is a scope tag, not two contracts, so a
    consumer reads one exported stream and filters by ``scope``.
    """

    DOCUMENT = "document"
    CORPUS = "corpus"


class DocumentFeatures(BaseModel):
    """The fixed per-document statistical + quality feature set (S7).

    Computed over one parsed :class:`DocumentRecord`'s assembled markdown. This is the
    consolidated, **fixed** feature set the S7 issue calls for (the private stack's 9
    enrichers RECAST into a lean closed set — no dynamic registry), so the schema is
    stable and a seeded run is byte-identical:

    * ``char_count`` / ``word_count`` / ``sentence_count`` — raw size measures.
    * ``unique_word_count`` — distinct lowercased word types (the vocabulary size).
    * ``type_token_ratio`` — ``unique_word_count / word_count`` (lexical diversity;
      0.0 for an empty document).
    * ``density`` — non-whitespace characters as a fraction of all characters (0..1):
      how information-dense the text is versus padding/layout whitespace.
    * ``readability`` — a normalised 0..1 readability score (higher = easier), derived
      from average sentence length and average word length (a Flesch-style reading-ease
      mapped into ``[0, 1]``), dependency-free and deterministic.
    * ``zipf_alpha`` — the fitted Zipfian exponent α of the word-frequency
      distribution (a well-formed natural-language corpus sits near 1.0); 0.0 when
      there are too few distinct words to fit.
    * ``compression_ratio`` — ``compressed_size / raw_size`` under zlib (0..1-ish): a
      proxy for redundancy/repetitiveness (boilerplate compresses hard, so a low ratio
      flags a repetitive/templated document); 0.0 for empty text.
    * ``avg_word_length`` / ``avg_sentence_length`` — the shape measures the
      readability score is built from, surfaced for downstream tuning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    char_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    sentence_count: int = Field(..., ge=0)
    unique_word_count: int = Field(..., ge=0)
    type_token_ratio: float = Field(..., ge=0.0, le=1.0)
    density: float = Field(..., ge=0.0, le=1.0)
    readability: float = Field(..., ge=0.0, le=1.0)
    zipf_alpha: float = Field(..., ge=0.0)
    compression_ratio: float = Field(..., ge=0.0)
    avg_word_length: float = Field(..., ge=0.0)
    avg_sentence_length: float = Field(..., ge=0.0)


class EntityCoOccurrence(BaseModel):
    """One cross-document co-occurrence: an ordered entity-text pair + its count (S7).

    The pair is stored in a canonical order (``left <= right`` lexicographically) so
    the same unordered pair is counted once regardless of the order the two mentions
    appeared in a document. ``document_count`` is the number of *distinct documents*
    in which both entities co-occur — the cross-document co-occurrence signal, not a
    raw within-document tally.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    left: str = Field(..., min_length=1)
    right: str = Field(..., min_length=1)
    document_count: int = Field(..., ge=1)


class CorpusFeatures(BaseModel):
    """The fixed cross-document statistical + quality feature set (S7).

    Computed over the whole run — every parsed document and (when an Entity-Extraction
    Stage is upstream) every :class:`EntityMention`. The consolidated cross-document
    half of the RECAST enricher set:

    * ``document_count`` — parsed documents profiled.
    * ``total_words`` / ``total_unique_words`` — corpus size + vocabulary.
    * ``corpus_zipf_alpha`` — the Zipfian exponent fitted over the *whole-corpus*
      word-frequency distribution (the corpus-level Zipf α the issue names).
    * ``language_coverage`` — per-``Classification.language`` document counts (source
      coverage by language).
    * ``category_coverage`` / ``sensitivity_coverage`` — the same source-coverage
      breakdown by content category and sensitivity.
    * ``entity_frequency`` — cross-document entity mention counts, keyed by the
      mention surface text (how often each entity appears corpus-wide).
    * ``entity_type_frequency`` — mention counts keyed by entity *label*.
    * ``entity_type_consensus`` — per entity text, the fraction of its mentions that
      carry its single most-common label (1.0 = every mention agrees on the type; a
      lower value flags an entity the extractor typed inconsistently across the
      corpus). Only entities seen at least twice are scored.
    * ``co_occurrences`` — the top cross-document entity co-occurrence pairs
      (:class:`EntityCoOccurrence`), most-frequent first, capped for a bounded report.

    Every map is emitted in a deterministic order (sorted), so a seeded run is
    byte-identical.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    document_count: int = Field(..., ge=0)
    total_words: int = Field(..., ge=0)
    total_unique_words: int = Field(..., ge=0)
    corpus_zipf_alpha: float = Field(..., ge=0.0)
    language_coverage: dict[str, int] = Field(default_factory=dict)
    category_coverage: dict[str, int] = Field(default_factory=dict)
    sensitivity_coverage: dict[str, int] = Field(default_factory=dict)
    entity_frequency: dict[str, int] = Field(default_factory=dict)
    entity_type_frequency: dict[str, int] = Field(default_factory=dict)
    entity_type_consensus: dict[str, float] = Field(default_factory=dict)
    co_occurrences: list[EntityCoOccurrence] = Field(default_factory=list)


class FeatureRecord(Record):
    """A corpus-level Profiling result: a typed FeatureSet at one scope (CONTEXT ``Profiling``).

    Profiling computes statistical and quality features over the corpus. A FeatureRecord
    is an inter-Stage :class:`Record` — it carries Provenance and Classification like
    every other carrier — so it threads through the DAG, is checkpointed/resumed
    uniformly, and exports directly. Exactly one of ``document``/``corpus`` is populated,
    selected by ``scope``:

    * ``scope == DOCUMENT``: ``document`` holds the per-document :class:`DocumentFeatures`
      and ``document_record_id`` links back to the profiled DocumentRecord; its Provenance
      is that document's lineage, its Classification inherited unchanged. ``corpus`` is
      ``None``.
    * ``scope == CORPUS``: ``corpus`` holds the cross-document :class:`CorpusFeatures`;
      ``document_record_id`` is ``None`` and its Provenance is the synthetic corpus-scope
      lineage (the whole run is its source). ``document`` is ``None``.

    The scope/payload consistency is contract-enforced, so a Provider cannot emit a
    document-scope record with corpus features (or vice-versa) or leave both empty.
    """

    scope: FeatureScope
    document_record_id: str | None = Field(
        default=None, description="Parent DocumentRecord id (DOCUMENT scope only)."
    )
    document: DocumentFeatures | None = Field(default=None)
    corpus: CorpusFeatures | None = Field(default=None)

    @model_validator(mode="after")
    def _scope_payload_consistent(self) -> FeatureRecord:
        if self.scope is FeatureScope.DOCUMENT:
            if self.document is None:
                msg = "A DOCUMENT-scope FeatureRecord must carry a 'document' feature set."
                raise ValueError(msg)
            if self.corpus is not None:
                msg = "A DOCUMENT-scope FeatureRecord must not carry 'corpus' features."
                raise ValueError(msg)
            if self.document_record_id is None:
                msg = "A DOCUMENT-scope FeatureRecord must carry the profiled document_record_id."
                raise ValueError(msg)
        else:  # CORPUS
            if self.corpus is None:
                msg = "A CORPUS-scope FeatureRecord must carry a 'corpus' feature set."
                raise ValueError(msg)
            if self.document is not None:
                msg = "A CORPUS-scope FeatureRecord must not carry 'document' features."
                raise ValueError(msg)
            if self.document_record_id is not None:
                msg = "A CORPUS-scope FeatureRecord must not name a single document_record_id."
                raise ValueError(msg)
        return self


# -- S8: Disambiguation ------------------------------------------------------


class Evidence(BaseModel):
    """The justification linking an inference back to its source mentions (CONTEXT ``Evidence``).

    Evidence is the record linking a knowledge-graph edge — or, in S8, a Disambiguation
    merge/link decision — back to the specific source mentions and offsets that justify
    it, with a confidence value. It is deliberately distinct from :class:`Provenance`:
    Provenance is *source lineage* (which file, which page, which char span a record came
    from); Evidence is *justification for an inference* (which mentions made us believe two
    surface strings name the same entity, and how strongly). A merge that folds two
    mentions into one canonical entity carries the Evidence for *why* — so no merge is a
    silent, unexplained act.

    ``mention_ids`` are the ``record_id``s of the source :class:`EntityMention`s (or, for a
    relation, the endpoints) the inference rests on; ``document_ids`` the distinct source
    documents they came from (the cross-document reach of the evidence); ``snippet`` an
    optional short human-readable justification (never raw sensitive content — a surface
    string or the strategy name), and ``confidence`` the 0..1 strength of the justification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    mention_ids: list[str] = Field(
        default_factory=list, description="record_ids of the source mentions that justify this."
    )
    document_ids: list[str] = Field(
        default_factory=list, description="Distinct source document ids the evidence spans."
    )
    snippet: str = Field(default="", description="Short human-readable justification.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Strength of the justification.")


class MergeStrategy(StrEnum):
    """Which resolver strategy justified a merge/link decision (ported EntityResolver cascade).

    The ported ``resolution/entity_resolver.py`` asset (ASSET-INVENTORY.md, 5/5,
    PORT-AS-IS) resolves entity mentions with a **5-strategy cascade**, tried in order of
    descending precision so the strongest available signal wins and the reason is
    auditable:

    * ``EXACT`` — identical normalised surface strings (case/whitespace-folded).
    * ``ALIAS`` — a configured alias/synonym maps one surface to another's canonical.
    * ``ACRONYM`` — an acronym matches the initials of a multi-word name
      (``IBM`` ↔ ``International Business Machines``).
    * ``SUBSTRING`` — one name is a whole-word substring of the other
      (``Acme`` ↔ ``Acme Corporation``), gated by a length ratio so ``a`` ⊄ ``about``.
    * ``EMBEDDING`` — a vector-similarity match above a threshold (the CPU reference
      Provider has no embedder, so this strategy is inert there and is the pluggable
      FAISS/GPU upgrade path, S8 AC).

    ``KB_LINK`` is the linking (not merge) disposition: the GLinker pipeline matched a
    mention to a canonical entity in an external KB (L2→L3→L4→L0). Recorded as the
    justification on a :class:`CanonicalEntity` that carries a ``kb_id``.
    """

    EXACT = "exact"
    ALIAS = "alias"
    ACRONYM = "acronym"
    SUBSTRING = "substring"
    EMBEDDING = "embedding"
    KB_LINK = "kb_link"


class MergeDecision(BaseModel):
    """One audited, confidence-weighted merge/link decision (no silent over-merge, S8 AC).

    The ported EntityResolver keeps an **audited merge log**: every candidate merge —
    applied or not — is recorded with the strategy that proposed it, its confidence, and
    the Evidence, so a run never silently over-merges. A decision below the confidence
    policy is recorded with ``applied == False`` and is NOT folded into a canonical
    entity — the S8 acceptance criterion "merges below the confidence policy are not
    applied and are logged".

    ``source_text``/``target_text`` are the two surface strings the decision related
    (``source`` folded into ``target``); ``strategy`` the cascade rung that proposed it;
    ``confidence`` its 0..1 score; ``applied`` whether the confidence policy accepted it;
    ``evidence`` the justification (the mentions/offsets). ``reason`` is a short
    human-readable note (e.g. ``"acronym: IBM ~ International Business Machines"``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    source_text: str = Field(..., min_length=1, description="Surface folded into the target.")
    target_text: str = Field(..., min_length=1, description="Surface kept as the canonical.")
    strategy: MergeStrategy
    confidence: float = Field(..., ge=0.0, le=1.0)
    applied: bool = Field(..., description="False = below policy, logged but not merged.")
    reason: str = Field(..., min_length=1, description="Human-readable justification.")
    evidence: Evidence


class CanonicalEntity(Record):
    """A canonical entity: a merged cross-document cluster of mentions (CONTEXT ``Disambiguation``).

    Disambiguation links entity mentions to canonical entities and merges duplicates across
    documents. A CanonicalEntity is the *entity-scope* output: one per resolved cluster,
    carrying the chosen canonical surface (``canonical_text``) and ``entity_type``, the
    ``member_mention_ids`` (the ``record_id``s of every :class:`EntityMention` folded into
    it), the ``source_document_ids`` it spans (its cross-document reach — the affected-set
    key S11 keys incremental recompute off), the ``merge_decisions`` that built it (the
    audit trail), and its :class:`Evidence`. When the cluster linked to an external KB, the
    matched id is recorded in ``kb_id`` (``None`` for an unlinked entity — the graceful
    fallback, S8 AC).

    Like every inter-Stage carrier it is a :class:`Record`, so it carries Provenance and
    Classification and threads through the DAG / exports. Its Provenance is the *canonical*
    mention's lineage — page-accurate via :class:`~latence_core.pagemap.PageOffsetIndex`, so
    a downstream KG edge can cite "page 7 of the contract", the S8 page-map-integration
    sub-task — and its Classification is inherited from that mention. ``mention_count`` is
    the size of the cluster; a singleton entity (no duplicate found) is valid — it is a
    canonical entity of one mention, not an error.
    """

    canonical_text: str = Field(..., min_length=1, description="The chosen canonical surface.")
    entity_type: str = Field(..., min_length=1, description="The entity label (majority vote).")
    member_mention_ids: list[str] = Field(
        ..., min_length=1, description="record_ids of every mention folded into this entity."
    )
    source_document_ids: list[str] = Field(
        default_factory=list, description="Distinct source documents this entity appears in."
    )
    kb_id: str | None = Field(
        default=None, description="External KB id when linked; None when unlinked (fallback)."
    )
    merge_decisions: list[MergeDecision] = Field(
        default_factory=list, description="The audited merges that built this cluster."
    )
    evidence: Evidence

    @property
    def mention_count(self) -> int:
        """The number of source mentions folded into this canonical entity."""
        return len(self.member_mention_ids)

    @property
    def linked(self) -> bool:
        """True if this entity linked to an external KB (carries a kb_id)."""
        return self.kb_id is not None

    @model_validator(mode="after")
    def _canonical_is_a_member(self) -> CanonicalEntity:
        # The member list must be non-empty (enforced above) and de-duplicated so the
        # cluster size is unambiguous and a mention is not double-counted.
        if len(set(self.member_mention_ids)) != len(self.member_mention_ids):
            msg = "A CanonicalEntity's member_mention_ids must be unique (no double-counting)."
            raise ValueError(msg)
        return self


class NormalizedRelation(Record):
    """A relation remapped onto canonical entities, label normalised (ported RelationNormalizer).

    The ported ``relations/normalizer.py`` asset (ASSET-INVENTORY.md, 4.5/5, PORT-AS-IS):
    GLinker-style relation L2 resolution — fuzzy + alias label normalisation, type
    filtering, and **inverse detection + direction swap** (an ``employed_by`` relation
    whose configured inverse is ``employs`` is rewritten to the canonical direction). A
    NormalizedRelation is the *relation-scope* Disambiguation output: it takes a source
    :class:`RelationMention` and rewrites its endpoints from mention ids to the
    ``head_entity_id``/``tail_entity_id`` of the :class:`CanonicalEntity`s they resolved to,
    and its ``label`` to the normalised canonical relation label. ``inverted`` records
    whether the head/tail were swapped to reach the canonical direction (so the rewrite is
    auditable). ``source_relation_id`` links back to the original RelationMention.

    Like every carrier it is a :class:`Record` (Provenance + Classification, inherited from
    the source relation), so it threads through the DAG and exports. A relation whose head
    or tail did not resolve to a canonical entity is dropped by the Provider (not emitted as
    a broken relation) and counted in the Quality Report — never silently mis-linked.
    """

    source_relation_id: str = Field(..., min_length=1, description="Parent RelationMention id.")
    label: str = Field(..., min_length=1, description="The normalised canonical relation label.")
    head_entity_id: str = Field(..., min_length=1, description="Canonical head entity record_id.")
    tail_entity_id: str = Field(..., min_length=1, description="Canonical tail entity record_id.")
    head_text: str = Field(..., min_length=1, description="Canonical head surface string.")
    tail_text: str = Field(..., min_length=1, description="Canonical tail surface string.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    inverted: bool = Field(
        default=False, description="True if head/tail were swapped to the canonical direction."
    )
    evidence: Evidence

    @model_validator(mode="after")
    def _endpoints_distinct(self) -> NormalizedRelation:
        if self.head_entity_id == self.tail_entity_id:
            msg = (
                "A NormalizedRelation's head and tail must be distinct canonical entities "
                f"(both are {self.head_entity_id!r})."
            )
            raise ValueError(msg)
        return self


class DisambiguationScope(StrEnum):
    """The scope a :class:`DisambiguationRecord` describes (CONTEXT ``Disambiguation``).

    Disambiguation is a *corpus-level* Stage, so — like Profiling (ADR-0025) — it emits a
    single scoped carrier rather than two contracts. ``ENTITY`` records carry one
    :class:`CanonicalEntity` each (the entity-linking + merge half); ``RELATION`` records
    carry one :class:`NormalizedRelation` each (the relation-normalisation half). A consumer
    reads one exported stream and filters by ``scope``; the DAG, the checkpoint, and Export
    all treat one carrier.
    """

    ENTITY = "entity"
    RELATION = "relation"


class DisambiguationRecord(Record):
    """A corpus-level Disambiguation result: one canonical entity OR one normalized relation.

    The scoped inter-Stage carrier for the Disambiguation Stage, mirroring
    :class:`FeatureRecord`'s scope split (ADR-0025/0026). Exactly one of
    ``entity``/``relation`` is populated, selected by ``scope``:

    * ``scope == ENTITY``: ``entity`` holds the :class:`CanonicalEntity`; ``relation`` is
      ``None``. Its Provenance is the canonical mention's page-accurate lineage and its
      Classification is inherited from that mention.
    * ``scope == RELATION``: ``relation`` holds the :class:`NormalizedRelation`;
      ``entity`` is ``None``. Its Provenance/Classification are the source relation's.

    The scope/payload consistency is contract-enforced, so a Provider cannot emit an
    entity-scope record with a relation payload (or leave both empty). Records are emitted
    deterministically (all ENTITY records in canonical-text order, then all RELATION records)
    so a seeded run is byte-identical (Baseline bar).
    """

    scope: DisambiguationScope
    entity: CanonicalEntity | None = Field(default=None)
    relation: NormalizedRelation | None = Field(default=None)

    @model_validator(mode="after")
    def _scope_payload_consistent(self) -> DisambiguationRecord:
        if self.scope is DisambiguationScope.ENTITY:
            if self.entity is None:
                msg = "An ENTITY-scope DisambiguationRecord must carry an 'entity'."
                raise ValueError(msg)
            if self.relation is not None:
                msg = "An ENTITY-scope DisambiguationRecord must not carry a 'relation'."
                raise ValueError(msg)
        else:  # RELATION
            if self.relation is None:
                msg = "A RELATION-scope DisambiguationRecord must carry a 'relation'."
                raise ValueError(msg)
            if self.entity is not None:
                msg = "A RELATION-scope DisambiguationRecord must not carry an 'entity'."
                raise ValueError(msg)
        return self


# -- S9: Graph Assembly ------------------------------------------------------


class GraphNode(Record):
    """One node in the canonical knowledge graph (CONTEXT ``Graph Assembly``).

    Graph Assembly builds the canonical knowledge graph — nodes, edges, and per-edge
    Evidence — from disambiguated entities and relations. A GraphNode is the graph's
    entity half: exactly one node per :class:`CanonicalEntity`, carrying a
    **deterministic** ``node_id`` (content-addressed: a sha256 over the corpus id and the
    canonical entity id, so the same entity always yields the same node id and two runs over
    the same corpus produce the same graph — the S9 "deterministic IDs" acceptance
    criterion), the ``canonical_name`` and ``label`` (entity type), the ``entity_id`` of the
    source :class:`CanonicalEntity`, the ``member_mention_ids`` folded into it, its
    ``source_document_ids`` (cross-document reach), an optional external ``kb_id`` link, and
    the :class:`Evidence` back to the source mentions that justify it.

    Like every inter-Stage carrier it is a :class:`Record`, so it carries Provenance
    (inherited from the canonical entity — page-accurate via PageOffsetIndex) and
    Classification. ``properties`` is an open bag of node attributes materialised into the KG
    export (confidence, mention_count, document_count), kept JSON-serialisable so it round-
    trips through Parquet/TTL/GraphML.
    """

    node_id: str = Field(..., min_length=1, description="Deterministic content-addressed id.")
    entity_id: str = Field(..., min_length=1, description="Source CanonicalEntity record_id.")
    label: str = Field(..., min_length=1, description="The entity type (node label).")
    canonical_name: str = Field(..., min_length=1, description="The canonical surface string.")
    member_mention_ids: list[str] = Field(
        default_factory=list, description="record_ids of every mention behind this node."
    )
    source_document_ids: list[str] = Field(
        default_factory=list, description="Distinct source documents this node appears in."
    )
    kb_id: str | None = Field(default=None, description="External KB id when linked; else None.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Node confidence (cluster).")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Open node attribute bag (JSON-serialisable)."
    )
    evidence: Evidence

    @property
    def linked(self) -> bool:
        """True if this node links to an external KB (carries a kb_id)."""
        return self.kb_id is not None


class GraphEdge(Record):
    """One edge in the canonical knowledge graph, with per-edge Evidence (S9 Graph Assembly).

    A GraphEdge is the graph's relation half: exactly one edge per
    :class:`NormalizedRelation`, carrying a **deterministic** ``edge_id`` (content-addressed:
    a sha256 over the corpus id and the source relation id), the ``source_node_id`` /
    ``target_node_id`` it connects (the deterministic ids of the head/tail
    :class:`GraphNode`s), the normalized relation ``label``, its ``confidence``, and its
    :class:`Evidence` — the record linking the edge back to the specific source mentions and
    offsets that justify it, with a confidence value (CONTEXT ``Evidence``, reused verbatim
    from S8 for per-edge justification: **every edge carries Evidence**, the S9 acceptance
    criterion). ``relation_id`` back-references the source :class:`NormalizedRelation`.

    Like every inter-Stage carrier it is a :class:`Record` (Provenance + Classification,
    inherited from the source relation). The head and tail node must be distinct (no
    self-loop). ``properties`` is an open bag of edge attributes materialised into the KG
    export (inverted, source_document_ids), kept JSON-serialisable.
    """

    edge_id: str = Field(..., min_length=1, description="Deterministic content-addressed id.")
    relation_id: str = Field(..., min_length=1, description="Source NormalizedRelation record_id.")
    source_node_id: str = Field(..., min_length=1, description="Head GraphNode node_id.")
    target_node_id: str = Field(..., min_length=1, description="Tail GraphNode node_id.")
    label: str = Field(..., min_length=1, description="The normalized relation label (edge label).")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Edge confidence.")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Open edge attribute bag (JSON-serialisable)."
    )
    evidence: Evidence

    @model_validator(mode="after")
    def _endpoints_distinct(self) -> GraphEdge:
        if self.source_node_id == self.target_node_id:
            msg = (
                "A GraphEdge's source and target node must be distinct "
                f"(both are {self.source_node_id!r})."
            )
            raise ValueError(msg)
        return self


class GraphScope(StrEnum):
    """The scope a :class:`GraphRecord` describes (CONTEXT ``Graph Assembly``).

    Graph Assembly is a *corpus-level* Stage, so — like Disambiguation (ADR-0026) and
    Profiling (ADR-0025) — it emits a single scoped carrier rather than two contracts.
    ``NODE`` records carry one :class:`GraphNode` each (the graph's entity half); ``EDGE``
    records carry one :class:`GraphEdge` each (the graph's relation half). A consumer reads
    one exported stream and filters by ``scope``; the DAG, the checkpoint, and the KG Export
    all treat one carrier.
    """

    NODE = "node"
    EDGE = "edge"


class GraphRecord(Record):
    """A corpus-level Graph-Assembly result: one graph node OR one graph edge.

    The scoped inter-Stage carrier for the Graph Assembly Stage, mirroring
    :class:`DisambiguationRecord`'s scope split (ADR-0026). Exactly one of ``node``/``edge``
    is populated, selected by ``scope``:

    * ``scope == NODE``: ``node`` holds the :class:`GraphNode`; ``edge`` is ``None``. Its
      Provenance/Classification are the source canonical entity's (page-accurate).
    * ``scope == EDGE``: ``edge`` holds the :class:`GraphEdge`; ``node`` is ``None``. Its
      Provenance/Classification are the source normalized relation's.

    The scope/payload consistency is contract-enforced, so a Provider cannot emit a node-scope
    record with an edge payload (or leave both empty). Records are emitted deterministically
    (all NODE records in node-id order, then all EDGE records in edge-id order), so a seeded
    run is byte-identical (Baseline bar).
    """

    scope: GraphScope
    node: GraphNode | None = Field(default=None)
    edge: GraphEdge | None = Field(default=None)

    @model_validator(mode="after")
    def _scope_payload_consistent(self) -> GraphRecord:
        if self.scope is GraphScope.NODE:
            if self.node is None:
                msg = "A NODE-scope GraphRecord must carry a 'node'."
                raise ValueError(msg)
            if self.edge is not None:
                msg = "A NODE-scope GraphRecord must not carry an 'edge'."
                raise ValueError(msg)
        else:  # EDGE
            if self.edge is None:
                msg = "An EDGE-scope GraphRecord must carry an 'edge'."
                raise ValueError(msg)
            if self.node is not None:
                msg = "An EDGE-scope GraphRecord must not carry a 'node'."
                raise ValueError(msg)
        return self


# -- S11: Incremental deltas -------------------------------------------------


class DeltaKind(StrEnum):
    """How a document changed since the last committed Corpus Version (CONTEXT ``Delta``).

    A :class:`Delta` is a content-hash diff of the Source manifest against the last committed
    Corpus Version (ports ``delta_detector``). Each document falls into exactly one class:

    * ``ADDED`` — a document id not present in the parent Version (new source).
    * ``UPDATED`` — the same document id, a new content hash (the text or its derived
      extraction changed): doc-level Stages re-run on it, its old derived records are retracted.
    * ``DELETED`` — a document id in the parent Version absent from the new manifest: its
      records are retracted (or purged), affected clusters re-resolved.
    * ``UNCHANGED`` — same id, same hash: nothing re-runs; its committed records are carried
      forward verbatim.

    **When is ``UPDATED`` reachable?** (issue #42). ``UPDATED`` fires only when a document id
    *persists* across two manifests with a changed content hash — i.e. the Source assigns a
    **stable identity independent of content**. The reference ``source.local_folder``
    (:class:`~latence_core.stages.source.LocalFolderSource`) is content-addressed: a document's id
    IS ``sha256`` of its bytes, so any in-place edit mints a *new* id and retires the old one, and
    the detector classifies that as ``ADDED`` (new id) + ``DELETED`` (old id) — never ``UPDATED``.
    So ``UPDATED`` is unreachable under the content-addressed local-folder Source and is exercised
    only by a Source with content-independent stable ids (e.g. a path-keyed or database-keyed
    Source). It is retained (not removed) because it is a valid, contract-versioned classification
    the DeltaProcessor and ``DeltaRunner._commit`` already handle, and a stable-id Source is a
    supported Source variant; removing it would be a lossy contract change.
    """

    ADDED = "added"
    UPDATED = "updated"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class DeletionMode(StrEnum):
    """The disposition for a deleted document (CONTEXT ``Retraction`` / ``Purge``).

    Deletion is dual-mode (ADR-0018). ``RETRACTION`` is the default soft tombstone: the
    document and its derived records are excluded from every Export, the knowledge graph, and
    the RAG corpus, but **retained** for audit and rollback, with affected clusters
    re-resolved. ``PURGE`` is the hard GDPR / right-to-be-forgotten erasure: physical removal
    of the source, all derived records, and their Evidence, followed by reconciliation of the
    affected set — nothing derived survives.
    """

    RETRACTION = "retraction"
    PURGE = "purge"


class DeltaEntry(BaseModel):
    """One document's classification within a :class:`Delta` (CONTEXT ``Delta``).

    The content-hash diff assigns each document a :class:`DeltaKind`. ``content_hash`` is the
    deterministic SHA-256 the ported detector computed over the document's markdown plus its
    derived entity texts and relation labels (``None`` for a ``DELETED`` document, which no
    longer has a current hash). ``file_name`` is carried for the audit; identity is the
    ``document_id`` alone (never the file name).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    kind: DeltaKind
    document_id: str = Field(..., min_length=1)
    file_name: str = Field(default="")
    content_hash: str | None = Field(
        default=None, description="Deterministic content hash; None for a DELETED document."
    )

    @model_validator(mode="after")
    def _hash_consistent(self) -> DeltaEntry:
        if self.kind is DeltaKind.DELETED:
            if self.content_hash is not None:
                msg = "A DELETED DeltaEntry has no current content_hash."
                raise ValueError(msg)
        elif self.content_hash is None:
            msg = f"A {self.kind.value} DeltaEntry must carry a content_hash."
            raise ValueError(msg)
        return self


class Delta(BaseModel):
    """The classified change set since the last committed Corpus Version (CONTEXT ``Delta``).

    Computed as a content-hash diff of the Source manifest against the parent Version's
    manifest (ports ``delta_detector``): every document is an ADDED / UPDATED / DELETED /
    UNCHANGED :class:`DeltaEntry`. ``parent_version`` is the Corpus Version this delta was
    computed against (``None`` for the very first, bootstrap delta against an empty corpus).
    ``deletion_mode`` records how deleted documents are being handled (Retraction by default,
    Purge on request). ``intentional_deletion`` records whether this delta's DELETED entries are
    *user-requested* tombstones (an explicit :meth:`retract`/:meth:`purge`) rather than the
    incidental delete-half of an ordinary in-place content edit: because document ids are
    content-addressed, editing a file mints a new id and retires the old one, so the detector
    classifies the old content-id DELETED even though the user retracted nothing. Only an
    ``intentional_deletion`` DELETED id is a durable tombstone that stays excluded at Source on
    every future run; an edit's incidental DELETED id must NOT be permanently blacklisted, or an
    edit-then-revert to earlier content would be silently dropped. ``drift`` is the cheap drift
    metric (fraction of the corpus changed), and ``triggered_reconciliation`` records whether it
    crossed the threshold and forced a full Reconciliation — so the trigger and its outcome are
    auditable (the S11 AC).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    parent_version: int | None = Field(
        default=None, ge=0, description="The Corpus Version diffed against; None on bootstrap."
    )
    deletion_mode: DeletionMode = Field(default=DeletionMode.RETRACTION)
    intentional_deletion: bool = Field(
        default=False,
        description=(
            "True only when the DELETED entries are a user-requested retract/purge (a durable "
            "tombstone), not the incidental delete-half of a content-addressed in-place edit."
        ),
    )
    entries: list[DeltaEntry] = Field(default_factory=list)
    drift: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of the corpus changed by this delta."
    )
    triggered_reconciliation: bool = Field(
        default=False,
        description="True if drift crossed the threshold and forced a full recompute.",
    )

    def _ids(self, kind: DeltaKind) -> list[str]:
        return sorted(e.document_id for e in self.entries if e.kind is kind)

    @property
    def added(self) -> list[str]:
        return self._ids(DeltaKind.ADDED)

    @property
    def updated(self) -> list[str]:
        return self._ids(DeltaKind.UPDATED)

    @property
    def deleted(self) -> list[str]:
        return self._ids(DeltaKind.DELETED)

    @property
    def unchanged(self) -> list[str]:
        return self._ids(DeltaKind.UNCHANGED)

    @property
    def changed(self) -> list[str]:
        """The documents a delta must re-process (added + updated), sorted."""
        return sorted(self.added + self.updated)

    @property
    def is_empty(self) -> bool:
        """True if nothing changed (no add/update/delete) — an idempotent no-op re-run."""
        return not (self.added or self.updated or self.deleted)


class CorpusVersion(BaseModel):
    """An immutable, numbered committed state of the corpus (CONTEXT ``Corpus Version``).

    A Delta run reads Version N and transactionally produces Version N+1 (via the ported WAL +
    atomic manifest swap); any prior Version is inspectable for audit and rollback. A
    CorpusVersion is the manifest of that committed state: its ``version`` number, the
    ``parent_version`` it succeeded (``None`` for the initial Version 0), the :class:`Delta`
    that produced it, the live ``record_count`` after the delta, a deterministic
    ``fingerprint`` (a content hash over the committed record ids — identical input + config ⇒
    identical fingerprint, the G1 determinism criterion), and ``reconciled`` (True when this
    Version was produced by a full Reconciliation rather than an incremental affected-set
    recompute). ``created_ts`` is excluded from the fingerprint so re-running a delta is
    byte-identical apart from the timestamp.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    version: int = Field(..., ge=0, description="This Corpus Version's number.")
    parent_version: int | None = Field(default=None, ge=0)
    delta: Delta
    record_count: int = Field(..., ge=0, description="Live derived-record count after the delta.")
    document_count: int = Field(..., ge=0, description="Live source-document count post-delta.")
    fingerprint: str = Field(..., min_length=1, description="Deterministic content fingerprint.")
    reconciled: bool = Field(
        default=False,
        description="True if produced by a full Reconciliation, not incremental.",
    )
    created_ts: float = Field(
        default=0.0, description="Commit wall-clock (excluded from fingerprint)."
    )

    @model_validator(mode="after")
    def _parent_ordered(self) -> CorpusVersion:
        if self.parent_version is not None and self.parent_version >= self.version:
            msg = (
                f"CorpusVersion {self.version} must succeed a lower parent "
                f"(got parent {self.parent_version})."
            )
            raise ValueError(msg)
        return self


class DeltaChurn(BaseModel):
    """The delta run's churn roll-up for the Quality Report (S11 AC).

    "Delta churn (docs ±, entities created/merged/split, edges ±, drift) lands in the Quality
    Report" — the S11 acceptance criterion, kept to counts (no raw text), so it is safe to
    persist. ``documents_added`` / ``documents_updated`` / ``documents_deleted`` classify the
    delta; ``documents_unchanged`` are carried forward untouched. ``entities_created`` /
    ``entities_merged`` / ``entities_split`` are the affected-set recompute's effect on
    canonical entities (a bridging doc that MERGES two clusters, a deleted doc that SPLITS
    one); ``edges_added`` / ``edges_retracted`` the graph patch. ``records_retracted`` /
    ``records_purged`` count the soft-tombstoned vs physically-erased derived records.
    ``canonical_types_added`` / ``type_aliases_added`` / ``canonical_types_total`` are the T4
    type-vocabulary churn: how the delta EXTENDED the persisted canonical type vocabulary (new
    canonical entries vs raw labels that aliased into an existing one).
    ``drift`` is the metric and ``reconciliation_triggered`` whether it forced a full recompute,
    with ``deletion_mode`` recording how deletes were handled. ``from_version`` / ``to_version``
    bracket the transaction.

    ``affected_set_size`` / ``corpus_record_count`` are the perf-baseline witness that the
    recompute was affected-set-scoped, not O(corpus): on an incremental (non-reconciliation)
    delta ``affected_set_size`` (the corpus-level records the blocking-neighborhood recompute
    actually re-resolved) is ``<= corpus_record_count`` (the whole current corpus-level set),
    and a genuinely local change touches only a small fraction — the measurable ADR-0018
    "recompute only the affected set" bound. On a full Reconciliation the two are equal.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    from_version: int | None = Field(default=None, ge=0)
    to_version: int = Field(..., ge=0)
    documents_added: int = 0
    documents_updated: int = 0
    documents_deleted: int = 0
    documents_unchanged: int = 0
    # W17 (ADR-0043): the incremental-extraction perf witness. ``documents_extracted`` are the
    # NEW/CHANGED documents that actually ran the doc-level Stages (parse/chunk/extract/redact)
    # this delta; ``documents_reused`` are the UNCHANGED documents whose committed doc-level
    # records were reused from the prior Corpus Version (their doc-level Stages skipped). When the
    # induced schema is frozen, extraction is O(``documents_extracted``), not O(corpus) — so
    # ``documents_reused`` growing while ``documents_extracted`` stays small IS the measurable
    # boundary shift. A ``--reinduce`` delta (or the very first run) reuses nothing:
    # ``documents_reused == 0`` and every document is extracted.
    documents_reused: int = 0
    documents_extracted: int = 0
    # T4 (ADR-0043 amended): the canonical TYPE-vocabulary churn. Per-chunk induction produces no
    # single corpus schema to freeze, so what a delta carries across runs is the canonical type
    # vocabulary (ADR-0054), PERSISTED in the Corpus Version and EXTENDED — never recomputed —
    # by each delta. ``canonical_types_added`` counts the genuinely new canonical entries this
    # delta appended; ``type_aliases_added`` the raw labels that folded INTO an existing canonical
    # cluster (the no-churn half: they aliased instead of renaming anything);
    # ``canonical_types_total`` the size of the committed vocabulary afterwards. All three are
    # zero for a pipeline with no Type-Consolidation Stage.
    canonical_types_added: int = Field(default=0, ge=0)
    type_aliases_added: int = Field(default=0, ge=0)
    canonical_types_total: int = Field(default=0, ge=0)
    entities_created: int = 0
    entities_merged: int = 0
    entities_split: int = 0
    edges_added: int = 0
    edges_retracted: int = 0
    records_retracted: int = 0
    records_purged: int = 0
    affected_set_size: int = Field(default=0, ge=0)
    corpus_record_count: int = Field(default=0, ge=0)
    drift: float = Field(default=0.0, ge=0.0, le=1.0)
    reconciliation_triggered: bool = False
    deletion_mode: DeletionMode = Field(default=DeletionMode.RETRACTION)
