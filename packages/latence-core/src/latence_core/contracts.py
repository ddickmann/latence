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
SCHEMA_VERSION = 6


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
    """

    model_config = ConfigDict(extra="forbid")

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
    risk_markers: list[RiskMarker] = Field(
        default_factory=list, description="Content-Screening flags that propagate into the corpus."
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

    The structured, auditable record of a single redaction: *what* type of PII was
    found (``pii_type``, from the Provider's configured type set — the Capability
    names none), *where* it sat in the parent document's assembled markdown
    (``char_start``/``char_end``, the same document-offset coordinate system every
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


class RedactionRecord(Record):
    """A per-document Redaction result: the masked text variant + the PII spans (S6).

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
