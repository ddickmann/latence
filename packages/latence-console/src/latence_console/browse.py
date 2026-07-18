"""The debugging loop, derived: quarantine, chunk inspection, and the KG with its Evidence.

:mod:`latence_console.verdicts` turns a Quality Report into headline verdicts — the
*summary* half of the console. This module is the *browse* half (#188): it turns the typed
records :class:`~latence_console.artifacts.RunArtifacts` reads back into the three answers a
person debugging a corpus actually needs.

* **What was held back, and why** — :func:`quarantine_entries` (removed from the pipeline
  entirely) and :func:`flag_entries` (kept, but carrying a risk marker into the corpus, so a
  RAG consumer can exclude it).
* **Why is this chunk wrong** — :func:`inspect_chunk` resolves a chunk back through its
  **Provenance** to the source document, the source *page*, and the character offsets it
  occupies there, and states its Classification and its Redaction disposition. The offsets
  are resolved through the record's own ``offset_map``/``page_map`` with the framework's own
  resolvers (:class:`~latence_core.offsetmap.OffsetIndex`,
  :class:`~latence_core.pagemap.PageOffsetIndex`) — never re-derived here, so the console
  cannot disagree with the pipeline about where a chunk came from.
* **Why does the graph believe this** — :func:`edge_views` renders every edge with its
  **Evidence**: the source mentions resolved to their surface text, file, page and character
  offsets, and the confidence. Evidence is a required field on
  :class:`~latence_core.contracts.GraphEdge`, so an edge without it cannot exist; a mention
  id the run did not checkpoint is shown as *unresolved*, never dropped.

Every text this module surfaces is the **PII-safe** text: a chunk's ``masked_content`` when
Redaction produced one, and never the parent document's raw markdown (ADR-0042/0053 —
the corpus is redacted by default, and an inspector that re-showed the raw text would be a
hole straight through that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from latence_core.contracts import ChunkRecord
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex

if TYPE_CHECKING:
    from latence_core.contracts import GraphEdge, GraphNode, PageMap, RiskMarker

    from latence_console.artifacts import RunArtifacts

# How much of a chunk's corpus text the inspector shows before it truncates. The inspector
# exists to explain a chunk, not to be a document reader.
EXCERPT_CHARS = 1200


# -- Quarantine / screening --------------------------------------------------


@dataclass(frozen=True)
class QuarantineEntry:
    """One record Screening removed from the pipeline, with the reason it recorded."""

    file_name: str
    file_type: str
    size_bytes: int
    checkpoint: str
    detector: str
    category: str
    reason: str
    document_id: str
    source_uri: str


@dataclass(frozen=True)
class FlagEntry:
    """One risk marker that survived onto a chunk in the corpus (kept, not removed)."""

    chunk_id: str
    file_name: str
    checkpoint: str
    detector: str
    category: str
    score: float
    reason: str


def quarantine_entries(artifacts: RunArtifacts) -> list[QuarantineEntry]:
    """Every quarantined record, newest checkpoint order preserved, by file name."""
    entries = [
        QuarantineEntry(
            file_name=record.provenance.file_name,
            file_type=record.provenance.file_type,
            size_bytes=record.quarantined_bytes,
            checkpoint=record.checkpoint.value,
            detector=record.detector,
            category=record.category,
            reason=record.reason,
            document_id=record.provenance.document_id,
            source_uri=record.provenance.source_uri,
        )
        for record in artifacts.quarantined()
    ]
    return sorted(entries, key=lambda e: (e.file_name, e.category))


def flag_entries(artifacts: RunArtifacts) -> list[FlagEntry]:
    """Every risk-flagged chunk, one row per marker.

    Derived from the chunks themselves — the markers that actually *survived into the
    corpus*, which is the property a RAG consumer filters on — not from the screening
    sidecar's intent. The checkpoint each marker came from is recovered by joining the
    sidecar's FLAG findings on ``(document_id, category)``.
    """
    checkpoints = artifacts.flag_checkpoints()
    entries: list[FlagEntry] = []
    for chunk in artifacts.chunks():
        for marker in chunk.risk_markers:
            entries.append(
                FlagEntry(
                    chunk_id=chunk.record_id,
                    file_name=chunk.provenance.file_name,
                    checkpoint=checkpoints.get(
                        (chunk.provenance.document_id, marker.category), "unknown"
                    ),
                    detector=marker.detector,
                    category=marker.category,
                    score=marker.score,
                    reason=marker.reason,
                )
            )
    return entries


# -- The chunk inspector -----------------------------------------------------


@dataclass(frozen=True)
class ChunkRow:
    """One row of the chunk list: enough to find the chunk you are looking for."""

    chunk_id: str
    file_name: str
    index: int
    pages: str
    char_span: str
    token_count: int
    flagged: bool
    redacted: bool


@dataclass(frozen=True)
class PIISpanView:
    """One detected PII span, located in the chunk AND in the source document."""

    pii_type: str
    detector: str
    action: str
    placeholder: str
    chunk_span: str
    source_span: str
    pages: str
    confidence: float


@dataclass(frozen=True)
class RedactionDisposition:
    """What Redaction did to this chunk — including "nothing, and that is the point"."""

    applied: bool
    control_disabled: bool
    spans: list[PIISpanView]
    summary: str


@dataclass(frozen=True)
class ChunkInspection:
    """A chunk resolved back to where it came from and what was done to it."""

    chunk_id: str
    index: int
    document_id: str
    document_record_id: str
    file_name: str
    file_type: str
    source_uri: str
    source_system: str | None
    provenance_pages: str
    provenance_span: str
    resolved_span: str
    resolved_pages: str
    document_pages: int | None
    document_chars: int | None
    document_parsed: bool
    document_error: str | None
    language: str
    category: str
    sensitivity: str
    token_count: int
    character_count: int
    redaction: RedactionDisposition
    risk_markers: list[RiskMarker] = field(default_factory=list)
    excerpt: str = ""
    excerpt_truncated: bool = False
    excerpt_is_masked: bool = False


def chunk_rows(artifacts: RunArtifacts) -> list[ChunkRow]:
    """Every chunk in the run, in source-document then chunk order."""
    return [
        ChunkRow(
            chunk_id=chunk.record_id,
            file_name=chunk.provenance.file_name,
            index=chunk.index,
            pages=_page_range(chunk.provenance.page_start, chunk.provenance.page_end),
            char_span=_span(chunk.provenance.char_start, chunk.provenance.char_end),
            token_count=chunk.token_count,
            flagged=bool(chunk.risk_markers),
            redacted=chunk.masked_content is not None,
        )
        for chunk in artifacts.chunks()
    ]


def inspect_chunk(artifacts: RunArtifacts, chunk: ChunkRecord) -> ChunkInspection:
    """Resolve one chunk: source document, page, character offsets, and disposition."""
    provenance = chunk.provenance
    document = artifacts.document(provenance.document_id)
    page_map = chunk.page_map or (document.page_map if document is not None else None)

    resolved_start, resolved_end = _resolved_span(chunk)
    masked = chunk.masked_content
    text = masked if masked is not None else chunk.content

    return ChunkInspection(
        chunk_id=chunk.record_id,
        index=chunk.index,
        document_id=provenance.document_id,
        document_record_id=chunk.document_record_id,
        file_name=provenance.file_name,
        file_type=provenance.file_type,
        source_uri=provenance.source_uri,
        source_system=provenance.source_system,
        provenance_pages=_page_range(provenance.page_start, provenance.page_end),
        provenance_span=_span(provenance.char_start, provenance.char_end),
        resolved_span=_span(resolved_start, resolved_end),
        resolved_pages=_pages_for_span(page_map, resolved_start, resolved_end),
        document_pages=len(page_map.pages) if page_map is not None else None,
        document_chars=page_map.total_chars if page_map is not None else None,
        document_parsed=document is not None,
        document_error=document.error if document is not None else None,
        language=chunk.classification.language,
        category=chunk.classification.category or "—",
        sensitivity=chunk.classification.sensitivity,
        token_count=chunk.token_count,
        character_count=len(chunk.content),
        redaction=_redaction_disposition(chunk, page_map),
        risk_markers=list(chunk.risk_markers),
        excerpt=text[:EXCERPT_CHARS],
        excerpt_truncated=len(text) > EXCERPT_CHARS,
        excerpt_is_masked=masked is not None,
    )


def _resolved_span(chunk: ChunkRecord) -> tuple[int | None, int | None]:
    """The chunk's span in the ORIGINAL document markdown, via its own ``offset_map``.

    ``content`` is markup-stripped, so ``char_start + len(content)`` overstates nothing but
    understates the true extent once markup was stripped inside the chunk (ADR-0031). The
    offset map is what corrects that; without one the console reports the Provenance span
    unchanged rather than guessing.
    """
    if chunk.offset_map is None:
        return chunk.provenance.char_start, chunk.provenance.char_end
    index = OffsetIndex(chunk.offset_map)
    return index.resolve_span(0, chunk.offset_map.length)


def _redaction_disposition(chunk: ChunkRecord, page_map: PageMap | None) -> RedactionDisposition:
    """What Redaction did — stated so an empty span list is never read as "clean"."""
    offsets = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
    spans: list[PIISpanView] = []
    for span in chunk.pii_spans:
        if offsets is not None:
            source_start, source_end = offsets.resolve_span(span.char_start, span.char_end)
        else:
            source_start, source_end = span.char_start, span.char_end
        pages = _page_range(span.page_start, span.page_end)
        if pages == "—":
            pages = _pages_for_span(page_map, source_start, source_end)
        spans.append(
            PIISpanView(
                pii_type=span.pii_type,
                detector=span.detector,
                action=span.action.value,
                placeholder=span.placeholder or "—",
                chunk_span=_span(span.char_start, span.char_end),
                source_span=_span(source_start, source_end),
                pages=pages,
                confidence=span.confidence,
            )
        )

    applied = chunk.masked_content is not None
    if chunk.redaction_disabled_for_sensitive:
        summary = (
            f"This chunk is classified {chunk.classification.sensitivity} but the redaction "
            "policy ran no detectors. An empty PII list here means the control was OFF — "
            "not that the chunk is clean."
        )
    elif not applied:
        summary = (
            "No Redaction Stage produced a masked variant of this chunk, so the corpus "
            "exports its text unchanged."
        )
    elif spans:
        summary = (
            f"Redaction ran and handled {len(spans)} PII span(s); the corpus exports the "
            "masked text below, never the original."
        )
    else:
        summary = (
            "Redaction ran and found no PII in this chunk; the corpus exports the masked "
            "text, which is identical to the original."
        )
    return RedactionDisposition(
        applied=applied,
        control_disabled=chunk.redaction_disabled_for_sensitive,
        spans=spans,
        summary=summary,
    )


# -- The KG browser ----------------------------------------------------------


@dataclass(frozen=True)
class MentionEvidence:
    """One source mention an inference rests on, located in its source document."""

    mention_id: str
    text: str
    label: str
    confidence: float
    file_name: str
    pages: str
    char_span: str
    resolved: bool


@dataclass(frozen=True)
class EdgeView:
    """One KG edge together with the Evidence that justifies it (never one without the other)."""

    edge_id: str
    label: str
    confidence: float
    source_node_id: str
    target_node_id: str
    source_name: str
    target_name: str
    inferred: bool
    evidence_confidence: float
    snippet: str
    document_count: int
    mentions: list[MentionEvidence]


@dataclass(frozen=True)
class EntityView:
    """One KG node: the canonical entity a set of mentions was folded into."""

    node_id: str
    canonical_name: str
    label: str
    kb_id: str | None
    confidence: float
    mention_count: int
    document_count: int
    edge_count: int


def entity_views(artifacts: RunArtifacts) -> list[EntityView]:
    """Every entity in the graph, by name, with how many edges touch it."""
    degree: dict[str, int] = {}
    for edge in artifacts.graph_edges():
        degree[edge.source_node_id] = degree.get(edge.source_node_id, 0) + 1
        degree[edge.target_node_id] = degree.get(edge.target_node_id, 0) + 1
    return [_entity_view(node, degree.get(node.node_id, 0)) for node in artifacts.graph_nodes()]


def entity_view(artifacts: RunArtifacts, node: GraphNode) -> EntityView:
    """One entity, with its edge degree."""
    degree = sum(
        1
        for edge in artifacts.graph_edges()
        if node.node_id in (edge.source_node_id, edge.target_node_id)
    )
    return _entity_view(node, degree)


def _entity_view(node: GraphNode, degree: int) -> EntityView:
    return EntityView(
        node_id=node.node_id,
        canonical_name=node.canonical_name,
        label=node.label,
        kb_id=node.kb_id,
        confidence=node.confidence,
        mention_count=len(node.member_mention_ids),
        document_count=len(node.source_document_ids),
        edge_count=degree,
    )


def edge_views(artifacts: RunArtifacts, node_id: str | None = None) -> list[EdgeView]:
    """Every edge (optionally only those touching ``node_id``), each with its Evidence."""
    names = {node.node_id: node.canonical_name for node in artifacts.graph_nodes()}
    views: list[EdgeView] = []
    for edge in artifacts.graph_edges():
        if node_id is not None and node_id not in (edge.source_node_id, edge.target_node_id):
            continue
        views.append(_edge_view(artifacts, edge, names))
    return views


def _edge_view(
    artifacts: RunArtifacts, edge: GraphEdge, names: dict[str, str]
) -> EdgeView:
    evidence = edge.evidence
    mentions = [
        _mention_evidence(artifacts, mention_id) for mention_id in evidence.mention_ids
    ]
    return EdgeView(
        edge_id=edge.edge_id,
        label=edge.label,
        confidence=edge.confidence,
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        source_name=names.get(edge.source_node_id, edge.source_node_id),
        target_name=names.get(edge.target_node_id, edge.target_node_id),
        inferred=bool(edge.properties.get("is_predicted") or edge.properties.get("inferred")),
        evidence_confidence=evidence.confidence,
        snippet=evidence.snippet,
        document_count=len(evidence.document_ids),
        mentions=mentions,
    )


def _mention_evidence(artifacts: RunArtifacts, mention_id: str) -> MentionEvidence:
    """A cited mention resolved to text + offsets, or an honest 'not checkpointed' row.

    An Evidence entry is never dropped for being unresolvable: the point of showing Evidence
    is that the reader learns what the inference rests on, including when the run no longer
    holds the record it cites.
    """
    mention = artifacts.mention(mention_id)
    if mention is None:
        return MentionEvidence(
            mention_id=mention_id,
            text="(mention not in this run's checkpoints)",
            label="—",
            confidence=0.0,
            file_name="—",
            pages="—",
            char_span="—",
            resolved=False,
        )
    return MentionEvidence(
        mention_id=mention_id,
        text=mention.text,
        label=mention.label,
        confidence=mention.confidence,
        file_name=mention.provenance.file_name,
        pages=_page_range(mention.provenance.page_start, mention.provenance.page_end),
        char_span=_span(mention.provenance.char_start, mention.provenance.char_end),
        resolved=True,
    )


# -- formatting helpers ------------------------------------------------------


def _page_range(start: int | None, end: int | None) -> str:
    if start is None and end is None:
        return "—"
    if start is None or end is None:
        return str(start if start is not None else end)
    return str(start) if start == end else f"{start}–{end}"


def _span(start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return "—"
    return f"{start}–{end}"


def _pages_for_span(page_map: PageMap | None, start: int | None, end: int | None) -> str:
    """The source page(s) a character span falls on, via the framework's own resolver."""
    if page_map is None or start is None or end is None:
        return "—"
    index = PageOffsetIndex(page_map)
    first = index.page_for_offset(start)
    last = index.page_for_offset(max(end - 1, start))
    return _page_range(first, last)
