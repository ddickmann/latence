"""Built-in INLINE-CITATION cross-reference Provider (deterministic, zero-shot, zero-dep).

The gap this closes (dogfood measured). Its sibling
:class:`~latence_core.stages.header_refs.HeaderReferenceExtractor` recovers a document's
*structured-header* cross-references (an RFC's ``Obsoletes:``/``Updates:`` lines) by anchoring the
document's self id and a header line **that must both sit in the SAME front-matter chunk**. That is
exact for metadata headers, but it structurally cannot see a document's *inline body* citations —
the "RFC 7230" / "[RFC9110]" references scattered through the document BODY — because a body chunk
carries no self declaration, so header_refs has no subject to hang the reference on and yields
nothing. On a real RFC corpus that inline body-citation class is the overwhelming majority of the
gold cross-reference graph (measured: 1039 of 1105 gold ``references`` edges), so a header-only KG
recovers only a small fraction of the corpus's reference edges and is too weak to drive genuine
multi-hop retrieval.

This Provider recovers exactly those inline references, **deterministically** and with real
Evidence. Unlike header_refs it resolves each document's *self* id **once, at document granularity**
— from a front-matter self declaration in ANY of the document's chunks, OR, failing that, from the
chunk :class:`~latence_core.contracts.Provenance`'s ``file_name`` (e.g. ``rfc9110.txt``) — and
carries it forward to that document's OTHER chunks. It then mines a configurable inline citation
pattern (e.g. ``RFC[\\s-]?(\\d{3,5})``) out of EVERY chunk of the document and links
``source_doc -> cited_id`` as a directed :class:`~latence_core.contracts.RelationMention` (default
label ``references``), synthesizing the source-doc node AND one node per distinct cited id so the
Graph gets real endpoints to connect. Because it fulfils the SAME
:class:`~latence_core.capability.FusedEntityRelationExtractor` seam as gliner2 and header_refs
(``extract_fused`` returns mentions + relations), it slots into a Pipeline as an extraction Stage
whose mentions AND relations thread downstream to Disambiguation and Graph Assembly with no other
change — and, using a disjoint ``#iref-`` record-id namespace, it composes ALONGSIDE both a learned
fused Stage AND header_refs without id collision (all feed the same Disambiguation input; the Runner
merges by carrier type).

Duplicate-and-contradiction hygiene. A cited id is emitted at most ONCE per source document (a
citation repeated across body chunks folds to one node + one edge). A self-citation (a document
citing its own id) is dropped (no self-loop). And any id already carried as a header relation for
that document — the ``exclude_relations`` block mirrors header_refs's ``Obsoletes:``/``Updates:``
header grammar — is dropped from the inline ``references`` set, so an id that is an ``obsoletes``
edge does not ALSO become a weaker, contradictory ``references`` edge when it happens to be
mentioned inline. The source-doc node is synthesized only when at least one inline edge survives, so
a document with no inline citations adds no isolated node (the KG stays byte-identical to a run
without the Stage for such a document).

Zero-shot (ADR-0007 CPU-first, ADR-0016 no extra deps, ADR-0004 ships in core like the reference
Chunker / Screeners / gazetteer / ``relation.pattern`` / ``header_refs``): the self recogniser, the
file-name fallback, the inline citation pattern, the identifier rendering, the relation label, and
the excluded header relations come **entirely from Stage** ``config`` — core hardcodes no relation
types and no RFC literals.

Determinism: documents are processed in first-seen input order, chunks within a document in input
order, citations within a chunk in match order, and each cited id is anchored at its FIRST
occurrence — so identical input + config yields byte-identical output (the Baseline bar). Offsets
and pages are resolved EXACTLY like the learned providers and like header_refs: a chunk-local model
offset is lifted to a true original-markdown offset through the chunk's stripped-``content`` ->
original ``offset_map`` via :class:`~latence_core.offsetmap.OffsetIndex` (a pre-v11 chunk degrades
to the ``char_start + local`` shift), and each span's source page is resolved through the
document-scoped :class:`~latence_core.page_index.PageIndexResolver` (degrading to the chunk's
inherited page range when no map is available). A document whose self id resolves from neither the
front matter nor the file name yields nothing (never an error, never a dangling edge).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.capability import FusedExtraction
from latence_core.contracts import EntityMention, RelationMention
from latence_core.offsetmap import OffsetIndex
from latence_core.page_index import PageIndexResolver, pages_for_covering_span
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)
from latence_core.stages.header_refs import _span_provenance

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord

# Default per-document cited-id fan-out cap: at most this many DISTINCT inline references are mined
# out of one source document, so a pathological body (thousands of "RFC n" tokens) cannot drive the
# synthesized mention/relation count — and every downstream graph edge — unbounded. Sized generously
# above any realistic document's reference list. Configurable via ``max_targets_per_document``; a
# value ``<= 0`` disables the cap (unbounded — not recommended on untrusted corpora).
_MAX_TARGETS_PER_DOCUMENT = 1000


class _ExcludeRule:
    """One compiled header relation whose target ids are EXCLUDED from the inline reference set."""

    __slots__ = ("line",)

    def __init__(self, line: re.Pattern[str]) -> None:
        self.line = line


class InlineReferenceExtractor:
    """Zero-shot deterministic INLINE-CITATION cross-reference Provider (fused NER + RE).

    Fulfils BOTH the :class:`~latence_core.capability.EntityExtractor` and the
    :class:`~latence_core.capability.FusedEntityRelationExtractor` seams: ``extract_fused`` returns
    the synthesized source-doc + cited-id mentions AND the inline reference relations between them
    together (one pass), and ``extract`` exposes the NER half alone.

    Config (every recogniser is supplied per Pipeline — zero-shot; core hardcodes none):
        citation_pattern: str  — REQUIRED. A regex whose **first capturing group** is ONE cited
            identifier as it appears in a document's body (e.g. ``r"(?i)RFCs?[\\s-]?(\\d{3,5})"`` to
            catch ``RFC 7230`` / ``[RFC9110]`` / ``RFC-3986``). Applied to EVERY chunk of a
            document; each match's captured id becomes a ``references`` target.
        self_pattern: str  — a regex whose **first capturing group** is the document's own
            identifier as it appears in its front matter (e.g.
            ``r"(?im)^Request for Comments:\\s*(\\d+)"``). The FIRST chunk (in input order) that
            matches names the whole document's subject. May be omitted when ``source_id_pattern`` is
            supplied.
        source_id_pattern: str  — a regex whose **first capturing group** is the document's own
            identifier as it appears in its provenance ``file_name`` (e.g.
            ``r"(?i)rfc[\\s_-]?(\\d+)"`` for ``rfc9110.txt``). The FALLBACK subject resolver when no
            chunk carries a
            ``self_pattern`` match. May be omitted when ``self_pattern`` is supplied. At least ONE
            of ``self_pattern`` / ``source_id_pattern`` is REQUIRED.
        relation_label: str  — the label stamped on every synthesized inline edge (default
            ``"references"``).
        id_template: str  — how a mined raw id is rendered into the canonical mention surface / node
            name (default ``"{id}"``; e.g. ``"RFC {id}"`` so both endpoints read ``RFC 7230`` and
            disambiguate against an in-text ``RFC 7230`` mention). Must contain ``{id}``.
        entity_label: str  — the entity type stamped on every synthesized identifier mention
            (default ``"reference"``).
        exclude_relations: dict[str, dict]  — OPTIONAL. Header relations whose target ids must NOT
            also become inline ``references`` edges for the same document (so an ``obsoletes`` id is
            not double-emitted as a weaker reference). Each value declares a ``header`` regex whose
            **first capturing group** carries the referenced ids (e.g.
            ``{"obsoletes": {"header": r"(?im)^Obsoletes:\\s*([\\d,\\s]+)"}}``); the ids are mined
            with ``target_pattern``. The relation LABELS here are ignored (only their ids matter) —
            this is the exclusion set, not an emission set.
        target_pattern: str  — a regex whose **first capturing group** is ONE id, applied to each
            ``exclude_relations`` header capture to mine the ids to exclude (default ``r"(\\d+)"``).
        default_confidence: float  — the confidence stamped on every emitted edge (default ``1.0``:
            a matched inline citation is a certain, deterministic signal).
        case_sensitive: bool  — compile the patterns case-sensitively (default ``False``). A
            caller-supplied inline flag (``(?i)``/``(?m)``) in a pattern is always honoured on top.
        max_targets_per_document: int  — per-document DISTINCT-id fan-out cap (default 1000). A
            value ``<= 0`` disables it (unbounded — not recommended on untrusted corpora).

    A cited id is emitted at most ONCE per source document; a self-citation and any
    ``exclude_relations`` id are dropped. The source-doc node is synthesized only when at least one
    inline edge survives (a document with no inline citations adds no isolated node).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        PolyForm-Noncommercial-1.0.0.

        A regex inline-citation extractor ships no model weights (``model_id=None``) and is a
        pure-Python part of ``latence-core`` (PolyForm-Noncommercial-1.0.0, verified). Deterministic
        (mentions/ relations emitted in document + chunk + first-occurrence order), CPU-only, no
        batching, no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-21",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._case_sensitive = bool(config.get("case_sensitive", False))
        flags = 0 if self._case_sensitive else re.IGNORECASE
        self._default_confidence = float(config.get("default_confidence", 1.0))

        self._entity_label = str(config.get("entity_label", "reference"))
        if not self._entity_label:
            msg = "InlineReferenceExtractor requires a non-empty 'entity_label'."
            raise ValueError(msg)

        self._relation_label = str(config.get("relation_label", "references"))
        if not self._relation_label:
            msg = "InlineReferenceExtractor requires a non-empty 'relation_label'."
            raise ValueError(msg)

        self._id_template = str(config.get("id_template", "{id}"))
        if "{id}" not in self._id_template:
            msg = "InlineReferenceExtractor 'id_template' must contain the '{id}' placeholder."
            raise ValueError(msg)

        self._max_targets = int(config.get("max_targets_per_document", _MAX_TARGETS_PER_DOCUMENT))

        citation_pattern = config.get("citation_pattern")
        if not citation_pattern or not isinstance(citation_pattern, str):
            msg = (
                "InlineReferenceExtractor requires a non-empty 'citation_pattern' regex in its "
                "config (the first capturing group is ONE cited identifier as it appears inline)."
            )
            raise ValueError(msg)
        self._citation_re = self._compile(citation_pattern, "citation_pattern", flags)

        self_pattern = config.get("self_pattern")
        source_id_pattern = config.get("source_id_pattern")
        if not self_pattern and not source_id_pattern:
            msg = (
                "InlineReferenceExtractor requires at least one of 'self_pattern' (the document's "
                "front-matter self id) or 'source_id_pattern' (its id in the provenance file name)."
            )
            raise ValueError(msg)
        self._self_re = (
            self._compile(str(self_pattern), "self_pattern", flags) if self_pattern else None
        )
        self._source_id_re = (
            self._compile(str(source_id_pattern), "source_id_pattern", flags)
            if source_id_pattern
            else None
        )

        self._target_re = self._compile(
            str(config.get("target_pattern", r"(\d+)")), "target_pattern", flags
        )
        self._exclude_rules = self._compile_exclude(config.get("exclude_relations"), flags)

    # -- config compilation -------------------------------------------------

    @staticmethod
    def _compile(pattern: str, name: str, flags: int) -> re.Pattern[str]:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            msg = f"InlineReferenceExtractor {name!r} is not a valid regex: {exc}"
            raise ValueError(msg) from exc
        if compiled.groups < 1:
            msg = (
                f"InlineReferenceExtractor {name!r} must have a capturing group for the identifier."
            )
            raise ValueError(msg)
        return compiled

    def _compile_exclude(self, relations: Any, flags: int) -> list[_ExcludeRule]:
        if relations is None:
            return []
        if not isinstance(relations, dict):
            msg = (
                "InlineReferenceExtractor 'exclude_relations' must be a mapping of label -> "
                f"{{'header': regex}} (got {type(relations).__name__})."
            )
            raise TypeError(msg)
        rules: list[_ExcludeRule] = []
        for label, spec in relations.items():
            label_str = str(label)
            if not isinstance(spec, dict):
                msg = (
                    f"InlineReferenceExtractor exclude relation {label_str!r} config must be a "
                    f"mapping (got {type(spec).__name__})."
                )
                raise TypeError(msg)
            header = spec.get("header")
            if not header or not isinstance(header, str):
                msg = (
                    f"InlineReferenceExtractor exclude relation {label_str!r} must declare a "
                    "'header' regex whose first capturing group carries the ids to exclude."
                )
                raise ValueError(msg)
            line = self._compile(header, f"exclude_relations[{label_str!r}].header", flags)
            rules.append(_ExcludeRule(line))
        return rules

    # -- fused extraction ---------------------------------------------------

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Group chunks by document; mine each document's self id + inline citations.

        For each document (first-seen order): resolve the self id ONCE (front-matter
        ``self_pattern`` in any chunk, else ``source_id_pattern`` against the provenance file name);
        build the excluded-id set from ``exclude_relations``; then walk the document's chunks in
        order emitting one ``references`` edge per DISTINCT cited id (skipping the self id and every
        excluded id).
        A document whose self id resolves from neither source, or that has no surviving citation,
        contributes nothing (never an error). Emission is document order, then chunk order, then
        first-occurrence order — byte-identical per run.
        """
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        # ONE resolver for the whole call, purely so the drift roll-up is per-call. It is
        # stateless (v19): each chunk carries its own page spans, so a chunk that cites
        # nothing teaches it nothing and skipping one costs nothing.
        resolver = PageIndexResolver()
        for doc_chunks in self._group_by_document(chunks):
            doc_mentions, doc_relations = self._extract_document(doc_chunks, resolver)
            mentions.extend(doc_mentions)
            relations.extend(doc_relations)
        return FusedExtraction(mentions=mentions, relations=relations)

    def extract(self, chunks: Iterable[ChunkRecord]) -> list[EntityMention]:
        """The NER half alone (the synthesized identifier mentions) for an EntityExtractor Stage."""
        return self.extract_fused(chunks).mentions

    @staticmethod
    def _group_by_document(chunks: Iterable[ChunkRecord]) -> list[list[ChunkRecord]]:
        """Group chunks by ``document_record_id``, preserving first-seen document + chunk order."""
        groups: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.document_record_id, []).append(chunk)
        return list(groups.values())

    def _resolve_self(
        self, doc_chunks: list[ChunkRecord]
    ) -> tuple[str, ChunkRecord, int, int] | None:
        """Resolve the document's self id + the chunk/span to anchor the source-doc node on.

        Priority: the first chunk (input order) whose ``self_pattern`` matches names the subject and
        anchors it at that match's span (real front-matter Evidence). Failing that, the
        ``source_id_pattern`` is applied to the FIRST chunk's provenance ``file_name`` and the
        source-doc node is anchored at the start of that chunk (an empty span — the id is a property
        of the file, not of a text span). ``None`` when neither resolves.
        """
        if self._self_re is not None:
            for chunk in doc_chunks:
                match = self._self_re.search(chunk.content)
                if match is None:
                    continue
                self_id = self._first_group(match)
                if self_id is not None:
                    return self_id, chunk, match.start(1), match.end(1)
        if self._source_id_re is not None and doc_chunks:
            first = doc_chunks[0]
            match = self._source_id_re.search(first.provenance.file_name)
            if match is not None:
                self_id = self._first_group(match)
                if self_id is not None:
                    return self_id, first, 0, 0
        return None

    def _excluded_ids(self, doc_chunks: list[ChunkRecord]) -> set[str]:
        """The ids carried as header relations for this document — dropped from the inline set."""
        excluded: set[str] = set()
        for rule in self._exclude_rules:
            for chunk in doc_chunks:
                for header_match in rule.line.finditer(chunk.content):
                    captured = header_match.group(1)
                    if not captured:
                        continue
                    for target_match in self._target_re.finditer(captured):
                        target_id = self._first_group(target_match)
                        if target_id is not None:
                            excluded.add(target_id)
        return excluded

    def _extract_document(
        self, doc_chunks: list[ChunkRecord], resolver: PageIndexResolver
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        resolved = self._resolve_self(doc_chunks)
        if resolved is None:
            return [], []
        self_id, self_chunk, self_start, self_end = resolved
        self_text = self._id_template.format(id=self_id)
        excluded = self._excluded_ids(doc_chunks)

        mentions: list[EntityMention] = []
        by_text: dict[str, EntityMention] = {}
        # The OFFSET map is genuinely chunk-scoped (it indexes this chunk's stripped
        # content), so its index stays cached per chunk here. The PAGE map is not — that
        # now lives behind ``resolver``, memoised per document.
        index_cache: dict[str, tuple[OffsetIndex | None, int]] = {}

        def indices(chunk: ChunkRecord) -> tuple[OffsetIndex | None, int]:
            cached = index_cache.get(chunk.record_id)
            if cached is None:
                offset_index = (
                    OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
                )
                cached = (offset_index, chunk.provenance.char_start or 0)
                index_cache[chunk.record_id] = cached
            return cached

        def get_or_create(
            chunk: ChunkRecord, text: str, local_start: int, local_end: int
        ) -> EntityMention:
            existing = by_text.get(text)
            if existing is not None:
                return existing
            offset_index, chunk_char_start = indices(chunk)
            if offset_index is not None:
                doc_start, doc_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_start = chunk_char_start + local_start
                doc_end = chunk_char_start + local_end
            index = len(mentions)
            mention = EntityMention(
                record_id=f"{chunk.document_record_id}#iref-entity-{index}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=index,
                label=self._entity_label,
                text=text,
                confidence=1.0,
                provenance=_span_provenance(
                    chunk.provenance,
                    doc_start,
                    doc_end,
                    resolver.pages_for_span(chunk, doc_start, doc_end),
                ),
                classification=chunk.classification,
            )
            mentions.append(mention)
            by_text[text] = mention
            return mention

        relations: list[RelationMention] = []
        head: EntityMention | None = None
        seen_targets: set[str] = set()
        for chunk in doc_chunks:
            if 0 < self._max_targets <= len(seen_targets):
                break
            for citation in self._citation_re.finditer(chunk.content):
                if 0 < self._max_targets <= len(seen_targets):
                    break
                target_id = self._first_group(citation)
                if target_id is None or target_id == self_id or target_id in excluded:
                    continue
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                # The source-doc node is synthesized lazily on the first surviving edge (index 0),
                # so a document with no inline citations adds no isolated node.
                if head is None:
                    head = get_or_create(self_chunk, self_text, self_start, self_end)
                target_text = self._id_template.format(id=target_id)
                tail = get_or_create(
                    chunk, target_text, citation.start(1), citation.end(1)
                )
                if tail.record_id == head.record_id:
                    continue
                relations.append(
                    self._relation(chunk, len(relations), head, tail)
                )
        return mentions, relations

    def _relation(
        self,
        chunk: ChunkRecord,
        index: int,
        head: EntityMention,
        tail: EntityMention,
    ) -> RelationMention:
        span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
        span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
        return RelationMention(
            record_id=f"{chunk.document_record_id}#iref-relation-{index}",
            document_record_id=chunk.document_record_id,
            index=index,
            label=self._relation_label,
            head_mention_id=head.record_id,
            tail_mention_id=tail.record_id,
            head_text=head.text,
            tail_text=tail.text,
            confidence=min(max(self._default_confidence, 0.0), 1.0),
            provenance=_span_provenance(
                head.provenance,
                span_start,
                span_end,
                # Genuinely cross-chunk: the head is the document's SELF declaration (often in
                # chunk 0) and the tail is wherever the citation was found, so no one chunk
                # spans it. Both endpoints resolved their own pages exactly from their own
                # chunk's spans, so the covering span's pages are their union.
                pages_for_covering_span(head.provenance, tail.provenance),
            ),
            classification=chunk.classification,
        )

    @staticmethod
    def _first_group(match: re.Match[str]) -> str | None:
        # Group 1 is guaranteed present by ``_compile`` (it rejects a group-less pattern), so
        # ``match.group(1)`` never raises; a group that matched empty / whitespace yields no id.
        value = match.group(1)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
