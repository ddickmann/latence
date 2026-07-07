"""Built-in reference RelationExtractor Provider (deterministic, zero-shot, zero-dep).

The S5 Relation Extraction reference Provider (ADR-0004; ships in core like the S3
Chunker/Screeners and the S4 gazetteer so the demo runs end to end, ADR-0007
CPU-first, ADR-0016 no extra deps). It relates the :class:`EntityMention`s found in
S4 by **configured, trigger-phrase rules** — a genuinely zero-shot seam: the
relation label set and how to recognise each relation come entirely from Stage
``config``, the Capability names no relation types, and core hardcodes none.

A rule for a relation label says: a mention of type ``head`` followed (within a
window) by a mention of type ``tail``, with one of the rule's **trigger** phrases
appearing in the document text *between* them, is that relation (head → tail). This
is not a bare co-occurrence stub (ADR-0013 rejects that — it undercuts the KG
quality story): the trigger phrase both *types* the relation and *gates* it, so a
Pipeline gets typed, justified edges deterministically, with no model download.

Each emitted :class:`~latence_core.contracts.RelationMention` carries:

* its ``label`` (from the configured relation label set) and a ``confidence``;
* the head/tail :class:`~latence_core.contracts.EntityMention` refs and surface
  strings;
* Provenance ``char_start``/``char_end`` — the half-open span in the parent
  document's assembled markdown that covers **both** endpoints (the earlier
  mention's start to the later mention's end), so the relation resolves back to
  source offsets;
* Provenance ``page_start``/``page_end`` — the source page(s) that covering span
  falls on, resolved from the parent document's page map (carried on the mentions'
  chunks) via :class:`~latence_core.pagemap.PageOffsetIndex` — the S5 round-trip
  (relation → offset → original page); degrading to the endpoints' own recorded
  pages when no page map is available;
* the parent document's Classification, inherited unchanged.

A document with fewer than two mentions, or no rule-matching pair, yields no
relations (never an error) — the S5 acceptance criterion.

Determinism: relations are emitted grouped by ``document_record_id`` (in first-seen
order) then ordered within a document by ``(char_start, label, head, tail)``, so
identical input + config yields byte-identical relations (the Baseline bar).

**Quadratic-fan-out guard (ADR-0033 / #63).** The candidate scan is
``O(mentions²)`` per document within the window. A low-cardinality corpus (many
same-typed mentions crammed inside one window) is a pathological input that drives
the emitted relation count — and every downstream graph edge — quadratic in the
document's mention count, an algorithmic memory-DoS lever (G1 Lane 6 measured
``~0.084·n²`` records; 5k docs → ~12 GB, killed). Independent of streaming, the
Provider caps the relations emitted *per document* at ``max_relations_per_document``
(default 10_000). The cap is applied **after** the deterministic candidate sort, so
the retained relations are the lowest-ordered (by covering-span start, then label,
then head/tail ids) — a stable, deterministic prefix, never a random subset. The
number of documents whose fan-out was capped and the number of relations dropped are
tallied on the Provider (``fanout_capped_documents`` / ``fanout_dropped_relations``)
and surfaced into the Quality Report's RelationQuality — the audited "capped" marker,
so a capped run is never silently truncated. This mirrors the discipline #33 already
applied to profiling's per-document pair fan-out.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.contracts import RelationMention
from latence_core.pagemap import PageOffsetIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, EntityMention, PageMap, Provenance


class _Rule:
    """One compiled relation rule: head/tail label gate + trigger recogniser."""

    __slots__ = ("label", "head_label", "tail_label", "trigger", "confidence", "symmetric")

    def __init__(
        self,
        label: str,
        head_label: str,
        tail_label: str,
        trigger: re.Pattern[str] | None,
        confidence: float,
        symmetric: bool,
    ) -> None:
        self.label = label
        self.head_label = head_label
        self.tail_label = tail_label
        self.trigger = trigger
        self.confidence = confidence
        self.symmetric = symmetric


# A candidate relation before it is turned into a contract: the rule that matched,
# the ordered head/tail mentions, and the covering document span.
_Candidate = tuple[_Rule, "EntityMention", "EntityMention", int, int]

# Default per-document relation fan-out cap (ADR-0033 / #63): the max relations one
# document may emit, bounding the O(mentions²) a low-cardinality corpus produces.
# Sized generously so it never trims a realistic document (real documents yield far
# fewer than 10k typed, triggered relations), but a pathological same-typed mention
# storm is bounded rather than driving records — and downstream graph edges —
# quadratic. Configurable per Pipeline via ``max_relations_per_document``.
_MAX_RELATIONS_PER_DOCUMENT = 10_000


class PatternRelationExtractor:
    """Zero-shot trigger-phrase RelationExtractor Provider.

    The relation label set and its recognisers come entirely from ``config`` —
    nothing hardcoded in core, so the same Provider extracts ``works_for`` for one
    Pipeline and ``binds_to`` for another (zero-shot, S5 seam).

    Config:
        relations: dict[str, dict]  — the relation label set. Each key is a relation
            type; each value configures how to find it:
              * ``head``: str  — the entity label of the head (subject) mention.
              * ``tail``: str  — the entity label of the tail (object) mention.
              * ``triggers``: list[str]  — literal phrases (case-insensitive by
                default) that must appear in the document text *between* the head and
                tail mention for the pair to be this relation. A rule with no triggers
                matches any head→tail pair in-window (pure co-occurrence — allowed but
                the caller opts into it explicitly per label).
              * ``patterns``: list[str]  — regexes tried against the between-text as an
                alternative to ``triggers``.
              * ``confidence``: float  — the score to stamp (default ``default_confidence``).
              * ``symmetric``: bool  — if true, also match tail→head order (the
                relation is undirected); the emitted relation still records a head and
                tail but the pair need not appear in declared order.
        default_confidence: float  — confidence when a rule sets none (default 0.75).
        case_sensitive: bool  — match triggers case-sensitively (default False).
        window_chars: int  — max document-char gap between the head end and the tail
            start for a pair to be considered (default 200). Keeps relations local.
        max_relations_per_document: int  — quadratic-fan-out guard (ADR-0033 / #63):
            the max relations one document may emit (default 10_000). Applied after the
            deterministic candidate sort, so the retained relations are the
            lowest-ordered stable prefix. Bounds the O(mentions²) a low-cardinality
            corpus produces. A value ``<= 0`` disables the cap (unbounded, the old
            behaviour) — not recommended on untrusted corpora.

    Audit attributes (read after ``relate`` returns, e.g. by the Runner into the
    Quality Report — the ADR-0033 "capped" marker):
        fanout_capped_documents: int  — how many documents hit the cap.
        fanout_dropped_relations: int  — total relations dropped across those documents.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._default_confidence = float(config.get("default_confidence", 0.75))
        self._case_sensitive = bool(config.get("case_sensitive", False))
        self._window_chars = int(config.get("window_chars", 200))
        self._max_relations_per_document = int(
            config.get("max_relations_per_document", _MAX_RELATIONS_PER_DOCUMENT)
        )
        self._rules = self._compile_rules(config.get("relations", {}))
        # Audited fan-out-guard tallies (ADR-0033 / #63), reset per ``relate`` call so a
        # reused Provider instance reports only the current run's capping.
        self.fanout_capped_documents = 0
        self.fanout_dropped_relations = 0

    # -- config compilation -------------------------------------------------

    def _compile_rules(self, relations: dict[str, Any] | list[str]) -> list[_Rule]:
        """Compile the zero-shot relation label set into head/tail + trigger rules.

        A list-of-labels form declares the vocabulary with no recognisers (matches
        nothing). The dict form requires ``head`` and ``tail`` labels per relation and
        compiles the literal ``triggers`` (escaped) and raw ``patterns`` into one
        alternation; a rule with neither triggers nor patterns is a pure in-window
        co-occurrence rule (``trigger`` is ``None``).
        """
        rules: list[_Rule] = []
        flags = 0 if self._case_sensitive else re.IGNORECASE
        if isinstance(relations, list):
            return rules
        for label, spec in relations.items():
            if not isinstance(spec, dict):
                msg = f"Relation {label!r} config must be a mapping (got {type(spec).__name__})."
                raise TypeError(msg)
            head = spec.get("head")
            tail = spec.get("tail")
            if not head or not tail:
                msg = (
                    f"Relation {label!r} must declare 'head' and 'tail' entity labels "
                    "(the mention types it connects)."
                )
                raise ValueError(msg)
            confidence = float(spec.get("confidence", self._default_confidence))
            symmetric = bool(spec.get("symmetric", False))
            alternatives: list[str] = []
            for trig in spec.get("triggers", []):
                alternatives.append(re.escape(str(trig)))
            alternatives.extend(str(pat) for pat in spec.get("patterns", []))
            trigger = (
                re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
                if alternatives
                else None
            )
            rules.append(
                _Rule(str(label), str(head), str(tail), trigger, confidence, symmetric)
            )
        return rules

    # -- extraction ---------------------------------------------------------

    def relate(
        self, chunks: Iterable[ChunkRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[RelationMention]:
        """Relate mentions per document; ``chunks`` supply text + the page map.

        Mentions are grouped by ``document_record_id`` (a relation only ever links two
        mentions of the same document). The document's assembled text is reconstructed
        from its chunks (each chunk carries its own document-level span), which the
        rules read to test trigger phrases *between* a candidate head and tail.
        """
        # Reset the fan-out-guard tallies so a reused Provider reports only this call.
        self.fanout_capped_documents = 0
        self.fanout_dropped_relations = 0
        chunk_list = list(chunks)
        # Reconstruct per-document text + page map from the chunks (each chunk knows
        # its own [char_start, char_end) in the parent document, S3). This lets a
        # trigger phrase be tested against the real between-mention text, and lets a
        # relation's covering span resolve to a page.
        doc_text = self._document_texts(chunk_list)
        doc_page_map = self._document_page_maps(chunk_list)

        mentions_by_doc: dict[str, list[EntityMention]] = {}
        doc_order: list[str] = []
        for m in mentions:
            if m.document_record_id not in mentions_by_doc:
                mentions_by_doc[m.document_record_id] = []
                doc_order.append(m.document_record_id)
            mentions_by_doc[m.document_record_id].append(m)

        if not self._rules:
            return
        for doc_id in doc_order:
            yield from self._relate_one_document(
                doc_id,
                mentions_by_doc[doc_id],
                doc_text.get(doc_id, ""),
                doc_page_map.get(doc_id),
            )

    def _relate_one_document(
        self,
        doc_id: str,
        mentions: list[EntityMention],
        text: str,
        page_map: PageMap | None,
    ) -> Iterator[RelationMention]:
        if len(mentions) < 2:
            # Fewer than two mentions ⇒ no relatable pair (S5 AC: no error).
            return
        # Mentions in document order, so the "between" text and covering span are
        # well-defined. Mentions always carry a char span (contract-enforced).
        ordered = sorted(mentions, key=lambda m: (m.provenance.char_start or 0, m.record_id))
        candidates = self._find_candidates(ordered, text)
        # Deterministic order within a document: by covering-span start, then label,
        # then head/tail ids.
        candidates.sort(key=lambda c: (c[3], c[0].label, c[1].record_id, c[2].record_id))
        # Quadratic-fan-out guard (ADR-0033 / #63): bound the relations one document may
        # emit. Applied AFTER the deterministic sort so the retained relations are the
        # lowest-ordered stable prefix (never a random subset — determinism preserved),
        # and the dropped count is audited into the Quality Report. A cap <= 0 disables it.
        cap = self._max_relations_per_document
        if 0 < cap < len(candidates):
            self.fanout_dropped_relations += len(candidates) - cap
            self.fanout_capped_documents += 1
            candidates = candidates[:cap]
        page_index = PageOffsetIndex(page_map) if page_map is not None else None
        for i, (rule, head, tail, span_start, span_end) in enumerate(candidates):
            provenance = self._relation_provenance(
                head, tail, span_start, span_end, page_index
            )
            yield RelationMention(
                record_id=f"{doc_id}#relation-{i}",
                document_record_id=doc_id,
                index=i,
                label=rule.label,
                head_mention_id=head.record_id,
                tail_mention_id=tail.record_id,
                head_text=head.text,
                tail_text=tail.text,
                confidence=rule.confidence,
                provenance=provenance,
                classification=head.classification,
            )

    def _find_candidates(
        self, ordered: list[EntityMention], text: str
    ) -> list[_Candidate]:
        """All rule-matching (head, tail) pairs within the window, non-duplicated.

        For every ordered pair of mentions close enough together, every rule whose
        head/tail labels fit tests its trigger against the between-text. A pair may
        satisfy at most one relation per label (first rule wins for that label), so
        the candidate set is deterministic. A symmetric rule also matches the reverse
        label order but always emits head=the rule's head-typed mention.
        """
        candidates: list[_Candidate] = []
        n = len(ordered)
        for a in range(n):
            first = ordered[a]
            first_start = first.provenance.char_start or 0
            first_end = first.provenance.char_end or first_start
            for b in range(a + 1, n):
                second = ordered[b]
                second_start = second.provenance.char_start or 0
                second_end = second.provenance.char_end or second_start
                gap = second_start - first_end
                if gap > self._window_chars:
                    # Ordered by start, so no later mention is closer — stop scanning.
                    break
                between = text[first_end:second_start] if text else ""
                span_start = min(first_start, second_start)
                span_end = max(first_end, second_end)
                matched_labels: set[str] = set()
                for rule in self._rules:
                    if rule.label in matched_labels:
                        continue
                    head, tail = self._orient(rule, first, second)
                    if head is None or tail is None:
                        continue
                    if rule.trigger is not None and not rule.trigger.search(between):
                        continue
                    candidates.append((rule, head, tail, span_start, span_end))
                    matched_labels.add(rule.label)
        return candidates

    @staticmethod
    def _orient(
        rule: _Rule, first: EntityMention, second: EntityMention
    ) -> tuple[EntityMention | None, EntityMention | None]:
        """Resolve which mention is head vs tail for a rule, or (None, None) if unfit.

        In declared order the head-typed mention is ``first`` and the tail-typed is
        ``second``. A symmetric rule also accepts the reverse (``first`` is the tail
        type, ``second`` the head type), still emitting the head-typed mention as head.
        """
        if first.label == rule.head_label and second.label == rule.tail_label:
            return first, second
        if rule.symmetric and first.label == rule.tail_label and second.label == rule.head_label:
            return second, first
        return None, None

    # -- text + page-map reconstruction from chunks -------------------------

    @staticmethod
    def _document_texts(chunks: list[ChunkRecord]) -> dict[str, str]:
        """Reconstruct each document's assembled text from its chunks' spans.

        Each chunk carries the half-open ``[char_start, char_end)`` it occupies in the
        parent document (S3), so writing every chunk's content at its own start
        reconstructs the document text at exact document offsets — the coordinate
        system a relation's covering span and its trigger between-text both use.
        """
        by_doc_chunks: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            by_doc_chunks.setdefault(chunk.document_record_id, []).append(chunk)
        texts: dict[str, str] = {}
        for doc_id, doc_chunks in by_doc_chunks.items():
            end = 0
            for chunk in doc_chunks:
                start = chunk.provenance.char_start or 0
                end = max(end, start + len(chunk.content))
            buf = [" "] * end
            for chunk in doc_chunks:
                start = chunk.provenance.char_start or 0
                buf[start : start + len(chunk.content)] = chunk.content
            texts[doc_id] = "".join(buf)
        return texts

    @staticmethod
    def _document_page_maps(chunks: list[ChunkRecord]) -> dict[str, PageMap | None]:
        """The parent page map per document (carried on chunks since S4), if present."""
        maps: dict[str, PageMap | None] = {}
        for chunk in chunks:
            if chunk.document_record_id not in maps and chunk.page_map is not None:
                maps[chunk.document_record_id] = chunk.page_map
        return maps

    @staticmethod
    def _relation_provenance(
        head: EntityMention,
        tail: EntityMention,
        span_start: int,
        span_end: int,
        page_index: PageOffsetIndex | None,
    ) -> Provenance:
        """Provenance for a relation: shared lineage + the covering span AND its page.

        The char span covers both endpoints (``[span_start, span_end)``). The page
        span is re-resolved from that covering span through the parent's page map so
        the relation cites the page(s) its endpoints actually fall on (``page_end``
        from the last included character so a relation straddling a page boundary
        records both pages). Absent a page map, it degrades to the union of the two
        mentions' own recorded pages. Source-document identity is inherited from the
        head mention (head and tail share a document).
        """
        base = head.provenance
        update: dict[str, int] = {"char_start": span_start, "char_end": span_end}
        if page_index is not None:
            update["page_start"] = page_index.page_for_offset(span_start)
            update["page_end"] = page_index.page_for_offset(max(span_end - 1, span_start))
        else:
            head_ps = head.provenance.page_start
            tail_ps = tail.provenance.page_start
            head_pe = head.provenance.page_end
            tail_pe = tail.provenance.page_end
            starts = [p for p in (head_ps, tail_ps) if p is not None]
            ends = [p for p in (head_pe, tail_pe) if p is not None]
            if starts:
                update["page_start"] = min(starts)
            if ends:
                update["page_end"] = max(ends)
        return base.model_copy(update=update)
