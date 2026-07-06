"""Built-in reference EntityExtractor Provider (deterministic, zero-shot, zero-dep).

The S4 Entity Extraction reference Provider (ADR-0004; ships in core like the S3
Chunker/Screeners so the demo runs end to end, ADR-0007 CPU-first, ADR-0016 no
extra deps). It is a **zero-shot** gazetteer/pattern extractor: the entity types
and how to recognise them are supplied entirely through Stage ``config`` — the
Capability names no labels and core hardcodes none. A Pipeline says "find these
labels with these seed terms / these regexes" and the Provider tags every
occurrence.

This is the CPU-viable, dependency-free, hash-stable default the Baseline bar
needs (no model download, so a seeded run is byte-identical). It is not a weaker
reinvention of a learned NER model — the real learned Provider is the separately
packaged ``latence-ner-gliner`` (a permissively-licensed GLiNER checkpoint via
plain ``transformers``, ADR-0016 heavy-dep isolation), a drop-in behind this same
seam. Shipping the deterministic reference in core keeps first contact
(``git clone`` → working extraction) free of torch.

Each emitted :class:`~latence_core.contracts.EntityMention` carries:

* its ``label`` (from the configured zero-shot label set) and a ``confidence``;
* Provenance ``char_start``/``char_end`` — the mention's half-open span **in the
  parent document's assembled markdown**, computed by offsetting the chunk-local
  match position by the chunk's own ``provenance.char_start`` (the chunk records
  its span in the parent document, S3), so the mention resolves back to the exact
  source offsets;
* Provenance ``page_start``/``page_end`` — the source page(s) that span falls on,
  resolved from the mention's *own* document offset through the parent's page map
  (carried on the chunk) via :class:`~latence_core.pagemap.PageOffsetIndex` — the
  S4 round-trip (mention → offset → original page). A mention on a later page of a
  multi-page chunk resolves to that page, not the chunk's first page. When the
  chunk carries no page map, it degrades to the chunk's inherited page range;
* the parent chunk's Classification, inherited unchanged.

An empty or near-empty chunk yields no mentions (never an error) — the S4
acceptance criterion.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.contracts import EntityMention
from latence_core.pagemap import PageOffsetIndex

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, Provenance

# A single non-overlapping match found in a chunk's text, before it is turned into
# a contract: the label it matched, the surface string, the chunk-local half-open
# span, and the confidence assigned.
_Match = tuple[str, str, int, int, float]


class GazetteerEntityExtractor:
    """Zero-shot gazetteer/regex EntityExtractor Provider.

    The label set and its recognisers come entirely from ``config`` — nothing is
    hardcoded in core, so the same Provider extracts ``PERSON``/``ORG`` for one
    Pipeline and ``GENE``/``DRUG`` for another (zero-shot, S4 AC).

    Config:
        labels: dict[str, dict]  — the zero-shot label set. Each key is an entity
            type; each value configures how to find it:
              * ``terms``: list[str]  — literal surface strings (case-insensitive,
                whole-word) to tag as this type. A gazetteer.
              * ``patterns``: list[str]  — regexes whose whole match is tagged as
                this type (e.g. an email or an id format).
              * ``confidence``: float  — the score to stamp on mentions of this
                type (default ``default_confidence``).
            ``labels`` may also be given as ``list[str]`` for pattern-free label
            declaration (produces the empty extractor for those types — useful when
            only the label vocabulary matters and recognisers arrive via config).
        default_confidence: float  — confidence when a label sets none (default 0.9).
        case_sensitive: bool  — match ``terms`` case-sensitively (default False).
        min_chars: int  — a chunk whose stripped length is below this is treated as
            near-empty and yields no mentions (default 1).

    Determinism: matches are emitted in ``(char_start, label)`` order, so identical
    input + config yields byte-identical mentions.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._default_confidence = float(config.get("default_confidence", 0.9))
        self._case_sensitive = bool(config.get("case_sensitive", False))
        self._min_chars = int(config.get("min_chars", 1))
        self._recognisers = self._compile_labels(config.get("labels", {}))

    # -- config compilation -------------------------------------------------

    def _compile_labels(
        self, labels: dict[str, Any] | list[str]
    ) -> list[tuple[str, re.Pattern[str], float]]:
        """Compile the zero-shot label set into ``(label, pattern, confidence)`` rules.

        A list-of-labels form declares the vocabulary with no recognisers (no
        mentions until terms/patterns are given). The dict form compiles each
        label's literal ``terms`` (word-boundary, escaped) and raw ``patterns`` into
        one alternation per label, so recognition is a single scan per label.
        """
        rules: list[tuple[str, re.Pattern[str], float]] = []
        flags = 0 if self._case_sensitive else re.IGNORECASE
        if isinstance(labels, list):
            # Vocabulary-only declaration: valid, but recognises nothing.
            return rules
        for label, spec in labels.items():
            if not isinstance(spec, dict):
                msg = f"Label {label!r} config must be a mapping (got {type(spec).__name__})."
                raise TypeError(msg)
            confidence = float(spec.get("confidence", self._default_confidence))
            alternatives: list[str] = []
            for term in spec.get("terms", []):
                # Whole-word, literal: escape the term and bound it so "Amazon" does
                # not match inside "Amazonian".
                alternatives.append(rf"\b{re.escape(str(term))}\b")
            alternatives.extend(str(pat) for pat in spec.get("patterns", []))
            if not alternatives:
                continue
            pattern = re.compile("|".join(f"(?:{alt})" for alt in alternatives), flags)
            rules.append((label, pattern, confidence))
        return rules

    # -- extraction ---------------------------------------------------------

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        for chunk in chunks:
            yield from self._extract_one(chunk)

    def _extract_one(self, chunk: ChunkRecord) -> Iterator[EntityMention]:
        text = chunk.content
        # Empty / near-empty chunk: no mentions, no error (S4 AC).
        if len(text.strip()) < self._min_chars:
            return
        matches = self._find_matches(text)
        # The chunk records its own span in the parent document (S3); a mention's
        # document-level offset is the chunk-local match offset shifted by the
        # chunk's start. The chunk always carries a char span (S3 invariant); fall
        # back to 0 only defensively so a mention always has an offset.
        chunk_char_start = chunk.provenance.char_start or 0
        # Resolve each mention's OWN page from the parent's page map (carried on the
        # chunk), not the chunk's whole page range: a mention on a later page of a
        # multi-page chunk must cite that page (S4 "correct source page"). Built once
        # per chunk. Absent (pre-S4 chunk / no page map) → inherit the chunk range.
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None
        for i, (label, surface, local_start, local_end, confidence) in enumerate(matches):
            provenance = self._mention_provenance(
                chunk.provenance,
                doc_char_start=chunk_char_start + local_start,
                doc_char_end=chunk_char_start + local_end,
                page_index=page_index,
            )
            yield EntityMention(
                record_id=f"{chunk.record_id}#entity-{i}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=i,
                label=label,
                text=surface,
                confidence=confidence,
                provenance=provenance,
                classification=chunk.classification,
            )

    def _find_matches(self, text: str) -> list[_Match]:
        """All non-overlapping matches across every label, deterministically ordered.

        Each label is scanned independently; overlaps *between* labels are resolved
        by preferring the earlier start, then the longer span, then the label name —
        so a fixed, deterministic set of non-overlapping mentions results (identical
        input ⇒ identical output). Within a label ``finditer`` is already
        non-overlapping.
        """
        raw: list[_Match] = []
        for label, pattern, confidence in self._recognisers:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                if end > start:
                    raw.append((label, m.group(0), start, end, confidence))
        # Deterministic ordering: by start, then longer span first, then label.
        raw.sort(key=lambda t: (t[2], -(t[3] - t[2]), t[0]))
        # Drop any match that overlaps one already accepted (first-wins under the
        # sort above), so the mention set is non-overlapping and stable.
        accepted: list[_Match] = []
        last_end = -1
        for match in raw:
            _, _, start, end, _ = match
            if start >= last_end:
                accepted.append(match)
                last_end = end
        return accepted

    @staticmethod
    def _mention_provenance(
        chunk_prov: Provenance,
        doc_char_start: int,
        doc_char_end: int,
        page_index: PageOffsetIndex | None,
    ) -> Provenance:
        """Provenance for a mention: chunk lineage + the mention's own span AND page.

        The char span is narrowed to the mention's own ``[doc_char_start,
        doc_char_end)`` in the parent document. The page span is re-resolved from
        that offset through the parent's page map (``page_index``) so a mention on a
        later page of a multi-page chunk cites *its own* page — the ``page_end`` uses
        the last included character (``doc_char_end - 1``) so a mention straddling a
        page boundary records both pages. When no page map is available the mention
        degrades to inheriting the chunk's page range.
        """
        update: dict[str, int] = {
            "char_start": doc_char_start,
            "char_end": doc_char_end,
        }
        if page_index is not None:
            update["page_start"] = page_index.page_for_offset(doc_char_start)
            update["page_end"] = page_index.page_for_offset(max(doc_char_end - 1, doc_char_start))
        return chunk_prov.model_copy(update=update)
