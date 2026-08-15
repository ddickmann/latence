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
  falls on, taken from the two endpoint mentions' own exactly-resolved pages via
  :func:`~latence_core.page_index.pages_for_covering_span` — the S5 round-trip
  (relation → offset → original page). A covering span is not a sub-chunk offset:
  its endpoints may sit in different chunks, so it is answered by arithmetic on the
  endpoints rather than by a page lookup against any one chunk;
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

**Input-side mention cap (#68).** The ``max_relations_per_document`` cap bounds the
*emitted output*, but :meth:`_find_candidates` still materialises the full
``O(mentions²)`` candidate list in memory **before** the sort trims it — so the
transient itself is unbounded even when the output is capped. The inner-loop ``break``
bounds that transient only under a small ``window_chars`` (the shipped default 200
keeps pairs local); a large operator-configured ``window_chars`` defeats the break and
lets one document's candidate list grow quadratic before the output cap ever applies
(operator-config-gated, the same trust tier as the output cap). Mirroring the #33
input-side discipline exactly (profiling caps the *distinct entities* that enter its
``O(n²)`` pairing), the Provider caps the *mentions per document* that enter the
candidate scan at ``max_mentions_per_document`` (default 5_000) — so the transient is
``O(cap²)`` regardless of ``window_chars``, not just the output. The cap keeps the
lowest-ordered mentions (by document char offset, then record id) — the same
document-order prefix the scan already relies on, so covering spans, page provenance,
and Evidence offsets of the retained relations are unchanged; it is a stable,
deterministic prefix, never a random subset. Documents whose mention set was trimmed
and the number of mentions dropped are tallied on the Provider
(``mention_capped_documents`` / ``mention_dropped_mentions``) and surfaced into the
Quality Report's RelationQuality — the audited marker, so a bounded document is never
silently truncated.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.contracts import RelationMention
from latence_core.page_index import pages_for_covering_span
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    REPO_LICENSE_SOURCE,
    ProviderProfile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord, EntityMention, Provenance


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

# Default per-document INPUT mention cap (#68): at most this many mentions from one
# document enter the O(mentions²) candidate scan, so the transient candidate list is
# O(cap²) rather than O(mentions²) — bounding it BEFORE the output cap trims, not just
# after (which the #63 ``max_relations_per_document`` cap alone leaves unbounded when a
# large ``window_chars`` defeats the inner-loop break). Sized generously above any
# realistic document's mention count so it never trims real input, but a pathological
# same-typed mention storm is bounded on the input side. This mirrors profiling's
# ``_MAX_ENTITIES_PER_DOCUMENT`` (#33). Configurable via ``max_mentions_per_document``;
# a value ``<= 0`` disables it (unbounded transient — not recommended on untrusted
# corpora with a wide ``window_chars``).
_MAX_MENTIONS_PER_DOCUMENT = 5_000


def _placed_chars(chunk: ChunkRecord) -> Iterator[tuple[int, str]]:
    """Yield ``(original_offset, char)`` for a chunk's surviving content (issue #74).

    Chunk ``content`` is markup-stripped and its ``offset_map`` records, per surviving
    run, ``(local_start, doc_start)`` — the original-markdown offset of the run's first
    character. Within a run the original offset of local char ``k`` is
    ``doc_start + (k - local_start)`` (stripping only deletes, so a run is a constant
    shift). Yielding each character at its TRUE original offset lets the document text be
    reconstructed in the mentions' coordinate system, so the relation between-text lookup
    aligns with the mention offsets. A chunk without an ``offset_map`` (pre-v11) degrades
    to the identity placement at ``char_start`` (the markup-free / legacy path).
    """
    content = chunk.content
    offset_map = chunk.offset_map
    if offset_map is None:
        base = chunk.provenance.char_start or 0
        for k, ch in enumerate(content):
            yield base + k, ch
        return
    segments = offset_map.segments
    n = len(content)
    for s, seg in enumerate(segments):
        run_start = seg.local_start
        run_end = segments[s + 1].local_start if s + 1 < len(segments) else n
        for k in range(run_start, min(run_end, n)):
            yield seg.doc_start + (k - run_start), content[k]


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
            lowest-ordered stable prefix. Bounds the O(mentions²) OUTPUT a
            low-cardinality corpus produces. A value ``<= 0`` disables the cap
            (unbounded, the old behaviour) — not recommended on untrusted corpora.
        max_mentions_per_document: int  — input-side mention cap (#68): the max
            mentions from one document that enter the O(mentions²) candidate scan
            (default 5_000), so the transient candidate list is O(cap²) — bounding the
            transient itself, not just the emitted output. The retained mentions are the
            lowest-ordered by document char offset (then record id), the same
            document-order prefix the scan relies on, so covering spans / page
            provenance / Evidence offsets are unchanged. A value ``<= 0`` disables it
            (unbounded transient) — not recommended on untrusted corpora with a wide
            ``window_chars``.

    Audit attributes (read after ``relate`` returns, e.g. by the Runner into the
    Quality Report — the audited "capped" markers):
        fanout_capped_documents: int  — how many documents hit the output relation cap
            (ADR-0033 / #63).
        fanout_dropped_relations: int  — total relations dropped across those documents.
        mention_capped_documents: int  — how many documents hit the input mention cap
            (#68).
        mention_dropped_mentions: int  — total mentions dropped across those documents.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic,
        Apache-2.0.

        A trigger-phrase relation extractor ships no model weights (``model_id=None``) and
        is a pure-Python part of ``latence-core`` (Apache-2.0, verified).
        Deterministic (candidates emitted in covering-span/label/id order), CPU-only, no batching,
        no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license=REPO_LICENSE_CODE,
            license_verified=True,
            license_code=REPO_LICENSE_CODE,
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-06",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._default_confidence = float(config.get("default_confidence", 0.75))
        self._case_sensitive = bool(config.get("case_sensitive", False))
        self._window_chars = int(config.get("window_chars", 200))
        self._max_relations_per_document = int(
            config.get("max_relations_per_document", _MAX_RELATIONS_PER_DOCUMENT)
        )
        self._max_mentions_per_document = int(
            config.get("max_mentions_per_document", _MAX_MENTIONS_PER_DOCUMENT)
        )
        self._rules = self._compile_rules(config.get("relations", {}))
        # Audited fan-out-guard tallies (ADR-0033 / #63), reset per ``relate`` call so a
        # reused Provider instance reports only the current run's capping.
        self.fanout_capped_documents = 0
        self.fanout_dropped_relations = 0
        # Audited input-side mention-cap tallies (#68), likewise reset per ``relate`` call.
        self.mention_capped_documents = 0
        self.mention_dropped_mentions = 0

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
        self.mention_capped_documents = 0
        self.mention_dropped_mentions = 0
        chunk_list = list(chunks)
        # Reconstruct per-document text + page map from the chunks (each chunk knows
        # its own [char_start, char_end) in the parent document, S3). This lets a
        # trigger phrase be tested against the real between-mention text, and lets a
        # relation's covering span resolve to a page.
        doc_text = self._document_texts(chunk_list)

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
            )

    def _relate_one_document(
        self,
        doc_id: str,
        mentions: list[EntityMention],
        text: str,
    ) -> Iterator[RelationMention]:
        if len(mentions) < 2:
            # Fewer than two mentions ⇒ no relatable pair (S5 AC: no error).
            return
        # Mentions in document order, so the "between" text and covering span are
        # well-defined. Mentions always carry a char span (contract-enforced).
        ordered = sorted(mentions, key=lambda m: (m.provenance.char_start or 0, m.record_id))
        # Input-side mention cap (#68): bound the mentions that enter the O(mentions²)
        # candidate scan BEFORE it materialises the transient, so the candidate list is
        # O(cap²) regardless of ``window_chars`` (the #63 output cap alone leaves the
        # transient unbounded when a wide window defeats the inner-loop break). The
        # lowest-ordered prefix is kept — the same document-order the scan relies on, so
        # covering spans / page provenance / Evidence offsets of retained relations are
        # unchanged. A cap <= 0 disables it. The drop is audited into the Quality Report.
        mention_cap = self._max_mentions_per_document
        if 0 < mention_cap < len(ordered):
            self.mention_dropped_mentions += len(ordered) - mention_cap
            self.mention_capped_documents += 1
            ordered = ordered[:mention_cap]
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
        for i, (rule, head, tail, span_start, span_end) in enumerate(candidates):
            provenance = self._relation_provenance(
                head,
                tail,
                span_start,
                span_end,
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
        """Reconstruct each document's text at TRUE original-markdown offsets.

        The reconstructed buffer is the coordinate system a relation's between-trigger
        lookup reads: ``text[first_end:second_start]`` where ``first_end`` and
        ``second_start`` are the mentions' *original-markdown* offsets (Entity
        Extraction stamps a mention's TRUE original offset through the chunk's
        ``offset_map``, ADR-0031). The reconstruction MUST therefore be in that same
        coordinate system.

        Chunk ``content`` is markup-stripped, so placing it naively at the chunk's
        ``char_start`` (``buf[start:start+len(content)] = content``) writes each
        surviving character one-for-one from ``char_start`` — which is only correct
        when *no* markup was stripped inside the chunk. Once a chunk contains stripped
        markup its content is shorter than its ``[char_start, char_end)`` span, so the
        naive placement shifts every character after the first stripped run left of its
        real offset (issue #74): the between-text lookup then reads the wrong region and
        a genuine trigger between two mentions is missed (false negative) — or a
        different span is shifted into the window (false positive).

        The fix places each chunk's content through its ``offset_map``: the map's
        ``(local_start, doc_start)`` runs give each surviving character's TRUE original
        offset (``doc_start + (local - local_start)`` within a run), so a character is
        written at exactly the offset the mention offsets use. Stripped-away positions
        (the markup) stay as space gaps — harmless, since triggers are matched against
        the surviving text and the window aligns with the mention coordinates. A chunk
        with no ``offset_map`` (pre-v11) degrades to the identity placement at
        ``char_start`` (unchanged behaviour on the markup-free / legacy path).
        """
        by_doc_chunks: dict[str, list[ChunkRecord]] = {}
        for chunk in chunks:
            by_doc_chunks.setdefault(chunk.document_record_id, []).append(chunk)
        texts: dict[str, str] = {}
        for doc_id, doc_chunks in by_doc_chunks.items():
            placements = [
                (orig, ch) for chunk in doc_chunks for orig, ch in _placed_chars(chunk)
            ]
            end = 0
            for orig, _ in placements:
                if orig + 1 > end:
                    end = orig + 1
            buf = [" "] * end
            for orig, ch in placements:
                buf[orig] = ch
            texts[doc_id] = "".join(buf)
        return texts

    @staticmethod
    def _relation_provenance(
        head: EntityMention,
        tail: EntityMention,
        span_start: int,
        span_end: int,
    ) -> Provenance:
        """Provenance for a relation: shared lineage + the covering span AND its page.

        The char span covers both endpoints (``[span_start, span_end)``), so the relation
        cites the page(s) its endpoints actually fall on and one straddling a page boundary
        records both. Those pages come from the endpoints themselves
        (:func:`~latence_core.page_index.pages_for_covering_span`) rather than from a lookup:
        a covering span is not a sub-chunk offset — its endpoints may be in different chunks
        — and each endpoint already resolved its own page exactly when it was emitted.

        This used to resolve the span against an arbitrary "anchor" chunk of the document,
        which only worked because that chunk carried the WHOLE document's map, and fell back
        to the endpoint union whenever it did not. The fallback was the correct answer all
        along; the lookup was the redundant path. Source-document identity is inherited from
        the head mention (head and tail share a document).
        """
        base = head.provenance
        update: dict[str, int] = {"char_start": span_start, "char_end": span_end}
        pages = pages_for_covering_span(head.provenance, tail.provenance)
        if pages is not None:
            update["page_start"], update["page_end"] = pages
        return base.model_copy(update=update)
