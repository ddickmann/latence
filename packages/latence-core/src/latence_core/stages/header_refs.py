"""Built-in structured-header cross-reference Provider (deterministic, zero-shot, zero-dep).

The gap this closes (dogfood-3-inspect). An entity-anchored RelationExtractor — the
learned fused ``fused_entity_relation.gliner2`` and the in-core trigger-phrase
:class:`~latence_core.stages.relation.PatternRelationExtractor` alike — can only relate two
mentions that BOTH surface as tagged entities *near each other* in the text. A document's most
explicit relations are often carried in **self-describing metadata headers** whose targets are
*bare identifiers*, not entity-shaped surface strings, and whose subject is the document itself
(named once, far away, in the front matter). An RFC's ``Obsoletes: 7230, 7231`` / ``Updates: 3864``
lines are the canonical example: ``7230`` is never tagged as an entity, and the subject
(``RFC 9110``) is declared only in the ``Request for Comments: 9110`` header — so zero-shot RE
recovers none of them and the induced knowledge graph misses the corpus's most explicit edges.

This Provider recovers exactly those relations, **deterministically** and with real Evidence: it
finds the document's own identifier (its *self* id) and, for each configured header relation,
mines the referenced target identifiers out of the header line — synthesizing an
:class:`~latence_core.contracts.EntityMention` for the self id AND for each referenced id (so the
Graph gets real nodes to connect) plus a directed :class:`~latence_core.contracts.RelationMention`
``self -> target``. Because it fulfils the SAME
:class:`~latence_core.capability.FusedEntityRelationExtractor` seam as gliner2 (``extract_fused``
returns mentions + relations), it slots into a Pipeline as an extraction Stage whose mentions AND
relations thread downstream to Disambiguation and Graph Assembly with no other change — and, using a
disjoint ``#href-`` record-id namespace, it composes ALONGSIDE a learned fused Stage without id
collision (both feed the same Disambiguation input, the Runner merges by carrier type).

Zero-shot (ADR-0007 CPU-first, ADR-0016 no extra deps, ADR-0004 ships in core like the reference
Chunker / Screeners / gazetteer / ``relation.pattern``): the self recogniser, the identifier
rendering, and every header relation's recogniser come **entirely from Stage** ``config`` — core
hardcodes no relation types and no ``Obsoletes``/``Updates`` literals. The same Provider mines
``Obsoletes:`` for one Pipeline and ``Supersedes:`` / ``References:`` / an e-mail ``In-Reply-To:``
for another, with no code change.

Determinism: mentions and relations are emitted in chunk (input) order, targets in header-line
order, so identical input + config yields byte-identical output (the Baseline bar). Offsets and
pages are resolved EXACTLY like the learned providers — chunk-local model offsets are lifted to true
original-markdown offsets through the chunk's stripped-``content``->original ``offset_map`` via
:class:`~latence_core.offsetmap.OffsetIndex` (a pre-v11 chunk degrades to the ``char_start + local``
shift), and each span's source page is resolved through the document-scoped
:class:`~latence_core.page_index.PageIndexResolver` (degrading to the chunk's inherited page range
when no page map is available for the document). The self id and a relation's header line are
matched **within one chunk** — for
front-matter metadata headers (RFC, e-mail, memo) the self declaration and the header lines are
co-located in the same leading chunk, so this is exact; a chunk carrying header lines but no self
declaration yields nothing (never an error, never a dangling edge).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from latence_core.capability import FusedExtraction
from latence_core.contracts import EntityMention, RelationMention
from latence_core.offsetmap import OffsetIndex
from latence_core.page_index import PageIndexResolver, pages_for_covering_span
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, Provenance

# Default per-relation, per-chunk target fan-out cap: at most this many referenced ids are mined out
# of one header line for one relation, so a pathological header (a comma list of thousands) cannot
# drive the synthesized mention/relation count — and every downstream graph edge — unbounded. Sized
# generously above any realistic metadata header. Configurable via ``max_targets_per_relation``; a
# value ``<= 0`` disables the cap (unbounded — not recommended on untrusted corpora).
_MAX_TARGETS_PER_RELATION = 1000


class _RelationRule:
    """One compiled header relation: its label + the line recogniser that carries the targets."""

    __slots__ = ("label", "line", "confidence")

    def __init__(self, label: str, line: re.Pattern[str], confidence: float) -> None:
        self.label = label
        self.line = line
        self.confidence = confidence


class HeaderReferenceExtractor:
    """Zero-shot deterministic structured-header cross-reference Provider (fused NER + RE).

    Fulfils BOTH the :class:`~latence_core.capability.EntityExtractor` and the
    :class:`~latence_core.capability.FusedEntityRelationExtractor` seams: ``extract_fused`` returns
    the synthesized identifier mentions AND the header relations between them together (one pass),
    and ``extract`` exposes the NER half alone.

    Config (every recogniser is supplied per Pipeline — zero-shot; core hardcodes none):
        self_pattern: str  — REQUIRED. A regex whose **first capturing group** is the document's own
            identifier as it appears in the text (e.g. ``r"Request for Comments:\\s*(\\d+)"`` for an
            RFC). The first match in a chunk anchors that chunk's header relations. A chunk with no
            self match yields nothing (the relations have no subject to hang on).
        relations: dict[str, dict]  — REQUIRED, non-empty. The header relation label set. Each key
            is a relation type; each value configures how to recognise it:
              * ``header``: str  — REQUIRED. A regex whose **first capturing group** is the raw
                substring carrying the referenced identifiers (e.g.
                ``r"(?im)^Obsoletes:\\s*([0-9,\\s]+)"``). Matched against the chunk text; every
                match contributes its targets.
              * ``confidence``: float  — the score to stamp on this relation's edges (default
                ``default_confidence``).
        target_pattern: str  — a regex whose **first capturing group** is ONE referenced
            identifier, applied to each ``header`` match's captured substring to mine the individual
            targets (default ``r"(\\d+)"`` — the bare-number RFC form). The self id is matched with
            this same pattern against ``self_pattern``'s captured group, so both share one id shape.
        id_template: str  — how a mined raw id is rendered into the canonical mention surface / node
            name (default ``"{id}"``; e.g. ``"RFC {id}"`` so a synthesized node reads ``RFC 7230``
            and disambiguates against an in-text ``RFC 7230`` mention). Must contain ``{id}``.
        entity_label: str  — the entity type stamped on every synthesized identifier mention
            (default ``"reference"``).
        default_confidence: float  — the confidence for a relation whose rule sets none (default
            ``1.0``: a matched metadata header is a certain, deterministic signal).
        case_sensitive: bool  — compile ``self_pattern`` / ``relations[*].header`` /
            ``target_pattern`` case-sensitively (default ``False``). Note a caller-supplied inline
            flag (e.g. ``(?i)``/``(?m)``) in a pattern is always honoured on top of this.
        max_targets_per_relation: int  — per-relation, per-chunk fan-out cap (default 1000); the max
            referenced ids mined from one header line for one relation, bounding the synthesized
            record count. A value ``<= 0`` disables it (unbounded — not recommended on untrusted
            corpora).

    The self id and a header line are recognised **within one chunk**: front-matter metadata headers
    (the self declaration and the ``Obsoletes:``/``Updates:`` lines) sit together in the leading
    chunk, so this is exact for the intended shape; header lines in a chunk with no self declaration
    are left alone. A header that lists the document's own id drops that self-referential target (no
    self-loop edge). Distinct target ids fold to one mention per canonical surface within a chunk,
    so a target named under two relations is one node with two edges.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0.

        A regex header-reference extractor ships no model weights (``model_id=None``) and is a
        pure-Python part of ``latence-core`` (Apache-2.0, verified). Deterministic (mentions/
        relations emitted in chunk + header-line order), CPU-only, no batching, no per-call cost.
        """
        return ProviderProfile(
            compute="cpu",
            memory_mb=16,
            model_id=None,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
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
            msg = "HeaderReferenceExtractor requires a non-empty 'entity_label'."
            raise ValueError(msg)
        self._id_template = str(config.get("id_template", "{id}"))
        if "{id}" not in self._id_template:
            msg = "HeaderReferenceExtractor 'id_template' must contain the '{id}' placeholder."
            raise ValueError(msg)
        self._max_targets = int(config.get("max_targets_per_relation", _MAX_TARGETS_PER_RELATION))

        self_pattern = config.get("self_pattern")
        if not self_pattern or not isinstance(self_pattern, str):
            msg = (
                "HeaderReferenceExtractor requires a non-empty 'self_pattern' regex in its config "
                "(the first capturing group is the document's own identifier)."
            )
            raise ValueError(msg)
        self._self_re = self._compile(self_pattern, "self_pattern", flags)

        self._target_re = self._compile(
            str(config.get("target_pattern", r"(\d+)")), "target_pattern", flags
        )

        relations = config.get("relations")
        if not relations or not isinstance(relations, dict):
            msg = (
                "HeaderReferenceExtractor requires a non-empty 'relations' mapping in its config "
                "(zero-shot RE: each header relation label + its line recogniser is supplied "
                "per Pipeline)."
            )
            raise ValueError(msg)
        self._rules = self._compile_rules(relations, flags)

    # -- config compilation -------------------------------------------------

    @staticmethod
    def _compile(pattern: str, name: str, flags: int) -> re.Pattern[str]:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            msg = f"HeaderReferenceExtractor {name!r} is not a valid regex: {exc}"
            raise ValueError(msg) from exc
        if compiled.groups < 1:
            msg = (
                f"HeaderReferenceExtractor {name!r} must have a capturing group for the identifier."
            )
            raise ValueError(msg)
        return compiled

    def _compile_rules(self, relations: dict[str, Any], flags: int) -> list[_RelationRule]:
        rules: list[_RelationRule] = []
        for label, spec in relations.items():
            label_str = str(label)
            if not label_str:
                msg = "HeaderReferenceExtractor relation labels must be non-empty."
                raise ValueError(msg)
            if not isinstance(spec, dict):
                msg = (
                    f"HeaderReferenceExtractor relation {label_str!r} config must be a mapping "
                    f"(got {type(spec).__name__})."
                )
                raise TypeError(msg)
            header = spec.get("header")
            if not header or not isinstance(header, str):
                msg = (
                    f"HeaderReferenceExtractor relation {label_str!r} must declare a 'header' "
                    "regex whose first capturing group carries the referenced identifiers."
                )
                raise ValueError(msg)
            line = self._compile(header, f"relations[{label_str!r}].header", flags)
            confidence = float(spec.get("confidence", self._default_confidence))
            rules.append(_RelationRule(label_str, line, confidence))
        return rules

    # -- fused extraction ---------------------------------------------------

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Mine each chunk's self id + header relations; return synthesized mentions + relations.

        For each chunk: the first ``self_pattern`` match names the subject; each ``relations[*]``
        header match contributes its ``target_pattern`` ids as objects. A self mention plus one
        mention per distinct target surface are synthesized (offsets/pages resolved like the learned
        providers), and a directed ``self -> target`` :class:`RelationMention` is emitted per
        (relation, target). A chunk with no self match, or no header match, contributes nothing
        (never an error). Emission is chunk order, then header-line order — byte-identical per run.
        """
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        # ONE resolver for the whole call, purely so the drift roll-up is per-call. It is
        # stateless (v19): each chunk carries its own page spans, so a chunk with no self
        # declaration teaches it nothing and skipping one costs nothing.
        resolver = PageIndexResolver()
        for chunk in chunks:
            chunk_mentions, chunk_relations = self._extract_chunk(chunk, resolver)
            mentions.extend(chunk_mentions)
            relations.extend(chunk_relations)
        return FusedExtraction(mentions=mentions, relations=relations)

    def extract(self, chunks: Iterable[ChunkRecord]) -> list[EntityMention]:
        """The NER half alone (the synthesized identifier mentions) for an EntityExtractor Stage."""
        return self.extract_fused(chunks).mentions

    def _extract_chunk(
        self, chunk: ChunkRecord, resolver: PageIndexResolver
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        content = chunk.content
        self_match = self._self_re.search(content)
        if self_match is None:
            return [], []
        self_id = self._first_group(self_match)
        if self_id is None:
            return [], []
        self_text = self._id_template.format(id=self_id)

        offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        chunk_char_start = chunk.provenance.char_start or 0

        mentions: list[EntityMention] = []
        by_text: dict[str, EntityMention] = {}

        def get_or_create(text: str, local_start: int, local_end: int) -> EntityMention:
            existing = by_text.get(text)
            if existing is not None:
                return existing
            if offset_index is not None:
                doc_start, doc_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_start = chunk_char_start + local_start
                doc_end = chunk_char_start + local_end
            index = len(mentions)
            mention = EntityMention(
                record_id=f"{chunk.record_id}#href-entity-{index}",
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

        # The subject mention is the chunk's self declaration (created first ⇒ index 0).
        head = get_or_create(self_text, self_match.start(1), self_match.end(1))

        relations: list[RelationMention] = []
        seen_edges: set[tuple[str, str]] = set()
        for rule in self._rules:
            for header_match in rule.line.finditer(content):
                group_start = header_match.start(1)
                captured = header_match.group(1)
                if not captured:
                    continue
                emitted_for_header = 0
                for target_match in self._target_re.finditer(captured):
                    if 0 < self._max_targets <= emitted_for_header:
                        break
                    target_id = self._first_group(target_match)
                    if target_id is None:
                        continue
                    if target_id == self_id:
                        continue  # a header listing the document's own id ⇒ no self-loop edge
                    target_text = self._id_template.format(id=target_id)
                    edge_key = (rule.label, target_text)
                    if edge_key in seen_edges:
                        continue
                    local_start = group_start + target_match.start(1)
                    local_end = group_start + target_match.end(1)
                    tail = get_or_create(target_text, local_start, local_end)
                    if tail.record_id == head.record_id:
                        continue
                    seen_edges.add(edge_key)
                    emitted_for_header += 1
                    relations.append(
                        self._relation(chunk, len(relations), rule, head, tail)
                    )
        return mentions, relations

    def _relation(
        self,
        chunk: ChunkRecord,
        index: int,
        rule: _RelationRule,
        head: EntityMention,
        tail: EntityMention,
    ) -> RelationMention:
        span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
        span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
        return RelationMention(
            record_id=f"{chunk.record_id}#href-relation-{index}",
            document_record_id=chunk.document_record_id,
            index=index,
            label=rule.label,
            head_mention_id=head.record_id,
            tail_mention_id=tail.record_id,
            head_text=head.text,
            tail_text=tail.text,
            confidence=min(max(rule.confidence, 0.0), 1.0),
            provenance=_span_provenance(
                head.provenance,
                span_start,
                span_end,
                # A covering span runs from one endpoint to the other, so it is not a
                # sub-chunk offset and no single chunk answers for it. Both endpoints already
                # resolved their own pages exactly, from their own chunk's spans, so the
                # covering span's pages are their union — arithmetic, not a lookup.
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


def _span_provenance(
    base: Provenance,
    doc_char_start: int,
    doc_char_end: int,
    pages: tuple[int, int] | None,
) -> Provenance:
    """A synthesized record's Provenance: the chunk lineage + the resolved char/page span.

    The char span is the true original-markdown span of the surface (or, for a relation, the
    covering span of both endpoints). ``pages`` is what was resolved for that span — from the
    chunk's own page spans for a surface, or from the two endpoints' already-exact pages for a
    relation's covering span (:func:`~latence_core.page_index.pages_for_covering_span`). ``None``
    only when neither endpoint of a relation recorded a page at all, which inherits the chunk's
    own ``page_start``/``page_end`` (never left ``None`` — the mention contract requires a page
    span).
    """
    update: dict[str, int] = {"char_start": doc_char_start, "char_end": doc_char_end}
    if pages is not None:
        update["page_start"], update["page_end"] = pages
    else:
        if base.page_start is not None:
            update["page_start"] = base.page_start
        if base.page_end is not None:
            update["page_end"] = base.page_end
    return base.model_copy(update=update)
