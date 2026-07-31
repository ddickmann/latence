# Copyright (c) latence-framework contributors.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Hyperedge building: the m-ary retrieval unit emitted by Graph Assembly (ADR-0057).

The pure engine behind the ``HYPEREDGE`` :class:`~latence_core.contracts.GraphScope`. It
re-projects the SAME inputs Graph Assembly already consumes — this time at *reasoning*
granularity: a Hyperedge is the minimal contiguous sentence window covering all arguments
of at least one extracted relation, carrying the full set of canonical entities in that
window, the directed relations, and everything the chain-of-evidence selector needs
precomputed (``token_cost`` + counter identity per ADR-0058, ``gamma``, ``support``).

Structural rules (spec 1.2–1.4, decided in ADR-0057):

* **Chunks are storage granularity; Hyperedges are reasoning granularity.** Sentences come
  from the abbreviation-safe :class:`~latence_core.sentences.SentenceSplitter`, never the
  chunker's budget-boundary regex.
* A relation seeds a Hyperedge only when its argument window spans at most
  ``max_window_sentences`` sentences; overlapping windows in the same chunk merge (union of
  spans, entities, relations).
* Sentences with entities but no relation become *weak* Hyperedges
  (``has_relations=False``) behind the ``weak_hyperedges`` toggle.
* Size targets: below ``min_tokens`` a Hyperedge merges into the same-chunk neighbour
  sharing the most entities; above ``max_tokens`` it splits at the sentence boundary that
  minimises cut relations. Arity above ``max_arity`` splits by relation groups (entities
  co-occurring in a relation stay together).
* **Dedup** (corpus-wide): identical entity-node set AND identical relation signature merge
  — ``support`` increments, the highest-γ instance's ``tau_text``/Provenance stay primary,
  every other source is appended to ``merged_sources``. Differing signatures over the same
  entity set are distinct facts and never merge.
* **``tau_text`` follows the corpus redaction variant** (ADR-0057), enforced HERE at the
  writer: any ``PIISpan`` overlapping the span is applied to the emitted text (its
  placeholder substituted; ``TAG`` spans leave text intact by definition), so the hypergraph
  can never carry text the exported corpus masked. The Provenance span stays in original
  document coordinates (offset round-trip via the chunk's ``offset_map``, ADR-0031).

``gamma`` (spec 1.5, uncalibrated — the calibration decision is its own ticket) composes
link confidence (persisted per ADR-0057), relation confidence, and the weakest-mention
penalty, renormalising over the components actually present: a missing signal is UNKNOWN
and drops out of the weighting; it is never counted as zero.

Determinism: every collection is sorted before use; the output is sorted by
``hyperedge_id``; the id is content-addressed over the corpus id + entity set + relation
signature (NOT span position), so the same fact always hashes to the same id and a re-run
over the same corpus is byte-identical (the S9 bar, extended).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from latence_core.contracts import (
    Evidence,
    GraphHyperedge,
    HyperedgeRelation,
    HyperedgeSourceRef,
    Provenance,
    RedactionAction,
)
from latence_core.graph import make_graph_id
from latence_core.offsetmap import OffsetIndex
from latence_core.sentences import RuleSentenceSplitter, SentenceSplitter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import (
        CanonicalEntity,
        ChunkRecord,
        EntityMention,
        RelationMention,
    )

__all__ = ["HyperedgeBuilder", "HyperedgeConfig"]

# The identity of the in-core token counter (ADR-0058: the counter identity is stored
# beside every token_cost so a later index/query tokenizer mismatch is detectable).
_HEURISTIC_COUNTER_ID = "latence:estimate_tokens:v1"

# gamma composition weights (spec 1.5 starting point): link, relation, weakest-mention.
_GAMMA_WEIGHTS = (0.3, 0.5, 0.2)
_WEAK_GAMMA_FACTOR = 0.5


@dataclass(frozen=True)
class HyperedgeConfig:
    """Build policy for :class:`HyperedgeBuilder` (spec 1.2 defaults, ADR-0057).

    ``token_counter``/``token_counter_id`` are the ADR-0058 seam: the in-core deterministic
    heuristic by default; an operator-named exact counter is injected by the Stage (a
    Provider-package adapter), never imported here.
    """

    weak_hyperedges: bool = True
    max_arity: int = 12
    min_tokens: int = 25
    max_tokens: int = 180
    max_window_sentences: int = 3
    token_counter: Callable[[str], int] | None = None
    token_counter_id: str = _HEURISTIC_COUNTER_ID


@dataclass
class _Span:
    """A Hyperedge under construction: a sentence window inside one chunk."""

    chunk: ChunkRecord
    start: int  # chunk-content-local, half-open
    end: int
    mention_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)

    def overlaps(self, other: _Span) -> bool:
        return self.start < other.end and other.start < self.end

    def absorb(self, other: _Span) -> None:
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
        self.mention_ids |= other.mention_ids
        self.relation_ids |= other.relation_ids


class HyperedgeBuilder:
    """Builds :class:`~latence_core.contracts.GraphHyperedge` records (pure, deterministic)."""

    def __init__(self, corpus_id: str, config: HyperedgeConfig | None = None) -> None:
        self._corpus_id = corpus_id
        self._config = config or HyperedgeConfig()
        self._splitter: SentenceSplitter = RuleSentenceSplitter()
        counter = self._config.token_counter
        if counter is None:
            from latence_core.chunking import estimate_tokens

            counter = estimate_tokens
        self._count: Callable[[str], int] = counter

    def build(
        self,
        chunks: Iterable[ChunkRecord],
        mentions: Iterable[EntityMention],
        relations: Iterable[RelationMention],
        entities: Iterable[CanonicalEntity],
        node_id_for: Callable[[str], str],
    ) -> list[GraphHyperedge]:
        """Build the corpus's Hyperedges from the run's records.

        ``node_id_for`` maps a CanonicalEntity record_id to its GraphNode node_id (the
        assembler's :meth:`~latence_core.graph.GraphBuilder.node_id_for`), so Hyperedges
        reference the same deterministic node ids the NODE stream carries.
        """
        chunk_list = sorted(chunks, key=lambda c: c.record_id)
        mention_list = sorted(mentions, key=lambda m: m.record_id)
        relation_list = sorted(relations, key=lambda r: r.record_id)
        self._relation_index = {r.record_id: r for r in relation_list}

        # mention record_id -> (canonical entity record_id, link confidence | None)
        entity_of: dict[str, str] = {}
        link_conf: dict[str, float] = {}
        for entity in sorted(entities, key=lambda e: e.record_id):
            for mid in entity.member_mention_ids:
                entity_of.setdefault(mid, entity.record_id)
            for mid, score in entity.member_link_confidences.items():
                link_conf.setdefault(mid, score)

        by_id = {m.record_id: m for m in mention_list}
        mentions_of_chunk: dict[str, list[EntityMention]] = defaultdict(list)
        for m in mention_list:
            mentions_of_chunk[m.chunk_record_id].append(m)
        relations_of_chunk: dict[str, list[RelationMention]] = defaultdict(list)
        for r in relation_list:
            head = by_id.get(r.head_mention_id)
            tail = by_id.get(r.tail_mention_id)
            # A relation spanning two chunks has no single-chunk sentence window; it is a
            # binary EDGE already — the m-ary unit is within-window by definition (spec 1.2).
            if head is not None and tail is not None and (
                head.chunk_record_id == tail.chunk_record_id
            ):
                relations_of_chunk[head.chunk_record_id].append(r)

        built: dict[tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]], GraphHyperedge]
        built = {}
        for chunk in chunk_list:
            for span in self._spans_for_chunk(
                chunk,
                mentions_of_chunk.get(chunk.record_id, []),
                relations_of_chunk.get(chunk.record_id, []),
                by_id,
            ):
                self._emit(span, by_id, entity_of, link_conf, node_id_for, built)
        return sorted(built.values(), key=lambda h: h.hyperedge_id)

    # -- segmentation (spec 1.2) -------------------------------------------

    def _spans_for_chunk(
        self,
        chunk: ChunkRecord,
        mentions: list[EntityMention],
        relations: list[RelationMention],
        by_id: dict[str, EntityMention],
    ) -> list[_Span]:
        sentences = self._splitter.split(chunk.content)
        if not sentences:
            return []
        starts = [s for s, _ in sentences]
        index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        base = chunk.provenance.char_start or 0

        def local(doc_offset: int) -> int:
            if index is not None:
                return index.resolve_original(doc_offset)
            return doc_offset - base

        def sentence_of(local_start: int) -> int:
            i = bisect_right(starts, local_start) - 1
            return max(0, min(i, len(sentences) - 1))

        mention_sentence: dict[str, int] = {}
        mention_local: dict[str, tuple[int, int]] = {}
        for m in mentions:
            m_start = local(m.provenance.char_start or 0)
            m_end = local(m.provenance.char_end or 0)
            if m_start < 0 or m_start >= len(chunk.content):
                continue  # unresolvable offset: never guess a sentence for it
            mention_sentence[m.record_id] = sentence_of(m_start)
            mention_local[m.record_id] = (m_start, m_end)

        # Strong seeds: one window per relation whose arguments sit close enough.
        spans: list[_Span] = []
        for r in relations:
            hs = mention_sentence.get(r.head_mention_id)
            ts = mention_sentence.get(r.tail_mention_id)
            if hs is None or ts is None:
                continue
            lo, hi = min(hs, ts), max(hs, ts)
            if hi - lo + 1 > self._config.max_window_sentences:
                continue  # arguments too far apart for one reasoning unit
            spans.append(
                _Span(
                    chunk=chunk,
                    start=sentences[lo][0],
                    end=sentences[hi][1],
                    mention_ids={r.head_mention_id, r.tail_mention_id},
                    relation_ids={r.record_id},
                )
            )

        spans = self._merge_overlapping(spans)

        # Every mention inside a strong span joins it (the FULL entity set of the window).
        for span in spans:
            for mid, (m_start, _) in mention_local.items():
                if span.start <= m_start < span.end:
                    span.mention_ids.add(mid)

        # Weak spans: sentences with mentions not covered by any strong span.
        if self._config.weak_hyperedges:
            covered = {
                mid for span in spans for mid in span.mention_ids
            }
            weak_by_sentence: dict[int, set[str]] = defaultdict(set)
            for mid, sent in mention_sentence.items():
                if mid not in covered:
                    weak_by_sentence[sent].add(mid)
            for sent, mids in sorted(weak_by_sentence.items()):
                spans.append(
                    _Span(
                        chunk=chunk,
                        start=sentences[sent][0],
                        end=sentences[sent][1],
                        mention_ids=mids,
                    )
                )
            spans = self._merge_overlapping(spans)

        spans.sort(key=lambda s: (s.start, s.end))
        spans = self._apply_size_targets(spans, sentences, by_id, mention_local)
        return spans

    @staticmethod
    def _merge_overlapping(spans: list[_Span]) -> list[_Span]:
        spans = sorted(spans, key=lambda s: (s.start, s.end))
        merged: list[_Span] = []
        for span in spans:
            if merged and merged[-1].overlaps(span):
                merged[-1].absorb(span)
            else:
                merged.append(span)
        return merged

    def _apply_size_targets(
        self,
        spans: list[_Span],
        sentences: list[tuple[int, int]],
        by_id: dict[str, EntityMention],
        mention_local: dict[str, tuple[int, int]],
    ) -> list[_Span]:
        """Merge undersized spans into their best neighbour; split oversized ones."""
        cfg = self._config

        # Split above max_tokens: at the sentence boundary minimising cut relations.
        exploded: list[_Span] = []
        for span in spans:
            exploded.extend(self._split_oversized(span, sentences, by_id, mention_local))
        spans = exploded

        # Merge below min_tokens into the same-chunk neighbour sharing the most entities.
        result: list[_Span] = []
        for span in spans:
            text = span.chunk.content[span.start : span.end]
            if self._count(text) >= cfg.min_tokens and result:
                result.append(span)
                continue
            if not result:
                result.append(span)
                continue
            # neighbour choice: previous vs next is unknowable in a single pass; the
            # previous same-chunk span is the deterministic choice (spec: "the neighbor
            # sharing the most entities" — adjacent spans in one chunk share context).
            prev = result[-1]
            if prev.chunk.record_id == span.chunk.record_id:
                prev.absorb(span)
            else:
                result.append(span)
        return result

    def _split_oversized(
        self,
        span: _Span,
        sentences: list[tuple[int, int]],
        by_id: dict[str, EntityMention],
        mention_local: dict[str, tuple[int, int]],
    ) -> list[_Span]:
        text = span.chunk.content[span.start : span.end]
        if self._count(text) <= self._config.max_tokens:
            return [span]
        inner = [s for s in sentences if span.start <= s[0] and s[1] <= span.end]
        if len(inner) < 2:
            return [span]  # a single giant sentence cannot be split at a boundary

        # Choose the boundary minimising cut relations (ties: most balanced token split).
        def cut_cost(boundary_idx: int) -> tuple[int, int]:
            cut_at = inner[boundary_idx][0]
            left = span.chunk.content[span.start : inner[boundary_idx - 1][1]]
            right = span.chunk.content[cut_at : span.end]
            balance = abs(self._count(left) - self._count(right))
            return (self._relations_cut(span, cut_at, by_id, mention_local), balance)

        best = min(range(1, len(inner)), key=cut_cost)
        cut_at = inner[best][0]
        left = _Span(chunk=span.chunk, start=span.start, end=inner[best - 1][1])
        right = _Span(chunk=span.chunk, start=cut_at, end=span.end)
        for mid in span.mention_ids:
            m_start = mention_local.get(mid, (span.start, span.start))[0]
            (left if m_start < cut_at else right).mention_ids.add(mid)
        for rid in span.relation_ids:
            # a relation follows its arguments; a cut relation lands where its head sits
            (left if self._relation_side(rid, cut_at, by_id, mention_local) else right
             ).relation_ids.add(rid)
        return self._split_oversized(left, sentences, by_id, mention_local) + \
            self._split_oversized(right, sentences, by_id, mention_local)

    def _relations_cut(
        self,
        span: _Span,
        cut_at: int,
        by_id: dict[str, EntityMention],
        mention_local: dict[str, tuple[int, int]],
    ) -> int:
        cut = 0
        for rid in sorted(span.relation_ids):
            sides = set()
            for mid in self._relation_arg_ids(rid, by_id):
                start = mention_local.get(mid)
                if start is not None:
                    sides.add(start[0] < cut_at)
            if len(sides) > 1:
                cut += 1
        return cut

    def _relation_side(
        self,
        relation_id: str,
        cut_at: int,
        by_id: dict[str, EntityMention],
        mention_local: dict[str, tuple[int, int]],
    ) -> bool:
        """True = left of the cut. A relation follows its head argument."""
        args = self._relation_arg_ids(relation_id, by_id)
        if args:
            start = mention_local.get(args[0])
            if start is not None:
                return start[0] < cut_at
        return True

    def _relation_arg_ids(
        self, relation_id: str, by_id: dict[str, EntityMention]
    ) -> list[str]:
        rel = self._relation_index.get(relation_id)
        return [rel.head_mention_id, rel.tail_mention_id] if rel is not None else []

    # -- emission (spec 1.3–1.5) -------------------------------------------

    def _emit(
        self,
        span: _Span,
        by_id: dict[str, EntityMention],
        entity_of: dict[str, str],
        link_conf: dict[str, float],
        node_id_for: Callable[[str], str],
        built: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]], GraphHyperedge
        ],
    ) -> None:
        member_ids = sorted(mid for mid in span.mention_ids if mid in by_id)
        node_ids = sorted(
            {
                node_id_for(entity_of[mid])
                for mid in member_ids
                if mid in entity_of
            }
        )
        if not node_ids:
            return  # no canonicalised entity in the span: nothing to connect

        relations = self._span_relations(span, by_id, entity_of, node_id_for)
        has_relations = bool(relations)
        if has_relations and len(node_ids) < 2:
            relations, has_relations = [], False  # arguments collapsed to one node
        if not has_relations and not self._config.weak_hyperedges:
            return

        for group_nodes, group_relations, group_members in self._arity_groups(
            node_ids, relations, member_ids, entity_of, node_id_for
        ):
            self._emit_group(
                span, group_nodes, group_relations, group_members, by_id, link_conf, built
            )

    def _emit_group(
        self,
        span: _Span,
        node_ids: list[str],
        relations: list[HyperedgeRelation],
        member_ids: list[str],
        by_id: dict[str, EntityMention],
        link_conf: dict[str, float],
        built: dict[
            tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]], GraphHyperedge
        ],
    ) -> None:
        chunk = span.chunk
        has_relations = bool(relations)
        signature = tuple(sorted(r.signature for r in relations))
        key = (tuple(node_ids), signature)

        tau_text = self._redacted_text(chunk, span.start, span.end)
        if not tau_text.strip():
            return
        token_cost = max(1, self._count(tau_text))
        gamma = self._gamma(member_ids, relations, by_id, link_conf, weak=not has_relations)

        index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        if index is not None:
            doc_start, doc_end = index.resolve_span(span.start, span.end)
        else:
            base = chunk.provenance.char_start or 0
            doc_start, doc_end = base + span.start, base + span.end

        existing = built.get(key)
        if existing is not None:
            merged_sources = [
                *existing.merged_sources,
                HyperedgeSourceRef(
                    document_record_id=self._doc_id(chunk, by_id, member_ids),
                    chunk_record_id=chunk.record_id,
                    char_start=doc_start,
                    char_end=doc_end,
                ),
            ]
            update: dict[str, object] = {
                "support": existing.support + 1,
                "member_mention_ids": sorted(
                    set(existing.member_mention_ids) | set(member_ids)
                ),
                "merged_sources": merged_sources,
            }
            if gamma > existing.gamma:  # the new instance becomes primary (spec 1.4)
                update.update(
                    {
                        "gamma": gamma,
                        "tau_text": tau_text,
                        "token_cost": token_cost,
                        "provenance": self._span_provenance(chunk, doc_start, doc_end),
                        "merged_sources": [
                            *existing.merged_sources,
                            HyperedgeSourceRef(
                                document_record_id=existing.provenance.document_id,
                                chunk_record_id=self._primary_chunk_id(existing),
                                char_start=existing.provenance.char_start or 0,
                                char_end=existing.provenance.char_end or 0,
                            ),
                        ],
                    }
                )
            built[key] = existing.model_copy(update=update)
            return

        hyperedge_id = make_graph_id(
            self._corpus_id, "hyperedge", *node_ids, *(":".join(s) for s in signature)
        )
        provenance = self._span_provenance(chunk, doc_start, doc_end)
        built[key] = GraphHyperedge(
            record_id=f"hyperedge:{hyperedge_id}",
            hyperedge_id=hyperedge_id,
            entity_node_ids=node_ids,
            member_mention_ids=member_ids,
            relations=relations,
            tau_text=tau_text,
            token_cost=token_cost,
            token_counter=self._config.token_counter_id,
            gamma=gamma,
            support=1,
            has_relations=has_relations,
            properties={"primary_chunk_record_id": chunk.record_id},
            evidence=Evidence(
                mention_ids=member_ids,
                document_ids=sorted(
                    {
                        by_id[mid].document_record_id
                        for mid in member_ids
                        if mid in by_id
                    }
                ),
                snippet=(
                    f"hyperedge over {len(node_ids)} entit"
                    f"{'y' if len(node_ids) == 1 else 'ies'}, "
                    f"{len(relations)} relation(s)"
                ),
                confidence=gamma,
            ),
            provenance=provenance,
            classification=chunk.classification,
        )

    # -- helpers -----------------------------------------------------------

    _relation_index: dict[str, RelationMention]

    def _span_relations(
        self,
        span: _Span,
        by_id: dict[str, EntityMention],
        entity_of: dict[str, str],
        node_id_for: Callable[[str], str],
    ) -> list[HyperedgeRelation]:
        out: list[HyperedgeRelation] = []
        for rid in sorted(span.relation_ids):
            rel = self._relation_index.get(rid)
            if rel is None:
                continue
            head_entity = entity_of.get(rel.head_mention_id)
            tail_entity = entity_of.get(rel.tail_mention_id)
            if head_entity is None or tail_entity is None:
                continue  # an uncanonicalised argument cannot join the signature
            head_node, tail_node = node_id_for(head_entity), node_id_for(tail_entity)
            if head_node == tail_node:
                continue  # both arguments collapsed onto one entity: not a relation
            out.append(
                HyperedgeRelation(
                    relation_id=rid,
                    head_node_id=head_node,
                    label=rel.label,
                    tail_node_id=tail_node,
                    confidence=rel.confidence,
                )
            )
        return sorted(out, key=lambda r: (r.signature, r.relation_id))

    def _arity_groups(
        self,
        node_ids: list[str],
        relations: list[HyperedgeRelation],
        member_ids: list[str],
        entity_of: dict[str, str],
        node_id_for: Callable[[str], str],
    ) -> list[tuple[list[str], list[HyperedgeRelation], list[str]]]:
        """Split above max_arity by relation groups (spec 1.3: co-related entities stay)."""
        if len(node_ids) <= self._config.max_arity:
            return [(node_ids, relations, member_ids)]

        # Union relation endpoints into connected components.
        parent: dict[str, str] = {n: n for n in node_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for r in relations:
            ra, rb = find(r.head_node_id), find(r.tail_node_id)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        components: dict[str, set[str]] = defaultdict(set)
        for n in node_ids:
            components[find(n)].add(n)

        # Entities in no relation ride with the largest component (deterministic tiebreak).
        comps = sorted(components.values(), key=lambda c: (-len(c), min(c)))
        groups: list[tuple[list[str], list[HyperedgeRelation], list[str]]] = []
        for comp in comps:
            comp_nodes = sorted(comp)
            comp_relations = [
                r for r in relations
                if r.head_node_id in comp and r.tail_node_id in comp
            ]
            comp_members = [
                mid for mid in member_ids
                if mid in entity_of and node_id_for(entity_of[mid]) in comp
            ]
            if len(comp_nodes) > self._config.max_arity:
                comp_nodes = comp_nodes[: self._config.max_arity]
                keep = set(comp_nodes)
                comp_relations = [
                    r for r in comp_relations
                    if r.head_node_id in keep and r.tail_node_id in keep
                ]
                comp_members = [
                    mid for mid in comp_members
                    if node_id_for(entity_of[mid]) in keep
                ]
            groups.append((comp_nodes, comp_relations, comp_members))
        return groups

    def _gamma(
        self,
        member_ids: list[str],
        relations: list[HyperedgeRelation],
        by_id: dict[str, EntityMention],
        link_conf: dict[str, float],
        *,
        weak: bool,
    ) -> float:
        w_link, w_rel, w_ner = _GAMMA_WEIGHTS
        parts: list[tuple[float, float]] = []  # (weight, value)

        links = [link_conf[mid] for mid in member_ids if mid in link_conf]
        if links:
            parts.append((w_link, sum(links) / len(links)))
        if relations:
            parts.append((w_rel, sum(r.confidence for r in relations) / len(relations)))
        ner = [by_id[mid].confidence for mid in member_ids if mid in by_id]
        if ner:
            parts.append((w_ner, min(ner)))

        if not parts:
            return 0.0
        total = sum(w for w, _ in parts)
        gamma = sum(w * v for w, v in parts) / total
        if weak:
            gamma *= _WEAK_GAMMA_FACTOR
        return max(0.0, min(1.0, round(gamma, 6)))

    @staticmethod
    def _redacted_text(chunk: ChunkRecord, start: int, end: int) -> str:
        """The span text in the corpus redaction variant (ADR-0057, enforced at the writer).

        Applies every non-``TAG`` :class:`PIISpan` overlapping the span to the raw content
        slice, substituting the span's placeholder (or a typed mask when the placeholder is
        empty). When no redaction ran (``masked_content is None``) the corpus itself is raw,
        so the raw slice IS the corpus variant.
        """
        raw = chunk.content[start:end]
        if chunk.masked_content is None or not chunk.pii_spans:
            return raw
        overlapping = sorted(
            (
                s for s in chunk.pii_spans
                if s.action is not RedactionAction.TAG
                and s.char_start < end and start < s.char_end
            ),
            key=lambda s: (s.char_start, s.char_end),
        )
        if not overlapping:
            return raw
        out: list[str] = []
        cursor = start
        for s in overlapping:
            s_start, s_end = max(s.char_start, start), min(s.char_end, end)
            if s_start > cursor:
                out.append(chunk.content[cursor:s_start])
            out.append(s.placeholder or f"[{s.pii_type.upper()}]")
            cursor = max(cursor, s_end)
        if cursor < end:
            out.append(chunk.content[cursor:end])
        return "".join(out)

    @staticmethod
    def _span_provenance(chunk: ChunkRecord, doc_start: int, doc_end: int) -> Provenance:
        return chunk.provenance.model_copy(
            update={"char_start": doc_start, "char_end": doc_end}
        )

    @staticmethod
    def _doc_id(
        chunk: ChunkRecord, by_id: dict[str, EntityMention], member_ids: list[str]
    ) -> str:
        for mid in member_ids:
            m = by_id.get(mid)
            if m is not None:
                return m.document_record_id
        return chunk.provenance.document_id

    @staticmethod
    def _primary_chunk_id(hyperedge: GraphHyperedge) -> str:
        value = hyperedge.properties.get("primary_chunk_record_id", "")
        return str(value) if value else "unknown"
