"""Gold-set harness — per-Stage precision/recall/F1 on the adopter's own corpus (S10).

The gap the private stack had (ADR-0008): it filtered on confidence thresholds it
never *measured*. The gold-set harness closes it. An adopter drops in 20–50 of their
own documents with hand annotations — the entities and relations that *should* have
been extracted — and the Quality Report reports precision/recall/F1 per Stage against
those annotations, on THEIR corpus, so the "high-quality, AI-ready data" claim is
auditable rather than asserted.

This module is the pure, deterministic scoring engine:

* :class:`GoldEntity` / :class:`GoldRelation` / :class:`GoldDocument` / :class:`GoldSet`
  are the typed annotation contracts an adopter authors (JSON or YAML), keyed by source
  file name so an annotation binds to the document it describes without knowing the
  content-addressed ``document_id`` the run computes.
* :func:`score_entities` / :func:`score_relations` match the run's extracted mentions /
  relations against the gold annotations and return a :class:`ConfusionCounts` (tp/fp/fn)
  overall and per label.

Matching is **offset-aware and deterministic** — no model, no randomness, no network —
so a seeded run is byte-identical (Baseline bar). An entity matches a gold entity when
they share the same label and their character spans overlap by at least a configurable
Jaccard threshold (default: any overlap, i.e. > 0); a relation matches when its label
matches and BOTH endpoints resolve to the same gold entities (by span overlap). Each
gold annotation and each prediction is consumed at most once (a greedy, order-stable
one-to-one assignment), so precision and recall are well-defined and never double-count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from latence_core.contracts import EntityMention, RelationMention


# -- annotation contracts ----------------------------------------------------


class GoldEntity(BaseModel):
    """One annotated entity mention in a gold document (adopter-authored).

    ``char_start``/``char_end`` are the half-open span the entity occupies in the
    parsed document's assembled markdown — the same document-offset coordinate system
    every extracted :class:`~latence_core.contracts.EntityMention` uses, so a gold
    annotation and a prediction are directly comparable. ``label`` is the entity type
    the extractor is expected to assign (matching is label-sensitive). ``text`` is
    optional, carried for human readability of the annotation file only — matching is
    by span + label, never by surface string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(..., min_length=1, description="Expected entity type.")
    char_start: int = Field(..., ge=0, description="Span start in the parsed document markdown.")
    char_end: int = Field(..., ge=0, description="Span end (half-open) in the parsed document.")
    text: str = Field(default="", description="Surface string (readability only; not matched).")

    @model_validator(mode="after")
    def _ordered(self) -> GoldEntity:
        if self.char_end < self.char_start:
            msg = (
                f"GoldEntity span is inverted: char_end {self.char_end} "
                f"< char_start {self.char_start}"
            )
            raise ValueError(msg)
        return self

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


class GoldRelation(BaseModel):
    """One annotated relation between two gold entities in a gold document.

    ``label`` is the expected relation type. ``head``/``tail`` are the head (subject)
    and tail (object) :class:`GoldEntity`s the relation connects, embedded inline so the
    annotation is self-contained and a relation match can be scored by resolving both
    endpoints to the run's extracted mentions independently of entity scoring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(..., min_length=1, description="Expected relation type.")
    head: GoldEntity
    tail: GoldEntity


class GoldDocument(BaseModel):
    """The gold annotations for one source document, keyed by its file name.

    ``file_name`` binds the annotation to the source document by name (e.g.
    ``contract-01.pdf``) — the stable key an adopter knows, not the content-addressed
    ``document_id`` the run computes. The harness maps annotations onto the run's
    extracted records by resolving each document's records to their Provenance
    ``file_name`` (unique within a corpus in the reference Source).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_name: str = Field(..., min_length=1, description="Source file name this annotates.")
    entities: list[GoldEntity] = Field(default_factory=list)
    relations: list[GoldRelation] = Field(default_factory=list)


class GoldSet(BaseModel):
    """A corpus of gold-annotated documents the adopter drops in (S10 harness input).

    Loaded from a JSON or YAML file (the same one-schema/dual-serialization discipline
    as the Pipeline, ADR-0006). ``match_min_overlap`` is the Jaccard-overlap threshold a
    prediction span must reach against a gold span to count as a match (0.0 = any overlap
    counts; 1.0 = exact span required) — a single knob the adopter tunes to their
    annotation precision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1)
    documents: list[GoldDocument] = Field(default_factory=list)
    match_min_overlap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Min span Jaccard overlap for a match (0=any overlap, 1=exact).",
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> GoldSet:
        """Validate a parsed JSON/YAML mapping into a :class:`GoldSet`."""
        return cls.model_validate(data)

    def by_file_name(self) -> dict[str, GoldDocument]:
        """The gold documents keyed by ``file_name`` (later duplicate wins, order-stable)."""
        return {doc.file_name: doc for doc in self.documents}

    @property
    def total_entities(self) -> int:
        return sum(len(doc.entities) for doc in self.documents)

    @property
    def total_relations(self) -> int:
        return sum(len(doc.relations) for doc in self.documents)


# -- scoring -----------------------------------------------------------------


class ConfusionCounts(BaseModel):
    """True/false-positive/negative counts + derived precision/recall/F1 (S10).

    The atomic scoring unit, computed overall for a Stage and once per label.
    ``true_positives`` are gold annotations a prediction matched; ``false_positives``
    predictions with no matching gold (over-extraction); ``false_negatives`` gold
    annotations no prediction matched (under-extraction). Precision/recall/F1 are the
    standard derivations, each 0.0 when its denominator is empty (no predictions ⇒
    precision 0.0; no gold ⇒ recall 0.0), so the report never divides by zero.
    """

    model_config = ConfigDict(extra="forbid")

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def predicted(self) -> int:
        return self.true_positives + self.false_positives

    @property
    def actual(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        if self.predicted == 0:
            return 0.0
        return round(self.true_positives / self.predicted, 6)

    @property
    def recall(self) -> float:
        if self.actual == 0:
            return 0.0
        return round(self.true_positives / self.actual, 6)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        if p + r == 0.0:
            return 0.0
        return round(2 * p * r / (p + r), 6)

    def add(self, other: ConfusionCounts) -> None:
        """Fold another count in (used to accumulate per-document into the corpus total)."""
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives


def _span_overlap_jaccard(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> float:
    """Jaccard overlap of two half-open char spans (0.0 when disjoint or degenerate).

    ``|intersection| / |union|`` over character positions. Two zero-length spans at the
    same offset are treated as a full (1.0) overlap so a point annotation still matches
    a point prediction; a zero-length span against a positive-length one has empty
    intersection ⇒ 0.0.
    """
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    intersection = max(inter_end - inter_start, 0)
    union = max(a_end, b_end) - min(a_start, b_start)
    if union == 0:
        # Both spans are zero-length: they match iff they sit at the same offset.
        return 1.0 if a_start == b_start else 0.0
    return intersection / union


def _greedy_match(
    gold: list[tuple[int, str, int, int]],
    pred: list[tuple[int, str, int, int]],
    min_overlap: float,
) -> ConfusionCounts:
    """Greedy one-to-one label+span match between gold and predicted spans.

    Each tuple is ``(id, label, char_start, char_end)``. A gold and a prediction match
    when their labels are equal AND their span Jaccard overlap is ``> min_overlap`` when
    ``min_overlap == 0`` (any overlap) or ``>= min_overlap`` otherwise. The assignment
    is greedy in a deterministic order: predictions are tried in input order, each taking
    the first still-unmatched gold with the **highest** overlap (ties broken by gold input
    order), so the result is stable and one-to-one — no gold or prediction is counted
    twice. Leftover predictions are false positives; leftover gold are false negatives.
    """
    counts = ConfusionCounts()
    matched_gold: set[int] = set()
    for _pid, p_label, p_start, p_end in pred:
        best_idx: int | None = None
        best_overlap = 0.0
        for gi, (_gid, g_label, g_start, g_end) in enumerate(gold):
            if gi in matched_gold or g_label != p_label:
                continue
            overlap = _span_overlap_jaccard(p_start, p_end, g_start, g_end)
            qualifies = overlap > 0.0 if min_overlap == 0.0 else overlap >= min_overlap
            if qualifies and overlap > best_overlap:
                best_overlap = overlap
                best_idx = gi
        if best_idx is None:
            counts.false_positives += 1
        else:
            matched_gold.add(best_idx)
            counts.true_positives += 1
    counts.false_negatives = len(gold) - len(matched_gold)
    return counts


def score_entities(
    mentions: Iterable[EntityMention],
    gold: GoldSet,
    file_name_of: Mapping[str, str],
) -> tuple[ConfusionCounts, dict[str, ConfusionCounts]]:
    """Score extracted entity mentions against the gold set (overall + per label).

    ``mentions`` are the run's extracted :class:`~latence_core.contracts.EntityMention`s;
    ``file_name_of`` maps a mention's ``document_record_id`` to its source file name (the
    key the gold annotations are authored against). Matching is per document: within each
    gold document the mentions of that document are matched against its gold entities by
    label + span overlap. Returns the corpus-total :class:`ConfusionCounts` and a
    per-label breakdown, both accumulated deterministically (documents in gold order,
    labels sorted).
    """
    gold_by_file = gold.by_file_name()
    # Group mentions by their source file name (only files present in the gold set matter
    # for scoring; a mention in an un-annotated document is neither a TP/FP/FN).
    mentions_by_file: dict[str, list[EntityMention]] = {}
    for m in mentions:
        file_name = file_name_of.get(m.document_record_id)
        if file_name is None or file_name not in gold_by_file:
            continue
        mentions_by_file.setdefault(file_name, []).append(m)

    overall = ConfusionCounts()
    per_label: dict[str, ConfusionCounts] = {}
    labels_seen: set[str] = set()
    for file_name, doc in gold_by_file.items():
        doc_mentions = mentions_by_file.get(file_name, [])
        gold_spans = [
            (i, g.label, g.char_start, g.char_end) for i, g in enumerate(doc.entities)
        ]
        pred_spans = [
            (
                i,
                m.label,
                m.provenance.char_start if m.provenance.char_start is not None else -1,
                m.provenance.char_end if m.provenance.char_end is not None else -1,
            )
            for i, m in enumerate(doc_mentions)
        ]
        labels_seen.update(g.label for g in doc.entities)
        labels_seen.update(m.label for m in doc_mentions)
        # Overall for the document.
        overall.add(_greedy_match(gold_spans, pred_spans, gold.match_min_overlap))
        # Per-label: restrict both sides to one label at a time.
        for label in sorted({g.label for g in doc.entities} | {m.label for m in doc_mentions}):
            g_l = [t for t in gold_spans if t[1] == label]
            p_l = [t for t in pred_spans if t[1] == label]
            per_label.setdefault(label, ConfusionCounts()).add(
                _greedy_match(g_l, p_l, gold.match_min_overlap)
            )
    return overall, dict(sorted(per_label.items()))


def score_relations(
    relations: Iterable[RelationMention],
    mentions: Iterable[EntityMention],
    gold: GoldSet,
    file_name_of: Mapping[str, str],
) -> tuple[ConfusionCounts, dict[str, ConfusionCounts]]:
    """Score extracted relations against the gold set (overall + per label).

    A predicted :class:`~latence_core.contracts.RelationMention` matches a gold relation
    when their labels are equal AND both endpoints align: the relation's head mention
    resolves (by span overlap + label) to the gold relation's head entity, and its tail to
    the gold tail. Endpoint resolution reuses the same span-overlap rule as entity scoring,
    so a relation is credited only when the entities it connects are the annotated ones —
    not merely a label match. ``mentions`` supplies the endpoint spans (a relation carries
    only the head/tail mention *ids*), looked up per document. Returns the corpus-total
    counts + a per-label breakdown, deterministically accumulated.
    """
    gold_by_file = gold.by_file_name()
    mention_by_id: dict[str, EntityMention] = {m.record_id: m for m in mentions}
    relations_by_file: dict[str, list[RelationMention]] = {}
    for r in relations:
        file_name = file_name_of.get(r.document_record_id)
        if file_name is None or file_name not in gold_by_file:
            continue
        relations_by_file.setdefault(file_name, []).append(r)

    overall = ConfusionCounts()
    per_label: dict[str, ConfusionCounts] = {}
    min_overlap = gold.match_min_overlap
    for file_name, doc in gold_by_file.items():
        doc_relations = relations_by_file.get(file_name, [])
        gold_rels = list(enumerate(doc.relations))
        pred_rels = list(enumerate(doc_relations))
        overall.add(_match_relations(gold_rels, pred_rels, mention_by_id, min_overlap))
        for label in sorted(
            {g.label for g in doc.relations} | {r.label for r in doc_relations}
        ):
            g_l = [(i, g) for i, g in gold_rels if g.label == label]
            p_l = [(i, r) for i, r in pred_rels if r.label == label]
            per_label.setdefault(label, ConfusionCounts()).add(
                _match_relations(g_l, p_l, mention_by_id, min_overlap)
            )
    return overall, dict(sorted(per_label.items()))


def _endpoint_matches(
    gold_entity: GoldEntity,
    mention: EntityMention | None,
    min_overlap: float,
) -> bool:
    """True if an extracted mention aligns with a gold endpoint (label + span overlap)."""
    if mention is None:
        return False
    if mention.label != gold_entity.label:
        return False
    m_start = mention.provenance.char_start
    m_end = mention.provenance.char_end
    if m_start is None or m_end is None:
        return False
    overlap = _span_overlap_jaccard(m_start, m_end, gold_entity.char_start, gold_entity.char_end)
    return overlap > 0.0 if min_overlap == 0.0 else overlap >= min_overlap


def _match_relations(
    gold: list[tuple[int, GoldRelation]],
    pred: list[tuple[int, RelationMention]],
    mention_by_id: Mapping[str, EntityMention],
    min_overlap: float,
) -> ConfusionCounts:
    """Greedy one-to-one match of predicted relations to gold relations (both endpoints).

    A prediction matches a gold relation when labels are equal and BOTH the head and tail
    extracted mentions align with the gold head and tail entities (span overlap + label).
    Greedy, deterministic (predictions in input order, first unmatched qualifying gold),
    one-to-one; leftovers are FP (predictions) / FN (gold).
    """
    counts = ConfusionCounts()
    matched_gold: set[int] = set()
    for _pi, rel in pred:
        head = mention_by_id.get(rel.head_mention_id)
        tail = mention_by_id.get(rel.tail_mention_id)
        chosen: int | None = None
        for gi, gold_rel in gold:
            if gi in matched_gold or gold_rel.label != rel.label:
                continue
            if _endpoint_matches(gold_rel.head, head, min_overlap) and _endpoint_matches(
                gold_rel.tail, tail, min_overlap
            ):
                chosen = gi
                break
        if chosen is None:
            counts.false_positives += 1
        else:
            matched_gold.add(chosen)
            counts.true_positives += 1
    counts.false_negatives = len(gold) - len(matched_gold)
    return counts
