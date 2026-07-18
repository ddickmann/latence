"""The Entity-Extraction, Relation-Extraction and Disambiguation suites, on RED^FM.

Three Stages, one pinned corpus (:mod:`latence_benchmark.redfm`), one shared discipline: the
real Provider is resolved from the real registry, fed real contracts, and scored against the
corpus's own gold annotations with :mod:`latence_benchmark.metrics`.

What each suite isolates — this is the part a benchmark usually gets wrong:

* **Entity Extraction** scores the extractor's spans against the gold spans. It reports two
  match criteria side by side: ``span`` (exact boundaries, type ignored — did it find the
  entity at all?) and ``span+type`` (boundaries *and* label). Reporting only the first would
  flatter a system that finds text but cannot type it; reporting only the second would hide a
  detector that works under a label set the corpus does not use.
* **Relation Extraction** is scored on **gold mentions**. The Stage is handed the corpus's own
  entity mentions, so its number measures relation extraction and not the entity extractor
  upstream of it — otherwise a weak NER silently caps the RE score and neither number means
  anything.
* **Disambiguation** is scored **pairwise over KB-linked mentions**: two mentions are gold-
  coreferent iff they carry the same Wikidata URI. Mentions the corpus could not link
  (``uri == "None"``) are excluded from both sides — they carry no identity, and treating them
  as one giant cluster would invent thousands of gold pairs that no annotator asserted.

The pinned Providers are the framework's dependency-free CPU built-ins, for the same reason the
Parse slice pins ``parser.pdfplumber``: the published number must be the number an adopter
reproduces from a clean clone, not one from a GPU stack they do not have. Each suite's note says
plainly what its pinned Provider structurally cannot do, and names the learned re-pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.capability import Disambiguator, EntityExtractor, RelationExtractor
from latence_core.contracts import DisambiguationScope, EntityMention, Provenance
from latence_core.pipeline import CapabilityKind

from latence_benchmark.chunking import as_chunk
from latence_benchmark.metrics import MetricScore, confusion, micro
from latence_benchmark.providers import load_provider
from latence_benchmark.redfm import (
    REDFM,
    REDFM_LANGUAGES,
    RedfmDocument,
    RedfmEntity,
    load_language,
)
from latence_benchmark.suite import StackPin, SuiteResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from latence_core.contracts import ChunkRecord

# -- pinned stack configs ----------------------------------------------------------------------

#: A language-agnostic proper-noun recogniser plus a numeric recogniser, expressed with **bounded
#: quantifiers only** — core's config-regex ReDoS guard (issue #28) rejects nested/adjacent
#: unbounded quantifiers, and rightly so.
# The token class deliberately excludes ``.``: including it swallows the sentence-final period
# into the last token ("Ada Lovelace." instead of "Ada Lovelace"), which is a boundary error on
# every sentence-final entity — a measurable, and avoidable, self-inflicted recall loss.
_PROPER_NOUN = r"[A-ZÀ-ÖØ-Þ][\w'’-]{0,29}(?: [A-ZÀ-ÖØ-Þ][\w'’-]{0,29}){0,4}"
_NUMBER = r"\d[\d.,]{0,14}"

#: The pinned Entity-Extraction stack. ``entity.gazetteer`` is the zero-dep CPU default, so this
#: is a *detection* baseline: capitalisation finds proper nouns in Latin-script languages and
#: nothing at all in Arabic or Chinese, which have no case distinction — the per-language slices
#: show exactly that, which is the point of publishing them. ``NUMBER`` is a genuinely typed
#: recogniser, so the ``span+type`` criterion is measuring something rather than being zero by
#: construction. The learned re-pin is ``--provider ner.gliner``.
ENTITY_STACK_PIN = StackPin(
    stage="entity_extraction",
    provider="entity.gazetteer",
    config={
        "labels": {
            "ENTITY": {"patterns": [_PROPER_NOUN], "confidence": 0.5},
            "NUMBER": {"patterns": [_NUMBER], "confidence": 0.5},
        },
        # Capitalisation IS the signal here, and this flag also governs the pattern flags:
        # case-insensitive matching would make the proper-noun pattern match every word.
        "case_sensitive": True,
    },
)

#: The pinned Relation-Extraction stack. ``relation.pattern`` with **type-prior co-occurrence**
#: rules: five predicates whose head/tail entity-type signature is distinctive enough to be worth
#: a rule. The signatures and the window were derived from RED^FM's **train** split (en+de) and
#: never from the scored ``test`` split — a rule tuned on the test set would publish a number
#: about itself. ``window_chars=250`` is ≈ the 90th percentile of the train split's head-end →
#: tail-start gap. The learned re-pins are ``--provider relation.gliner`` / ``relation.llm``.
RELATION_STACK_PIN = StackPin(
    stage="relation_extraction",
    provider="relation.pattern",
    config={
        "relations": {
            "location": {"head": "LOC", "tail": "LOC"},
            "country": {"head": "ORG", "tail": "LOC"},
            "inception": {"head": "ORG", "tail": "DATE"},
            "owned by": {"head": "ORG", "tail": "ORG"},
            "manufacturer": {"head": "MISC", "tail": "ORG"},
        },
        "window_chars": 250,
        "default_confidence": 0.5,
    },
)

#: The pinned Disambiguation stack: the zero-dep 5-strategy cascade at its defaults. It resolves
#: entities by *surface* — it has no Wikidata index — so it is measured against gold KB identity
#: with that stated up front. The KB-linking re-pin is ``--provider disambiguation.glinker``.
DISAMBIGUATION_STACK_PIN = StackPin(
    stage="disambiguation",
    provider="disambiguation.cascade",
    config={},
)


# -- shared plumbing ---------------------------------------------------------------------------


def _gold_mentions(
    document: RedfmDocument, chunk: ChunkRecord
) -> list[tuple[EntityMention, RedfmEntity]]:
    """Turn a document's gold annotations into real :class:`EntityMention` contracts.

    Used to feed Relation Extraction and Disambiguation, so their numbers are not hostage to the
    entity extractor. Each mention is paired with the annotation it came from, so a caller can
    read the gold URI without re-deriving the correspondence. Zero-length annotations are
    dropped (see :attr:`RedfmEntity.mentionable`) — they are not mentions.
    """
    pairs: list[tuple[EntityMention, RedfmEntity]] = []
    for index, entity in enumerate(document.entities):
        if not entity.mentionable:
            continue
        mention = EntityMention(
            record_id=f"{document.docid}#gold:{index}",
            document_record_id=document.docid,
            chunk_record_id=chunk.record_id,
            index=index,
            label=entity.entity_type,
            text=entity.surface,
            confidence=1.0,
            provenance=chunk.provenance.model_copy(
                update={"char_start": entity.start, "char_end": entity.end}
            ),
            classification=chunk.classification,
        )
        pairs.append((mention, entity))
    return pairs


def _excluded_note(count: int) -> list[str]:
    if not count:
        return []
    return [
        f"{count} zero-length gold annotation(s) were excluded from both sides — a zero-length "
        "span is not a mention any extractor could emit."
    ]


def _documents(root: Path, language: str, limit: int | None) -> tuple[RedfmDocument, ...]:
    documents = load_language(root, language)
    return documents[:limit] if limit is not None else documents


def _span(provenance: Provenance) -> tuple[int, int]:
    """The half-open char span a Stage stamped on a record, as a comparable key."""
    return (provenance.char_start or 0, provenance.char_end or 0)


def _subset_note(limit: int | None) -> list[str]:
    if limit is None:
        return []
    return [
        f"SUBSET RUN — capped at {limit} document(s) per language slice. These numbers are NOT "
        "comparable with full-benchmark results."
    ]


def _result(
    *,
    suite: str,
    stage: str,
    pin: StackPin,
    metrics: Sequence[MetricScore],
    documents_scored: int,
    notes: Sequence[str],
) -> SuiteResult:
    return SuiteResult(
        suite=suite,
        stage=stage,
        dataset=REDFM,
        stack_pin=pin,
        splits=(),
        documents_scored=documents_scored,
        metrics=tuple(metrics),
        headline_dimension="language",
        notes=tuple(notes),
    )


# -- the suites --------------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySuite:
    """Benchmarks the Entity Extraction Stage on RED^FM's multilingual gold spans."""

    name: str = "entity_extraction"
    stage: str = "entity_extraction"
    dataset: Any = REDFM
    stack_pin: StackPin = ENTITY_STACK_PIN
    languages: tuple[str, ...] = REDFM_LANGUAGES

    def run(self, root: Path, *, limit: int | None = None) -> SuiteResult:
        extractor: EntityExtractor = load_provider(
            self.stack_pin,
            capability=CapabilityKind.ENTITY_EXTRACTION,
            protocol=EntityExtractor,
        )
        per_language: list[MetricScore] = []
        typed_slices: list[MetricScore] = []
        documents_scored = 0
        excluded = 0

        for language in self.languages:
            documents = _documents(root, language, limit)
            documents_scored += len(documents)
            gold_spans: list[tuple[str, int, int]] = []
            gold_typed: list[tuple[str, int, int, str]] = []
            chunks: list[ChunkRecord] = []
            for document in documents:
                chunks.append(as_chunk(document.docid, document.text, language=language))
                for entity in document.entities:
                    if not entity.mentionable:
                        excluded += 1
                        continue
                    gold_spans.append((document.docid, entity.start, entity.end))
                    gold_typed.append(
                        (document.docid, entity.start, entity.end, entity.entity_type)
                    )

            predicted_spans: list[tuple[str, int, int]] = []
            predicted_typed: list[tuple[str, int, int, str]] = []
            for mention in extractor.extract(chunks):
                start, end = _span(mention.provenance)
                predicted_spans.append((mention.document_record_id, start, end))
                predicted_typed.append(
                    (mention.document_record_id, start, end, mention.label)
                )

            per_language.append(
                confusion("language", language, gold=gold_spans, predicted=predicted_spans)
            )
            typed_slices.append(
                confusion("match", language, gold=gold_typed, predicted=predicted_typed)
            )

        metrics = [
            *per_language,
            micro(per_language, dimension="match", name="span (boundaries only)"),
            micro(typed_slices, dimension="match", name="span+type"),
        ]
        notes = [
            "Scored with Provider 'entity.gazetteer' exactly as pinned: a dependency-free "
            "capitalisation + numeric recogniser, i.e. an entity DETECTION baseline. It has no "
            "notion of RED^FM's label inventory beyond NUMBER, so `span+type` is far below "
            "`span`; and Arabic and Chinese have no case distinction, so their per-language "
            "slices are near-zero BY CONSTRUCTION, not by defect. The learned re-pin is "
            "`--provider ner.gliner` (installs latence-ner-gliner).",
            "Per-language slices use the `span` (boundaries-only) criterion; the `match` "
            "dimension micro-averages both criteria over all languages.",
            *_excluded_note(excluded),
            *_subset_note(limit),
        ]
        return _result(
            suite=self.name,
            stage=self.stage,
            pin=self.stack_pin,
            metrics=metrics,
            documents_scored=documents_scored,
            notes=notes,
        )


@dataclass(frozen=True)
class RelationSuite:
    """Benchmarks the Relation Extraction Stage on RED^FM, from GOLD entity mentions."""

    name: str = "relation_extraction"
    stage: str = "relation_extraction"
    dataset: Any = REDFM
    stack_pin: StackPin = RELATION_STACK_PIN
    languages: tuple[str, ...] = REDFM_LANGUAGES

    def run(self, root: Path, *, limit: int | None = None) -> SuiteResult:
        extractor: RelationExtractor = load_provider(
            self.stack_pin,
            capability=CapabilityKind.RELATION_EXTRACTION,
            protocol=RelationExtractor,
        )
        scored_labels = frozenset(self.stack_pin.config.get("relations", {}))
        per_language: list[MetricScore] = []
        per_label: dict[str, list[MetricScore]] = {label: [] for label in sorted(scored_labels)}
        documents_scored = 0

        for language in self.languages:
            documents = _documents(root, language, limit)
            documents_scored += len(documents)
            chunks: list[ChunkRecord] = []
            mentions: list[EntityMention] = []
            gold: list[tuple[str, tuple[int, int], str, tuple[int, int]]] = []
            for document in documents:
                chunk = as_chunk(document.docid, document.text, language=language)
                chunks.append(chunk)
                mentions.extend(mention for mention, _ in _gold_mentions(document, chunk))
                for relation in document.relations:
                    if relation.predicate not in scored_labels:
                        # A predicate the pin declares no rule for is out of scope for BOTH
                        # sides — counting it as a false negative would charge the Stage for a
                        # label it was never asked to find, which is not a measurement.
                        continue
                    gold.append(
                        (
                            document.docid,
                            (relation.head.start, relation.head.end),
                            relation.predicate,
                            (relation.tail.start, relation.tail.end),
                        )
                    )

            by_id = {mention.record_id: mention for mention in mentions}
            predicted: list[tuple[str, tuple[int, int], str, tuple[int, int]]] = []
            for relation_mention in extractor.relate(chunks, mentions):
                head = by_id.get(relation_mention.head_mention_id)
                tail = by_id.get(relation_mention.tail_mention_id)
                if head is None or tail is None:  # pragma: no cover - ids come from `mentions`
                    continue
                predicted.append(
                    (
                        relation_mention.document_record_id,
                        _span(head.provenance),
                        relation_mention.label,
                        _span(tail.provenance),
                    )
                )

            per_language.append(confusion("language", language, gold=gold, predicted=predicted))
            for label in per_label:
                per_label[label].append(
                    confusion(
                        "predicate",
                        label,
                        gold=[g for g in gold if g[2] == label],
                        predicted=[p for p in predicted if p[2] == label],
                    )
                )

        metrics = [
            *per_language,
            *(
                micro(slices, dimension="predicate", name=label)
                for label, slices in sorted(per_label.items())
            ),
        ]
        notes = [
            "Scored with Provider 'relation.pattern' exactly as pinned, from GOLD entity "
            "mentions: the Stage is handed the corpus's own mentions, so this number measures "
            "relation extraction and not the entity extractor upstream of it.",
            "The pin declares five type-prior co-occurrence rules (see RELATION_STACK_PIN), "
            "derived from the TRAIN split. Scoring is restricted to those five predicates on "
            "BOTH sides — gold triples with other predicates are out of scope, not false "
            "negatives. A co-occurrence rule fires on every in-window head/tail type pair, so "
            "recall is high and precision is the honest weakness of a rule baseline. The "
            "learned re-pins are `--provider relation.gliner` / `relation.llm`.",
            f"Scored predicates: {', '.join(sorted(scored_labels))}.",
            *_subset_note(limit),
        ]
        return _result(
            suite=self.name,
            stage=self.stage,
            pin=self.stack_pin,
            metrics=metrics,
            documents_scored=documents_scored,
            notes=notes,
        )


@dataclass(frozen=True)
class DisambiguationSuite:
    """Benchmarks the Disambiguation Stage against RED^FM's gold Wikidata identities."""

    name: str = "disambiguation"
    stage: str = "disambiguation"
    dataset: Any = REDFM
    stack_pin: StackPin = DISAMBIGUATION_STACK_PIN
    languages: tuple[str, ...] = REDFM_LANGUAGES

    def run(self, root: Path, *, limit: int | None = None) -> SuiteResult:
        disambiguator: Disambiguator = load_provider(
            self.stack_pin,
            capability=CapabilityKind.DISAMBIGUATION,
            protocol=Disambiguator,
        )
        per_language: list[MetricScore] = []
        documents_scored = 0

        for language in self.languages:
            documents = _documents(root, language, limit)
            documents_scored += len(documents)
            chunks: list[ChunkRecord] = []
            mentions: list[EntityMention] = []
            uri_of: dict[str, str] = {}
            for document in documents:
                chunk = as_chunk(document.docid, document.text, language=language)
                chunks.append(chunk)
                for mention, entity in _gold_mentions(document, chunk):
                    mentions.append(mention)
                    if entity.linked:
                        uri_of[mention.record_id] = entity.uri

            records = disambiguator.disambiguate(mentions, (), chunks)
            predicted_clusters = [
                tuple(record.entity.member_mention_ids)
                for record in records
                if record.scope is DisambiguationScope.ENTITY and record.entity is not None
            ]
            per_language.append(
                _pairwise(
                    language,
                    gold_uri=uri_of,
                    predicted_clusters=predicted_clusters,
                )
            )

        notes = [
            "Scored with Provider 'disambiguation.cascade' exactly as pinned, over GOLD entity "
            "mentions, PAIRWISE: two mentions count as coreferent iff they share a gold Wikidata "
            "URI, and precision/recall are computed over mention PAIRS. Mentions the corpus left "
            "unlinked (uri == 'None') are excluded from both sides — an unlinked mention asserts "
            "no identity, and clustering them together would invent gold pairs.",
            "The pinned cascade resolves entities by SURFACE (no Wikidata index), so it cannot "
            "link two different surfaces of the same entity across languages, and it will merge "
            "two same-surfaced different entities. That is the measured trade-off, not a bug; "
            "the KB-linking re-pin is `--provider disambiguation.glinker`.",
            "Each language slice is disambiguated as its own corpus, so a cross-language merge "
            "is neither credited nor charged.",
            *_subset_note(limit),
        ]
        return _result(
            suite=self.name,
            stage=self.stage,
            pin=self.stack_pin,
            metrics=per_language,
            documents_scored=documents_scored,
            notes=notes,
        )


def _pairs(groups: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Every unordered within-group mention pair, as sorted 2-tuples."""
    out: set[tuple[str, str]] = set()
    for members in groups.values():
        ordered = sorted(members)
        for i, left in enumerate(ordered):
            for right in ordered[i + 1 :]:
                out.add((left, right))
    return out


def _pairwise(
    language: str,
    *,
    gold_uri: dict[str, str],
    predicted_clusters: Sequence[Sequence[str]],
) -> MetricScore:
    """Pairwise precision/recall of predicted clusters against gold KB identity."""
    gold_groups: dict[str, list[str]] = {}
    for mention_id, uri in gold_uri.items():
        gold_groups.setdefault(uri, []).append(mention_id)
    gold_pairs = _pairs(gold_groups)

    predicted_groups: dict[str, list[str]] = {}
    for index, members in enumerate(predicted_clusters):
        # Only KB-linked mentions carry a gold identity; the rest are outside the scored
        # universe entirely, on both sides.
        scored = [member for member in members if member in gold_uri]
        if len(scored) > 1:
            predicted_groups[str(index)] = scored
    predicted_pairs = _pairs(predicted_groups)

    true_positives = len(gold_pairs & predicted_pairs)
    return MetricScore(
        dimension="language",
        name=language,
        true_positives=true_positives,
        false_positives=len(predicted_pairs) - true_positives,
        false_negatives=len(gold_pairs) - true_positives,
    )
