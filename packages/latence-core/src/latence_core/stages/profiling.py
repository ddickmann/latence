"""Built-in reference Profiler Provider (deterministic, streaming, zero-dep).

The S7 Profiling reference Provider (ADR-0004; ships in core like the S3
Chunker/Screeners, the S4 gazetteer, the S5 trigger extractor and the S6 hybrid
redactor so the demo runs end to end, ADR-0007 CPU-first, ADR-0016 no extra deps).
It wraps the ported/consolidated enricher engine (:mod:`latence_core.profiling`) and
turns the run's corpus into typed :class:`~latence_core.contracts.FeatureRecord`s:

* one ``DOCUMENT``-scope record per parsed :class:`~latence_core.contracts.DocumentRecord`,
  carrying that document's :class:`~latence_core.contracts.DocumentFeatures` (density,
  readability, Zipf α, compression ratio, structure), its Provenance inherited and its
  Classification unchanged;
* exactly one ``CORPUS``-scope record carrying the cross-document
  :class:`~latence_core.contracts.CorpusFeatures` (entity frequency, co-occurrence, type
  consensus, source coverage), stamped with a synthetic corpus-scope Provenance.

The pass is **streaming** (S7 AC): documents are folded into a
:class:`~latence_core.profiling.CorpusAggregator` one at a time via a generator, so the
corpus text is never held whole in RAM — only the bounded aggregate state is, and that
state is explicitly capped (per-document co-occurrence fan-out + accumulated pair table)
so a large NER Provider cannot drive it to unbounded growth. Nothing spills to disk; the
frequency counters grow with the corpus vocabulary, which is inherent to the feature set.
The entity mentions are grouped by document once up front (the mention stream is small
relative to the corpus text and must be indexed by document to fold per-document).

Determinism: document-scope records are emitted in first-seen document order and the
single corpus-scope record last; every feature map is sorted by the engine — so a
seeded run is byte-identical (Baseline bar). A ``PARSE_ERROR`` document (no usable
content) is skipped, mirroring the Chunk/Redaction Stages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from latence_core.contracts import (
    Classification,
    CorpusFeatures,
    Disposition,
    DocumentFeatures,
    EntityCoOccurrence,
    FeatureRecord,
    FeatureScope,
    Provenance,
)
from latence_core.profiling import (
    _MAX_ENTITIES_PER_DOCUMENT,
    _MAX_TRACKED_PAIRS,
    CorpusAggregator,
    DocFeatureValues,
    MentionView,
    document_features,
)
from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord, EntityMention

# The synthetic corpus-scope Provenance identity. The CORPUS FeatureRecord's source
# is the whole run, not a single file, so its Provenance is a fixed sentinel (never a
# real file) — satisfying the Record contract (Provenance is required) while making
# the corpus scope unambiguous to a consumer. The document_id is a stable literal so
# a seeded run is byte-identical.
_CORPUS_SOURCE_URI = "latence:corpus"
_CORPUS_DOCUMENT_ID = "latence:corpus"
_UNKNOWN_LANGUAGE = "und"


class StatisticalProfiler:
    """Profiler Provider: corpus → per-document + corpus FeatureRecords (streaming).

    Config:
        max_co_occurrences: int  — cap on the number of top entity co-occurrence pairs
            in the corpus feature record, so the report stays bounded on a large corpus
            (default 50).
        max_entities_per_document: int  — cap on the distinct entities from one document
            that enter the co-occurrence pairing, so a large NER Provider cannot make one
            document build an O(distinct-entities²) pair explosion (default 200).
        max_tracked_pairs: int  — cap on the accumulated distinct co-occurring pairs held
            across the corpus, so the pair state cannot grow without bound (default
            100_000).
        corpus_sensitivity: str  — the ``Classification.sensitivity`` stamped on the
            CORPUS-scope record (the aggregate is not tied to one document's
            sensitivity). Default ``"internal"`` (a corpus profile is derived metadata,
            not public by default).
        corpus_category: str | None  — optional category stamped on the CORPUS record
            (default ``"corpus_profile"``).
    """

    # The declared descriptor (ADR-0036 §2): pure-code, CPU, deterministic, Apache-2.0. The
    # statistical Profiler folds per-document + cross-document features streaming — a pure-Python
    # part of ``latence-core`` (Apache-2.0, verified per ADR-0012), no model weights
    # (``model_id=None``), deterministic, CPU-only, non-batching, no per-call cost. Declared as a
    # ``PROFILE`` ClassVar (not a ``profile`` classmethod) BECAUSE the Profiler Capability's own
    # I/O method is named ``profile(self, documents, mentions)`` — a ``profile`` classmethod would
    # be shadowed by that instance method, so :func:`~latence_core.providers.profile.profile_of`
    # reads the ClassVar form here (the collision the P3-F2 hardening documented).
    PROFILE: ClassVar[ProviderProfile] = ProviderProfile(
        compute="cpu",
        memory_mb=32,
        model_id=None,
        license="Apache-2.0",
        license_verified=True,
        license_code="Apache-2.0",
        license_source=REPO_LICENSE_SOURCE,
        license_verified_on="2026-07-06",
        deterministic=True,
        batch=False,
        cost_per_1k=None,
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self._max_co_occurrences = int(config.get("max_co_occurrences", 50))
        self._max_entities_per_document = int(
            config.get("max_entities_per_document", _MAX_ENTITIES_PER_DOCUMENT)
        )
        self._max_tracked_pairs = int(config.get("max_tracked_pairs", _MAX_TRACKED_PAIRS))
        self._corpus_sensitivity = str(config.get("corpus_sensitivity", "internal"))
        category = config.get("corpus_category", "corpus_profile")
        self._corpus_category = None if category is None else str(category)

    def profile(
        self, documents: Iterable[DocumentRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[FeatureRecord]:
        # Index mentions by their parent document once. The mention stream is small
        # relative to the corpus text; grouping it lets each document fold its own
        # mentions without re-scanning. (This is the only non-streaming structure, and
        # it is bounded by the mention count, not the corpus text size.)
        mentions_by_doc: dict[str, list[MentionView]] = {}
        for mention in mentions:
            mentions_by_doc.setdefault(mention.document_record_id, []).append(
                MentionView(
                    document_id=mention.document_record_id,
                    text=mention.text,
                    label=mention.label,
                )
            )

        aggregator = CorpusAggregator(
            max_co_occurrences=self._max_co_occurrences,
            max_entities_per_document=self._max_entities_per_document,
            max_tracked_pairs=self._max_tracked_pairs,
        )
        seen_language = _UNKNOWN_LANGUAGE
        for document in documents:
            # A document that failed to parse has no usable content — skip it (mirrors
            # Chunk/Redaction). It contributes to neither per-document nor corpus stats.
            if document.disposition is Disposition.PARSE_ERROR or not document.content:
                continue
            doc_values = document_features(document.content)
            classification = document.classification
            seen_language = classification.language
            aggregator.add_document(
                word_frequencies=doc_values.word_frequencies,
                language=classification.language,
                category=classification.category,
                sensitivity=classification.sensitivity,
            )
            doc_mentions = mentions_by_doc.get(document.record_id, [])
            if doc_mentions:
                aggregator.add_mentions(doc_mentions)
            yield self._document_record(document, doc_values)

        # Exactly one corpus-scope record, emitted last, over the folded aggregate.
        yield self._corpus_record(aggregator, seen_language)

    def _document_record(
        self, document: DocumentRecord, values: DocFeatureValues
    ) -> FeatureRecord:
        features = DocumentFeatures(
            char_count=values.char_count,
            word_count=values.word_count,
            sentence_count=values.sentence_count,
            unique_word_count=values.unique_word_count,
            type_token_ratio=values.type_token_ratio,
            density=values.density,
            readability=values.readability,
            zipf_alpha=values.zipf_alpha,
            compression_ratio=values.compression_ratio,
            avg_word_length=values.avg_word_length,
            avg_sentence_length=values.avg_sentence_length,
        )
        # The per-document FeatureRecord's Provenance covers the whole document [0, len);
        # source identity is inherited from the DocumentRecord unchanged.
        provenance = document.provenance.model_copy(
            update={
                "char_start": 0,
                "char_end": len(document.content),
                "page_start": None,
                "page_end": None,
            }
        )
        return FeatureRecord(
            record_id=f"{document.record_id}#features",
            scope=FeatureScope.DOCUMENT,
            document_record_id=document.record_id,
            document=features,
            provenance=provenance,
            classification=document.classification,
        )

    def _corpus_record(
        self, aggregator: CorpusAggregator, language: str
    ) -> FeatureRecord:
        values = aggregator.finalize()
        features = CorpusFeatures(
            document_count=values.document_count,
            total_words=values.total_words,
            total_unique_words=values.total_unique_words,
            corpus_zipf_alpha=values.corpus_zipf_alpha,
            language_coverage=values.language_coverage,
            category_coverage=values.category_coverage,
            sensitivity_coverage=values.sensitivity_coverage,
            entity_frequency=values.entity_frequency,
            entity_type_frequency=values.entity_type_frequency,
            entity_type_consensus=values.entity_type_consensus,
            co_occurrences=[
                EntityCoOccurrence(left=left, right=right, document_count=count)
                for left, right, count in values.co_occurrences
            ],
        )
        # The CORPUS record's source is the whole run — a synthetic corpus-scope
        # Provenance (never a real file). file_size is the total corpus word count as a
        # rough size signal; the offsets cover [0, 0] (no single document extent).
        provenance = Provenance(
            source_uri=_CORPUS_SOURCE_URI,
            file_name="corpus",
            file_type="corpus",
            file_size=values.total_words,
            document_id=_CORPUS_DOCUMENT_ID,
            char_start=0,
            char_end=0,
        )
        classification = Classification(
            language=language,
            category=self._corpus_category,
            sensitivity=self._corpus_sensitivity,
        )
        return FeatureRecord(
            record_id="corpus#features",
            scope=FeatureScope.CORPUS,
            document_record_id=None,
            corpus=features,
            provenance=provenance,
            classification=classification,
        )


class LightweightProfiler:
    """Second reference Profiler Provider (#77): scale-oriented, co-occurrence-free.

    An alternative CPU Profiler behind the same :class:`~latence_core.capability.Profiler`
    seam as :class:`StatisticalProfiler`, satisfying the identical typed contract — one
    ``DOCUMENT``-scope :class:`FeatureRecord` per parsed document plus exactly one
    ``CORPUS``-scope record, both carrying the fixed feature set with Provenance +
    Classification. Pure Python, streaming, deterministic, zero extra deps (ADR-0007
    CPU-first, ADR-0016 thin core); no model, so no license note (ADR-0012).

    The difference vs the reference is a scale trade-off: it **skips the O(distinct-
    entities²) co-occurrence pairing** entirely (the corpus record's ``co_occurrences`` is
    always empty) while still computing the cheap O(mentions) tallies — entity frequency,
    type frequency, and type consensus. On a huge corpus with a hostile NER fan-out the
    reference's pairing is the dominant cost even with its per-document cap; this Provider
    removes that cost by construction, trading the co-occurrence feature for a strictly
    linear-in-mentions pass. The per-document features are the same fixed set the contract
    mandates (so they match the reference document-for-document — that is the required
    parity), and the corpus features match except the intentionally-empty co-occurrences.
    Benchmarked against the reference in ``test_profiling_second_provider``.

    Config:
        corpus_sensitivity: str  — sensitivity stamped on the CORPUS record (default
            ``"internal"``).
        corpus_category: str | None  — category stamped on the CORPUS record (default
            ``"corpus_profile"``).
    """

    # The declared descriptor (ADR-0036 §2): the lightweight Profiler is the scale-oriented
    # streaming feature pass — a pure-Python part of ``latence-core`` (Apache-2.0, verified per
    # ADR-0012), no model weights, deterministic, CPU-only, non-batching, no per-call cost.
    # Declared as a ``PROFILE`` ClassVar (not a ``profile`` classmethod) because the Profiler
    # Capability's I/O method is itself named ``profile`` (see StatisticalProfiler.PROFILE).
    PROFILE: ClassVar[ProviderProfile] = ProviderProfile(
        compute="cpu",
        memory_mb=32,
        model_id=None,
        license="Apache-2.0",
        license_verified=True,
        license_code="Apache-2.0",
        license_source=REPO_LICENSE_SOURCE,
        license_verified_on="2026-07-06",
        deterministic=True,
        batch=False,
        cost_per_1k=None,
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        # Reuse the reference's contract-stamping (identical per-document + corpus record
        # shape), but force the co-occurrence pairing off: max_entities_per_document=0 makes
        # the aggregator record no pairs, so the pass is linear in the mention count.
        self._delegate = StatisticalProfiler(
            {
                "max_co_occurrences": 0,
                "max_entities_per_document": 0,
                "max_tracked_pairs": 0,
                "corpus_sensitivity": str(config.get("corpus_sensitivity", "internal")),
                "corpus_category": config.get("corpus_category", "corpus_profile"),
            }
        )

    def profile(
        self, documents: Iterable[DocumentRecord], mentions: Iterable[EntityMention]
    ) -> Iterator[FeatureRecord]:
        # Same streaming fold as the reference (documents one at a time); the delegate's
        # aggregator, configured with a zero entity cap, never builds the O(n²) pair table.
        yield from self._delegate.profile(documents, mentions)
