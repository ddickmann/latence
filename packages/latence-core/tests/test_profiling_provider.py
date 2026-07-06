"""StatisticalProfiler (S7, in-core reference): corpus → per-document + corpus features.

Unit tests for the reference Provider: it emits one DOCUMENT-scope FeatureRecord per
parsed document (with the per-document feature set) plus exactly one CORPUS-scope record
(with the cross-document aggregates), skips PARSE_ERROR documents, folds entity mentions
into the corpus entity features, is streaming (emits as a generator), and is
deterministic. Also the Parse-only degradation (no mentions -> empty entity aggregates).
"""

from __future__ import annotations

from latence_core.contracts import (
    Classification,
    Disposition,
    DocumentRecord,
    EntityMention,
    FeatureScope,
    PageMap,
    PageSpan,
    Provenance,
)
from latence_core.stages.profiling import StatisticalProfiler


def _prov(text: str, doc_id: str, *, page_end: int = 1) -> Provenance:
    return Provenance(
        source_uri=f"file:///c/{doc_id}.txt",
        file_name=f"{doc_id}.txt",
        file_type="txt",
        file_size=len(text),
        document_id=f"sha256:{doc_id}",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=page_end,
    )


def _doc(
    text: str,
    doc_id: str = "d1",
    *,
    language: str = "en",
    sensitivity: str = "public",
    category: str | None = "memo",
    disposition: Disposition = Disposition.PARSED,
    error: str | None = None,
) -> DocumentRecord:
    page_map = None
    if disposition is Disposition.PARSED:
        page_map = PageMap(
            pages=[PageSpan(page_number=1, char_start=0, char_end=len(text))],
            total_chars=len(text),
        )
    return DocumentRecord(
        record_id=doc_id,
        provenance=_prov(text, doc_id),
        classification=Classification(
            language=language, category=category, sensitivity=sensitivity
        ),
        content=text if disposition is Disposition.PARSED else "",
        page_map=page_map,
        disposition=disposition,
        error=error,
    )


def _mention(doc_id: str, text: str, label: str, index: int) -> EntityMention:
    prov = Provenance(
        source_uri=f"file:///c/{doc_id}.txt",
        file_name=f"{doc_id}.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc_id}",
        char_start=index,
        char_end=index + len(text),
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=f"{doc_id}#chunk-0#entity-{index}",
        document_record_id=doc_id,
        chunk_record_id=f"{doc_id}#chunk-0",
        index=index,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
    )


# -- happy path ---------------------------------------------------------------


def test_emits_one_document_record_per_doc_plus_one_corpus_record() -> None:
    docs = [_doc("The cat sat on the mat.", "d1"), _doc("The dog ran home.", "d2")]
    records = list(StatisticalProfiler().profile(docs, []))
    doc_records = [r for r in records if r.scope is FeatureScope.DOCUMENT]
    corpus_records = [r for r in records if r.scope is FeatureScope.CORPUS]
    assert len(doc_records) == 2
    assert len(corpus_records) == 1
    # The corpus record is emitted last.
    assert records[-1].scope is FeatureScope.CORPUS


def test_document_record_carries_features_and_provenance() -> None:
    [doc_rec, _corpus] = list(StatisticalProfiler().profile([_doc("The cat sat.", "d1")], []))
    assert doc_rec.document is not None
    assert doc_rec.document.word_count == 3
    assert doc_rec.document_record_id == "d1"
    assert doc_rec.provenance.document_id == "sha256:d1"
    assert doc_rec.classification.sensitivity == "public"


def test_corpus_record_source_coverage() -> None:
    docs = [
        _doc("The cat sat.", "d1", language="en", sensitivity="public"),
        _doc("Der Hund lief.", "d2", language="de", sensitivity="internal", category=None),
    ]
    records = list(StatisticalProfiler().profile(docs, []))
    corpus = next(r for r in records if r.scope is FeatureScope.CORPUS)
    assert corpus.corpus is not None
    assert corpus.corpus.document_count == 2
    assert corpus.corpus.language_coverage == {"de": 1, "en": 1}
    assert corpus.corpus.sensitivity_coverage == {"internal": 1, "public": 1}


def test_mentions_fold_into_corpus_entity_features() -> None:
    docs = [_doc("Alice works at Acme.", "d1"), _doc("Acme is a firm.", "d2")]
    mentions = [
        _mention("d1", "Alice", "PERSON", 0),
        _mention("d1", "Acme", "ORG", 15),
        _mention("d2", "Acme", "ORG", 0),
    ]
    records = list(StatisticalProfiler().profile(docs, mentions))
    corpus = next(r for r in records if r.scope is FeatureScope.CORPUS)
    assert corpus.corpus is not None
    assert corpus.corpus.entity_frequency == {"Acme": 2, "Alice": 1}
    assert corpus.corpus.entity_type_frequency == {"ORG": 2, "PERSON": 1}
    pairs = {(c.left, c.right): c.document_count for c in corpus.corpus.co_occurrences}
    assert pairs[("Acme", "Alice")] == 1


def test_parse_only_pipeline_has_empty_entity_aggregates() -> None:
    records = list(StatisticalProfiler().profile([_doc("The cat sat.", "d1")], []))
    corpus = next(r for r in records if r.scope is FeatureScope.CORPUS)
    assert corpus.corpus is not None
    assert corpus.corpus.entity_frequency == {}
    assert corpus.corpus.co_occurrences == []


# -- degradations -------------------------------------------------------------


def test_parse_error_document_is_skipped() -> None:
    docs = [
        _doc("The cat sat.", "d1"),
        _doc("", "d2", disposition=Disposition.PARSE_ERROR, error="corrupt"),
    ]
    records = list(StatisticalProfiler().profile(docs, []))
    doc_records = [r for r in records if r.scope is FeatureScope.DOCUMENT]
    assert len(doc_records) == 1
    assert doc_records[0].document_record_id == "d1"
    corpus = next(r for r in records if r.scope is FeatureScope.CORPUS)
    assert corpus.corpus is not None
    assert corpus.corpus.document_count == 1


def test_empty_corpus_yields_only_corpus_record() -> None:
    records = list(StatisticalProfiler().profile([], []))
    assert len(records) == 1
    assert records[0].scope is FeatureScope.CORPUS
    assert records[0].corpus is not None
    assert records[0].corpus.document_count == 0


def test_profile_is_a_generator_streaming() -> None:
    gen = StatisticalProfiler().profile([_doc("The cat sat.", "d1")], [])
    # A generator yields lazily; the first item is a document-scope record.
    first = next(gen)
    assert first.scope is FeatureScope.DOCUMENT


# -- determinism --------------------------------------------------------------


def test_profile_is_deterministic() -> None:
    docs = [_doc("The cat sat on the mat.", "d1"), _doc("The dog ran home fast.", "d2")]
    mentions = [_mention("d1", "cat", "ANIMAL", 4), _mention("d2", "dog", "ANIMAL", 4)]
    a = [r.model_dump_json() for r in StatisticalProfiler().profile(docs, mentions)]
    b = [r.model_dump_json() for r in StatisticalProfiler().profile(docs, mentions)]
    assert a == b


def test_co_occurrence_cap_config() -> None:
    docs = [_doc("word text here now.", "d1")]
    mentions = [_mention("d1", name, "X", i) for i, name in enumerate(["A", "B", "C", "D"])]
    profiler = StatisticalProfiler({"max_co_occurrences": 2})
    records = list(profiler.profile(docs, mentions))
    corpus = next(r for r in records if r.scope is FeatureScope.CORPUS)
    assert corpus.corpus is not None
    assert len(corpus.corpus.co_occurrences) == 2


def test_pair_state_caps_are_wired_from_config() -> None:
    """The DoS caps (issue #33) are threaded from Provider config into the aggregator.

    Adversarial regression: a single document with many distinct entities must not build an
    O(n²) pair explosion. With ``max_entities_per_document`` wired through, only k*(k-1)/2
    pairs are ever formed. Would FAIL if the Provider dropped the cap on the wire.
    """
    docs = [_doc("filler text for the doc.", "d1")]
    mentions = [_mention("d1", f"E{i:03d}", "X", i) for i in range(300)]
    profiler = StatisticalProfiler(
        {"max_entities_per_document": 5, "max_co_occurrences": 1000}
    )
    records = list(profiler.profile(docs, mentions))
    corpus = next(r for r in records if r.scope is FeatureScope.CORPUS)
    assert corpus.corpus is not None
    # Capped fan-out: at most 5*4/2 = 10 pairs, not 300*299/2 = 44_850.
    assert len(corpus.corpus.co_occurrences) == 10
    # Every mention still counted in entity frequency (only pairing is capped).
    assert len(corpus.corpus.entity_frequency) == 300
