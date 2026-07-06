"""FeatureRecord contract (S7): scope/payload consistency + Provenance/Classification.

The Baseline-bar contract checks: a FeatureRecord always carries Provenance and
Classification (inherited from Record), the scope tag and the populated payload must
agree (DOCUMENT ⇒ document features + a document_record_id, no corpus; CORPUS ⇒ corpus
features, no document features, no document_record_id), and SCHEMA_VERSION is stamped.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    CorpusFeatures,
    DocumentFeatures,
    EntityCoOccurrence,
    FeatureRecord,
    FeatureScope,
    Provenance,
)
from pydantic import ValidationError


def _prov() -> Provenance:
    return Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=0,
        char_end=100,
    )


def _classification() -> Classification:
    return Classification(language="en", sensitivity="public")


def _doc_features() -> DocumentFeatures:
    return DocumentFeatures(
        char_count=100,
        word_count=20,
        sentence_count=3,
        unique_word_count=15,
        type_token_ratio=0.75,
        density=0.8,
        readability=0.6,
        zipf_alpha=1.1,
        compression_ratio=0.5,
        avg_word_length=4.2,
        avg_sentence_length=6.6,
    )


def _corpus_features() -> CorpusFeatures:
    return CorpusFeatures(
        document_count=2,
        total_words=40,
        total_unique_words=25,
        corpus_zipf_alpha=1.05,
        language_coverage={"en": 2},
        sensitivity_coverage={"public": 2},
        entity_frequency={"Acme": 3},
        entity_type_frequency={"ORG": 3},
        entity_type_consensus={"Acme": 1.0},
        co_occurrences=[EntityCoOccurrence(left="Acme", right="Alice", document_count=2)],
    )


def test_document_scope_record_is_valid_and_carries_contracts() -> None:
    record = FeatureRecord(
        record_id="d1#features",
        scope=FeatureScope.DOCUMENT,
        document_record_id="d1",
        document=_doc_features(),
        provenance=_prov(),
        classification=_classification(),
    )
    assert record.scope is FeatureScope.DOCUMENT
    assert record.provenance is not None
    assert record.classification is not None
    assert record.corpus is None
    assert record.schema_version == SCHEMA_VERSION


def test_corpus_scope_record_is_valid() -> None:
    record = FeatureRecord(
        record_id="corpus#features",
        scope=FeatureScope.CORPUS,
        corpus=_corpus_features(),
        provenance=_prov(),
        classification=_classification(),
    )
    assert record.scope is FeatureScope.CORPUS
    assert record.document is None
    assert record.document_record_id is None


def test_missing_provenance_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureRecord(
            record_id="d1#features",
            scope=FeatureScope.DOCUMENT,
            document_record_id="d1",
            document=_doc_features(),
            classification=_classification(),  # type: ignore[call-arg]
        )


def test_missing_classification_rejected() -> None:
    with pytest.raises(ValidationError):
        FeatureRecord(
            record_id="d1#features",
            scope=FeatureScope.DOCUMENT,
            document_record_id="d1",
            document=_doc_features(),
            provenance=_prov(),  # type: ignore[call-arg]
        )


def test_document_scope_without_document_features_rejected() -> None:
    with pytest.raises(ValidationError, match="must carry a 'document' feature set"):
        FeatureRecord(
            record_id="d1#features",
            scope=FeatureScope.DOCUMENT,
            document_record_id="d1",
            provenance=_prov(),
            classification=_classification(),
        )


def test_document_scope_with_corpus_features_rejected() -> None:
    with pytest.raises(ValidationError, match="must not carry 'corpus' features"):
        FeatureRecord(
            record_id="d1#features",
            scope=FeatureScope.DOCUMENT,
            document_record_id="d1",
            document=_doc_features(),
            corpus=_corpus_features(),
            provenance=_prov(),
            classification=_classification(),
        )


def test_document_scope_without_document_id_rejected() -> None:
    with pytest.raises(ValidationError, match="must carry the profiled document_record_id"):
        FeatureRecord(
            record_id="d1#features",
            scope=FeatureScope.DOCUMENT,
            document=_doc_features(),
            provenance=_prov(),
            classification=_classification(),
        )


def test_corpus_scope_without_corpus_features_rejected() -> None:
    with pytest.raises(ValidationError, match="must carry a 'corpus' feature set"):
        FeatureRecord(
            record_id="corpus#features",
            scope=FeatureScope.CORPUS,
            provenance=_prov(),
            classification=_classification(),
        )


def test_corpus_scope_with_document_id_rejected() -> None:
    with pytest.raises(ValidationError, match="must not name a single document_record_id"):
        FeatureRecord(
            record_id="corpus#features",
            scope=FeatureScope.CORPUS,
            document_record_id="d1",
            corpus=_corpus_features(),
            provenance=_prov(),
            classification=_classification(),
        )


def test_feature_record_round_trips_json() -> None:
    record = FeatureRecord(
        record_id="corpus#features",
        scope=FeatureScope.CORPUS,
        corpus=_corpus_features(),
        provenance=_prov(),
        classification=_classification(),
    )
    restored = FeatureRecord.model_validate_json(record.model_dump_json())
    assert restored == record
