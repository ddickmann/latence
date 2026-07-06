"""RelationMention contract + RelationQuality report models (S5).

Validates the head/tail + Provenance-span enforcement at the relation boundary and
the confidence-histogram / per-type aggregation the Quality Report exposes.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    Provenance,
    RelationMention,
)
from latence_core.quality import RelationQuality
from pydantic import ValidationError


def _prov(**overrides: object) -> Provenance:
    base: dict[str, object] = {
        "source_uri": "file:///c/a.txt",
        "file_name": "a.txt",
        "file_type": "txt",
        "file_size": 100,
        "document_id": "sha256:doc1",
        "char_start": 10,
        "char_end": 40,
        "page_start": 1,
        "page_end": 1,
    }
    base.update(overrides)
    return Provenance(**base)  # type: ignore[arg-type]


def _relation(**overrides: object) -> RelationMention:
    base: dict[str, object] = {
        "record_id": "doc1#relation-0",
        "document_record_id": "doc1",
        "index": 0,
        "label": "works_for",
        "head_mention_id": "doc1#chunk-0#entity-0",
        "tail_mention_id": "doc1#chunk-0#entity-1",
        "head_text": "Alice",
        "tail_text": "Acme Corp",
        "confidence": 0.9,
        "provenance": _prov(),
        "classification": Classification(language="en"),
    }
    base.update(overrides)
    return RelationMention(**base)  # type: ignore[arg-type]


def test_schema_version_is_current() -> None:
    # SCHEMA_VERSION is corpus-wide (7 after S7 Profiling); a RelationMention
    # serialized now stamps the current version, still a valid RelationMention.
    assert SCHEMA_VERSION == 9
    assert _relation().schema_version == SCHEMA_VERSION


def test_valid_relation_round_trips() -> None:
    r = _relation()
    again = RelationMention.model_validate_json(r.model_dump_json())
    assert again == r


def test_relation_requires_distinct_head_and_tail() -> None:
    with pytest.raises(ValidationError, match="distinct mentions"):
        _relation(head_mention_id="same", tail_mention_id="same")


def test_relation_requires_char_span() -> None:
    with pytest.raises(ValidationError, match="char_start and char_end"):
        _relation(provenance=_prov(char_start=None, char_end=None))


def test_relation_rejects_inverted_span() -> None:
    with pytest.raises(ValidationError, match="inverted"):
        _relation(provenance=_prov(char_start=40, char_end=10))


def test_relation_requires_page_span() -> None:
    with pytest.raises(ValidationError, match="page_start and page_end"):
        _relation(provenance=_prov(page_start=None, page_end=None))


def test_relation_requires_provenance_and_classification() -> None:
    # Record base enforces both at every boundary (Baseline bar contract check).
    with pytest.raises(ValidationError):
        RelationMention(  # type: ignore[call-arg]
            record_id="x",
            document_record_id="doc1",
            index=0,
            label="works_for",
            head_mention_id="h",
            tail_mention_id="t",
            head_text="Alice",
            tail_text="Acme",
            confidence=0.9,
        )


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        _relation(confidence=1.5)
    with pytest.raises(ValidationError):
        _relation(confidence=-0.1)


def test_relation_quality_offset_resolving_flag() -> None:
    q = RelationQuality(relations_total=3, relations_offset_resolving=3)
    assert q.offset_resolving is True
    q2 = RelationQuality(relations_total=3, relations_offset_resolving=2)
    assert q2.offset_resolving is False


def test_relation_quality_defaults_empty() -> None:
    q = RelationQuality()
    assert q.relations_total == 0
    assert q.per_type == {}
    assert q.confidence.count == 0
    assert q.offset_resolving is True  # vacuously (0 == 0)
