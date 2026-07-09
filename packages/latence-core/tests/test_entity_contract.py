"""EntityMention contract + EntityQuality report models (S4).

Validates the Provenance-span enforcement at the mention boundary and the
confidence-histogram / per-type aggregation the Quality Report exposes.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    EntityMention,
    Provenance,
)
from latence_core.quality import ConfidenceHistogram, EntityQuality
from pydantic import ValidationError


def _prov(**overrides: object) -> Provenance:
    base: dict[str, object] = {
        "source_uri": "file:///c/a.txt",
        "file_name": "a.txt",
        "file_type": "txt",
        "file_size": 100,
        "document_id": "sha256:doc1",
        "char_start": 10,
        "char_end": 15,
        "page_start": 1,
        "page_end": 1,
    }
    base.update(overrides)
    return Provenance(**base)  # type: ignore[arg-type]


def _mention(**overrides: object) -> EntityMention:
    base: dict[str, object] = {
        "record_id": "doc1#chunk-0#entity-0",
        "document_record_id": "doc1",
        "chunk_record_id": "doc1#chunk-0",
        "index": 0,
        "label": "PERSON",
        "text": "Alice",
        "confidence": 0.9,
        "provenance": _prov(),
        "classification": Classification(language="en"),
    }
    base.update(overrides)
    return EntityMention(**base)  # type: ignore[arg-type]


def test_schema_version_is_current() -> None:
    # SCHEMA_VERSION is corpus-wide (14 after the W4 induced-labels field). An
    # EntityMention serialized now stamps the current version, still a valid EntityMention.
    assert SCHEMA_VERSION == 14
    assert _mention().schema_version == SCHEMA_VERSION


def test_valid_mention_round_trips() -> None:
    m = _mention()
    again = EntityMention.model_validate_json(m.model_dump_json())
    assert again == m


def test_mention_requires_char_span() -> None:
    with pytest.raises(ValidationError, match="char_start and char_end"):
        _mention(provenance=_prov(char_start=None, char_end=None))


def test_mention_rejects_inverted_span() -> None:
    with pytest.raises(ValidationError, match="inverted"):
        _mention(provenance=_prov(char_start=20, char_end=10))


def test_mention_requires_page_span() -> None:
    with pytest.raises(ValidationError, match="page_start and page_end"):
        _mention(provenance=_prov(page_start=None, page_end=None))


def test_mention_requires_provenance_and_classification() -> None:
    # Record base enforces both at every boundary (Baseline bar contract check).
    with pytest.raises(ValidationError):
        EntityMention(  # type: ignore[call-arg]
            record_id="x",
            document_record_id="doc1",
            chunk_record_id="doc1#chunk-0",
            index=0,
            label="PERSON",
            text="Alice",
            confidence=0.9,
        )


def test_confidence_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        _mention(confidence=1.5)
    with pytest.raises(ValidationError):
        _mention(confidence=-0.1)


def test_confidence_histogram_binning() -> None:
    hist = ConfidenceHistogram()
    for c in (0.05, 0.15, 0.95, 1.0):
        hist.add(c)
    assert hist.bins[0] == 1  # 0.05
    assert hist.bins[1] == 1  # 0.15
    assert hist.bins[9] == 2  # 0.95 and exactly-1.0 both land in the top bin
    assert hist.count == 4


def test_confidence_histogram_mean_empty_is_zero() -> None:
    assert ConfidenceHistogram().mean == 0.0


def test_entity_quality_offset_resolving_flag() -> None:
    q = EntityQuality(mentions_total=3, mentions_offset_resolving=3)
    assert q.offset_resolving is True
    q2 = EntityQuality(mentions_total=3, mentions_offset_resolving=2)
    assert q2.offset_resolving is False
