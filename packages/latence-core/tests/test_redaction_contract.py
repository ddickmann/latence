"""RedactionRecord + PIISpan contracts and RedactionQuality report model (S6).

Validates the span-ordering / placeholder-consistency enforcement at the Redaction
boundary, the "raw PII value is never stored" shape of a PIISpan, and the per-type /
per-action aggregation the Quality Report exposes.
"""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    PIISpan,
    Provenance,
    RedactionAction,
    RedactionRecord,
)
from latence_core.quality import RedactionQuality
from pydantic import ValidationError


def _prov(**overrides: object) -> Provenance:
    base: dict[str, object] = {
        "source_uri": "file:///c/a.txt",
        "file_name": "a.txt",
        "file_type": "txt",
        "file_size": 100,
        "document_id": "sha256:doc1",
        "char_start": 0,
        "char_end": 40,
    }
    base.update(overrides)
    return Provenance(**base)  # type: ignore[arg-type]


def _span(**overrides: object) -> PIISpan:
    base: dict[str, object] = {
        "pii_type": "email",
        "detector": "regex:email",
        "char_start": 8,
        "char_end": 22,
        "action": RedactionAction.MASK,
        "placeholder": "[EMAIL]",
    }
    base.update(overrides)
    return PIISpan(**base)  # type: ignore[arg-type]


def _record(**overrides: object) -> RedactionRecord:
    base: dict[str, object] = {
        "record_id": "doc1#redaction",
        "document_record_id": "doc1",
        "masked_content": "Contact [EMAIL] now.",
        "policy": "public",
        "pii_spans": [_span()],
        "provenance": _prov(),
        "classification": Classification(language="en", sensitivity="public"),
    }
    base.update(overrides)
    return RedactionRecord(**base)  # type: ignore[arg-type]


# -- schema version guard ----------------------------------------------------


def test_schema_version_is_six() -> None:
    assert SCHEMA_VERSION == 6
    assert _record().schema_version == 6


# -- PIISpan ------------------------------------------------------------------


def test_pii_span_carries_no_raw_value() -> None:
    # A PIISpan has offset/type/action/placeholder but NO field for the raw PII value
    # — the audit trail can never itself become a PII leak (S6 AC).
    span = _span()
    assert "value" not in span.model_dump()
    assert set(span.model_dump()) == {
        "schema_version",
        "pii_type",
        "detector",
        "char_start",
        "char_end",
        "action",
        "placeholder",
        "page_start",
        "page_end",
        "confidence",
    }


def test_pii_span_inverted_span_rejected() -> None:
    with pytest.raises(ValidationError):
        _span(char_start=20, char_end=8)


def test_non_tag_span_requires_placeholder() -> None:
    # A mask/replace/hash span must carry the placeholder it wrote; only a TAG span
    # may have an empty placeholder.
    with pytest.raises(ValidationError):
        _span(action=RedactionAction.MASK, placeholder="")
    tag = _span(action=RedactionAction.TAG, placeholder="")
    assert tag.action is RedactionAction.TAG
    assert tag.placeholder == ""


def test_pii_span_length() -> None:
    assert _span(char_start=8, char_end=22).length == 14


# -- RedactionRecord ----------------------------------------------------------


def test_redaction_record_requires_provenance_and_classification() -> None:
    with pytest.raises(ValidationError):
        RedactionRecord(  # type: ignore[call-arg]
            record_id="r", document_record_id="doc1", masked_content="x"
        )


def test_redaction_record_requires_char_span_on_provenance() -> None:
    with pytest.raises(ValidationError):
        _record(provenance=_prov(char_start=None, char_end=None))


def test_redaction_record_rejects_overlapping_spans() -> None:
    # Spans must be ordered and non-overlapping so the masked variant is
    # reconstructable and the audit is unambiguous.
    a = _span(char_start=8, char_end=22)
    b = _span(char_start=15, char_end=30, pii_type="phone", placeholder="[PHONE]")
    with pytest.raises(ValidationError):
        _record(pii_spans=[a, b])


def test_redaction_record_accepts_ordered_spans_and_counts() -> None:
    a = _span(char_start=8, char_end=22)
    b = _span(char_start=25, char_end=37, pii_type="phone", placeholder="[PHONE]")
    rec = _record(pii_spans=[a, b])
    assert rec.pii_count == 2


def test_clean_document_has_no_spans() -> None:
    rec = _record(masked_content="Nothing sensitive here.", pii_spans=[])
    assert rec.pii_count == 0


# -- RedactionQuality ---------------------------------------------------------


def test_redaction_quality_defaults() -> None:
    q = RedactionQuality()
    assert q.pii_total == 0
    assert q.documents_with_pii == 0
    assert q.per_type == {}
    assert q.per_action == {}
