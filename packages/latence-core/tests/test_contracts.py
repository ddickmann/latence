"""Contract validation: Provenance/Classification required at every boundary."""

from __future__ import annotations

import pytest
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    DocumentRecord,
    ParserInput,
    Provenance,
    Record,
)
from pydantic import ValidationError


def _provenance() -> Provenance:
    return Provenance(
        source_uri="file:///corpus/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=10,
        document_id="sha256:abc",
    )


def _classification() -> Classification:
    return Classification(language="en")


def test_record_carries_schema_version() -> None:
    rec = Record(record_id="r1", provenance=_provenance(), classification=_classification())
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.provenance.schema_version == SCHEMA_VERSION


def test_record_missing_provenance_raises() -> None:
    with pytest.raises(ValidationError):
        Record(record_id="r1", classification=_classification())  # type: ignore[call-arg]


def test_record_missing_classification_raises() -> None:
    with pytest.raises(ValidationError):
        Record(record_id="r1", provenance=_provenance())  # type: ignore[call-arg]


def test_provenance_requires_core_fields() -> None:
    with pytest.raises(ValidationError):
        Provenance(  # type: ignore[call-arg]
            source_uri="",  # empty -> min_length violation
            file_name="a.txt",
            file_type="txt",
            file_size=10,
            document_id="sha256:abc",
        )


def test_provenance_is_frozen() -> None:
    prov = _provenance()
    with pytest.raises(ValidationError):
        prov.file_size = 20  # type: ignore[misc]


def test_document_record_roundtrips_json() -> None:
    rec = DocumentRecord(
        record_id="r1",
        provenance=_provenance(),
        classification=_classification(),
        content="hello",
    )
    restored = DocumentRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Classification(language="en", bogus="x")  # type: ignore[call-arg]


def test_parser_input_requires_provenance_and_classification() -> None:
    with pytest.raises(ValidationError):
        ParserInput(content=b"x")  # type: ignore[call-arg]


def test_parser_input_carries_raw_bytes_and_roundtrips() -> None:
    raw = ParserInput(
        provenance=_provenance(),
        classification=_classification(),
        content=b"\x00\x01raw bytes",
    )
    assert raw.schema_version == SCHEMA_VERSION
    assert raw.config == {}
    # Round-trips through JSON with the raw bytes intact (checkpoint durability).
    restored = ParserInput.model_validate_json(raw.model_dump_json())
    assert restored == raw
    assert restored.content == b"\x00\x01raw bytes"


def test_parser_input_accepts_str_content_and_config() -> None:
    raw = ParserInput(
        provenance=_provenance(),
        classification=_classification(),
        content="already text",
        config={"encoding": "latin-1"},
    )
    assert raw.content == "already text"
    assert raw.config["encoding"] == "latin-1"
