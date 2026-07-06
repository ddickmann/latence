"""Versioned data contracts shared across every Stage boundary.

One schema, dual serialization (ADR-0006): these Pydantic v2 models are the
single canonical definition of inter-Stage data. Every :class:`Record` carries
both :class:`Provenance` (immutable source lineage) and :class:`Classification`
(descriptive content attributes); validation fails if either is missing, so a
record can never silently lose its source or attributes at a Stage boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Bump on any breaking change to a contract below. Serialized with every record
# so downstream consumers can detect and reject an incompatible schema.
SCHEMA_VERSION = 1


class Provenance(BaseModel):
    """The immutable chain from a record back to its source.

    Required fields identify the source document unambiguously; the optional
    offset fields locate a derived record within it (populated by Parse/Chunk).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    source_uri: str = Field(..., min_length=1, description="Storage URI read from.")
    file_name: str = Field(..., min_length=1)
    file_type: str = Field(..., min_length=1, description="Lowercased extension, no dot.")
    file_size: int = Field(..., ge=0, description="Source size in bytes.")
    document_id: str = Field(..., min_length=1, description="Content-addressed source doc id.")
    source_system: str | None = Field(default=None, description="Originating system.")
    page_start: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)


class Classification(BaseModel):
    """Descriptive attributes of a document's content.

    Attached at ingest / after Parse and inherited by every downstream record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    language: str = Field(..., min_length=1, description="BCP-47-ish tag; 'und' if unknown.")
    category: str | None = Field(default=None, description="Content category, e.g. 'contract'.")
    sensitivity: str = Field(default="unknown", description="'public'/'internal'/'confidential'.")


class Record(BaseModel):
    """Base inter-Stage record: always carries Provenance and Classification.

    Both are required with no default, so constructing a record without them
    raises ``pydantic.ValidationError`` — the enforced Stage-boundary check.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    record_id: str = Field(..., min_length=1)
    provenance: Provenance
    classification: Classification


class DocumentRecord(Record):
    """A whole-document record: the parsed Parse output for S1.

    ``content`` is markdown text; ``media_type`` records how to interpret it.
    """

    content: str = Field(default="")
    media_type: str = Field(default="text/markdown")


class ParserInput(BaseModel):
    """A raw document handed to a Parser: source lineage plus undecoded content.

    This is the input side of the Parse seam (ADR-0019). Source produces a
    ``ParserInput`` per source document — Provenance and Classification stamped,
    but ``content`` still the raw bytes (or already-decoded text) exactly as read
    from Storage, NOT yet turned into markdown. The Parser owns decoding and the
    turn into a :class:`DocumentRecord`, so the seam fits real parsers (which
    consume PDF/image/office BYTES), not just the S1 plain-text passthrough.

    ``config`` carries per-input Parser hints (e.g. a forced encoding or an OCR
    language) so a Source or the Runner can steer a Parser without a new contract.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=SCHEMA_VERSION)
    provenance: Provenance
    classification: Classification
    content: bytes | str = Field(
        ..., description="Raw, undecoded source content — bytes for binary formats, str if text."
    )
    config: dict[str, Any] = Field(
        default_factory=dict, description="Per-input Parser hints (encoding, OCR lang, …)."
    )
