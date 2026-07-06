"""Declarative Pipeline contract (ADR-0015).

A versioned YAML/JSON document is the canonical Pipeline declaration: an ordered
list of Stages, each binding a Capability kind to a Provider (plugin name + config),
plus Storage URIs. A typed Pydantic builder constructs and validates the same
object, and can round-trip back to YAML — config-as-data that platform teams diff
and commit, and what makes the one-line Provider swap literally true.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from latence_core.storage import Storage

PIPELINE_SCHEMA_VERSION = 1


class CapabilityKind(StrEnum):
    """The Capability a Stage fulfils.

    S1 spine: source/parse/export. S3 adds Chunk and Screening's two checkpoints
    (intake before Parse, content after Chunk). S4 adds Entity Extraction (typed
    entity mentions found in chunked text). S5 adds Relation Extraction (typed
    relations between mentions) and a *fused* Entity+Relation Extraction — one
    Provider fulfilling both adjacent Stages in one pass (ADR-0013), so the Pipeline
    is not forced to one Provider per Stage.
    """

    SOURCE = "source"
    INTAKE_SCREENING = "intake_screening"
    PARSE = "parse"
    CHUNK = "chunk"
    CONTENT_SCREENING = "content_screening"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    FUSED_ENTITY_RELATION = "fused_entity_relation"
    EXPORT = "export"


class Stage(BaseModel):
    """One node in the Pipeline DAG: a named Stage bound to a Provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, description="Unique Stage name within the Pipeline.")
    capability: CapabilityKind
    provider: str = Field(..., min_length=1, description="Entry-point name of the Provider plugin.")
    config: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class Pipeline(BaseModel):
    """A DAG of Stages over a set of input documents, declared once."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=PIPELINE_SCHEMA_VERSION)
    name: str = Field(..., min_length=1)
    storage_uri: str = Field(..., min_length=1, description="Base URI for state and exports.")
    stages: list[Stage] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_dag(self) -> Pipeline:
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            msg = f"Duplicate Stage names: {dupes}"
            raise ValueError(msg)
        known = set(names)
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in known:
                    msg = f"Stage {stage.name!r} depends on unknown Stage {dep!r}"
                    raise ValueError(msg)
                if dep == stage.name:
                    msg = f"Stage {stage.name!r} cannot depend on itself"
                    raise ValueError(msg)
        return self

    @classmethod
    def from_yaml_text(cls, text: str) -> Pipeline:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            msg = "Pipeline YAML must be a mapping at the top level."
            raise ValueError(msg)
        return cls.model_validate(data)

    @classmethod
    def load(cls, uri: str, storage: Storage) -> Pipeline:
        return cls.from_yaml_text(storage.read_text(uri))

    def to_yaml_text(self) -> str:
        text: str = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        return text
