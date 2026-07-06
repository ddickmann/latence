"""Quality Report — a first-class run artifact (ADR-0008).

The S1 skeleton reports what the walking skeleton can substantiate: document
counts, per-Stage timing, and a contract-completeness check (every record has
Provenance and Classification). Later slices extend this same artifact with
per-Stage quality metrics, KG statistics, and gold-set precision/recall.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

QUALITY_SCHEMA_VERSION = 1


class StageMetrics(BaseModel):
    """Per-Stage observability rolled into the Quality Report."""

    model_config = ConfigDict(extra="forbid")

    name: str
    capability: str
    provider: str
    records_in: int = 0
    records_out: int = 0
    duration_seconds: float = 0.0
    from_checkpoint: bool = Field(
        default=False, description="True if this Stage was skipped and resumed from a checkpoint."
    )


class CompletenessCheck(BaseModel):
    """Contract-completeness: every exported record carries Provenance + Classification."""

    model_config = ConfigDict(extra="forbid")

    records_checked: int = 0
    records_with_provenance: int = 0
    records_with_classification: int = 0

    @property
    def passed(self) -> bool:
        return (
            self.records_with_provenance == self.records_checked
            and self.records_with_classification == self.records_checked
        )


class QualityReport(BaseModel):
    """The per-run report emitted alongside the data."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=QUALITY_SCHEMA_VERSION)
    pipeline_name: str
    run_id: str
    document_count: int = 0
    total_duration_seconds: float = 0.0
    stages: list[StageMetrics] = Field(default_factory=list)
    completeness: CompletenessCheck = Field(default_factory=CompletenessCheck)
    contracts_complete: bool = True

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
