"""Quality Report — a first-class run artifact (ADR-0008).

The S1 skeleton reports what the walking skeleton can substantiate: document
counts, per-Stage timing, and a contract-completeness check (every record has
Provenance and Classification). Later slices extend this same artifact with
per-Stage quality metrics, KG statistics, and gold-set precision/recall.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from latence_core.contracts import ScreeningFinding

# v3 (S3): the report gains ``chunk`` (Chunk-Stage counts + offset-integrity) and
# ``screening`` (both checkpoints' quarantine/flag findings).
# v4 (S4): the report gains ``entities`` (Entity-Extraction per-type mention counts
# and the confidence distribution).
QUALITY_SCHEMA_VERSION = 4


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


class ParseQuality(BaseModel):
    """Parse-Stage quality: page-drift diagnostics + parse-error counts (S2).

    Emitted into the Quality Report when a run parsed documents. ``page_drift_*``
    aggregate the fuzzy offset→page recoveries the ported ``PageOffsetIndex`` made
    while the Runner audited each parsed record's page map — a clean parse has zero
    drift. ``parse_errors`` counts documents the Parser could not decode (emitted
    as ``PARSE_ERROR`` records rather than crashing the run). ``bytes_after_parse``
    sums the assembled-markdown byte size of every parsed record — the observable
    Provenance 'size-after-parse' rolled up across the Stage.
    """

    model_config = ConfigDict(extra="forbid")

    documents_parsed: int = 0
    parse_errors: int = 0
    pages_total: int = 0
    page_map_present: int = 0
    bytes_after_parse: int = 0
    page_drift_resolutions: int = 0
    page_drift_recovered: int = 0
    page_drift_max_chars: int = 0

    @property
    def page_maps_clean(self) -> bool:
        """True if every offset→page audit resolved exactly (no fuzzy recovery)."""
        return self.page_drift_recovered == 0


class ChunkQuality(BaseModel):
    """Chunk-Stage quality: chunk/token counts + offset-preservation integrity (S3).

    Emitted when a run chunked documents. ``chunks_offset_preserving`` counts
    chunks whose Provenance carried a valid source char span AND page span — the
    S3 round-trip invariant (chunk → offset → original page). ``offset_preserving``
    is True only when every chunk preserved its offsets. ``chunks_flagged`` counts
    chunks Content Screening marked with a risk marker.
    """

    model_config = ConfigDict(extra="forbid")

    chunks_total: int = 0
    chunks_offset_preserving: int = 0
    chunks_flagged: int = 0
    tokens_total: int = 0

    @property
    def offset_preserving(self) -> bool:
        """True if every emitted chunk carried a resolvable source offset+page span."""
        return self.chunks_offset_preserving == self.chunks_total


class ScreeningQuality(BaseModel):
    """Screening quality: quarantine + flag dispositions across both checkpoints (S3).

    Emitted when a run had a Screening Stage. Counts are rolled up from the
    per-record :class:`~latence_core.contracts.ScreeningFinding`s, which are also
    carried verbatim in ``findings`` so every Quarantine disposition and reason is
    inspectable in the Quality Report (S3 AC: "Quarantine dispositions and reasons
    appear in the Quality Report").
    """

    model_config = ConfigDict(extra="forbid")

    intake_screened: int = 0
    content_screened: int = 0
    quarantined: int = 0
    flagged: int = 0
    passed: int = 0
    findings: list[ScreeningFinding] = Field(default_factory=list)

    @property
    def total_screened(self) -> int:
        return self.intake_screened + self.content_screened


class ConfidenceHistogram(BaseModel):
    """A fixed-bin histogram of mention confidences (the confidence distribution, S4).

    Ten equal-width bins over ``[0.0, 1.0]``: ``bins[i]`` counts mentions whose
    confidence falls in ``[i/10, (i+1)/10)`` — except a confidence of exactly 1.0,
    which lands in the top bin. Deterministic and dependency-free, so a seeded run
    yields the identical distribution.
    """

    model_config = ConfigDict(extra="forbid")

    bins: list[int] = Field(default_factory=lambda: [0] * 10)

    def add(self, confidence: float) -> None:
        """Increment the bin ``confidence`` falls in (1.0 lands in the top bin)."""
        clamped = min(max(confidence, 0.0), 1.0)
        idx = min(int(clamped * 10), 9)
        self.bins[idx] += 1

    @property
    def count(self) -> int:
        return sum(self.bins)

    @property
    def mean(self) -> float:
        """Mean confidence, estimated at each bin's midpoint (0.0 if empty)."""
        total = self.count
        if total == 0:
            return 0.0
        weighted = sum(count * ((i + 0.5) / 10) for i, count in enumerate(self.bins))
        return round(weighted / total, 6)


class EntityQuality(BaseModel):
    """Entity-Extraction quality: per-type mention counts + confidence distribution (S4).

    Emitted when a run had an Entity-Extraction Stage. ``mentions_total`` is the
    number of typed :class:`~latence_core.contracts.EntityMention`s produced;
    ``per_type`` breaks that down by entity label (the zero-shot label set the
    Pipeline configured); ``confidence`` is the fixed-bin distribution of mention
    confidences. ``mentions_offset_resolving`` counts mentions whose Provenance
    carried a valid source char span AND page span — the S4 round-trip invariant
    (mention → offset → original page).
    """

    model_config = ConfigDict(extra="forbid")

    mentions_total: int = 0
    mentions_offset_resolving: int = 0
    chunks_with_mentions: int = 0
    per_type: dict[str, int] = Field(default_factory=dict)
    confidence: ConfidenceHistogram = Field(default_factory=ConfidenceHistogram)

    @property
    def offset_resolving(self) -> bool:
        """True if every emitted mention carried a resolvable source offset+page span."""
        return self.mentions_offset_resolving == self.mentions_total


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
    # Present when the run had a Parse Stage; ``None`` for a Parse-free pipeline.
    parse: ParseQuality | None = None
    # Present when the run had a Chunk Stage / a Screening Stage respectively.
    chunk: ChunkQuality | None = None
    screening: ScreeningQuality | None = None
    # Present when the run had an Entity-Extraction Stage.
    entities: EntityQuality | None = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)
