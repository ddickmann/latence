"""latence-core: data contracts, Capability protocols, Pipeline, local Runner, Storage.

The thin, near-zero-dependency spine of the Latence framework (ADR-0016).
"""

from __future__ import annotations

from latence_core.capability import (
    ENTRY_POINT_GROUP,
    Chunker,
    ContentScreener,
    Export,
    IntakeScreener,
    Parser,
    ProviderRegistry,
    ScreeningOutcome,
    Source,
)
from latence_core.chunking import Chunk, ChunkBudget, chunk_document, estimate_tokens, strip_markup
from latence_core.contracts import (
    SCHEMA_VERSION,
    ChunkRecord,
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    PageMapAssembly,
    PageSpan,
    ParserInput,
    Provenance,
    QuarantineRecord,
    Record,
    RiskMarker,
    ScreeningCheckpoint,
    ScreeningFinding,
    ScreeningVerdict,
)
from latence_core.language import UNKNOWN_LANGUAGE, detect_language
from latence_core.pagemap import PageDriftDiagnostics, PageOffsetIndex
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.quality import (
    ChunkQuality,
    CompletenessCheck,
    ParseQuality,
    QualityReport,
    ScreeningQuality,
    StageMetrics,
)
from latence_core.runner import LocalRunner, topological_order
from latence_core.storage import SchemeNotAllowedError, Storage

__all__ = [
    "ENTRY_POINT_GROUP",
    "SCHEMA_VERSION",
    "UNKNOWN_LANGUAGE",
    "CapabilityKind",
    "Chunk",
    "ChunkBudget",
    "ChunkQuality",
    "ChunkRecord",
    "Chunker",
    "Classification",
    "CompletenessCheck",
    "ContentScreener",
    "Disposition",
    "DocumentRecord",
    "Export",
    "IntakeScreener",
    "LocalRunner",
    "PageDriftDiagnostics",
    "PageMap",
    "PageMapAssembly",
    "PageOffsetIndex",
    "PageSpan",
    "ParseQuality",
    "Parser",
    "ParserInput",
    "Pipeline",
    "Provenance",
    "ProviderRegistry",
    "QualityReport",
    "QuarantineRecord",
    "Record",
    "RiskMarker",
    "SchemeNotAllowedError",
    "ScreeningCheckpoint",
    "ScreeningFinding",
    "ScreeningOutcome",
    "ScreeningQuality",
    "ScreeningVerdict",
    "Source",
    "Stage",
    "StageMetrics",
    "Storage",
    "chunk_document",
    "detect_language",
    "estimate_tokens",
    "strip_markup",
    "topological_order",
]

__version__ = "0.1.0"
