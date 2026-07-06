"""latence-core: data contracts, Capability protocols, Pipeline, local Runner, Storage.

The thin, near-zero-dependency spine of the Latence framework (ADR-0016).
"""

from __future__ import annotations

from latence_core.capability import (
    ENTRY_POINT_GROUP,
    Export,
    Parser,
    ProviderRegistry,
    Source,
)
from latence_core.contracts import (
    SCHEMA_VERSION,
    Classification,
    Disposition,
    DocumentRecord,
    PageMap,
    PageMapAssembly,
    PageSpan,
    ParserInput,
    Provenance,
    Record,
)
from latence_core.language import UNKNOWN_LANGUAGE, detect_language
from latence_core.pagemap import PageDriftDiagnostics, PageOffsetIndex
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.quality import CompletenessCheck, ParseQuality, QualityReport, StageMetrics
from latence_core.runner import LocalRunner, topological_order
from latence_core.storage import SchemeNotAllowedError, Storage

__all__ = [
    "ENTRY_POINT_GROUP",
    "SCHEMA_VERSION",
    "UNKNOWN_LANGUAGE",
    "CapabilityKind",
    "Classification",
    "CompletenessCheck",
    "Disposition",
    "DocumentRecord",
    "Export",
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
    "Record",
    "SchemeNotAllowedError",
    "Source",
    "Stage",
    "StageMetrics",
    "Storage",
    "detect_language",
    "topological_order",
]

__version__ = "0.1.0"
