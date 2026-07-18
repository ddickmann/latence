"""``latence-sink-qdrant`` — load a latence Export into Qdrant, once, idempotently.

The first of the first-party sink family (#156). It is a **loader, not an index**: the files the
pipeline exported stay the source of truth (ADR-0017) and this package is an additional, optional
step that projects them into a vector store; it holds no index and runs no search (ADR-0048).

One command::

    latence-sink-qdrant load ./out/export --collection corpus --url http://localhost:6333

One object::

    from latence_sink_qdrant import QdrantSink
    report = QdrantSink({"collection": "corpus", "url": "http://localhost:6333"}).load("./out/export")

And one guarantee worth stating plainly: re-running converges. Records whose content is unchanged
are not re-upserted, changed records are overwritten at the same derived point id, and records the
Export no longer contains — the retracted or purged ones — are deleted and *counted back to zero*
before the load reports success.
"""

from __future__ import annotations

from latence_sink_qdrant.errors import (
    SinkAuthError,
    SinkConfigError,
    SinkError,
    SinkExportError,
    SinkLoadError,
)
from latence_sink_qdrant.export_reader import ExportRow, read_export
from latence_sink_qdrant.sink import (
    DEFAULT_ID_NAMESPACE,
    DIGEST_KEY,
    LOAD_ID_KEY,
    RESERVED_PAYLOAD_KEYS,
    LoadReport,
    QdrantSink,
    point_id,
)

__all__ = [
    "DEFAULT_ID_NAMESPACE",
    "DIGEST_KEY",
    "LOAD_ID_KEY",
    "RESERVED_PAYLOAD_KEYS",
    "ExportRow",
    "LoadReport",
    "QdrantSink",
    "SinkAuthError",
    "SinkConfigError",
    "SinkError",
    "SinkExportError",
    "SinkLoadError",
    "point_id",
    "read_export",
]
