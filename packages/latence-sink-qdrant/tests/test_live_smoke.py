"""The live smoke path: the same load, against a real Qdrant.

Skipped unless you opt in — the offline suite is the merge gate (CPU-first, no services,
ADR-0007/0016), and this is the out-of-band validation that the faithful fake is telling the truth::

    docker run -p 6333:6333 qdrant/qdrant
    uv pip install 'qdrant-client>=1.10'
    LATENCE_QDRANT_URL=http://localhost:6333 uv run pytest packages/latence-sink-qdrant -k live

It asserts the *same* three properties the offline tests assert — load, converge, and make a
retraction verifiably absent — through the real client, real HTTP and a real collection, so a drift
between the fake's contract and Qdrant's shows up here as a failure rather than as a silent false
green offline. It cleans its collection up on the way in and out.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from latence_sink_qdrant import QdrantSink

if TYPE_CHECKING:
    from pathlib import Path

QDRANT_URL = os.environ.get("LATENCE_QDRANT_URL")

# Two independent gates: the URL env var (opt-in) and the optional client dep (checked inside the
# test with importorskip, so a missing dep reads as "not installed", never as a failure).
pytestmark = pytest.mark.skipif(
    not QDRANT_URL, reason="set LATENCE_QDRANT_URL to run the live smoke"
)

_DIMENSION = 16


def _export(tmp_path: Path, record_ids: list[str]) -> str:
    export = JsonlParquetExport(
        {"embedder": {"provider": "embedding.hashing", "config": {"dimension": _DIMENSION}}}
    )
    records = [
        ChunkRecord(
            record_id=rid,
            provenance=Provenance(
                source_uri="file:///c/a.txt",
                file_name="a.txt",
                file_type="txt",
                file_size=10,
                document_id="sha256:d1",
                char_start=0,
                char_end=5,
                page_start=1,
                page_end=1,
            ),
            classification=Classification(language="en", sensitivity="public"),
            document_record_id="d1",
            index=0,
            content=f"live text for {rid}",
            token_count=4,
        )
        for rid in record_ids
    ]
    export.export(records, Storage(), str(tmp_path))
    return str(tmp_path)


def test_live_load_converges_and_honours_a_retraction(tmp_path: Path) -> None:
    pytest.importorskip("qdrant_client", reason="install latence-sink-qdrant[client]")
    from qdrant_client import QdrantClient

    collection = f"latence-smoke-{uuid.uuid4().hex[:8]}"
    client = QdrantClient(url=QDRANT_URL)
    try:
        config = {"collection": collection, "url": QDRANT_URL}

        first = QdrantSink(config).load(_export(tmp_path / "v1", ["c1", "c2", "c3"]))
        assert (first.created_collection, first.upserted, first.deleted) == (True, 3, 0)

        again = QdrantSink(config).load(_export(tmp_path / "v1", ["c1", "c2", "c3"]))
        assert (again.upserted, again.unchanged, again.deleted) == (0, 3, 0)

        after = QdrantSink(config).load(_export(tmp_path / "v2", ["c1", "c3"]))
        assert after.deleted == 1
        assert QdrantSink(config).verify_absent(["c2"]) == []
        assert QdrantSink(config).verify_absent(["c1"]) == ["c1"]
    finally:
        client.delete_collection(collection)
