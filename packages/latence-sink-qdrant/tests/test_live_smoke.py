"""The live smoke path: the same load, against a real Qdrant.

Skipped unless you opt in — the offline suite is the merge gate (CPU-first, no services,
ADR-0007/0016), and this is the out-of-band validation that the faithful fake is telling the truth::

    docker run -p 6333:6333 qdrant/qdrant
    uv pip install 'qdrant-client>=1.10'
    LATENCE_QDRANT_URL=http://localhost:6333 uv run pytest packages/latence-sink-qdrant -k live

It asserts the *same* properties the offline tests assert — load, converge, make a retraction
verifiably absent, and (T-RET-1) **serve a sparse query off a named sparse vector** — through the
real client, real HTTP and a real collection, so a drift between the fake's contract and Qdrant's
shows up here as a failure rather than as a silent false green offline. It cleans its collections up
on the way in and out.

The sparse test is the one leg the fake cannot stand in for. The fake models the *shape* of a named
sparse vector and rejects what Qdrant rejects, but it implements no ANN and no scoring (there is no
search in it at all — the sink never searches, ADR-0048). Whether a real Qdrant actually *ranks* a
sparse query the way the load assumes is a claim only a real server can settle, so it is settled
here and not asserted offline.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import pytest
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.embedding import HashingEmbedder
from latence_core.signals import SparseHashingEmbedder
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from latence_sink_qdrant import QdrantSink, point_id

if TYPE_CHECKING:
    from pathlib import Path

QDRANT_URL = os.environ.get("LATENCE_QDRANT_URL")

# Two independent gates: the URL env var (opt-in) and the optional client dep (checked inside the
# test with importorskip, so a missing dep reads as "not installed", never as a failure).
pytestmark = pytest.mark.skipif(
    not QDRANT_URL, reason="set LATENCE_QDRANT_URL to run the live smoke"
)

_DIMENSION = 16
_VOCAB = 4096


def _records(texts: dict[str, str]) -> list[ChunkRecord]:
    return [
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
            content=text,
            token_count=max(1, len(text.split())),
        )
        for rid, text in texts.items()
    ]


def _export(tmp_path: Path, record_ids: list[str]) -> str:
    export = JsonlParquetExport(
        {"embedder": {"provider": "embedding.hashing", "config": {"dimension": _DIMENSION}}}
    )
    export.export(
        _records({rid: f"live text for {rid}" for rid in record_ids}), Storage(), str(tmp_path)
    )
    return str(tmp_path)


def _hybrid_export(tmp_path: Path, texts: dict[str, str]) -> str:
    """An Export carrying BOTH signals — the shape a ``vector: both`` load projects."""
    export = JsonlParquetExport(
        {
            "embedder": {"provider": "embedding.hashing", "config": {"dimension": _DIMENSION}},
            "sparse_embedder": {"provider": "sparse.hashing", "config": {"vocab_size": _VOCAB}},
        }
    )
    export.export(_records(texts), Storage(), str(tmp_path))
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


def test_live_sparse_query_returns_a_document_the_dense_query_does_not(tmp_path: Path) -> None:
    """T-RET-1 acceptance #3, against a real Qdrant binary.

    The claim under test is the one the whole ticket exists for: **the sparse leg can surface a
    document the dense leg does not return.** Before this change the collection had no sparse index
    at all, so ``query_points(..., using="sparse")`` could only ever answer nothing — and no offline
    test could say so, because the fake had no notion of a sparse vector.

    It is set up so the answer is deterministic rather than lucky. Two documents share no
    vocabulary. The dense query vector is the hashing embedding of the *second* document's exact
    text, so the dense leg's top hit is that document (cosine 1.0). The sparse query is the term
    weights of the *first* document's exact text, which the second document has zero overlap with —
    so the sparse leg returns the first and Qdrant does not return the second at all.
    """
    pytest.importorskip("qdrant_client", reason="install latence-sink-qdrant[client]")
    from qdrant_client import QdrantClient, models

    lexical = "quokka quokka wombat"
    semantic = "helvetica kerning ligature"

    collection = f"latence-sparse-{uuid.uuid4().hex[:8]}"
    client = QdrantClient(url=QDRANT_URL)
    try:
        export = _hybrid_export(tmp_path, {"lexical": lexical, "semantic": semantic})
        report = QdrantSink(
            {"collection": collection, "url": QDRANT_URL, "vector": "both"}
        ).load(export)
        assert (report.created_collection, report.upserted) == (True, 2)
        assert report.sparse_vector_name == "sparse"

        # 1. The real server agrees the collection carries the named sparse index.
        params = client.get_collection(collection).config.params
        assert set(params.sparse_vectors or {}) == {"sparse"}
        assert set(params.vectors or {}) == {"dense"}

        # 2. The dense leg's top hit is the document whose text the query vector IS.
        dense_query = next(HashingEmbedder({"dimension": _DIMENSION}).embed([semantic]))
        dense_hits = client.query_points(
            collection_name=collection, query=dense_query, using="dense", limit=1
        ).points
        assert [str(h.id) for h in dense_hits] == [point_id("semantic")]

        # 3. The sparse leg returns the OTHER document — the one dense did not return — and it is
        #    scored, i.e. the index is genuinely there and genuinely queried.
        terms = next(SparseHashingEmbedder({"vocab_size": _VOCAB}).encode_sparse([lexical]))
        sparse_hits = client.query_points(
            collection_name=collection,
            query=models.SparseVector(
                indices=list(terms.indices), values=[float(v) for v in terms.values]
            ),
            using="sparse",
            limit=10,
        ).points
        assert [str(h.id) for h in sparse_hits] == [point_id("lexical")]
        assert sparse_hits[0].score > 0
        assert point_id("lexical") not in {str(h.id) for h in dense_hits}

        # 4. And the term weights live in the VECTOR, not in the payload, where they were before.
        (stored,) = client.retrieve(
            collection_name=collection, ids=[point_id("lexical")], with_vectors=True
        )
        assert "sparse_indices" not in (stored.payload or {})
        assert sorted(stored.vector["sparse"].indices) == sorted(terms.indices)
    finally:
        client.delete_collection(collection)
