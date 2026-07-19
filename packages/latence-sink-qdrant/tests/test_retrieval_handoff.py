"""The sink→backend handoff: a collection this sink loads is queryable by the shipped backend.

The two halves of the same product meet here. :class:`~latence_sink_qdrant.sink.QdrantSink` (#183)
*writes* a collection out of the Export; :class:`latence_retrieval.QdrantBackend` (#173) *reads*
that exact collection back as :class:`~latence_retrieval.Candidate`\\ s. Neither package's own
suite can catch a disagreement about the payload key the chunk text lives under — the sink never
searches (ADR-0048) and the backend never loads — so the contract between them is asserted here,
end to end: a **real** Export written by the framework's own ``JsonlParquetExport``, loaded by the
real sink into the shipped :class:`~latence_sink_qdrant.testing.FakeQdrantClient`, and then the
**verbatim stored payload** handed to a **default-configured** ``QdrantBackend``.

The regression it pins: the Export's text column is ``content``, so that is the payload key the
sink writes; a backend defaulting to some *other* key would return every Candidate with
``text == ""`` — a green retrieval over a collection that carries the text perfectly well, which
then silently starves every downstream rerank/pack/BM25 processor (they all score on
``Candidate.text``).
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from latence_retrieval import (
    BackendSearchError,
    Capability,
    QdrantBackend,
    RetrievalQuery,
)
from latence_sink_qdrant import testing as sink_testing
from latence_sink_qdrant.testing import FakeQdrantClient
from test_qdrant_sink import (  # the sibling suite's REAL-Export fixtures, reused verbatim
    _COLLECTION,
    _DIMENSION,
    sink,
    stored_payload,
    write_export,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _qdrant_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the (unimported, extra-gated) ``qdrant_client.models`` the backend loads.

    The model shapes are the SHIPPED :mod:`latence_sink_qdrant.testing` ones, which mirror the real
    ``qdrant_client.models`` — so no hand-rolled stub can drift from the client's contract here.
    """
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = sink_testing.FakeQdrantClient  # type: ignore[attr-defined]
    module.models = sink_testing  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", sink_testing)


class _ScoredPoint:
    """A ``qdrant_client`` ``ScoredPoint``-like: the three fields the backend reads."""

    def __init__(self, id: Any, score: float, payload: dict[str, Any] | None) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class _QueryResponse:
    def __init__(self, points: list[_ScoredPoint]) -> None:
        self.points = points


class _SearchingClient:
    """A minimal read side for the loaded collection: ``query_points`` over stored payloads.

    It re-implements no ANN maths (it returns the stored points in insertion order); the point of
    this fake is that the payloads it serves are the ones the **real sink** actually wrote.
    """

    def __init__(self, points: list[_ScoredPoint]) -> None:
        self._points = points

    def query_points(
        self,
        collection_name: str,
        query: Any,
        using: str | None = None,
        query_filter: Any | None = None,
        limit: int = 10,
        with_payload: bool = False,
    ) -> _QueryResponse:
        return _QueryResponse(self._points[:limit])


def _loaded_points(tmp_path: Path, records: dict[str, str]) -> list[_ScoredPoint]:
    """Load a real Export through the real sink and return its stored points, payloads verbatim."""
    client = FakeQdrantClient()
    export = write_export(tmp_path, records)
    report = sink(client).load(export)
    assert report.upserted == len(records)
    return [
        _ScoredPoint(id=str(rid), score=1.0, payload=stored_payload(client, rid))
        for rid in records
    ]


def _backend(points: list[_ScoredPoint], **over: Any) -> QdrantBackend:
    cfg: dict[str, Any] = {
        "collection": _COLLECTION,
        "capabilities": ["dense"],
        "dimension": _DIMENSION,
    }
    cfg.update(over)
    return QdrantBackend(cfg, client=_SearchingClient(points))


def _dense_query() -> RetrievalQuery:
    return RetrievalQuery(mode=Capability.DENSE, vector=tuple([0.0] * _DIMENSION))


def test_default_backend_reads_the_text_the_sink_wrote(tmp_path: Path) -> None:
    """DEFAULT config, both halves shipped: the chunk text survives the handoff."""
    points = _loaded_points(tmp_path, {"c1": "alpha text", "c2": "beta text"})

    hits = _backend(points).search(_dense_query())

    assert {c.text for c in hits} == {"alpha text", "beta text"}
    # …and the text is not duplicated into the metadata it was lifted out of.
    assert all("content" not in c.metadata for c in hits)
    # the rest of the Export's columns still ride through as opaque provenance
    assert all(c.metadata["record_id"] in {"c1", "c2"} for c in hits)


def test_a_wrong_text_key_is_loud_not_an_empty_string(tmp_path: Path) -> None:
    """A configured key that the payload does not carry is a typed error, never silent "" (ADR-0034).

    Without this, a mis-keyed backend returns a full page of Candidates whose ``text`` is empty and
    the pipeline reports a green retrieval; the reranker scores nothing and the Packer packs
    nothing.
    """
    points = _loaded_points(tmp_path, {"c1": "alpha text"})

    with pytest.raises(BackendSearchError, match="text_key"):
        _backend(points, text_key="text").search(_dense_query())
