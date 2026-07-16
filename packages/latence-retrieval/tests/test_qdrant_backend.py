"""The reference :class:`QdrantBackend`, tested at the :class:`RetrievalBackend` seam (#173).

Two boundaries, both offline (no ``qdrant-client`` needed — ADR-0016/0049):

* the **adapter guards + config validation** — asserted through the public interface (``search`` in,
  typed errors out; construction rejections), proving the adapter rejects *exactly* what a real store
  rejects — an undeclared mode/filter, a wrong-width dense vector — and does so **before the wire**
  (no import of the client), never a silent empty result;
* the **translation + result mapping** — driven through a faithful fake ``qdrant_client`` module
  (the same honest fake-module boundary the rerank/pack tests use): the adapter turns one
  ``RetrievalQuery`` into a single ``query_points`` call (dense vector / sparse ``SparseVector`` +
  ``using``, an exact-equality ``Filter``, ``limit``) and maps each ``ScoredPoint`` back to a
  ``Candidate`` — asserted on the observable output, never the internal translation, so the tests
  survive a refactor of *how* the query is built.

Anti-false-green: the fake ``qdrant_client`` mirrors the real ``query_points`` contract — an explicit
kwarg signature (no ``**kwargs`` swallow, so a kwarg the real API lacks would fail here too), a
``.points`` list of ``ScoredPoint``-likes, descending-score order truncated to ``limit``, an
exact-per-field ``must`` filter (missing field ⇒ excluded), and a raise on an unknown collection —
and re-implements none of Qdrant's ANN maths (faithful to the *contract*, not the semantics, exactly
as ``FakeBackend`` is). A real-store smoke test is present and gated on the dep + a URL env var
(skipped offline; live-store validation is out of band per the AC).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any

import pytest
from latence_retrieval import (
    BackendCapabilityError,
    BackendDimensionError,
    BackendSearchError,
    Candidate,
    Capability,
    QdrantBackend,
    RetrievalBackend,
    RetrievalQuery,
)

# --- the faithful fake qdrant_client module ---------------------------------------------------


class _MatchValue:
    def __init__(self, value: Any) -> None:
        self.value = value


class _FieldCondition:
    def __init__(self, key: str, match: _MatchValue) -> None:
        self.key = key
        self.match = match


class _Filter:
    def __init__(self, must: list[_FieldCondition] | None = None) -> None:
        self.must = must or []


class _SparseVector:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class _ScoredPoint:
    def __init__(self, id: Any, score: float, payload: dict[str, Any] | None) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class _QueryResponse:
    def __init__(self, points: list[_ScoredPoint]) -> None:
        self.points = points


def _match(payload: dict[str, Any], query_filter: _Filter | None) -> bool:
    if query_filter is None:
        return True
    return all(payload.get(c.key) == c.match.value for c in query_filter.must)


class _FakeQdrantClient:
    """A faithful fake ``QdrantClient``: mirrors the ``query_points`` contract, not Qdrant's ANN."""

    def __init__(self, store: dict[str, list[_ScoredPoint]] | None = None, **conn: Any) -> None:
        self.store = store if store is not None else {}
        self.conn = conn
        self.calls: list[dict[str, Any]] = []

    # Explicit signature (no **kwargs): a kwarg the real query_points lacks would raise here too.
    def query_points(
        self,
        collection_name: str,
        query: Any,
        using: str | None = None,
        query_filter: _Filter | None = None,
        limit: int = 10,
        with_payload: bool = False,
    ) -> _QueryResponse:
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "using": using,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        if collection_name not in self.store:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        points = [p for p in self.store[collection_name] if _match(p.payload or {}, query_filter)]
        points.sort(key=lambda p: p.score, reverse=True)
        return _QueryResponse(points[:limit])


def _make_models_module() -> types.ModuleType:
    models_mod = types.ModuleType("qdrant_client.models")
    models_mod.Filter = _Filter  # type: ignore[attr-defined]
    models_mod.FieldCondition = _FieldCondition  # type: ignore[attr-defined]
    models_mod.MatchValue = _MatchValue  # type: ignore[attr-defined]
    models_mod.SparseVector = _SparseVector  # type: ignore[attr-defined]
    return models_mod


def _install_fake_qdrant(
    monkeypatch: pytest.MonkeyPatch, store: dict[str, list[_ScoredPoint]]
) -> list[_FakeQdrantClient]:
    """Install a faithful fake ``qdrant_client`` and return the list of clients it constructs."""
    constructed: list[_FakeQdrantClient] = []
    models_mod = _make_models_module()

    class _BoundClient(_FakeQdrantClient):
        def __init__(self, **conn: Any) -> None:
            super().__init__(store=store, **conn)
            constructed.append(self)

    module = types.ModuleType("qdrant_client")
    module.QdrantClient = _BoundClient  # type: ignore[attr-defined]
    module.models = models_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models_mod)
    return constructed


_CORPUS = {
    "docs": [
        _ScoredPoint(id=1, score=0.9, payload={"text": "alpha", "tenant": "acme", "year": 2024}),
        _ScoredPoint(id=2, score=0.5, payload={"text": "bravo", "tenant": "acme", "year": 2023}),
        _ScoredPoint(id=3, score=0.7, payload={"text": "charlie", "tenant": "globex"}),
    ]
}


def _dense_backend(**over: Any) -> QdrantBackend:
    cfg: dict[str, Any] = {
        "collection": "docs",
        "capabilities": ["dense", "filter"],
        "dimension": 3,
    }
    cfg.update(over)
    return QdrantBackend(cfg)


# --- seam identity ----------------------------------------------------------------------------


def test_qdrant_backend_satisfies_the_retrieval_backend_seam() -> None:
    # runtime_checkable Protocol: the real adapter IS a RetrievalBackend (its second implementation).
    assert isinstance(_dense_backend(), RetrievalBackend)


def test_declares_only_its_configured_capabilities() -> None:
    assert _dense_backend().capabilities == frozenset({Capability.DENSE, Capability.FILTER})


# --- config validation (construction never touches the wire) ----------------------------------


def test_requires_a_collection() -> None:
    with pytest.raises(ValueError, match="collection"):
        QdrantBackend({"capabilities": ["dense"], "dimension": 3})


def test_requires_non_empty_capabilities() -> None:
    with pytest.raises(ValueError, match="capabilities"):
        QdrantBackend({"collection": "docs", "capabilities": []})


def test_rejects_an_unknown_capability_name() -> None:
    with pytest.raises(ValueError, match="unknown capability"):
        QdrantBackend({"collection": "docs", "capabilities": ["dense", "nonsense"]})


@pytest.mark.parametrize("unsupported", ["bm25", "multivector"])
def test_rejects_a_capability_this_reference_adapter_cannot_serve(unsupported: str) -> None:
    # Honest declaration: a backend must not advertise a mode it cannot faithfully translate.
    with pytest.raises(ValueError, match="does not serve"):
        QdrantBackend({"collection": "docs", "capabilities": [unsupported]})


def test_dense_capability_requires_a_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        QdrantBackend({"collection": "docs", "capabilities": ["dense"]})


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "3"])
def test_rejects_a_bad_dimension(bad: Any) -> None:
    with pytest.raises(ValueError, match="dimension"):
        QdrantBackend({"collection": "docs", "capabilities": ["dense"], "dimension": bad})


def test_sparse_backend_needs_no_dimension() -> None:
    backend = QdrantBackend({"collection": "docs", "capabilities": ["sparse"]})
    assert backend.capabilities == frozenset({Capability.SPARSE})


def test_rejects_an_unknown_config_key() -> None:
    with pytest.raises(ValueError, match="unknown config key"):
        QdrantBackend({"collection": "docs", "capabilities": ["dense"], "dimension": 3, "oops": 1})


@pytest.mark.parametrize("key", ["vector_name", "sparse_vector_name", "text_key"])
def test_rejects_a_bad_string_option(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _dense_backend(**{key: 123})


# --- guards fire BEFORE the wire (no client import needed) -------------------------------------


def test_rejects_undeclared_capability_without_touching_the_store() -> None:
    backend = QdrantBackend({"collection": "docs", "capabilities": ["dense"], "dimension": 3})
    with pytest.raises(BackendCapabilityError, match="sparse"):
        backend.search(RetrievalQuery(mode=Capability.SPARSE, sparse_terms={"1": 0.5}))


def test_rejects_filter_when_filter_not_declared() -> None:
    backend = QdrantBackend({"collection": "docs", "capabilities": ["dense"], "dimension": 3})
    with pytest.raises(BackendCapabilityError, match="filter"):
        backend.search(
            RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), filter={"tenant": "acme"})
        )


def test_rejects_a_wrong_dimension_vector() -> None:
    with pytest.raises(BackendDimensionError, match="width 2 but the index dimension is 3"):
        _dense_backend().search(RetrievalQuery(mode=Capability.DENSE, vector=(0.1, 0.2)))


# --- dense search behaviour (through the faithful fake module) --------------------------------


def test_dense_search_returns_hits_in_descending_score_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch, _CORPUS)
    hits = _dense_backend().search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))
    assert [c.id for c in hits] == ["1", "3", "2"]  # ids stringified, store-sorted by score


def test_dense_search_maps_payload_to_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant(monkeypatch, _CORPUS)
    hits = _dense_backend().search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))
    top = hits[0]
    assert top.id == "1"
    assert top.score == pytest.approx(0.9)
    assert top.text == "alpha"  # text_key extracted…
    assert top.metadata == {"tenant": "acme", "year": 2024}  # …and dropped from metadata


def test_dense_search_honours_top_k_as_the_store_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _install_fake_qdrant(monkeypatch, _CORPUS)
    hits = _dense_backend().search(
        RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), top_k=2)
    )
    assert [c.id for c in hits] == ["1", "3"]
    assert clients[0].calls[0]["limit"] == 2  # top_k threaded to query_points(limit=…)


def test_dense_search_translates_the_filter_to_an_exact_equality_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch, _CORPUS)
    hits = _dense_backend().search(
        RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), filter={"tenant": "acme"})
    )
    assert [c.id for c in hits] == ["1", "2"]  # only acme rows, still score-ordered


def test_filter_excludes_a_candidate_missing_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant(monkeypatch, _CORPUS)
    # 'charlie' (id 3) has no 'year' → excluded by a year predicate, as a store's filter would.
    hits = _dense_backend().search(
        RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0), filter={"year": 2024})
    )
    assert [c.id for c in hits] == ["1"]


def test_dense_search_passes_the_configured_vector_name(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _install_fake_qdrant(monkeypatch, _CORPUS)
    _dense_backend(vector_name="dense_vec").search(
        RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0))
    )
    call = clients[0].calls[0]
    assert call["using"] == "dense_vec"
    assert call["query"] == [0.0, 0.0, 0.0]  # the raw dense vector
    assert call["with_payload"] is True


def test_missing_payload_yields_empty_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    store = {"docs": [_ScoredPoint(id="x", score=1.0, payload=None)]}
    _install_fake_qdrant(monkeypatch, store)
    (only,) = _dense_backend().search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))
    assert only.text == ""
    assert only.metadata == {}


# --- sparse search behaviour ------------------------------------------------------------------


def test_sparse_search_builds_a_sparse_vector_from_the_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients = _install_fake_qdrant(monkeypatch, _CORPUS)
    backend = QdrantBackend(
        {"collection": "docs", "capabilities": ["sparse"], "sparse_vector_name": "splade"}
    )
    hits = backend.search(RetrievalQuery(mode=Capability.SPARSE, sparse_terms={"7": 0.4, "12": 1.5}))
    assert [c.id for c in hits] == ["1", "3", "2"]
    call = clients[0].calls[0]
    assert call["using"] == "splade"
    sparse = call["query"]
    assert isinstance(sparse, _SparseVector)
    assert sparse.indices == [7, 12]  # string keys → integer token ids (ADR-0053)
    assert sparse.values == [0.4, 1.5]


def test_sparse_search_rejects_a_non_integer_term(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant(monkeypatch, _CORPUS)
    backend = QdrantBackend({"collection": "docs", "capabilities": ["sparse"]})
    with pytest.raises(ValueError, match="not an integer index"):
        backend.search(RetrievalQuery(mode=Capability.SPARSE, sparse_terms={"revenue": 0.9}))


# --- lazy client build + connection threading -------------------------------------------------


def test_builds_the_real_client_lazily_from_connection_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No injected client: construction must NOT build one (no client yet)…
    clients = _install_fake_qdrant(monkeypatch, _CORPUS)
    backend = _dense_backend(url="http://localhost:6333", api_key="secret", prefer_grpc=True)
    assert clients == []  # constructing the backend opened no connection
    # …the client is built on first search, with the connection kwargs threaded verbatim.
    backend.search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))
    assert len(clients) == 1
    assert clients[0].conn == {
        "url": "http://localhost:6333",
        "api_key": "secret",  # pragma: allowlist secret
        "prefer_grpc": True,
    }


def test_an_injected_client_is_reused_and_no_new_one_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch, {})  # module present for `models`, but store is empty
    injected = _FakeQdrantClient(store=_CORPUS)
    backend = QdrantBackend(
        {"collection": "docs", "capabilities": ["dense"], "dimension": 3}, client=injected
    )
    hits = backend.search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))
    assert [c.id for c in hits] == ["1", "3", "2"]
    assert injected.calls  # the injected client was the one queried


# --- native store faults + missing dependency -------------------------------------------------


def test_unknown_collection_maps_to_a_typed_backend_search_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_qdrant(monkeypatch, {})  # no 'docs' collection in the store
    with pytest.raises(BackendSearchError, match="qdrant search failed"):
        _dense_backend().search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))


def test_native_fault_message_leaks_no_raw_store_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_qdrant(monkeypatch, {})
    try:
        _dense_backend(api_key="super-secret-key").search(  # pragma: allowlist secret
            RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0))
        )
    except BackendSearchError as exc:
        assert "super-secret-key" not in str(exc)
        assert "does not exist" not in str(exc)  # no raw store text, only the exception type name
    else:  # pragma: no cover
        pytest.fail("expected a BackendSearchError")


def test_missing_dependency_raises_a_clear_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(ImportError, match=r"latence-retrieval\[backend-qdrant\]"):
        _dense_backend().search(RetrievalQuery(mode=Capability.DENSE, vector=(0.0, 0.0, 0.0)))


# --- real-store boundary (gated on the dep + a URL env var; skipped offline) ------------------


@pytest.mark.skipif(
    importlib.util.find_spec("qdrant_client") is None
    or not os.environ.get("LATENCE_QDRANT_URL"),
    reason="live Qdrant validation is out of band (set LATENCE_QDRANT_URL + install the extra)",
)
def test_live_qdrant_smoke() -> None:  # pragma: no cover - runs only against a live store
    """With the real client + a live collection, dense search returns candidates.

    Skipped offline (the AC: live-store validation is done out of band). ``LATENCE_QDRANT_URL`` and
    ``LATENCE_QDRANT_COLLECTION`` name a reachable store; nothing is hardcoded.
    """
    url = os.environ["LATENCE_QDRANT_URL"]
    collection = os.environ.get("LATENCE_QDRANT_COLLECTION", "latence_smoke")
    dimension = int(os.environ.get("LATENCE_QDRANT_DIM", "384"))
    backend = QdrantBackend(
        {
            "collection": collection,
            "capabilities": ["dense", "filter"],
            "dimension": dimension,
            "url": url,
        }
    )
    hits = backend.search(
        RetrievalQuery(mode=Capability.DENSE, vector=tuple([0.0] * dimension), top_k=3)
    )
    assert all(isinstance(c, Candidate) for c in hits)
