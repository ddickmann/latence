"""A faithful in-memory Qdrant client — the offline test surface for the sink.

**The rule this file exists to obey**: a fake that accepts what the real dependency *rejects*
manufactures false green. So :class:`FakeQdrantClient` is written from the rejection side first. It
enforces, with the same HTTP status codes a real Qdrant returns:

======  ==========================================================================================
status  what the real store rejects, and this fake rejects too
======  ==========================================================================================
403     any call when the api key does not match the one the server expects
404     any operation on a collection that does not exist
409     ``create_collection`` on a collection that already exists
400     a point id that is neither an unsigned integer nor a UUID — **a raw content-addressed**
        ``record_id`` **string is not a legal Qdrant point id**; a vector whose width is not the
        collection's dimension; a named-vector collection given an unnamed vector (or vice versa,
        or an unknown vector name); a payload that is not a JSON-serialisable string-keyed object
======  ==========================================================================================

It is faithful to the *contract*, not to Qdrant's internals: it stores points in a dict and it
re-implements none of the HNSW/ANN machinery (there is no search here — the sink never searches,
ADR-0048). Its method signatures are explicit — no ``**kwargs`` swallow — so a kwarg the real client
does not take would fail here too.

The filter subset it implements is exactly the one the sink uses: ``must`` / ``must_not`` lists of
``FieldCondition(key, MatchValue(value))``, ANDed, with real Qdrant's semantics that a point
*lacking* the key does not match the condition (so it survives a ``must``, and satisfies a
``must_not``) — which is what makes the reconciling sweep also collect points written by an older
loader that never stamped the marker at all.

Adopters can use this class too: it is shipped, not test-local, so a downstream project that wires
the sink into its own pipeline can test that wiring with no Qdrant running.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


class FakeUnexpectedResponse(Exception):
    """The fake's analogue of ``qdrant_client.http.exceptions.UnexpectedResponse``.

    Carries a ``status_code`` because that is the only part of the native fault the sink reads (to
    tell an auth rejection from any other store fault); everything else about it stays opaque, just
    as the sink treats the real exception.
    """

    def __init__(self, status_code: int, reason: str) -> None:
        super().__init__(f"Unexpected Response: {status_code} ({reason})")
        self.status_code = status_code
        self.reason = reason


# --- the model shapes (mirroring ``qdrant_client.models``) ---------------------------------------


class Distance:
    """Qdrant's distance metrics, with their wire values."""

    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"
    MANHATTAN = "Manhattan"


#: The exact set of wire values ``VectorParams.distance`` may carry.
_DISTANCE_VALUES = frozenset(
    {Distance.COSINE, Distance.DOT, Distance.EUCLID, Distance.MANHATTAN}
)


@dataclass(frozen=True)
class VectorParams:
    """A vector index definition: fixed width + metric."""

    size: int
    distance: str


@dataclass(frozen=True)
class PointStruct:
    """One point to upsert: id, vector, payload."""

    id: Any
    vector: Any
    payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class MatchValue:
    """Exact-value match on a payload field."""

    value: Any


@dataclass(frozen=True)
class FieldCondition:
    """``payload[key] == match.value``."""

    key: str
    match: MatchValue


@dataclass(frozen=True)
class Filter:
    """A conjunction of conditions, and/or a conjunction of negated conditions."""

    must: list[FieldCondition] = field(default_factory=list)
    must_not: list[FieldCondition] = field(default_factory=list)


@dataclass(frozen=True)
class FilterSelector:
    """Select points to delete by filter (rather than by explicit id list)."""

    filter: Filter


@dataclass(frozen=True)
class PointIdsList:
    """Select points to delete by explicit id list."""

    points: list[Any]


@dataclass(frozen=True)
class Record:
    """A retrieved point (Qdrant's ``Record``)."""

    id: Any
    payload: dict[str, Any] | None = None
    vector: Any | None = None


@dataclass(frozen=True)
class CountResult:
    """The result of ``count``."""

    count: int


class _Models:
    """The ``qdrant_client.models`` namespace, as far as the sink uses it."""

    Distance = Distance
    VectorParams = VectorParams
    PointStruct = PointStruct
    MatchValue = MatchValue
    FieldCondition = FieldCondition
    Filter = Filter
    FilterSelector = FilterSelector
    PointIdsList = PointIdsList
    Record = Record


MODELS = _Models()


# --- the client ----------------------------------------------------------------------------------


class FakeQdrantClient:
    """An in-memory Qdrant that rejects what a real Qdrant rejects.

    Args:
        api_key: the key this client presents (mirrors ``QdrantClient(api_key=...)``).
        server_api_key: the key the simulated server requires; when set and different from
            ``api_key``, **every** call raises a 403 — the real first-live-run failure mode.
        fail_on: a set of method names that raise a 500 the moment they are called, for driving the
            "the store died mid-load" path (the sink must then abort *before* deleting anything).

    Inspection surface for tests (all observations of the *store*, never of the sink's internals):
        :attr:`points` — ``{collection: {point_id: (vector, payload)}}``;
        :attr:`upserted_ids` — every point id passed to :meth:`upsert`, in order, across calls;
        :attr:`calls` — the method names invoked, in order.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        server_api_key: str | None = None,
        fail_on: set[str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._server_api_key = server_api_key
        #: Method names that raise a 500 on entry. Mutable, so a test can make a *healthy* store
        #: start failing mid-run — the only way to exercise "the store died half way through".
        self.fail_on = set(fail_on or ())
        self._config: dict[str, VectorParams | dict[str, VectorParams]] = {}
        self.points: dict[str, dict[Any, tuple[Any, dict[str, Any]]]] = {}
        self.upserted_ids: list[Any] = []
        self.calls: list[str] = []

    @property
    def models(self) -> _Models:
        """The model shapes this client understands (how the sink resolves them offline)."""
        return MODELS

    # -- collection lifecycle ---------------------------------------------------------------

    def collection_exists(self, collection_name: str) -> bool:
        self._enter("collection_exists")
        return collection_name in self.points

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | dict[str, VectorParams],
    ) -> bool:
        self._enter("create_collection")
        if collection_name in self.points:
            raise FakeUnexpectedResponse(409, f"Collection `{collection_name}` already exists!")
        if isinstance(vectors_config, VectorParams):
            self._check_params(vectors_config)
        elif isinstance(vectors_config, dict) and vectors_config:
            for name, params in vectors_config.items():
                if not isinstance(name, str) or not name:
                    raise FakeUnexpectedResponse(400, "vector name must be a non-empty string")
                self._check_params(params)
        else:
            raise FakeUnexpectedResponse(400, "vectors_config must be VectorParams or a mapping")
        self._config[collection_name] = vectors_config
        self.points[collection_name] = {}
        return True

    @staticmethod
    def _check_params(params: object) -> None:
        if not isinstance(params, VectorParams):
            raise FakeUnexpectedResponse(400, "expected VectorParams")
        if not isinstance(params.size, int) or isinstance(params.size, bool) or params.size < 1:
            raise FakeUnexpectedResponse(400, "vector size must be a positive integer")
        if params.distance not in _DISTANCE_VALUES:
            raise FakeUnexpectedResponse(400, f"unknown distance {params.distance!r}")

    # -- points -----------------------------------------------------------------------------

    def upsert(
        self,
        collection_name: str,
        points: list[PointStruct],
        wait: bool = True,
        ordering: str | None = None,
    ) -> bool:
        self._enter("upsert")
        store = self._collection(collection_name)
        config = self._config[collection_name]
        if not isinstance(points, list) or not points:
            raise FakeUnexpectedResponse(400, "points must be a non-empty list")
        staged: list[tuple[Any, Any, dict[str, Any]]] = []
        for point in points:
            if not isinstance(point, PointStruct):
                raise FakeUnexpectedResponse(400, "each point must be a PointStruct")
            pid = _validate_point_id(point.id)
            vector = _validate_vector(point.vector, config)
            payload = _validate_payload(point.payload)
            staged.append((pid, vector, payload))
        # All-or-nothing per request, like a real batch upsert: a rejected point does not leave the
        # earlier points of the same request applied.
        for pid, vector, payload in staged:
            store[pid] = (vector, payload)
            self.upserted_ids.append(pid)
        return True

    def retrieve(
        self,
        collection_name: str,
        ids: list[Any],
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> list[Record]:
        self._enter("retrieve")
        store = self._collection(collection_name)
        found: list[Record] = []
        for raw in ids:
            pid = _validate_point_id(raw)
            if pid not in store:
                continue  # a real retrieve silently omits ids it does not hold
            vector, payload = store[pid]
            if with_payload is True:
                shown: dict[str, Any] | None = dict(payload)
            elif with_payload is False:
                shown = None
            else:
                shown = {k: payload[k] for k in with_payload if k in payload}
            found.append(Record(id=pid, payload=shown, vector=vector if with_vectors else None))
        return found

    def set_payload(
        self,
        collection_name: str,
        payload: dict[str, Any],
        points: list[Any],
        wait: bool = True,
        ordering: str | None = None,
    ) -> bool:
        self._enter("set_payload")
        store = self._collection(collection_name)
        patch = _validate_payload(payload)
        for raw in points:
            pid = _validate_point_id(raw)
            if pid not in store:
                continue  # a real set_payload on an unknown id is a no-op, not an error
            vector, existing = store[pid]
            store[pid] = (vector, {**existing, **patch})
        return True

    def delete(
        self,
        collection_name: str,
        points_selector: FilterSelector | PointIdsList,
        wait: bool = True,
        ordering: str | None = None,
    ) -> bool:
        self._enter("delete")
        store = self._collection(collection_name)
        if isinstance(points_selector, FilterSelector):
            doomed = [
                pid
                for pid, (_, payload) in store.items()
                if _matches(payload, points_selector.filter)
            ]
        elif isinstance(points_selector, PointIdsList):
            doomed = [_validate_point_id(pid) for pid in points_selector.points]
        else:
            raise FakeUnexpectedResponse(
                400, "points_selector must be a FilterSelector or PointIdsList"
            )
        for pid in doomed:
            store.pop(pid, None)
        return True

    def count(
        self,
        collection_name: str,
        count_filter: Filter | None = None,
        exact: bool = True,
    ) -> CountResult:
        self._enter("count")
        store = self._collection(collection_name)
        if count_filter is None:
            return CountResult(count=len(store))
        return CountResult(
            count=sum(1 for _, payload in store.values() if _matches(payload, count_filter))
        )

    # -- internals --------------------------------------------------------------------------

    def _enter(self, method: str) -> None:
        self.calls.append(method)
        if self._server_api_key is not None and self._api_key != self._server_api_key:
            raise FakeUnexpectedResponse(403, "Invalid api-key")
        if method in self.fail_on:
            raise FakeUnexpectedResponse(500, f"simulated store failure during {method}")

    def _collection(self, name: str) -> dict[Any, tuple[Any, dict[str, Any]]]:
        if name not in self.points:
            raise FakeUnexpectedResponse(404, f"Collection `{name}` doesn't exist!")
        return self.points[name]


def _validate_point_id(raw: Any) -> Any:
    """Qdrant point ids are unsigned integers or UUIDs — nothing else.

    This is the single most important rejection in the file: the corpus's ``record_id`` is a
    content-addressed *string*, and a loader that handed it to Qdrant directly would fail against a
    real server. A fake that accepted it would hide exactly that bug.
    """
    if isinstance(raw, bool):
        raise FakeUnexpectedResponse(400, "value is not a valid point ID")
    if isinstance(raw, int):
        if raw < 0:
            raise FakeUnexpectedResponse(400, "point ID integers must be unsigned")
        return raw
    if isinstance(raw, str):
        try:
            return str(uuid.UUID(raw))
        except ValueError:
            raise FakeUnexpectedResponse(
                400,
                "value is not a valid point ID: expected an unsigned integer or a UUID string",
            ) from None
    raise FakeUnexpectedResponse(400, "value is not a valid point ID")


def _validate_vector(raw: Any, config: VectorParams | dict[str, VectorParams]) -> Any:
    """A vector must match the collection's declared shape: width, and named-vs-unnamed."""
    if isinstance(config, VectorParams):
        if isinstance(raw, dict):
            raise FakeUnexpectedResponse(
                400, "collection has a single unnamed vector; a named vector map was given"
            )
        values = _floats(raw)
        if len(values) != config.size:
            raise FakeUnexpectedResponse(
                400,
                f"Wrong input: Vector dimension error: expected dim: {config.size}, "
                f"got {len(values)}",
            )
        return values
    if not isinstance(raw, dict):
        raise FakeUnexpectedResponse(
            400, "collection has named vectors; an unnamed vector was given"
        )
    out: dict[str, list[float]] = {}
    for name, values in raw.items():
        if name not in config:
            raise FakeUnexpectedResponse(
                400, f"Wrong input: Not existing vector name error: {name}"
            )
        floats = _floats(values)
        if len(floats) != config[name].size:
            raise FakeUnexpectedResponse(
                400,
                f"Wrong input: Vector dimension error: expected dim: {config[name].size}, "
                f"got {len(floats)}",
            )
        out[name] = floats
    missing = set(config) - set(out)
    if missing:
        raise FakeUnexpectedResponse(400, f"Wrong input: missing vector(s) {sorted(missing)}")
    return out


def _floats(raw: Any) -> list[float]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise FakeUnexpectedResponse(400, "vector must be a non-empty list of numbers")
    out: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FakeUnexpectedResponse(400, "vector must contain numbers only")
        out.append(float(value))
    return out


def _validate_payload(raw: Any) -> dict[str, Any]:
    """A payload is a JSON object with string keys — the wire cannot carry anything else."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise FakeUnexpectedResponse(400, "payload must be a JSON object")
    for key in raw:
        if not isinstance(key, str):
            raise FakeUnexpectedResponse(400, "payload keys must be strings")
    try:
        json.dumps(raw)
    except (TypeError, ValueError):
        raise FakeUnexpectedResponse(400, "payload must be JSON-serialisable") from None
    return dict(raw)


def _matches(payload: dict[str, Any], filt: Filter) -> bool:
    """Qdrant's ``must`` / ``must_not`` semantics over exact-value field conditions.

    A point that does not *have* the key does not match the condition — so it fails a ``must`` and
    passes a ``must_not``. That is real Qdrant behaviour, and the sink depends on it.
    """
    for cond in filt.must:
        if cond.key not in payload or payload[cond.key] != cond.match.value:
            return False
    return all(
        not (cond.key in payload and payload[cond.key] == cond.match.value)
        for cond in filt.must_not
    )


__all__ = [
    "MODELS",
    "CountResult",
    "Distance",
    "FakeQdrantClient",
    "FakeUnexpectedResponse",
    "FieldCondition",
    "Filter",
    "FilterSelector",
    "MatchValue",
    "PointIdsList",
    "PointStruct",
    "Record",
    "VectorParams",
]
