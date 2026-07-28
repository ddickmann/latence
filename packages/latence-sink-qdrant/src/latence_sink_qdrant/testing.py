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
        or an unknown vector name); a payload that is not a JSON-serialisable string-keyed object;
        a collection declaring no vector index at all; a **sparse** vector whose ``indices`` and
        ``values`` are not parallel, whose indices are not unique non-negative integers, or whose
        values are not numbers; a sparse name colliding with a dense one; a dense list sent under
        a sparse name (or a ``SparseVector`` under a dense one); a sparse config that is not a
        *named* mapping — Qdrant has no unnamed sparse vector
======  ==========================================================================================

**Sparse vectors are modelled here for the same reason the point-id rejection is** (T-RET-1,
``issues/dogfood/qdrant-native-sparse.md``): before this, the class contained zero occurrences of
"sparse", so the suite was *structurally unable* to catch a sink that created dense-only
``VectorParams`` while the retrieval side already queried ``using=<sparse vector name>``. A fake that
cannot express a feature manufactures green just as surely as one that accepts what the real store
rejects — so the sparse surface is written from the rejection side first, exactly like the rest.

It is faithful to the *contract*, not to Qdrant's internals: it stores points in a dict and it
re-implements none of the HNSW/ANN machinery (there is no search here — the sink never searches,
ADR-0048). Its method signatures are explicit — no ``**kwargs`` swallow — so a kwarg the real client
does not take would fail here too.

The filter subset it implements is exactly the one the sink uses: ``must`` / ``must_not`` /
``should`` lists of ``FieldCondition(key, MatchValue(value))`` and
``IsEmptyCondition(PayloadField(key))``, with real Qdrant's semantics — the clauses are ANDed, a
non-empty ``should`` needs one match, a point *lacking* the key does not match a ``FieldCondition``
(so it survives a ``must`` and satisfies a ``must_not``) but *does* match ``IsEmpty``. That last
pair is what lets the reconciling sweep collect points no loader ever stamped while leaving points
another corpus explicitly owns alone.

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
class SparseIndexParams:
    """Storage/tuning knobs of a sparse index (mirrors ``models.SparseIndexParams``)."""

    on_disk: bool | None = None
    full_scan_threshold: int | None = None


@dataclass(frozen=True)
class SparseVectorParams:
    """A **named** sparse vector index definition.

    Deliberately width-less: a sparse index has no fixed dimension — that is the whole point of it —
    so there is no ``size`` to validate a vector against, only the shape of the vector itself.
    """

    index: SparseIndexParams | None = None
    modifier: str | None = None


@dataclass(frozen=True)
class SparseVector:
    """One sparse vector: parallel ``indices`` (term ids) and ``values`` (weights).

    Mutable list fields, so this is deliberately **not** hashable — matching
    ``qdrant_client.models.SparseVector``, which is a pydantic model, not a key. Equality is
    field-wise, which is what a test asserting a round-trip needs.
    """

    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)


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
class PayloadField:
    """A payload key, as ``IsEmptyCondition`` addresses it."""

    key: str


@dataclass(frozen=True)
class IsEmptyCondition:
    """``payload[key]`` is missing, null, or an empty list — Qdrant's "is empty"."""

    is_empty: PayloadField


@dataclass(frozen=True)
class Filter:
    """A conjunction of clauses: every ``must``, no ``must_not``, and — if given — some ``should``.

    Qdrant ANDs the clauses of one filter and requires *at least one* ``should`` condition to match
    when the list is non-empty (``min_should`` defaults to 1); an empty ``should`` list constrains
    nothing.
    """

    must: list[FieldCondition | IsEmptyCondition] = field(default_factory=list)
    must_not: list[FieldCondition | IsEmptyCondition] = field(default_factory=list)
    should: list[FieldCondition | IsEmptyCondition] = field(default_factory=list)


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


@dataclass(frozen=True)
class CollectionParams:
    """The vector shape a collection declares — what ``get_collection`` reports back.

    ``vectors`` mirrors what was passed to ``create_collection``: a bare :class:`VectorParams` for
    the single unnamed dense vector, or a ``{name: VectorParams}`` mapping (``{}`` for a
    sparse-only collection). ``sparse_vectors`` is always a mapping, because Qdrant sparse vectors
    are always named.
    """

    vectors: dict[str, VectorParams] | VectorParams
    sparse_vectors: dict[str, SparseVectorParams]


@dataclass(frozen=True)
class CollectionConfig:
    """``CollectionInfo.config`` — as far as anything here reads it."""

    params: CollectionParams


@dataclass(frozen=True)
class CollectionInfo:
    """The result of ``get_collection``: the store's own account of a collection's shape."""

    config: CollectionConfig


class _Models:
    """The ``qdrant_client.models`` namespace, as far as the sink uses it."""

    Distance = Distance
    VectorParams = VectorParams
    SparseVectorParams = SparseVectorParams
    SparseIndexParams = SparseIndexParams
    SparseVector = SparseVector
    PointStruct = PointStruct
    MatchValue = MatchValue
    FieldCondition = FieldCondition
    Filter = Filter
    FilterSelector = FilterSelector
    IsEmptyCondition = IsEmptyCondition
    PayloadField = PayloadField
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
        self._sparse_config: dict[str, dict[str, SparseVectorParams]] = {}
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
        vectors_config: VectorParams | dict[str, VectorParams] | None = None,
        sparse_vectors_config: dict[str, SparseVectorParams] | None = None,
    ) -> bool:
        """Declare a collection's dense and/or sparse indexes.

        ``vectors_config`` may now legitimately be an **empty mapping** — a sparse-only collection
        is a real Qdrant shape (a SPLADE index with no dense side) — but a collection declaring
        *neither* a dense nor a sparse index is a 400: it could never be queried, and accepting it
        would let a mis-wired loader create a collection that silently answers nothing.
        """
        self._enter("create_collection")
        if collection_name in self.points:
            raise FakeUnexpectedResponse(409, f"Collection `{collection_name}` already exists!")
        sparse = self._check_sparse_config(sparse_vectors_config)
        if isinstance(vectors_config, VectorParams):
            self._check_params(vectors_config)
            dense_names = {""}
        elif isinstance(vectors_config, dict):
            for name, params in vectors_config.items():
                if not isinstance(name, str) or not name:
                    raise FakeUnexpectedResponse(400, "vector name must be a non-empty string")
                self._check_params(params)
            dense_names = set(vectors_config)
        elif vectors_config is None:
            dense_names = set()
        else:
            raise FakeUnexpectedResponse(400, "vectors_config must be VectorParams or a mapping")

        if not dense_names and not sparse:
            raise FakeUnexpectedResponse(
                400, "a collection must declare at least one dense or sparse vector"
            )
        collision = dense_names & set(sparse)
        if collision:
            raise FakeUnexpectedResponse(
                400,
                f"vector name(s) {sorted(collision)} are declared both dense and sparse; one point "
                "carries one vector map, so a name cannot mean two indexes",
            )
        self._config[collection_name] = vectors_config if vectors_config is not None else {}
        self._sparse_config[collection_name] = sparse
        self.points[collection_name] = {}
        return True

    def get_collection(self, collection_name: str) -> CollectionInfo:
        """The store's own account of a collection's declared vector shape.

        The only way to ask a live Qdrant "does this collection actually carry the sparse index I
        am about to upsert into?" — which is what turns an otherwise-opaque 400 on the first upsert
        into a named error.
        """
        self._enter("get_collection")
        self._collection(collection_name)
        return CollectionInfo(
            config=CollectionConfig(
                params=CollectionParams(
                    vectors=self._config[collection_name],
                    sparse_vectors=dict(self._sparse_config[collection_name]),
                )
            )
        )

    @staticmethod
    def _check_sparse_config(raw: object) -> dict[str, SparseVectorParams]:
        """Sparse vectors are **always named** — there is no unnamed sparse vector in Qdrant."""
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise FakeUnexpectedResponse(
                400, "sparse_vectors_config must be a mapping of name -> SparseVectorParams"
            )
        for name, params in raw.items():
            if not isinstance(name, str) or not name:
                raise FakeUnexpectedResponse(400, "sparse vector name must be a non-empty string")
            if not isinstance(params, SparseVectorParams):
                raise FakeUnexpectedResponse(400, "expected SparseVectorParams")
        return dict(raw)

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
        sparse_config = self._sparse_config[collection_name]
        if not isinstance(points, list) or not points:
            raise FakeUnexpectedResponse(400, "points must be a non-empty list")
        staged: list[tuple[Any, Any, dict[str, Any]]] = []
        for point in points:
            if not isinstance(point, PointStruct):
                raise FakeUnexpectedResponse(400, "each point must be a PointStruct")
            pid = _validate_point_id(point.id)
            vector = _validate_vector(point.vector, config, sparse_config)
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

    def scroll(
        self,
        collection_name: str,
        scroll_filter: Filter | None = None,
        limit: int = 10,
        offset: Any | None = None,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
    ) -> tuple[list[Record], Any | None]:
        """Page through the collection in point-id order, like the real ``scroll``.

        Returns ``(records, next_page_offset)``; ``next_page_offset`` is the id the *next* call
        should start from, or ``None`` when the last page has been handed out. Ordering is by point
        id (Qdrant's own scroll order) so pagination is stable, and a page already handed out can be
        deleted without disturbing the offset of the pages still to come — which is exactly what the
        sink's sweep does.
        """
        self._enter("scroll")
        store = self._collection(collection_name)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise FakeUnexpectedResponse(400, "limit must be a positive integer")
        ordered = sorted(store.items(), key=lambda kv: str(kv[0]))
        if offset is not None:
            start = str(_validate_point_id(offset))
            ordered = [kv for kv in ordered if str(kv[0]) >= start]
        matched = [
            kv for kv in ordered if scroll_filter is None or _matches(kv[1][1], scroll_filter)
        ]
        page = matched[:limit]
        next_offset = matched[limit][0] if len(matched) > limit else None
        records = []
        for pid, (vector, payload) in page:
            if with_payload is True:
                shown: dict[str, Any] | None = dict(payload)
            elif with_payload is False:
                shown = None
            else:
                shown = {k: payload[k] for k in with_payload if k in payload}
            records.append(Record(id=pid, payload=shown, vector=vector if with_vectors else None))
        return records, next_offset

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


def _validate_vector(
    raw: Any,
    config: VectorParams | dict[str, VectorParams],
    sparse_config: dict[str, SparseVectorParams] | None = None,
) -> Any:
    """A vector must match the collection's declared shape: width, named-vs-unnamed, dense-vs-sparse.

    With no sparse index declared this is exactly what it always was — an unnamed collection takes a
    bare list and a named one takes a map — so a dense-only load's stored shape is untouched. With a
    sparse index declared the point carries **one** vector map spanning both kinds, and each name is
    checked against the kind it was declared as: a dense list under a sparse name, or a
    :class:`SparseVector` under a dense name, is the same 400 a real Qdrant returns.
    """
    sparse_config = sparse_config or {}
    if not sparse_config:
        if isinstance(config, VectorParams):
            if isinstance(raw, dict):
                raise FakeUnexpectedResponse(
                    400, "collection has a single unnamed vector; a named vector map was given"
                )
            return _dense(raw, config)
        if not isinstance(raw, dict):
            raise FakeUnexpectedResponse(
                400, "collection has named vectors; an unnamed vector was given"
            )
        out: dict[str, Any] = {}
        for name, values in raw.items():
            if name not in config:
                raise FakeUnexpectedResponse(
                    400, f"Wrong input: Not existing vector name error: {name}"
                )
            out[name] = _dense(values, config[name])
        _require_complete(out, set(config))
        return out

    dense_config = {"": config} if isinstance(config, VectorParams) else dict(config or {})
    if not isinstance(raw, dict):
        # The unnamed dense vector of a collection that ALSO has a sparse index is addressed by
        # its real name — the empty string — because the point's sparse half needs a name too.
        raise FakeUnexpectedResponse(
            400, "collection has named vectors; an unnamed vector was given"
        )
    mixed: dict[str, Any] = {}
    for name, values in raw.items():
        if name in dense_config:
            if isinstance(values, SparseVector):
                raise FakeUnexpectedResponse(
                    400, f"Wrong input: vector {name!r} is dense; a SparseVector was given"
                )
            mixed[name] = _dense(values, dense_config[name])
        elif name in sparse_config:
            mixed[name] = _sparse(values, name)
        else:
            raise FakeUnexpectedResponse(
                400, f"Wrong input: Not existing vector name error: {name}"
            )
    _require_complete(mixed, set(dense_config) | set(sparse_config))
    return mixed


def _require_complete(given: dict[str, Any], declared: set[str]) -> None:
    missing = declared - set(given)
    if missing:
        raise FakeUnexpectedResponse(400, f"Wrong input: missing vector(s) {sorted(missing)}")


def _dense(raw: Any, params: VectorParams) -> list[float]:
    values = _floats(raw)
    if len(values) != params.size:
        raise FakeUnexpectedResponse(
            400,
            f"Wrong input: Vector dimension error: expected dim: {params.size}, "
            f"got {len(values)}",
        )
    return values


def _sparse(raw: Any, name: str) -> SparseVector:
    """A sparse vector is two parallel lists — and every way they can disagree is a 400.

    An **empty** sparse vector is not one of them: a chunk whose redacted text has no surviving
    terms genuinely has no term weights, and rejecting it would make redaction a load failure.
    """
    if not isinstance(raw, SparseVector):
        raise FakeUnexpectedResponse(
            400, f"Wrong input: vector {name!r} is sparse; a SparseVector is required"
        )
    if not isinstance(raw.indices, (list, tuple)) or not isinstance(raw.values, (list, tuple)):
        raise FakeUnexpectedResponse(
            400, "sparse vector indices and values must both be lists of the same length"
        )
    if len(raw.indices) != len(raw.values):
        raise FakeUnexpectedResponse(
            400,
            f"Wrong input: sparse vector {name!r} has {len(raw.indices)} indices and "
            f"{len(raw.values)} values; they must be of the same length",
        )
    indices: list[int] = []
    for index in raw.indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise FakeUnexpectedResponse(400, "sparse vector indices must be integers")
        if index < 0:
            raise FakeUnexpectedResponse(
                400, "sparse vector indices must be non-negative integers"
            )
        indices.append(int(index))
    if len(set(indices)) != len(indices):
        raise FakeUnexpectedResponse(
            400, "sparse vector indices must be unique — a duplicate term id is ambiguous"
        )
    values: list[float] = []
    for value in raw.values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FakeUnexpectedResponse(400, "sparse vector values must be numbers")
        values.append(float(value))
    return SparseVector(indices=indices, values=values)


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


def _condition_matches(payload: dict[str, Any], cond: FieldCondition | IsEmptyCondition) -> bool:
    """One condition against one payload, with real Qdrant's missing-key semantics.

    A point that does not *have* the key does not match a :class:`FieldCondition` — so it fails a
    ``must`` and passes a ``must_not`` — and it *does* match an :class:`IsEmptyCondition`, which is
    exactly how the sink tells "a point nobody stamped" from "a point another corpus owns".
    """
    if isinstance(cond, IsEmptyCondition):
        value = payload.get(cond.is_empty.key)
        return value is None or value == []
    return cond.key in payload and payload[cond.key] == cond.match.value


def _matches(payload: dict[str, Any], filt: Filter) -> bool:
    """Qdrant's ``must`` / ``must_not`` / ``should`` semantics over one point's payload."""
    if not all(_condition_matches(payload, cond) for cond in filt.must):
        return False
    if any(_condition_matches(payload, cond) for cond in filt.must_not):
        return False
    return not filt.should or any(_condition_matches(payload, cond) for cond in filt.should)


__all__ = [
    "MODELS",
    "CollectionConfig",
    "CollectionInfo",
    "CollectionParams",
    "CountResult",
    "Distance",
    "FakeQdrantClient",
    "FakeUnexpectedResponse",
    "FieldCondition",
    "Filter",
    "FilterSelector",
    "IsEmptyCondition",
    "MatchValue",
    "PayloadField",
    "PointIdsList",
    "PointStruct",
    "Record",
    "SparseIndexParams",
    "SparseVector",
    "SparseVectorParams",
    "VectorParams",
]
