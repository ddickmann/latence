"""The one-shot Qdrant sink loader: an Export in, a converged collection out.

**What it is.** :class:`QdrantSink` is a *loader*, not an index. It reads the Export the pipeline
already wrote — the files that stay the source of truth (ADR-0017) — and projects it into a Qdrant
collection, then it is done. It holds nothing, caches nothing, and runs no search (ADR-0048): the
whole object is one public verb, :meth:`QdrantSink.load`, plus one auditing verb,
:meth:`QdrantSink.verify_absent`. Everything behind them — collection creation, point-id derivation,
digest-based change detection, bounded batching, the reconciling sweep, native-fault mapping — is
hidden work the caller never has to think about.

**Why the point id is derived and not the ``record_id``.** Qdrant point ids are unsigned integers or
UUIDs — it *rejects* an arbitrary string id. The corpus's ``record_id`` is a content-addressed
string, so the sink derives a deterministic UUIDv5 from it (:func:`point_id`) under a fixed
namespace. Deterministic derivation is what makes the load idempotent: the same record always lands
on the same point, so a re-run overwrites rather than duplicates. The ``record_id`` itself rides in
the payload, so nothing is lost and a filter can still address it.

**How a re-run converges (the three cases, all in one pass).**

* *unchanged* — the stored point's content digest equals the incoming one ⇒ **no upsert**. This is
  the AC's "Delta runs upsert only what changed", and it holds whether the Export handed to the sink
  is a full corpus or an incremental one (ADR-0018): the sink compares content, so it does not have
  to be told which mode it is in.
* *new or changed* — upserted at the derived id.
* *gone* — a record that was in the collection and is **not** in this Export. This is the
  Retraction/Purge case (ADR-0018/0029: after a retract or purge the pipeline re-exports the
  surviving corpus, and the deleted document's derived records are simply not in it). Upserting can
  never express a deletion, so the sink stamps every point it touched in this run with a unique
  ``_latence_load_id`` and then deletes, by filter, everything in the collection that is *not*
  stamped with it — and **verifies convergence by counting the stale set back to zero** before
  reporting success. That count is the "documented verification step": the load fails loudly rather
  than reporting a green load over a collection that still serves purged data.

Reconciling assumes the collection is this Export's projection (the natural, and default, setup).
Loading an *incremental* Export into a collection that also holds records the increment omits would
sweep them, so that case sets ``reconcile: false`` and the collection converges on the next full
load. Both are one config key, and the docstring of :meth:`load` says which to use when.

**Failure containment.** A store fault anywhere in the upsert phase aborts the load *before* the
sweep. A half-uploaded corpus therefore never deletes the records it did not reach — the worst case
is a collection with stale-but-present data, never one with silently-erased data.

The sweep is guarded for the same reason. Its filter is collection-*global* (``must_not(marker)``),
so it is the stale set only while nothing else has stamped the collection. Before deleting anything
the sink checks that exactly as many points carry its marker as it processed records; a load
running concurrently restamps those points and trips the check, which fails the *losing* load
loudly and deletes nothing rather than erasing a live index behind a green report.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_core.storage import Storage

from latence_sink_qdrant.errors import (
    SinkAuthError,
    SinkConfigError,
    SinkError,
    SinkExportError,
    SinkLoadError,
)
from latence_sink_qdrant.export_reader import ExportRow, read_export

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Payload key holding the content digest of the record as loaded — the change detector.
DIGEST_KEY = "_latence_digest"
#: Payload key stamped with the id of the load that last touched the point — the sweep marker.
LOAD_ID_KEY = "_latence_load_id"
#: Payload keys the sink owns. An Export column of the same name would be silently overwritten (and
#: would corrupt the reconcile marker), so it is rejected loudly instead.
RESERVED_PAYLOAD_KEYS = frozenset({DIGEST_KEY, LOAD_ID_KEY})

#: The default UUIDv5 namespace for deriving a point id from a ``record_id``. Fixed forever: a
#: change here would re-key every collection ever loaded. Override ``id_namespace`` (per corpus) to
#: share one collection between two corpora whose ``record_id`` spaces could collide.
DEFAULT_ID_NAMESPACE = uuid.UUID("6f6c1a4e-2b3d-5f7a-9c1e-4d2f8a0b6c31")

#: Distance metrics Qdrant's ``VectorParams`` accepts, lowercased. Validated at construction (no
#: client import needed) and mapped onto ``models.Distance`` when the collection is created.
_DISTANCES = frozenset({"cosine", "dot", "euclid", "manhattan"})

_CONNECTION_KEYS = frozenset(
    {"url", "host", "port", "grpc_port", "prefer_grpc", "https", "api_key", "timeout", "location"}
)
_SINK_KEYS = frozenset(
    {
        "collection",
        "basename",
        "batch_size",
        "distance",
        "vector_name",
        "reconcile",
        "load_id",
        "id_namespace",
    }
)


def point_id(record_id: str, namespace: uuid.UUID = DEFAULT_ID_NAMESPACE) -> str:
    """The deterministic Qdrant point id for a corpus ``record_id``.

    A UUIDv5 (SHA-1 over ``namespace + record_id``): stable across processes, machines and runs, so
    the *same* record always upserts onto the *same* point — the property the whole idempotency
    story rests on — and valid as a Qdrant point id, which an arbitrary corpus ``record_id`` string
    is not.
    """
    return str(uuid.uuid5(namespace, record_id))


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What one :meth:`QdrantSink.load` actually did — the numbers the CLI prints.

    Every field is an observed count, not an intention: ``upserted + unchanged`` is the number of
    Export records processed, and ``deleted`` is the number of stale points the reconciling sweep
    removed (0 on a first load, and exactly the retracted/purged set on a re-load after a deletion).
    """

    collection: str
    dimension: int
    load_id: str
    created_collection: bool = False
    upserted: int = 0
    unchanged: int = 0
    deleted: int = 0
    reconciled: bool = False

    @property
    def records(self) -> int:
        """Records read from the Export in this load."""
        return self.upserted + self.unchanged

    def summary(self) -> str:
        """One human line for the CLI (and for a run log)."""
        sweep = (
            f"deleted {self.deleted} stale point(s)"
            if self.reconciled
            else "reconcile disabled (no stale sweep)"
        )
        created = "created" if self.created_collection else "existing"
        return (
            f"collection {self.collection!r} ({created}, dim {self.dimension}): "
            f"{self.records} record(s) read — upserted {self.upserted}, "
            f"unchanged {self.unchanged}, {sweep} [load_id={self.load_id}]"
        )


@dataclass(frozen=True, slots=True)
class _Point:
    """One export row resolved to the exact shape a store call needs."""

    id: str
    record_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    digest: str = ""


class QdrantSink:
    """Load an Export into a Qdrant collection, idempotently and convergently.

    Config (all keys optional except ``collection``; an unknown key is a loud
    :class:`~latence_sink_qdrant.errors.SinkConfigError`, never a silently-ignored setting):

        collection: str — the target collection. **Required.**
        basename: str — the Export file stem (default ``"records"``, matching the Export Stage).
        batch_size: int — records per store round-trip (default 256). Bounds memory *and* the
            request size; the Export is streamed, so this is the only thing that grows with it.
        distance: str — ``"cosine"`` (default) / ``"dot"`` / ``"euclid"`` / ``"manhattan"``. Used
            **only** when the collection is created; an existing collection's metric is left alone.
        vector_name: str | None — the named vector to write (``None`` ⇒ the collection's single
            unnamed vector). Named *sparse* vectors are deliberately out of scope: the Export's
            ``sparse_indices`` / ``sparse_values`` columns ride through as payload, and sparse
            wiring is the engine's concern (#156).
        reconcile: bool — default ``True``. Sweep points not present in this Export (the
            Retraction/Purge path). Set ``False`` when loading an *incremental* Export into a
            collection that also holds records the increment omits.
        load_id: str | None — an audit **label** for the run. The marker stamped on every touched
            point is always unique per load (a reused marker would make the sweep a silent no-op);
            a label is prefixed onto it as ``"<label>:<nonce>"``, so a named run stays greppable in
            the payload and in the report. Omitted ⇒ the bare nonce.
        id_namespace: str — UUID namespace for point-id derivation (default
            :data:`DEFAULT_ID_NAMESPACE`).
        url / host / port / grpc_port / prefer_grpc / https / api_key / timeout / location —
            connection kwargs threaded verbatim to ``QdrantClient`` when it is built lazily.
            Ignored when a ``client`` is injected.

    Dependencies are accepted, never created: pass ``client`` (a pooled ``QdrantClient``, or the
    shipped :class:`~latence_sink_qdrant.testing.FakeQdrantClient`) and/or ``models`` to drive the
    sink offline. With neither, ``qdrant_client`` is imported lazily on the first store call — so
    constructing a sink, and validating its config, never needs the heavy dep or a connection.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        client: Any | None = None,
        models: Any | None = None,
    ) -> None:
        cfg = dict(config or {})
        unknown = set(cfg) - _SINK_KEYS - _CONNECTION_KEYS
        if unknown:
            msg = (
                f"QdrantSink: unknown config key(s) {sorted(unknown)}; "
                f"valid keys: {sorted(_SINK_KEYS | _CONNECTION_KEYS)}."
            )
            raise SinkConfigError(msg)

        collection = cfg.get("collection")
        if not isinstance(collection, str) or not collection:
            msg = "QdrantSink: 'collection' (a non-empty string) is required."
            raise SinkConfigError(msg)
        self._collection = collection

        self._basename = _require_name(cfg.get("basename", "records"), "basename")
        self._batch_size = _require_positive_int(cfg.get("batch_size", 256), "batch_size")

        distance = cfg.get("distance", "cosine")
        if not isinstance(distance, str) or distance.lower() not in _DISTANCES:
            msg = (
                f"QdrantSink 'distance' must be one of {sorted(_DISTANCES)}, got {distance!r}."
            )
            raise SinkConfigError(msg)
        self._distance = distance.lower()

        vector_name = cfg.get("vector_name")
        if vector_name is not None and (not isinstance(vector_name, str) or not vector_name):
            msg = (
                "QdrantSink 'vector_name' must be a non-empty string or omitted, "
                f"got {vector_name!r}."
            )
            raise SinkConfigError(msg)
        self._vector_name: str | None = vector_name

        reconcile = cfg.get("reconcile", True)
        if not isinstance(reconcile, bool):
            msg = f"QdrantSink 'reconcile' must be a bool, got {reconcile!r}."
            raise SinkConfigError(msg)
        self._reconcile = reconcile

        load_id = cfg.get("load_id")
        if load_id is not None and (not isinstance(load_id, str) or not load_id):
            msg = f"QdrantSink 'load_id' must be a non-empty string or omitted, got {load_id!r}."
            raise SinkConfigError(msg)
        self._configured_load_id: str | None = load_id

        namespace = cfg.get("id_namespace")
        if namespace is None:
            self._namespace = DEFAULT_ID_NAMESPACE
        else:
            try:
                self._namespace = uuid.UUID(str(namespace))
            except ValueError:
                msg = f"QdrantSink 'id_namespace' must be a UUID string, got {namespace!r}."
                raise SinkConfigError(msg) from None

        self._client = client
        self._models = models
        self._client_kwargs = {k: cfg[k] for k in _CONNECTION_KEYS if k in cfg}

    # -- public surface ---------------------------------------------------------------------

    def load(self, export_dir: str, storage: Storage | None = None) -> LoadReport:
        """Load the Export at ``export_dir`` into the collection and return what happened.

        The whole command in one call: create the collection if it is absent (at the Export's own
        vector width), stream the Export in bounded batches, upsert only the records whose content
        changed, and — unless ``reconcile`` is off — delete every point the Export no longer
        contains, verifying the stale count back to zero before returning.

        Raises :class:`~latence_sink_qdrant.errors.SinkExportError` if the Export is unusable (the
        source of truth is never repaired here), and
        :class:`~latence_sink_qdrant.errors.SinkLoadError` /
        :class:`~latence_sink_qdrant.errors.SinkAuthError` if the store faults — in which case the
        reconciling sweep has **not** run, so nothing was deleted.
        """
        store = storage or Storage.for_uri(export_dir)
        load_id = _load_marker(self._configured_load_id)
        batches = read_export(
            export_dir, store, basename=self._basename, batch_size=self._batch_size
        )

        dimension: int | None = None
        created = False
        upserted = 0
        unchanged = 0
        for rows in batches:
            points = [self._to_point(row, load_id) for row in rows]
            if dimension is None:
                dimension = len(points[0].vector)
                created = self._ensure_collection(dimension)
            changed, same = self._partition(points)
            if changed:
                self._upsert(changed)
                upserted += len(changed)
            if same:
                unchanged += len(same)
                if self._reconcile:
                    # Unchanged records are NOT re-upserted (that is the whole point), but they
                    # must still carry this run's sweep marker or the reconcile below would delete
                    # them as stale. A payload-only stamp is O(id), never a vector rewrite — and
                    # with reconciling off there is no sweep, so it is skipped entirely.
                    self._restamp(same, load_id)

        if dimension is None:
            # An Export with zero records. If the collection does not exist there is nothing to
            # project and nothing to reconcile against; if it does exist, the sweep below is
            # exactly "every document was purged" and must run.
            if not self._collection_exists():
                return LoadReport(collection=self._collection, dimension=0, load_id=load_id)
            dimension = 0

        deleted = self._sweep(load_id, upserted + unchanged) if self._reconcile else 0
        return LoadReport(
            collection=self._collection,
            dimension=dimension,
            load_id=load_id,
            created_collection=created,
            upserted=upserted,
            unchanged=unchanged,
            deleted=deleted,
            reconciled=self._reconcile,
        )

    def verify_absent(self, record_ids: Iterable[str]) -> list[str]:
        """Return the subset of ``record_ids`` still present in the collection.

        The auditable half of the GDPR story, and the step a Retraction/Purge runbook ends with:
        after re-loading the post-deletion Export, ask the *store* — not the pipeline, and not a
        flag — whether the deleted records are gone. An empty list is the evidence. (Qdrant marks
        deleted vectors in its HNSW graph and reclaims them on optimisation; the guarantee here is
        the one that matters for erasure: the point and its payload are no longer retrievable or
        searchable.)
        """
        wanted = list(dict.fromkeys(record_ids))
        if not wanted:
            return []
        if not self._collection_exists():
            return []
        by_point = {point_id(rid, self._namespace): rid for rid in wanted}
        found = self._call(
            "retrieve",
            lambda c: c.retrieve(
                collection_name=self._collection,
                ids=list(by_point),
                with_payload=False,
                with_vectors=False,
            ),
        )
        present = {by_point[str(rec.id)] for rec in found if str(rec.id) in by_point}
        return [rid for rid in wanted if rid in present]

    # -- the store steps --------------------------------------------------------------------

    def _to_point(self, row: ExportRow, load_id: str) -> _Point:
        reserved = RESERVED_PAYLOAD_KEYS & set(row.payload)
        if reserved:
            msg = (
                f"The Export carries column(s) {sorted(reserved)}, which collide with the sink's "
                "own payload bookkeeping (the change digest and the reconcile marker). Rename the "
                "column(s) in the Export; without that the sink cannot tell a stale point from a "
                "fresh one."
            )
            raise SinkExportError(msg)
        digest = _digest(row.vector, row.payload)
        payload = {**row.payload, DIGEST_KEY: digest, LOAD_ID_KEY: load_id}
        return _Point(
            id=point_id(row.record_id, self._namespace),
            record_id=row.record_id,
            vector=row.vector,
            payload=payload,
            digest=digest,
        )

    def _partition(self, points: list[_Point]) -> tuple[list[_Point], list[_Point]]:
        """Split a batch into (must-upsert, already-current) by comparing stored content digests."""
        stored = self._call(
            "retrieve",
            lambda c: c.retrieve(
                collection_name=self._collection,
                ids=[p.id for p in points],
                with_payload=[DIGEST_KEY],
                with_vectors=False,
            ),
        )
        digests = {str(rec.id): (rec.payload or {}).get(DIGEST_KEY) for rec in stored}
        changed = [p for p in points if digests.get(p.id) != p.digest]
        same = [p for p in points if digests.get(p.id) == p.digest]
        return changed, same

    def _upsert(self, points: list[_Point]) -> None:
        models = self._load_models()
        structs = [
            models.PointStruct(id=p.id, vector=self._vector_arg(p.vector), payload=p.payload)
            for p in points
        ]
        self._call(
            "upsert",
            lambda c: c.upsert(collection_name=self._collection, points=structs, wait=True),
        )

    def _restamp(self, points: list[_Point], load_id: str) -> None:
        self._call(
            "set_payload",
            lambda c: c.set_payload(
                collection_name=self._collection,
                payload={LOAD_ID_KEY: load_id},
                points=[p.id for p in points],
                wait=True,
            ),
        )

    def _sweep(self, load_id: str, touched: int) -> int:
        """Delete everything this load did not touch, then prove the stale set is empty.

        ``touched`` is how many records this load processed — every one of them was either upserted
        with this run's marker or payload-restamped with it, so *exactly* that many points must
        carry the marker when the sweep begins. That equality is checked **before** anything is
        deleted, because the sweep's filter is collection-*global*: it deletes by
        ``must_not(marker)``, which is the stale set only while nothing else has stamped the
        collection. A load that overlaps this one restamps the very points this one just declared
        current, and an unguarded sweep would then delete them all — erasing a live index while
        both runs report success. On a mismatch the load fails loudly and deletes nothing, so the
        worst case stays the documented one: stale-but-present data, never silently-erased data.
        """
        models = self._load_models()
        mine = models.Filter(
            must=[models.FieldCondition(key=LOAD_ID_KEY, match=models.MatchValue(value=load_id))]
        )
        marked = self._count(mine)
        if marked != touched:
            msg = (
                f"refusing to sweep collection {self._collection!r}: this load processed {touched} "
                f"record(s) but {marked} point(s) carry its marker {load_id!r}. Another load is "
                "running against this collection concurrently (or the Export repeats a record_id). "
                "Nothing was deleted — the collection still holds the data this load read. Re-run "
                "the load once no other load is in flight."
            )
            raise SinkLoadError(msg)

        stale = models.Filter(
            must_not=[
                models.FieldCondition(key=LOAD_ID_KEY, match=models.MatchValue(value=load_id))
            ]
        )
        before = self._count(stale)
        if before:
            self._call(
                "delete",
                lambda c: c.delete(
                    collection_name=self._collection,
                    points_selector=models.FilterSelector(filter=stale),
                    wait=True,
                ),
            )
        after = self._count(stale)
        if after:
            # The verification step, run automatically: a load that leaves stale points behind is
            # a FAILED load, not a green one with a caveat. Anything else would let a purged
            # record keep serving.
            msg = (
                f"reconcile did not converge on collection {self._collection!r}: {after} point(s) "
                "not written by this load are still present after the sweep. The collection may "
                "still serve retracted or purged records — do not treat this load as complete."
            )
            raise SinkLoadError(msg)
        return before

    def _count(self, filt: Any) -> int:
        result = self._call(
            "count",
            lambda c: c.count(collection_name=self._collection, count_filter=filt, exact=True),
        )
        return int(result.count)

    def _collection_exists(self) -> bool:
        return bool(
            self._call("collection_exists", lambda c: c.collection_exists(self._collection))
        )

    def _ensure_collection(self, dimension: int) -> bool:
        """Create the collection at the Export's vector width if absent. Returns whether it was."""
        if self._collection_exists():
            return False
        models = self._load_models()
        params = models.VectorParams(
            size=dimension, distance=getattr(models.Distance, self._distance.upper())
        )
        config = {self._vector_name: params} if self._vector_name else params
        self._call(
            "create_collection",
            lambda c: c.create_collection(
                collection_name=self._collection, vectors_config=config
            ),
        )
        return True

    def _vector_arg(self, vector: list[float]) -> Any:
        return {self._vector_name: vector} if self._vector_name else vector

    # -- client plumbing --------------------------------------------------------------------

    def _call(self, what: str, fn: Any) -> Any:
        """Run one store call, mapping any native fault to a typed, leak-free sink error."""
        client = self._resolve_client()
        try:
            return fn(client)
        except SinkError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in (401, 403):
                msg = (
                    f"qdrant rejected the credentials on collection {self._collection!r} "
                    f"({what}). Check the api key / auth configuration for the target cluster."
                )
                raise SinkAuthError(msg) from exc
            msg = (
                f"qdrant {what} failed on collection {self._collection!r}: {type(exc).__name__}. "
                "No records were deleted by this load."
            )
            raise SinkLoadError(msg) from exc

    def _resolve_client(self) -> Any:
        if self._client is None:
            self._client = self._import_qdrant()[0](**self._client_kwargs)
        return self._client

    def _load_models(self) -> Any:
        """The model namespace the store call shapes come from.

        Resolution order — explicit injection, then the injected client's own namespace (how the
        shipped :class:`~latence_sink_qdrant.testing.FakeQdrantClient` supplies faithful shapes
        offline), then the real ``qdrant_client.models``, imported lazily so nothing heavy is needed
        to construct a sink or validate its config.
        """
        if self._models is None:
            self._models = getattr(self._client, "models", None) or self._import_qdrant()[1]
        return self._models

    @staticmethod
    def _import_qdrant() -> tuple[Any, Any]:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'qdrant-client' package is required to load into a live Qdrant. Install it "
                "with: pip install 'latence-sink-qdrant[client]'"
            )
            raise ImportError(msg) from exc
        return QdrantClient, models


def _load_marker(label: str | None) -> str:
    """The value this load stamps on every point it touches — unique per load, always.

    ``load_id`` is the run's **label**, not the marker itself. The sweep deletes exactly
    ``must_not(_latence_load_id == marker)``, so a marker reused across runs matches the *previous*
    run's points too, i.e. matches nothing: the sweep silently becomes a no-op and the convergence
    check passes over an empty-by-construction stale set, leaving retracted/purged points serving.
    So a configured label is *prefixed* onto a fresh nonce — the run stays auditable (filter the
    collection on the ``"<label>:"`` prefix, and the summary prints the whole marker) while the
    marker stays per-run unique. Without a label the marker is the bare nonce, exactly as before.
    """
    nonce = uuid.uuid4().hex
    return f"{label}:{nonce}" if label else nonce


def _digest(vector: list[float], payload: dict[str, Any]) -> str:
    """A stable content fingerprint of one record as it would be stored.

    Covers the vector *and* the payload, so a re-embed, a re-chunk, a redaction change or a
    provenance change all count as "changed" — and nothing else does. Sorted keys + a canonical
    float repr make it identical across processes and machines (the sink must agree with the run
    that wrote the point last month, on another host).
    """
    material = json.dumps(
        {"v": [float(x) for x in vector], "p": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _require_name(value: object, key: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"QdrantSink '{key}' must be a non-empty string, got {value!r}."
        raise SinkConfigError(msg)
    return value


def _require_positive_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = f"QdrantSink '{key}' must be an int >= 1, got {value!r}."
        raise SinkConfigError(msg)
    return value


__all__ = [
    "DEFAULT_ID_NAMESPACE",
    "DIGEST_KEY",
    "LOAD_ID_KEY",
    "RESERVED_PAYLOAD_KEYS",
    "LoadReport",
    "QdrantSink",
    "point_id",
]
