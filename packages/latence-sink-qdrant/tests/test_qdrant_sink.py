"""The Qdrant sink, tested at its two real boundaries (#183, the first slice of #156).

Every test drives the *public* surface — ``QdrantSink.load`` / ``QdrantSink.verify_absent`` in, a
:class:`LoadReport` and an observable store state out — and every fixture is a **real Export**,
written by the framework's own ``JsonlParquetExport`` Provider, not a hand-rolled parquet. So these
tests exercise the exact file the sink meets in production and survive any refactor of *how* the
sink builds its calls.

The store is the shipped :class:`~latence_sink_qdrant.testing.FakeQdrantClient`, which is faithful
on the rejection side (404 unknown collection, 409 duplicate create, 403 bad key, 400 for a
non-UUID point id / wrong-width vector / named-vs-unnamed mismatch / non-JSON payload). Two tests
below assert those rejections directly, because the anti-false-green rule stands or falls on them:
a fake that accepted a raw ``record_id`` as a point id would hide a bug that fails on every real
Qdrant.

The load-bearing test is :func:`test_retracted_record_is_verifiably_absent_after_reload`: a plain
upsert loader — the obvious implementation — passes every other test in this file and fails that
one, because upserting cannot express a deletion.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from latence_sink_qdrant import (
    DIGEST_KEY,
    LOAD_ID_KEY,
    QdrantSink,
    SinkAuthError,
    SinkConfigError,
    SinkExportError,
    SinkLoadError,
    SinkVerificationError,
    point_id,
)
from latence_sink_qdrant.testing import (
    CountResult,
    FakeQdrantClient,
    FakeUnexpectedResponse,
    Filter,
)

if TYPE_CHECKING:
    from pathlib import Path

_DIMENSION = 32
_COLLECTION = "corpus"


# --- fixtures: a REAL Export ---------------------------------------------------------------------


def _chunk(record_id: str, text: str, file_name: str = "a.txt") -> ChunkRecord:
    prov = Provenance(
        source_uri=f"file:///c/{file_name}",
        file_name=file_name,
        file_type="txt",
        file_size=100,
        document_id="sha256:d1",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=record_id,
        provenance=prov,
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=0,
        content=text,
        token_count=max(1, len(text.split())),
    )


def write_export(
    out_dir: Path,
    records: dict[str, str],
    *,
    dimension: int = _DIMENSION,
    embedder: bool = True,
) -> str:
    """Write a real ``records.parquet`` for ``{record_id: text}`` and return its directory URI."""
    config: dict[str, Any] = {}
    if embedder:
        config["embedder"] = {"provider": "embedding.hashing", "config": {"dimension": dimension}}
    export = JsonlParquetExport(config)
    export.export(
        [_chunk(rid, text) for rid, text in records.items()],
        Storage(),
        str(out_dir),
    )
    return str(out_dir)


def write_rows_export(
    out_dir: Path, rows: list[tuple[str, str, str]], *, dimension: int = _DIMENSION
) -> str:
    """Write a real ``records.parquet`` for ``(record_id, text, file_name)`` **rows**.

    The ``dict``-keyed :func:`write_export` cannot express the one thing the pipeline routinely
    produces and the sink has to survive: two rows carrying the *same* ``record_id``. That is not a
    corrupt Export — ids are content-addressed (``sha256`` of the document bytes), so two
    byte-identical documents in a corpus share one, while their Provenance still differs.
    """
    export = JsonlParquetExport(
        {"embedder": {"provider": "embedding.hashing", "config": {"dimension": dimension}}}
    )
    export.export(
        [_chunk(rid, text, file_name=name) for rid, text, name in rows], Storage(), str(out_dir)
    )
    return str(out_dir)


def stored_payload(client: FakeQdrantClient, record_id: str) -> dict[str, Any]:
    return client.points[_COLLECTION][point_id(record_id)][1]


def stored_record_ids(client: FakeQdrantClient) -> set[str]:
    return {payload["record_id"] for _, payload in client.points[_COLLECTION].values()}


@pytest.fixture
def client() -> FakeQdrantClient:
    return FakeQdrantClient()


def sink(client: FakeQdrantClient, **config: Any) -> QdrantSink:
    return QdrantSink({"collection": _COLLECTION, **config}, client=client)


# --- AC 1: one command loads an Export ------------------------------------------------------------


def test_load_creates_the_collection_and_maps_every_row(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """Create-if-absent, embedding as the vector, remaining fields as payload — in one call."""
    export = write_export(tmp_path, {"c1": "alpha text", "c2": "beta text"})

    report = sink(client).load(export)

    assert report.created_collection is True
    assert report.dimension == _DIMENSION
    assert (report.upserted, report.unchanged, report.deleted) == (2, 0, 0)
    assert stored_record_ids(client) == {"c1", "c2"}

    # the vector is the Export's own embedding column, at the Export's own width
    table = pq.read_table(f"{export}/records.parquet")  # type: ignore[no-untyped-call]
    by_id = {row["record_id"]: row for row in table.to_pylist()}
    vector, payload = client.points[_COLLECTION][point_id("c1")]
    assert vector == [pytest.approx(v) for v in by_id["c1"]["embedding"]]
    assert len(vector) == _DIMENSION

    # every remaining exported column rides through as payload, verbatim...
    assert payload["content"] == "alpha text"
    assert json.loads(payload["provenance"])["file_name"] == "a.txt"
    assert payload["record_id"] == "c1"
    # ...and the vector is NOT duplicated into the payload
    assert "embedding" not in payload


def test_load_batches_the_upserts(tmp_path: Path, client: FakeQdrantClient) -> None:
    """``batch_size`` bounds the round-trip, so a corpus is never one giant request."""
    export = write_export(tmp_path, {f"c{i}": f"text {i}" for i in range(5)})

    report = sink(client, batch_size=2).load(export)

    assert report.upserted == 5
    assert client.calls.count("upsert") == 3  # 2 + 2 + 1


def test_an_existing_collection_is_not_recreated(tmp_path: Path, client: FakeQdrantClient) -> None:
    """Create-if-*absent*: a second load must not raise the store's 409."""
    export = write_export(tmp_path, {"c1": "alpha"})
    sink(client).load(export)

    report = sink(client).load(export)

    assert report.created_collection is False
    assert client.calls.count("create_collection") == 1


def test_a_named_vector_is_wired_when_configured(tmp_path: Path, client: FakeQdrantClient) -> None:
    """With ``vector_name`` the collection is created named and the point carries a named vector."""
    export = write_export(tmp_path, {"c1": "alpha"})

    sink(client, vector_name="dense").load(export)

    vector, _ = client.points[_COLLECTION][point_id("c1")]
    assert isinstance(vector, dict)
    assert len(vector["dense"]) == _DIMENSION


# --- AC 2: idempotent; delta upserts only what changed --------------------------------------------


def test_point_ids_are_uuids_derived_from_the_record_id(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """Identity is content-addressed and *derived* — the store never sees the raw record_id as id."""
    export = write_export(tmp_path, {"c1": "alpha"})

    sink(client).load(export)

    assert set(client.points[_COLLECTION]) == {point_id("c1")}
    assert "c1" not in client.points[_COLLECTION]


def test_the_store_rejects_a_raw_record_id_as_a_point_id(client: FakeQdrantClient) -> None:
    """The fake mirrors Qdrant's own rejection — this is why point ids must be derived.

    Anti-false-green: if this fake accepted a content-addressed ``record_id`` string as a point id,
    a loader that used one directly would pass its whole offline suite and fail on the first real
    server. It rejects it, with Qdrant's own 400.
    """
    client.create_collection(_COLLECTION, client.models.VectorParams(size=2, distance="Cosine"))

    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.upsert(_COLLECTION, [client.models.PointStruct(id="c1", vector=[1.0, 2.0])])

    assert excinfo.value.status_code == 400
    assert "point ID" in str(excinfo.value)


def test_reloading_an_unchanged_export_upserts_nothing(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """Idempotence, asserted on the store's behaviour: the second load issues no upsert at all."""
    export = write_export(tmp_path, {"c1": "alpha", "c2": "beta"})
    sink(client).load(export)
    before = dict(client.points[_COLLECTION])
    client.upserted_ids.clear()

    report = sink(client).load(export)

    assert (report.upserted, report.unchanged, report.deleted) == (0, 2, 0)
    assert client.upserted_ids == []
    # ...and the collection is unchanged apart from the (payload-only) sweep marker
    for pid, (vector, payload) in client.points[_COLLECTION].items():
        assert vector == before[pid][0]
        assert {k: v for k, v in payload.items() if k != LOAD_ID_KEY} == {
            k: v for k, v in before[pid][1].items() if k != LOAD_ID_KEY
        }


def test_a_delta_run_upserts_only_the_changed_record(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """One record's content changes ⇒ exactly one point is rewritten, the rest are left alone."""
    first = write_export(tmp_path / "v1", {"c1": "alpha", "c2": "beta", "c3": "gamma"})
    sink(client).load(first)
    client.upserted_ids.clear()

    second = write_export(tmp_path / "v2", {"c1": "alpha", "c2": "beta REVISED", "c3": "gamma"})
    report = sink(client).load(second)

    assert (report.upserted, report.unchanged, report.deleted) == (1, 2, 0)
    assert client.upserted_ids == [point_id("c2")]
    assert stored_payload(client, "c2")["content"] == "beta REVISED"
    assert stored_payload(client, "c1")["content"] == "alpha"


def test_a_re_embed_alone_counts_as_changed(tmp_path: Path, client: FakeQdrantClient) -> None:
    """The change digest covers the VECTOR, not only the payload — a re-embed is a real change.

    Byte-identical payload, different embedding (a new Embedder over the same text). If the digest
    covered the payload only, the store would keep serving last month's vectors forever.
    """

    def export_with(vector: list[float], into: Path) -> str:
        into.mkdir(parents=True, exist_ok=True)
        pq.write_table(  # type: ignore[no-untyped-call]
            pa.table({"record_id": ["c1"], "content": ["alpha"], "embedding": [vector]}),
            into / "records.parquet",
        )
        return str(into)

    sink(client).load(export_with([0.1, 0.2], tmp_path / "v1"))
    client.upserted_ids.clear()

    report = sink(client).load(export_with([0.9, 0.8], tmp_path / "v2"))

    assert report.upserted == 1
    assert client.upserted_ids == [point_id("c1")]
    assert client.points[_COLLECTION][point_id("c1")][0] == [0.9, 0.8]
    assert stored_payload(client, "c1")[DIGEST_KEY]


# --- AC 3: Retraction / Purge honoured -------------------------------------------------------------


def test_retracted_record_is_verifiably_absent_after_reload(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """THE regression test: a record dropped from the Export is deleted from the collection.

    After ``latence retract`` / ``latence purge`` the pipeline re-exports the surviving corpus; the
    deleted document's derived records are simply not in it. An upsert-only loader converges on
    every other case in this file and silently keeps serving the retracted record here.
    """
    full = write_export(tmp_path / "v1", {"c1": "alpha", "c2": "beta", "c3": "gamma"})
    the_sink = sink(client)
    the_sink.load(full)
    assert the_sink.verify_absent(["c2"]) == ["c2"]  # present before the retraction

    after_retraction = write_export(tmp_path / "v2", {"c1": "alpha", "c3": "gamma"})
    report = sink(client).load(after_retraction)

    assert report.deleted == 1
    assert stored_record_ids(client) == {"c1", "c3"}
    assert point_id("c2") not in client.points[_COLLECTION]
    # the documented verification step, from the operator's side
    assert sink(client).verify_absent(["c2"]) == []
    assert sink(client).verify_absent(["c1", "c2"]) == ["c1"]


def test_a_configured_load_id_still_sweeps_a_retracted_record(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """A stable configured ``load_id`` must not turn the sweep into a vacuous no-op.

    ``load_id`` is the run's *label*; the value the sweep filters on has to differ between runs or
    ``must_not(_latence_load_id == label)`` matches nothing and the previous run's points survive.
    With the label reused as the marker, the load reports ``deleted 0`` and its own convergence
    check passes over an empty-by-construction stale set — a green load still serving purged data.
    """
    full = write_export(tmp_path / "v1", {"c1": "alpha", "c2": "beta", "c3": "gamma"})
    sink(client, load_id="nightly-load").load(full)
    first_marker = stored_payload(client, "c1")[LOAD_ID_KEY]

    after_retraction = write_export(tmp_path / "v2", {"c1": "alpha", "c3": "gamma"})
    report = sink(client, load_id="nightly-load").load(after_retraction)

    assert report.deleted == 1
    assert stored_record_ids(client) == {"c1", "c3"}
    assert sink(client).verify_absent(["c2"]) == []
    # the marker is per-run unique even though the label is pinned — including on the *unchanged*
    # points, which are restamped rather than re-upserted
    assert stored_payload(client, "c1")[LOAD_ID_KEY] != first_marker
    # ...and the label is still recoverable for the audit trail
    assert first_marker.startswith("nightly-load:")
    assert report.load_id.startswith("nightly-load:")
    assert "load_id=nightly-load" in report.summary()


def test_without_a_configured_load_id_the_marker_is_a_bare_random_hex(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The default path is untouched by the label/marker split: a bare 32-char hex nonce."""
    export = write_export(tmp_path, {"c1": "alpha"})

    report = sink(client).load(export)

    marker = stored_payload(client, "c1")[LOAD_ID_KEY]
    assert marker == report.load_id
    assert len(marker) == 32
    assert ":" not in marker
    int(marker, 16)


def test_purging_every_document_empties_the_collection(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """An Export with no records is a legitimate post-purge state, and it must reach the store."""
    full = write_export(tmp_path / "v1", {"c1": "alpha", "c2": "beta"})
    sink(client).load(full)

    empty = write_export(tmp_path / "v2", {})
    report = sink(client).load(empty)

    assert report.deleted == 2
    assert client.points[_COLLECTION] == {}


def test_an_unstamped_point_from_another_loader_is_swept(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """A point with no sweep marker at all (a hand-loaded or legacy point) is stale, and goes."""
    export = write_export(tmp_path, {"c1": "alpha"})
    sink(client).load(export)
    client.upsert(
        _COLLECTION,
        [
            client.models.PointStruct(
                id=point_id("legacy"), vector=[0.5] * _DIMENSION, payload={"record_id": "legacy"}
            )
        ],
    )

    report = sink(client).load(export)

    assert report.deleted == 1
    assert stored_record_ids(client) == {"c1"}


_NS_A = "11111111-2222-3333-4444-555555555555"
_NS_B = "66666666-7777-8888-9999-aaaaaaaaaaaa"


def test_second_corpus_in_a_shared_collection_does_not_sweep_the_first(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The setup ``id_namespace`` exists for: two corpora, one collection, both intact.

    ``id_namespace`` is documented as the way to share one collection between two corpora whose
    ``record_id`` spaces could collide. A collection-*global* sweep makes that advice destructive:
    loading corpus B deletes every point of corpus A while both loads exit 0 — and ``verify_absent``
    then hands back an empty list for A's records, i.e. erasure evidence for a corpus nobody
    retracted. The sweep must own only the namespace partition it wrote.
    """
    corpus_a = write_export(tmp_path / "a", {"A#chunk-0": "alpha", "A#chunk-1": "alef"})
    corpus_b = write_export(tmp_path / "b", {"B#chunk-0": "beta"})

    report_a = sink(client, id_namespace=_NS_A).load(corpus_a)
    report_b = sink(client, id_namespace=_NS_B).load(corpus_b)

    assert (report_a.upserted, report_a.deleted) == (2, 0)
    assert (report_b.upserted, report_b.deleted) == (1, 0)
    assert stored_record_ids(client) == {"A#chunk-0", "A#chunk-1", "B#chunk-0"}
    # and the audit verb must not certify A as erased
    auditor = sink(client, id_namespace=_NS_A)
    assert auditor.verify_absent(["A#chunk-0", "A#chunk-1"]) == ["A#chunk-0", "A#chunk-1"]


def test_retraction_sweeps_only_the_loading_namespace(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """Namespace scoping must not cost convergence: A's own retraction still reaches the store."""
    corpus_a = write_export(tmp_path / "a", {"A#chunk-0": "alpha", "A#chunk-1": "alef"})
    corpus_b = write_export(tmp_path / "b", {"B#chunk-0": "beta"})
    sink(client, id_namespace=_NS_A).load(corpus_a)
    sink(client, id_namespace=_NS_B).load(corpus_b)

    shrunk_a = write_export(tmp_path / "a2", {"A#chunk-0": "alpha"})
    report = sink(client, id_namespace=_NS_A).load(shrunk_a)

    assert report.deleted == 1
    assert stored_record_ids(client) == {"A#chunk-0", "B#chunk-0"}
    assert sink(client, id_namespace=_NS_A).verify_absent(["A#chunk-1"]) == []
    assert sink(client, id_namespace=_NS_B).verify_absent(["B#chunk-0"]) == ["B#chunk-0"]


def test_reconcile_off_keeps_records_absent_from_the_export(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The documented incremental-Export escape hatch — and its honest trade-off."""
    full = write_export(tmp_path / "v1", {"c1": "alpha", "c2": "beta"})
    sink(client).load(full)

    partial = write_export(tmp_path / "v2", {"c1": "alpha"})
    report = sink(client, reconcile=False).load(partial)

    assert report.reconciled is False
    assert report.deleted == 0
    assert stored_record_ids(client) == {"c1", "c2"}
    assert "delete" not in client.calls


def test_verify_absent_refuses_to_prove_erasure_against_a_collection_that_does_not_exist(
    client: FakeQdrantClient,
) -> None:
    """A missing collection is "I looked at nothing", never the affirmative "they are absent".

    ``verify_absent`` returning ``[]`` is the erasure evidence the Retraction/Purge runbook ends
    with, so the one thing it must never do is hand that evidence back for a collection it never
    inspected — a typo'd ``--collection`` or a stale ``--url`` would otherwise certify records that
    are still live in the real collection.
    """
    with pytest.raises(SinkVerificationError) as raised:
        sink(client).verify_absent(["c1"])

    message = str(raised.value)
    assert _COLLECTION in message
    assert "--collection" in message


def test_verify_absent_certifies_nothing_when_the_records_live_in_another_collection(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The operator-typo scenario end to end: `c1` is live in `corpus`, audited in `corpuss`."""
    sink(client).load(write_export(tmp_path, {"c1": "alpha"}))

    with pytest.raises(SinkVerificationError):
        sink(client, collection="corpuss").verify_absent(["c1"])

    assert stored_record_ids(client) == {"c1"}  # still live, and the audit said so by refusing


def test_an_overlapping_load_never_erases_the_points_this_load_declared_current(
    tmp_path: Path,
) -> None:
    """Two overlapping loads must not empty the collection while both report success.

    The sweep deletes by a collection-*global* ``must_not(my marker)`` filter, so its correctness
    argument ("everything I touched carries my marker") stops holding the instant another load
    restamps the same points. Unguarded, the losing run deletes every point it had itself just
    counted as ``unchanged`` — the exact "silently-erased data" the module docstring promises can
    never happen. It must refuse instead: stale-but-present beats erased (ADR-0034, fail-closed).
    """
    export = write_export(tmp_path, {"c1": "alpha", "c2": "beta", "c3": "gamma"})

    class Overlapping(FakeQdrantClient):
        """Runs a whole competing load the moment this load restamps its unchanged points."""

        def __init__(self) -> None:
            super().__init__()
            self.overlap: str | None = None

        def set_payload(self, *args: Any, **kwargs: Any) -> bool:
            result = super().set_payload(*args, **kwargs)
            if self.overlap is not None:
                competing, self.overlap = self.overlap, None
                sink(self).load(competing)  # the cron re-load, start to finish, in the gap
            return result

    client = Overlapping()
    sink(client).load(export)

    client.overlap = export
    with pytest.raises(SinkLoadError) as excinfo:
        sink(client).load(export)

    assert "concurrent" in str(excinfo.value).lower()
    assert stored_record_ids(client) == {"c1", "c2", "c3"}
    # The competing load's own sweep is the only delete anyone was entitled to issue.
    assert client.calls.count("delete") == 0


def test_a_restamp_inside_the_guard_delete_window_never_erases_an_exported_record(
    tmp_path: Path,
) -> None:
    """The guard is a *pre*-check; the sweep must not act on a condition re-derived after it.

    The guard counts markers at one instant and the filter-delete re-evaluates ``must_not(my
    marker)`` at another. A concurrent load restamping a point *in that window* moves it into the
    delete's reach — so the **winning** load erases a record it had itself just declared current,
    reports exit 0, and its own post-sweep convergence count (0, by construction) certifies the
    corrupted collection as converged.

    The only condition a concurrent writer cannot flip is the Export (ADR-0017): ``c0`` is *in* the
    Export this load read, so nothing this load does may delete it — while the genuinely retracted
    ``c2`` must still go.
    """
    full = write_export(tmp_path / "v1", {"c0": "alpha", "c1": "beta", "c2": "gamma"})
    after_retraction = write_export(tmp_path / "v2", {"c0": "alpha", "c1": "beta"})

    class RestampInWindow(FakeQdrantClient):
        """Lands one competing ``set_payload`` the moment the sweep guard has been answered."""

        def __init__(self) -> None:
            super().__init__()
            self.intruder: str | None = None

        def count(
            self,
            collection_name: str,
            count_filter: Any | None = None,
            exact: bool = True,
        ) -> Any:
            result = super().count(collection_name, count_filter=count_filter, exact=exact)
            # The guard's own count (the only one filtering *for* this load's marker) has just
            # been answered: the window between the check and the act is now open.
            if self.intruder is not None and count_filter is not None and count_filter.must:
                marker, self.intruder = self.intruder, None
                self.set_payload(collection_name, {LOAD_ID_KEY: marker}, [point_id("c0")])
            return result

    client = RestampInWindow()
    sink(client).load(full)

    client.intruder = "a-competing-load"
    report = sink(client).load(after_retraction)

    assert stored_record_ids(client) == {"c0", "c1"}  # c0 is live and stays live
    assert report.deleted == 1  # exactly the retracted record, honestly counted
    assert sink(client).verify_absent(["c0"]) == ["c0"]  # never certified as erased


# --- failure containment ---------------------------------------------------------------------------


def test_a_store_fault_mid_load_never_deletes_anything(tmp_path: Path) -> None:
    """A half-uploaded corpus must not trigger the sweep — stale data beats erased data."""
    export = write_export(tmp_path / "v1", {"c1": "alpha", "c2": "beta"})
    healthy = FakeQdrantClient()
    sink(healthy).load(export)
    before = dict(healthy.points[_COLLECTION])

    healthy.fail_on = {"upsert"}  # the store dies half way through the next load
    bigger = write_export(tmp_path / "v2", {"c1": "alpha", "c2": "beta", "c3": "gamma"})
    with pytest.raises(SinkLoadError) as excinfo:
        sink(healthy).load(bigger)

    assert "No records were deleted" in str(excinfo.value)
    assert healthy.calls.count("delete") == 0
    assert healthy.points[_COLLECTION] == before


class ResponseHandlingException(Exception):
    """qdrant-client's ``ResponseHandlingException`` as the sink sees it.

    Named to match the real class exactly (the sink classifies it by class name, importing nothing
    from the optional ``qdrant_client``). It is a **client-side** response-handling failure and
    carries **no** ``status_code`` — the distinguishing mark from a server ``UnexpectedResponse``.
    A qdrant-client newer than the server raises exactly this from ``count`` on a version skew,
    while every other call parses fine — the production bug (server 1.12.4 vs client 1.18.0) the
    scroll fallback exists for.
    """


class _CountVersionSkewClient(FakeQdrantClient):
    """A store whose ``count`` response the client cannot parse, while every other call works.

    It stays faithful on the rejection side: a bad key still 403s and a missing collection still
    404s (both surface *before* the skew), so only a genuine client-newer-than-server ``count``
    parse — the one thing raw HTTP and every other call survive — trips the fallback.
    """

    def count(
        self, collection_name: str, count_filter: Filter | None = None, exact: bool = True
    ) -> CountResult:
        self._enter("count")
        self._collection(collection_name)
        raise ResponseHandlingException(
            "Unable to parse response: server version 1.12.4 vs client 1.18.0"
        )


def test_a_count_the_client_cannot_parse_falls_back_to_scroll_not_a_failed_load(
    tmp_path: Path,
) -> None:
    """A qdrant-client/server version skew breaks *only* the client-side ``count`` parse.

    The load is otherwise correct — the upsert lands and the collection holds every point — so the
    reconcile guard's count must fall back to the version-robust ``scroll`` path and the load must
    succeed, not exit 2 on a false ``count failed`` red (the tutorial §4.1 look-for is
    ``N points upserted``). Pre-fix ``_count`` had no fallback: the ``ResponseHandlingException``
    was mapped to ``SinkLoadError`` and the load raised.
    """
    client = _CountVersionSkewClient()
    export = write_export(tmp_path, {"c1": "alpha text", "c2": "beta text"})

    report = QdrantSink({"collection": _COLLECTION}, client=client).load(export)

    assert (report.upserted, report.unchanged, report.deleted) == (2, 0, 0)
    assert stored_record_ids(client) == {"c1", "c2"}
    # the fast count was attempted, and the scroll fallback actually ran
    assert "count" in client.calls
    assert "scroll" in client.calls

    # An idempotent re-run is equally unbothered by the broken count: no duplication...
    again = QdrantSink({"collection": _COLLECTION}, client=client).load(export)
    assert (again.upserted, again.unchanged, again.deleted) == (0, 2, 0)
    assert stored_record_ids(client) == {"c1", "c2"}

    # ...and the retract-and-sweep still converges through the scroll-counted guard.
    after = QdrantSink({"collection": _COLLECTION}, client=client).load(
        write_export(tmp_path / "v2", {"c1": "alpha text"})
    )
    assert after.deleted == 1
    assert stored_record_ids(client) == {"c1"}


def test_a_bad_api_key_is_a_typed_auth_error(tmp_path: Path) -> None:
    """The 403 an operator hits on the first live run has its own type and its own remedy."""
    export = write_export(tmp_path, {"c1": "alpha"})
    guarded = FakeQdrantClient(api_key="wrong-key", server_api_key="the-real-key")

    with pytest.raises(SinkAuthError) as excinfo:
        sink(guarded).load(export)

    message = str(excinfo.value)
    assert "credentials" in message
    assert "wrong-key" not in message  # no key, url or store text ever leaks into the error


def test_a_wrong_width_vector_is_rejected_by_the_store(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """Loading a 64-dim corpus into a 32-dim collection fails loudly, as a real Qdrant would."""
    client.create_collection(
        _COLLECTION, client.models.VectorParams(size=_DIMENSION, distance="Cosine")
    )
    export = write_export(tmp_path, {"c1": "alpha"}, dimension=_DIMENSION * 2)

    with pytest.raises(SinkLoadError) as excinfo:
        sink(client).load(export)

    assert "upsert failed" in str(excinfo.value)


def test_the_store_rejects_operations_on_a_missing_collection(client: FakeQdrantClient) -> None:
    """The fake mirrors Qdrant's 404 rather than silently creating or ignoring the collection."""
    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.count("nope", None, True)
    assert excinfo.value.status_code == 404

    with pytest.raises(FakeUnexpectedResponse) as duplicate:
        client.create_collection(_COLLECTION, client.models.VectorParams(size=2, distance="Dot"))
        client.create_collection(_COLLECTION, client.models.VectorParams(size=2, distance="Dot"))
    assert duplicate.value.status_code == 409


def test_the_store_rejects_a_named_vector_for_an_unnamed_collection(
    client: FakeQdrantClient,
) -> None:
    """Named-vs-unnamed is a real 400, so the sink's ``vector_name`` wiring cannot be half-right."""
    client.create_collection(_COLLECTION, client.models.VectorParams(size=2, distance="Cosine"))

    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.upsert(
            _COLLECTION,
            [client.models.PointStruct(id=point_id("c1"), vector={"dense": [1.0, 2.0]})],
        )

    assert excinfo.value.status_code == 400


# --- the Export is the source of truth: it is validated, never repaired -----------------------------


def test_a_missing_export_is_an_actionable_error(tmp_path: Path, client: FakeQdrantClient) -> None:
    with pytest.raises(SinkExportError) as excinfo:
        sink(client).load(str(tmp_path / "nowhere"))
    assert "records.parquet" in str(excinfo.value)


def test_an_export_without_embeddings_is_refused(tmp_path: Path, client: FakeQdrantClient) -> None:
    """No Embedder wired ⇒ no vectors ⇒ the load would build a collection answering no query."""
    export = write_export(tmp_path, {"c1": "alpha"}, embedder=False)

    with pytest.raises(SinkExportError) as excinfo:
        sink(client).load(export)

    assert "'embedding' column" in str(excinfo.value)
    assert "Wire an Embedder" in str(excinfo.value)
    assert client.calls == []  # it failed before touching the store


def test_ragged_vector_widths_are_refused(tmp_path: Path, client: FakeQdrantClient) -> None:
    """One collection has one dimension; an Export with two cannot be loaded at all."""
    table = pa.table(
        {
            "record_id": ["c1", "c2"],
            "content": ["a", "b"],
            "embedding": [[0.1, 0.2], [0.1, 0.2, 0.3]],
        }
    )
    pq.write_table(table, tmp_path / "records.parquet")  # type: ignore[no-untyped-call]

    with pytest.raises(SinkExportError) as excinfo:
        sink(client).load(str(tmp_path))

    assert "3-dimensional" in str(excinfo.value)


def test_a_blank_record_id_is_refused_without_quoting_the_corpus(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The error names the bad row by POSITION; it never quotes a column value (R4 regression).

    The Export's ``content`` column is redacted chunk text — exactly what the redaction floor
    (ADR-0044) exists to keep out of the places humans copy-paste. An error message is one of those
    places: it reaches stderr, the operator's log, and the ticket pasted from it. So this path is
    counts-and-schema only (ADR-0034), the same discipline the retrieval backend already holds.
    Truncating a row repr does not satisfy it: with a blank ``record_id`` the Export's column order
    puts ``content`` well inside the first 200 characters.
    """
    secret = "Patient Jane Doe SSN 123-45-6789 card 4111111111111111"
    table = pa.table(
        {
            "record_id": ["c1", ""],
            "schema_version": [1, 1],
            "provenance": ['{"offset": 0}', '{"offset": 1}'],
            "classification": ['{"level": "internal"}', '{"level": "internal"}'],
            "content": ["alpha", secret],
            "embedding": [[0.1, 0.2], [0.3, 0.4]],
        }
    )
    pq.write_table(table, tmp_path / "records.parquet")  # type: ignore[no-untyped-call]

    with pytest.raises(SinkExportError) as excinfo:
        sink(client).load(str(tmp_path))

    message = str(excinfo.value)
    assert secret not in message  # the whole point
    assert "123-45-6789" not in message
    assert "alpha" not in message  # no row's content, not just the offending one
    assert "row 1" in message  # ...and it is still actionable: which row
    assert "record_id" in message  # column NAMES are schema, and stay


def test_a_column_colliding_with_the_sink_bookkeeping_is_refused(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The sweep marker must be unforgeable, so a column of that name is a loud error."""
    table = pa.table(
        {"record_id": ["c1"], "embedding": [[0.1, 0.2]], LOAD_ID_KEY: ["forged"]}
    )
    pq.write_table(table, tmp_path / "records.parquet")  # type: ignore[no-untyped-call]

    with pytest.raises(SinkExportError) as excinfo:
        sink(client).load(str(tmp_path))

    assert LOAD_ID_KEY in str(excinfo.value)


# --- config is validated eagerly, offline ----------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "needle"),
    [
        ({}, "'collection'"),
        ({"collection": ""}, "'collection'"),
        ({"collection": "c", "batch_size": 0}, "'batch_size'"),
        ({"collection": "c", "batch_size": True}, "'batch_size'"),
        ({"collection": "c", "distance": "hamming"}, "'distance'"),
        ({"collection": "c", "reconcile": "yes"}, "'reconcile'"),
        ({"collection": "c", "vector_name": ""}, "'vector_name'"),
        ({"collection": "c", "id_namespace": "not-a-uuid"}, "'id_namespace'"),
        ({"collection": "c", "colection": "typo"}, "unknown config key"),
    ],
)
def test_bad_config_fails_at_construction(config: dict[str, Any], needle: str) -> None:
    """Before a file is read and before a connection is opened — a typo costs one second."""
    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink(config)
    assert needle in str(excinfo.value)


def test_constructing_a_sink_needs_no_client_and_no_connection() -> None:
    """The heavy client is imported lazily: config validation is pure and offline."""
    built = QdrantSink({"collection": "c", "url": "http://example:6333", "api_key": "k"})
    assert isinstance(built, QdrantSink)


def test_the_report_summarises_what_happened(tmp_path: Path, client: FakeQdrantClient) -> None:
    export = write_export(tmp_path, {"c1": "alpha"})
    report = sink(client, load_id="run-7").load(export)
    summary = report.summary()
    assert "corpus" in summary
    assert "upserted 1" in summary
    assert "load_id=run-7" in summary
    assert report.records == 1


# --- two byte-identical documents: one record_id, one point (R3 regression) ----------------------

#: A corpus holding the same document twice — the duplicate a real document library always has.
#: Both rows carry the id latence-core derives from the *bytes* (``sha256`` of the content, then
#: ``#chunk-<i>``), because identical content genuinely is the same record; only Provenance differs.
_DUPLICATED = "sha256:2e02f2c2#chunk-0"
_TWO_COPIES = [
    (_DUPLICATED, "the same words", "a.txt"),
    (_DUPLICATED, "the same words", "b_copy.txt"),
]


def _file_name_of(client: FakeQdrantClient, record_id: str) -> str:
    return str(json.loads(stored_payload(client, record_id)["provenance"])["file_name"])


def test_a_corpus_with_two_byte_identical_documents_loads_and_converges(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """Duplicated content must load, not deadlock the default reconciling path.

    Before the fix the sweep's guard compared Export **rows** against marked **points**: two rows
    collapsing onto one deterministic point made ``touched=2`` and ``marked=1``, so every default
    (``reconcile: true``) load of such a corpus raised ``SinkLoadError`` — permanently, and blaming
    a concurrent load that does not exist. The only escape was ``reconcile: false``, i.e. turning
    off the sweep the erasure runbook depends on.
    """
    export = write_rows_export(tmp_path, _TWO_COPIES)

    report = sink(client).load(export)

    # One record, one point — and the counts say so in the units they name.
    assert report.upserted == 1
    assert report.duplicate_rows == 1
    assert report.points == 1
    assert report.records == 2  # rows read from the Export
    assert report.deleted == 0
    assert list(client.points[_COLLECTION]) == [point_id(_DUPLICATED)]
    # Last row wins, exactly as an upsert of a repeated id behaves on a real Qdrant.
    assert _file_name_of(client, _DUPLICATED) == "b_copy.txt"

    # ...and re-running is idempotent rather than a second failure.
    again = sink(client).load(export)
    assert (again.upserted, again.unchanged, again.duplicate_rows, again.deleted) == (0, 1, 1, 0)
    assert list(client.points[_COLLECTION]) == [point_id(_DUPLICATED)]


def test_the_report_counts_points_not_rows_with_reconcile_off(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The non-sweeping path reported "upserted 2" over a collection holding one point."""
    export = write_rows_export(tmp_path, _TWO_COPIES)

    report = sink(client, reconcile=False).load(export)

    assert report.upserted == len(client.points[_COLLECTION]) == 1
    assert report.duplicate_rows == 1
    assert "1 repeated record_id(s) collapsed" in report.summary()


def test_duplicate_record_ids_are_counted_once_across_batches(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The repeat is recognised even when the two rows never meet in one batch.

    ``batch_size=1`` puts each copy in its own round-trip, so a within-batch de-duplication alone
    would still over-count and still trip the sweep's guard. The second row is recognised from the
    marker this load already stamped on the point — which is also why ``load`` never has to hold a
    corpus-sized set of ids (ADR-0033).
    """
    export = write_rows_export(tmp_path, _TWO_COPIES)

    report = sink(client, batch_size=1).load(export)

    assert (report.upserted, report.duplicate_rows, report.points) == (1, 1, 1)
    assert _file_name_of(client, _DUPLICATED) == "b_copy.txt"


def test_a_duplicated_document_is_still_verifiably_absent_after_retraction(
    tmp_path: Path, client: FakeQdrantClient
) -> None:
    """The erasure runbook has to keep working for the corpus that provoked the bug.

    This is the reason the fix could not be "document ``reconcile: false``": with the sweep off, the
    purged record stays live and searchable, and ``verify_absent`` says so.
    """
    full = write_rows_export(tmp_path / "v1", [*_TWO_COPIES, ("c1", "alpha", "c1.txt")])
    sink(client).load(full)
    assert sink(client).verify_absent([_DUPLICATED]) == [_DUPLICATED]

    after_purge = write_rows_export(tmp_path / "v2", [("c1", "alpha", "c1.txt")])
    report = sink(client).load(after_purge)

    assert report.deleted == 1
    assert stored_record_ids(client) == {"c1"}
    assert sink(client).verify_absent([_DUPLICATED]) == []


# --- the credential never lives in the config file (R5-docs_truth-7 regression) ------------------
#
# The README's only credential example used to be ``api_key: ${QDRANT_API_KEY}``, and nothing in
# the framework shell-expands a config file — the literal 17 characters were handed to
# ``QdrantClient`` as the key, and the operator learned about it as an opaque 401 from a secured
# cluster. These four assert the real path (``api_key_env`` names the VARIABLE) and that the old
# documented syntax now fails loudly at construction instead of at the store.


def test_api_key_env_resolves_the_credential_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "s3cr3t")

    built = QdrantSink(
        {"collection": "c", "url": "http://example:6333", "api_key_env": "QDRANT_API_KEY"}
    )

    kwargs = built._client_kwargs  # noqa: SLF001 - the kwargs are what reaches QdrantClient
    assert kwargs["api_key"] == "s3cr3t"
    # The indirection key is a sink key: QdrantClient has no such kwarg and would reject it.
    assert "api_key_env" not in kwargs


def test_an_unexpanded_placeholder_api_key_is_rejected_not_sent_to_the_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact README-follower's config: a `${VAR}` nothing expands must not become the key."""
    monkeypatch.setenv("QDRANT_API_KEY", "s3cr3t")

    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink(
            {"collection": "c", "url": "http://example:6333", "api_key": "${QDRANT_API_KEY}"}
        )

    message = str(excinfo.value)
    assert "${QDRANT_API_KEY}" in message  # names what it found
    assert "api_key_env" in message  # and what to write instead
    assert "s3cr3t" not in message  # never a credential value


def test_an_unset_api_key_env_variable_is_a_typed_error_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink({"collection": "c", "api_key_env": "QDRANT_API_KEY"})

    assert "QDRANT_API_KEY" in str(excinfo.value)


def test_an_explicit_api_key_still_wins_over_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """So ``--api-key`` keeps overriding a config file that names a variable."""
    monkeypatch.setenv("QDRANT_API_KEY", "from-env")

    built = QdrantSink(
        {"collection": "c", "api_key": "from-flag", "api_key_env": "QDRANT_API_KEY"}
    )

    assert built._client_kwargs["api_key"] == "from-flag"  # noqa: SLF001


def test_a_real_key_containing_a_dollar_sign_is_not_mistaken_for_a_placeholder() -> None:
    """Only the unambiguous braced form is template syntax; refusing a valid key is worse."""
    built = QdrantSink({"collection": "c", "api_key": "ab$cd12$xy"})

    assert built._client_kwargs["api_key"] == "ab$cd12$xy"  # noqa: SLF001
