"""Named sparse vectors: the load half of SPLADE (T-RET-1, `issues/dogfood/qdrant-native-sparse.md`).

The defect this file exists to pin down: the sink created **dense-only** ``VectorParams``, so the
Export's ``sparse_indices`` / ``sparse_values`` columns rode into the *payload* as ordinary lists —
stored, never queryable — while :class:`latence_retrieval.qdrant.QdrantBackend` already issued
``query_points(..., using=<sparse vector name>)``. A query for an index that was never built.

The reason no test caught it is the reason the first half of this file is about the **fake**:
``FakeQdrantClient`` contained zero occurrences of "sparse", so the suite was structurally unable to
fail. A fake that cannot express the feature manufactures green; a fake that *accepts* what a real
Qdrant rejects manufactures worse. So the sparse rejections are asserted here directly, in the same
shape as the file-level rejections the fake already models (400/404/409/403).

What stays out of reach offline, and is marked so rather than faked: whether a real Qdrant server
ranks a sparse query the way we expect. That is ``test_live_smoke``-shaped, gated on
``LATENCE_QDRANT_URL``, and proven on the pod (T-RET-3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from latence_sink_qdrant import (
    QdrantSink,
    SinkConfigError,
    SinkExportError,
    SinkLoadError,
    point_id,
)
from latence_sink_qdrant.testing import (
    FakeQdrantClient,
    FakeUnexpectedResponse,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

if TYPE_CHECKING:
    from pathlib import Path

_DIMENSION = 8
_VOCAB = 64
_COLLECTION = "corpus"


# --- fixtures: a REAL Export, with the sparse columns a SparseEmbedder writes ---------------------


def _chunk(record_id: str, text: str) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
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


def write_sparse_export(
    out_dir: Path,
    records: dict[str, str],
    *,
    dense: bool = True,
    sparse: bool = True,
) -> str:
    """A real ``records.parquet`` carrying the dense and/or sparse signal columns."""
    config: dict[str, Any] = {}
    if dense:
        config["embedder"] = {
            "provider": "embedding.hashing",
            "config": {"dimension": _DIMENSION},
        }
    if sparse:
        config["sparse_embedder"] = {
            "provider": "sparse.hashing",
            "config": {"vocab_size": _VOCAB},
        }
    JsonlParquetExport(config).export(
        [_chunk(rid, text) for rid, text in records.items()], Storage(), str(out_dir)
    )
    return str(out_dir)


def rewrite_column(export_dir: str, name: str, column: pa.Array | pa.ChunkedArray) -> None:
    """Replace one column of an existing Export in place (nothing else moves).

    The only way to say "the same corpus, re-embedded sparsely" without also changing the dense
    vector and the payload — which is exactly what the digest test has to isolate.
    """
    uri = f"{export_dir}/records.parquet"
    table = pq.read_table(uri)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    table = table.set_column(table.schema.get_field_index(name), name, column)
    pq.write_table(table, uri)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs


def rename_columns(export_dir: str, mapping: dict[str, str]) -> None:
    """Rename Export columns in place — how a differently-named sparse signal is simulated."""
    uri = f"{export_dir}/records.parquet"
    table = pq.read_table(uri)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
    table = table.rename_columns([mapping.get(n, n) for n in table.schema.names])
    pq.write_table(table, uri)  # type: ignore[no-untyped-call]  # pyarrow ships no stubs


def sink(client: FakeQdrantClient, **config: Any) -> QdrantSink:
    return QdrantSink({"collection": _COLLECTION, **config}, client=client)


# --- the fake: it must be able to EXPRESS sparse, and REJECT what a real Qdrant rejects -----------


def test_fake_stores_a_named_sparse_vector_alongside_the_dense_one() -> None:
    """The capability the fake lacked entirely — without it no test here could ever go red."""
    client = FakeQdrantClient()
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={"dense": VectorParams(size=3, distance="Cosine")},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    client.upsert(
        collection_name=_COLLECTION,
        points=[
            PointStruct(
                id="11111111-1111-5111-8111-111111111111",
                vector={
                    "dense": [1.0, 2.0, 3.0],
                    "sparse": SparseVector(indices=[2, 7], values=[0.5, 0.25]),
                },
                payload={"record_id": "r1"},
            )
        ],
    )
    (record,) = client.retrieve(
        collection_name=_COLLECTION,
        ids=["11111111-1111-5111-8111-111111111111"],
        with_vectors=True,
    )
    # ``with_vectors=True`` must actually carry vectors back — narrowing the Optional is itself
    # the assertion that it did (real Qdrant returns ``vector=None`` when it did not).
    assert record.vector is not None
    assert record.vector["dense"] == [1.0, 2.0, 3.0]
    assert record.vector["sparse"] == SparseVector(indices=[2, 7], values=[0.5, 0.25])


def test_fake_reports_the_sparse_config_of_a_collection() -> None:
    client = FakeQdrantClient()
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config=VectorParams(size=3, distance="Cosine"),
        sparse_vectors_config={"splade": SparseVectorParams()},
    )
    info = client.get_collection(_COLLECTION)
    assert set(info.config.params.sparse_vectors) == {"splade"}


@pytest.mark.parametrize(
    ("vector", "reason"),
    [
        (SparseVector(indices=[1, 2], values=[0.5]), "same length"),
        (SparseVector(indices=[1, 1], values=[0.5, 0.5]), "duplicate"),
        (SparseVector(indices=[-1], values=[0.5]), "non-negative"),
        (SparseVector(indices=["a"], values=[0.5]), "integer"),  # type: ignore[list-item]
        (SparseVector(indices=[1], values=["x"]), "number"),  # type: ignore[list-item]
    ],
)
def test_fake_rejects_a_malformed_sparse_vector(vector: SparseVector, reason: str) -> None:
    """Every one of these is a 400 from a real Qdrant; a fake that took them would hide the bug."""
    client = FakeQdrantClient()
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.upsert(
            collection_name=_COLLECTION,
            points=[
                PointStruct(id=1, vector={"sparse": vector}),
            ],
        )
    assert excinfo.value.status_code == 400
    assert reason in str(excinfo.value)


def test_fake_rejects_a_sparse_vector_under_an_undeclared_name() -> None:
    client = FakeQdrantClient()
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.upsert(
            collection_name=_COLLECTION,
            points=[PointStruct(id=1, vector={"splade": SparseVector(indices=[1], values=[1.0])})],
        )
    assert excinfo.value.status_code == 400


def test_fake_rejects_a_dense_list_under_a_sparse_name_and_vice_versa() -> None:
    client = FakeQdrantClient()
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={"dense": VectorParams(size=2, distance="Cosine")},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    with pytest.raises(FakeUnexpectedResponse) as dense_under_sparse:
        client.upsert(
            collection_name=_COLLECTION,
            points=[PointStruct(id=1, vector={"dense": [1.0, 2.0], "sparse": [1.0, 2.0]})],
        )
    assert dense_under_sparse.value.status_code == 400
    with pytest.raises(FakeUnexpectedResponse) as sparse_under_dense:
        client.upsert(
            collection_name=_COLLECTION,
            points=[
                PointStruct(
                    id=1,
                    vector={
                        "dense": SparseVector(indices=[1], values=[1.0]),
                        "sparse": SparseVector(indices=[1], values=[1.0]),
                    },
                )
            ],
        )
    assert sparse_under_dense.value.status_code == 400


def test_fake_rejects_a_collection_declaring_no_vectors_at_all() -> None:
    client = FakeQdrantClient()
    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.create_collection(collection_name=_COLLECTION, vectors_config={})
    assert excinfo.value.status_code == 400


def test_fake_rejects_a_sparse_name_colliding_with_a_dense_name() -> None:
    """One point carries ONE vector map, so a name cannot mean two indexes."""
    client = FakeQdrantClient()
    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config={"v": VectorParams(size=2, distance="Cosine")},
            sparse_vectors_config={"v": SparseVectorParams()},
        )
    assert excinfo.value.status_code == 400


def test_fake_rejects_a_sparse_config_that_is_not_a_named_mapping() -> None:
    """Sparse vectors in Qdrant are ALWAYS named — there is no unnamed sparse vector."""
    client = FakeQdrantClient()
    with pytest.raises(FakeUnexpectedResponse) as excinfo:
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=2, distance="Cosine"),
            sparse_vectors_config=SparseVectorParams(),  # type: ignore[arg-type]
        )
    assert excinfo.value.status_code == 400


def test_fake_accepts_an_empty_sparse_vector() -> None:
    """A chunk whose redacted text has no surviving terms is the empty SparseVector, not an error."""
    client = FakeQdrantClient()
    client.create_collection(
        collection_name=_COLLECTION,
        vectors_config={},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    client.upsert(
        collection_name=_COLLECTION,
        points=[PointStruct(id=1, vector={"sparse": SparseVector()})],
    )
    (record,) = client.retrieve(collection_name=_COLLECTION, ids=[1], with_vectors=True)
    assert record.vector is not None
    assert record.vector["sparse"] == SparseVector(indices=[], values=[])


# --- the sink: the collection must carry a named sparse vector ------------------------------------


def test_created_collection_carries_a_named_sparse_vector(tmp_path: Path) -> None:
    """The ticket's acceptance #2, and the whole defect in one assertion.

    Before the fix the sink created dense-only ``VectorParams`` and this collection had no sparse
    index at all — so ``query_points(..., using="sparse")`` could only ever return nothing.
    """
    export = write_sparse_export(tmp_path, {"r1": "alpha beta", "r2": "gamma delta"})
    client = FakeQdrantClient()
    report = sink(client, vector="both").load(export)

    info = client.get_collection(_COLLECTION)
    assert set(info.config.params.sparse_vectors) == {"sparse"}
    # Sparse implies a *named* dense config too — an unnamed ``VectorParams`` cannot coexist with a
    # named sparse vector in Qdrant, so the isinstance narrowing is part of the claim, not scaffolding.
    assert isinstance(info.config.params.vectors, dict)
    assert set(info.config.params.vectors) == {"dense"}
    assert report.sparse_vector_name == "sparse"
    assert report.points == 2


def test_the_sparse_signal_lands_in_the_vector_not_the_payload(tmp_path: Path) -> None:
    """Queryable, not merely stored: the two columns leave the payload and become the vector."""
    export = write_sparse_export(tmp_path, {"r1": "alpha beta gamma"})
    client = FakeQdrantClient()
    sink(client, vector="both").load(export)

    stored_vector, payload = client.points[_COLLECTION][point_id("r1")]
    assert isinstance(stored_vector["sparse"], SparseVector)
    assert stored_vector["sparse"].indices  # the hashing reference emits terms for this text
    assert len(stored_vector["sparse"].indices) == len(stored_vector["sparse"].values)
    assert "sparse_indices" not in payload
    assert "sparse_values" not in payload


def test_sparse_only_load_writes_no_dense_vector(tmp_path: Path) -> None:
    """A SPLADE-only index: no Embedder on the Export, so there is no dense vector to demand."""
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"}, dense=False)
    client = FakeQdrantClient()
    report = sink(client, vector="sparse").load(export)

    info = client.get_collection(_COLLECTION)
    assert info.config.params.vectors == {}
    assert set(info.config.params.sparse_vectors) == {"sparse"}
    assert report.dimension == 0
    assert report.points == 1
    stored_vector, _ = client.points[_COLLECTION][point_id("r1")]
    assert set(stored_vector) == {"sparse"}


def test_sparse_source_names_the_export_column_pair(tmp_path: Path) -> None:
    """``--sparse-source splade`` reads ``splade_indices``/``splade_values`` and names the vector."""
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    rename_columns(export, {"sparse_indices": "splade_indices", "sparse_values": "splade_values"})
    client = FakeQdrantClient()
    report = sink(client, vector="both", sparse_source="splade").load(export)

    info = client.get_collection(_COLLECTION)
    assert set(info.config.params.sparse_vectors) == {"splade"}
    assert report.sparse_vector_name == "splade"


def test_sparse_vector_name_can_differ_from_the_source_column(tmp_path: Path) -> None:
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    client = FakeQdrantClient()
    sink(client, vector="both", sparse_vector_name="splade-v3").load(export)
    info = client.get_collection(_COLLECTION)
    assert set(info.config.params.sparse_vectors) == {"splade-v3"}


def test_missing_sparse_columns_are_a_loud_export_error(tmp_path: Path) -> None:
    """Asking for a signal the Export does not carry must name the columns, not load silently."""
    export = write_sparse_export(tmp_path, {"r1": "alpha"}, sparse=False)
    with pytest.raises(SinkExportError) as excinfo:
        sink(FakeQdrantClient(), vector="both").load(export)
    assert "sparse_indices" in str(excinfo.value)
    assert "sparse_values" in str(excinfo.value)


def test_ragged_sparse_columns_are_a_loud_export_error(tmp_path: Path) -> None:
    """Parallel columns of different lengths are what a real Qdrant 400s on — caught at the Export."""
    export = write_sparse_export(tmp_path, {"r1": "alpha beta gamma"})
    rewrite_column(export, "sparse_values", pa.array([[1.0]], type=pa.list_(pa.float64())))
    with pytest.raises(SinkExportError) as excinfo:
        sink(FakeQdrantClient(), vector="both").load(export)
    assert "r1" in str(excinfo.value)


def test_duplicate_sparse_term_ids_are_caught_at_the_export(tmp_path: Path) -> None:
    """A repeated term id is a 400 from Qdrant — so it is caught before a single point is written.

    The ragged-column case already fails at the Export; this one is the same class of corruption and
    must not be the exception that gets discovered halfway through a load as an opaque store fault.
    """
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    rewrite_column(export, "sparse_indices", pa.array([[5, 5]], type=pa.list_(pa.int64())))
    client = FakeQdrantClient()
    with pytest.raises(SinkExportError) as excinfo:
        sink(client, vector="both").load(export)
    assert "r1" in str(excinfo.value)
    assert "upsert" not in client.calls


def test_sparse_knobs_are_rejected_when_sparse_is_not_being_loaded(tmp_path: Path) -> None:
    """A knob that silently does nothing is the failure mode this framework refuses everywhere.

    The message must name the *conflict* — the key AND the mode that makes it inert. An "unknown
    config key" would pass a laxer assertion here while telling the operator nothing about why a
    sparse flag they deliberately set loaded no sparse vector.
    """
    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink({"collection": _COLLECTION, "sparse_source": "splade"})
    message = str(excinfo.value)
    assert "sparse_source" in message
    assert "'dense'" in message


def test_vector_name_is_rejected_on_a_sparse_only_load() -> None:
    """The mirror image: there is no dense vector to name, so naming one is inert too."""
    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink({"collection": _COLLECTION, "vector": "sparse", "vector_name": "dense"})
    message = str(excinfo.value)
    assert "vector_name" in message
    assert "'sparse'" in message  # the mode that makes it inert, named — not just "unknown key"


def test_a_dense_and_sparse_name_collision_is_refused() -> None:
    """One point carries ONE vector map, so a shared name silently keeps only one of the two."""
    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink(
            {
                "collection": _COLLECTION,
                "vector": "both",
                "vector_name": "v",
                "sparse_vector_name": "v",
            }
        )
    assert "must differ" in str(excinfo.value)


def test_unknown_vector_mode_is_a_config_error() -> None:
    with pytest.raises(SinkConfigError) as excinfo:
        QdrantSink({"collection": _COLLECTION, "vector": "hybrid"})
    message = str(excinfo.value)
    assert "'hybrid'" in message
    assert "'sparse'" in message and "'both'" in message


def test_an_empty_sparse_vector_loads_rather_than_failing(tmp_path: Path) -> None:
    """A chunk whose text carries no terms is a legitimate point, not a failed load.

    Redaction can empty a chunk's indexable text (ADR-0044), and a loader that treated the
    resulting empty ``SparseVector`` as an error would make a *correct* redaction abort the load.
    """
    export = write_sparse_export(tmp_path, {"r1": "alpha"})
    rewrite_column(export, "sparse_indices", pa.array([[]], type=pa.list_(pa.int64())))
    rewrite_column(export, "sparse_values", pa.array([[]], type=pa.list_(pa.float64())))
    client = FakeQdrantClient()
    report = sink(client, vector="both").load(export)

    assert report.points == 1
    stored_vector, _ = client.points[_COLLECTION][point_id("r1")]
    assert stored_vector["sparse"] == SparseVector(indices=[], values=[])


def test_sparse_load_into_an_existing_dense_only_collection_is_named_not_a_raw_400(
    tmp_path: Path,
) -> None:
    """The collection was created dense-only by an earlier load; the sparse index is not there.

    ``_ensure_collection`` does not touch an existing collection, so without this check the load
    would die on an opaque 400 from the first upsert.
    """
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    client = FakeQdrantClient()
    sink(client).load(export)  # dense-only, as today
    with pytest.raises(SinkLoadError) as excinfo:
        sink(client, vector="both").load(export)
    assert "sparse" in str(excinfo.value)


# --- the invariants the sparse path must not break ------------------------------------------------


def test_dense_only_load_is_unchanged_by_the_sparse_feature(tmp_path: Path) -> None:
    """The off-by-default guarantee: no sparse config, and the columns still ride in the payload.

    Also pins the *digest*: adding sparse to the change detector unconditionally would re-upsert
    every point of every existing collection on the next load.
    """
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    client = FakeQdrantClient()
    report = sink(client).load(export)

    assert client.get_collection(_COLLECTION).config.params.sparse_vectors == {}
    vector, payload = client.points[_COLLECTION][point_id("r1")]
    assert isinstance(vector, list)  # the single unnamed dense vector, exactly as before
    assert payload["sparse_indices"] == [0, 30] or payload["sparse_indices"]
    assert "sparse_values" in payload

    from latence_sink_qdrant.sink import DIGEST_KEY, _digest

    bare = {k: v for k, v in payload.items() if not k.startswith("_latence_")}
    assert payload[DIGEST_KEY] == _digest(vector, bare)
    assert report.sparse_vector_name is None


def test_sparse_reload_upserts_nothing(tmp_path: Path) -> None:
    export = write_sparse_export(tmp_path, {"r1": "alpha beta", "r2": "gamma"})
    client = FakeQdrantClient()
    sink(client, vector="both").load(export)
    again = sink(client, vector="both").load(export)
    assert (again.upserted, again.unchanged, again.deleted) == (0, 2, 0)


def test_a_changed_sparse_vector_alone_is_re_upserted(tmp_path: Path) -> None:
    """The digest must cover the sparse signal, or a re-embed would never reach the store.

    Only the sparse column changes here — the dense vector and every payload column are the bytes
    of the first load — so a digest that ignored sparse would report ``unchanged`` and leave the
    collection serving the old term weights forever.
    """
    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    client = FakeQdrantClient()
    first = sink(client, vector="both").load(export)
    assert first.upserted == 1

    rewrite_column(export, "sparse_indices", pa.array([[3, 9]], type=pa.list_(pa.int64())))
    second = sink(client, vector="both").load(export)
    assert (second.upserted, second.unchanged) == (1, 0)
    vector, _ = client.points[_COLLECTION][point_id("r1")]
    assert vector["sparse"].indices == [3, 9]


def test_purge_removes_the_sparse_point_too(tmp_path: Path) -> None:
    """A Retraction converges on a sparse collection exactly as on a dense one."""
    export = write_sparse_export(tmp_path, {"r1": "alpha", "r2": "beta"})
    client = FakeQdrantClient()
    sink(client, vector="both").load(export)

    survivors = write_sparse_export(tmp_path / "after", {"r1": "alpha"})
    report = sink(client, vector="both").load(survivors)
    assert report.deleted == 1
    assert sink(client, vector="both").verify_absent(["r2"]) == []
    assert point_id("r2") not in client.points[_COLLECTION]


# --- the CLI hop: a flag that stops before the sink is not wired ----------------------------------


@pytest.fixture
def cli_store(monkeypatch: pytest.MonkeyPatch) -> FakeQdrantClient:
    """The CLI builds its sinks over one shared in-memory store, and records their configs."""
    from latence_sink_qdrant import cli

    client = FakeQdrantClient()
    seen: list[dict[str, Any]] = []

    def _factory(config: dict[str, Any]) -> QdrantSink:
        seen.append(config)
        return QdrantSink(config, client=client)

    monkeypatch.setattr(cli, "QdrantSink", _factory)
    client.configs = seen  # type: ignore[attr-defined]  # test-only inspection handle
    return client


def test_cli_loads_named_sparse_vectors_end_to_end(
    tmp_path: Path, cli_store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--vector both --sparse-source sparse`` reaches the store, not just the argument namespace."""
    from latence_sink_qdrant import cli

    export = write_sparse_export(tmp_path, {"r1": "alpha beta"})
    code = cli.main(
        ["load", export, "--collection", _COLLECTION, "--vector", "both", "--sparse-source", "sparse"]
    )
    assert code == 0
    assert cli_store.configs[-1]["vector"] == "both"  # type: ignore[attr-defined]
    info = cli_store.get_collection(_COLLECTION)
    assert set(info.config.params.sparse_vectors) == {"sparse"}
    assert "sparse" in capsys.readouterr().out


def test_cli_without_the_flag_emits_no_sparse_keys(
    tmp_path: Path, cli_store: FakeQdrantClient
) -> None:
    """Unset flags never override — the dense-only command's config is byte-identical to today."""
    from latence_sink_qdrant import cli

    export = write_sparse_export(tmp_path, {"r1": "alpha"})
    assert cli.main(["load", export, "--collection", _COLLECTION]) == 0
    config = cli_store.configs[-1]  # type: ignore[attr-defined]
    assert config == {"collection": _COLLECTION}


def test_cli_rejects_an_unknown_vector_mode(tmp_path: Path) -> None:
    from latence_sink_qdrant import cli

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["load", str(tmp_path), "--vector", "hybrid"])


def test_cli_sparse_vector_name_reaches_the_collection(
    tmp_path: Path, cli_store: FakeQdrantClient
) -> None:
    """The fourth hop of the field→flag→config→store chain, for the name the QUERY side needs."""
    from latence_sink_qdrant import cli

    export = write_sparse_export(tmp_path, {"r1": "alpha"}, dense=False)
    code = cli.main(
        [
            "load",
            export,
            "--collection",
            _COLLECTION,
            "--vector",
            "sparse",
            "--sparse-vector-name",
            "splade-v3",
        ]
    )
    assert code == 0
    info = cli_store.get_collection(_COLLECTION)
    assert set(info.config.params.sparse_vectors) == {"splade-v3"}
    assert info.config.params.vectors == {}


def test_cli_names_the_conflict_when_a_sparse_flag_cannot_apply(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--sparse-source`` without ``--vector`` exits 2 with the conflict named, not a traceback."""
    from latence_sink_qdrant import cli

    export = write_sparse_export(tmp_path, {"r1": "alpha"})
    assert cli.main(["load", export, "--collection", _COLLECTION, "--sparse-source", "x"]) == 2
    assert "sparse_source" in capsys.readouterr().err
