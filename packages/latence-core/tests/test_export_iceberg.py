"""The OPTIONAL Iceberg-table Export output (#184, ADR-0017).

Every test here drives the REAL client (``pyiceberg.catalog.rest.RestCatalog``) over real HTTP
against a REAL, local, offline Iceberg REST catalog (``tests/_iceberg_rest.local_rest_catalog`` —
the OSS ``SqlCatalog`` behind the spec's routes). Nothing about the catalog, the metadata, the Avro
manifests or the snapshot commits is simulated, so a claim that "the corpus is now a governed
table" is proved by reading the rows back out of Iceberg, not by asserting that a flag was set.

Covered:

* enabled ⇒ the corpus is registered as an Iceberg table in the catalog and scans back to exactly
  the exported rows;
* the table schema mirrors the Parquet consumer contract column-for-column (including the opt-in
  signal columns) and carries the contract version stamp on the table AND on the snapshot;
* DISABLED IS BYTE-IDENTICAL — ``records.jsonl`` / ``records.parquet`` are byte-for-byte the same
  with the block absent and with it present, and nothing extra is written when it is absent;
* delta runs: ``append`` adds a snapshot (rows accumulate), ``overwrite`` replaces the contents in
  one commit — the two documented dispositions, asserted on the data that comes back;
* malformed config fails loudly at WIRING time, before any record is read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
from _iceberg_rest import local_rest_catalog
from latence_core.contracts import SCHEMA_VERSION, ChunkRecord, Classification, Provenance
from latence_core.errors import ConfigError, ProviderError
from latence_core.iceberg import (
    _PUBLISHED_DATA_FILE,
    PROP_CONTRACT_VERSION,
    PROP_MODE,
    PROP_RECORD_VERSIONS,
    PROP_SOURCE,
    IcebergCorpusTable,
)
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from pyiceberg.catalog.rest import RestCatalog

if TYPE_CHECKING:
    from collections.abc import Callable

_NAMESPACE = "lake.latence"
_TABLE = "corpus"


def _chunk(text: str, record_id: str) -> ChunkRecord:
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


def _iceberg_config(uri: str, mode: str = "append", **extra: Any) -> dict[str, Any]:
    return {
        "iceberg": {
            "uri": uri,
            "namespace": _NAMESPACE,
            "table": _TABLE,
            "mode": mode,
            **extra,
        }
    }


def _scan(uri: str) -> list[dict[str, Any]]:
    catalog = RestCatalog("assert", uri=uri)
    table = catalog.load_table((*_NAMESPACE.split("."), _TABLE))
    rows: list[dict[str, Any]] = table.scan().to_arrow().to_pylist()
    return sorted(rows, key=lambda row: str(row["record_id"]))


def _load_table(uri: str) -> Any:
    return RestCatalog("assert", uri=uri).load_table((*_NAMESPACE.split("."), _TABLE))


def _table_data_dir(uri: str) -> str:
    """Where the Export must publish its data file: the target TABLE's own location (R6).

    Requires the catalog to be up, so a test that needs this after the server has stopped captures
    it inside the ``local_rest_catalog`` block (or uses :func:`_warehouse_data_files`).
    """
    return f"{str(_load_table(uri).location()).rstrip('/')}/data"


def _warehouse_data_files(catalog_root: Path) -> list[str]:
    """Every Iceberg data file the Export published anywhere in the catalog's warehouse.

    Searched from the warehouse ROOT rather than one known directory, so a data file published to
    the wrong place inside the warehouse is still found (and a test asserting "none" cannot pass
    just because it looked in a directory that does not exist).
    """
    return sorted(p.name for p in (catalog_root / "warehouse").rglob("records-*.parquet"))


def test_enabled_registers_the_corpus_as_an_iceberg_table(tmp_path: Path) -> None:
    """The headline AC: enable one config block ⇒ the corpus IS a governed table."""
    with local_rest_catalog(tmp_path / "catalog") as uri:
        out = tmp_path / "export"
        export = JsonlParquetExport(_iceberg_config(uri))
        records = [_chunk("alpha text", "c1"), _chunk("beta text", "c2")]
        uris = export.export(records, Storage(), str(out))

        rows = _scan(uri)
        assert [row["record_id"] for row in rows] == ["c1", "c2"]
        assert [row["content"] for row in rows] == ["alpha text", "beta text"]
        # The registered data file is reported as an output URI, it is a real file on disk, and it
        # lives in the TABLE's own location — never in the ephemeral run tree (R6).
        data_dir = _table_data_dir(uri)
        data_uris = [u for u in uris if u.startswith(f"{data_dir}/")]
        assert len(data_uris) == 1, uris
        assert Path(data_uris[0].removeprefix("file://")).exists()
        assert _warehouse_data_files(tmp_path / "catalog") == [data_uris[0].rsplit("/", 1)[-1]]
        # ADR-0017: the deliverable is still the file, and it is untouched.
        assert (out / "records.parquet").exists()
        assert (out / "records.jsonl").exists()


def test_table_schema_mirrors_the_parquet_contract_and_carries_the_version_stamp(
    tmp_path: Path,
) -> None:
    """Same columns as the Parquet consumer contract, same version stamp — including opt-ins."""
    with local_rest_catalog(tmp_path / "catalog") as uri:
        out = tmp_path / "export"
        config = _iceberg_config(uri)
        # An opt-in signal column must ride along: the table mirrors what THIS Export writes.
        config["embedder"] = {"provider": "embedding.hashing", "config": {"dimension": 8}}
        export = JsonlParquetExport(config)
        export.export([_chunk("alpha", "c1")], Storage(), str(out))

        parquet_columns = pq.read_table(  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
            out / "records.parquet"
        ).schema.names
        table = _load_table(uri)
        assert [field.name for field in table.schema().fields] == parquet_columns
        assert "embedding" in parquet_columns

        # Version-stamped consistently with the Parquet export: the table property equals the
        # contract constant that the Parquet ``schema_version`` column carries per row.
        assert table.properties[PROP_CONTRACT_VERSION] == str(SCHEMA_VERSION)
        assert table.properties[PROP_RECORD_VERSIONS] == str(SCHEMA_VERSION)
        assert table.properties[PROP_MODE] == "append"
        assert table.properties[PROP_SOURCE].endswith("records.parquet")
        parquet_table = pq.read_table(  # type: ignore[no-untyped-call]  # pyarrow ships no stubs
            out / "records.parquet"
        )
        parquet_versions = set(parquet_table.column("schema_version").to_pylist())
        assert parquet_versions == {SCHEMA_VERSION}
        # The stamp is on the SNAPSHOT too, so a governance reader can date any historical commit.
        snapshot = table.current_snapshot()
        assert snapshot is not None
        assert snapshot.summary is not None
        assert snapshot.summary[PROP_CONTRACT_VERSION] == str(SCHEMA_VERSION)


def test_disabled_leaves_the_export_byte_identical(tmp_path: Path) -> None:
    """OFF BY DEFAULT: the existing deliverable is byte-for-byte unchanged, and nothing extra."""
    records = [_chunk("alpha text", "c1"), _chunk("beta text", "c2")]

    plain = tmp_path / "plain"
    plain_uris = JsonlParquetExport().export(records, Storage(), str(plain))
    assert plain_uris == [f"{plain}/records.jsonl", f"{plain}/records.parquet"]
    assert sorted(p.name for p in plain.iterdir()) == ["records.jsonl", "records.parquet"]

    with local_rest_catalog(tmp_path / "catalog") as uri:
        enabled = tmp_path / "enabled"
        JsonlParquetExport(_iceberg_config(uri)).export(records, Storage(), str(enabled))

    for name in ("records.jsonl", "records.parquet"):
        assert (plain / name).read_bytes() == (enabled / name).read_bytes(), name


def test_disabled_never_imports_pyiceberg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Off by default costs nothing: the optional dependency is not even imported."""
    import builtins

    real_import = builtins.__import__

    def guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pyiceberg"):
            msg = f"pyiceberg must not be imported when the Iceberg output is off (got {name})"
            raise AssertionError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    JsonlParquetExport().export([_chunk("alpha", "c1")], Storage(), str(tmp_path / "out"))


def test_append_mode_accumulates_snapshots_across_delta_runs(tmp_path: Path) -> None:
    """``append`` is the delta-run disposition: each run's records are added, both survive."""
    with local_rest_catalog(tmp_path / "catalog") as uri:
        config = _iceberg_config(uri, mode="append")
        JsonlParquetExport(config).export([_chunk("alpha", "c1")], Storage(), str(tmp_path / "r1"))
        JsonlParquetExport(config).export([_chunk("beta", "c2")], Storage(), str(tmp_path / "r2"))

        assert [row["record_id"] for row in _scan(uri)] == ["c1", "c2"]
        assert len(_load_table(uri).history()) == 2


def test_overwrite_mode_replaces_the_table_contents(tmp_path: Path) -> None:
    """``overwrite`` is the full-re-export disposition: the second run IS the table."""
    with local_rest_catalog(tmp_path / "catalog") as uri:
        config = _iceberg_config(uri, mode="overwrite")
        JsonlParquetExport(config).export(
            [_chunk("alpha", "c1"), _chunk("beta", "c2")], Storage(), str(tmp_path / "r1")
        )
        assert [row["record_id"] for row in _scan(uri)] == ["c1", "c2"]

        JsonlParquetExport(config).export([_chunk("gamma", "c3")], Storage(), str(tmp_path / "r2"))
        assert [row["record_id"] for row in _scan(uri)] == ["c3"]
        # Time travel still reaches the superseded snapshot — the old data file is unreferenced,
        # not destroyed (the documented retention posture).
        assert len(_load_table(uri).history()) >= 2


def test_the_export_tree_is_byte_identical_across_two_fresh_runs(tmp_path: Path) -> None:
    """DETERMINISM: enabling Iceberg must not make the export deliverable non-reproducible.

    The exact scenario ``latence stack validate`` runs: two fresh runs over identical records into
    two independent run roots, compared with the REAL harness comparator
    (``stacks.validate._export_artifacts``, which walks ``export/`` RECURSIVELY and keys by the
    run-relative path) under the REAL check (``_check_determinism`` is a plain ``==``).

    Pre-fix the data file was named ``records-<uuid4>.parquet`` *inside the export tree*, so the two
    runs produced disjoint keys and the harness reported ``export diverged`` on a perfectly
    deterministic pipeline. The name is content-addressed now (identical corpus bytes ⇒ identical
    name), and since R6 the file lives in the TABLE's location rather than the export tree — so the
    deliverable is untouched by the optional block, and determinism is asserted where the artifact
    now actually lands.
    """
    from latence_core.stacks.validate import _export_artifacts

    records = [_chunk("alpha text", "c1"), _chunk("beta text", "c2")]
    published: list[str] = []
    with local_rest_catalog(tmp_path / "catalog") as uri:
        config = _iceberg_config(uri)
        data_dir = ""
        for run in ("run-1", "run-2"):
            uris = JsonlParquetExport(config).export(
                records, Storage(), str(tmp_path / run / "export")
            )
            data_dir = data_dir or _table_data_dir(uri)
            published.append(next(u for u in uris if u.startswith(f"{data_dir}/")))
        # ``append`` still accumulated a snapshot per run (the documented delta disposition is
        # untouched by content addressing), and both runs' rows scan back.
        assert len(_load_table(uri).history()) == 2

    # The derived artifact is deterministic: two independent runs over identical records
    # content-address onto ONE path (non-vacuous — the URIs are real and were really published).
    assert len(published) == 2
    assert published[0] == published[1], published
    assert _warehouse_data_files(tmp_path / "catalog") == [published[0].rsplit("/", 1)[-1]]

    primary = _export_artifacts(Storage(), str(tmp_path / "run-1"))
    second = _export_artifacts(Storage(), str(tmp_path / "run-2"))
    # The export deliverable is non-empty and carries nothing Iceberg-specific at all: enabling the
    # block cannot perturb the tree ``stack validate`` compares (R6).
    assert sorted(primary) == ["records.jsonl", "records.parquet"]
    assert primary == second, (
        f"export diverged — primary_only={sorted(set(primary) - set(second))}, "
        f"second_only={sorted(set(second) - set(primary))}"
    )


def test_the_data_file_name_is_content_addressed(tmp_path: Path) -> None:
    """The name IS the sha256 of the published bytes: stable for equal content, distinct else."""
    import hashlib

    def publish(run: str, text: str) -> str:
        with local_rest_catalog(tmp_path / f"catalog-{run}") as uri:
            out = tmp_path / run / "export"
            uris = JsonlParquetExport(_iceberg_config(uri)).export(
                [_chunk(text, "c1")], Storage(), str(out)
            )
            data_dir = _table_data_dir(uri)
        return next(u for u in uris if u.startswith(f"{data_dir}/"))

    same_a = publish("a", "alpha text")
    same_b = publish("b", "alpha text")
    other = publish("c", "a completely different body of text")

    assert same_a.rsplit("/", 1)[-1] == same_b.rsplit("/", 1)[-1]
    assert other.rsplit("/", 1)[-1] != same_a.rsplit("/", 1)[-1]
    # And the token really is the digest of the file the catalog points at.
    published = Path(same_a.removeprefix("file://")).read_bytes()
    digest = hashlib.sha256(published).hexdigest()
    assert same_a.rsplit("/", 1)[-1] == f"records-{digest}.parquet"


def test_a_purge_of_the_run_artifacts_leaves_the_registered_table_readable(
    tmp_path: Path,
) -> None:
    """R6: the catalog's data file must NOT live in the tree the framework's own Purge deletes.

    The Export writes into ``<storage_uri>/_latence/runs/<run_id>/export`` (``Runner._export_dir``)
    and ``DeltaRunner._purge_run_artifacts`` — the GDPR-erasure step of EVERY purge, for ANY
    document — ``remove_tree``s every entry under ``_latence/runs``. Publishing the Iceberg data
    file under the Export's ``out_dir`` therefore put a file the committed snapshot references by
    absolute path inside the most ephemeral directory in the system: one unrelated purge and
    ``table.scan()`` raised ``FileNotFoundError`` forever (in ``append`` mode the re-export lands
    under a NEW run id, so the dangling path stays referenced).

    Driven through the REAL purge method with the REAL run layout — not a hand-rolled rmtree.
    """
    from latence_core.delta_runner import DeltaRunner
    from latence_core.pipeline import CapabilityKind, Pipeline, Stage

    storage_uri = str(tmp_path / "lake")
    run_dir = f"{storage_uri}/_latence/runs/run-1"
    pipeline = Pipeline(
        name="p",
        storage_uri=storage_uri,
        stages=[Stage(name="export", capability=CapabilityKind.EXPORT, provider="jsonl_parquet")],
    )

    with local_rest_catalog(tmp_path / "catalog") as uri:
        JsonlParquetExport(_iceberg_config(uri)).export(
            [_chunk("alpha text", "c1")], Storage(), f"{run_dir}/export"
        )
        assert [row["record_id"] for row in _scan(uri)] == ["c1"]

        DeltaRunner()._purge_run_artifacts(pipeline)  # noqa: SLF001 - the real erasure step
        assert not Path(run_dir).exists(), "the purge must really have wiped the run tree"

        # The derived table is still readable: its data file never lived in the purged tree.
        assert [row["record_id"] for row in _scan(uri)] == ["c1"]


def test_the_jsonl_deliverable_matches_the_registered_table(tmp_path: Path) -> None:
    """Same data, two faces: every JSONL record_id is in the table and vice versa."""
    with local_rest_catalog(tmp_path / "catalog") as uri:
        out = tmp_path / "export"
        records = [_chunk(f"text {i}", f"c{i}") for i in range(5)]
        JsonlParquetExport(_iceberg_config(uri)).export(records, Storage(), str(out))

        jsonl_ids = {
            json.loads(line)["record_id"]
            for line in (out / "records.jsonl").read_text().splitlines()
        }
        assert {str(row["record_id"]) for row in _scan(uri)} == jsonl_ids


@pytest.mark.parametrize(
    "block",
    [
        pytest.param("not-a-mapping", id="not-a-mapping"),
        pytest.param({}, id="missing-uri"),
        pytest.param({"uri": ""}, id="empty-uri"),
        pytest.param({"uri": "http://x", "mode": "upsert"}, id="unknown-mode"),
        pytest.param({"uri": "http://x", "namespace": ""}, id="empty-namespace"),
        pytest.param({"uri": "http://x", "namespace": 7}, id="namespace-wrong-type"),
        pytest.param({"uri": "http://x", "table": ""}, id="empty-table"),
        pytest.param({"uri": "http://x", "properties": "nope"}, id="properties-wrong-type"),
    ],
)
def test_malformed_config_fails_at_wiring_time(block: Any) -> None:
    """A bad catalog config is caught before a single record is read, never mid-pipeline."""
    with pytest.raises(ConfigError):
        JsonlParquetExport({"iceberg": block})


def test_a_catalog_fault_on_table_load_surfaces_as_a_typed_provider_error(tmp_path: Path) -> None:
    """ADR-0034: a transient catalog fault is a ProviderError, never a raw pyiceberg traceback.

    A second run loads the table the first run created; the catalog answers 503 (a routine
    transient server fault). Pre-fix, ``pyiceberg.exceptions.ServiceUnavailableError`` escaped
    ``JsonlParquetExport.export`` unwrapped, because the ``_catalog()`` / ``_ensure_table()`` calls
    sat outside ``register``'s normalising ``except``.
    """
    faulting = {"on": False}

    def fault(method: str, path: str) -> int | None:
        if faulting["on"] and method == "GET" and path.endswith(f"/tables/{_TABLE}"):
            return 503
        return None

    with local_rest_catalog(tmp_path / "catalog", fault=fault) as uri:
        export = JsonlParquetExport(_iceberg_config(uri))
        export.export([_chunk("alpha", "c1")], Storage(), str(tmp_path / "export1"))

        faulting["on"] = True
        with pytest.raises(ProviderError) as caught:
            export.export([_chunk("beta", "c2")], Storage(), str(tmp_path / "export2"))

    message = str(caught.value)
    assert uri in message
    assert _TABLE in message
    # The native cause is preserved (diagnosable), not the surfaced type (ADR-0034).
    assert type(caught.value.__cause__).__module__.startswith("pyiceberg")
    # The deliverable itself is complete — the optional table is what failed.
    assert (tmp_path / "export2" / "records.parquet").exists()


def test_a_lost_create_race_reloads_the_table_instead_of_failing(tmp_path: Path) -> None:
    """A concurrent writer winning the create race must not fail this export.

    The catalog performs this client's create and then answers 409, exactly as it would if another
    exporter had created the table a moment earlier. Pre-fix, the raw
    ``pyiceberg.exceptions.TableAlreadyExistsError`` escaped ``JsonlParquetExport.export``.
    """
    with local_rest_catalog(tmp_path / "catalog", lose_first_create=True) as uri:
        out = tmp_path / "export"
        JsonlParquetExport(_iceberg_config(uri)).export(
            [_chunk("alpha", "c1")], Storage(), str(out)
        )
        rows = _scan(uri)

    assert [row["record_id"] for row in rows] == ["c1"]


def test_a_run_that_drops_a_signal_column_is_refused_not_null_backfilled(tmp_path: Path) -> None:
    """Schema drift in the NARROWING direction must fail loudly, not NULL out the table.

    Run 1 writes ``embedding``; run 2 has the embedder unwired (a config edit, a second pipeline
    writing the same table, a delta run missing the signal). Pre-fix, ``tx.add_files`` accepted the
    narrower Parquet and back-filled the missing column with NULLs — in ``overwrite`` mode the whole
    governed table's ``embedding`` column silently became NULL while the schema still advertised it,
    with exit 0 and no warning (ADR-0034: no silent errors).
    """
    with local_rest_catalog(tmp_path / "catalog") as uri:
        config = _iceberg_config(uri, mode="overwrite")
        config["embedder"] = {"provider": "embedding.hashing", "config": {"dimension": 8}}
        records = [_chunk("alpha", "c1"), _chunk("beta", "c2")]
        JsonlParquetExport(config).export(records, Storage(), str(tmp_path / "r1"))
        assert all(row["embedding"] is not None for row in _scan(uri))

        narrowed = _iceberg_config(uri, mode="overwrite")  # same table, embedder unwired
        with pytest.raises(ProviderError) as caught:
            JsonlParquetExport(narrowed).export(records, Storage(), str(tmp_path / "r2"))

        message = str(caught.value)
        assert "embedding" in message
        assert _TABLE in message
        # The governed table is untouched: the previous run's vectors are still there.
        rows = _scan(uri)
        assert [row["record_id"] for row in rows] == ["c1", "c2"]
        assert all(row["embedding"] is not None for row in rows)
        # ADR-0017: the deliverable itself was still written — the optional table is what refused.
        assert (tmp_path / "r2" / "records.parquet").exists()


def test_a_run_that_adds_a_signal_column_is_refused_with_a_typed_error(tmp_path: Path) -> None:
    """The mirror direction: widening is refused too, and by OUR typed error, not pyiceberg's."""
    with local_rest_catalog(tmp_path / "catalog") as uri:
        records = [_chunk("alpha", "c1")]
        JsonlParquetExport(_iceberg_config(uri)).export(records, Storage(), str(tmp_path / "r1"))

        widened = _iceberg_config(uri)
        widened["embedder"] = {"provider": "embedding.hashing", "config": {"dimension": 8}}
        with pytest.raises(ProviderError) as caught:
            JsonlParquetExport(widened).export(records, Storage(), str(tmp_path / "r2"))

        assert "embedding" in str(caught.value)
        assert [row["record_id"] for row in _scan(uri)] == ["c1"]


def _iceberg_data_files(catalog_root: Path) -> list[str]:
    """Every published Iceberg data file in the catalog's warehouse (where R6 moved them)."""
    return _warehouse_data_files(catalog_root)


def _commit_fault(status: int) -> Callable[[str, str], int | None]:
    """Answer the catalog's COMMIT endpoint (and only that) with ``status``.

    The commit route is ``POST /v1/namespaces/<ns>/tables/<table>``; table CREATE is the sibling
    ``POST .../tables``. Faulting only the former reproduces the operationally common case: the
    table resolves fine, the corpus copy is published, and the snapshot commit is what fails.
    """

    def fault(method: str, path: str) -> int | None:
        if method == "POST" and path.endswith(f"/tables/{_TABLE}"):
            return status
        return None

    return fault


def test_a_failed_commit_leaves_no_orphan_data_file_in_the_deliverable(tmp_path: Path) -> None:
    """A commit that definitively did not land must not leave its corpus-sized copy behind.

    ``register`` publishes the data file BEFORE opening the transaction, so a commit failure used
    to leave a full byte-for-byte copy of the corpus (``content`` included) in the table's location
    that no manifest references, and that ``expire_snapshots`` can never collect (it only deletes
    files recorded in an expired snapshot's manifests). The operator's ``remove_orphan_files``
    could eventually reach it, but "eventually, if they run it" is not a disposal guarantee.
    """
    with local_rest_catalog(tmp_path / "catalog", fault=_commit_fault(503)) as uri:
        out = tmp_path / "run"
        with pytest.raises(ProviderError) as caught:
            JsonlParquetExport(_iceberg_config(uri)).export(
                [_chunk("alpha", "c1")], Storage(), str(out)
            )

    # The deliverable is intact (ADR-0017) — only the derived, uncommitted copy is gone.
    assert (out / "records.parquet").exists()
    assert _iceberg_data_files(tmp_path / "catalog") == []
    assert "removed" in str(caught.value)


def test_an_unknown_commit_outcome_keeps_the_data_file_on_purpose(tmp_path: Path) -> None:
    """The one case where the copy must SURVIVE: the commit may actually have landed.

    A 5xx that pyiceberg maps to ``CommitStateUnknownException`` means the catalog may have applied
    the commit before the response was lost. Deleting a file a live snapshot references would
    corrupt the table, so the file is kept and the typed error says so.
    """
    with local_rest_catalog(tmp_path / "catalog", fault=_commit_fault(500)) as uri:
        out = tmp_path / "run"
        with pytest.raises(ProviderError) as caught:
            JsonlParquetExport(_iceberg_config(uri)).export(
                [_chunk("alpha", "c1")], Storage(), str(out)
            )

    assert len(_iceberg_data_files(tmp_path / "catalog")) == 1
    assert "UNKNOWN" in str(caught.value)


def _warehouse_leftovers(catalog_root: Path) -> list[str]:
    """Every file of ANY name the Export left in the warehouse — temp copies included.

    ``_warehouse_data_files`` only sees finalized ``records-<digest>.parquet`` names, so it cannot
    catch a half-published ``records.parquet.<token>.tmp``. This one globs everything and keeps
    what carries corpus bytes.
    """
    return sorted(
        str(p.relative_to(catalog_root))
        for p in (catalog_root / "warehouse").rglob("*")
        if p.is_file() and "records.parquet" in p.name
    )


def _fail_publishing_the_data_file(
    monkeypatch: pytest.MonkeyPatch, *, and_sweep_fails: bool = False
) -> None:
    """Make the data-file PUBLICATION fail — the step before the catalog is ever contacted.

    A transport drop / ENOSPC / permission-denied rename on the warehouse side, i.e. the failure
    mode ``_publish_data_file`` sits exposed to: the corpus has already been streamed into
    ``temp_uri``, and the rename that would give it its content-addressed name is what dies.
    """
    real_finalize = Storage.finalize_atomic
    real_remove = Storage.remove

    def flaky_finalize(self: Storage, tmp_uri: str, uri: str) -> None:
        if "/data/" in uri and uri.endswith(".parquet"):
            raise OSError("simulated transport drop")
        real_finalize(self, tmp_uri, uri)

    def flaky_remove(self: Storage, uri: str) -> None:
        if uri.endswith(".tmp"):
            raise PermissionError("warehouse refused the delete")
        real_remove(self, uri)

    monkeypatch.setattr(Storage, "finalize_atomic", flaky_finalize)
    if and_sweep_fails:
        monkeypatch.setattr(Storage, "remove", flaky_remove)


def test_a_failed_data_file_publication_leaves_no_orphan_temp_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure DURING the publication must not strand the corpus-sized temp copy either.

    The commit-failure path was covered; the publication itself was not. ``_publish_data_file``
    streams the whole corpus (chunk ``content`` and all) into ``Storage.temp_uri`` and only then
    renames it, so a failure in between left a full-corpus ``…records.parquet.<token>.tmp`` in the
    customer's warehouse that no snapshot references and no ``expire_snapshots`` can collect — and
    because ``temp_uri`` mints a FRESH random token per call, every retry against a flaky store
    dropped another distinct copy. The typed error also reported the path as ``<not yet
    published>``, so the operator was never told what to sweep.
    """
    _fail_publishing_the_data_file(monkeypatch)
    with local_rest_catalog(tmp_path / "catalog") as uri:
        out = tmp_path / "run"
        export = JsonlParquetExport(_iceberg_config(uri))
        for _ in range(3):  # retries against a flaky store must not accumulate copies
            with pytest.raises(ProviderError) as caught:
                export.export([_chunk("SECRET alpha", "c1")], Storage(), str(out))

    assert _warehouse_leftovers(tmp_path / "catalog") == []
    assert _iceberg_data_files(tmp_path / "catalog") == []
    # The deliverable itself is untouched (ADR-0017).
    assert (out / "records.parquet").exists()
    # ADR-0034: the typed error names the real path and its disposal, never "<not yet published>".
    message = str(caught.value)
    assert "<not yet published>" not in message
    assert ".tmp" in message
    assert "removed" in message


def test_a_failed_publication_whose_sweep_also_fails_names_the_path_to_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the temp copy cannot be swept, the operator is TOLD which path still holds the corpus.

    Best-effort cleanup that silently gives up would be exactly the silent error ADR-0034 forbids:
    the file is a full copy of the corpus text, so the failure to remove it rides along on the same
    typed error rather than disappearing.
    """
    _fail_publishing_the_data_file(monkeypatch, and_sweep_fails=True)
    with local_rest_catalog(tmp_path / "catalog") as uri:
        out = tmp_path / "run"
        with pytest.raises(ProviderError) as caught:
            JsonlParquetExport(_iceberg_config(uri)).export(
                [_chunk("SECRET alpha", "c1")], Storage(), str(out)
            )

    leftovers = _warehouse_leftovers(tmp_path / "catalog")
    assert len(leftovers) == 1, leftovers
    message = str(caught.value)
    assert Path(leftovers[0]).name in message
    assert "could not be removed" in message
    assert "simulated transport drop" in message  # the original failure is never masked


def test_a_failed_commit_never_deletes_a_data_file_an_earlier_run_committed(
    tmp_path: Path,
) -> None:
    """Cleanup is scoped to what THIS run published — never to a file a snapshot already points at.

    Two runs of identical bytes into the same table content-address onto the SAME path. The
    first run commits it; the second run's commit fails. Removing that path would break the first
    run's snapshot, so only a file this run actually created may be discarded.
    """
    faulting = {"on": False}

    def fault(method: str, path: str) -> int | None:
        if faulting["on"]:
            return _commit_fault(503)(method, path)
        return None

    records = [_chunk("alpha", "c1")]
    out = tmp_path / "run"
    with local_rest_catalog(tmp_path / "catalog", fault=fault) as uri:
        export = JsonlParquetExport(_iceberg_config(uri))
        export.export(records, Storage(), str(out))
        committed = _iceberg_data_files(tmp_path / "catalog")
        assert len(committed) == 1

        faulting["on"] = True
        with pytest.raises(ProviderError):
            export.export(records, Storage(), str(out))

        assert _iceberg_data_files(tmp_path / "catalog") == committed
        assert [row["record_id"] for row in _scan(uri)] == ["c1"]


def _stale_existence_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the published data file's ``exists()`` probe answer with a STALE ``False``.

    That is precisely what a concurrent publisher observes: two Exports of identical corpus content
    content-address onto the SAME path, and the one whose ``exists()`` runs before the other's
    ``finalize_atomic`` lands is told the file is absent. It then re-finalizes the same path and
    believes it created it. Nothing about the code under test is stubbed here — only the
    interleaving is pinned, so the race is reproducible instead of a 1-in-N flake.
    """
    real_exists = Storage.exists

    def racing_exists(self: Storage, uri: str) -> bool:
        if "/data/" in uri and _PUBLISHED_DATA_FILE.search(uri):
            return False
        return bool(real_exists(self, uri))

    monkeypatch.setattr(Storage, "exists", racing_exists)


def test_a_concurrent_publisher_never_deletes_a_committed_runs_data_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost publish race must not let a failed commit destroy a COMMITTED table (R7-…-4).

    ``published_here`` was decided by a check-then-act pair — ``exists(data_uri)`` then
    ``finalize_atomic(tmp_uri, data_uri)``. Two concurrent Exports of identical content both see
    ``exists() == False``, both rename onto the same content-addressed path, and both claim
    authorship. When the second one's commit then failed, ``_discard_unreferenced`` deleted the
    shared file even though the FIRST run had already committed a snapshot referencing it: the
    manifest records that absolute path forever, so the governed table became permanently
    unreadable while the run that broke it reported success.
    """
    faulting = {"on": False}

    def fault(method: str, path: str) -> int | None:
        return _commit_fault(503)(method, path) if faulting["on"] else None

    records = [_chunk("alpha", "c1")]
    out = tmp_path / "run"
    with local_rest_catalog(tmp_path / "catalog", fault=fault) as uri:
        export = JsonlParquetExport(_iceberg_config(uri))
        export.export(records, Storage(), str(out))  # run A: publishes AND commits
        committed = _iceberg_data_files(tmp_path / "catalog")
        assert len(committed) == 1

        # Run B: identical corpus, its existence probe answers with the pre-rename world, and its
        # commit fails. It must NOT sweep the file A's snapshot points at.
        _stale_existence_check(monkeypatch)
        faulting["on"] = True
        with pytest.raises(ProviderError) as caught:
            export.export(records, Storage(), str(out))

        assert _iceberg_data_files(tmp_path / "catalog") == committed
        assert [row["record_id"] for row in _scan(uri)] == ["c1"]  # A's table still reads

    # ADR-0034: the operator is told the file was kept, and why.
    assert "KEPT" in str(caught.value)


def test_an_unestablishable_reference_keeps_the_data_file_and_names_it(tmp_path: Path) -> None:
    """When the catalog cannot say whether a snapshot uses the file, the file SURVIVES.

    The other half of R7-data_contracts_sinks-4: a deletion that cannot be justified is not a
    cleanup. If the metadata refresh AND the manifest walk both fail there is no evidence the file
    is unreferenced, so the safe answer is the same one ``CommitStateUnknownException`` gets — keep
    it, and tell the operator the exact path and why (ADR-0034).
    """

    class _UnreadableTable:
        """A table whose metadata cannot be reached (catalog down mid-failure-handling)."""

        def refresh(self) -> Any:
            raise OSError("catalog unreachable")

        @property
        def metadata(self) -> Any:
            raise OSError("catalog unreachable")

    target = tmp_path / "data" / f"records-{'a' * 64}.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corpus bytes")
    table_obj = IcebergCorpusTable(
        _iceberg_config("http://localhost:1")["iceberg"], default_table=_TABLE
    )

    assert table_obj._snapshot_references(_UnreadableTable(), str(target)) is None
    note = table_obj._discard_unreferenced(
        Storage(), str(target), RuntimeError("commit rejected"), _UnreadableTable()
    )

    assert target.exists()  # the whole point: unprovable ⇒ untouched
    assert "KEPT" in note
    assert str(target) in note


# -- Purge reaches the derived table (audit R7-invariants-1) ---------------------------------


def _delta_pipeline(store: str, docs_dir: Path, catalog_uri: str) -> Any:
    """A real Source->Parse->Chunk->Export pipeline whose Export publishes an Iceberg table."""
    from latence_core.pipeline import Pipeline

    iceberg = json.dumps(
        {"uri": catalog_uri, "namespace": _NAMESPACE, "table": _TABLE, "mode": "append"}
    )
    return Pipeline.from_yaml_text(
        f"""
name: iceberg-purge-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}', sensitivity: internal, extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
    config: {{iceberg: {iceberg}}}
"""
    )


def test_purge_erases_the_purged_document_from_the_derived_iceberg_table(tmp_path: Path) -> None:
    """A Purge must reach the catalog warehouse — in the DEFAULT ``append`` mode (R7).

    Before the fix the purge run merely APPENDED the survivors as a second data file: the first
    data file (the whole pre-purge corpus) stayed referenced by the table's CURRENT snapshot, so an
    ordinary ``table.scan()`` — not time travel — still returned the erased document's text, and its
    bytes stayed in the warehouse forever, while the tutorial promised removal "everywhere".
    """
    import hashlib

    from latence_core.delta_runner import DeltaRunner

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "purged.txt").write_text("Alice Marchetti works at Acme.\n", encoding="utf-8")
    (docs / "keep.txt").write_text("Bob Ndiaye is here.\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"

    with local_rest_catalog(tmp_path / "catalog") as uri:
        pipeline = _delta_pipeline(store, docs, uri)
        DeltaRunner().apply_delta(pipeline, run_id="v0")

        # The pre-purge table really does carry the document that is about to be erased.
        before = " ".join(str(row["content"]) for row in _scan(uri))
        assert "Alice Marchetti" in before
        assert "Bob Ndiaye" in before

        purged_id = "sha256:" + hashlib.sha256((docs / "purged.txt").read_bytes()).hexdigest()
        DeltaRunner().purge(pipeline, run_id="pur", document_ids=[purged_id])

        after = " ".join(str(row["content"]) for row in _scan(uri))
        # 1. The erased document is gone from the table's CURRENT snapshot...
        assert "Alice Marchetti" not in after
        # 2. ...and the survivor is still there: the erasure is total but the purge's own re-run
        #    republishes the surviving corpus, so `append` converges without a mode change.
        assert "Bob Ndiaye" in after

    # 3. The bytes are physically gone from the catalog warehouse too — a delete-all snapshot that
    #    left the corpus copy on disk would not be a GDPR erasure.
    leaked = [
        path
        for path in (tmp_path / "catalog").rglob("*.parquet")
        if b"Alice Marchetti" in path.read_bytes()
    ]
    assert leaked == [], f"purged text survived in the warehouse: {[str(p) for p in leaked]}"


def test_purge_without_an_iceberg_block_never_touches_pyiceberg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OFF BY DEFAULT survives the new purge hook: no block ⇒ no import, no behaviour change."""
    import builtins
    import hashlib

    from latence_core.delta_runner import DeltaRunner
    from latence_core.pipeline import Pipeline

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Alice works at Acme.\n", encoding="utf-8")
    store = f"file://{tmp_path}/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: plain-slice
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', sensitivity: internal, extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: export
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
"""
    )
    DeltaRunner().apply_delta(pipeline, run_id="v0")

    real_import = builtins.__import__

    def guard(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "pyiceberg":
            msg = "pyiceberg must not be imported by a purge on a stack with no iceberg block"
            raise AssertionError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    doc_id = "sha256:" + hashlib.sha256((docs / "a.txt").read_bytes()).hexdigest()
    report = DeltaRunner().purge(pipeline, run_id="pur", document_ids=[doc_id])
    assert report.delta is not None
    assert report.delta.documents_deleted == 1
