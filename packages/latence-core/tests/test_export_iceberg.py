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
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
from _iceberg_rest import local_rest_catalog
from latence_core.contracts import SCHEMA_VERSION, ChunkRecord, Classification, Provenance
from latence_core.errors import ConfigError
from latence_core.iceberg import (
    PROP_CONTRACT_VERSION,
    PROP_MODE,
    PROP_RECORD_VERSIONS,
    PROP_SOURCE,
)
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from pyiceberg.catalog.rest import RestCatalog

if TYPE_CHECKING:
    from pathlib import Path

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
        # The registered data file is reported as an output URI, and it is a real file on disk.
        data_uris = [u for u in uris if "/iceberg-data/" in u]
        assert len(data_uris) == 1
        assert (tmp_path / data_uris[0].removeprefix(str(tmp_path) + "/")).exists()
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


def test_disabled_never_imports_pyiceberg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
