"""The ``latence-sink-qdrant`` command: argument merging, exit codes, and what it prints.

The CLI's whole job is *composition* — merge a config file with flags (flags win, unset flags never
override), hand the result to :class:`~latence_sink_qdrant.sink.QdrantSink`, and turn the outcome
into an exit code an operator's script can branch on. So that is what these tests assert, through
``main(argv)``, with the sink's client swapped for the shipped faithful fake (the loading behaviour
itself is covered at the sink's own boundary in ``test_qdrant_sink.py``).

Exit codes are part of the contract:
``0`` success · ``1`` ``verify-absent`` found records that should be gone · ``2`` a sink error.
"""

from __future__ import annotations

import builtins
import json
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage
from latence_sink_qdrant import cli
from latence_sink_qdrant.errors import SinkError
from latence_sink_qdrant.sink import QdrantSink
from latence_sink_qdrant.testing import FakeQdrantClient

if TYPE_CHECKING:
    from pathlib import Path

_DIMENSION = 8


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeQdrantClient:
    """Make the CLI build sinks over one shared in-memory store instead of a real client."""
    client = FakeQdrantClient()
    seen: list[dict[str, Any]] = []

    def _factory(config: dict[str, Any]) -> QdrantSink:
        seen.append(config)
        return QdrantSink(config, client=client)

    monkeypatch.setattr(cli, "QdrantSink", _factory)
    client.configs = seen  # type: ignore[attr-defined]  # test-only inspection handle
    return client


def _export(tmp_path: Path, record_ids: list[str]) -> str:
    export = JsonlParquetExport(
        {"embedder": {"provider": "embedding.hashing", "config": {"dimension": _DIMENSION}}}
    )
    records = [
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
            content=f"text for {rid}",
            token_count=3,
        )
        for rid in record_ids
    ]
    export.export(records, Storage(), str(tmp_path))
    return str(tmp_path)


def test_load_prints_a_summary_and_exits_zero(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    export = _export(tmp_path, ["c1", "c2"])

    code = cli.main(["load", export, "--collection", "corpus", "--url", "http://localhost:6333"])

    assert code == 0
    assert "upserted 2" in capsys.readouterr().out
    assert store.points["corpus"].keys()


def test_flags_override_the_config_file(tmp_path: Path, store: FakeQdrantClient) -> None:
    """A cluster's connection block lives in one file; the day-to-day flags win over it."""
    config = tmp_path / "sink.yaml"
    config.write_text(
        "collection: from-file\nurl: http://from-file:6333\nbatch_size: 999\n", encoding="utf-8"
    )
    export = _export(tmp_path / "export", ["c1"])

    code = cli.main(
        ["load", export, "--config", str(config), "--collection", "from-flag", "--batch-size", "7"]
    )

    assert code == 0
    used = store.configs[-1]  # type: ignore[attr-defined]
    assert used["collection"] == "from-flag"
    assert used["batch_size"] == 7
    assert used["url"] == "http://from-file:6333"  # untouched by the (unset) flag


def test_a_json_config_file_works_too(tmp_path: Path, store: FakeQdrantClient) -> None:
    config = tmp_path / "sink.json"
    config.write_text(json.dumps({"collection": "corpus", "batch_size": 4}), encoding="utf-8")
    export = _export(tmp_path / "export", ["c1"])

    assert cli.main(["load", export, "--config", str(config)]) == 0
    assert store.configs[-1]["batch_size"] == 4  # type: ignore[attr-defined]


def test_no_reconcile_flag_reaches_the_sink(tmp_path: Path, store: FakeQdrantClient) -> None:
    export = _export(tmp_path, ["c1"])

    assert cli.main(["load", export, "--collection", "corpus", "--no-reconcile"]) == 0
    assert store.configs[-1]["reconcile"] is False  # type: ignore[attr-defined]


def test_verify_absent_exits_one_while_the_record_is_still_there(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The audit step an operator can branch on: non-zero, and it names the offenders."""
    export = _export(tmp_path, ["c1", "c2"])
    cli.main(["load", export, "--collection", "corpus"])

    code = cli.main(["verify-absent", "c2", "--collection", "corpus"])

    assert code == 1
    assert "STILL PRESENT" in capsys.readouterr().err


def test_verify_absent_exits_zero_after_the_record_is_gone(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["load", _export(tmp_path / "v1", ["c1", "c2"]), "--collection", "corpus"])
    cli.main(["load", _export(tmp_path / "v2", ["c1"]), "--collection", "corpus"])

    code = cli.main(["verify-absent", "c2", "--collection", "corpus"])

    assert code == 0
    assert "verified absent" in capsys.readouterr().out


def test_verify_absent_against_a_typoed_collection_exits_two_and_certifies_nothing(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The README runbook's step 4 must never print a green for a collection it never inspected."""
    cli.main(["load", _export(tmp_path, ["c1"]), "--collection", "corpus"])

    code = cli.main(["verify-absent", "c1", "--collection", "corpuss"])

    captured = capsys.readouterr()
    assert code == 2
    assert "verified absent" not in captured.out
    assert "corpuss" in captured.err
    assert "c1" in {payload["record_id"] for _, payload in store.points["corpus"].values()}


def test_a_sink_error_exits_two_with_the_message(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["load", str(tmp_path / "nowhere"), "--collection", "corpus"])

    assert code == 2
    assert "records.parquet" in capsys.readouterr().err


def test_a_missing_collection_is_a_clean_error_not_a_traceback(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["load", _export(tmp_path, ["c1"])])

    assert code == 2
    assert "'collection'" in capsys.readouterr().err


def test_the_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_a_missing_qdrant_client_extra_exits_two_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The likeliest first-run failure: the console script ships in the BASE install, the driver
    is the optional ``[client]`` extra. It must surface as the documented typed exit 2 with the
    install hint — never as a native ``ImportError`` traceback printing absolute paths (ADR-0034).
    """
    real_import = builtins.__import__

    def _no_qdrant(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            msg = "No module named 'qdrant_client'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_qdrant)

    code = cli.main(
        ["verify-absent", "doc-7#chunk-0", "--collection", "corpus", "--url", "http://localhost:6333"]
    )

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ")
    assert "latence-sink-qdrant[client]" in err
    assert "Traceback" not in err


def test_import_guard_raises_a_typed_sink_error_for_library_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typed contract errors.py claims holds for every caller, not only the CLI."""
    real_import = builtins.__import__

    def _no_qdrant(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            msg = "No module named 'qdrant_client'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_qdrant)

    with pytest.raises(SinkError) as excinfo:
        QdrantSink._import_qdrant()

    assert "latence-sink-qdrant[client]" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_a_missing_config_file_exits_two_not_a_traceback(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(["verify-absent", "c1", "--config", str(tmp_path / "nope.yaml")])

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ")
    assert "nope.yaml" in err


def test_a_malformed_config_file_exits_two_not_a_traceback(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1,\n  b: {\n", encoding="utf-8")

    code = cli.main(["verify-absent", "c1", "--config", str(bad)])

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ")
    assert "bad.yaml" in err


def test_a_malformed_json_config_file_exits_two_not_a_traceback(
    tmp_path: Path, store: FakeQdrantClient, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    code = cli.main(["verify-absent", "c1", "--config", str(bad)])

    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("error: ")
    assert "bad.json" in err
