"""The run list: discovered from Storage, with real status and real timestamps (#186 AC 2)."""

from __future__ import annotations

from datetime import datetime

from latence_console.runs import RunIndex, RunStatus
from latence_core.storage import Storage
from reports import healthy_report, write_run


def test_lists_every_run_that_has_a_manifest(populated: str) -> None:
    index = RunIndex(Storage.for_uri(populated), populated)
    runs = index.list_runs()
    assert sorted(s.run_id for s in runs) == [
        "run-0001",
        "run-0002",
        "run-0003",
        "run-0004",
        "run-0005",
    ]


def test_the_list_is_newest_first(populated: str) -> None:
    stamps = [s.started_at for s in RunIndex(Storage.for_uri(populated), populated).list_runs()]
    assert all(stamp is not None for stamp in stamps)
    assert stamps == sorted(stamps, reverse=True)  # type: ignore[type-var]


def test_status_is_derived_from_the_report_not_a_stored_flag(populated: str) -> None:
    index = RunIndex(Storage.for_uri(populated), populated)
    status = {s.run_id: s.status for s in index.list_runs()}
    assert status["run-0001"] is RunStatus.PASSED
    assert status["run-0002"] is RunStatus.ATTENTION  # completeness gap
    assert status["run-0003"] is RunStatus.FAILED  # a Stage raised
    assert status["run-0005"] is RunStatus.UNFINISHED  # manifest but no report


def test_a_stage_error_outranks_a_stored_contracts_complete_true(
    storage: Storage, storage_uri: str
) -> None:
    """The verdict follows the evidence, never the report's own summary boolean.

    A report can carry ``contracts_complete=True`` while a Stage recorded an error — the
    console must still call that run FAILED. Asserting the *outcome* here (not that some
    flag was read) is what keeps this test honest.
    """
    report = healthy_report("liar")
    broken = report.model_copy(
        update={
            "contracts_complete": True,
            "stages": [
                report.stages[0].model_copy(
                    update={"error_count": 1, "error_category": "storage"}
                )
            ],
        }
    )
    write_run(storage, storage_uri, "liar", broken)
    index = RunIndex(storage, storage_uri)
    summary = index.summary("liar")
    assert summary is not None
    assert summary.status is RunStatus.FAILED


def test_timestamps_come_from_the_files_on_storage(populated: str) -> None:
    index = RunIndex(Storage.for_uri(populated), populated)
    summary = index.summary("run-0001")
    assert summary is not None
    assert isinstance(summary.started_at, datetime)
    assert isinstance(summary.finished_at, datetime)
    assert summary.started_at.tzinfo is not None
    assert summary.finished_at >= summary.started_at


def test_unfinished_run_has_no_finish_time(populated: str) -> None:
    index = RunIndex(Storage.for_uri(populated), populated)
    summary = index.summary("run-0005")
    assert summary is not None
    assert summary.started_at is not None
    assert summary.finished_at is None


def test_delta_runs_are_marked_as_updates(populated: str) -> None:
    index = RunIndex(Storage.for_uri(populated), populated)
    by_id = {s.run_id: s for s in index.list_runs()}
    assert by_id["run-0004"].is_delta is True
    assert by_id["run-0001"].is_delta is False


def test_a_directory_without_a_manifest_is_not_a_run(
    storage: Storage, storage_uri: str
) -> None:
    storage.write_text(f"{storage_uri}/_latence/runs/scratch/notes.txt", "hello")
    index = RunIndex(storage, storage_uri)
    assert index.run_ids() == []


def test_empty_storage_lists_nothing(storage: Storage, storage_uri: str) -> None:
    assert RunIndex(storage, storage_uri).list_runs() == []


def test_a_corrupt_report_does_not_break_the_list(storage: Storage, storage_uri: str) -> None:
    write_run(storage, storage_uri, "torn", healthy_report("torn"))
    storage.write_text(f"{storage_uri}/_latence/runs/torn/quality-report.json", "{not json")
    index = RunIndex(storage, storage_uri)
    summary = index.summary("torn")
    assert summary is not None
    assert summary.status is RunStatus.UNFINISHED
    assert index.report("torn") is None
