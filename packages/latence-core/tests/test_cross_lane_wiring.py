"""The seams each lane built, actually consumed by the modules that used to hold copies.

Every deepening lane owned a disjoint set of files, so three consumers were left holding the old
implementation next to the new seam. These tests assert the CONSUMPTION, because a seam nothing
consumes is a second copy of the problem, not a fix:

* the Runner's dispatch enforces the Protocol the ``CapabilityDescriptor`` DECLARES (L2 ← L3), so a
  Capability whose declaration changes cannot keep being run against the old Protocol;
* ``LocalRunner._write_checkpoint`` publishes through ``Storage.atomic_writer`` (L8 ← L3), so the
  checkpoint write inherits the sweep instead of re-implementing it — the one that matters, since
  C6 put the *Provider* inside that write and a flaky store strands a full checkpoint per attempt;
* the Quality registry is held against the sections the descriptors claim (L2 ← L5), so a
  Capability cannot declare a report section that nothing fills.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.local import LocalFileSystem
from latence_core.capability_descriptor import dag_protocol_for, descriptor_for, descriptors
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.errors import ContractError
from latence_core.pipeline import CapabilityKind, Stage
from latence_core.quality_report_builder import (
    QUALITY_AGGREGATORS,
    check_aggregator_coverage,
)
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _chunk(index: int) -> ChunkRecord:
    text = f"chunk {index}"
    return ChunkRecord(
        record_id=f"c{index}",
        provenance=Provenance(
            source_uri="file:///c/a.txt",
            file_name="a.txt",
            file_type="txt",
            file_size=100,
            document_id="sha256:d1",
            char_start=0,
            char_end=len(text),
            page_start=1,
            page_end=1,
        ),
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=index,
        content=text,
        token_count=2,
    )


# -- L2 ← L3: the run-time capability check reads the descriptor -------------------------------


class _NotAnything:
    """A Provider object that satisfies no Capability Protocol."""


def _stage_of(kind: CapabilityKind) -> Stage:
    return Stage(name="s", capability=kind, provider="x.y")


@pytest.mark.parametrize(
    "kind",
    [k for k in CapabilityKind if descriptor_for(k).dag_node],
)
def test_dispatch_rejects_a_provider_against_the_declared_protocol(kind: CapabilityKind) -> None:
    """The refusal names the Protocol the DESCRIPTOR declares, for every DAG Capability.

    This was seventeen hand-written ``isinstance`` branches, each with its own transcribed message
    — a capability→Protocol table that had to be kept in step with the descriptor by memory. The
    message is now derived from ``dag_protocol_for``, so it cannot name something else.
    """
    stage = _stage_of(kind)
    with pytest.raises(ContractError) as excinfo:
        LocalRunner._capability_provider(stage, _NotAnything(), dag_protocol_for(kind))  # noqa: SLF001
    assert str(excinfo.value) == (
        f"Provider 'x.y' does not satisfy the {dag_protocol_for(kind).__name__} capability."
    )


def test_a_branch_that_narrows_to_the_wrong_protocol_is_refused() -> None:
    """Naming a Protocol the Capability does not declare fails loudly, not silently.

    The static narrowing each dispatch branch performs is a second statement of the same fact. It
    is asserted against the declaration rather than trusted, so a branch cannot end up enforcing
    one Protocol while claiming another.
    """
    parse = _stage_of(CapabilityKind.PARSE)
    chunker = dag_protocol_for(CapabilityKind.CHUNK)
    with pytest.raises(ContractError, match="declares Parser"):
        LocalRunner._capability_provider(parse, _NotAnything(), chunker)  # noqa: SLF001


# -- L8 ← L3: the checkpoint write publishes through the atomic_writer seam ---------------------


def _fail_the_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport drop on the publish half: the temp file is complete, the rename dies."""
    def flaky_mv(self: LocalFileSystem, path1: str, path2: str, **kwargs: Any) -> Any:
        msg = "simulated transport drop"
        raise OSError(msg)

    monkeypatch.setattr(LocalFileSystem, "mv", flaky_mv)


def test_a_failed_checkpoint_publish_leaves_no_orphan_and_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename dies: nothing is published AND nothing full-size is stranded beside it.

    Before this wiring the Runner minted its own temp name and swept only the *Provider-failure*
    path; a failure in the publish itself left the whole serialized checkpoint behind under a
    random temp name that no retry ever reuses. The behaviour asserted is what is on disk.
    """
    storage = Storage.for_uri(f"file://{tmp_path}")
    uri = f"file://{tmp_path}/run/stage.jsonl"
    _fail_the_rename(monkeypatch)

    with pytest.raises(OSError, match="simulated transport drop"):
        LocalRunner()._write_checkpoint(storage, uri, [_chunk(i) for i in range(3)])  # noqa: SLF001

    assert not (tmp_path / "run" / "stage.jsonl").exists()
    assert [p.name for p in tmp_path.rglob("*.tmp")] == []


def test_a_failing_record_stream_leaves_no_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C6's real case: the Provider runs INSIDE the write and raises part-way through.

    The half-written temp file must go, the original error must reach the caller untouched, and
    the previous checkpoint (if any) must still be the one on disk.
    """
    storage = Storage.for_uri(f"file://{tmp_path}")
    uri = f"file://{tmp_path}/run/stage.jsonl"
    storage.write_atomic(uri, b'{"previous": true}\n')

    def exploding() -> Iterator[ChunkRecord]:
        yield _chunk(0)
        msg = "provider died mid-stream"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="provider died mid-stream"):
        LocalRunner()._write_checkpoint(storage, uri, exploding())  # noqa: SLF001

    assert storage.read_text(uri) == '{"previous": true}\n'
    assert [p.name for p in tmp_path.rglob("*.tmp")] == []


def test_the_checkpoint_bytes_are_unchanged_by_the_seam(tmp_path: Path) -> None:
    """Newline-delimited ``model_dump_json``, trailing newline iff non-empty — the on-disk format.

    The delta engine's reader and the Airflow adapter's typed decode both depend on it, so the
    substitution is only admissible if the artifact is byte-identical.
    """
    storage = Storage.for_uri(f"file://{tmp_path}")
    uri = f"file://{tmp_path}/run/stage.jsonl"
    records = [_chunk(i) for i in range(3)]

    written = LocalRunner()._write_checkpoint(storage, uri, records)  # noqa: SLF001

    assert written == 3
    assert storage.read_text(uri) == "".join(r.model_dump_json() + "\n" for r in records)

    empty_uri = f"file://{tmp_path}/run/empty.jsonl"
    assert LocalRunner()._write_checkpoint(storage, empty_uri, []) == 0  # noqa: SLF001
    assert storage.read_text(empty_uri) == ""


# -- L2 ← L5: the Quality registry covers every section a Capability claims ---------------------


def test_every_capability_claimed_quality_section_has_an_aggregator() -> None:
    """The shipped registry covers the descriptors — the gate that runs at import."""
    check_aggregator_coverage()


def test_a_registry_missing_a_claimed_section_is_refused() -> None:
    """Dropping the aggregator for a claimed section fails loudly rather than emptying it.

    Without this the section is simply ``None`` in every report forever: a Quality artifact that
    under-reports is indistinguishable from a run with nothing to report.
    """
    claimed = {d.quality_section for d in descriptors() if d.quality_section is not None}
    survivors = tuple(
        factory
        for factory in QUALITY_AGGREGATORS
        if getattr(getattr(factory, "__self__", factory), "section_field", None) != "graph"
    )
    assert "graph" in claimed
    assert len(survivors) == len(QUALITY_AGGREGATORS) - 1

    with pytest.raises(ContractError, match="graph"):
        check_aggregator_coverage(survivors)


def test_cross_cutting_sections_are_not_required_to_be_claimed() -> None:
    """``drift`` and ``goldset`` belong to no Capability; the gate is one-directional by design."""
    claimed = {d.quality_section for d in descriptors() if d.quality_section is not None}
    filled = {
        getattr(getattr(factory, "__self__", factory), "section_field", None)
        for factory in QUALITY_AGGREGATORS
    }
    assert {"drift", "goldset"} <= filled - claimed
    check_aggregator_coverage()
