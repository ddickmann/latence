"""``Storage.atomic_writer`` / ``atomic_writers``: the streamed write-atomic primitive (ADR-0010).

The two-call ``temp_uri`` / ``finalize_atomic`` protocol these replace pushed a cleanup obligation
onto every caller that its signature never stated — sweep the temp path when the write dies between
the two calls — and one of its three callers actually did it. Because each temp name carries a
fresh random token, the cost of the other two was not "one recoverable file" but *another* full
copy per failed attempt: silent, billable, unreferenced objects still holding the run's bytes.

Every test here asserts the BEHAVIOUR that obligation exists to buy (no orphan survives a failure,
the original error is never masked, the destination is published or nothing is), never that a flag
was set. The failure paths are driven by making the REAL filesystem operation fail — the fault is
injected at the ``fsspec`` rename/delete, standing in for a transport drop, an ENOSPC, or a store
that refuses a delete — so the tests measure what survives on disk, not which method was called.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fsspec.implementations.local import LocalFileSystem
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_core.stages.export import JsonlParquetExport
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterator


def _leftover_temp_files(directory: Path) -> list[str]:
    """Every ``.tmp`` sibling left under ``directory`` — the orphans a failed write must not leave."""
    return sorted(str(p.relative_to(directory)) for p in directory.rglob("*.tmp") if p.is_file())


def _fail_the_rename(monkeypatch: pytest.MonkeyPatch, *, matching: str) -> None:
    """Make the publish half of the atomic write fail for destinations containing ``matching``.

    A transport drop / ENOSPC / permission-denied rename: the temp file already holds the complete
    payload and the operation that would give it its real name is what dies.

    Patched at the *filesystem* (``fsspec``) level rather than on ``Storage.finalize_atomic``, so
    the fault is injected somewhere both the buffered and the streamed write reach no matter which
    Storage method they route through — the test then measures behaviour, not call structure.
    """
    real_mv = LocalFileSystem.mv

    def flaky_mv(self: LocalFileSystem, path1: str, path2: str, **kwargs: Any) -> Any:
        if matching in str(path2):
            msg = "simulated transport drop"
            raise OSError(msg)
        return real_mv(self, path1, path2, **kwargs)

    monkeypatch.setattr(LocalFileSystem, "mv", flaky_mv)


def _fail_the_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the temp-file delete fail — a store that refuses the cleanup."""
    real_remove = Storage.remove

    def flaky_remove(self: Storage, uri: str) -> None:
        if uri.endswith(".tmp"):
            msg = "store refused the delete"
            raise PermissionError(msg)
        real_remove(self, uri)

    monkeypatch.setattr(Storage, "remove", flaky_remove)


# ---------------------------------------------------------------------------
# The happy path: publish-or-nothing, with no leftovers either way.
# ---------------------------------------------------------------------------


def test_atomic_writer_publishes_the_stream_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """A clean block publishes the destination and leaves no ``.tmp`` sibling behind."""
    storage = Storage()
    uri = str(tmp_path / "out" / "records.jsonl")
    with storage.atomic_writer(uri) as fh:
        fh.write(b"one\n")
        fh.write(b"two\n")
        # Nothing is visible at the destination until the block exits.
        assert not storage.exists(uri)
    assert storage.read_bytes(uri) == b"one\ntwo\n"
    assert _leftover_temp_files(tmp_path) == []


def test_atomic_writer_creates_the_destination_parent(tmp_path: Path) -> None:
    """The parent directory is created for the caller — a streaming ``open`` does not create one.

    Both migrated callers used to do this by hand before minting the temp path; the obligation
    belongs with the primitive that needs it.
    """
    uri = str(tmp_path / "deep" / "nested" / "run" / "records.jsonl")
    with Storage().atomic_writer(uri) as fh:
        fh.write(b"x")
    assert Storage().read_bytes(uri) == b"x"


def test_atomic_writers_publishes_the_whole_set_only_after_every_stream_closes(
    tmp_path: Path,
) -> None:
    """A multi-file deliverable appears as a set: nothing is published mid-block."""
    storage = Storage()
    a = str(tmp_path / "export" / "records.jsonl")
    b = str(tmp_path / "export" / "records.parquet")
    with storage.atomic_writers([a, b]) as (jsonl, parquet):
        jsonl.write(b"row\n")
        parquet.write(b"PAR1")
        assert not storage.exists(a)
        assert not storage.exists(b)
    assert storage.read_bytes(a) == b"row\n"
    assert storage.read_bytes(b) == b"PAR1"
    assert _leftover_temp_files(tmp_path) == []


# ---------------------------------------------------------------------------
# The obligation the interface now owns: nothing survives a failure.
# ---------------------------------------------------------------------------


def test_a_body_that_raises_leaves_no_partial_write(tmp_path: Path) -> None:
    """An exception between opening the stream and publishing it sweeps the temp file.

    The exact leak the two-call protocol shipped: the handle closes, the destination is never
    written, and the temp file stays on the store holding whatever was streamed into it.
    """
    storage = Storage()
    uri = str(tmp_path / "run" / "checkpoint.jsonl")
    boom = RuntimeError("stage failed half way through the corpus")
    with pytest.raises(RuntimeError) as caught, storage.atomic_writer(uri) as fh:
        fh.write(b"a lot of records\n")
        raise boom
    assert caught.value is boom  # the original failure, not a cleanup error
    assert not storage.exists(uri)
    assert _leftover_temp_files(tmp_path) == []


def test_repeated_failures_do_not_accumulate_copies(tmp_path: Path) -> None:
    """Retrying against a flaky store must not strand ANOTHER copy each time.

    The unbounded-growth half of the defect: every temp name carries a fresh random token, so an
    un-swept failure is not one recoverable file but one per attempt.
    """
    storage = Storage()
    uri = str(tmp_path / "run" / "checkpoint.jsonl")
    for _ in range(5):
        with pytest.raises(RuntimeError), storage.atomic_writer(uri) as fh:
            fh.write(b"the whole corpus\n")
            msg = "flaky store"
            raise RuntimeError(msg)
    assert _leftover_temp_files(tmp_path) == []


def test_a_failed_publish_leaves_no_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in the publish itself (not the body) is swept too, and is not masked."""
    _fail_the_rename(monkeypatch, matching="checkpoint.jsonl")
    storage = Storage()
    uri = str(tmp_path / "run" / "checkpoint.jsonl")
    with pytest.raises(OSError, match="simulated transport drop"), storage.atomic_writer(uri) as fh:
        fh.write(b"records\n")
    assert not storage.exists(uri)
    assert _leftover_temp_files(tmp_path) == []


def test_a_failed_sweep_is_reported_on_the_original_error_never_masking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the store refuses the delete, the operator is TOLD which path still holds the bytes.

    Best-effort cleanup that silently gives up is exactly the silent error ADR-0034 forbids — and
    replacing the real failure with the cleanup's failure would be worse. The original exception
    propagates unchanged, carrying a PEP-678 note that names the surviving path.
    """
    _fail_the_rename(monkeypatch, matching="checkpoint.jsonl")
    _fail_the_sweep(monkeypatch)
    storage = Storage()
    uri = str(tmp_path / "run" / "checkpoint.jsonl")
    with (
        pytest.raises(OSError, match="simulated transport drop") as caught,
        storage.atomic_writer(uri) as fh,
    ):
        fh.write(b"records\n")

    leftovers = _leftover_temp_files(tmp_path)
    assert len(leftovers) == 1, leftovers
    notes = " ".join(getattr(caught.value, "__notes__", ()))
    assert "could not be removed" in notes
    assert Path(leftovers[0]).name in notes
    assert "store refused the delete" in notes


def test_a_successful_sweep_is_reported_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disposal is never silent: a swept temp file is recorded on the propagating error."""
    _fail_the_rename(monkeypatch, matching="checkpoint.jsonl")
    with (
        pytest.raises(OSError, match="simulated transport drop") as caught,
        Storage().atomic_writer(str(tmp_path / "run" / "checkpoint.jsonl")) as fh,
    ):
        fh.write(b"records\n")
    notes = " ".join(getattr(caught.value, "__notes__", ()))
    assert "was removed" in notes
    assert ".tmp" in notes


def test_a_partially_published_set_names_what_landed_and_never_claims_nothing_did(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish loop is N renames: when it dies mid-way, the notes must name what LANDED.

    ``atomic_writers`` documents this as its honest limit — "a failure *during* the publish loop
    leaves the already-renamed siblings in place and sweeps the rest; the exception's notes name
    what landed". The sweep is a cleanup helper and knows only what it removed; asserting
    publication state from there produced the exact inversion of the truth: a published
    ``a.jsonl`` sitting on the store while every note said nothing was published.

    That is the Export's live shape (``records.jsonl`` + ``records.parquet`` ride one block), so
    the operator reading "nothing was published" would skip cleaning up a published,
    corpus-incomplete deliverable that a downstream reader will happily consume.
    """
    _fail_the_rename(monkeypatch, matching="b.parquet")
    storage = Storage()
    out = tmp_path / "export"
    a, b, c = (str(out / name) for name in ("a.jsonl", "b.parquet", "c.bin"))
    with (
        pytest.raises(OSError, match="simulated transport drop") as caught,
        storage.atomic_writers([a, b, c]) as (first, second, third),
    ):
        first.write(b"landed\n")
        second.write(b"never landed")
        third.write(b"never landed")

    # The first rename really did land: this is the state the notes have to describe.
    assert storage.read_bytes(a) == b"landed\n"
    assert not storage.exists(b)
    assert not storage.exists(c)
    assert _leftover_temp_files(tmp_path) == []

    notes = getattr(caught.value, "__notes__", ())
    joined = " ".join(notes)
    assert "nothing was published" not in joined, joined
    assert a in joined, joined  # the published sibling is named
    # Named once, as a publication-state note — not repeated per swept file.
    landed_notes = [note for note in notes if a in note]
    assert len(landed_notes) == 1, notes
    # And the sweep still reports both disposals.
    assert sum("was removed" in note for note in notes) == 2, notes


def test_a_failed_first_rename_still_reports_that_nothing_was_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the loop dies on its FIRST target, nothing landed — and that is still said, once."""
    _fail_the_rename(monkeypatch, matching="a.jsonl")
    storage = Storage()
    out = tmp_path / "export"
    a, b = (str(out / name) for name in ("a.jsonl", "b.parquet"))
    with (
        pytest.raises(OSError, match="simulated transport drop") as caught,
        storage.atomic_writers([a, b]) as (first, second),
    ):
        first.write(b"x")
        second.write(b"y")

    assert not storage.exists(a)
    assert not storage.exists(b)
    notes = getattr(caught.value, "__notes__", ())
    assert sum("nothing was published" in note for note in notes) == 1, notes


def test_atomic_writers_sweeps_every_stream_when_one_fails(tmp_path: Path) -> None:
    """A four-file deliverable that dies mid-write leaves four fewer orphans, not four."""
    storage = Storage()
    out = tmp_path / "export"
    uris = [str(out / name) for name in ("a.jsonl", "b.parquet", "c.parquet", "d.parquet")]
    with pytest.raises(RuntimeError), storage.atomic_writers(uris) as streams:
        for stream in streams:
            stream.write(b"partial")
        msg = "embedder blew up on batch 2"
        raise RuntimeError(msg)
    assert _leftover_temp_files(tmp_path) == []
    assert [uri for uri in uris if storage.exists(uri)] == []


# ---------------------------------------------------------------------------
# The buffered face rides the same primitive.
# ---------------------------------------------------------------------------


def test_write_atomic_leaves_no_temp_file_when_the_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The BUFFERED write leaked the same way: ``write_bytes`` then a failing ``mv``.

    ``write_atomic`` is the Runner's WAL/manifest/Quality-Report primitive, so every failed run
    against a flaky store used to leave a full copy of whatever it was writing next to it.
    """
    _fail_the_rename(monkeypatch, matching="manifest.json")
    uri = str(tmp_path / "run" / "manifest.json")
    with pytest.raises(OSError, match="simulated transport drop"):
        Storage().write_atomic(uri, b'{"run": 1}')
    assert _leftover_temp_files(tmp_path) == []


def test_write_atomic_is_byte_identical_to_write_bytes(tmp_path: Path) -> None:
    """Re-basing the buffered write on the streaming primitive changed no output byte."""
    storage = Storage()
    payload = b'{"a": 1}\n\xff\x00binary\n'
    atomic = str(tmp_path / "a" / "x.json")
    plain = str(tmp_path / "b" / "x.json")
    storage.write_atomic(atomic, payload)
    storage.write_bytes(plain, payload)
    assert storage.read_bytes(atomic) == storage.read_bytes(plain) == payload


# ---------------------------------------------------------------------------
# The content-addressed destination (the Iceberg data file's late naming).
# ---------------------------------------------------------------------------


def test_destination_names_the_publish_target_once_the_bytes_exist(tmp_path: Path) -> None:
    """An artifact whose name derives from its own contents publishes under the resolved name."""
    storage = Storage()
    seed = str(tmp_path / "data" / "records.parquet")
    written: list[bytes] = []

    def _named_after_content() -> str:
        return str(tmp_path / "data" / f"records-{len(b''.join(written))}.parquet")

    with storage.atomic_writer(seed, destination=_named_after_content) as fh:
        written.append(b"12345")
        fh.write(b"12345")
    assert not storage.exists(seed)
    assert storage.read_bytes(str(tmp_path / "data" / "records-5.parquet")) == b"12345"
    assert _leftover_temp_files(tmp_path) == []


def test_a_none_destination_publishes_nothing_and_sweeps(tmp_path: Path) -> None:
    """Returning ``None`` is the "identical file is already there, leave it alone" case."""
    storage = Storage()
    seed = str(tmp_path / "data" / "records.parquet")
    with storage.atomic_writer(seed, destination=lambda: None) as fh:
        fh.write(b"already published elsewhere")
    assert not storage.exists(seed)
    assert _leftover_temp_files(tmp_path) == []


def test_a_destination_that_raises_still_sweeps(tmp_path: Path) -> None:
    """Resolving the name is part of the publication, so its failure sweeps like any other."""
    storage = Storage()

    def _boom() -> str:
        msg = "the existence probe timed out"
        raise OSError(msg)

    target = str(tmp_path / "data" / "records.parquet")
    with (
        pytest.raises(OSError, match="existence probe"),
        storage.atomic_writer(target, destination=_boom) as fh,
    ):
        fh.write(b"corpus")
    assert _leftover_temp_files(tmp_path) == []


# ---------------------------------------------------------------------------
# The migrated caller: the Export's deliverable.
# ---------------------------------------------------------------------------


def _chunk(text: str, record_id: str) -> ChunkRecord:
    provenance = Provenance(
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
        provenance=provenance,
        classification=Classification(language="en", sensitivity="public"),
        document_record_id="d1",
        index=0,
        content=text,
        token_count=max(1, len(text.split())),
    )


def _records_that_die_after(count: int) -> Iterator[ChunkRecord]:
    """A record stream that fails part-way — a Source/embedder blowing up mid-corpus."""
    for index in range(count):
        yield _chunk(f"chunk {index}", f"c{index}")
    msg = "the record stream failed half way through the corpus"
    raise RuntimeError(msg)


def _export_config(**extra: Any) -> dict[str, Any]:
    return dict(extra)


def test_a_failed_export_leaves_no_orphaned_temp_files(tmp_path: Path) -> None:
    """An Export that dies mid-corpus leaves NO ``.tmp`` files behind — the caller-level proof.

    The Export opens up to four streams inside one block and published them at the end; an
    exception in between closed the handles and abandoned every temp path. On an object store
    those are silent, billable, unreferenced objects — and a retried run drops another set, since
    every temp name carries a fresh random token.
    """
    out = tmp_path / "run" / "export"
    export = JsonlParquetExport(_export_config())
    with pytest.raises(RuntimeError, match="half way through the corpus"):
        export.export(_records_that_die_after(3), Storage(), str(out))

    assert _leftover_temp_files(tmp_path) == []
    # And nothing half-published either.
    assert not (out / "records.jsonl").exists()
    assert not (out / "records.parquet").exists()


def test_a_repeatedly_failing_export_does_not_accumulate_temp_files(tmp_path: Path) -> None:
    """Three failed attempts leave three times nothing, not three abandoned deliverables."""
    out = tmp_path / "run" / "export"
    export = JsonlParquetExport(_export_config())
    for _ in range(3):
        with pytest.raises(RuntimeError, match="half way through the corpus"):
            export.export(_records_that_die_after(2), Storage(), str(out))
    assert _leftover_temp_files(tmp_path) == []


def test_a_successful_export_is_unchanged_by_the_migration(tmp_path: Path) -> None:
    """The deliverable itself is untouched: same files, same bytes, no leftovers."""
    out = tmp_path / "run" / "export"
    storage = Storage()
    records = [_chunk("alpha beta", "c0"), _chunk("gamma", "c1")]
    written = JsonlParquetExport(_export_config()).export(records, storage, str(out))
    assert [uri.rsplit("/", 1)[-1] for uri in written] == ["records.jsonl", "records.parquet"]
    lines = storage.read_text(str(out / "records.jsonl")).splitlines()
    assert len(lines) == 2
    assert _leftover_temp_files(tmp_path) == []
