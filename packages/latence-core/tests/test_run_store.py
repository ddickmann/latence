"""RunStore — the owned run layout + checkpoint decode interface (Wave 2-E, regression).

``latence_core.run_store`` re-homes two decisions that used to be scattered: WHERE a run's
files live (paths minted by hand at ~12 sites across six packages) and HOW a checkpoint's
typed records are decoded/written (``LocalRunner`` privates reached into via
``# noqa: SLF001``). These tests pin the re-homing's contract:

* **layout parity** — every :class:`RunLayout` path is byte-identical to the string the
  ``LocalRunner`` private it replaced has always minted (the delegation lock: if either side
  drifted, resume/delta/console would read from a place the runner never wrote);
* **read/count** — ``RunStore.read`` yields the runner's own records, typed, in order; a
  missing checkpoint yields nothing (the "skip it" answer every collection pass wants); the
  explicit ``record_type`` override decodes with a caller-chosen contract; ``count`` is the
  exact record count without decoding;
* **write** — ``RunStore.write`` produces a byte-identical file to the checkpoint the local
  Runner wrote for the same records, publishes atomically (no ``.tmp`` stranded), and
  ``wal=True`` logs the same per-batch ``stage_batch`` WAL events;
* **decode** — the ``Record`` sentinel routes a mixed fused-Stage checkpoint through the
  per-line EntityMention/RelationMention discrimination.

Each would fail if the RunStore stopped being the same decision the runner executes.
"""

from __future__ import annotations

from pathlib import Path

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    Provenance,
    Record,
    RelationMention,
)
from latence_core.pipeline import Pipeline
from latence_core.run_store import (
    RunLayout,
    RunStore,
    decode_records,
    run_dir_for,
    runs_root_for,
)
from latence_core.runner import LocalRunner
from latence_core.storage import Storage


def _pipeline(docs_dir: Path, store: str) -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: t
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}'}}
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


def test_layout_matches_the_runners_own_paths(docs_dir: Path, tmp_path: Path) -> None:
    """Every RunLayout path equals the string the LocalRunner private mints (delegation lock)."""
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    runner = LocalRunner()
    layout = RunLayout(p.storage_uri, "r1")

    assert runs_root_for(store) == f"{store}/_latence/runs"
    assert run_dir_for(store, "r1") == f"{store}/_latence/runs/r1"
    assert layout.run_dir == runner.run_dir(p, "r1")
    assert layout.manifest_uri == runner._manifest_uri(p, "r1")  # noqa: SLF001
    assert layout.wal_uri == runner._wal_uri(p, "r1")  # noqa: SLF001
    assert layout.report_uri == runner.report_uri(p, "r1")
    assert layout.report_markdown_uri == f"{layout.run_dir}/quality-report.md"
    assert layout.export_dir == runner._export_dir(p, "r1")  # noqa: SLF001
    assert layout.checkpoint_uri("chunk") == runner._checkpoint_uri(p, "r1", "chunk")  # noqa: SLF001
    assert layout.quarantine_uri("chunk") == runner._quarantine_uri(p, "r1", "chunk")  # noqa: SLF001
    assert layout.findings_uri("chunk") == runner._findings_uri(p, "r1", "chunk")  # noqa: SLF001
    assert layout.type_vocabulary_uri("tc") == runner._type_vocabulary_uri(p, "r1", "tc")  # noqa: SLF001

    # The run-scoped face exposes the identical strings.
    s = Storage.for_uri(store)
    rs = RunStore(s, p, "r1")
    assert rs.layout == layout
    assert (rs.run_dir, rs.manifest_uri, rs.wal_uri, rs.report_uri) == (
        layout.run_dir,
        layout.manifest_uri,
        layout.wal_uri,
        layout.report_uri,
    )
    assert rs.checkpoint_uri("chunk") == layout.checkpoint_uri("chunk")


def test_read_and_count_round_trip_the_runners_checkpoints(
    docs_dir: Path, tmp_path: Path
) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    LocalRunner().run(p, run_id="r1")
    s = Storage.for_uri(store)
    rs = RunStore(s, p, "r1")
    chunk_stage = next(st for st in p.stages if st.name == "chunk")

    records = list(rs.read(chunk_stage))
    assert records and all(isinstance(r, ChunkRecord) for r in records)
    # Byte-identical to re-serializing what the runner's own decode reads back.
    on_disk = s.read_bytes(rs.checkpoint_uri("chunk"))
    assert on_disk == ("\n".join(r.model_dump_json() for r in records) + "\n").encode("utf-8")
    # The exact no-decode count.
    assert rs.count("chunk") == len(records)
    # The explicit record_type override decodes with the caller's contract.
    override = [
        r.record_id
        for r in rs.read(chunk_stage, record_type=ChunkRecord)
        if isinstance(r, ChunkRecord)
    ]
    assert override == [r.record_id for r in records if isinstance(r, ChunkRecord)]
    # A Stage with no finalized checkpoint yields nothing and counts zero.
    ghost = chunk_stage.model_copy(update={"name": "never-ran"})
    assert list(rs.read(ghost)) == []
    assert rs.count("never-ran") == 0


def test_write_is_byte_identical_to_the_runners_checkpoint(
    docs_dir: Path, tmp_path: Path
) -> None:
    store = f"file://{tmp_path}/store"
    p = _pipeline(docs_dir, store)
    LocalRunner().run(p, run_id="r1")
    s = Storage.for_uri(store)
    src = RunStore(s, p, "r1")
    chunk_stage = next(st for st in p.stages if st.name == "chunk")
    records = list(src.read(chunk_stage))
    runner_bytes = s.read_bytes(src.checkpoint_uri("chunk"))

    # Re-write the same records through RunStore.write into a second run's layout.
    dst = RunStore(s, p, "r2")
    written = dst.write("chunk", iter(records), wal=True)
    assert written == len(records)
    assert s.read_bytes(dst.checkpoint_uri("chunk")) == runner_bytes
    # Atomic publish: no temp file stranded next to the checkpoint.
    ckpt_dir = Path(dst.checkpoint_uri("chunk").removeprefix("file://")).parent
    assert not list(ckpt_dir.glob("*.tmp*"))
    # wal=True logged the per-batch progress events to this run's WAL.
    wal_text = s.read_text(dst.wal_uri)
    assert '"event": "stage_batch"' in wal_text
    assert '"stage": "chunk"' in wal_text
    # An empty write still publishes an (empty) checkpoint — the sticky-skip shape.
    assert dst.write("empty", iter([])) == 0
    assert s.read_bytes(dst.checkpoint_uri("empty")) == b""


def test_decode_records_discriminates_the_mixed_carrier() -> None:
    prov = Provenance(
        source_uri="mem://d",
        file_name="d.txt",
        file_type="txt",
        file_size=64,
        document_id="d1",
        char_start=0,
        char_end=15,
        page_start=1,
        page_end=1,
    )
    cls = Classification(language="en")
    entity = EntityMention(
        record_id="e-1",
        document_record_id="d1",
        chunk_record_id="c1",
        index=0,
        label="TOKEN",
        text="head_mention_id",  # the adversarial value: a substring test would misroute this
        confidence=0.9,
        provenance=prov,
        classification=cls,
    )
    relation = RelationMention(
        record_id="r-1",
        document_record_id="d1",
        index=0,
        label="works_for",
        head_mention_id="e-1",
        tail_mention_id="e-2",
        head_text="head_mention_id",
        tail_text="Alice",
        confidence=0.88,
        provenance=prov,
        classification=cls,
    )
    text = entity.model_dump_json() + "\n" + relation.model_dump_json() + "\n"
    decoded = decode_records(text, Record)
    assert [type(r) for r in decoded] == [EntityMention, RelationMention]
    assert decoded == [entity, relation]
