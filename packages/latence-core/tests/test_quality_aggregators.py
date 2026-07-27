"""C5: the Quality Report is FOLDED out of the record streams, never gathered into 13 lists.

The :class:`~latence_core.quality_report_builder.StageRecordReader` seam is lazy by construction —
``stage_records`` yields ``StageOutput`` one at a time so the Runner can release a Stage's retained
output and stream it back from its checkpoint (#69, ADR-0033). The report builder used to throw that
away: every ``_records_of_kind`` call did ``result.extend(...)`` into a list, and ``build()`` held
fourteen whole-corpus lists simultaneously live until the ``QualityReport(...)`` constructor
returned — with the entity-mention stream (the largest in any corpus) decoded and retained TWICE,
once as ``mentions`` and once as the byte-identical ``disambig_mentions_in``.

C5 replaced the gather with a fold: a :class:`~latence_core.quality_report_builder.QualityAggregator`
per report section, each declaring the ``CapabilityKind``s it subscribes to, and one pass over the
DAG that streams each subscribed Stage's records ONCE and discards them.

These tests hold that line, and each one asserts an observable behaviour rather than a flag:

* **byte-identity** — a broad run's report bytes still equal a fixture captured from the
  pre-C5 gather implementation (ADR-0008 H-F1: the report's bytes are the contract);
* **each Stage is decoded once** — the duplicated entity-mention decode is gone;
* **the fold never materialises the corpus** — peak allocation during ``build()`` is a small
  fraction of one whole-corpus record list;
* **the seam is real** — a caller-supplied aggregator fills a report section without any edit to
  ``build()`` (the L2 ``CapabilityDescriptor`` wiring point);
* **every registered aggregator names a real report field**.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from _c5_broad_stack import FIXTURE, run_broad, timing_free
from latence_core.contracts import (
    Classification,
    EntityMention,
    Provenance,
    ScreeningFinding,
)
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.quality import ParseQuality, QualityReport
from latence_core.quality_report_builder import (
    QUALITY_AGGREGATORS,
    QualityAggregator,
    QualityReportBuilder,
    QualityRunContext,
    StageRecordReader,
)
from latence_core.runner import StageOutput, topological_order
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pydantic import BaseModel



def test_folded_report_is_byte_identical_to_the_pre_c5_gather_fixture(tmp_path: Path) -> None:
    """The folded report's bytes equal the fixture captured from the gather implementation.

    ADR-0008's H-F1 amendment makes the report's BYTES the contract, so the C5 rewrite is only
    admissible if it is invisible in the artifact. The fixture was produced by running this exact
    pipeline against the pre-C5 builder; every field except the wall-clock durations is compared
    exactly.
    """
    report = run_broad(tmp_path)
    expected = FIXTURE.read_text(encoding="utf-8")
    assert timing_free(report) == expected

    # Not vacuous: the fixture has to actually contain the sections the fold reproduces.
    populated = [
        name
        for name in (
            "parse",
            "chunk",
            "screening",
            "entities",
            "relations",
            "redaction",
            "profiling",
            "type_consolidation",
            "disambiguation",
            "graph",
            "context_enrichment",
            "drift",
            "goldset",
        )
        if getattr(report, name) is not None
    ]
    assert len(populated) == 13, f"fixture covers only {populated}"


def test_rendered_report_is_byte_identical_to_the_pre_c5_render(tmp_path: Path) -> None:
    """``render()`` still emits the same Markdown after the section list became derived.

    The rendered "Stage metrics present:" line used to come from a hand-maintained ``section_map``
    tuple; it is now derived from the report model's own section-marked fields. The order and the
    display names must not move.
    """
    report = run_broad(tmp_path)
    rendered = report.render()
    assert (
        "## Stage metrics present: Parse, Chunk, Screening, Entities, Relations, Redaction, "
        "Profiling, TypeConsolidation, Disambiguation, Graph, ContextEnrichment" in rendered
    )


# ---------------------------------------------------------------------------
# The streaming properties.
# ---------------------------------------------------------------------------


def _mention(i: int) -> EntityMention:
    """A fresh, valid EntityMention — allocated on demand so retention is measurable."""
    return EntityMention(
        record_id=f"m-{i}",
        provenance=Provenance(
            source_uri="file:///corpus/doc.txt",
            file_name="doc.txt",
            file_type="txt",
            file_size=1024,
            document_id="doc-1",
            page_start=1,
            page_end=1,
            char_start=i,
            char_end=i + 4,
        ),
        classification=Classification(language="en", sensitivity="internal"),
        document_record_id="doc-1",
        chunk_record_id=f"chunk-{i % 8}",
        index=i % 8,
        label="org",
        text=f"Acme {i}",
        confidence=0.5,
    )


class _CountingLazyReader:
    """A :class:`StageRecordReader` that GENERATES records lazily and counts its decodes.

    Two properties matter and neither is available from a list-backed fake:

    * every record is allocated at ``yield`` time, so anything the consumer retains shows up as
      live memory (the fold must not);
    * ``calls`` records how many times each Stage's stream was opened, so a duplicated decode is
      directly observable;
    * ``yielded`` records how many records have left the reader SO FAR, so a consumer that drains a
      stream eagerly (rather than pulling one record at a time) is directly observable too.
    """

    def __init__(self, per_stage: Mapping[str, int]) -> None:
        self._per_stage = dict(per_stage)
        self.calls: dict[str, int] = {}
        self.yielded = 0

    def stage_records(
        self,
        stage_name: str,
        outputs: Mapping[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> Iterator[StageOutput]:
        self.calls[stage_name] = self.calls.get(stage_name, 0) + 1
        for i in range(self._per_stage.get(stage_name, 0)):
            self.yielded += 1
            yield _mention(i)

    def read_findings(
        self, storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> list[ScreeningFinding]:
        return []


def _mentions_pipeline(store: str) -> Pipeline:
    """A DAG with one entity Stage — the stream the old builder decoded twice."""
    return Pipeline(
        name="c5-stream",
        storage_uri=store,
        stages=[
            Stage(name="source", capability=CapabilityKind.SOURCE, provider="source.local_folder"),
            Stage(
                name="entities",
                capability=CapabilityKind.ENTITY_EXTRACTION,
                provider="entity.gazetteer",
                depends_on=["source"],
            ),
        ],
    )


def _build_with(reader: _CountingLazyReader, store: str, n: int) -> QualityReport:
    pipeline = _mentions_pipeline(store)
    ordered = topological_order(list(pipeline.stages))
    builder = QualityReportBuilder(reader)
    return builder.build(
        pipeline,
        "stream",
        Storage.for_uri(store),
        ordered,
        [],
        {},
        iter(()),
    )


def test_each_stage_stream_is_decoded_exactly_once(tmp_path: Path) -> None:
    """The entity-mention stream is opened ONCE, not twice.

    The pre-C5 ``build()`` made two byte-identical ``_records_of_kinds`` calls over
    ``(ENTITY_EXTRACTION, FUSED_ENTITY_RELATION)`` — one for ``mentions``, one for
    ``disambig_mentions_in`` — so the largest stream in any corpus was decoded and retained twice.
    """
    store = f"file://{tmp_path}/store"
    reader = _CountingLazyReader({"entities": 64})
    report = _build_with(reader, store, 64)

    assert reader.calls["entities"] == 1, (
        f"the entity stream was decoded {reader.calls['entities']}x, expected exactly 1"
    )
    # The section is still computed from that single pass.
    assert report.entities is not None
    assert report.entities.mentions_total == 64


def test_the_fold_never_materialises_the_whole_corpus(tmp_path: Path) -> None:
    """Peak allocation during ``build()`` is a small fraction of ONE whole-corpus list.

    The memory claim is the point of C5, so it is measured rather than asserted structurally:
    ``full`` is what materialising the stream once actually costs, and the fold's peak must be far
    below it. The pre-C5 builder retained the mention stream twice (``mentions`` +
    ``disambig_mentions_in``), so its peak was ``>= 2 * full``.
    """
    store = f"file://{tmp_path}/store"
    n = 4000

    # Calibrate: what does ONE materialised copy of this stream cost?
    calib = _CountingLazyReader({"entities": n})
    pipeline = _mentions_pipeline(store)
    tracemalloc.start()
    materialised = list(
        calib.stage_records(
            "entities", {}, storage=Storage.for_uri(store), pipeline=pipeline, run_id="stream"
        )
    )
    _, full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(materialised) == n
    del materialised

    reader = _CountingLazyReader({"entities": n})
    tracemalloc.start()
    report = _build_with(reader, store, n)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert report.entities is not None
    assert report.entities.mentions_total == n
    assert peak < full * 0.25, (
        f"build() peaked at {peak} bytes; one materialised copy of the stream is {full} bytes — "
        "the fold is retaining the corpus"
    )


def test_final_records_pulls_no_sink_record_until_the_report_asks_for_one(
    tmp_path: Path,
) -> None:
    """``final_records`` hands back a STREAM of the sink records, not a whole-corpus list.

    ``build()`` folds the sink records exactly once (``_fold_final_records`` — completeness +
    distinct-document count) and nothing else in the process touches them, so the sinks never need
    to exist all at once. ``final_records`` used to ``list.extend`` every sink Stage's stream and
    return the list, which is the SAME defeat of the lazy reader seam that C5 removed from the
    per-kind gathers: report-time peak stayed ``O(corpus)`` after every Stage had already been
    streamed and released (#69, ADR-0033).

    Measured at the seam: after the call returns, the reader must have emitted NOTHING; pulling one
    record must cost exactly one record.
    """
    store = f"file://{tmp_path}/store"
    pipeline = _mentions_pipeline(store)
    ordered = topological_order(list(pipeline.stages))
    n = 512
    reader = _CountingLazyReader({"entities": n})
    builder = QualityReportBuilder(reader)

    stream = builder.final_records(
        ordered, {}, storage=Storage.for_uri(store), pipeline=pipeline, run_id="sinks"
    )
    assert reader.yielded == 0, (
        f"final_records drained {reader.yielded} of {n} sink records before the report asked for "
        "one — the sinks are being materialised, so report-time peak is O(corpus)"
    )

    iterator = iter(stream)
    next(iterator)
    assert reader.yielded == 1, (
        f"pulling ONE sink record cost {reader.yielded} records — the stream is not incremental"
    )
    # Not vacuous: the sink Stage really does have a corpus-sized stream behind it, and it is the
    # DAG's sink (nothing depends on ``entities``), so this is the stream the report reads.
    assert sum(1 for _ in iterator) == n - 1


def test_the_report_phase_never_materialises_the_sink_records(tmp_path: Path) -> None:
    """Peak allocation across ``final_records`` + ``build()`` is a fraction of ONE sink copy.

    The companion to :func:`test_the_fold_never_materialises_the_whole_corpus`: that one measures
    the aggregator fold, this one measures the OTHER half of the report phase — the sink stream the
    completeness check and the document count are folded from. Both must be streamed for the report
    phase to be ``O(1)`` in the corpus; a materialised sink list made the fold's win irrelevant
    (~99.6% of report-time peak was the sink list).
    """
    store = f"file://{tmp_path}/store"
    pipeline = _mentions_pipeline(store)
    ordered = topological_order(list(pipeline.stages))
    n = 4000

    # Calibrate: what does ONE materialised copy of the sink stream cost?
    calib = _CountingLazyReader({"entities": n})
    tracemalloc.start()
    materialised = list(
        calib.stage_records(
            "entities", {}, storage=Storage.for_uri(store), pipeline=pipeline, run_id="sinks"
        )
    )
    _, full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(materialised) == n
    del materialised

    reader = _CountingLazyReader({"entities": n})
    builder = QualityReportBuilder(reader)
    storage = Storage.for_uri(store)
    tracemalloc.start()
    tracemalloc.reset_peak()
    final_records = builder.final_records(
        ordered, {}, storage=storage, pipeline=pipeline, run_id="sinks"
    )
    report = builder.build(pipeline, "sinks", storage, ordered, [], {}, final_records)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The sinks really were read (the completeness check saw every one of them).
    assert report.completeness.records_checked == n
    assert peak < full * 0.25, (
        f"the report phase peaked at {peak} bytes; one materialised copy of the sink stream is "
        f"{full} bytes — the sinks are being retained"
    )


# ---------------------------------------------------------------------------
# The seam.
# ---------------------------------------------------------------------------


class _ShoutingParseAggregator:
    """A caller-supplied aggregator that fills the ``parse`` section its own way.

    Stands in for what lane L2's ``CapabilityDescriptor`` will supply per Capability: the builder
    takes the factory list as a dependency, so a new metric never edits ``build()``.
    """

    section_field: ClassVar[str] = "parse"
    kinds: ClassVar[tuple[CapabilityKind, ...]] = (CapabilityKind.ENTITY_EXTRACTION,)

    def __init__(self) -> None:
        self._seen = 0

    @classmethod
    def for_run(cls, context: QualityRunContext) -> _ShoutingParseAggregator | None:
        return cls()

    def observe(self, stage: Stage, record: StageOutput) -> None:
        self._seen += 1

    def finish(self) -> BaseModel | None:
        return ParseQuality(documents_parsed=self._seen)


def test_a_supplied_aggregator_fills_a_section_without_editing_build(tmp_path: Path) -> None:
    """A custom aggregator reaches the report through the registry seam alone."""
    store = f"file://{tmp_path}/store"
    pipeline = _mentions_pipeline(store)
    ordered = topological_order(list(pipeline.stages))
    reader = _CountingLazyReader({"entities": 7})
    builder = QualityReportBuilder(reader, aggregators=(_ShoutingParseAggregator.for_run,))
    report = builder.build(
        pipeline, "seam", Storage.for_uri(store), ordered, [], {}, iter(())
    )
    assert report.parse is not None
    assert report.parse.documents_parsed == 7
    # Only the supplied aggregator ran: no built-in section was filled.
    assert report.entities is None


def test_every_registered_aggregator_names_a_real_report_field() -> None:
    """A typo in ``section_field`` must be impossible to ship silently.

    ``QualityReport`` forbids extra fields, so a bad name would raise at run time on the first
    pipeline that reached it — this makes it a structural failure at import-plus-test time instead.
    """
    fields = set(QualityReport.model_fields)
    for factory in QUALITY_AGGREGATORS:
        name = getattr(factory, "__self__", factory)
        section = getattr(name, "section_field", None)
        assert section in fields, f"{factory} declares unknown report field {section!r}"


def test_the_protocol_is_the_test_surface() -> None:
    """The public protocol is satisfied structurally — no base class, no registration hook."""
    aggregator: QualityAggregator = _ShoutingParseAggregator()
    reader: StageRecordReader = _CountingLazyReader({})
    assert aggregator.finish() is not None
    assert reader.read_findings(
        Storage.for_uri("file:///tmp"), Pipeline(
            name="x",
            storage_uri="file:///tmp",
            stages=[
                Stage(name="s", capability=CapabilityKind.SOURCE, provider="source.local_folder")
            ],
        ), "r", "s"
    ) == []
