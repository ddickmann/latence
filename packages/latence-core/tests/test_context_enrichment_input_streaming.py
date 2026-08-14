"""Context Enrichment's INPUT streaming — the two-carrier case (ADR-0033, dogfood-3).

``ContextHeaderEnricher`` was already bounded on its own side: it folds the mention stream on
arrival and projects ``GraphNode``/``GraphEdge`` to compact views, so no input record is retained
(``test_context_enrichment_memory_shape``). That made the Stage's own retention small — and left
the RUNNER as the binding constraint. ``CapabilityKind.CONTEXT_ENRICHMENT`` was not in
a streamed-input Capability, so the executor gathered its parents and handed the dispatch ONE
materialised list of every parent record, which the branch then partitioned by carrier:

    enrich_chunks = [r for r in gathered if isinstance(r, ChunkRecord)]
    enrich_graph  = [r for r in gathered if isinstance(r, GraphRecord)]

At UDA scale (220 129 chunks / 516 512 canonical entities / ~90 k edges, a fixture-measured
~5.1 kB per ``GraphNode``) that is GiBs of pydantic records resident for the Stage's whole
execution, inside a container with a hard 29 GB cgroup cap that OOM-killed this pipeline twice.
Since the Provider consumes each of its three inputs exactly once, the Runner can hand it one
INDEPENDENT lazy stream per carrier instead (the Capability's declared ``PER_CARRIER`` shape).

These tests pin that bound at the Runner level; each fails if the eager gather comes back:

* the Runner materialises NOTHING for the Stage, at any corpus size
  (``test_context_enrichment_input_is_never_materialised``);
* the chunk stream is really lazy — pulling the *k*-th chunk has read exactly *k* records back
  from the parent checkpoint (``test_chunk_stream_is_pulled_one_record_at_a_time``);
* the graph carrier — the expensive one — is never all resident: sampled mid-stream, only the
  record in flight is alive (``test_graph_records_are_never_all_resident``);
* streaming cost no output: the enriched checkpoint is BYTE-IDENTICAL to the same run with both
  carriers materialised, and ``records_in`` still reports the whole input
  (``test_streamed_run_is_byte_identical_to_a_materialised_run``);
* a missing ``depends_on`` edge still raises the exact typed ``ContractError`` through the
  streamed presence check, for either carrier
  (``test_mis_wired_context_enrichment_still_errors_streamed``).
"""

from __future__ import annotations

import gc
from importlib import metadata
from typing import TYPE_CHECKING, Any, TypeVar

import pytest
from latence_core import ProviderRegistry
from latence_core.contracts import GraphRecord
from latence_core.errors import ContractError
from latence_core.pipeline import Pipeline
from latence_core.quality import QualityReport
from latence_core.runner import LocalRunner, RunRecords, StageOutput
from latence_core.stage_inputs import StageInputs
from latence_core.stages.context_enrichment import ContextHeaderEnricher
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from importlib import metadata as _metadata
    from pathlib import Path

    from latence_core.contracts import ChunkRecord, EntityMention, Record
    from latence_core.pipeline import Stage

#: How many GraphRecords may be alive at once while the Stage reads the KG. A streamed carrier
#: holds the record in flight plus the handful the reader/filter frames are passing along; the
#: gathered form held every one of them for the Stage's whole execution.
_RESIDENT_GRAPH_SLACK = 5

#: Where in the graph stream residency is sampled. It must be MID-stream, not after the drain: a
#: fully-decoded list handed over as ``iter(list)`` is released when the iterator is exhausted, so
#: an end-of-stream sample would read as bounded even though the peak was O(corpus).
_RESIDENCY_SAMPLE_AT = 16


def _pipeline(store: str, docs_dir: Path, *, enricher: str = "context.kg_header") -> Pipeline:
    """The W13 spine: source→parse→chunk→screen→entities→relations→disambiguate→graph→enrich."""
    return Pipeline.from_yaml_text(
        f"""
name: ctx-input-stream
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
    config: {{max_tokens: 64, overlap_tokens: 0, min_tokens: 2}}
  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [content_screen]
    config:
      labels:
        ORG: {{terms: [Acme Corporation, Acme, International Business Machines, IBM, Globex]}}
        PERSON: {{terms: [Alice, Bob, Carla]}}
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [content_screen, entities]
    config:
      window_chars: 200
      relations:
        works_for: {{head: PERSON, tail: ORG, triggers: [works at, joined]}}
        partner_of: {{head: ORG, tail: ORG, triggers: [partnered with]}}
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config: {{corpus_id: ctx-input-stream}}
  - name: enrich
    capability: context_enrichment
    provider: {enricher}
    depends_on: [content_screen, graph]
    config: {{max_neighbors_per_entity: 4, max_triples: 6}}
"""
    )


def _write_docs(tmp_path: Path, n: int) -> Path:
    """A corpus whose chunk AND graph record counts both grow with ``n``."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    for i in range(n):
        (docs / f"doc{i:03d}.txt").write_text(
            "\n\n".join(
                f"# Section {s} of document {i}\n\n"
                f"Alice works at Acme. Bob joined International Business Machines. "
                f"Carla works at Globex{i}. Acme partnered with Globex{i}. "
                f"IBM partnered with Acme Corporation in section {s}."
                for s in range(3)
            ),
            encoding="utf-8",
        )
    return docs


def _checkpoint_lines(storage: Storage, base: str, stage: str) -> int:
    text = storage.read_text(f"{base}/checkpoints/{stage}.jsonl")
    return sum(1 for line in text.splitlines() if line.strip())


def _registry_with(name: str, cls: type) -> ProviderRegistry:
    """A ProviderRegistry with one extra, test-injected entry point (the seam-test idiom)."""
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access
    ep: _metadata.EntryPoint = metadata.EntryPoint(
        name=name, value="test:injected", group="latence.providers"
    )
    object.__setattr__(ep, "load", lambda: cls)
    discovered[name] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


class _InputSpyRunner(LocalRunner):
    """Records, per Stage, how many input records the Runner MATERIALISED before dispatch.

    ``None`` means NOTHING was materialised — a lazy stream (or, for a per-carrier Stage, streams
    the dispatch opens itself) is all the Stage ever gets; an ``int`` is the length of the list the
    Runner built, the transient these tests bound. Read off
    :attr:`~latence_core.stage_inputs.StageInputs.materialised`, so the spy asks the input whether
    it was materialised rather than inferring it from a parameter's runtime type — and it never
    touches the Stage's records, which for a streamed Capability are one-shot.
    """

    def __init__(self) -> None:
        super().__init__()
        self.materialised_input: dict[str, int | None] = {}

    def _dispatch_stage(
        self,
        stage: Stage,
        provider: object,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        inputs: StageInputs,
    ) -> Iterator[StageOutput]:
        self.materialised_input[stage.name] = inputs.materialised
        return super()._dispatch_stage(stage, provider, storage, pipeline, run_id, inputs)


_T = TypeVar("_T", bound=StageOutput)


class _MaterialisingInputs(StageInputs):
    """A ``StageInputs`` whose per-carrier streams are fully read before the Provider sees them.

    The pre-fix memory shape restored at exactly the one seam that changed: ``stream_of`` — the
    per-carrier read Context Enrichment takes — hands over a fully-materialised carrier LIST
    (wrapped in an iterator so the signature is unchanged), which is what
    ``[r for r in gathered if isinstance(...)]`` produced. Everything else delegates.
    """

    def stream_of(self, expected: type[_T]) -> Iterator[_T]:
        return iter(list(super().stream_of(expected)))


class _MaterialisingRunner(LocalRunner):
    """The parity control: the same run with every per-carrier input collected up front.

    This is the pre-fix shape restored at the one seam that changed — the Provider receives
    fully-materialised carrier LISTS (exactly what ``[r for r in gathered if isinstance(...)]``
    produced) instead of lazy streams — so a byte-difference between this run and the streamed one
    is a difference streaming caused, not a difference in the corpus.
    """

    def stage_inputs(
        self,
        stage: Stage,
        outputs: dict[str, list[StageOutput]],
        *,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
    ) -> StageInputs:
        return _MaterialisingInputs(
            stage,
            pipeline,
            RunRecords(self, outputs, storage, pipeline, run_id),
            overlay=self._reuse_overlay,
        )


class _ProbeEnricher:
    """A ContextEnricher that records the memory shape of the streams the Runner hands it.

    Mirrors :class:`~latence_core.stages.context_enrichment.ContextHeaderEnricher`'s consumption
    order — the whole graph, then the whole mention stream, then the chunks threaded 1:1 — so what
    it measures is what the real Provider would experience.
    """

    graph_seen: list[int] = []  # noqa: RUF012 - test-only shared probe
    live_graph_mid_stream: list[int] = []  # noqa: RUF012 - test-only shared probe
    chunk_pulls: list[int] = []  # noqa: RUF012 - test-only shared probe
    chunk_reads: list[int] = [0]  # noqa: RUF012 - test-only shared counter

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def enrich(
        self,
        chunks: Iterable[ChunkRecord],
        mentions: Iterable[EntityMention],
        graph_records: Iterable[GraphRecord],
    ) -> Iterator[ChunkRecord]:
        seen = 0
        for _record in graph_records:
            seen += 1
            if seen == _RESIDENCY_SAMPLE_AT:
                # Mid-stream: a streamed carrier has decoded ``seen`` records and dropped all but
                # the one in flight; a materialised one has decoded (and is holding) every record
                # in the run before the Provider was called at all.
                gc.collect()
                _ProbeEnricher.live_graph_mid_stream.append(
                    sum(1 for obj in gc.get_objects() if type(obj) is GraphRecord)
                )
        _ProbeEnricher.graph_seen.append(seen)
        for _mention in mentions:
            pass
        for chunk in chunks:
            _ProbeEnricher.chunk_pulls.append(_ProbeEnricher.chunk_reads[0])
            yield chunk

    @staticmethod
    def _reset() -> None:
        _ProbeEnricher.graph_seen = []
        _ProbeEnricher.live_graph_mid_stream = []
        _ProbeEnricher.chunk_pulls = []
        _ProbeEnricher.chunk_reads = [0]


def _count_reads_of(
    monkeypatch: pytest.MonkeyPatch, suffix: str, counter: list[int]
) -> None:
    """Count every record streamed back from the checkpoint whose URI ends with ``suffix``."""
    original = LocalRunner._stream_checkpoint

    def _counting_stream(
        self: LocalRunner,
        storage: Storage,
        uri: str,
        record_type: type[StageOutput] | type[Record],
    ) -> Iterator[StageOutput]:
        for record in original(self, storage, uri, record_type):
            if uri.endswith(suffix):
                counter[0] += 1
            yield record

    monkeypatch.setattr(LocalRunner, "_stream_checkpoint", _counting_stream)


def test_context_enrichment_input_is_never_materialised(tmp_path: Path) -> None:
    """The Runner builds no input list for the Stage — at 3 documents or at 12.

    Pre-fix this recorded ``len(chunks) + len(graph records)`` and grew linearly with the corpus:
    the gathered list WAS the Stage's input. The bound requires that no such list exists, so the
    Stage materialises NOTHING and reads each carrier as its own stream.
    """
    counts: dict[str, int | None] = {}
    for tag, docs in (("small", 3), ("big", 12)):
        doc_dir = _write_docs(tmp_path / tag, docs)
        runner = _InputSpyRunner()
        runner.run(_pipeline(f"file://{tmp_path}/{tag}/store", doc_dir), run_id=tag)
        counts[tag] = runner.materialised_input["enrich"]
        storage = Storage.for_uri(f"file://{tmp_path}/{tag}/store")
        base = f"file://{tmp_path}/{tag}/store/_latence/runs/{tag}"
        # The test has teeth only if the input the Runner USED to hold really did grow.
        counts[f"{tag}-input"] = _checkpoint_lines(storage, base, "content_screen") + (
            _checkpoint_lines(storage, base, "graph")
        )

    small_input, big_input = counts["small-input"], counts["big-input"]
    assert small_input is not None and big_input is not None
    assert big_input > small_input > 0, "the corpus did not grow; the assertion below is vacuous"
    assert counts["small"] is None and counts["big"] is None, (
        f"the context_enrichment Stage received a MATERIALISED input: {counts['small']} records "
        f"at 3 documents, {counts['big']} at 12 (the run's own input grew {small_input} -> "
        f"{big_input}). The Runner is holding the corpus-level gather ADR-0033 bounds — the very "
        f"transient that OOM-killed the UDA run inside its 29 GB cgroup."
    )


def test_chunk_stream_is_pulled_one_record_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pulling the k-th chunk has read exactly k records back from the parent's checkpoint.

    The laziness proof behind the previous test: a fully-read list wrapped in a non-``Sized``
    iterator would pass there and fail here. It also pins the 1:1 threading the Stage's contract
    requires — every chunk arrives, in order, exactly once.
    """
    _ProbeEnricher._reset()  # noqa: SLF001 - test-only probe reset
    _count_reads_of(monkeypatch, "checkpoints/content_screen.jsonl", _ProbeEnricher.chunk_reads)

    docs = _write_docs(tmp_path / "lazy", 6)
    LocalRunner(registry=_registry_with("context.probe", _ProbeEnricher)).run(
        _pipeline(f"file://{tmp_path}/lazy/store", docs, enricher="context.probe"),
        run_id="lazy",
    )

    pulls = _ProbeEnricher.chunk_pulls
    assert len(pulls) > 3, "the probe enricher saw too few chunks to be a test"
    # The counter spans the whole run (the chunk checkpoint is also read by Entity/Relation
    # Extraction and by the corpus-wide mention scan), so the k-th pull is measured from where
    # this Stage's own stream began: one read per pull, none of them ahead of the Provider.
    expected = [pulls[0] + k for k in range(len(pulls))]
    assert pulls == expected, (
        f"the chunk carrier was not pulled lazily: reads-per-pull {pulls[:5]}… over {len(pulls)} "
        f"chunks. Streaming means the k-th pull has read exactly k records; a materialised "
        f"carrier reads the whole checkpoint before the Provider sees anything."
    )


def test_graph_records_are_never_all_resident(tmp_path: Path) -> None:
    """While the Provider folds the graph, only the record in flight is alive.

    The graph side is the expensive one: a fixture-measured ~5.1 kB per ``GraphNode`` is ~2.5 GiB
    of nodes alone at UDA scale. The Provider projects each record to a compact view on arrival,
    so at any point mid-stream nothing but the record being read should exist — unless the Runner
    decoded them all up front, which is what the gathered partition did.
    """
    _ProbeEnricher._reset()  # noqa: SLF001 - test-only probe reset

    docs = _write_docs(tmp_path / "resident", 8)
    LocalRunner(registry=_registry_with("context.probe", _ProbeEnricher)).run(
        _pipeline(f"file://{tmp_path}/resident/store", docs, enricher="context.probe"),
        run_id="resident",
    )

    assert _ProbeEnricher.graph_seen, "the enrich Stage never ran"
    total = _ProbeEnricher.graph_seen[0]
    live = _ProbeEnricher.live_graph_mid_stream[0]
    assert total > 4 * _RESIDENCY_SAMPLE_AT, (
        f"only {total} GraphRecords in the run — too few for the residency assertion to bite"
    )
    assert live <= _RESIDENT_GRAPH_SLACK, (
        f"{live} GraphRecords were alive after reading only {_RESIDENCY_SAMPLE_AT} of the run's "
        f"{total}. A streamed carrier holds the record in flight; this is the whole corpus, "
        f"decoded before the Provider was called and resident for the Stage's execution."
    )


def test_streamed_run_is_byte_identical_to_a_materialised_run(tmp_path: Path) -> None:
    """Bounded memory cost no output: same enriched checkpoint, same ``records_in``.

    The parity control materialises both carriers at the one seam that changed, so any difference
    here — a dropped chunk, a reordered stream, a header built from a different KG — would be
    caused by streaming. ``records_in`` is checked too: it is now counted from the parents'
    checkpoint lines rather than measured off a list, and must still report the whole input.
    """
    docs = _write_docs(tmp_path / "parity", 5)
    store = f"file://{tmp_path}/parity/store"
    storage = Storage.for_uri(store)

    LocalRunner().run(_pipeline(store, docs), run_id="streamed")
    _MaterialisingRunner().run(_pipeline(store, docs), run_id="gathered")

    base_streamed = f"{store}/_latence/runs/streamed"
    base_gathered = f"{store}/_latence/runs/gathered"
    enriched = storage.read_text(f"{base_streamed}/checkpoints/enrich.jsonl")
    assert enriched.strip(), "the enrich Stage produced nothing"
    assert enriched == storage.read_text(f"{base_gathered}/checkpoints/enrich.jsonl"), (
        "the streamed run's enriched chunks differ from the materialised run's — streaming "
        "changed the output"
    )

    report = QualityReport.model_validate_json(
        storage.read_text(f"{base_streamed}/quality-report.json")
    )
    metrics = next(m for m in report.stages if m.name == "enrich")
    expected_in = _checkpoint_lines(storage, base_streamed, "content_screen") + (
        _checkpoint_lines(storage, base_streamed, "graph")
    )
    assert metrics.records_in == expected_in, (
        f"records_in for the enrich Stage is {metrics.records_in}, but its parents hold "
        f"{expected_in} records — the streamed count no longer reports the whole input."
    )


@pytest.mark.parametrize(
    ("depends_on", "expected"),
    [
        (["graph"], "received no ChunkRecords"),
        (["content_screen"], "received no GraphRecords"),
    ],
)
def test_mis_wired_context_enrichment_still_errors_streamed(
    tmp_path: Path, depends_on: list[str], expected: str
) -> None:
    """A missing ``depends_on`` edge still raises the same typed error, either side.

    The eager check validated a materialised list; the streamed one scans the parents for the
    carrier TYPES present. Streaming must not turn a wiring mistake into a silently-empty header
    (ADR-0034) — the Provider would happily produce chunks with no ``context_header`` at all.
    """
    docs = _write_docs(tmp_path / "wired", 2)
    pipeline = _pipeline(f"file://{tmp_path}/wired/store", docs)
    for stage in pipeline.stages:
        if stage.name == "enrich":
            stage.depends_on = depends_on

    with pytest.raises(ContractError, match=expected):
        LocalRunner().run(pipeline, run_id="wired")


def test_reference_enricher_still_satisfies_the_capability() -> None:
    """The probe stands in for the real Provider only if the real one has the same shape."""
    assert hasattr(ContextHeaderEnricher, "enrich")
    assert set(_ProbeEnricher.enrich.__annotations__) == set(
        ContextHeaderEnricher.enrich.__annotations__
    )
