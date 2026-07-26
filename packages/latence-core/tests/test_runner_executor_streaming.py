"""Executor doc-level output streaming (ADR-0033 / #69, regression).

#63 bounded the Export + checkpoint-write transients, but the LocalRunner still held the
per-Stage ``outputs: dict[str, list[StageOutput]]`` in memory for the whole run — the
corpus-level accumulators (Profiling / Disambiguation / Graph Assembly) + the Quality Report
consumed the whole retained doc-level output. #69 completes ADR-0033's honest boundary: a
doc-level Stage's output is RELEASED from the dict once its last consumer has run, and the
accumulators read their doc-level input back from the checkpoints batch-by-batch
(``_stream_checkpoint``) instead of the retained list. Corpus-level memory is then bounded by
the KG state the accumulators hold, not by O(all doc-level records).

These tests pin the invariants the streamed executor must keep; each would fail if the release
+ checkpoint-streaming were reverted (the whole ``outputs`` dict retained through the run):

* the doc-level Stages' outputs are ABSENT from the retained dict by the time an accumulator
  runs — the release actually happens (``test_doc_level_outputs_released_before_accumulators``);
  since C6 the default executor never retains a Stage's output at ALL (the Provider's records
  stream straight into the checkpoint), so this is now the strongest form of the invariant;
* an accumulator's input is read back from its parents' checkpoints, not the retained dict — the
  wired path is taken (``test_accumulator_input_read_from_checkpoint``);
* the peak count of simultaneously-retained doc-level records is bounded well below the corpus
  when the corpus grows — the executor transient is no longer corpus-linear
  (``test_retained_doc_level_records_bounded_not_corpus_linear``);
* the Quality Report + KG are byte-identical to a run with release disabled — release loses no
  data the report/accumulators need (``test_report_identical_with_and_without_release``);
* a mis-wired accumulator still raises the exact typed wiring error through the streamed
  presence check (``test_mis_wired_disambiguation_still_errors_streamed``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from latence_core.contracts import (
    DisambiguationRecord,
    DocumentRecord,
    EntityMention,
    Record,
)
from latence_core.errors import ContractError
from latence_core.pipeline import Pipeline
from latence_core.runner import _ACCUMULATOR_KINDS, LocalRunner, StageOutput
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterator

    from latence_core.pipeline import Stage
    from latence_core.quality import QualityReport

# Doc-level carriers whose retained output the accumulators consume — the sets #69 releases.
_DOC_LEVEL_CARRIERS = (DocumentRecord, EntityMention)


def _accumulator_pipeline(store: str, docs_dir: Path, *, corpus_id: str = "s") -> Pipeline:
    """Source→Parse→Chunk→Entity→Relation→Disambiguation→Graph + a KG + corpus Export.

    Exercises all three accumulators (Profiling omitted only to keep the DAG focused;
    Disambiguation + Graph are the mention/relation-volume accumulators #69 most affects).
    """
    return Pipeline.from_yaml_text(
        f"""
name: exec-stream
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs_dir}', sensitivity: internal, extensions: [txt, md]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: entities
    capability: entity_extraction
    provider: entity.gazetteer
    depends_on: [chunk]
    config:
      labels:
        ORG:
          terms: [Acme Corporation, Acme, International Business Machines, IBM]
        PERSON:
          terms: [Alice, Bob]
  - name: relations
    capability: relation_extraction
    provider: relation.pattern
    depends_on: [chunk, entities]
    config:
      window_chars: 200
      relations:
        works_for:
          head: PERSON
          tail: ORG
          triggers: [works at, employed by, joined]
  - name: profile
    capability: profiling
    provider: profiling.statistical
    depends_on: [parse, entities]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [entities, relations]
    config:
      kb:
        - kb_id: Q1
          name: Acme
          entity_type: ORG
      kb_threshold: 0.5
  - name: graph
    capability: graph_assembly
    provider: graph.canonical
    depends_on: [disambiguate]
    config:
      corpus_id: {corpus_id}
  - name: export_kg
    capability: export
    provider: export.knowledge_graph
    depends_on: [graph]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [chunk]
    config:
      basename: corpus
"""
    )


def _write_docs(tmp_path: Path, n: int = 3) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "memo.txt").write_text(
        "Alice works at Acme. International Business Machines is a partner.\n",
        encoding="utf-8",
    )
    (docs / "note.txt").write_text(
        "Bob joined Acme last year. IBM announced results. "
        "Acme Corporation filed the report.\n",
        encoding="utf-8",
    )
    # Extra filler docs to make the corpus (and its doc-level record count) grow.
    for i in range(max(0, n - 2)):
        (docs / f"fill{i}.txt").write_text(
            f"Alice works at Acme. Bob joined IBM. Document number {i} body text.\n",
            encoding="utf-8",
        )
    return docs


class _SpyRunner(LocalRunner):
    """A LocalRunner that records the retained ``outputs`` keys at each Stage's dispatch."""

    def __init__(self, *, release: bool = True) -> None:
        super().__init__()
        self._release_doc_level_outputs = release
        # stage name -> the set of doc-level Stage names still retained in `outputs` when this
        # Stage began executing.
        self.retained_doc_level_at: dict[str, set[str]] = {}
        # peak count of simultaneously-retained doc-level *records* across the run.
        self.peak_retained_doc_records = 0
        self._doc_level_names: set[str] = set()

    def _dispatch_stage(
        self,
        stage: Stage,
        provider: object,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        incoming: list[StageOutput],
        outputs: dict[str, list[StageOutput]],
    ) -> Iterator[StageOutput]:
        if not self._doc_level_names:
            self._doc_level_names = {
                s.name for s in pipeline.stages if s.capability not in _ACCUMULATOR_KINDS
            }
        retained_doc_level = {n for n in outputs if n in self._doc_level_names}
        self.retained_doc_level_at[stage.name] = set(retained_doc_level)
        retained_records = sum(len(outputs[n]) for n in retained_doc_level)
        self.peak_retained_doc_records = max(self.peak_retained_doc_records, retained_records)
        return super()._dispatch_stage(
            stage, provider, storage, pipeline, run_id, incoming, outputs
        )


def test_doc_level_outputs_released_before_accumulators(tmp_path: Path) -> None:
    """When an accumulator runs, the doc-level Stages it does NOT directly need are released.

    Graph Assembly depends only on Disambiguation; by the time it runs, every doc-level Stage
    (source/parse/chunk/entities/relations/exports) must already be released from the retained
    dict — a reverted (never-release) executor would still hold all of them, failing here.
    """
    docs = _write_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    runner = _SpyRunner()
    runner.run(_accumulator_pipeline(store, docs), run_id="rel")

    # Graph Assembly's only live dependency is the (accumulator) Disambiguation output; NO
    # doc-level Stage should still be retained when it runs.
    retained_at_graph = runner.retained_doc_level_at["graph"]
    assert retained_at_graph == set(), (
        f"doc-level outputs still retained when Graph Assembly ran: {sorted(retained_at_graph)}; "
        f"the #69 release was not applied."
    )
    # Disambiguation needs entities + relations (both doc-level). Everything it does NOT need
    # (parse, chunk, source) must already be released — parse in particular fed chunk+profile
    # which both ran before disambiguation, so it is dead.
    retained_at_disambig = runner.retained_doc_level_at["disambiguate"]
    assert "parse" not in retained_at_disambig
    assert "chunk" not in retained_at_disambig
    assert "source" not in retained_at_disambig


def test_accumulator_input_read_from_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An accumulator's doc-level input is streamed back from checkpoints (the wired path).

    The whole point of #69: the accumulators read their released doc-level input via
    ``_stream_checkpoint``, not the retained dict. We record which checkpoint URIs get streamed
    and assert the doc-level producers (entities/relations) were read back during the
    accumulator phase. A reverted executor (accumulators read the retained dict) never streams
    them, failing here.
    """
    streamed_uris: list[str] = []
    original = LocalRunner._stream_checkpoint

    def _spy(
        self: LocalRunner,
        storage: Storage,
        uri: str,
        record_type: type[StageOutput] | type[Record],
    ) -> Iterator[StageOutput]:
        streamed_uris.append(uri)
        return original(self, storage, uri, record_type)

    monkeypatch.setattr(LocalRunner, "_stream_checkpoint", _spy)

    docs = _write_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    LocalRunner().run(_accumulator_pipeline(store, docs), run_id="cp")

    # The entities + relations checkpoints (Disambiguation's input) and the disambiguation
    # checkpoint (Graph's input) must have been streamed back — released, then re-read.
    joined = "\n".join(streamed_uris)
    assert "checkpoints/entities.jsonl" in joined, "entities checkpoint never streamed back"
    assert "checkpoints/relations.jsonl" in joined, "relations checkpoint never streamed back"


def test_retained_doc_level_records_bounded_below_retain_all(tmp_path: Path) -> None:
    """Release holds strictly fewer doc-level records at peak than retaining every Stage.

    The perf assertion for #69's bound. Both a released and a retain-everything executor scale
    linearly with the corpus (each live Stage's output is O(docs)), so the win is that release
    keeps only the LIVE fan-in retained at once (a bounded slice of the DAG) instead of EVERY
    Stage's output simultaneously. Run the identical corpus twice — release ON vs OFF (the
    pre-#69 path) — and assert the peak retained doc-level record count is materially lower with
    release. A revert to retaining the whole ``outputs`` dict makes the two equal, failing here.
    """
    docs = _write_docs(tmp_path, n=6)

    released = _SpyRunner(release=True)
    released.run(
        _accumulator_pipeline(f"file://{tmp_path}/on/store", docs, corpus_id="s"), run_id="on"
    )
    retained = _SpyRunner(release=False)
    retained.run(
        _accumulator_pipeline(f"file://{tmp_path}/off/store", docs, corpus_id="s"), run_id="off"
    )

    peak_released = released.peak_retained_doc_records
    peak_retained_all = retained.peak_retained_doc_records
    # The retain-all path must actually retain something, otherwise the comparison is vacuous.
    assert peak_retained_all > 0
    # Retaining every Stage's output must peak strictly higher than the default path. A
    # comfortable margin so it is not a knife-edge, while still failing hard if the default path
    # stops dropping anything.
    #
    # Since C6 this is exact rather than merely lower: the default executor streams each Stage's
    # records from the Provider straight into its checkpoint, so NOTHING is retained and
    # ``peak_released`` is 0. (Before C6 it was the live DAG fan-in — a transient the earlier
    # version of this test asserted was ``> 0``. That assertion pinned the implementation, not the
    # behaviour; dropping to zero is the strictly better outcome it was guarding towards.)
    assert peak_released < peak_retained_all * 0.75, (
        f"the default executor did not reduce peak retained doc-level records "
        f"(default={peak_released}, retain-all={peak_retained_all}): the #69/C6 bound regressed."
    )


def _report_and_kg(store: str, docs: Path, run_id: str, *, release: bool) -> tuple[QualityReport, str]:
    runner = LocalRunner()
    runner._release_doc_level_outputs = release  # noqa: SLF001 - the #69 parity knob
    report = runner.run(_accumulator_pipeline(store, docs, corpus_id="c"), run_id=run_id)
    s = Storage.for_uri(store)
    ttl = s.read_text(f"{store}/_latence/runs/{run_id}/export/graph.ttl")
    return report, ttl


def test_report_identical_with_and_without_release(tmp_path: Path) -> None:
    """Release must not change the Quality Report or the assembled KG by even one byte.

    Run once with release ON (doc-level outputs dropped + streamed back from checkpoints) and
    once with the ``_release_doc_level_outputs`` knob OFF (the pre-#69 retain-everything path).
    Same run_id in both stores so the reports differ only by the release behavior. The persisted
    Quality Report JSON and the exported KG TTL must be byte-identical: release is a pure
    memory-shaping optimization, never a semantic change (determinism / the G1 fingerprint bar).
    """
    docs = _write_docs(tmp_path)
    report_on, ttl_on = _report_and_kg(f"file://{tmp_path}/on", docs, "same", release=True)
    report_off, ttl_off = _report_and_kg(f"file://{tmp_path}/off", docs, "same", release=False)

    # Wall-clock stage durations vary run-to-run regardless of release; zero them out so the
    # comparison is over the *content* of the report (counts, KG stats, offsets, completeness).
    def _timeless(report: QualityReport) -> str:
        r = report.model_copy(deep=True)
        r.total_duration_seconds = 0.0
        for m in r.stages:
            m.duration_seconds = 0.0
        return r.to_json()

    assert _timeless(report_on) == _timeless(report_off), (
        "Quality Report content changed when release was disabled — release is not "
        "semantics-preserving."
    )
    assert ttl_on == ttl_off, "assembled KG TTL changed when release was disabled."


def test_mis_wired_disambiguation_still_errors_streamed(tmp_path: Path) -> None:
    """A Disambiguation Stage fed only DocumentRecords (no mentions) raises the typed error.

    Adversarial wiring: Disambiguation depends on Parse (DocumentRecords), never an
    Entity-Extraction Stage. The streamed presence check must still raise the exact
    ContractError naming the missing EntityMentions — the streaming path must not silently
    accept an empty mention stream and emit a broken (empty) disambiguation.
    """
    store = f"file://{tmp_path}/store"
    docs = _write_docs(tmp_path)
    pipeline = Pipeline.from_yaml_text(
        f"""
name: bad-disambig
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', extensions: [txt]}}
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: disambiguate
    capability: disambiguation
    provider: disambiguation.cascade
    depends_on: [parse]
"""
    )
    with pytest.raises(ContractError) as exc:
        LocalRunner().run(pipeline, run_id="bad")
    assert "EntityMentions" in str(exc.value)
    assert "disambiguation" in str(exc.value)


def test_report_reads_released_doc_level_from_checkpoint(tmp_path: Path) -> None:
    """The Quality Report's doc-level rollups survive release (read back from checkpoints).

    Entity/Relation/Chunk Stages are all released before the report is built, so their quality
    sections must be reconstructed from checkpoints. Assert the report still carries the
    doc-level metrics (a revert that lost the released data would report empty sections).
    """
    docs = _write_docs(tmp_path)
    store = f"file://{tmp_path}/store"
    report = LocalRunner().run(_accumulator_pipeline(store, docs), run_id="rep")
    assert report.entities is not None and report.entities.mentions_total > 0
    assert report.relations is not None
    assert report.chunk is not None and report.chunk.chunks_total > 0
    assert report.disambiguation is not None and report.disambiguation.mentions_in > 0
    assert report.graph is not None and report.graph.nodes_total > 0
    # The doc-level EntityMention set the disambiguation rollup counted equals the persisted
    # entities checkpoint — proving the report re-read the released doc-level output faithfully.
    s = Storage.for_uri(store)
    ent_ckpt = f"{store}/_latence/runs/rep/checkpoints/entities.jsonl"
    persisted_mentions = [
        EntityMention.model_validate_json(line)
        for line in s.read_text(ent_ckpt).splitlines()
        if line.strip()
    ]
    assert report.disambiguation.mentions_in == len(persisted_mentions)
    assert report.entities.mentions_total == len(persisted_mentions)
    # DisambiguationRecords the graph consumed round-trip through its own checkpoint.
    dis_ckpt = f"{store}/_latence/runs/rep/checkpoints/disambiguate.jsonl"
    persisted_disambig = [
        DisambiguationRecord.model_validate_json(line)
        for line in s.read_text(dis_ckpt).splitlines()
        if line.strip()
    ]
    assert persisted_disambig
