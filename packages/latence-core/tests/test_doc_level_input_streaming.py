"""Doc-level Stage INPUT streaming (ADR-0033, regression for audit R2).

C6 streamed the **output** side: ``_dispatch_stage`` hands the Provider's iterator back
uncollapsed and the Runner writes it record-by-record into the checkpoint, so a doc-level Stage's
records never exist as a list. Its **input** side was still materialised whole-corpus:
``LocalRunner._gather_inputs`` concatenated every parent's records into one
``list[StageOutput]`` that lived in ``ctx.incoming`` for the Stage's entire execution — and
``_as_type`` then built a SECOND full list of the same size (a third under
``extract_on: masked``, via ``model_copy``). An Entity Stage over a 1M-chunk corpus therefore
still held ~1M ChunkRecords while the ADR recorded that transient as ``O(checkpoint batch)``.

These tests pin the input side of the bound, for the doc-level Stages whose Capability Protocol
lets the Runner stream it (Parse, Chunk, Schema Induction, Entity Extraction, Redaction, Fused):

* the Runner never builds the input list at all — what reaches the Stage is a lazy stream, at
  every corpus size (``test_doc_level_stage_input_is_never_materialised_whole_corpus``);
* the stream is really lazy, not a list in a wrapper: pulling the *k*-th input record reads
  exactly *k* records back from the parent's checkpoint
  (``test_entity_stage_pulls_its_input_one_record_at_a_time``);
* a mis-wired doc-level Stage still fails with the same typed ``ContractError`` rather than
  silently consuming the wrong carrier (``test_mis_wired_streamed_stage_still_raises_typed_error``).
"""

from __future__ import annotations

from collections.abc import Sized
from typing import TYPE_CHECKING, Any

import pytest
from latence_core import ProviderRegistry
from latence_core.contracts import ChunkRecord, EntityMention, Record
from latence_core.errors import ContractError
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner, StageOutput
from latence_core.storage import Storage

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from importlib import metadata as _metadata
    from pathlib import Path

    from latence_core.pipeline import Stage

from importlib import metadata

# The doc-level Stages whose input the Runner streams (their Providers take an ``Iterable`` and
# consume it once). Screening/Relation/Export are excluded on purpose: their Provider Protocol
# materialises the input itself (an outcome object / a two-carrier partition / a file write).
_STREAMED_STAGE_NAMES = ("parse", "chunk", "entities")


def _pipeline(store: str, docs_dir: Path, *, entity_provider: str = "entity.gazetteer") -> Pipeline:
    """Source→Parse→Chunk→Entity — the shortest DAG with three streamable doc-level Stages."""
    return Pipeline.from_yaml_text(
        f"""
name: doc-input-stream
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
  - name: entities
    capability: entity_extraction
    provider: {entity_provider}
    depends_on: [chunk]
    config:
      labels:
        ORG:
          terms: [Acme Corporation, Acme, International Business Machines, IBM]
        PERSON:
          terms: [Alice, Bob]
"""
    )


def _write_docs(tmp_path: Path, n: int) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    for i in range(n):
        (docs / f"doc{i:03d}.txt").write_text(
            "\n\n".join(
                f"# Section {s}\n\nAlice works at Acme. Bob joined IBM. "
                f"Document {i} section {s} body text long enough to chunk."
                for s in range(3)
            ),
            encoding="utf-8",
        )
    return docs


class _InputSpyRunner(LocalRunner):
    """Records, per Stage, how many input records the Runner MATERIALISED before dispatch.

    ``None`` means the Stage received a lazy stream (nothing materialised); an ``int`` is the
    length of the list the Runner built — the transient this test bounds.
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
        incoming: Iterable[StageOutput],
        outputs: dict[str, list[StageOutput]],
    ) -> Iterator[StageOutput]:
        self.materialised_input[stage.name] = (
            len(incoming) if isinstance(incoming, Sized) else None
        )
        return super()._dispatch_stage(
            stage, provider, storage, pipeline, run_id, incoming, outputs
        )


def _run_and_spy(tmp_path: Path, *, docs: int, tag: str) -> _InputSpyRunner:
    doc_dir = _write_docs(tmp_path / tag, docs)
    runner = _InputSpyRunner()
    runner.run(_pipeline(f"file://{tmp_path}/{tag}/store", doc_dir), run_id=tag)
    return runner


def test_doc_level_stage_input_is_never_materialised_whole_corpus(tmp_path: Path) -> None:
    """A streamable doc-level Stage's input is a lazy stream at ANY corpus size.

    Pre-fix the Runner called ``_gather_inputs`` for every non-accumulator Stage and handed the
    resulting ``list`` to the executor, so this recorded a count that grew exactly linearly with
    the corpus (the auditor measured 48 → 192 over a 4× scale-up). The bound ADR-0033 claims
    requires that no such list exists at all.
    """
    small = _run_and_spy(tmp_path, docs=4, tag="small")
    big = _run_and_spy(tmp_path, docs=16, tag="big")

    for name in _STREAMED_STAGE_NAMES:
        assert small.materialised_input[name] is None and big.materialised_input[name] is None, (
            f"Stage {name!r} received a MATERIALISED input list: "
            f"{small.materialised_input[name]} records at 4 documents, "
            f"{big.materialised_input[name]} records at 16 documents — the input transient is "
            f"corpus-linear, not O(checkpoint batch). ADR-0033's 'Honest memory boundary' "
            f"claims this transient is bounded."
        )


class _ProbeExtractor:
    """An EntityExtractor that records how much of the parent checkpoint was read per pull."""

    # reads-seen-so-far at the moment each input chunk was pulled, across the whole run.
    pulls: list[int] = []  # noqa: RUF012 - test-only shared probe
    reads: list[int] = [0]  # noqa: RUF012 - test-only shared counter

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def extract(self, chunks: Iterable[ChunkRecord]) -> Iterator[EntityMention]:
        for _chunk in chunks:
            _ProbeExtractor.pulls.append(_ProbeExtractor.reads[0])
        return
        yield  # pragma: no cover - makes this a generator function


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


def test_entity_stage_pulls_its_input_one_record_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pulling the k-th input chunk reads exactly k records back from the parent checkpoint.

    The laziness proof behind the previous test: a non-``Sized`` wrapper around a fully-read list
    would pass there and fail here. Pre-fix every chunk was read (and decoded, and held) before
    the Provider saw its first one, so the first pull already recorded the whole corpus.
    """
    _ProbeExtractor.pulls = []
    _ProbeExtractor.reads = [0]

    original = LocalRunner._stream_checkpoint

    def _counting_stream(
        self: LocalRunner,
        storage: Storage,
        uri: str,
        record_type: type[StageOutput] | type[Record],
    ) -> Iterator[StageOutput]:
        for record in original(self, storage, uri, record_type):
            if uri.endswith("checkpoints/chunk.jsonl"):
                _ProbeExtractor.reads[0] += 1
            yield record

    monkeypatch.setattr(LocalRunner, "_stream_checkpoint", _counting_stream)

    docs = _write_docs(tmp_path / "lazy", 6)
    pipeline = _pipeline(
        f"file://{tmp_path}/lazy/store", docs, entity_provider="entity.probe"
    )
    LocalRunner(registry=_registry_with("entity.probe", _ProbeExtractor)).run(
        pipeline, run_id="lazy"
    )

    assert len(_ProbeExtractor.pulls) > 3, "the probe extractor saw too few chunks to be a test"
    assert _ProbeExtractor.pulls == list(range(1, len(_ProbeExtractor.pulls) + 1)), (
        f"the Entity Stage's input was not pulled lazily: reads-per-pull "
        f"{_ProbeExtractor.pulls[:5]}… over {len(_ProbeExtractor.pulls)} chunks. Streaming means "
        f"the k-th pull has read exactly k records; a materialised input reads the whole "
        f"checkpoint before the Provider sees anything."
    )


def test_mis_wired_streamed_stage_still_raises_typed_error(tmp_path: Path) -> None:
    """A doc-level Stage fed the wrong carrier still fails with the typed ContractError.

    Streaming the input must not turn a wiring mistake into a silently-empty Stage (ADR-0034).
    Chunk is wired straight to Source, so it receives ParserInputs where it requires
    DocumentRecords.
    """
    docs = _write_docs(tmp_path / "wired", 2)
    store = f"file://{tmp_path}/wired/store"
    pipeline = Pipeline.from_yaml_text(
        f"""
name: mis-wired
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config: {{path: 'file://{docs}', sensitivity: internal, extensions: [txt]}}
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [source]
"""
    )
    with pytest.raises(ContractError, match="requires DocumentRecord inputs but received"):
        LocalRunner().run(pipeline, run_id="wired")
