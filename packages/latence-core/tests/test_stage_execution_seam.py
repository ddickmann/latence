"""StageExecution — the Stage-execution seam's own contract (C3 + C6).

The interface a Runner adapter must learn is now three names: ``execute(stage, ctx)``,
``outcome.records``, ``outcome.metrics(records_in=...)``. These tests pin what the module
guarantees behind them, and in particular the ONE ordering that makes lazy Stage output sound.

**The drain ordering.** Under the pre-C6 shape the Provider teardown lived in
``_execute_stage``'s ``finally``: it fired when the dispatch *returned*. Make the dispatch lazy
and that ``finally`` fires **before the Provider has produced a single record** — every tally
reads zero and ``del provider`` + ``empty_cuda_cache`` free the model mid-generation. The teardown
therefore moved into the producing generator's own ``finally``, which the outcome owns. That is
not a detail; it is the whole reason a Runner cannot get streaming wrong by accident.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from importlib import metadata
from pathlib import Path
from typing import IO, Any

import pytest
from latence_core import ProviderRegistry
from latence_core.contracts import ParserInput
from latence_core.errors import ContractError
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.stage_execution import StageContext
from latence_core.storage import Storage


class _CountingParser:
    """A Parser whose ``endpoint_retries`` tally only reaches its final value at the LAST record.

    This is the shape of every real endpoint-backed Provider: the counter grows as the Provider
    works. A Runner that drains it before consuming the stream reads 0.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self.endpoint_retries = 0
        self.freed_after: int | None = None
        self.records_yielded = 0

    def parse(self, inputs: Iterable[ParserInput]) -> Iterator[Any]:
        from latence_parser_plaintext.provider import PlaintextParser

        inner = PlaintextParser(self._config)
        for record in inner.parse(inputs):
            self.endpoint_retries += 1
            self.records_yielded += 1
            yield record


def _registry(name: str, cls: type) -> ProviderRegistry:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access
    ep = metadata.EntryPoint(name=name, value="test:injected", group="latence.providers")
    object.__setattr__(ep, "load", lambda: cls)
    discovered[name] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def _pipeline(store: str, docs: Path, parser: str = "parser.plaintext") -> Pipeline:
    return Pipeline.from_yaml_text(
        f"""
name: seam
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: {parser}
    depends_on: [source]
"""
    )


def _docs(tmp_path: Path, n: int = 4) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(n):
        (docs / f"d{i}.txt").write_text(f"Acme document number {i}.\n", encoding="utf-8")
    return docs


def test_provider_tallies_are_drained_after_the_stream_not_before(tmp_path: Path) -> None:
    """The C6 correctness core: the retry tally reflects the WHOLE run, not zero.

    Draining eagerly (the pre-C6 ``finally``) under lazy iteration yields
    ``endpoint_retries == 0`` here, because the Provider had not produced anything yet.
    """
    docs = _docs(tmp_path, n=4)
    store = f"file://{tmp_path}/store"
    report = LocalRunner(registry=_registry("parser.counting", _CountingParser)).run(
        _pipeline(store, docs, parser="parser.counting"), run_id="r"
    )

    parse = next(m for m in report.stages if m.name == "parse")
    assert parse.records_out == 4
    assert parse.endpoint_retries == 4, (
        "the Provider's tally was read before its stream was consumed — every per-Stage counter "
        "(endpoint retries, schema-induction columns, relation fan-out) would silently be zero."
    )


def test_records_are_streamed_not_materialised(tmp_path: Path) -> None:
    """A Stage's records reach the checkpoint incrementally — the Provider is still mid-stream.

    A materialising Runner has ``records_yielded == 4`` before the first record is ever written;
    the streaming one is at 1. Asserted through the writer, so it is a property of the wiring, not
    of the seam in isolation.
    """
    docs = _docs(tmp_path, n=4)
    store = f"file://{tmp_path}/store"
    provider_holder: list[_CountingParser] = []
    yielded_at_first_write: list[int] = []

    class _Spy(_CountingParser):
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            super().__init__(config)
            provider_holder.append(self)

    class _SpyRunner(LocalRunner):
        def _stream_batches(
            self,
            fh: IO[bytes],
            storage: Storage,
            records: Iterable[Any],
            **kwargs: Any,
        ) -> tuple[int, int]:
            def _watch() -> Iterator[Any]:
                for record in records:
                    if provider_holder and not yielded_at_first_write:
                        yielded_at_first_write.append(provider_holder[0].records_yielded)
                    yield record

            return super()._stream_batches(fh, storage, _watch(), **kwargs)

    runner = _SpyRunner(registry=_registry("parser.counting", _Spy))
    runner.run(_pipeline(store, docs, parser="parser.counting"), run_id="r")

    assert yielded_at_first_write == [1], (
        f"the Provider had produced {yielded_at_first_write} records by the time the first one "
        f"reached the checkpoint writer; streaming means exactly one. A materialising dispatch "
        f"(list(provider...)) reports the whole corpus slice here."
    )


def test_metrics_before_the_stream_is_consumed_is_a_loud_error(tmp_path: Path) -> None:
    """A Runner that forgets to drain gets a RuntimeError, never a quietly-zeroed metrics row."""
    docs = _docs(tmp_path, n=2)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    storage = Storage.for_uri(store)
    runner = LocalRunner()
    stage = pipeline.stages[0]

    outcome = runner.stage_execution.execute(
        stage,
        StageContext(
            storage=storage,
            pipeline=pipeline,
            run_id="r",
            incoming=[],
            outputs={},
            runner="local",
        ),
    )
    with pytest.raises(RuntimeError, match="before its record stream was consumed"):
        outcome.metrics(records_in=0)


def test_record_stream_is_one_shot(tmp_path: Path) -> None:
    """Taking the records twice is a loud error, not a silently empty second read."""
    docs = _docs(tmp_path, n=2)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs)
    runner = LocalRunner()
    outcome = runner.stage_execution.execute(
        pipeline.stages[0],
        StageContext(
            storage=Storage.for_uri(store),
            pipeline=pipeline,
            run_id="r",
            incoming=[],
            outputs={},
            runner="local",
        ),
    )
    assert len(list(outcome.records)) == 2
    with pytest.raises(RuntimeError, match="already taken"):
        _ = outcome.records


def test_a_consumer_side_failure_still_reports_the_time_and_retries_it_burned(
    tmp_path: Path,
) -> None:
    """The FAILED row is honest even when the exception came from the CONSUMER, not the Provider.

    The producing generator is *suspended at a yield* when the checkpoint/WAL/storage write raises,
    so its ``finally`` — the one that stops the clock, drains the Provider tallies and releases the
    model — has not run when the Runner's ``except`` branch calls ``metrics(error=...)``. Reading
    the counters there without finalizing the stream reports ``duration_seconds=0.0`` and
    ``endpoint_retries=0`` for a Stage that burned real time and real retries, and leaks the
    Provider (no ``del`` / no CUDA-cache release). The outcome owns the Provider's whole lifetime,
    so finalizing is its job, not a fourth name a Runner adapter has to learn.
    """
    torn_down: list[bool] = []

    class _RetryingParser:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self._config = config or {}
            self.endpoint_retries = 0

        def parse(self, inputs: Iterable[ParserInput]) -> Iterator[Any]:
            from latence_parser_plaintext.provider import PlaintextParser

            inner = PlaintextParser(self._config)
            try:
                for record in inner.parse(inputs):
                    self.endpoint_retries += 1
                    time.sleep(0.005)  # real, measurable work behind each record
                    yield record
            finally:
                torn_down.append(True)

    class _FullDiskRunner(LocalRunner):
        """The consumer fails: the checkpoint writer dies on the 4th record."""

        def _stream_batches(
            self,
            fh: IO[bytes],
            storage: Storage,
            records: Iterable[Any],
            **kwargs: Any,
        ) -> tuple[int, int]:
            if kwargs.get("stage_name") != "parse":
                return super()._stream_batches(fh, storage, records, **kwargs)
            seen = 0
            for _record in records:
                seen += 1
                if seen == 4:
                    raise OSError("[Errno 28] No space left on device")
            return seen, 0  # pragma: no cover - the writer always fails here

    docs = _docs(tmp_path, n=6)
    store = f"file://{tmp_path}/store"
    runner = _FullDiskRunner(registry=_registry("parser.retrying", _RetryingParser))

    with pytest.raises(OSError, match="No space left on device"):
        runner.run(_pipeline(store, docs, parser="parser.retrying"), run_id="r")

    import json

    storage = Storage.for_uri(store)
    report_uri = f"{store}/_latence/runs/r/quality-report.json"
    stages = {s["name"]: s for s in json.loads(storage.read_text(report_uri))["stages"]}
    parse = stages["parse"]
    assert parse["error_count"] == 1
    assert parse["endpoint_retries"] == 4, (
        "the FAILED row reported the retries as "
        f"{parse['endpoint_retries']} — the Provider burned 4. The producing generator was "
        "suspended at its yield and its tally drain never ran."
    )
    assert parse["duration_seconds"] > 0.0, (
        "the FAILED row reported duration_seconds=0.0 for a Stage that did ~20ms of real work: "
        "the clock is stopped in a finally that a consumer-side failure never reaches."
    )
    assert torn_down == [True], (
        "the Provider's stream was never closed, so the seam's teardown (del provider + "
        "empty_cuda_cache) never ran — on a GPU stack the model stays resident."
    )


def test_stage_failures_surface_from_the_stream_with_populated_failure_metrics(
    tmp_path: Path,
) -> None:
    """A contract violation still raises typed — and the FAILED row still carries its columns.

    ``execute`` deliberately never raises: the failure comes out of the record stream, so the
    outcome (and therefore the Stage's device/profile columns) is still in the caller's hands when
    it builds the failed metrics row.
    """

    class _NotAParser:
        def __init__(self, config: dict[str, Any] | None = None) -> None:
            self._config = config or {}

    docs = _docs(tmp_path, n=1)
    store = f"file://{tmp_path}/store"
    pipeline = _pipeline(store, docs, parser="parser.bogus")
    runner = LocalRunner(registry=_registry("parser.bogus", _NotAParser))

    with pytest.raises(ContractError):
        runner.run(pipeline, run_id="r")

    storage = Storage.for_uri(store)
    report_uri = f"{store}/_latence/runs/r/quality-report.json"
    assert storage.exists(report_uri)
    import json

    stages = {s["name"]: s for s in json.loads(storage.read_text(report_uri))["stages"]}
    assert stages["parse"]["error_count"] == 1
    assert stages["parse"]["error_category"] == "contract"
    # The failed Stage's checkpoint was never published, so a resume re-runs it (no torn write) —
    # and the half-written temp file is cleaned up rather than left beside the checkpoints.
    assert not storage.exists(f"{store}/_latence/runs/r/checkpoints/parse.jsonl")
    leftovers = [
        u for u in storage.ls(f"{store}/_latence/runs/r/checkpoints") if "parse" in u
    ]
    assert leftovers == []
