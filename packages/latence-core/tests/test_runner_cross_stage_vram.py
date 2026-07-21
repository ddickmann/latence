"""dogfood-2b-process — the Runner reclaims a Stage's model between Stages (cross-stage VRAM).

The dogfood proof crashed at the mandatory ``redact`` Stage with a native CUDA
``OutOfMemoryError``: on the tutorial's 32 GB Blackwell pod the enterprise-sota stack loads a
fresh multi-GiB model per learned Stage (extract, embedding, disambiguation, redaction), and
because nothing released the earlier Stages' models the card was ~31 GiB resident by the time
``redaction.gliner2`` needed its 6.28 GiB attention-bias forward — only ~248 MiB free, so every
micro-batch retry (down to batch-of-one) OOM'd and NO run dir / records were ever written.

The root cause is a **cross-stage VRAM lifecycle** gap, not a redaction-batch gap: a torch
``nn.Module`` is held alive by a reference cycle, so dropping the last Python reference to a
Provider does NOT free its several-GiB model by refcount — and Python's own cyclic GC is triggered
by *Python* object-count thresholds the tiny Provider wrapper never trips (its weight is off-heap in
CUDA). Between two model Stages the collector may therefore never fire, and the models pile up. The
fix makes the Runner reclaim each Stage's model the moment its dispatch returns: drop the strong
reference and run an explicit ``gc.collect()`` (+ ``torch.cuda.empty_cache()`` where torch is
present), so the next Stage's model allocates on a near-empty card.

This test reproduces the leak **torch-free** and deterministically. Two chained model Stages each
build a stand-in "model" kept alive ONLY by a self-cycle (the nn.Module shape); automatic GC is
disabled for the run so the ONLY thing that can collect a finished Stage's model is the Runner's own
explicit ``gc.collect()``. Pre-fix the Runner never collects, so when the SECOND model Stage runs
the FIRST Stage's model is still resident (the pile-up that OOM'd the pod); post-fix the first
Stage's model has been reclaimed before the second runs.
"""

from __future__ import annotations

import gc
import weakref
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from latence_core import ProviderRegistry
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

# Module-level observation state, reset per test by ``_reset()``.
_LIVE_MODELS: set[str] = set()  # tags of stand-in models NOT yet reclaimed (their cycle survives)
_LIVE_WHEN_SECOND_RAN: list[frozenset[str]] = []  # snapshot taken while the 2nd model Stage runs


def _reset() -> None:
    _LIVE_MODELS.clear()
    _LIVE_WHEN_SECOND_RAN.clear()


class _CyclicModel:
    """A stand-in for a resident GPU model held alive ONLY by a reference cycle (nn.Module shape).

    ``self._self_ref = self`` makes the object unreachable-except-through-itself once its owning
    Provider is dropped: refcount alone never frees it (the cycle), so ONLY a ``gc.collect()``
    reclaims it — exactly the property that let real models pile up across Stages on the pod. A
    ``weakref.finalize`` removes the tag from :data:`_LIVE_MODELS` the instant the collector reclaims
    the cycle, so the live set is a faithful, side-effect-free readout of what is still resident.
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._self_ref = self  # the cycle: survives refcount-0, needs gc.collect() to reclaim
        _LIVE_MODELS.add(tag)
        weakref.finalize(self, _LIVE_MODELS.discard, tag)


class _CyclicModelRedactor:
    """A Redaction Provider (chunk→chunk) that loads a cyclic stand-in model tagged by config.

    Declares NO ProviderProfile so device routing follows the request and the Stage always executes
    (a gpu-only profile would be skip-with-flag'd on the CPU-only offline host and never construct a
    model). ``redact`` is an identity pass — it yields the same ChunkRecords — but the SECOND Stage
    to run snapshots the live-model set so the test can assert the FIRST Stage's model was reclaimed
    before it ran.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._tag = str(cfg["tag"])
        self._observe = bool(cfg.get("observe", False))
        self._model = _CyclicModel(self._tag)  # the only strong ref to the model is this attribute

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        if self._observe:
            # The first Stage has finished dispatching by now; whether its model is still in
            # _LIVE_MODELS is exactly the cross-stage-reclaim question this test pins.
            _LIVE_WHEN_SECOND_RAN.append(frozenset(_LIVE_MODELS))
        yield from chunks


def _registry_with(providers: dict[str, type]) -> ProviderRegistry:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access (mirrors sibling tests)
    for name, cls in providers.items():
        ep = metadata.EntryPoint(name=name, value="test:injected", group="latence.providers")
        object.__setattr__(ep, "load", lambda cls=cls: cls)
        discovered[name] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def _pipeline(store: str, docs_dir: Path) -> Pipeline:
    # source → parse → chunk → redact_a → redact_b : two chained model Stages, unambiguous order.
    return Pipeline.from_yaml_text(
        f"""
name: cross-stage-vram
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs_dir}'
      sensitivity: internal
      extensions: [txt]
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: redact_a
    capability: redaction
    provider: redaction.cyclic_a
    depends_on: [chunk]
    config: {{tag: redact_a}}
  - name: redact_b
    capability: redaction
    provider: redaction.cyclic_b
    depends_on: [redact_a]
    config: {{tag: redact_b, observe: true}}
"""
    )


def _docs(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "memo.txt").write_text("Acme Corp shipped the order to Berlin.\n", encoding="utf-8")
    return docs


def test_runner_reclaims_each_stage_model_before_the_next_stage_runs(tmp_path: Path) -> None:
    """REGRESSION (dogfood-2b-process): the FIRST model Stage's model is reclaimed before the SECOND.

    Pre-fix (no cross-stage reclaim) this FAILS: with automatic GC disabled the first Stage's
    cycle-held model is never collected, so ``redact_a`` is still resident when ``redact_b`` runs —
    the pile-up that left ~248 MiB free and OOM'd the redaction forward on the 32 GB pod.
    """
    _reset()
    docs = _docs(tmp_path)
    store = f"file://{tmp_path}/store"
    registry = _registry_with(
        {"redaction.cyclic_a": _CyclicModelRedactor, "redaction.cyclic_b": _CyclicModelRedactor}
    )

    # Disable automatic cyclic GC for the duration of the run so the ONLY thing that can reclaim a
    # finished Stage's cycle-held model is the Runner's OWN explicit gc.collect() — making the
    # cross-stage-reclaim behaviour the sole variable. gc.collect() still runs when called directly
    # (a manual collection ignores the disable flag), so the fix is exercised; the ambient collector
    # is silenced so a pre-fix run cannot be rescued by an incidental collection.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        report = LocalRunner(registry=registry).run(_pipeline(store, docs), run_id="vram")
    finally:
        if gc_was_enabled:
            gc.enable()

    assert report.contracts_complete
    # The second model Stage ran exactly once and took its snapshot.
    assert len(_LIVE_WHEN_SECOND_RAN) == 1
    live_when_second_ran = _LIVE_WHEN_SECOND_RAN[0]
    assert live_when_second_ran == frozenset({"redact_b"}), (
        "when the second model Stage ran, only its own model must be resident — the first Stage's "
        f"model must already have been reclaimed. Observed live models: {live_when_second_ran} "
        "(pre-fix this includes 'redact_a': the cross-stage VRAM leak that OOM'd the pod)."
    )
