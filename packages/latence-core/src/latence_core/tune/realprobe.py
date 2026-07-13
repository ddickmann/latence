"""The REAL registry-backed :class:`SampleProbe` — runs a Provider under a knob setting (W11).

The measuring half of :mod:`latence_core.tune`. It reuses the bake-off's timing/sample approach
(:mod:`latence_core.bakeoff.harness`) rather than reinventing it: it stages the SAME bundled
deterministic corpus, holds every OTHER stage of a base stack constant, swaps the tuned Provider's
knobs onto its stage, runs the stack through the :class:`~latence_core.runner.LocalRunner`, and:

* **times** the target-stage run (the K repeats' seconds → the tuner's median throughput);
* **fingerprints** the target-stage CHECKPOINT bytes (the newline-delimited ``model_dump_json``
  records — labels/spans/offsets/text, the deterministic G1 fingerprint the bake-off already uses
  for
  its determinism cross-check) → the correctness subject the gate compares to the baseline.

Crucially it NEVER raises for a bad candidate: a load OR **inference-time** fault (a
``BackendCompilerFailed`` on the first forward — the W9 lesson — surfaces from the Runner's stage
execution) is caught and returned as :attr:`ProbeOutcome.error`, so the gate rejects it as a
documented rejection. Heavy (runner/stage) imports are pulled in at call time so a bare
``import latence_core`` stays thin (ADR-0016); this module is only reached on a rig where the tuned
Provider is installed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING

from latence_core.tune.probe import ProbeOutcome

if TYPE_CHECKING:
    from latence_core.capability import ProviderRegistry
    from latence_core.pipeline import Pipeline
    from latence_core.runner import LocalRunner
    from latence_core.storage import Storage
    from latence_core.tune.report import KnobSetting


class RegistrySampleProbe:
    """Runs a Provider through the LocalRunner on the sample, timing + fingerprinting it (W11).

    Constructed with a base stack path + a Storage root it owns (state/checkpoints/corpus live under
    it, per-candidate-namespaced so two candidates never collide). It stages the bundled corpus ONCE
    (deterministic bytes), then :meth:`probe` runs one candidate: swap the knobs onto the Provider's
    stage, warmup once, time K runs, fingerprint the target-stage checkpoint — or catch a fault into
    the outcome.
    """

    def __init__(
        self,
        base_stack_path: Path,
        *,
        storage_root: str,
        registry: ProviderRegistry | None = None,
    ) -> None:
        from latence_core.capability import ProviderRegistry  # noqa: PLC0415 - heavy import at call
        from latence_core.storage import Storage  # noqa: PLC0415

        self._base_stack_path = base_stack_path
        self._root = storage_root.rstrip("/")
        self._storage: Storage = Storage.for_uri(storage_root)
        self._registry = registry or ProviderRegistry()
        self._corpus_dir: str | None = None
        # Providers for which the one-time process-global cost (model download, first-ever
        # load, torch.compile cache population) has already been paid by a global warmup. The
        # FIRST candidate of a provider must NOT be the one paying that in its timed region — that
        # is exactly the cold-baseline leak that produced the pod's bogus 17720× speedup. See
        # :meth:`_warm_globals_once`.
        self._globals_warmed: set[str] = set()

    def _ensure_corpus(self) -> str:
        """Stage the bundled corpus once; return its dir (the SAME sample per candidate)."""
        if self._corpus_dir is None:
            from latence_core.stacks.corpus import stage_validation_corpus  # noqa: PLC0415

            corpus_dir = f"{self._root}/_tune-corpus"
            stage_validation_corpus(self._storage, corpus_dir)
            self._corpus_dir = corpus_dir
        return self._corpus_dir

    def _stage_for_provider(self, pipeline: Pipeline, provider: str) -> str | None:
        """The stage name bound to ``provider`` (top-level), or ``None`` if none references it."""
        for stage in pipeline.stages:
            if stage.provider == provider:
                return stage.name
        return None

    def probe(
        self, provider: str, setting: KnobSetting, *, sample_size: int, repeats: int
    ) -> ProbeOutcome:
        """Run ``provider`` under ``setting`` on the sample; never raise (fault → outcome.error)."""
        try:
            return self._probe_inner(provider, setting, sample_size=sample_size, repeats=repeats)
        except Exception as exc:  # noqa: BLE001 - a bad candidate is a rejection, never an aborted sweep
            # This is the W9 catch: a load OR inference-time fault (BackendCompilerFailed on the
            # first
            # forward) surfaces here from the Runner; the tuner's gate classifies + rejects it.
            return ProbeOutcome(error=exc)

    def _probe_inner(
        self, provider: str, setting: KnobSetting, *, sample_size: int, repeats: int
    ) -> ProbeOutcome:
        from latence_core.runner import LocalRunner  # noqa: PLC0415 - heavy import at call time
        from latence_core.stacks.validate import load_stack  # noqa: PLC0415

        corpus_dir = self._ensure_corpus()
        # Per-candidate Storage namespace so two candidates never collide on run state.
        cand_root = f"{self._root}/candidates/{provider.replace('.', '_')}/{setting.label}"
        base = load_stack(self._base_stack_path, storage_uri=cand_root, corpus_dir=corpus_dir)
        stage_name = self._stage_for_provider(base, provider)
        if stage_name is None:
            # The provider is not a top-level stage in this base stack (e.g. a nested embedder). It
            # is
            # not directly runnable through the stage seam here, so it is not sweepable via this
            # base
            # stack — surface a clear error the gate records as a skip rather than a fabricated
            # number.
            msg = f"provider {provider!r} is not a top-level stage in {self._base_stack_path.name}"
            raise ValueError(msg)

        swapped = self._apply_knobs(base, stage_name, setting)
        runner = LocalRunner(self._registry)

        # (1) GLOBAL warmup — paid ONCE per provider, before ANY candidate of it is timed. This
        # absorbs the one-time process-global cost (model DOWNLOAD, first-ever load, torch.compile
        # cache population) that would otherwise land ONLY in the FIRST candidate's timed region —
        # the cold-baseline leak behind the pod's bogus 17720× speedup (baseline paid download +
        # cold load + compile-JIT while every LATER candidate hit warm process-global caches). Run
        # on the BASELINE knobs (the plain/fp32 path) so the download + module import are staged for
        # every candidate identically. UNTIMED, in its own run namespace.
        self._warm_globals_once(runner, provider, base)

        # (2) PER-CANDIDATE warmup — pays THIS candidate's specific (dtype/quantize/compile) path
        # JIT/first-call cost, so the compiled/quantized forward is warm before we time it. A fresh
        # provider instance is built per LocalRunner.run (the model is instance-cached, not process-
        # cached), so this warmup precedes the timed runs of THIS candidate for parity with the
        # global-warmed globals. UNTIMED.
        runner.run(swapped, run_id="tune-warmup")

        timings: list[float] = []
        for i in range(max(1, repeats)):
            start = time.perf_counter()
            runner.run(swapped, run_id=f"tune-timed-{i}")
            timings.append(time.perf_counter() - start)

        fingerprint = self._checkpoint_fingerprint(
            Storage_for(cand_root), swapped, "tune-timed-0", stage_name
        )
        return ProbeOutcome(fingerprint=fingerprint, timings=timings)

    def _warm_globals_once(self, runner: LocalRunner, provider: str, base: Pipeline) -> None:
        """Pay the one-time PROCESS-GLOBAL cost for ``provider`` once, before it is ever timed.

        The model download, the first-ever load, and the ``torch.compile`` inductor-cache
        population are process-global one-time costs. Left to the natural sweep order they would
        land entirely in the FIRST candidate's (the baseline's) timed region, while every later
        candidate hits the now-warm global caches — the exact asymmetry that produced the pod's
        bogus 17720× speedup (a 0.028 r/s cold baseline vs a 500 r/s warm winner). We run one
        UNTIMED baseline-knobs pass in its own ``tune-global-warmup`` namespace so that when the
        baseline candidate is subsequently probed, its per-candidate warmup + timed runs see the
        same warm globals every other candidate sees. Idempotent per provider (a set guard), so
        the cost is paid exactly once no matter how many candidates the grid holds.
        """
        if provider in self._globals_warmed:
            return
        self._globals_warmed.add(provider)
        # Baseline knobs = the plain path; the download + import are what we want staged, and the
        # baseline is always the first candidate so this mirrors what it will time. A fault here is
        # NOT swallowed: it surfaces to :meth:`probe`'s guard as this provider's outcome.error, so a
        # genuinely un-loadable provider is still recorded as a rejection (never a fabricated skip).
        runner.run(base, run_id="tune-global-warmup")

    @staticmethod
    def _apply_knobs(base: Pipeline, stage_name: str, setting: KnobSetting) -> Pipeline:
        """Copy ``base`` with ``setting``'s knobs shallow-merged onto the target stage."""
        new_stages = []
        for stage in base.stages:
            if stage.name == stage_name and setting.knobs:
                merged = {**stage.config, **setting.knobs}
                new_stages.append(stage.model_copy(update={"config": merged}))
            else:
                new_stages.append(stage.model_copy())
        return base.model_copy(update={"stages": new_stages})

    @staticmethod
    def _checkpoint_fingerprint(
        storage: Storage, pipeline: Pipeline, run_id: str, stage_name: str
    ) -> str:
        """A deterministic digest of the target-stage checkpoint bytes (the correctness subject).

        The stage checkpoint (``<run_dir>/checkpoints/<stage_name>.jsonl``) is the newline-delimited
        ``model_dump_json`` record set the stage emitted — labels/spans/offsets/text. A candidate
        that
        changes ANY of those (a wrong dtype that flips a label, a bad batch slice) yields a
        different
        digest and the gate rejects it (``output_changed``). An empty-but-stable stage yields the
        digest of ``b""`` (still comparable). The digest — not the raw bytes — keeps the outcome
        light
        and never carries record content into a committed artifact.
        """
        run_dir = f"{pipeline.storage_uri.rstrip('/')}/_latence/runs/{run_id}"
        uri = f"{run_dir}/checkpoints/{stage_name}.jsonl"
        raw = storage.read_bytes(uri) if storage.exists(uri) else b""
        return hashlib.sha256(raw).hexdigest()


def Storage_for(uri: str) -> Storage:  # noqa: N802 - thin lazy wrapper to keep the heavy import local
    """Lazily build a :class:`~latence_core.storage.Storage` for ``uri`` (heavy import lazy)."""
    from latence_core.storage import Storage  # noqa: PLC0415

    return Storage.for_uri(uri)
