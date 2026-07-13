"""The ``latence tune`` autotuner (W11-autotune, ADR-0036): the correctness-gated per-device profiler.

Anti-false-green bar (offline-buildable; the REAL device sweep is the maintainer's pod run). A fake
:class:`SampleProbe` with a KNOWN knob→(speed, correctness, crash) profile drives the SAME correctness
gate + fastest-eligible pick the real registry probe drives, so these prove:

1. **pick-correct** — the tuner picks the FASTEST setting that is correct AND non-crashing.
2. **reject-crash** — a knob whose fake forward raises ``BackendCompilerFailed`` at INFERENCE (the
   W9 lesson — not just load-time) is rejected, named ``compile_failure``.
3. **reject-wrong** — a knob that changes the output (fingerprint != baseline) is rejected.
4. **fingerprint-write / fingerprint-guard** — the tune writes a fingerprinted config, and the
runner
   overlay is NOT applied on a device-fingerprint mismatch (falls back to defaults + warns).
5. **determinism** — same device+sample+seed → byte-identical config.

No real GPU numbers are fabricated: every timing here is a fake, synthetic per-candidate delay.
Seeded/deterministic: ``LATENCE_CUDA`` + ``LATENCE_DEVICE`` are pinned so the fingerprint is fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from latence_core.pipeline import CapabilityKind, Pipeline, Stage
from latence_core.providers.perf import is_compile_backend_failure
from latence_core.tune.fingerprint import (
    LATENCE_DEVICE_ENV,
    DeviceFingerprint,
    probe_device_fingerprint,
)
from latence_core.tune.overlay import overlay_tuned_config
from latence_core.tune.probe import ProbeOutcome
from latence_core.tune.report import KnobSetting, TuneReport
from latence_core.tune.sweep import knob_grid_for_provider
from latence_core.tune.tuner import Tuner, _pick_winner

if TYPE_CHECKING:
    from collections.abc import Callable


# -- A faithful compile-backend-failure exception (the W9 lesson) ------------------------------


class BackendCompilerFailed(Exception):
    """A stand-in named EXACTLY like the torch class the pod hit (gliner2-multi-v1 + torch 2.4.1).

    The gate recognises it torch-free by class NAME (``is_compile_backend_failure``), so this
    faithful
    stand-in exercises the real detection path without importing torch — the whole point of the
    torch-free compile-fault reasoning.
    """


# -- A fake SampleProbe with a KNOWN knob→(speed, correctness, crash) profile ------------------


class FakeProbe:
    """Drives the tuner's gate with a scripted per-knob-setting outcome (no model, no torch, no GPU).

    ``script`` maps a knob-setting LABEL to a factory that builds its :class:`ProbeOutcome` — a
    (fingerprint, seconds) success, or a crash. Any label not in the script falls back to a
    correct-but-slow baseline-fingerprinted outcome, so a test only scripts the candidates it cares
    about. Records every probed label so a test can assert what was swept.
    """

    def __init__(
        self,
        baseline_fingerprint: str,
        script: dict[str, Callable[[], ProbeOutcome]] | None = None,
        *,
        default_seconds: float = 1.0,
    ) -> None:
        self._baseline_fp = baseline_fingerprint
        self._script = script or {}
        self._default_seconds = default_seconds
        self.probed: list[str] = []

    def probe(
        self, provider: str, setting: KnobSetting, *, sample_size: int, repeats: int
    ) -> ProbeOutcome:
        self.probed.append(setting.label)
        factory = self._script.get(setting.label)
        if factory is not None:
            return factory()
        # Unscripted → correct (baseline fingerprint) but slow.
        return ProbeOutcome(
            fingerprint=self._baseline_fp, timings=[self._default_seconds] * max(1, repeats)
        )


_FP = DeviceFingerprint(cuda=True, gpu_name="Fake GPU", cpu_count=8)


# -- The sweep grid ---------------------------------------------------------------------------


def test_knob_grid_baseline_is_first_and_deterministic() -> None:
    """The grid's FIRST candidate is the all-baseline setting (fp32/plain/compile-off/batch-1)."""
    grid = knob_grid_for_provider("parser.lighton")
    assert grid, "parser.lighton must be tunable"
    baseline = grid[0]
    assert baseline.knobs["dtype"] == "float32"
    assert baseline.knobs["compile"] is False
    assert baseline.knobs["ocr_batch_size"] == 1
    # Deterministic: the same provider yields the same ordered grid every call.
    assert [k.label for k in grid] == [k.label for k in knob_grid_for_provider("parser.lighton")]


def test_non_tunable_provider_has_empty_grid() -> None:
    """A CPU-deterministic in-core Provider has no perf knobs — an empty grid, not swept."""
    assert knob_grid_for_provider("entity.gazetteer") == []
    assert knob_grid_for_provider("chunk.markdown") == []


# -- (1) pick-correct: the fastest CORRECT setting wins ----------------------------------------


def test_picks_the_fastest_correct_setting() -> None:
    """The tuner picks the fastest CORRECT + non-crashing candidate (pick-correct)."""
    grid = knob_grid_for_provider("embedding.sentence_transformers")
    baseline_label = grid[0].label
    # Script: baseline slow-but-correct (2.0s); one accelerated candidate correct AND fast (0.5s).
    fast = grid[3]  # some non-baseline candidate
    probe = FakeProbe(
        "FP-CORRECT",
        {
            baseline_label: lambda: ProbeOutcome(fingerprint="FP-CORRECT", timings=[2.0, 2.0, 2.0]),
            fast.label: lambda: ProbeOutcome(fingerprint="FP-CORRECT", timings=[0.5, 0.5, 0.5]),
        },
    )
    tuning = Tuner(probe, sample_size=8).tune_provider("embedding.sentence_transformers")
    assert tuning.winner.label == fast.label
    # Its throughput beats the baseline's → a speedup > 1 (4x here: 2.0s → 0.5s).
    assert tuning.speedup is not None
    assert tuning.speedup > 1.0
    assert tuning.baseline_throughput is not None
    assert tuning.winner_throughput is not None
    assert tuning.winner_throughput > tuning.baseline_throughput


# -- (2) reject-crash: an INFERENCE-time BackendCompilerFailed is rejected ----------------------


def test_rejects_inference_time_compile_crash_named_compile_failure() -> None:
    """A knob whose forward raises BackendCompilerFailed at INFERENCE is rejected (the W9 lesson).

    The fake returns the crash as ``ProbeOutcome.error`` — exactly how the real probe surfaces an
    inference-time fault from the Runner — and the gate rejects it, naming it ``compile_failure`` so
    the honest report reads "compile: rejected (BackendCompilerFailed)". A fast candidate that
    CRASHES
    must NOT win over the slow-but-correct baseline.
    """
    grid = knob_grid_for_provider("fused_entity_relation.gliner2")
    baseline_label = grid[0].label
    # Find a compile:true candidate — the one the pod's crash lives on.
    compile_true = next(k for k in grid if k.knobs.get("compile") is True)

    probe = FakeProbe(
        "FP-BASE",
        {
            baseline_label: lambda: ProbeOutcome(fingerprint="FP-BASE", timings=[3.0, 3.0, 3.0]),
            # It would be BLAZING fast (0.01s) — but it CRASHES at inference. Must be rejected.
            compile_true.label: lambda: ProbeOutcome(
                error=BackendCompilerFailed("dynamo backend failed on the first forward")
            ),
        },
    )
    tuning = Tuner(probe, sample_size=8).tune_provider("fused_entity_relation.gliner2")

    # The crashing compile candidate is NOT the winner (slow-but-correct beats fast-but-crashing).
    assert tuning.winner.label != compile_true.label
    # It is recorded as a REJECTION, named compile_failure (not a generic crash).
    rejected = {r.setting.label: r for r in tuning.rejected}
    assert compile_true.label in rejected
    assert rejected[compile_true.label].reason == "compile_failure"
    assert "BackendCompilerFailed" in rejected[compile_true.label].detail


def test_compile_fault_detected_torch_free_through_cause_chain() -> None:
    """A BackendCompilerFailed WRAPPED in a typed error is still recognised (cause-chain walk)."""

    class ProviderError(Exception):
        pass

    try:
        try:
            raise BackendCompilerFailed("inductor lowering failed")
        except BackendCompilerFailed as cause:
            raise ProviderError("provider redaction.gliner2 op batch_extract failed") from cause
    except ProviderError as wrapped:
        assert is_compile_backend_failure(wrapped)
    # A plain error that is NOT a compile fault is not misclassified.
    assert not is_compile_backend_failure(ValueError("just a config error"))


def test_compile_fault_detected_by_module_prefix_and_message() -> None:
    """A dynamo/inductor error is detected by its module prefix OR a telltale message, torch-free."""
    # A generic-named exception DEFINED in the torch._dynamo module namespace is a compile fault.
    dynamo_exc = type("SomeError", (Exception,), {"__module__": "torch._dynamo.exc"})("boom")
    assert is_compile_backend_failure(dynamo_exc)
    # A bare RuntimeError whose MESSAGE names the backend failure is caught conservatively.
    assert is_compile_backend_failure(RuntimeError("BackendCompilerFailed: inductor lowering"))
    # A plain RuntimeError with no compile signal is not.
    assert not is_compile_backend_failure(RuntimeError("network timeout"))


# -- (3) reject-wrong: a knob that CHANGES OUTPUT is rejected -----------------------------------


def test_rejects_a_knob_that_changes_output() -> None:
    """A candidate whose output fingerprint != the baseline's is rejected (output_changed).

    Even if it is the fastest, a setting that changes labels/spans/offsets vs the fp32 baseline is
    correctness-gated out — fast-but-wrong loses to slow-but-correct.
    """
    grid = knob_grid_for_provider("embedding.sentence_transformers")
    baseline_label = grid[0].label
    wrong = grid[2]
    probe = FakeProbe(
        "FP-RIGHT",
        {
            baseline_label: lambda: ProbeOutcome(fingerprint="FP-RIGHT", timings=[2.0, 2.0, 2.0]),
            # Blazing fast (0.01s) but a DIFFERENT fingerprint → the output changed → rejected.
            wrong.label: lambda: ProbeOutcome(fingerprint="FP-WRONG", timings=[0.01, 0.01, 0.01]),
        },
    )
    tuning = Tuner(probe, sample_size=8).tune_provider("embedding.sentence_transformers")
    assert tuning.winner.label != wrong.label
    rejected = {r.setting.label: r for r in tuning.rejected}
    assert wrong.label in rejected
    assert rejected[wrong.label].reason == "output_changed"


def test_when_every_acceleration_rejected_baseline_wins_slow_but_correct() -> None:
    """If every accelerated candidate crashes/changes output, the baseline wins (slow-but-correct)."""
    grid = knob_grid_for_provider("parser.lighton")
    baseline_label = grid[0].label

    def make_outcome(label: str) -> Callable[[], ProbeOutcome]:
        def factory() -> ProbeOutcome:
            if label == baseline_label:
                return ProbeOutcome(fingerprint="OK", timings=[5.0])
            # Every non-baseline candidate crashes.
            return ProbeOutcome(error=RuntimeError("CUDA out of memory"))

        return factory

    script = {k.label: make_outcome(k.label) for k in grid}
    tuning = Tuner(FakeProbe("OK", script), sample_size=8).tune_provider("parser.lighton")
    assert tuning.winner.label == baseline_label
    assert tuning.eligible_count == 1  # only the baseline
    assert tuning.speedup == 1.0
    # Every non-baseline candidate is a recorded rejection.
    assert len(tuning.rejected) == len(grid) - 1


def test_baseline_crash_makes_provider_untunable_but_recorded() -> None:
    """A BASELINE crash → no correctness reference → un-tunable, recorded honestly (no fake win)."""
    grid = knob_grid_for_provider("parser.lighton")
    baseline_label = grid[0].label
    probe = FakeProbe(
        "unused",
        {baseline_label: lambda: ProbeOutcome(error=RuntimeError("model failed to load"))},
    )
    tuning = Tuner(probe, sample_size=8).tune_provider("parser.lighton")
    assert tuning.eligible_count == 0
    assert tuning.rejected and tuning.rejected[0].setting.label == baseline_label


# -- warmup isolation: the cold FIRST call must NOT leak into the baseline's timed throughput ----


class ColdStartFakeProbe:
    """A fake provider with a one-time PROCESS-GLOBAL cold cost, timed as the real probe times it.

    Models the pod pathology precisely. The underlying model pays a one-time cost (download + first
    load + torch.compile cache population) that is PROCESS-GLOBAL: once ANY run has paid it, every
    later run — of any candidate — is warm. But the model itself is re-loaded per run (instance-
    cached, discarded each run), so while the globals are still cold EVERY timed run pays the cold
    per-record time, not just the first. That is why a cold baseline's *median* is slow (all K runs
    cold), reproducing the pod's 0.028 r/s baseline rather than a single excludable outlier.

    ``global_warmup`` toggles the fix: when True the probe runs one UNTIMED global-warmup forward
    BEFORE any candidate is ever timed (mirroring realprobe's ``_warm_globals_once``), so the globals
    are warm by the time the baseline's timed runs execute — the baseline is timed WARM, identical to
    every later candidate. When False the globals are still cold when the baseline is timed, so ALL
    of the baseline's timed runs are cold (the un-isolated leak the guard must catch).
    """

    def __init__(
        self,
        fingerprint: str,
        *,
        cold_seconds: float = 50.0,
        warm_seconds: float = 0.02,
        global_warmup: bool = True,
    ) -> None:
        self._fp = fingerprint
        self._cold = cold_seconds
        self._warm = warm_seconds
        self._globals_warm = False  # process-global: flipped by the first forward
        self._global_warmup = global_warmup

    def _run_forwards(self, n: int) -> list[float]:
        """``n`` per-record times at the CURRENT global temperature (warm iff globals are warm).

        The model is re-loaded per run (instance-cached, discarded each run), so while the globals
        are cold EVERY forward is cold — the cold baseline's whole median is slow, not one outlier.
        """
        secs = self._warm if self._globals_warm else self._cold
        return [secs] * n

    def _mark_globals_warm(self) -> None:
        """A completed run has populated the process-global disk/compile caches — warm from now on."""
        self._globals_warm = True

    def probe(
        self, provider: str, setting: KnobSetting, *, sample_size: int, repeats: int
    ) -> ProbeOutcome:
        if self._global_warmup:
            # Global warmup (the fix): one UNTIMED run pays the one-time global cost and marks the
            # globals warm BEFORE any per-candidate timing — realprobe's ``_warm_globals_once``. So
            # the FIRST candidate (the baseline) is timed warm, identical to every later candidate.
            self._run_forwards(1)
            self._mark_globals_warm()
        # TIMED runs at the CURRENT global temperature. Without the global warmup, the first
        # candidate (the baseline) still sees cold globals here — so it times cold; the completed
        # baseline then warms the globals for every later candidate, which time warm. That is the
        # pod's cold-baseline-vs-warm-winner leak; the global warmup above defuses it.
        timings = self._run_forwards(max(1, repeats))
        self._mark_globals_warm()
        return ProbeOutcome(fingerprint=self._fp, timings=timings)


def test_warmup_excludes_cold_first_call_from_baseline_throughput() -> None:
    """The tuner's per-candidate throughput reflects the WARM steady-state, not the cold first call.

    With warmup isolation, the BASELINE (the first candidate, the one that would otherwise pay the
    one-time download/load/compile cost in its timed region) is timed WARM — so its throughput is
    on the same warm footing as every later candidate and the baseline-vs-winner ratio is SANE, not
    inflated by the cold start (the pod's 17720x artifact).
    """
    tuning = Tuner(
        ColdStartFakeProbe("FP", cold_seconds=50.0, warm_seconds=0.02, global_warmup=True),
        sample_size=8,
    ).tune_provider("fused_entity_relation.gliner2")
    # Baseline timed warm → its throughput is the warm rate (8 / 0.02 = 400 r/s), NOT the cold
    # 8/50 = 0.16 r/s. Every eligible candidate ran warm, so the ratio is ~1x (sane), never the
    # hundreds-to-thousands a leaked cold baseline would fabricate.
    assert tuning.baseline_throughput is not None
    assert tuning.baseline_throughput > 100.0  # warm, not the ~0.16 r/s cold rate
    assert tuning.speedup is not None
    assert tuning.speedup <= 2.0  # sane; all candidates warm
    assert not tuning.speedup_suspect  # a sane ratio is never flagged


def test_implausible_speedup_guard_flags_a_leaked_cold_baseline() -> None:
    """A cold-baseline leak (an implausible ratio) is re-measured warm / flagged, never signed real.

    If a cold first-call DID leak into the baseline's timed region (warmup disabled here to
    simulate the un-isolated bug), the raw winner/baseline ratio is hundreds-to-thousands. The
    tuner's guard re-measures the baseline once — now warm (the global cold cost was already paid on
    the first forward) — and keeps the faster warm timing, collapsing the ratio to sane. The report
    must never print the fabricated multiplier.
    """
    # global_warmup=False → the FIRST candidate probed (the baseline) times the cold forwards; every
    # later candidate is warm. This reproduces the leak the fix guards against.
    probe = ColdStartFakeProbe("FP", cold_seconds=50.0, warm_seconds=0.02, global_warmup=False)
    tuning = Tuner(probe, sample_size=8).tune_provider("fused_entity_relation.gliner2")
    # The guard re-measured the baseline warm and reconciled the ratio to a plausible value —
    # nothing suspect is signed, and no absurd multiplier survives into the report.
    assert tuning.speedup is not None
    assert tuning.speedup <= 100.0  # collapsed to plausible by the warm re-measure
    assert not tuning.speedup_suspect
    assert "SUSPECT" not in _render_one(tuning)
    assert "17720" not in _render_one(tuning)


def _render_one(tuning: object) -> str:
    """Render a one-provider report so a test can assert on the signed-report text."""
    report = TuneReport(
        stack="s.yaml",
        sample_size=8,
        seed=0,
        device_fingerprint=_FP,
        providers=[tuning],  # type: ignore[list-item]
    )
    return report.render()


# -- (5) determinism: same device+sample+seed → identical config -------------------------------


def test_deterministic_same_device_sample_seed_identical_config() -> None:
    """Two sweeps with the same device+sample+seed produce a byte-identical tuned config."""
    grid = knob_grid_for_provider("embedding.sentence_transformers")
    fast = grid[3]

    def make_probe() -> FakeProbe:
        return FakeProbe(
            "FP",
            {
                grid[0].label: lambda: ProbeOutcome(fingerprint="FP", timings=[2.0, 2.0]),
                fast.label: lambda: ProbeOutcome(fingerprint="FP", timings=[0.5, 0.5]),
            },
        )

    report_a = Tuner(make_probe(), sample_size=8, seed=7).tune_stack(
        ["embedding.sentence_transformers"], stack_name="s.yaml", fingerprint=_FP
    )
    report_b = Tuner(make_probe(), sample_size=8, seed=7).tune_stack(
        ["embedding.sentence_transformers"], stack_name="s.yaml", fingerprint=_FP
    )
    assert report_a.to_yaml() == report_b.to_yaml()
    # And it round-trips through YAML unchanged (the committed tuned/<device>.yaml shape).
    assert TuneReport.from_yaml_text(report_a.to_yaml()).to_yaml() == report_a.to_yaml()


def test_pick_winner_tie_breaks_toward_fewer_knobs_then_label() -> None:
    """Equal throughput ties break toward fewer pinned knobs, then label — a total, stable order."""
    a = (KnobSetting(knobs={"dtype": "bfloat16", "compile": True}), ProbeOutcome(), 100.0)
    b = (KnobSetting(knobs={"dtype": "bfloat16"}), ProbeOutcome(), 100.0)
    winner, _tp = _pick_winner([a, b])
    assert winner.label == "dtype=bfloat16"  # fewer knobs wins the tie


# -- (4) fingerprint write + guard -------------------------------------------------------------


def test_tune_report_carries_and_writes_the_device_fingerprint() -> None:
    """The tuned config is fingerprinted to the device it was measured on (the overlay match key)."""
    report = Tuner(FakeProbe("FP"), sample_size=8).tune_stack(
        ["embedding.sentence_transformers"], stack_name="gpu-sota.yaml", fingerprint=_FP
    )
    assert report.device_fingerprint.matches(_FP)
    reloaded = TuneReport.from_yaml_text(report.to_yaml())
    assert reloaded.device_fingerprint.key == _FP.key


def _tiny_pipeline(provider: str, knobs_key: str) -> Pipeline:
    return Pipeline(
        name="tiny",
        storage_uri="file:///tmp/tune-test",
        stages=[
            Stage(
                name="embed_stage",
                capability=CapabilityKind.EMBEDDING,
                provider=provider,
                config={knobs_key: "seed"},
            )
        ],
    )


def test_overlay_applied_on_fingerprint_match() -> None:
    """On a MATCHING fingerprint, the winning knobs are shallow-merged onto the stage config."""
    grid = knob_grid_for_provider("embedding.sentence_transformers")
    fast = grid[3]
    report = Tuner(
        FakeProbe(
            "FP",
            {
                grid[0].label: lambda: ProbeOutcome(fingerprint="FP", timings=[2.0]),
                fast.label: lambda: ProbeOutcome(fingerprint="FP", timings=[0.5]),
            },
        ),
        sample_size=8,
    ).tune_stack(["embedding.sentence_transformers"], stack_name="s.yaml", fingerprint=_FP)

    pipeline = _tiny_pipeline("embedding.sentence_transformers", "dimension")
    result = overlay_tuned_config(pipeline, report, device_fingerprint=_FP)
    assert result.applied
    stage = result.pipeline.stages[0]
    # The winning knobs are present AND the stage's own keys are preserved (config-only merge).
    for key, value in fast.knobs.items():
        assert stage.config[key] == value
    assert stage.config["dimension"] == "seed"  # untouched
    assert "embed_stage" in result.stages_overlaid


def test_overlay_NOT_applied_on_fingerprint_mismatch_falls_back_to_defaults() -> None:
    """The fingerprint GUARD: a tune from a DIFFERENT device is NOT applied — defaults + a warn.

    The reviewer's guard against silently applying a cross-GPU tune (where a winning ``compile:
    true``
    could crash). The pipeline comes back UNCHANGED and ``applied`` is False, with a reason naming
    both
    devices.
    """
    report = Tuner(FakeProbe("FP"), sample_size=8).tune_stack(
        ["embedding.sentence_transformers"], stack_name="s.yaml", fingerprint=_FP
    )
    other_device = DeviceFingerprint(cuda=True, gpu_name="A DIFFERENT GPU", cpu_count=64)
    assert not other_device.matches(_FP)

    pipeline = _tiny_pipeline("embedding.sentence_transformers", "dimension")
    result = overlay_tuned_config(pipeline, report, device_fingerprint=other_device)
    assert not result.applied
    assert result.pipeline == pipeline  # byte-identical: defaults, nothing overlaid
    assert result.stages_overlaid == ()
    assert "NOT applying" in result.reason


def test_fingerprint_pinned_deterministic_under_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LATENCE_DEVICE pins the whole fingerprint so a seeded sweep's key is reproducible on any host."""
    monkeypatch.setenv("LATENCE_CUDA", "1")
    monkeypatch.setenv(LATENCE_DEVICE_ENV, "Pinned Test GPU")
    a = probe_device_fingerprint()
    b = probe_device_fingerprint()
    assert a.key == b.key
    assert a.gpu_name == "Pinned Test GPU"
    assert a.cuda is True


def test_report_render_names_winners_and_rejections() -> None:
    """The honest report renders WINNERS and REJECTIONS with reasons (never fabricated)."""
    grid = knob_grid_for_provider("fused_entity_relation.gliner2")
    compile_true = next(k for k in grid if k.knobs.get("compile") is True)
    report = Tuner(
        FakeProbe(
            "OK",
            {
                grid[0].label: lambda: ProbeOutcome(fingerprint="OK", timings=[1.0]),
                compile_true.label: lambda: ProbeOutcome(
                    error=BackendCompilerFailed("boom at inference")
                ),
            },
        ),
        sample_size=8,
    ).tune_stack(["fused_entity_relation.gliner2"], stack_name="gpu-sota.yaml", fingerprint=_FP)
    rendered = report.render()
    assert "Tune report" in rendered
    assert "Rejected knobs" in rendered
    assert "compile_failure" in rendered
    assert "fused_entity_relation.gliner2" in rendered
