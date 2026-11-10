"""``latence tune`` — the per-device optimal-settings profiler (correctness-gated, W11-autotune).

A client's GPU decides which perf knobs win — and which even WORK (compile support is
env-dependent: gliner2-multi-v1 + torch 2.4.1 raised ``BackendCompilerFailed`` at inference on one
pod). :mod:`latence_core.tune` profiles the actual device, sweeps each heavy Provider's perf knobs
(dtype/quantize/compile/batch_size/ocr_batch_size/max_concurrency) on timed stack runs, applies
a HARD CORRECTNESS GATE (reject any setting that raises ANY error — including inference-time compile
failures via :func:`~latence_core.providers.perf.is_compile_backend_failure` — OR changes output
beyond tolerance vs the fp32/plain baseline), picks the fastest ELIGIBLE setting per Provider, and
writes a device-fingerprinted ``tuned/<device>.yaml``. The runner overlays a tuned config's
per-Provider knobs ONLY when the ``device_fingerprint`` matches (else warn + use stack defaults).

Split like the bake-off: a **pure contract + rendering half** (:mod:`latence_core.tune.report`,
:mod:`latence_core.tune.fingerprint`, :mod:`latence_core.tune.sweep`,
:mod:`latence_core.tune.overlay`
— Pydantic models + torch-free logic, no runner/stage imports, so a bare ``import latence_core``
stays
thin) and a **measuring half** (:mod:`latence_core.tune.tuner` + the real :class:`SampleProbe` in
:mod:`latence_core.tune.probe`) that runs each candidate through the registry. The correctness gate
+ the fastest-eligible pick live in the pure half and are exercised offline by fake Probes with a
known knob→(speed, correctness, crash) profile.
"""

from __future__ import annotations

from latence_core.tune.fingerprint import DeviceFingerprint, probe_device_fingerprint
from latence_core.tune.overlay import OverlayResult, overlay_tuned_config
from latence_core.tune.probe import ProbeOutcome, SampleProbe
from latence_core.tune.report import (
    KnobSetting,
    ProviderTuning,
    RejectedKnob,
    TuneReport,
)
from latence_core.tune.sweep import knob_grid_for_provider
from latence_core.tune.tuner import Tuner, tune_stack

__all__ = [
    "DeviceFingerprint",
    "KnobSetting",
    "OverlayResult",
    "ProbeOutcome",
    "ProviderTuning",
    "RejectedKnob",
    "SampleProbe",
    "TuneReport",
    "Tuner",
    "knob_grid_for_provider",
    "overlay_tuned_config",
    "probe_device_fingerprint",
    "tune_stack",
]
