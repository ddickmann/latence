"""The tuner core — sweep + HARD CORRECTNESS GATE + fastest-eligible pick (W11, the crux).

Pure logic over the :class:`~latence_core.tune.probe.SampleProbe` seam: given a Provider and its
candidate :class:`~latence_core.tune.report.KnobSetting` grid, this probes each candidate, applies
the correctness gate, and picks the fastest ELIGIBLE one — producing a :class:`ProviderTuning`. It
is
torch-free and Provider-agnostic (the fake Probe in the tests exercises the SAME gate as the real
one), so the anti-false-green bar (pick-correct / reject-crash / reject-wrong / fingerprint-guard)
holds by construction.

The correctness gate (the whole point):

1. The BASELINE candidate (fp32/plain, compile off) is probed FIRST and defines the correctness
   reference — its output fingerprint is what every accelerated candidate must match. If the
   baseline
   itself crashes the Provider is un-tunable on this box (recorded honestly, no winner beyond it).
2. Every other candidate is REJECTED if it (a) raised ANY error — including an inference-time
   ``BackendCompilerFailed`` / dynamo error (recognised torch-free via
   :func:`~latence_core.providers.perf.is_compile_backend_failure`, named ``compile_failure`` so the
   report is precise), OR (b) produced an output fingerprint != the baseline's (``output_changed``).
3. Among the ELIGIBLE candidates (baseline always eligible) the fastest by median throughput wins.
   Ties break toward the candidate with FEWER pinned knobs (prefer the simpler setting), then by
   label — a total, deterministic order, so a seeded sweep yields one byte-identical winner.

Slow-but-correct always beats fast-but-crashing/wrong: a box that cannot compile simply never gets
``compile: true``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from latence_core.providers.perf import is_compile_backend_failure
from latence_core.tune.fingerprint import DeviceFingerprint, probe_device_fingerprint
from latence_core.tune.probe import ProbeOutcome
from latence_core.tune.report import (
    KnobSetting,
    ProviderTuning,
    RejectedKnob,
    TuneReport,
)
from latence_core.tune.sweep import is_tunable_provider, knob_grid_for_provider

if TYPE_CHECKING:
    from latence_core.tune.probe import SampleProbe

# The default figure numerator + timed-run count (N defaults to 8; K timed runs after a warmup so a
# cold-start does not skew the median). N does not size anything: the real probe times a run of
# the whole stack over the bundled validation corpus, and N is only what the median is divided
# into (see ``_throughput``).
DEFAULT_SAMPLE_SIZE = 8
DEFAULT_REPEATS = 3
DEFAULT_SEED = 0

# The implausible-speedup guard (the anti-17720× rail). A tuned perf knob (dtype/quantize/compile/
# batching) on the SAME model buys single- to low-double-digit throughput multipliers, never
# hundreds. A winner/baseline ratio above this ceiling is not a real speedup — it is a MEASUREMENT
# fault, almost always a cold-baseline leak (the baseline paid a one-time download/load/compile cost
# INSIDE its timed region while later candidates ran warm). Rather than sign a fabricated number,
# the tuner RE-MEASURES the baseline once (a fresh, now-warm timing) and, if the ratio is still
# implausible, marks the reported speedup as suspect so a signed report never prints e.g. "17720x".
IMPLAUSIBLE_SPEEDUP = 100.0

# A short cap on the exception detail folded into a (potentially repo-committed) tuned config /
# report — the same no-content discipline the bake-off keeps (a native fault against a real corpus
# must not echo input content into an artifact). Type name + a bounded message is enough to triage.
_MAX_DETAIL = 200


def _capped(exc: BaseException) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    if len(detail) > _MAX_DETAIL:
        detail = detail[: _MAX_DETAIL - 3] + "..."
    return detail


def _throughput(sample_size: int, outcome: ProbeOutcome) -> float | None:
    """``sample_size`` over the outcome's median timed-run seconds, or ``None`` (untimed).

    This is the figure the report prints. It is a rate of records only if the probe really ran
    ``sample_size`` records per timed run. The registry probe does not: it runs the whole stack
    over the bundled validation corpus whatever ``sample_size`` is, so the figure is good for
    comparing settings on one device and nothing else.

    A zero/negative median (an instant fake run) yields ``None`` rather than a divide-by-zero or an
    infinite rate — the candidate is still eligible (it ran correctly), it just carries no measured
    throughput. Real timed runs are always positive.
    """
    median = outcome.median_seconds()
    if median is None or median <= 0.0 or sample_size <= 0:
        return None
    return round(sample_size / median, 6)


def _pick_winner(
    candidates: list[tuple[KnobSetting, ProbeOutcome, float | None]],
) -> tuple[KnobSetting, float | None]:
    """The fastest ELIGIBLE candidate — ties toward fewer knobs, then label (a total order).

    ``candidates`` is the list of eligible ``(setting, outcome, throughput)`` (the baseline always
    present). Sort key: higher throughput first (``None`` throughput sorts last — an unmeasured but
    correct candidate never beats a measured one), then fewer pinned knobs, then label —
    deterministic
    so a seeded sweep yields one byte-identical winner.
    """

    def sort_key(item: tuple[KnobSetting, ProbeOutcome, float | None]) -> tuple[float, int, str]:
        setting, _outcome, throughput = item
        # Negate throughput so the highest sorts first; None → -inf-equivalent (sorts last).
        neg_tp = -(throughput if throughput is not None else float("-inf"))
        return (neg_tp, len(setting.knobs), setting.label)

    best = min(candidates, key=sort_key)
    return best[0], best[2]


class Tuner:
    """Sweeps + correctness-gates + picks the fastest-eligible knobs per Provider (W11).

    Constructed with a :class:`~latence_core.tune.probe.SampleProbe` (the real registry probe on a
    rig, a fake with a known profile in the tests) and the sample/repeats/seed.
    :meth:`tune_provider`
    produces one :class:`ProviderTuning`; :meth:`tune_stack` sweeps a set of Providers into a full
    :class:`TuneReport` fingerprinted to the current (or an injected) device.
    """

    def __init__(
        self,
        probe: SampleProbe,
        *,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        repeats: int = DEFAULT_REPEATS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._probe = probe
        self._sample_size = sample_size
        self._repeats = repeats
        self._seed = seed

    def tune_provider(self, provider: str) -> ProviderTuning:
        """Sweep ONE Provider's grid, correctness-gate, and pick the fastest eligible setting."""
        grid = knob_grid_for_provider(provider)
        if not grid:
            # Not a tunable Provider — the "winner" is the empty baseline, nothing swept.
            return ProviderTuning(provider=provider, winner=KnobSetting())

        baseline_setting = grid[0]
        baseline_outcome = self._probe.probe(
            provider, baseline_setting, sample_size=self._sample_size, repeats=self._repeats
        )
        baseline_tp = _throughput(self._sample_size, baseline_outcome)

        # If the BASELINE crashes, the Provider is un-tunable on this box: no correctness reference
        # exists, so nothing can be eligible. Record it honestly — the baseline is the winner (it is
        # the safest available setting) but it carries the crash as a rejection so the report is
        # loud.
        if baseline_outcome.crashed:
            reason, detail = self._classify_error(baseline_outcome.error)
            return ProviderTuning(
                provider=provider,
                winner=baseline_setting,
                candidate_count=len(grid),
                eligible_count=0,
                rejected=[RejectedKnob(setting=baseline_setting, reason=reason, detail=detail)],
            )

        eligible: list[tuple[KnobSetting, ProbeOutcome, float | None]] = [
            (baseline_setting, baseline_outcome, baseline_tp)
        ]
        rejected: list[RejectedKnob] = []

        for setting in grid[1:]:
            outcome = self._probe.probe(
                provider, setting, sample_size=self._sample_size, repeats=self._repeats
            )
            reject = self._gate(setting, outcome, baseline_outcome.fingerprint)
            if reject is not None:
                rejected.append(reject)
                continue
            eligible.append((setting, outcome, _throughput(self._sample_size, outcome)))

        winner, winner_tp = _pick_winner(eligible)

        # The implausible-speedup guard (the anti-17720× rail). If the winner/baseline ratio is
        # above the plausible ceiling, the baseline was almost certainly measured COLD (it paid a
        # one-time download/load/compile cost in its timed region while later candidates ran warm).
        # Re-measure the baseline ONCE — now that the process-global caches are warm — and prefer
        # that timing if it is faster (a lower cold time is the leak; keep the faster, warm one). If
        # the ratio is STILL implausible, flag it suspect so the signed report never prints a
        # fabricated multiplier.
        baseline_tp, winner_tp, speedup, suspect = self._reconcile_speedup(
            provider, baseline_setting, baseline_tp, winner, winner_tp
        )
        if speedup is None and winner == baseline_setting:
            speedup = 1.0
        return ProviderTuning(
            provider=provider,
            winner=winner,
            baseline_throughput=baseline_tp,
            winner_throughput=winner_tp,
            speedup=speedup,
            speedup_suspect=suspect,
            candidate_count=len(grid),
            eligible_count=len(eligible),
            rejected=rejected,
        )

    def _reconcile_speedup(
        self,
        provider: str,
        baseline_setting: KnobSetting,
        baseline_tp: float | None,
        winner: KnobSetting,
        winner_tp: float | None,
    ) -> tuple[float | None, float | None, float | None, bool]:
        """Compute the speedup, re-measuring a suspiciously-slow (cold) baseline once (the guard).

        Returns ``(baseline_throughput, winner_throughput, speedup, suspect)``. When the raw ratio
        exceeds :data:`IMPLAUSIBLE_SPEEDUP` the baseline is re-probed (its warmup + timed runs now
        hit warm process-global caches); the FASTER of the two baseline throughputs is kept (the
        cold one is the leak). If even the re-measured ratio is implausible the speedup is returned
        with ``suspect=True`` so the report flags rather than signs it.
        """
        speedup = self._ratio(baseline_tp, winner_tp)
        if speedup is None or speedup <= IMPLAUSIBLE_SPEEDUP:
            return (baseline_tp, winner_tp, speedup, False)

        # Re-measure the baseline warm (globals are now populated from the first sweep pass).
        remeasured = self._probe.probe(
            provider, baseline_setting, sample_size=self._sample_size, repeats=self._repeats
        )
        if not remeasured.crashed:
            warm_tp = _throughput(self._sample_size, remeasured)
            if warm_tp is not None and (baseline_tp is None or warm_tp > baseline_tp):
                baseline_tp = warm_tp
        speedup = self._ratio(baseline_tp, winner_tp)
        suspect = speedup is not None and speedup > IMPLAUSIBLE_SPEEDUP
        return (baseline_tp, winner_tp, speedup, suspect)

    @staticmethod
    def _ratio(baseline_tp: float | None, winner_tp: float | None) -> float | None:
        """``winner_tp / baseline_tp`` rounded, or ``None`` when either is missing/non-positive."""
        if baseline_tp is not None and baseline_tp > 0.0 and winner_tp is not None:
            return round(winner_tp / baseline_tp, 6)
        return None

    def _gate(
        self, setting: KnobSetting, outcome: ProbeOutcome, baseline_fingerprint: str | None
    ) -> RejectedKnob | None:
        """The correctness gate: ``None`` if the candidate is ELIGIBLE, else a rejection.

        Rejects (a) any crash — naming a compile-backend failure precisely (the W9 inference-time
        lesson) — and (b) an output whose fingerprint differs from the baseline
        (``output_changed``).
        Only a non-crashing, output-identical candidate passes.
        """
        if outcome.crashed:
            reason, detail = self._classify_error(outcome.error)
            return RejectedKnob(setting=setting, reason=reason, detail=detail)
        if outcome.fingerprint != baseline_fingerprint:
            return RejectedKnob(
                setting=setting,
                reason="output_changed",
                detail=(
                    f"output fingerprint {outcome.fingerprint!r} != baseline "
                    f"{baseline_fingerprint!r} (beyond tolerance)"
                ),
            )
        return None

    @staticmethod
    def _classify_error(error: BaseException | None) -> tuple[str, str]:
        """Classify a probe error as ``compile_failure`` (the W9 lesson) or a generic ``crash``.

        A compile-backend failure (``BackendCompilerFailed`` / a dynamo/inductor error, possibly
        WRAPPED in a typed ProviderError) is recognised torch-free via
        :func:`~latence_core.providers.perf.is_compile_backend_failure`, named ``compile_failure``
        so the report reads "compile: rejected (BackendCompilerFailed)"; anything else (an OOM, a
        real
        bug) is a ``crash``. Both are rejections — slow-but-correct beats fast-but-crashing — but
        the
        precise name is the honest report the reviewer checks for.
        """
        if error is None:  # pragma: no cover - defensive; callers only pass a real error here
            return ("crash", "")
        if is_compile_backend_failure(error):
            return ("compile_failure", _capped(error))
        return ("crash", _capped(error))

    def tune_stack(
        self,
        providers: list[str],
        *,
        stack_name: str,
        fingerprint: DeviceFingerprint | None = None,
    ) -> TuneReport:
        """Sweep every tunable Provider in ``providers`` into a fingerprinted :class:`TuneReport`.

        A non-tunable Provider (no knob family) is skipped silently — only the heavy/GPU/endpoint
        Providers are tuned. ``fingerprint`` defaults to the current device's (probed once); an
        injected fingerprint keeps a seeded test deterministic on any host.
        """
        fp = fingerprint or probe_device_fingerprint()
        tunings = [self.tune_provider(p) for p in providers if is_tunable_provider(p)]
        return TuneReport(
            stack=stack_name,
            sample_size=self._sample_size,
            seed=self._seed,
            device_fingerprint=fp,
            providers=tunings,
        )


def tune_stack(
    stack_path: str,
    *,
    probe: SampleProbe,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
    fingerprint: DeviceFingerprint | None = None,
) -> TuneReport:
    """Load a stack, collect its tunable Providers, and tune them into a :class:`TuneReport`.

    The single functional entry point the CLI drives with the REAL registry probe. Reads the stack's
    stage Providers (the placeholders are irrelevant to the roster, so the raw YAML is parsed
    directly), keeps the tunable ones IN STACK ORDER (deduped), and sweeps them via ``probe``.
    """
    from pathlib import Path  # noqa: PLC0415 - local to keep the module import light

    import yaml  # noqa: PLC0415

    data = yaml.safe_load(Path(stack_path).read_text(encoding="utf-8"))
    providers: list[str] = []
    seen: set[str] = set()
    for stage in data.get("stages", []) if isinstance(data, dict) else []:
        for name in _stage_provider_names(stage):
            if name not in seen:
                seen.add(name)
                providers.append(name)
    tuner = Tuner(probe, sample_size=sample_size, repeats=repeats, seed=seed)
    return tuner.tune_stack(providers, stack_name=Path(stack_path).name, fingerprint=fingerprint)


def _stage_provider_names(stage: object) -> list[str]:
    """Every Provider name a stage references — its own ``provider`` + any nested ``embedder``.

    A nested Export/Disambiguation ``config['embedder']['provider']`` (the ADR-0017 opt-in embedder,
    which is itself a heavy/GPU Provider) is a tune target too, so it is collected alongside the
    stage's top-level Provider.
    """
    if not isinstance(stage, dict):
        return []
    names: list[str] = []
    top = stage.get("provider")
    if isinstance(top, str):
        names.append(top)
    config = stage.get("config")
    if isinstance(config, dict):
        embedder = config.get("embedder")
        if isinstance(embedder, dict):
            nested = embedder.get("provider")
            if isinstance(nested, str):
                names.append(nested)
    return names
