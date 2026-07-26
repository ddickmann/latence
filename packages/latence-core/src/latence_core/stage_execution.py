"""StageExecution — everything it takes to run ONE Stage, behind one small interface.

Before this module a "Runner" was not a thing you could *be*. The local Runner's ``_run`` loop
held, inline, five concerns no second Runner could reach: provider load, device routing against
the declared :class:`~latence_core.providers.profile.ProviderProfile`, the skip-with-flag posture
(ADR-0036 §2), the three per-Stage tally drains (endpoint retries, schema-induction columns,
relation fan-out), and the ~15-field :class:`~latence_core.quality.StageMetrics` assembly. The
only seam a second Runner could reuse — ``LocalRunner._execute_stage`` — dispatched and drained
*one* of those tallies, so the Airflow adapter re-implemented the metrics by hand with 7 of the 22
columns and re-derived none of the device routing. Same Pipeline, two different Quality Reports
and two different failure modes on a GPU-only Provider — exactly what ADR-0003 ("the identical
Stages under a pluggable Runner") forbids.

This module is that missing depth. A Runner adapter now learns **one call**::

    outcome = stage_execution.execute(stage, ctx)
    records_out = write_them_somewhere(outcome.records)   # lazy, streaming
    metrics = outcome.metrics(records_in=n)               # every column, always

and gets device routing, skip-with-flag, typed-error categorisation and a fully-populated
``StageMetrics`` for free. It **cannot** forget a column, because it never names one.

Streaming and the drain ordering (ADR-0033)
-------------------------------------------
:attr:`StageOutcome.records` is a **lazy, one-shot** iterator: the Provider's own
``Iterable``-in/``Iterator``-out contract is threaded straight through to the caller's sink, so a
Stage's records are never materialised into a list on the way to a checkpoint. That is only sound
because the per-Stage teardown — draining the Provider's tallies, dropping the strong reference,
and handing the CUDA cache back — happens **after the stream is consumed**, inside the producing
generator's own ``finally``. Doing it eagerly (the shape ``_execute_stage``'s ``finally`` had)
would read every tally as zero and free the model mid-generation. The outcome is the natural home
for that ordering: it owns the Provider's whole lifetime, so a Runner adapter cannot get it wrong.

``metrics()`` refuses to build a StageMetrics before the stream has been drained — a Runner that
forgets to consume the records gets a loud :class:`RuntimeError`, never a silently zeroed row.

Internal composition
--------------------
The capability→Protocol dispatch itself is supplied as a :class:`StageDispatch` collaborator
rather than inlined here. It is **not** part of this module's interface — no Runner adapter names
it, constructs one, or varies it; :class:`~latence_core.runner.LocalRunner` composes itself in as
the one implementation. It is a small internal part behind a deep interface (the dispatch body is
~400 lines of capability-specific input narrowing that reaches a dozen of the engine's checkpoint
and stream helpers; relocating it is a mechanical follow-up that changes no behaviour, and is
deliberately not bundled with this behavioural change).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from latence_core.errors import ConfigError, error_category
from latence_core.pipeline import CapabilityKind
from latence_core.providers.device import DeviceDecision, select_device
from latence_core.providers.perf import empty_cuda_cache
from latence_core.providers.profile import ProviderProfile, profile_of
from latence_core.quality import StageMetrics

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from latence_core.capability import ProviderRegistry
    from latence_core.pipeline import Pipeline, Stage
    from latence_core.runner import StageOutput
    from latence_core.storage import Storage


@dataclass(frozen=True)
class StageContext:
    """Everything a Stage needs from its Runner that is not the Stage itself.

    ``incoming`` is the Stage's gathered input (empty for a corpus-level accumulator, which
    streams its input instead); ``outputs`` is the per-Stage record map the accumulator streams
    *from* — a live dict under the local Runner, the XCom store under Airflow. ``runner`` is the
    name stamped onto the Stage's metrics, which is the only place the two Runners are allowed to
    differ.
    """

    storage: Storage
    pipeline: Pipeline
    run_id: str
    incoming: list[StageOutput]
    outputs: dict[str, list[StageOutput]]
    runner: str = "local"


class StageDispatch(Protocol):
    """The capability→Protocol dispatch StageExecution drives (an internal part, not a seam)."""

    def dispatch_stage(
        self,
        stage: Stage,
        provider: object,
        storage: Storage,
        pipeline: Pipeline,
        run_id: str,
        incoming: list[StageOutput],
        outputs: dict[str, list[StageOutput]],
    ) -> Iterator[StageOutput]:
        """Narrow the Stage's input to its capability's carrier and return the Provider's stream."""
        ...  # pragma: no cover - Protocol


@dataclass
class _Tallies:
    """The per-Stage counters a Provider reports, read off it once its stream is exhausted."""

    endpoint_retries: int = 0
    induction: dict[str, int | bool] = field(default_factory=dict)


class StageOutcome:
    """One Stage's result: its records (lazily) AND the metrics row that describes producing them.

    The two halves are inseparable on purpose. ``records_out``, ``duration_seconds``, the
    endpoint-retry count and every schema-induction column are only knowable *after* the Provider's
    stream has been consumed, so the outcome measures them itself as the caller drains it. A
    Runner therefore cannot report a row that disagrees with the records it wrote.

    * :attr:`records` — the Stage's output, lazily. **One-shot**: iterate it exactly once, and to
      completion, before calling :meth:`metrics`.
    * :attr:`skipped` — True when the Stage was skipped-with-flag (a GPU-only Provider on a
      CPU-only host, ADR-0036 §2). The record stream is then empty and the Provider was never
      constructed — the Stage is *reported*, never crashed, under every Runner.
    * :meth:`metrics` — the Stage's :class:`~latence_core.quality.StageMetrics`, every column.
    """

    def __init__(
        self,
        stage: Stage,
        *,
        profile_fields: dict[str, object],
        produce: Callable[[_Tallies], Iterator[StageOutput]],
        tallies: _Tallies,
        runner: str,
        skipped: bool = False,
        from_checkpoint: bool = False,
    ) -> None:
        self._stage = stage
        self._profile_fields = profile_fields
        self._produce = produce
        self._tallies = tallies
        self._runner = runner
        self._skipped = skipped
        self._from_checkpoint = from_checkpoint
        self._records_out = 0
        self._started_at = time.perf_counter()
        self._elapsed: float | None = None
        self._taken = False
        self._drained = False

    @property
    def skipped(self) -> bool:
        """True when the Stage was skipped-with-flag rather than executed (ADR-0036 §2)."""
        return self._skipped

    @property
    def records_out(self) -> int:
        """How many records the Stage produced. Exact once :attr:`records` has been drained."""
        return self._records_out

    @property
    def records(self) -> Iterator[StageOutput]:
        """The Stage's output records, lazily. One-shot — iterate once, to completion.

        Consuming this is what actually runs the Provider. The Provider's teardown (tally drain,
        strong-reference drop, CUDA cache release) fires when the stream is exhausted or abandoned,
        never before it has produced anything.
        """
        if self._taken:
            msg = (
                f"StageOutcome.records for stage {self._stage.name!r} was already taken. "
                f"The record stream is one-shot: iterate it exactly once, to completion, then "
                f"call metrics()."
            )
            raise RuntimeError(msg)
        self._taken = True
        return self._counted()

    def _counted(self) -> Iterator[StageOutput]:
        try:
            for record in self._produce(self._tallies):
                self._records_out += 1
                yield record
        finally:
            # Stops the clock on BOTH the success and the failure path, so a failed Stage's
            # metrics report the time it actually burned rather than zero.
            self._elapsed = time.perf_counter() - self._started_at
            self._drained = True

    def metrics(
        self, *, records_in: int, error: BaseException | None = None
    ) -> StageMetrics:
        """This Stage's metrics row — every column, derived, never hand-assembled by a Runner.

        ``error`` builds the FAILED row instead: ``records_out=0``, ``error_count=1`` and the
        stable typed *category* (ADR-0034) — counts and a category only, never the exception
        message, which can quote a record.

        Raises :class:`RuntimeError` if called on a successful execution whose record stream was
        never consumed: the counters would all read zero and the row would be a quiet lie.
        """
        duration = self._elapsed if self._elapsed is not None else 0.0
        if error is not None:
            return StageMetrics(
                name=self._stage.name,
                capability=self._stage.capability.value,
                provider=self._stage.provider,
                records_in=records_in,
                records_out=0,
                duration_seconds=round(duration, 6),
                runner=self._runner,
                endpoint_retries=self._tallies.endpoint_retries,
                error_count=1,
                error_category=error_category(error),
                **self._profile_fields,  # type: ignore[arg-type]
            )
        if not self._drained and not self._skipped and not self._from_checkpoint:
            msg = (
                f"StageOutcome.metrics() for stage {self._stage.name!r} was called before its "
                f"record stream was consumed. Drain `outcome.records` first — records_out, the "
                f"duration and every Provider tally are only known once the Stage has run."
            )
            raise RuntimeError(msg)
        return StageMetrics(
            name=self._stage.name,
            capability=self._stage.capability.value,
            provider=self._stage.provider,
            records_in=records_in,
            records_out=self._records_out,
            duration_seconds=round(duration, 6),
            from_checkpoint=self._from_checkpoint,
            runner=self._runner,
            endpoint_retries=self._tallies.endpoint_retries,
            **self._tallies.induction,  # type: ignore[arg-type]
            **self._profile_fields,  # type: ignore[arg-type]
        )


class StageExecution:
    """Runs one Stage: load → route → skip-or-dispatch → stream → drain → measure.

    Accepts its dependencies (a :class:`StageDispatch` and a
    :class:`~latence_core.capability.ProviderRegistry`) and returns a :class:`StageOutcome`; it
    performs no scheduling, writes no run state and owns no substrate. Those are a Runner's job,
    and they are the *only* things that differ between the local Runner and an adopter's.

    ``fanout_by_stage`` is the live relation fan-out tally the Quality Report reads; the executor
    folds each relation Stage's counters into it as that Stage finishes, so a report built off it
    carries the fan-out columns regardless of which Runner ran the Stage.
    """

    def __init__(
        self,
        dispatch: StageDispatch,
        registry: ProviderRegistry,
        *,
        fanout_by_stage: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        self._dispatch = dispatch
        self._registry = registry
        self._fanout_by_stage = fanout_by_stage if fanout_by_stage is not None else {}

    def execute(
        self,
        stage: Stage,
        ctx: StageContext,
        *,
        replay: Iterable[StageOutput] | None = None,
    ) -> StageOutcome:
        """Execute ``stage`` under ``ctx`` and return its outcome. Never raises for the Stage.

        Every failure a Stage can produce — an unloadable Provider, a malformed
        :class:`~latence_core.providers.profile.ProviderProfile` declaration, a contract violation,
        anything the Provider itself raises — surfaces when the caller drains
        :attr:`StageOutcome.records`, not from this call. That keeps the outcome (and therefore the
        Stage's profile/device columns) available to the caller's ``except`` branch, so a FAILED
        row is as fully populated as a successful one.

        ``replay`` short-circuits execution with records the Runner already has — the local
        Runner's resume path, reading a finalized checkpoint. The Provider is never constructed,
        but the profile/device columns are still routed and stamped, which is what makes a
        device-skipped Stage stay skipped-with-flag across a resume instead of quietly reappearing
        as an ordinary Stage.
        """
        profile, deferred = self._resolve_profile(stage)
        requested_device = str(stage.config.get("device", "auto"))
        if deferred is None:
            decision = self._route_device(stage, profile)
        else:
            # A load/import error is DEFERRED, not mapped: it must re-surface as the real error
            # when the stream re-loads the Provider. A malformed profile declaration is fatal but
            # is likewise raised from the stream, so it flows through the caller's failure
            # handling. Neither is ever skipped-with-flag, which would mask it.
            decision = DeviceDecision(
                device=None, skipped=False, reason="provider load deferred to execution"
            )
        profile_fields = _profile_metric_fields(profile, requested_device, decision)
        tallies = _Tallies()

        if replay is not None:
            replayed = replay

            def _replay(_t: _Tallies) -> Iterator[StageOutput]:
                yield from replayed

            return StageOutcome(
                stage,
                profile_fields=profile_fields,
                produce=_replay,
                tallies=tallies,
                runner=ctx.runner,
                skipped=decision.skipped,
                from_checkpoint=True,
            )

        if decision.skipped:
            # Skip-with-flag (ADR-0036 §2 / G1): a GPU-only Provider on a CPU-only host is
            # REPORTED, never crashed and never given a fabricated number. Structurally identical
            # under every Runner now that it lives here rather than inline in one Runner's loop.
            def _empty(_t: _Tallies) -> Iterator[StageOutput]:
                return iter(())

            return StageOutcome(
                stage,
                profile_fields=profile_fields,
                produce=_empty,
                tallies=tallies,
                runner=ctx.runner,
                skipped=True,
            )

        def _run(t: _Tallies) -> Iterator[StageOutput]:
            if deferred is not None and not isinstance(deferred, _DeferredLoad):
                raise deferred
            provider_cls = self._registry.load(stage.provider)
            provider = provider_cls(stage.config)
            try:
                yield from self._dispatch.dispatch_stage(
                    stage,
                    provider,
                    ctx.storage,
                    ctx.pipeline,
                    ctx.run_id,
                    ctx.incoming,
                    ctx.outputs,
                )
            finally:
                # THE ordering that makes lazy Stage output sound. Every counter below is written
                # by the Provider *while* it generates; reading them before the stream is drained
                # (the shape the old eager ``finally`` had) reports zeros for all of them. Runs on
                # the failure path too, so a failed Stage still reports the retries it burned.
                t.endpoint_retries = _provider_retry_count(provider)
                t.induction = _provider_induction_tally(provider)
                if stage.capability in (
                    CapabilityKind.RELATION_EXTRACTION,
                    CapabilityKind.FUSED_ENTITY_RELATION,
                ):
                    self._fanout_by_stage[stage.name] = _provider_fanout(provider)
                # Cross-stage VRAM lifecycle (dogfood-2b-process): each Stage constructs its own
                # Provider, so once its stream is exhausted nothing downstream needs this model.
                # Under lazy iteration this now fires when the records are actually done, not
                # before the first one was produced — freeing it earlier freed the model
                # mid-generation. ``gc.collect()`` inside the seam is what really releases a
                # cycle-held nn.Module; torch-free/guarded (ADR-0016) and never raises.
                del provider
                empty_cuda_cache(provider_name=f"stage:{stage.name}")

        return StageOutcome(
            stage,
            profile_fields=profile_fields,
            produce=_run,
            tallies=tallies,
            runner=ctx.runner,
        )

    # -- profile + device routing (P3-F1/P3-F2, ADR-0036 §2) ----------------

    def _resolve_profile(
        self, stage: Stage
    ) -> tuple[ProviderProfile | None, BaseException | None]:
        """The Stage Provider's declared profile, or the deferred error that stands in for it.

        Three outcomes, deliberately kept distinct (P3-F2 §2):

        * **load/import error** (unknown provider, missing heavy dep) → a :class:`_DeferredLoad`
          sentinel: the columns route as "no profile", and execution re-loads and re-raises the
          real error. Never mapped or skipped here, which would double-handle or mask it.
        * **profile declared but MALFORMED** (``profile_of`` raises, or a ``profile()`` body does)
          → a typed :class:`~latence_core.errors.ConfigError` returned for the stream to raise. An
          author bug must fail LOUD, not be read as "no profile" and run anyway.
        * **no profile declared** → ``(None, None)``: follow the request.

        Reading the profile is a cached class import and never executes the Provider, so this is
        cheap enough to re-derive on the resume path too — which is what makes the skip flag
        STICKY across a resume.
        """
        try:
            provider_cls = self._registry.load(stage.provider)
        except Exception:  # noqa: BLE001 - re-surfaces on the execution path, deliberately
            return None, _DeferredLoad()
        try:
            return profile_of(provider_cls), None
        except Exception as exc:  # noqa: BLE001 - any profile-declaration error is an author bug
            return None, ConfigError(
                f"Provider {stage.provider!r} for stage {stage.name!r} declares a "
                f"malformed ProviderProfile (invalid declaration or profile() error): {exc}"
            )

    @staticmethod
    def _route_device(stage: Stage, profile: ProviderProfile | None) -> DeviceDecision:
        """Route the Stage's requested device against its Provider's declared profile.

        Device *placement* stays the adapter's job (ADR-0003/0036 §2) — this only routes and
        records. A non-string config value is coerced to ``str`` first so a typo surfaces as an
        "unknown device" message rather than a ``TypeError``.
        """
        return select_device(str(stage.config.get("device", "auto")), profile)


class _DeferredLoad(Exception):
    """Sentinel: the Provider failed to load; the real error re-surfaces on the execution path."""


def _profile_metric_fields(
    profile: ProviderProfile | None, requested: str, decision: DeviceDecision
) -> dict[str, object]:
    """The StageMetrics profile/device columns for a Stage (P3-F1).

    Surfaces the declared profile's ``compute``/``model_id``/``license`` (all ``None`` when the
    Provider declares no profile — never fabricated) plus the device routing: the requested
    ``device``, the ``device_selected`` the seam resolved, and the ``skipped``/``skip_reason``
    skip-with-flag markers.
    """
    return {
        "compute": profile.compute if profile is not None else None,
        "model_id": profile.model_id if profile is not None else None,
        "license": profile.license if profile is not None else None,
        "device": requested,
        "device_selected": decision.device,
        "skipped": decision.skipped,
        "skip_reason": decision.reason if decision.skipped else None,
    }


def _provider_retry_count(provider: object) -> int:
    """The endpoint-retry count a Provider exposes, or 0 if it exposes none.

    An endpoint-backed performance Provider (``latence-*-endpoint``) tallies how many transient
    endpoint failures it retried through in an ``endpoint_retries`` int attribute; an
    in-process/CPU Provider has no such attribute. Read defensively so any Provider shape is
    accepted (the seam names no base class — ADR-0004).
    """
    value = getattr(provider, "endpoint_retries", 0)
    return value if isinstance(value, int) else 0


def _provider_fanout(provider: object) -> tuple[int, int, int, int]:
    """The relation fan-out-guard tally a Provider exposes.

    Returns ``(output-capped-docs, dropped-relations, input-capped-docs, dropped-mentions)``. A
    RelationExtractor Provider enforcing the ADR-0033 / #63 per-document relation OUTPUT cap
    tallies it in ``fanout_capped_documents``/``fanout_dropped_relations``; the #68 per-document
    mention INPUT cap tallies it in ``mention_capped_documents``/``mention_dropped_mentions``. Any
    Provider without an attribute (or with a non-int) contributes ``0`` for it — read defensively
    so the seam names no base class (ADR-0004).
    """

    def _int_attr(name: str) -> int:
        value = getattr(provider, name, 0)
        return value if isinstance(value, int) else 0

    return (
        _int_attr("fanout_capped_documents"),
        _int_attr("fanout_dropped_relations"),
        _int_attr("mention_capped_documents"),
        _int_attr("mention_dropped_mentions"),
    )


def _provider_induction_tally(provider: object) -> dict[str, int | bool]:
    """The schema-induction columns a LabelInducer Provider exposes (W4 ADR-0038, T5 cost).

    Two families, read off the SAME optional-attribute seam: the W4 fail-open tally (how many
    documents fell back to config labels, and the LOUD empty-after-sanitization subset) and the T5
    cost columns (calls issued, dedup cache hits, endpoint-reported tokens, whether a call budget
    bound the run). A Provider exposing none of them contributes an EMPTY map, so absent columns
    keep their ``StageMetrics`` defaults rather than being written as zeros.
    """
    counters = (
        "induction_fallbacks",
        "induction_documents",
        "induction_empty_after_sanitization",
        "induction_calls",
        "induction_cache_hits",
        "induction_prompt_tokens",
        "induction_completion_tokens",
    )
    tally: dict[str, int | bool] = {}
    for name in counters:
        value = getattr(provider, name, None)
        # bool is an int subclass — exclude it so a mis-typed flag never lands in a count.
        if isinstance(value, int) and not isinstance(value, bool):
            tally[name] = value
    exhausted = getattr(provider, "induction_budget_exhausted", None)
    if isinstance(exhausted, bool):
        tally["induction_budget_exhausted"] = exhausted
    return tally
