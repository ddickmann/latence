"""Lightweight tracing spans — a thin, optional observability seam (#73).

The Hardening bar (DEFINITION-OF-DONE "Observability: tracing spans …") asks for
structured per-Stage span events a downstream tracer could consume — *without* pulling a
heavy OpenTelemetry dependency into ``latence-core`` (ADR-0016: the core stays near-zero
dependency). So this is deliberately not OTel: it is a stdlib-only seam that emits a
structured event per Stage ``begin`` / ``end`` (and per checkpoint ``batch``), each stamped
with a **run-scoped correlation id** so every event of one run is greppable together and an
adopter can fan them into a real tracer (OTel, a log pipeline) at the edge if they want one.

Two design rules the tests pin:

* **Counts only, never content.** A span's ``fields`` carry the Stage name, capability,
  provider, record counts, duration, and — on a failure — the typed error *category*
  (:mod:`latence_core.errors`) and message. They **never** carry a record's text, a PII
  surface, a secret, or a document body. The same no-PII discipline the Quality Report
  keeps: a span is safe to ship to a shared tracer.
* **Determinism / opt-in.** The default sink is a :class:`LoggingSpanSink` that formats a
  span as a single structured ``logging`` record on the ``latence.trace`` logger at DEBUG —
  silent unless an operator turns that logger on, so a default run's stdout is unchanged
  (Baseline determinism). A caller who wants the events (a test, a downstream tracer) passes
  a :class:`CollectingSpanSink` or their own :class:`SpanSink`. No wall-clock leaks into the
  event *payload* (``duration_ms`` is measured by the caller via ``time.perf_counter`` and is
  the only timing field), so a seeded run's span *shape* is byte-stable even though the
  duration value itself varies.

The correlation id is ``"<run_id>"`` — the run is the trace; each Stage span is a child
keyed by ``stage``. That is enough for a downstream tracer to reconstruct the per-run tree
without this module owning a span-context stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

# Every span the framework emits carries only these safe, low-cardinality field kinds
# (str / int / float / bool). The tracer never puts a record, a PII surface, or a secret in
# a span — enforced by construction (the Runner only ever passes counts + categories) and
# asserted by the no-PII regression test.
SpanField = str | int | float | bool

_TRACE_LOGGER_NAME = "latence.trace"


@dataclass(frozen=True)
class Span:
    """One structured trace event: a Stage ``begin`` / ``end`` / ``batch`` of a run.

    ``correlation_id`` is the run-scoped id every span of a single run shares (the run *is*
    the trace). ``stage`` names the Stage; ``event`` is ``"stage_begin"``, ``"stage_end"``,
    or ``"stage_batch"``. ``fields`` are the safe, counts-only attributes — never record
    content. ``duration_ms`` is present on an ``end`` event (the Stage's wall time, the only
    timing the caller measured); ``error_category`` / ``error_message`` are present only on a
    *failed* ``end`` event (the typed :mod:`latence_core.errors` category + message).
    """

    correlation_id: str
    stage: str
    event: str
    fields: dict[str, SpanField] = field(default_factory=dict)
    duration_ms: float | None = None
    error_category: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, SpanField]:
        """A flat, JSON-able dict of the span — what a :class:`SpanSink` records/ships.

        Deterministic key order; ``None`` optionals are omitted so a ``begin`` event and a
        clean ``end`` event carry no empty error keys.
        """
        out: dict[str, SpanField] = {
            "correlation_id": self.correlation_id,
            "stage": self.stage,
            "event": self.event,
        }
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        if self.error_category is not None:
            out["error_category"] = self.error_category
        if self.error_message is not None:
            out["error_message"] = self.error_message
        for key in sorted(self.fields):
            out[key] = self.fields[key]
        return out


class SpanSink(Protocol):
    """Where spans go — the thin seam a downstream tracer plugs into (no OTel dep in core)."""

    def emit(self, span: Span) -> None: ...


class LoggingSpanSink:
    """The default sink: one structured ``logging`` record per span on ``latence.trace``.

    Emitted at DEBUG so a default run is silent (the operator opts in by turning the
    ``latence.trace`` logger on), keeping stdout unchanged for a normal run. The record's
    ``msg`` is a stable ``"trace <event> <stage>"`` and every span field is attached as
    ``extra`` so a structured-logging handler (JSON formatter, log pipeline) picks them up as
    first-class fields rather than an opaque string.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(_TRACE_LOGGER_NAME)

    def emit(self, span: Span) -> None:
        payload = span.as_dict()
        # ``correlation_id``/``stage``/``event`` are dedicated LogRecord-safe keys; the rest
        # (counts, categories) ride as extra. Guard against a field colliding with a reserved
        # LogRecord attribute name so ``logging`` never raises on ``extra``.
        extra = {k: v for k, v in payload.items() if k not in _RESERVED_LOGRECORD_KEYS}
        self._logger.debug("trace %s %s", span.event, span.stage, extra=extra)


class CollectingSpanSink:
    """An in-memory sink that keeps every span — the test + local-inspection sink.

    Deterministic: spans are appended in emission order, so a seeded run's span sequence is
    reproducible and a regression test can assert the exact begin/end pairing (and that no
    field carries content).
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def emit(self, span: Span) -> None:
        self.spans.append(span)


# LogRecord attribute names ``logging`` reserves; a span field colliding with one would make
# ``logger.debug(..., extra=...)`` raise, so :class:`LoggingSpanSink` drops those keys from
# ``extra`` (they are never span field names in practice — this is belt-and-suspenders).
_RESERVED_LOGRECORD_KEYS = frozenset(
    {"name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
     "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
     "relativeCreated", "thread", "threadName", "processName", "process", "message"}
)


class RunTracer:
    """A run-scoped span emitter: stamps every span with the run's correlation id (#73).

    Constructed once per run with the run id (the correlation id) and a :class:`SpanSink`
    (default :class:`LoggingSpanSink`). The Runner calls :meth:`stage_begin` /
    :meth:`stage_end` / :meth:`stage_batch` around each Stage; each builds a :class:`Span`
    stamped with the correlation id and hands it to the sink. Passing ``sink=None`` and the
    default logger keeps a normal run silent — the seam is inert until an operator wires a
    real sink or turns the trace logger on.
    """

    def __init__(self, correlation_id: str, sink: SpanSink | None = None) -> None:
        self.correlation_id = correlation_id
        self._sink: SpanSink = sink if sink is not None else LoggingSpanSink()

    def stage_begin(
        self, stage: str, *, capability: str, provider: str, records_in: int
    ) -> None:
        self._sink.emit(
            Span(
                correlation_id=self.correlation_id,
                stage=stage,
                event="stage_begin",
                fields={
                    "capability": capability,
                    "provider": provider,
                    "records_in": records_in,
                },
            )
        )

    def stage_batch(self, stage: str, *, batch_index: int, records_so_far: int) -> None:
        self._sink.emit(
            Span(
                correlation_id=self.correlation_id,
                stage=stage,
                event="stage_batch",
                fields={"batch": batch_index, "records": records_so_far},
            )
        )

    def stage_end(
        self,
        stage: str,
        *,
        capability: str,
        provider: str,
        records_in: int,
        records_out: int,
        duration_ms: float,
        error_category: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self._sink.emit(
            Span(
                correlation_id=self.correlation_id,
                stage=stage,
                event="stage_end",
                fields={
                    "capability": capability,
                    "provider": provider,
                    "records_in": records_in,
                    "records_out": records_out,
                },
                duration_ms=duration_ms,
                error_category=error_category,
                error_message=error_message,
            )
        )
