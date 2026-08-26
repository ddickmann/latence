"""Endpoint lifecycle + ref-counted lease + retry-with-jitter (PORTED, S12).

Ported and generalized from the private ``pipeline-worker/endpoint_lifecycle.py`` and
``pipeline-worker/retry_utils.py`` assets: the endpoint lifecycle / ref-counted lease was
ported WITH HARDENING and generalized off RunPod, and the retry-with-jitter + error classing
was taken as an ALGORITHM ONLY. The private originals hard-wired RunPod serverless endpoint IDs, a
Supabase job table, and B2 for weights; this port keeps *only the algorithm* — reference-counted
lease so N concurrent Stages sharing one heavy inference endpoint spin it up once and tear it down
once, plus the exponential-backoff-with-jitter retry that classes 5xx/429 as retryable and 4xx as
terminal — behind a substrate-agnostic seam. It names no vendor: the ``activate``/``deactivate``
hooks are supplied by the caller (a vLLM autoscaler, a Triton model-repository load, a RunPod
serverless resume, or a no-op for an always-on OpenAI-compatible endpoint), so the same lease
drives any of them.

Nothing here imports a network client: this is the pure control-plane algorithm the endpoint
Providers (``latence-*-endpoint``) and any Runner adapter lean on. Determinism for tests is via
injected ``monotonic`` / ``sleep`` / ``rng`` callables — no wall-clock, no real sleeping, no
salted RNG — so a seeded run is byte-stable (Baseline bar).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from latence_core.errors import ProviderError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# HTTP status classes for the ported error-classing (retry_utils): a 429 (rate limit) or any 5xx
# (server/overloaded) is transient and retryable; a 4xx other than 429 is a terminal client error
# (bad request, auth, not-found) that retrying cannot fix, so it fails fast.
_RATE_LIMITED = 429
_SERVER_ERROR_FLOOR = 500


def is_retryable_status(status: int) -> bool:
    """Class an HTTP status as retryable (ported error-classing).

    ``429`` and any ``5xx`` are transient (rate-limit / server-overload) → retryable. Every other
    ``4xx`` is a terminal client error retrying cannot fix → not retryable. A ``<400`` status is a
    success and is not an error at all (returns ``False`` — nothing to retry).
    """
    if status == _RATE_LIMITED:
        return True
    return status >= _SERVER_ERROR_FLOOR


class EndpointError(ProviderError):
    """An endpoint call failed. Carries the HTTP ``status`` when the failure was an HTTP error.

    A :class:`~latence_core.errors.ProviderError` (a Provider — here a remote inference
    endpoint — failed at runtime, #73) — and, transitively, still a ``RuntimeError``, so the
    pre-taxonomy retry loop's ``except`` clauses keep catching it. ``status is None`` marks a
    transport-level failure (connection reset, timeout) with no HTTP status — treated as
    retryable, since a dropped connection to a heavy inference endpoint is the canonical
    transient fault the ported retry exists for. ``retryable`` is computed from the HTTP
    ``status`` here (429/5xx retryable, other 4xx terminal) rather than the base class's plain
    flag, so it stays the single source of truth for the retry classing.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        # Seed the base ProviderError.retryable from the status classing so the taxonomy's
        # ``retryable`` and this class's status-derived one always agree.
        super().__init__(
            message, retryable=(status is None or is_retryable_status(status))
        )
        self.status = status

    @property
    def retryable(self) -> bool:
        if self.status is None:
            return True
        return is_retryable_status(self.status)


def classify_exception(exc: Exception) -> EndpointError:
    """Normalize any client exception into a classed :class:`EndpointError` for the ported retry.

    An exception exposing an HTTP ``status_code`` (the OpenAI client's ``APIStatusError`` shape) is
    classed by that status (429/5xx retryable, other 4xx terminal); anything else — a timeout, a
    dropped connection, a client-library error with no status — is a transport error, and an
    :class:`EndpointError` with ``status is None`` is retryable by construction.

    This is the ADR-0034 native-error → typed-error boundary that EVERY OpenAI-client Provider sits
    behind (``embedder``/``ner``/``parser`` endpoints, ``parser.lighton-vllm``, ``relation.llm``,
    ``schema-inducer``). It was six byte-identical copies, one per package, differing only in how
    much of the rule each docstring restated — six independently-editable answers to the one
    question "is this failure worth retrying?". It belongs next to :class:`EndpointError`,
    :class:`RetryPolicy`, :func:`call_with_retry` and :func:`is_retryable_status`, which are what
    it feeds and which those packages already import from here.

    Deliberately duck-typed on ``status_code`` rather than on ``openai.APIStatusError``: this
    module imports no network client (ADR-0016), and the same attribute is exposed by httpx-based
    and vendor clients alike, so a Provider on a different client gets the same classing free.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return EndpointError(str(exc), status=status)
    return EndpointError(str(exc))


async def async_noop_sleep(_delay: float) -> None:
    """The async backoff sleep the endpoint Providers inject: a no-op, awaited.

    :func:`async_call_with_retry` requires an awaitable ``sleep``; the endpoint Providers pass this
    one, the async twin of the sync path's ``lambda _delay: None``. In their batch context the
    ported retry's value is the CLASSING plus the ``endpoint_retries`` tally, not wall-time
    backoff, and an instant sleep keeps a seeded test instant and byte-stable. A deployment that
    wants real pacing injects ``asyncio.sleep`` instead — the seam is the injection point, and
    nothing here forecloses it.
    """
    return None


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with full jitter (ported ``retry_utils`` algorithm).

    ``attempts`` total tries (so ``attempts - 1`` retries); it is clamped to at least 1 in
    ``__post_init__`` so a configured ``retry_attempts: 0`` means "one try, no retries" rather
    than skipping the call. The nth retry sleeps a random duration
    in ``[0, min(base * 2**n, max_delay)]`` — "full jitter", the private original's shape — which
    de-correlates a thundering herd of Stages retrying one overloaded endpoint at once. A terminal
    (non-retryable) error is re-raised immediately; the retry only ever fires on a retryable one.
    """

    attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        # Clamp ``attempts`` to at least 1 so an operator-supplied ``retry_attempts: 0`` (or a
        # negative) still runs the call exactly once with no retries, rather than skipping the loop
        # entirely and falling through to the ``assert last_error is not None`` guard in
        # :func:`call_with_retry` — which would raise a bare ``AssertionError`` (and, under ``-O``,
        # ``raise None``). Mirrors the ``max(1, ...)`` batch-size guards elsewhere: zero means "one
        # try, no retries", the natural read of "no retry attempts". Frozen dataclass → set via
        # ``object.__setattr__``.
        if self.attempts < 1:
            object.__setattr__(self, "attempts", 1)

    def backoff(self, retry_index: int, rand: float) -> float:
        """The (jittered) sleep before the ``retry_index``-th retry (0-based).

        ``rand`` is a caller-supplied uniform ``[0, 1)`` draw (injected so tests are deterministic).
        Full jitter: ``rand * min(base_delay * 2**retry_index, max_delay)``.
        """
        ceiling = min(self.base_delay * (2**retry_index), self.max_delay)
        return float(rand * ceiling)


class _Activator(Protocol):
    def __call__(self) -> None: ...


@dataclass
class EndpointLease:
    """A single heavy inference endpoint with reference-counted activation (ported lifecycle).

    The ported algorithm: the first :meth:`acquire` activates the endpoint (calls ``activate``);
    every subsequent acquire while it is live just bumps the reference count; the last
    :meth:`release` (count back to zero) deactivates it (calls ``deactivate``). So N Stages sharing
    one GLiNER-large / PaddleOCR-VL endpoint pay the cold-start once and the teardown once, never
    per-Stage — the private stack's whole reason for the lease (amortizing RunPod serverless spin-up
    across a burst of work), generalized off RunPod to any ``activate``/``deactivate`` pair.

    ``activate``/``deactivate`` default to no-ops, which is exactly right for an always-on
    OpenAI-compatible / vLLM / Triton endpoint that needs no lifecycle management — the lease still
    ref-counts, it just has nothing to spin up. Thread-safe: the count and the live flag are guarded
    by a lock, so concurrent Stages on the same lease can't double-activate or tear down under a
    live user.
    """

    name: str
    activate: _Activator | None = None
    deactivate: _Activator | None = None
    _refcount: int = field(default=0, init=False)
    _live: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def refcount(self) -> int:
        with self._lock:
            return self._refcount

    @property
    def live(self) -> bool:
        with self._lock:
            return self._live

    def acquire(self) -> None:
        """Take a reference; activate the endpoint on the 0→1 transition."""
        activate_now = False
        with self._lock:
            if self._refcount == 0 and not self._live:
                activate_now = True
                self._live = True
            self._refcount += 1
        if activate_now and self.activate is not None:
            try:
                self.activate()
            except Exception:
                # Activation failed: roll the lease back to inactive so a retry can try again,
                # rather than leaving it wedged "live" with nothing behind it.
                with self._lock:
                    self._live = False
                    self._refcount -= 1
                raise

    def release(self) -> None:
        """Drop a reference; deactivate the endpoint on the 1→0 transition."""
        deactivate_now = False
        with self._lock:
            if self._refcount == 0:
                msg = f"EndpointLease {self.name!r} released more times than acquired."
                raise RuntimeError(msg)
            self._refcount -= 1
            if self._refcount == 0 and self._live:
                deactivate_now = True
                self._live = False
        if deactivate_now and self.deactivate is not None:
            self.deactivate()

    def __enter__(self) -> EndpointLease:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class EndpointPool:
    """Registry of named :class:`EndpointLease`s so co-located Stages share one live endpoint.

    Keyed by endpoint name (typically the base URL + model): two Stages configured against the
    same heavy endpoint get the *same* lease and so share its reference count — the ported
    co-location the private worker used to run NER and RE against one vLLM server without paying two
    cold-starts. Thread-safe registration.
    """

    def __init__(self) -> None:
        self._leases: dict[str, EndpointLease] = {}
        self._lock = threading.Lock()

    def lease(
        self,
        name: str,
        *,
        activate: _Activator | None = None,
        deactivate: _Activator | None = None,
    ) -> EndpointLease:
        with self._lock:
            existing = self._leases.get(name)
            if existing is not None:
                return existing
            created = EndpointLease(name=name, activate=activate, deactivate=deactivate)
            self._leases[name] = created
            return created


def call_with_retry(
    fn: Callable[[], object],
    policy: RetryPolicy,
    *,
    rng: Callable[[], float],
    sleep: Callable[[float], None],
    on_retry: Callable[[], None] | None = None,
) -> object:
    """Invoke ``fn`` with exponential-backoff-with-jitter retry on retryable failures (ported).

    Retries only on a retryable :class:`EndpointError` (429 / 5xx / transport). A terminal error
    (4xx other than 429) is re-raised on the first failure — retrying an auth or bad-request error
    only burns the endpoint budget. After the last attempt the final error is re-raised so the
    caller (a Stage / a Runner) surfaces it rather than silently swallowing it — the S12 AC
    "endpoint failures are retried/surfaced". ``rng`` (uniform ``[0,1)``) and ``sleep`` are injected
    so the retry is deterministic and instant under test. ``on_retry`` (optional) is called once
    per retry actually taken, so a Provider can tally how many transient failures it absorbed for
    the Quality Report (S12 observability).
    """
    last_error: EndpointError | None = None
    for attempt in range(policy.attempts):
        try:
            return fn()
        except EndpointError as exc:
            last_error = exc
            if not exc.retryable:
                raise
            if attempt == policy.attempts - 1:
                raise
            if on_retry is not None:
                on_retry()
            sleep(policy.backoff(attempt, rng()))
    # Unreachable: ``policy.attempts`` is clamped to >= 1 (RetryPolicy.__post_init__), so the loop
    # always runs at least once and either returns or raises. Guards the type-checker.
    assert last_error is not None  # noqa: S101
    raise last_error


async def call_with_retry_async(
    fn: Callable[[], Awaitable[object]],
    policy: RetryPolicy,
    *,
    rng: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    on_retry: Callable[[], None] | None = None,
) -> object:
    """Async twin of :func:`call_with_retry` — awaits ``fn`` with the SAME backoff+classing (W6).

    The per-item network call in the concurrent LLM Stages (``label_inducer.llm`` / ``relation.llm``
    ) is an ``await``, so it cannot ride the sync :func:`call_with_retry`. This coroutine applies
    the identical ported algorithm — retry only on a retryable :class:`EndpointError` (429 / 5xx /
    transport), a terminal 4xx re-raised on the first failure, the final error surfaced after the
    last attempt — but ``await``s the supplied coroutine and an injected async ``sleep`` (e.g.
    :func:`asyncio.sleep`, or an instant no-op under test). ``rng`` (uniform ``[0,1)``), ``sleep``,
    and ``on_retry`` have the exact same contract as the sync version, so a Provider tallies the
    same ``endpoint_retries`` and a seeded test is byte-stable and instant. Backoff jitter is
    per-item (each concurrent item draws its own ``rng``), so a burst of items retrying one
    overloaded endpoint de-correlates — the whole reason the ported retry exists.
    """
    last_error: EndpointError | None = None
    for attempt in range(policy.attempts):
        try:
            return await fn()
        except EndpointError as exc:
            last_error = exc
            if not exc.retryable:
                raise
            if attempt == policy.attempts - 1:
                raise
            if on_retry is not None:
                on_retry()
            await sleep(policy.backoff(attempt, rng()))
    # Unreachable: ``policy.attempts`` is clamped to >= 1 (see the sync twin); guards the checker.
    assert last_error is not None  # noqa: S101
    raise last_error
