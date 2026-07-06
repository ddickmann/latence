"""Ported endpoint lifecycle: ref-counted lease + retry-with-jitter error-classing (S12).

Exercises the pure control-plane algorithm the endpoint Providers and the second Runner lean on:
the reference-counted :class:`EndpointLease` (activate once on 0→1, deactivate once on 1→0),
the :class:`EndpointPool` co-location, and :func:`call_with_retry` classing 5xx/429/transport as
retryable and other 4xx as terminal. Deterministic: the clock/rng/sleep are injected, so nothing
sleeps and nothing depends on wall-time.
"""

from __future__ import annotations

import pytest
from latence_core.endpoint import (
    EndpointError,
    EndpointLease,
    EndpointPool,
    RetryPolicy,
    call_with_retry,
    is_retryable_status,
)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (503, True), (400, False), (401, False), (404, False), (200, False)],
)
def test_error_classing(status: int, retryable: bool) -> None:
    assert is_retryable_status(status) is retryable
    assert EndpointError("x", status=status).retryable is retryable


def test_transport_failure_is_retryable() -> None:
    # No HTTP status (a dropped connection / timeout) is treated as retryable.
    assert EndpointError("connection reset").retryable is True


def test_lease_activates_once_and_deactivates_once() -> None:
    events: list[str] = []
    lease = EndpointLease(
        name="ep",
        activate=lambda: events.append("up"),
        deactivate=lambda: events.append("down"),
    )
    lease.acquire()  # 0 -> 1: activates
    assert lease.live is True
    assert lease.refcount == 1
    lease.acquire()  # 1 -> 2: no activation
    lease.acquire()  # 2 -> 3
    assert events == ["up"]
    lease.release()  # 3 -> 2
    lease.release()  # 2 -> 1
    assert events == ["up"]  # still live
    lease.release()  # 1 -> 0: deactivates
    assert events == ["up", "down"]
    assert lease.live is False
    assert lease.refcount == 0


def test_lease_context_manager_ref_counts() -> None:
    events: list[str] = []
    lease = EndpointLease(
        name="ep", activate=lambda: events.append("up"), deactivate=lambda: events.append("down")
    )
    with lease:
        with lease:
            assert lease.refcount == 2
        assert lease.refcount == 1
        assert lease.live is True
    assert lease.refcount == 0
    assert events == ["up", "down"]


def test_lease_defaults_are_noops() -> None:
    # An always-on endpoint needs no lifecycle: the lease still ref-counts, activate/deactivate
    # default to no-ops.
    lease = EndpointLease(name="always-on")
    lease.acquire()
    assert lease.live is True
    lease.release()
    assert lease.live is False


def test_over_release_raises() -> None:
    lease = EndpointLease(name="ep")
    lease.acquire()
    lease.release()
    with pytest.raises(RuntimeError, match="released more times than acquired"):
        lease.release()


def test_activation_failure_rolls_back() -> None:
    def boom() -> None:
        raise RuntimeError("cold start failed")

    lease = EndpointLease(name="ep", activate=boom)
    with pytest.raises(RuntimeError, match="cold start failed"):
        lease.acquire()
    # The failed activation left the lease inactive so a later acquire can retry.
    assert lease.live is False
    assert lease.refcount == 0


def test_pool_shares_one_lease_by_name() -> None:
    pool = EndpointPool()
    a = pool.lease("http://vllm/v1|model-x")
    b = pool.lease("http://vllm/v1|model-x")
    c = pool.lease("http://vllm/v1|model-y")
    assert a is b  # same name -> same lease (co-located Stages share activation)
    assert a is not c


def test_retry_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}
    slept: list[float] = []

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise EndpointError("overloaded", status=503)
        return "ok"

    result = call_with_retry(
        flaky,
        RetryPolicy(attempts=4, base_delay=1.0),
        rng=lambda: 0.5,
        sleep=slept.append,
    )
    assert result == "ok"
    assert calls["n"] == 3
    # Two retries: full-jitter sleeps of 0.5 * min(1*2^0,30)=0.5 and 0.5 * min(1*2^1,30)=1.0.
    assert slept == [0.5, 1.0]


def test_retry_gives_up_and_surfaces_after_last_attempt() -> None:
    def always_503() -> str:
        raise EndpointError("still down", status=503)

    with pytest.raises(EndpointError, match="still down"):
        call_with_retry(
            always_503, RetryPolicy(attempts=3), rng=lambda: 0.0, sleep=lambda _d: None
        )


def test_terminal_error_is_not_retried() -> None:
    calls = {"n": 0}

    def bad_request() -> str:
        calls["n"] += 1
        raise EndpointError("bad request", status=400)

    with pytest.raises(EndpointError, match="bad request"):
        call_with_retry(
            bad_request, RetryPolicy(attempts=5), rng=lambda: 0.0, sleep=lambda _d: None
        )
    # A 4xx (other than 429) fails fast: exactly one call, no retries.
    assert calls["n"] == 1


def test_backoff_is_capped_by_max_delay() -> None:
    policy = RetryPolicy(base_delay=10.0, max_delay=15.0)
    # 10 * 2^2 = 40, capped to 15; rand=1.0 -> full ceiling.
    assert policy.backoff(2, 1.0) == 15.0
