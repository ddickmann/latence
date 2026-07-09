"""A faithful offline mock of the ASYNC ``openai`` client for the schema-inducer safety-net tests.

Returns the REAL chat-completions response shape — ``response.choices[0].message.content`` (a JSON
string) plus ``response.choices[0].finish_reason`` — so a Provider that reads either field is
exercised exactly as against the live API. Since W6 the Provider fans its per-item calls out with
:class:`~openai.AsyncOpenAI`, so this mock exposes ``AsyncOpenAI`` and ``create`` is a **coroutine**.
Supports:

  * a **sequence of scripted replies** (``(content, finish_reason)`` tuples), consumed one per
    ``create`` call, so a multi-call safety net (truncation → doubled-budget retry, malformed →
    repair retry) can be driven deterministically;
  * a **raised exception** per call (an ``APIStatusError``-shaped object carrying ``status_code``,
    or a bare transport error) so the network/auth/terminal-4xx fail-open paths are exercised;
  * an optional **per-call delay** (``asyncio.sleep``) so a timing test can assert the wall-clock of
    N concurrent items is ``ceil(N / concurrency) × delay`` — proving the fan-out is real — plus a
    **shuffled completion order** (fast items finishing before slow ones) so a determinism test
    proves the index-keyed reassembly is completion-order-independent.

Installed into ``sys.modules['openai']`` for the duration of one test (the same test-double pattern
the in-tree conformance stub + the other endpoint Providers use), so nothing hits the network and no
``openai`` install is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any


class FakeAPIStatusError(Exception):
    """Stands in for the OpenAI client's ``APIStatusError``: carries an HTTP ``status_code``."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _Script:
    """A shared, mutable script of replies/exceptions consumed across the fake client's calls."""

    #: Each item is either ("reply", content, finish_reason) or ("raise", exception).
    steps: list[tuple[str, Any, Any]] = []
    #: Every ``create`` call's kwargs, recorded so a test can assert max_tokens/response_format.
    calls: list[dict[str, Any]] = []
    #: Optional per-call delay (seconds) awaited inside ``create`` — drives the concurrency-timing
    #: and shuffled-completion determinism tests. 0 ⇒ no delay.
    delay: float = 0.0
    #: Live count of ``create`` coroutines currently between their pop and their return (i.e. sleeping
    #: on the injected delay) and the high-water mark of that count. The high-water mark is the
    #: BOUND check: it must be ``<= max_concurrency`` (the semaphore holds) yet ``> 1`` (the fan-out
    #: is genuinely concurrent). Without this an unbounded run would still pass the timing test.
    in_flight: int = 0
    max_in_flight: int = 0
    #: Guards ``steps``/``calls`` mutation across concurrently-awaited ``create`` coroutines. asyncio
    #: is single-threaded so this is belt-and-suspenders, but it keeps the pop/append atomic.
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        cls.steps = []
        cls.calls = []
        cls.delay = 0.0
        cls.in_flight = 0
        cls.max_in_flight = 0

    @classmethod
    def next_step(cls) -> tuple[str, Any, Any]:
        with cls._lock:
            if not cls.steps:
                # Exhausted script: default to a well-formed empty-shape reply so a Provider that
                # calls more times than scripted still gets a valid response (never an IndexError).
                return (
                    "reply",
                    '{"entity_types": [], "relation_types": [], "pii_types": []}',
                    "stop",
                )
            return cls.steps.pop(0)


class _FakeCompletions:
    async def create(self, **kwargs: Any) -> Any:
        with _Script._lock:  # noqa: SLF001 - the fake owns the shared script
            _Script.calls.append(kwargs)
        kind, a, b = _Script.next_step()
        # The step is popped in dispatch (input) order BEFORE any sleep, so a reply is bound to a job
        # by its dispatch position; the sleep only shifts COMPLETION order. A per-step delay (packed
        # into the finish_reason slot for a ``reply_after`` step) lets a test reverse completion order
        # while keeping the reply→job binding stable — the determinism-under-shuffle proof.
        per_step_delay = 0.0
        if kind == "reply_after":
            b, per_step_delay = b
            kind = "reply"
        delay = per_step_delay or _Script.delay
        # Track how many coroutines are concurrently between here and their return. The high-water
        # mark is the semaphore-bound check: it must never exceed max_concurrency, yet exceed 1 to
        # prove the fan-out is genuine. asyncio is single-threaded, so a plain read-modify-write
        # around the ``await`` is safe (no interleaving except at the await point itself).
        _Script.in_flight += 1
        _Script.max_in_flight = max(_Script.max_in_flight, _Script.in_flight)
        try:
            if delay:
                await asyncio.sleep(delay)
            if kind == "raise":
                raise a
            message = types.SimpleNamespace(content=a)
            choice = types.SimpleNamespace(message=message, finish_reason=b)
            return types.SimpleNamespace(choices=[choice])
        finally:
            _Script.in_flight -= 1


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class FakeAsyncOpenAI:
    """Stands in for ``openai.AsyncOpenAI``: ``.chat.completions.create`` is a coroutine (W6)."""

    def __init__(self, **_kwargs: Any) -> None:
        self.chat = _FakeChat()


def reply(content: str, finish_reason: str = "stop") -> tuple[str, str, str]:
    """A scripted successful reply step: the given content + finish_reason."""
    return ("reply", content, finish_reason)


def reply_after(content: str, delay: float, finish_reason: str = "stop") -> tuple[str, str, Any]:
    """A scripted reply that completes only after ``delay`` seconds — reverses completion order.

    The reply is still BOUND to its dispatch (input) position; only its completion is delayed, so a
    determinism test can make later jobs finish first and assert the output is still input-ordered.
    """
    return ("reply_after", content, (finish_reason, delay))


def raises(exc: Exception) -> tuple[str, Exception, None]:
    """A scripted exception step: ``create`` raises ``exc`` (network/auth/terminal fault)."""
    return ("raise", exc, None)


@contextlib.contextmanager
def install(
    steps: list[tuple[str, Any, Any]], *, delay: float = 0.0
) -> Iterator[type[_Script]]:
    """Install the fake ``openai`` module driven by ``steps``; yields the ``_Script`` for asserts.

    Each step is built by :func:`reply` or :func:`raises`. ``delay`` (seconds) makes every
    ``create`` await ``asyncio.sleep(delay)`` so a timing test can measure the concurrency. The
    yielded ``_Script`` exposes ``calls`` (every ``create`` kwargs) so a test can assert the request
    (max_tokens doubling, ``response_format=json_object``, temperature 0, the repair instruction).
    """
    _Script.reset()
    _Script.steps = list(steps)
    _Script.delay = delay
    module = types.ModuleType("openai")
    module.AsyncOpenAI = FakeAsyncOpenAI  # type: ignore[attr-defined]
    saved = sys.modules.get("openai")
    sys.modules["openai"] = module
    try:
        yield _Script
    finally:
        # NOTE: do NOT reset ``calls`` here — a test asserts ``script.calls`` AFTER the ``with``
        # block exits (the induce() run happens inside the block, the asserts read it after). Only
        # uninstall the fake module. ``reset()`` at the next ``install`` clears it for the next test.
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)
