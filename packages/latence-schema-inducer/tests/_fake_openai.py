"""A faithful offline mock of the ``openai`` client for the schema-inducer safety-net tests.

Returns the REAL chat-completions response shape — ``response.choices[0].message.content`` (a JSON
string) plus ``response.choices[0].finish_reason`` — so a Provider that reads either field is
exercised exactly as against the live API. Supports:

  * a **sequence of scripted replies** (``(content, finish_reason)`` tuples), consumed one per
    ``create`` call, so a multi-call safety net (truncation → doubled-budget retry, malformed →
    repair retry) can be driven deterministically;
  * a **raised exception** per call (an ``APIStatusError``-shaped object carrying ``status_code``,
    or a bare transport error) so the network/auth/terminal-4xx fail-open paths are exercised.

Installed into ``sys.modules['openai']`` for the duration of one test (the same test-double pattern
the in-tree conformance stub + the other endpoint Providers use), so nothing hits the network and no
``openai`` install is required.
"""

from __future__ import annotations

import contextlib
import sys
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

    @classmethod
    def reset(cls) -> None:
        cls.steps = []
        cls.calls = []

    @classmethod
    def next_step(cls) -> tuple[str, Any, Any]:
        if not cls.steps:
            # Exhausted script: default to a well-formed empty-shape reply so a Provider that calls
            # more times than scripted still gets a valid response (never an IndexError crash).
            return ("reply", '{"entity_types": [], "relation_types": [], "pii_types": []}', "stop")
        return cls.steps.pop(0)


class _FakeCompletions:
    def create(self, **kwargs: Any) -> Any:
        _Script.calls.append(kwargs)
        kind, a, b = _Script.next_step()
        if kind == "raise":
            raise a
        message = types.SimpleNamespace(content=a)
        choice = types.SimpleNamespace(message=message, finish_reason=b)
        return types.SimpleNamespace(choices=[choice])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class FakeOpenAI:
    """Stands in for ``openai.OpenAI``: exposes ``.chat.completions.create`` off the shared script."""

    def __init__(self, **_kwargs: Any) -> None:
        self.chat = _FakeChat()


def reply(content: str, finish_reason: str = "stop") -> tuple[str, str, str]:
    """A scripted successful reply step: the given content + finish_reason."""
    return ("reply", content, finish_reason)


def raises(exc: Exception) -> tuple[str, Exception, None]:
    """A scripted exception step: ``create`` raises ``exc`` (network/auth/terminal fault)."""
    return ("raise", exc, None)


@contextlib.contextmanager
def install(steps: list[tuple[str, Any, Any]]) -> Iterator[type[_Script]]:
    """Install the fake ``openai`` module driven by ``steps``; yields the ``_Script`` for asserts.

    Each step is built by :func:`reply` or :func:`raises`. The yielded ``_Script`` exposes
    ``calls`` (every ``create`` kwargs) so a test can assert the request (max_tokens doubling,
    ``response_format=json_object``, temperature 0, the repair instruction).
    """
    _Script.reset()
    _Script.steps = list(steps)
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI  # type: ignore[attr-defined]
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
