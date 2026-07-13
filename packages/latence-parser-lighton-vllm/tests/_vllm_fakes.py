"""Shared test-double state for the served-vLLM parser tests (imported as ``_vllm_fakes``).

Kept OUT of ``conftest.py`` (and imported by it) so the tests import these symbols via
``from _vllm_fakes import ...`` rather than ``from conftest import ...`` — the repo convention
(cf. ``latence-schema-inducer``'s ``_fake_openai``). Two ``conftest`` modules with the same basename
can collide under a whole-tree ``pytest`` run; a uniquely-named module never does.

:class:`VllmScript` is the per-page scripted-reply + timing + concurrency accounting the faithful
async vLLM mock reads (see ``conftest._FakeCompletions``). :class:`StatusError` stands in for the
OpenAI client's ``APIStatusError`` (an HTTP ``status_code``-bearing exception) so a test can force a
retryable/terminal endpoint failure on a specific page.
"""

from __future__ import annotations

from typing import Any

# The page-image sentinel the fake rasteriser emits and the fake vLLM "reads". A str carrying the
# page's intended markdown, so the fake decode is deterministic + inspectable (never a real bitmap).
FAKE_PAGE_MARKER = "FAKE_PAGE::"


class VllmScript:
    """Per-page scripted overrides + timing + concurrency accounting for the fake vLLM endpoint.

    By default the fake vLLM decodes the page image the Provider POSTed back to that page's markdown
    (the faithful round-trip). ``overrides`` lets a test force a specific reply/exception FOR A
    SPECIFIC PAGE MARKER (keyed by the page's markdown), e.g. to raise a terminal 4xx on one page
    (per-page fail-open) or a retryable-then-success sequence. ``delay_for`` sets a per-page
    ``asyncio.sleep`` (so a test can reverse completion order / prove concurrency). ``max_in_flight``
    records the peak number of concurrent ``create`` coroutines (the semaphore bound).
    """

    # marker markdown -> list of steps; each step is ("reply", str) or ("raise", Exception).
    overrides: dict[str, list[tuple[str, Any]]] = {}
    # marker markdown -> per-call delay seconds (default 0).
    delay_for: dict[str, float] = {}
    default_delay: float = 0.0
    calls: int = 0
    in_flight: int = 0
    max_in_flight: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.overrides = {}
        cls.delay_for = {}
        cls.default_delay = 0.0
        cls.calls = 0
        cls.in_flight = 0
        cls.max_in_flight = 0


class StatusError(Exception):
    """Stand-in for the OpenAI client's ``APIStatusError`` (carries an HTTP ``status_code``)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
