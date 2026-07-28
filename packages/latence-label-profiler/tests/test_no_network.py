"""Acceptance 4: ZERO network calls — proved by making a network call impossible, not asserted.

The standing rule in this repo is that a fake must REJECT what the real dependency rejects. Here
the "fake" is a socket layer that raises on any attempt to open one, plus a poisoned
:mod:`urllib.request`. If any part of the profiler ever reached for the network — a model download,
a stopword-list fetch, a telemetry ping — these tests turn red rather than quietly working.
"""

from __future__ import annotations

import socket
import sys
import urllib.request  # imported HERE: importing it under the poisoned socket pulls ssl and dies
from typing import TYPE_CHECKING, Any, NoReturn

import pytest
from latence_label_profiler import CorpusProfiler, ProfilerConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

_CORPUS = [
    "The commodity is classified under the tariff heading of the customs schedule.",
    "Each commodity code maps to a tariff heading in the combined nomenclature.",
    "The customs declaration references the tariff heading and the commodity code.",
    "Contact compliance@example.org about the export licence for the dual use item.",
] * 4


class _NetworkAttempted(AssertionError):
    """Raised instead of opening a socket, so a network call is a FAILING test, not a slow one."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def _boom(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise _NetworkAttempted("latence-label-profiler attempted a network call")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)
    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    monkeypatch.setattr(socket, "gethostbyname", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    yield


def test_the_guard_itself_bites(no_network: None) -> None:
    """Bite-test: without this, a profiler that never ran would 'pass' the no-network test."""
    with pytest.raises(_NetworkAttempted):
        socket.create_connection(("example.com", 80))


def test_a_full_profile_runs_with_the_network_removed(no_network: None) -> None:
    profile = CorpusProfiler(ProfilerConfig(seed=0, min_mention_mass=1)).profile(_CORPUS)
    assert profile.lexicon_candidates()
    assert profile.to_json()


def test_the_package_imports_no_network_module() -> None:
    """A static check to back the dynamic one: no http/urllib/requests anywhere in the package."""
    import latence_label_profiler  # noqa: F401

    forbidden = ("requests", "httpx", "urllib3", "aiohttp", "http.client")
    loaded = {
        name
        for name, module in sys.modules.items()
        if getattr(module, "__name__", "").startswith("latence_label_profiler")
    }
    assert loaded, "the package did not import"
    for module_name in forbidden:
        assert module_name not in sys.modules or not any(
            module_name in getattr(sys.modules[name], "__dict__", {}) for name in loaded
        )


def test_the_profile_declares_zero_cost_and_zero_calls() -> None:
    """The cost seam an operator compares against an LLM-induction run."""
    profile = CorpusProfiler(ProfilerConfig(seed=0, min_mention_mass=1)).profile(_CORPUS)
    assert profile.method.network_calls == 0
    assert profile.method.model_id is None
    assert profile.method.cost_usd == 0.0
