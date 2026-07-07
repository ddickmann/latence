"""P3-F1 — ProviderRegistry.profile() / .profiles() surfacing (ADR-0036 §2).

Deterministic + seeded (entry-point discovery + an injected fake entry point, no I/O):
a declared provider returns its real profile; a provider that declares none returns
None (never fabricated); a provider whose import RAISES is skipped-and-flagged (omitted
from the map), never crashing the sweep. And the single-name profile() path is cached.
"""

from __future__ import annotations

from importlib import metadata
from typing import Any

import pytest
from latence_core import ProviderProfile, ProviderRegistry


def test_profile_returns_declared_for_in_core_reference() -> None:
    reg = ProviderRegistry()
    prof = reg.profile("entity.gazetteer")
    assert prof is not None
    assert prof.compute == "cpu"
    assert prof.license == "Apache-2.0"
    assert prof.license_verified is True


def test_profile_is_cached() -> None:
    reg = ProviderRegistry()
    first = reg.profile("relation.pattern")
    second = reg.profile("relation.pattern")
    assert first is second  # cached instance, no re-read


def test_profile_unknown_name_raises_keyerror() -> None:
    reg = ProviderRegistry()
    with pytest.raises(KeyError, match="No Provider registered"):
        reg.profile("does.not.exist")


class _NoProfileProvider:
    """A registered provider that declares NO profile — must surface as None."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}


class _RaisingEntryPoint:
    """A fake EntryPoint whose ``load()`` raises — simulates a missing heavy dep."""

    name = "broken.provider"

    def load(self) -> type:
        msg = "simulated missing heavy dependency (torch)"
        raise ImportError(msg)


def _reg_with_injected(**providers: type) -> ProviderRegistry:
    """A registry pre-seeded so ``names()`` includes injected fakes alongside real ones."""
    reg = ProviderRegistry()
    # Force discovery of the real entry points, then splice in the fakes.
    discovered = dict(reg._discover())  # noqa: SLF001 - test-only seam access
    for name, cls in providers.items():
        # A minimal EntryPoint-like object load()-ing to the class.
        ep = metadata.EntryPoint(name=name, value="test:injected", group="latence.providers")
        object.__setattr__(ep, "load", lambda cls=cls: cls)
        discovered[name] = ep
    reg._entry_points = discovered  # noqa: SLF001
    return reg


def test_profile_none_for_provider_declaring_none() -> None:
    reg = _reg_with_injected(**{"test.no_profile": _NoProfileProvider})
    assert reg.profile("test.no_profile") is None


def test_profiles_maps_declared_and_none() -> None:
    reg = _reg_with_injected(**{"test.no_profile": _NoProfileProvider})
    profiles = reg.profiles()
    # A real declared provider has a real profile.
    assert isinstance(profiles["entity.gazetteer"], ProviderProfile)
    # The no-profile provider is present with a None value (declared absence).
    assert "test.no_profile" in profiles
    assert profiles["test.no_profile"] is None


def test_profiles_skips_import_error_provider() -> None:
    reg = ProviderRegistry()
    discovered = dict(reg._discover())  # noqa: SLF001
    discovered["broken.provider"] = _RaisingEntryPoint()  # type: ignore[assignment]
    reg._entry_points = discovered  # noqa: SLF001
    # The broken provider is in names() but its import raises...
    assert "broken.provider" in reg.names()
    profiles = reg.profiles()
    # ...so profiles() SKIPS it (omitted), never crashing the whole sweep.
    assert "broken.provider" not in profiles
    # ...and the real providers are still mapped.
    assert isinstance(profiles["entity.gazetteer"], ProviderProfile)


def test_profiles_covers_all_in_core_reference_declarations() -> None:
    # The in-core references named by the issue each surface a real profile via the map.
    reg = ProviderRegistry()
    profiles = reg.profiles()
    for name in (
        "entity.gazetteer",
        "relation.pattern",
        "chunk.markdown",
        "screening.intake_signature",
    ):
        assert isinstance(profiles[name], ProviderProfile), name
        assert profiles[name].compute == "cpu"  # type: ignore[union-attr]
