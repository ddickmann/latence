"""Resolving a suite's pinned Provider through the framework's own registry.

Every suite measures a **real Stage**, resolved by entry-point name from the same
``latence.providers`` registry the Runner uses and instantiated the same way
(``provider_cls(config)``). The Parse tracer slice established the two rules this module
generalises so the four #187 suites cannot quietly break them:

* **A pin is validated as a real stack fragment.** The pin is pushed through core's own
  :class:`~latence_core.pipeline.Stage` model, so a typo'd Provider name or a capability
  mismatch fails here rather than producing a mystery zero.
* **A missing Provider is named, never substituted.** Silently falling back to another Provider
  would change what the published number means while leaving the page looking identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from latence_core.capability import ProviderRegistry
from latence_core.pipeline import Stage

if TYPE_CHECKING:
    from latence_core.pipeline import CapabilityKind

    from latence_benchmark.suite import StackPin

P = TypeVar("P")


class ProviderUnavailableError(RuntimeError):
    """The pinned Provider is not installed in this environment."""


def load_provider(
    pin: StackPin,
    *,
    capability: CapabilityKind,
    protocol: type[P] | Any,
    registry: ProviderRegistry | None = None,
) -> P:
    """Resolve and instantiate ``pin``'s Provider, checked against ``protocol``.

    :param protocol: The Capability protocol the Provider must satisfy (``Parser``,
        ``EntityExtractor``, ...). A Provider registered under the right name but missing the
        Capability's method is rejected here rather than at the first call, so the failure names
        the real problem. It is typed loosely because a ``Protocol`` class is not a concrete
        ``type[P]`` to mypy; the caller pins the result's type by annotating it
        (``parser: Parser = load_provider(...)``), and the runtime ``isinstance`` below is what
        actually enforces the Capability.
    :raises ProviderUnavailableError: The pinned Provider is not installed.
    :raises TypeError: It is installed but does not satisfy the Capability.
    """
    stage = Stage(
        name=pin.stage,
        capability=capability,
        provider=pin.provider,
        config=dict(pin.config),
    )
    registry = registry if registry is not None else ProviderRegistry()
    try:
        provider_cls = registry.load(stage.provider)
    except KeyError as exc:
        msg = (
            f"The {pin.stage} benchmark is pinned to Provider {stage.provider!r}, which is not "
            "installed. Install the package that registers it and re-run; the harness never "
            "substitutes a different Provider silently, because that would change what the "
            "published number means."
        )
        raise ProviderUnavailableError(msg) from exc
    provider = provider_cls(dict(stage.config))
    if not isinstance(provider, protocol):
        msg = (
            f"Provider {stage.provider!r} does not satisfy the "
            f"{getattr(protocol, '__name__', protocol)} Capability, "
            f"so it cannot be benchmarked as a {pin.stage} Stage."
        )
        raise TypeError(msg)
    return provider
