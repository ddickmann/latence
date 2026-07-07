"""``ProviderProfile`` — a provider's declared, typed descriptor (ADR-0036 §2).

Every Provider (CONTEXT ``Provider``) declares a small, frozen ``ProviderProfile``:
what compute it needs, its approximate memory footprint, its model id + verified
SPDX license, whether it is deterministic and batch-capable, and (for endpoint/API
providers) its per-1k cost. The registry surfaces it (``ProviderRegistry.profile``),
and it drives the Phase-3 harnesses: skipping GPU providers on a CPU-only host
(:mod:`~latence_core.providers.device`, the G1 posture), device routing, and the
license/cost/compute columns of the bake-off (P3-F5) — all from **one declared
object**, so adapters and harnesses never churn.

This module is deliberately dependency-light — a Pydantic v2 contract in the same
style as :mod:`latence_core.contracts`, with **no torch/heavy deps** (ADR-0016) — so
``pip install latence-core`` stays lean and the registry can read a provider's
profile off the class object it already loads.

A provider declares its profile via a ``profile`` classmethod (preferred — a provider
can compute it from config) OR a ``PROFILE`` ClassVar; :class:`HasProfile` is the
``runtime_checkable`` protocol the registry uses to detect either structurally, and
:func:`profile_of` reads whichever a provider class declares.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The sentinel a provider uses when its license could not be verified in-slice
# (weights AND code, per ADR-0012 / the research-diligence directive). It is NEVER
# guessed — an unverifiable license is declared ``UNVERIFIED`` with
# ``license_verified=False``, so an enterprise adopter reads an honest gap rather
# than a fabricated permissive claim.
UNVERIFIED_LICENSE = "UNVERIFIED"

# The compute a provider requires: it runs on CPU only, needs a GPU, or runs on
# either (following the requested device / availability). Drives device routing and
# the skip-with-flag posture (:mod:`~latence_core.providers.device`).
Compute = Literal["cpu", "gpu", "either"]


class ProviderProfile(BaseModel):
    """A provider's declared capabilities + enterprise-readiness columns (ADR-0036 §2).

    Frozen and strict (``extra="forbid"``) — the descriptor is a stable, immutable
    fact about a provider, read by the registry, the device seam, and the bake-off.
    Fabricating one is forbidden: a provider that declares none surfaces as ``None``
    from the registry, never a synthesised default (P3-F1 scope §3).

    Fields (ADR-0036 §2):

    * ``compute`` — ``"cpu"`` | ``"gpu"`` | ``"either"``: what the adapter needs.
    * ``memory_mb`` — approximate peak resident footprint in MiB, **declared** by the
      provider (not measured here). ``>= 0``.
    * ``model_id`` — the HF/model identifier, or ``None`` for a pure-code provider
      (gazetteer, pattern) that ships no weights.
    * ``license`` — an SPDX identifier (e.g. ``"Apache-2.0"``, ``"MIT"``) or the
      sentinel :data:`UNVERIFIED_LICENSE` when it could not be verified in-slice.
    * ``license_verified`` — ``True`` only when BOTH weights and code were checked per
      ADR-0012. The :data:`UNVERIFIED_LICENSE` invariant forbids a verified unknown.
    * ``deterministic`` — whether identical input + config yields identical output
      (the Baseline/G1 determinism signal a bake-off reports).
    * ``batch`` — whether the adapter supports batched calls.
    * ``cost_per_1k`` — for an endpoint/API provider, the cost per 1k units; ``None``
      for a local/in-process provider (no per-call cost). ``>= 0`` when present.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    compute: Compute
    memory_mb: int = Field(
        ..., ge=0, description="Approx peak resident MiB (declared, not measured)."
    )
    model_id: str | None = Field(
        default=None, description="HF/model id, or None for a pure-code provider."
    )
    license: str = Field(
        ...,
        min_length=1,
        description=f"SPDX id (e.g. 'Apache-2.0') or the sentinel {UNVERIFIED_LICENSE!r}.",
    )
    license_verified: bool = Field(
        ..., description="True only when BOTH weights and code were verified (ADR-0012)."
    )
    deterministic: bool = Field(
        ..., description="Identical input + config yields identical output."
    )
    batch: bool = Field(..., description="Whether the adapter supports batched calls.")
    cost_per_1k: float | None = Field(
        default=None,
        ge=0,
        description="Per-1k cost for an endpoint/API provider; None for a local provider.",
    )

    @model_validator(mode="after")
    def _unverified_is_never_verified(self) -> ProviderProfile:
        """Enforce the UNVERIFIED invariant: an unknown license is never verified.

        A silently-verified unknown would let a fabricated permissive claim slip past
        the research-diligence directive (ADR-0012). If ``license`` is the
        :data:`UNVERIFIED_LICENSE` sentinel, ``license_verified`` MUST be ``False``.
        """
        if self.license == UNVERIFIED_LICENSE and self.license_verified:
            msg = (
                f"license={UNVERIFIED_LICENSE!r} requires license_verified=False "
                "(an unverifiable license is never silently verified — ADR-0012)."
            )
            raise ValueError(msg)
        return self


@runtime_checkable
class HasProfile(Protocol):
    """Structural protocol: a provider class that declares a :class:`ProviderProfile`.

    A provider surfaces its profile via a ``profile`` classmethod (preferred, so a
    provider can compute it from config) OR a ``PROFILE`` ClassVar. This protocol names
    the ``profile`` classmethod shape so the registry can detect a declaration
    ``runtime_checkable``-y — **without** importing the provider's heavy deps beyond the
    class object it already loaded. :func:`profile_of` is the reader that accepts either
    the classmethod or the ClassVar form.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """Return the provider's declared :class:`ProviderProfile`."""
        ...


def profile_of(provider_cls: type) -> ProviderProfile | None:
    """Read a provider class's declared profile, or ``None`` if it declares none.

    Detects either declaration form on the class object the registry already loaded,
    **never fabricating** a profile for a provider that declares none (P3-F1 §3):

    * a ``profile`` **classmethod** (preferred) — called with no args; or
    * a ``PROFILE`` **ClassVar** holding a :class:`ProviderProfile`.

    A provider with neither returns ``None``. A provider whose ``profile`` attribute is
    present but is not a callable/classmethod returning a :class:`ProviderProfile` (or a
    ``PROFILE`` that is not one) raises ``TypeError`` — a malformed declaration is a
    hard error, not a silently-dropped profile, so a typo surfaces rather than reading
    as "no profile declared".
    """
    # Prefer the classmethod form. ``getattr`` on the class returns the bound
    # classmethod; a plain function assigned as ``profile`` also works (called with no
    # args). We only treat it as a declaration when it is callable — a data attribute
    # named ``profile`` that is a ProviderProfile is handled as the ClassVar case below.
    prof_attr = getattr(provider_cls, "profile", None)
    if callable(prof_attr):
        result = prof_attr()
        if not isinstance(result, ProviderProfile):
            msg = (
                f"Provider {provider_cls.__name__!r} has a 'profile' classmethod that did "
                f"not return a ProviderProfile (got {type(result).__name__})."
            )
            raise TypeError(msg)
        return result

    class_var = getattr(provider_cls, "PROFILE", None)
    if class_var is not None:
        if not isinstance(class_var, ProviderProfile):
            msg = (
                f"Provider {provider_cls.__name__!r} has a 'PROFILE' attribute that is not "
                f"a ProviderProfile (got {type(class_var).__name__})."
            )
            raise TypeError(msg)
        return class_var

    # A ``profile`` attribute that is present but neither callable nor absent (e.g. a
    # bare ProviderProfile assigned to ``profile`` instead of ``PROFILE``) is a
    # malformed declaration — surface it rather than reading as "no profile".
    if prof_attr is not None:
        if isinstance(prof_attr, ProviderProfile):
            return prof_attr
        msg = (
            f"Provider {provider_cls.__name__!r} has a 'profile' attribute that is neither a "
            f"classmethod returning a ProviderProfile nor a ProviderProfile "
            f"(got {type(prof_attr).__name__}); declare a 'profile' classmethod or a "
            f"'PROFILE' ClassVar."
        )
        raise TypeError(msg)
    return None
