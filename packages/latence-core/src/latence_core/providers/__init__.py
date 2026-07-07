"""Provider ecosystem seam: the declared :class:`ProviderProfile` + device routing.

Phase-3 Wave-0 foundation (ADR-0036 §1/§2). This subpackage is the small, declared
contract every Wave-1 provider adapter is written against — dependency-light, with
**no torch/CUDA/heavy deps** (ADR-0016). It carries two things:

* :mod:`~latence_core.providers.profile` — the :class:`ProviderProfile` descriptor a
  provider declares (compute/memory/model/license/determinism/batch/cost) plus the
  structural :class:`HasProfile` protocol the registry uses to detect a declaration
  without importing a provider's heavy deps beyond the class object it already loads.
* :mod:`~latence_core.providers.device` — the dep-light device seam: a torch-free
  :func:`~latence_core.providers.device.cuda_available` probe and
  :func:`~latence_core.providers.device.select_device`, which routes a requested
  device against a profile and returns a skip-with-flag :class:`DeviceDecision` for a
  GPU-only provider on a CPU-only host (the G1 posture — flag, never fail).
"""

from __future__ import annotations

from latence_core.providers.device import (
    LATENCE_CUDA_ENV,
    DeviceDecision,
    cuda_available,
    select_device,
)
from latence_core.providers.profile import HasProfile, ProviderProfile, profile_of

__all__ = [
    "LATENCE_CUDA_ENV",
    "DeviceDecision",
    "HasProfile",
    "ProviderProfile",
    "cuda_available",
    "profile_of",
    "select_device",
]
