"""Dep-light device routing seam (ADR-0036 §1/§2, ADR-0016).

Device *placement* stays the adapter's job (it moves its own model to the resolved
device, ADR-0003/0036 §2); this seam does NOT move models. It answers two questions,
dependency-light and **without importing torch** (torch is not a core dep, ADR-0016):

* :func:`cuda_available` — is a CUDA GPU present? Deterministic under an explicit
  ``LATENCE_CUDA=0|1`` env override (for CPU-only CI and seeded tests); otherwise a
  best-effort probe that defaults to ``False`` when unknown.
* :func:`select_device` — given a requested device (``auto|cpu|cuda``) and a provider's
  :class:`~latence_core.providers.profile.ProviderProfile`, return a
  :class:`DeviceDecision`: the effective device (``"cpu"``/``"cuda"``), or a
  **skip-with-flag** decision (``skipped=True`` + a reason) when a GPU-only provider is
  requested on a CPU-only host. It **never raises** for that case — the G1 posture is
  flag-and-skip, so the stack harness records the skip in its report rather than
  crashing (no fabricated GPU numbers).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from latence_core.providers.profile import ProviderProfile

# The env override that pins CUDA availability deterministically. ``LATENCE_CUDA=1``
# forces available, ``LATENCE_CUDA=0`` forces unavailable — the CI/test knob that makes
# a CPU-only host reproducible and lets a test exercise the cuda path without a GPU.
LATENCE_CUDA_ENV = "LATENCE_CUDA"

# The device a config value may request. ``auto`` follows availability; ``cpu`` pins
# CPU; ``cuda`` asks for the GPU (skip-with-flag on a CPU-only host).
RequestedDevice = Literal["auto", "cpu", "cuda"]

# The valid requested-device strings, for validating a raw config value.
_VALID_REQUESTED: frozenset[str] = frozenset({"auto", "cpu", "cuda"})

# Truthy/falsy spellings the env override accepts (case-insensitive). Anything else is
# ignored (falls through to the best-effort probe) rather than guessed.
_ENV_TRUE: frozenset[str] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE: frozenset[str] = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class DeviceDecision:
    """The typed outcome of :func:`select_device` — a routing result, not a bare string.

    So a harness can record a skip in its report (never a fabricated GPU number):

    * ``device`` — the effective device (``"cpu"`` or ``"cuda"``) to place the model on,
      or ``None`` when ``skipped`` (there is no device to place on).
    * ``skipped`` — ``True`` when a GPU-only provider was requested on a CPU-only host:
      the provider is skipped-with-flag (the G1 posture), not run and not crashed.
    * ``reason`` — a human-readable explanation of the routing (why this device, or why
      the skip), for the bake-off / stack-validation report.
    """

    device: str | None
    skipped: bool
    reason: str


def _env_override() -> bool | None:
    """Read the ``LATENCE_CUDA`` override: ``True``/``False`` if set + recognised, else ``None``.

    An unset or unrecognised value returns ``None`` so the caller falls through to the
    best-effort probe — the override is honored first (determinism for CI/tests), but a
    garbage value is never silently treated as either truthy or falsy.
    """
    raw = os.environ.get(LATENCE_CUDA_ENV)
    if raw is None:
        return None
    lowered = raw.strip().lower()
    if lowered in _ENV_TRUE:
        return True
    if lowered in _ENV_FALSE:
        return False
    return None


def cuda_available() -> bool:
    """Detect a CUDA GPU **without importing torch** (ADR-0016), override-driven for CI.

    Probe order (deterministic under the override):

    1. Honor an explicit ``LATENCE_CUDA=0|1`` env override first — the CI/test knob that
       makes a CPU-only host reproducible and lets a test drive the cuda path.
    2. Otherwise, a best-effort torch-free probe: the presence of the ``nvidia-smi``
       tool on ``PATH`` OR an ``/dev/nvidia*`` device file. Either is a strong signal a
       CUDA device exists; neither imports torch or a CUDA library.
    3. Default to ``False`` when unknown — a CPU-only host is the safe assumption, so a
       GPU provider is skipped-with-flag rather than attempted and crashed.

    Under the override this is fully deterministic (no filesystem/PATH dependence), which
    is what the seeded conformance/stack tests rely on.
    """
    override = _env_override()
    if override is not None:
        return override

    # Best-effort, torch-free. ``nvidia-smi`` on PATH is the usual driver-present signal.
    if shutil.which("nvidia-smi") is not None:
        return True
    # A CUDA device node is a second signal on a Linux host without the CLI on PATH.
    try:
        for entry in os.listdir("/dev"):
            if entry.startswith("nvidia") and entry != "nvidiactl":
                # nvidia0, nvidia1, ... are per-GPU device nodes. ``nvidiactl`` exists
                # even without a usable GPU on some setups, so it is not counted alone.
                return True
    except OSError:
        # No /dev, or not readable (non-Linux, sandbox) — unknown, default False below.
        pass
    return False


def _normalize_requested(requested: str) -> RequestedDevice:
    """Validate + normalise a raw ``device`` config value to ``auto|cpu|cuda``.

    A missing/empty value is treated as ``auto`` (the natural default — follow
    availability). An unrecognised string is a config error surfaced to the caller as a
    ``ValueError`` (device routing is a declared config knob, not a free-form string).
    """
    if not requested:
        return "auto"
    lowered = requested.strip().lower()
    if lowered not in _VALID_REQUESTED:
        allowed = ", ".join(sorted(_VALID_REQUESTED))
        msg = f"Unknown device request {requested!r}; expected one of: {allowed}."
        raise ValueError(msg)
    # Narrowed to the Literal by the membership check above.
    return lowered  # type: ignore[return-value]


def select_device(requested: str, profile: ProviderProfile | None) -> DeviceDecision:
    """Route a requested device against a provider's profile (skip-with-flag, never raise).

    ``requested`` is the ``device`` config value (``auto|cpu|cuda``); ``profile`` is the
    provider's declared :class:`~latence_core.providers.profile.ProviderProfile` (or
    ``None`` when it declares none). The G1 posture is **skip-with-flag, never raise** for
    the GPU-on-CPU-only case, so the stack harness records the skip.

    Routing (per ADR-0036 §2):

    * ``profile.compute == "cpu"`` → always ``"cpu"``. Even ``requested="cuda"`` resolves
      to CPU (with a reason noting the override) — a CPU-only provider has no GPU path, so
      this is not a skip, just a clamp.
    * ``profile.compute == "gpu"`` → needs a GPU. If CUDA is available, resolve to the
      requested device (``cpu`` is still honored if explicitly asked, else ``cuda``); if
      NOT available, **skip-with-flag** — ``requested="auto"`` or ``"cuda"`` both skip,
      ``requested="cpu"`` clamps to CPU only if the provider could run there, which a
      ``compute=="gpu"`` provider cannot, so it also skips.
    * ``profile.compute == "either"`` (or ``profile is None``) → follow the request:
      ``cpu`` → CPU; ``cuda`` → CUDA if available else **skip-with-flag**; ``auto`` →
      CUDA when available, else CPU.

    Never raises for the skip case; a genuinely malformed ``requested`` string is still a
    ``ValueError`` (config error), surfaced before any routing.
    """
    req = _normalize_requested(requested)
    has_cuda = cuda_available()
    compute = profile.compute if profile is not None else "either"

    if compute == "cpu":
        # A CPU-only provider always resolves to CPU; an explicit cuda request is a clamp,
        # not a skip (the provider simply has no GPU path).
        if req == "cuda":
            return DeviceDecision(
                device="cpu",
                skipped=False,
                reason="provider is cpu-only; requested cuda clamped to cpu",
            )
        return DeviceDecision(device="cpu", skipped=False, reason="provider is cpu-only")

    if compute == "gpu":
        if not has_cuda:
            # The G1 posture: a GPU-only provider on a CPU-only host is skipped-with-flag,
            # never run and never crashed — the harness records the skip (no fake numbers).
            return DeviceDecision(
                device=None,
                skipped=True,
                reason="provider requires a GPU but no CUDA device is available (skipped)",
            )
        if req == "cpu":
            # CUDA is present but the caller explicitly pinned CPU for a GPU-only provider:
            # it cannot run on CPU, so skip-with-flag rather than silently mis-placing it.
            return DeviceDecision(
                device=None,
                skipped=True,
                reason="provider requires a GPU but cpu was explicitly requested (skipped)",
            )
        return DeviceDecision(
            device="cuda", skipped=False, reason="provider requires a GPU; cuda available"
        )

    # compute == "either" (or no profile): follow the request against availability.
    if req == "cpu":
        return DeviceDecision(device="cpu", skipped=False, reason="cpu requested")
    if req == "cuda":
        if has_cuda:
            return DeviceDecision(
                device="cuda", skipped=False, reason="cuda requested; cuda available"
            )
        return DeviceDecision(
            device=None,
            skipped=True,
            reason="cuda requested but no CUDA device is available (skipped)",
        )
    # auto: prefer CUDA when present, else CPU.
    if has_cuda:
        return DeviceDecision(
            device="cuda", skipped=False, reason="auto resolved to cuda (available)"
        )
    return DeviceDecision(device="cpu", skipped=False, reason="auto resolved to cpu (no cuda)")
