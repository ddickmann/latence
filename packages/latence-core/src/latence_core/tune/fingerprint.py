"""The device fingerprint — what makes a tuned config reusable on THIS box and no other (W11).

A tuned config is only valid for the device it was measured on: a knob that wins on a Blackwell can
crash or regress on an Ada, and a ``compile: true`` that a driver supports on one pod raises
``BackendCompilerFailed`` on another. So every ``tuned/<device>.yaml`` records a
:class:`DeviceFingerprint`, and the runner overlay (:mod:`latence_core.tune.overlay`) applies the
tune ONLY when the current device's fingerprint MATCHES — else it warns and falls back to stack
defaults (never silently applies a mismatched tune).

Dep-light + torch-free-to-import (ADR-0016): the fingerprint is gathered **without importing torch**
where possible (``nvidia-smi`` for the GPU name/VRAM, ``os`` for CPU count), reusing the same
override-driven CUDA probe the device seam uses
(:func:`~latence_core.providers.device.cuda_available`)
so a seeded CPU-only test is fully deterministic. A ``torch``/``CUDA`` version is read ONLY if torch
is already importable, and its absence is an honest ``None`` field, never a crash.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # noqa: S404 - used with a fixed argv list + timeout, no shell (see _nvidia_smi)

from pydantic import BaseModel, ConfigDict, Field

from latence_core.providers.device import LATENCE_CUDA_ENV, cuda_available

# The fingerprint schema version — bumped if the fields that identify a device change, so a
# persisted
# tuned config carries the shape (and match semantics) it was written under.
FINGERPRINT_SCHEMA_VERSION = 1

# The env override a seeded test uses to pin the ENTIRE fingerprint deterministically (so a
# determinism test does not depend on the host's real GPU name/VRAM/cpu count). When set, its value
# IS the ``gpu_name`` and the rest of the probed fields are held at fixed, override-derived values —
# the same "override first, real probe second" discipline as ``LATENCE_CUDA``.
LATENCE_DEVICE_ENV = "LATENCE_DEVICE"


class DeviceFingerprint(BaseModel):
    """The identity of the device a tune was measured on — the overlay match key (W11).

    Frozen + strict (``extra="forbid"``): a fingerprint is an immutable fact about a box. Two
    fingerprints MATCH (:meth:`matches`) when their :attr:`key` — a stable hash over the identifying
    fields — is equal; the runner overlay applies a tuned config only on a match.

    Fields:

    * ``cuda`` — whether a CUDA device was present (the coarse CPU-vs-GPU split; a CPU tune never
      applies on a GPU host and vice-versa).
    * ``gpu_name`` — the CUDA device name (``nvidia-smi --query-gpu=name``) or ``None`` on CPU.
    * ``gpu_memory_mb`` — total VRAM in MiB (``nvidia-smi --query-gpu=memory.total``) or ``None``.
    * ``cpu_count`` — logical CPU count (``os.cpu_count``), for the CPU/endpoint knobs.
    * ``torch_version`` / ``cuda_version`` — read ONLY if torch is already importable (the compile
      support that bit the pod is torch/driver-version-specific), else ``None`` (honest, never
      faked).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=FINGERPRINT_SCHEMA_VERSION)
    cuda: bool = Field(description="Whether a CUDA device was present when tuned.")
    gpu_name: str | None = Field(default=None, description="CUDA device name, or None on CPU.")
    gpu_memory_mb: int | None = Field(default=None, description="Total VRAM (MiB), or None on CPU.")
    cpu_count: int | None = Field(default=None, description="Logical CPU count, or None.")
    torch_version: str | None = Field(default=None, description="torch version if importable.")
    cuda_version: str | None = Field(default=None, description="torch CUDA version if importable.")

    @property
    def key(self) -> str:
        """A short, stable hash over the identifying fields — the overlay match key.

        Deterministic (sorted, canonical) so the same device yields the same key across runs and
        processes; a change in ANY identifying field yields a different key, so a tune from a
        different GPU / a different torch build never matches. The schema_version is included so a
        fingerprint written under an older schema never spuriously matches a newer one.
        """
        parts = [
            f"v={self.schema_version}",
            f"cuda={self.cuda}",
            f"gpu={self.gpu_name or ''}",
            f"vram={self.gpu_memory_mb or 0}",
            f"cpu={self.cpu_count or 0}",
            f"torch={self.torch_version or ''}",
            f"cudav={self.cuda_version or ''}",
        ]
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return digest[:16]

    def matches(self, other: DeviceFingerprint) -> bool:
        """Whether ``other`` identifies the SAME device (equal :attr:`key`)."""
        return self.key == other.key

    def slug(self) -> str:
        """A filesystem-safe device slug for the default ``tuned/<slug>.yaml`` name.

        Human-readable where possible (the GPU name, sanitised) + the key suffix so two similar GPUs
        never collide on the same file. On CPU it is ``cpu-<key>``.
        """
        if not self.cuda or not self.gpu_name:
            base = "cpu"
        else:
            base = "".join(c if c.isalnum() else "-" for c in self.gpu_name.lower())
            base = "-".join(filter(None, base.split("-")))
        return f"{base}-{self.key}"


def _nvidia_smi(query: str) -> str | None:
    """Read ONE ``nvidia-smi --query-gpu`` field (first GPU), torch-free, or ``None``.

    A fixed argv list + a hard timeout + ``shell=False`` (no injection surface); any failure
    (tool absent, non-zero exit, timeout, empty output) is a ``None``, never a crash — the honest
    "could not read it" the fingerprint records rather than a fabricated value.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell, bounded timeout
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip().splitlines()
    if not value:
        return None
    first = value[0].strip()
    return first or None


def _torch_versions() -> tuple[str | None, str | None]:
    """``(torch_version, cuda_version)`` if torch is ALREADY importable, else ``(None, None)``.

    Never installs or forces a heavy import chain beyond ``import torch`` itself, and swallows any
    import/attribute error to ``(None, None)`` — a torch-free host records an honest absence, and
    the
    compile-support-is-version-specific reason for pinning the torch build is captured where torch
    is
    present without ever making torch a hard dependency (ADR-0016).
    """
    try:
        import torch  # noqa: PLC0415 - optional heavy dep, only imported to read its version
    except Exception:  # noqa: BLE001 - a torch-free host is the honest None case, never a crash
        return (None, None)
    torch_version = getattr(torch, "__version__", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    return (
        str(torch_version) if torch_version is not None else None,
        str(cuda_version) if cuda_version is not None else None,
    )


def probe_device_fingerprint() -> DeviceFingerprint:
    """Gather this box's :class:`DeviceFingerprint` — torch-free where possible, override-driven.

    Deterministic under ``LATENCE_DEVICE`` (the whole fingerprint is pinned to override-derived
    values) AND ``LATENCE_CUDA`` (the CUDA-present bit), so a seeded test yields the same key on any
    host. Otherwise: the CUDA bit from :func:`~latence_core.providers.device.cuda_available`, the
    GPU
    name/VRAM from ``nvidia-smi`` (torch-free), the CPU count from ``os.cpu_count``, and the
    torch/CUDA versions ONLY if torch is already importable.
    """
    override = os.environ.get(LATENCE_DEVICE_ENV)
    if override is not None and override.strip():
        # Fully-pinned deterministic fingerprint for seeded tests: the override names the device,
        # LATENCE_CUDA (defaulting True when a device is named) decides the cuda bit, and the rest
        # are held at fixed override-derived values so the key is reproducible on any host.
        name = override.strip()
        is_cuda = cuda_available() if LATENCE_CUDA_ENV in os.environ else True
        return DeviceFingerprint(
            cuda=is_cuda,
            gpu_name=name if is_cuda else None,
            gpu_memory_mb=None,
            cpu_count=None,
            torch_version=None,
            cuda_version=None,
        )

    is_cuda = cuda_available()
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    if is_cuda:
        gpu_name = _nvidia_smi("name")
        raw_vram = _nvidia_smi("memory.total")
        if raw_vram is not None:
            try:
                gpu_memory_mb = int(float(raw_vram))
            except ValueError:
                gpu_memory_mb = None
    torch_version, cuda_version = _torch_versions()
    return DeviceFingerprint(
        cuda=is_cuda,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_memory_mb,
        cpu_count=os.cpu_count(),
        torch_version=torch_version,
        cuda_version=cuda_version,
    )
