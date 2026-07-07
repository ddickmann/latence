"""P3-F1 — the dep-light device seam: cuda_available + select_device (ADR-0036 §1/§2).

Deterministic + seeded via the ``LATENCE_CUDA`` env override (the CI/test knob), so the
cuda path is exercised without a GPU and the no-cuda skip-with-flag path is exercised on
this CPU host. Asserts: the override drives cuda_available; routing for cpu / either / gpu
profiles; the G1 skip-with-flag posture (a GPU-only provider on a CPU-only host is
skipped, NEVER raised); and that no torch import is pulled into core.
"""

from __future__ import annotations

import pytest
from latence_core import DeviceDecision, cuda_available, select_device
from latence_core.providers.device import LATENCE_CUDA_ENV
from latence_core.providers.profile import ProviderProfile


def _profile(compute: str) -> ProviderProfile:
    is_pure_code = compute == "cpu"
    return ProviderProfile(
        compute=compute,  # type: ignore[arg-type]
        memory_mb=64,
        model_id=None if is_pure_code else "m",
        license="Apache-2.0",
        license_verified=True,
        license_code="Apache-2.0",
        license_weights=None if is_pure_code else "Apache-2.0",
        license_source="LICENSE" if is_pure_code else "https://huggingface.co/m",
        license_verified_on="2026-07-06",
        deterministic=is_pure_code,
        batch=False,
        cost_per_1k=None,
    )


# -- cuda_available: override-driven, defaults False when unknown --------------


def test_cuda_available_override_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    assert cuda_available() is True


def test_cuda_available_override_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "0")
    assert cuda_available() is False


def test_cuda_available_override_spellings(monkeypatch: pytest.MonkeyPatch) -> None:
    for truthy in ("true", "YES", "on", "1"):
        monkeypatch.setenv(LATENCE_CUDA_ENV, truthy)
        assert cuda_available() is True
    for falsy in ("false", "NO", "off", "0"):
        monkeypatch.setenv(LATENCE_CUDA_ENV, falsy)
        assert cuda_available() is False


def test_cuda_available_garbage_override_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unrecognised value is ignored (falls through to the probe), never guessed truthy.
    monkeypatch.setenv(LATENCE_CUDA_ENV, "maybe")
    # On this CPU-only host the best-effort probe returns False; the point is it does not
    # crash and does not treat "maybe" as truthy.
    assert cuda_available() is False


# -- select_device routing -----------------------------------------------------


def test_cpu_profile_always_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")  # even with cuda present
    d = select_device("auto", _profile("cpu"))
    assert d == DeviceDecision(device="cpu", skipped=False, reason="provider is cpu-only")


def test_cpu_profile_clamps_cuda_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    d = select_device("cuda", _profile("cpu"))
    assert d.device == "cpu"
    assert d.skipped is False
    assert "clamped" in d.reason


def test_gpu_profile_on_cpu_host_skips_with_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # THE G1 posture: a GPU-only provider on a CPU-only host is skipped-with-flag, not raised.
    monkeypatch.setenv(LATENCE_CUDA_ENV, "0")
    d = select_device("auto", _profile("gpu"))
    assert d.skipped is True
    assert d.device is None
    assert "no CUDA" in d.reason or "GPU" in d.reason


def test_gpu_profile_cuda_request_on_cpu_host_skips_not_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "0")
    # Even an EXPLICIT cuda request skips-with-flag rather than raising (default posture).
    d = select_device("cuda", _profile("gpu"))
    assert d.skipped is True
    assert d.device is None


def test_gpu_profile_with_cuda_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    d = select_device("auto", _profile("gpu"))
    assert d.skipped is False
    assert d.device == "cuda"


def test_gpu_profile_explicit_cpu_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    # CUDA present but caller pinned cpu for a GPU-only provider: it cannot run on cpu → skip.
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    d = select_device("cpu", _profile("gpu"))
    assert d.skipped is True
    assert d.device is None


def test_either_profile_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    assert select_device("auto", _profile("either")).device == "cuda"
    monkeypatch.setenv(LATENCE_CUDA_ENV, "0")
    assert select_device("auto", _profile("either")).device == "cpu"


def test_either_profile_cuda_request_no_cuda_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "0")
    d = select_device("cuda", _profile("either"))
    assert d.skipped is True
    assert d.device is None


def test_either_profile_cpu_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    d = select_device("cpu", _profile("either"))
    assert d.device == "cpu"
    assert d.skipped is False


def test_none_profile_follows_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # A provider that declares no profile is treated as ``either``.
    monkeypatch.setenv(LATENCE_CUDA_ENV, "0")
    assert select_device("auto", None).device == "cpu"
    assert select_device("cuda", None).skipped is True
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    assert select_device("auto", None).device == "cuda"


def test_bad_requested_device_raises() -> None:
    # A genuinely malformed request string is a config error (ValueError), surfaced BEFORE
    # any routing — distinct from the never-raise skip case.
    with pytest.raises(ValueError, match="Unknown device"):
        select_device("gpu", _profile("either"))


def test_empty_request_is_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LATENCE_CUDA_ENV, "1")
    assert select_device("", _profile("either")).device == "cuda"


def test_no_torch_import_in_device_seam() -> None:
    # ADR-0016: the device seam must not import torch. Reading the module source is a
    # cheap structural guard that a future edit does not smuggle a torch dep into core.
    # Guard the actual import statements, not the docstring mentions of the word "torch".
    import inspect

    from latence_core.providers import device
    from latence_core.providers import profile as profile_mod

    for mod in (device, profile_mod):
        src = inspect.getsource(mod)
        assert "import torch" not in src
        assert "from torch" not in src
