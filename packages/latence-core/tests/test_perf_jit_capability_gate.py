"""T-DEV-1 — the TorchScript-JIT kill is GATED on the detected compute capability.

``neutralize_torch_jit`` exists for exactly one fault: **FM-NVRTC-JIT**, the missing
``libnvrtc-builtins.so.13.0`` on a Blackwell / CUDA-13 box. That fault does not exist on sm_86.
Before this ticket the kill was unconditional at four call sites (``perf.py:306``, ``:679``,
``:1147`` and ``latence-disambig-glinker/backends.py:338``) — an Ampere tax on the three stages
that consumed **1,604 s of the prior run's 1,743 s** (DECISIONS.md §D2).

**The asymmetry that shapes every assertion below.** The nvrtc fault surfaces on the FIRST
FORWARD, so no load-time ``try/except`` can catch it: a wrong *skip* is a native **crash**, a wrong
*neutralize* is merely **slow**. So the gate FAILS SAFE — it neutralizes on unknown, unreadable,
torch-absent and ``>= 10.0``, and skips only on a positively-read known-good capability. Every test
here is written from that direction: the "leave it on" assertions are the narrow, positively-proven
case; the "kill it" assertions are the wide default.
"""

from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from latence_core.providers.perf import (
    JIT_SAFE_MAX_CAPABILITY,
    LATENCE_COMPUTE_CAPABILITY_ENV,
    TORCH_JIT_ENV,
    detect_compute_capability,
    neutralize_torch_jit,
    resolve_torch_jit_policy,
    torch_jit_must_be_neutralized,
)


class _FakeEnabled:
    """Mirrors ``torch.jit._state._enabled`` — the proxy torch parses ``PYTORCH_JIT`` into."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled


def _install_fake_torch(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int] | None
) -> Any:
    """A fake ``torch`` whose JIT starts LIVE and whose CUDA reports ``capability``.

    ``capability=None`` installs a torch with **no readable** ``cuda.get_device_capability`` (the
    CPU-only / stub shape) — which the gate must treat as *unknown*, not as *safe*.
    """
    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    state = types.ModuleType("torch.jit._state")
    enabled = _FakeEnabled(True)
    state._enabled = enabled  # type: ignore[attr-defined]
    state.disable = lambda: setattr(enabled, "enabled", False)  # type: ignore[attr-defined]
    jit._state = state  # type: ignore[attr-defined]
    torch.jit = jit  # type: ignore[attr-defined]
    if capability is not None:
        cuda = types.ModuleType("torch.cuda")
        cuda.get_device_capability = lambda *a, **k: capability  # type: ignore[attr-defined]
        torch.cuda = cuda  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch.cuda", cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.jit", jit)
    return torch


@pytest.fixture(autouse=True)
def _clean_jit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with the JIT switch UNSET and no capability override in force."""
    monkeypatch.delenv(TORCH_JIT_ENV, raising=False)
    monkeypatch.delenv(LATENCE_COMPUTE_CAPABILITY_ENV, raising=False)


def _no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the torch-free ``nvidia-smi`` probe report "no such tool" deterministically.

    Without this a test host that happens to HAVE a GPU would answer the probe and the assertion
    would measure the host rather than the code.
    """
    monkeypatch.setattr("shutil.which", lambda name: None)


# --- Acceptance 1: a known-good architecture leaves the JIT ON -----------------------------------


def test_ampere_sm86_leaves_the_jit_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 1 (red before T-DEV-1): on sm_86 the JIT is left ON.

    ``86`` has a healthy nvrtc path — the ``libnvrtc-builtins.so.13.0`` fault is a Blackwell /
    CUDA-13 condition. Pre-fix this failed: ``neutralize_torch_jit`` set ``PYTORCH_JIT=0`` and
    disabled the runtime state for **every** host, so the three heaviest stages ran unfused on
    hardware that never had the fault.
    """
    monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, "8.6")
    torch = _install_fake_torch(monkeypatch, (8, 6))

    assert neutralize_torch_jit(config={}) is False  # nothing was neutralized

    assert TORCH_JIT_ENV not in os.environ, "the env lever must not be pulled on a known-good arch"
    assert torch.jit._state._enabled.enabled is True, "the runtime lever must not be pulled either"


def test_every_capability_at_or_below_the_safe_ceiling_leaves_the_jit_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Volta (7.0) through Hopper (9.0) are all positively known-good, not just sm_86."""
    for capability in ((7, 0), (7, 5), (8, 0), (8, 6), (8, 9), JIT_SAFE_MAX_CAPABILITY):
        monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, f"{capability[0]}.{capability[1]}")
        _install_fake_torch(monkeypatch, capability)
        assert neutralize_torch_jit(config={}) is False, capability
        assert TORCH_JIT_ENV not in os.environ, capability


# --- Acceptance 2: Blackwell / CUDA-13 still gets the kill ----------------------------------------


def test_blackwell_sm120_is_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 2: capability 12.0 — the box the workaround exists for — still gets the kill."""
    monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, "12.0")
    torch = _install_fake_torch(monkeypatch, (12, 0))

    assert neutralize_torch_jit(config={}) is True

    assert os.environ[TORCH_JIT_ENV] == "0"
    assert torch.jit._state._enabled.enabled is False


def test_blackwell_datacenter_sm100_is_also_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """sm_100 (B100/B200/GB200) is on the KILL side even though the observed fault was sm_120.

    Deliberate over-reach in the fail-safe direction: sm_100 is the same Blackwell/CUDA-13
    generation, we have never run it, and the cost of being wrong is asymmetric (crash vs slow).
    """
    monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, "10.0")
    _install_fake_torch(monkeypatch, (10, 0))

    assert neutralize_torch_jit(config={}) is True
    assert os.environ[TORCH_JIT_ENV] == "0"


# --- Acceptance 3: every unknown is a kill (fail safe) --------------------------------------------


def test_torch_absent_is_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 3a: no torch at all → neutralize (the env lever still arms a later import)."""
    _no_nvidia_smi(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", None)

    neutralize_torch_jit(config={})

    assert os.environ[TORCH_JIT_ENV] == "0"


def test_torch_whose_cuda_probe_raises_is_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 3b: a torch whose ``get_device_capability`` RAISES is unknown, not safe."""
    _no_nvidia_smi(monkeypatch)
    torch = _install_fake_torch(monkeypatch, (8, 6))

    def _boom(*_a: Any, **_k: Any) -> tuple[int, int]:
        msg = "CUDA driver initialization failed"
        raise RuntimeError(msg)

    torch.cuda.get_device_capability = _boom

    assert neutralize_torch_jit(config={}) is True
    assert os.environ[TORCH_JIT_ENV] == "0"


def test_torch_without_a_cuda_module_is_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 3c: a torch exposing no ``cuda`` (a stub, a CPU wheel) is unknown → kill."""
    _no_nvidia_smi(monkeypatch)
    _install_fake_torch(monkeypatch, None)

    assert neutralize_torch_jit(config={}) is True
    assert os.environ[TORCH_JIT_ENV] == "0"


def test_unreadable_nvidia_smi_output_is_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 3d: ``nvidia-smi`` present but answering garbage is unknown → kill.

    An old driver whose ``nvidia-smi`` predates ``--query-gpu=compute_cap`` answers with an error
    string, not a capability. That must not be parsed into a false "known-good".
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="Field 'compute_cap' invalid\n"),
    )

    assert detect_compute_capability() is None
    # ``neutralize_torch_jit``'s return value is "the in-process JIT is known-off", which a
    # torch-free host cannot report; the DECISION is the assertion, plus the env lever it pulled.
    assert torch_jit_must_be_neutralized({}) is True
    neutralize_torch_jit(config={})
    assert os.environ[TORCH_JIT_ENV] == "0"


def test_nvidia_smi_missing_entirely_is_neutralized(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with neither torch nor ``nvidia-smi`` — the offline CI shape — is unknown → kill."""
    _no_nvidia_smi(monkeypatch)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    assert detect_compute_capability() is None
    assert torch_jit_must_be_neutralized({}) is True
    neutralize_torch_jit(config={})
    assert os.environ[TORCH_JIT_ENV] == "0"


# --- The torch-free probe reads a real ``nvidia-smi`` answer --------------------------------------


def test_capability_is_read_without_importing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe is torch-free (ADR-0016) AND ordering-safe.

    It cannot import torch to answer: ``PYTORCH_JIT`` is read by torch **at torch import**, so a
    probe that imported torch would burn the very lever it is deciding about. ``nvidia-smi`` is the
    same torch-free signal ``device.cuda_available()`` already uses.
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    calls: list[list[str]] = []

    def _run(cmd: list[str], **_k: Any) -> Any:
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="8.6\n")

    monkeypatch.setattr("subprocess.run", _run)

    assert detect_compute_capability() == (8, 6)
    assert "torch" not in sys.modules, "the capability probe must not import torch"
    assert calls and calls[0][0] == "nvidia-smi"


def test_multi_gpu_host_reports_the_lowest_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """``nvidia-smi`` prints one line per GPU; the LOWEST wins.

    A mixed box must not be declared known-good by its best card, and must not be declared unsafe
    by its worst — the JIT switch is process-wide, so the safe reading is the minimum, which is
    also the one that keeps the kill on when any card is a Blackwell.
    """
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="12.0\n8.6\n"),
    )

    assert detect_compute_capability() == (8, 6)


# --- The operator override ------------------------------------------------------------------------


def test_operator_can_force_the_kill_on_a_known_good_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """``torch_jit: off`` pins the neutralization even where the capability says it is unneeded."""
    monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, "8.6")
    _install_fake_torch(monkeypatch, (8, 6))

    assert neutralize_torch_jit(config={"torch_jit": "off"}) is True
    assert os.environ[TORCH_JIT_ENV] == "0"


def test_operator_can_force_the_jit_on_where_the_gate_would_kill_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``torch_jit: on`` is the escape hatch — it beats even a positively-detected Blackwell.

    Recorded as a test so the hatch is a decision, not an accident: an operator who has a patched
    nvrtc can take the throughput back, and nothing in the seam second-guesses them.
    """
    monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, "12.0")
    torch = _install_fake_torch(monkeypatch, (12, 0))

    assert neutralize_torch_jit(config={"torch_jit": "on"}) is False
    assert TORCH_JIT_ENV not in os.environ
    assert torch.jit._state._enabled.enabled is True


def test_invalid_torch_jit_value_is_a_hard_error() -> None:
    """A typo'd knob is a ConfigError at the caller, never a silently-guessed policy."""
    with pytest.raises(ValueError, match="invalid torch_jit"):
        resolve_torch_jit_policy({"torch_jit": "maybe"})


def test_torch_jit_policy_accepts_bools_and_the_usual_spellings() -> None:
    """``True``/``False`` and the string spellings map onto on/off; unset is ``auto``."""
    assert resolve_torch_jit_policy({}) == "auto"
    assert resolve_torch_jit_policy({"torch_jit": "auto"}) == "auto"
    for on in (True, "on", "true", "1", "yes"):
        assert resolve_torch_jit_policy({"torch_jit": on}) == "on", on
    for off in (False, "off", "false", "0", "no"):
        assert resolve_torch_jit_policy({"torch_jit": off}) == "off", off


# --- The bare call stays the unconditional applier -------------------------------------------------


def test_the_bare_call_is_still_the_unconditional_applier(monkeypatch: pytest.MonkeyPatch) -> None:
    """``neutralize_torch_jit()`` with NO config neutralizes unconditionally, as it always did.

    The gate is opt-in by handing the seam a Provider ``config``. Keeping the bare call
    unconditional is what lets the one function keep the one name — which the AST ordering guard
    (``test_perf_fp32_regression_guard.py``) matches on lexically, and which the pre-gate call
    sites and tests already depend on.
    """
    monkeypatch.setenv(LATENCE_COMPUTE_CAPABILITY_ENV, "8.6")
    torch = _install_fake_torch(monkeypatch, (8, 6))

    assert neutralize_torch_jit() is True
    assert os.environ[TORCH_JIT_ENV] == "0"
    assert torch.jit._state._enabled.enabled is False


# --- The four call sites are ALL gated -------------------------------------------------------------


_GATED_CALL_SITES = (
    "latence-core/src/latence_core/providers/perf.py",
    "latence-disambig-glinker/src/latence_disambig_glinker/backends.py",
)


def _packages_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_every_in_tree_neutralize_call_passes_a_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """STRUCTURAL: no in-tree call site may take the ungated path.

    The gate is only worth what its coverage is: one forgotten call site re-imposes the Ampere tax
    on whichever Stage routes through it. ``perf.py:306`` / ``:679`` / ``:1147`` and
    ``backends.py:338`` are the four; this asserts by AST that each hands over a ``config``, so a
    fifth site added later cannot quietly default to unconditional.
    """
    del monkeypatch
    offenders: list[str] = []
    for rel in _GATED_CALL_SITES:
        path = _packages_root() / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else "")
            )
            if name != "neutralize_torch_jit":
                continue
            if not any(kw.arg == "config" for kw in node.keywords):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "these neutralize_torch_jit() call sites take the UNGATED path and so kill the "
        f"TorchScript JIT on Ampere too (T-DEV-1): {offenders}"
    )
