"""T-DEV-2 — the micro-batch and thread defaults READ the machine, not a remembered pod.

Three constants encoded one specific box (DECISIONS.md §D3):

* ``DEFAULT_MICROBATCH_MAX_TEXTS = 4`` and ``DEFAULT_MICROBATCH_TOKEN_BUDGET = 3072`` — chosen
  because ``batch_size=32`` OOM'd and ``batch_size=4`` peaked at **21.6 GiB** on a **32 GB**
  Blackwell pod.
* ``DEFAULT_INTRA_OP_THREADS = 8`` — chosen against a **128-core** host, where torch's default
  intra-op pool (``os.cpu_count()``) oversubscribed to ~129 threads and spun at ~520% CPU with the
  GPU at 0%.

The target is a **24 GB RTX 3090**. 21.6 GiB peak leaves **2.4 GiB** there — before the OCR server
and before fragmentation. So the defaults are re-derived from what is actually detected, under two
hard constraints this file pins:

1. **Byte-identical on the old pod.** Fed 32 GiB / 128 cores the derivation must reproduce 4 / 3072
   / 8 exactly, so nothing about the previously-validated run changes.
2. **Detection may only LOWER the validated point, never raise it.** 4 texts × 768 tokens is the
   only micro-batch size we have ever measured a peak for; a bigger card is an operator decision,
   not an inference from one data point.

Unknown VRAM keeps the reference default rather than collapsing to 1 — and that is a *different*
call from the T-DEV-1 JIT gate on purpose. There the fault is an uncatchable first-forward native
crash, so unknown must fail safe. Here the fault is a CUDA OOM, which the providers' reactive
micro-batch backoff (:func:`~latence_core.providers.perf.is_cuda_oom`) already recovers from — so
crippling every un-probeable host to a batch of one would buy insurance against a recoverable event
at a permanent throughput price.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.providers.perf import (
    DEFAULT_GLINER_MAX_LEN,
    DEFAULT_INTRA_OP_THREADS,
    DEFAULT_MICROBATCH_MAX_TEXTS,
    DEFAULT_MICROBATCH_TOKEN_BUDGET,
    LATENCE_TOTAL_VRAM_MIB_ENV,
    REFERENCE_CPU_CORES,
    REFERENCE_VRAM_MIB,
    derive_intra_op_threads,
    derive_microbatch_max_texts,
    detect_total_vram_mib,
    effective_cpu_count,
    resolve_intra_op_threads,
    resolve_microbatch_bounds,
)

_GIB = 1024


@pytest.fixture(autouse=True)
def _no_vram_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LATENCE_TOTAL_VRAM_MIB_ENV, raising=False)


def _no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither torch nor ``nvidia-smi`` — so a test measures the code, not the host."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)


# --- Constraint 1: the 32 GB / 128-core pod is reproduced EXACTLY ---------------------------------


def test_the_reference_pod_reproduces_todays_constants_exactly() -> None:
    """Fed the box every constant was measured on, the derivation returns those constants.

    This is the no-regression clause of T-DEV-2: the 32 GB Blackwell pod's validated run must not
    change under a "derive it" refactor, or the derivation has silently invalidated the only
    measurement we have.
    """
    assert derive_microbatch_max_texts(REFERENCE_VRAM_MIB) == DEFAULT_MICROBATCH_MAX_TEXTS == 4
    assert (
        derive_microbatch_max_texts(REFERENCE_VRAM_MIB) * DEFAULT_GLINER_MAX_LEN
        == DEFAULT_MICROBATCH_TOKEN_BUDGET
        == 3072
    )
    assert derive_intra_op_threads(REFERENCE_CPU_CORES) == DEFAULT_INTRA_OP_THREADS == 8


# --- Constraint 2: a 24 GB card is bounded at or below the 32 GB card ------------------------------


def test_a_24gb_card_is_bounded_at_or_below_the_32gb_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACCEPTANCE 3 (red before T-DEV-2): the 3090's bound is <= the 32 GB pod's bound.

    Pre-fix ``resolve_microbatch_bounds({})`` returned ``(4, 3072)`` on **every** host, so a 24 GB
    card got a bound derived from 8 GB it does not have — 21.6 GiB of peak against 24 GiB of card,
    before the OCR server and before fragmentation.
    """
    monkeypatch.setenv(LATENCE_TOTAL_VRAM_MIB_ENV, str(32 * _GIB))
    reference_texts, reference_budget = resolve_microbatch_bounds({})

    monkeypatch.setenv(LATENCE_TOTAL_VRAM_MIB_ENV, str(24 * _GIB))
    texts, budget = resolve_microbatch_bounds({})

    assert texts < reference_texts, "a 24 GB card must not inherit the 32 GB card's text count"
    assert budget < reference_budget
    assert (texts, budget) == (3, 3 * DEFAULT_GLINER_MAX_LEN)


def test_the_bound_is_monotone_in_vram() -> None:
    """More VRAM never means a smaller bound, and every bound is at least one text.

    Monotonicity is what makes the derivation reviewable: an operator reading the number off a card
    size can predict it, and a pathologically small card degrades to a batch of one (which the
    micro-batcher already guarantees is always admitted) rather than to zero.
    """
    previous = 0
    for gib in (2, 4, 6, 8, 11, 12, 16, 20, 24, 32):
        bound = derive_microbatch_max_texts(gib * _GIB)
        assert bound >= 1, gib
        assert bound >= previous, gib
        previous = bound


def test_detection_never_raises_the_bound_above_the_measured_point() -> None:
    """An 80 GB A100 still gets 4 — the one point we have ever measured a peak for.

    Scaling *up* from a single measurement would be inventing data. More VRAM is reachable, but as
    an explicit ``microbatch_max_texts`` an operator sets and owns.
    """
    for gib in (40, 48, 80, 141):
        assert derive_microbatch_max_texts(gib * _GIB) == DEFAULT_MICROBATCH_MAX_TEXTS


def test_unknown_vram_keeps_the_reference_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No torch, no ``nvidia-smi`` (the offline CI shape) → the pre-T-DEV-2 default, unchanged.

    Deliberately NOT the T-DEV-1 fail-safe posture: an over-large micro-batch fails as a CUDA OOM,
    which the providers' reactive backoff recovers from, so "unknown" does not justify a permanent
    throughput cut on every CPU host.
    """
    _no_gpu(monkeypatch)
    assert detect_total_vram_mib() is None
    assert derive_microbatch_max_texts(None) == DEFAULT_MICROBATCH_MAX_TEXTS
    assert resolve_microbatch_bounds({}) == (
        DEFAULT_MICROBATCH_MAX_TEXTS,
        DEFAULT_MICROBATCH_TOKEN_BUDGET,
    )


# --- Constraint 3: explicit config still wins ------------------------------------------------------


def test_explicit_microbatch_config_beats_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """ACCEPTANCE 2: an operator's number wins over the detected one, in both directions."""
    monkeypatch.setenv(LATENCE_TOTAL_VRAM_MIB_ENV, str(24 * _GIB))
    assert resolve_microbatch_bounds({"microbatch_max_texts": 8})[0] == 8
    assert resolve_microbatch_bounds({"microbatch_token_budget": 9999})[1] == 9999
    assert resolve_microbatch_bounds({"microbatch_max_texts": 1, "microbatch_token_budget": 64}) == (
        1,
        64,
    )


def test_explicit_intra_op_threads_beats_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thread cap keeps its explicit override + its opt-out, unchanged by the derivation."""
    monkeypatch.setattr("os.cpu_count", lambda: 128)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(4)), raising=False)
    assert resolve_intra_op_threads({"intra_op_threads": 32}) == 32
    assert resolve_intra_op_threads({"intra_op_threads": 0}) is None


def test_invalid_microbatch_config_is_still_a_hard_error() -> None:
    """Detection does not soften the fail-fast contract on a malformed knob."""
    for bad in ("4", 0, -1, 1.5, True):
        with pytest.raises(ValueError, match="microbatch_max_texts"):
            resolve_microbatch_bounds({"microbatch_max_texts": bad})


# --- The thread cap must read the CONTAINER's cores, not the host's -------------------------------


def test_a_container_slice_is_not_told_to_run_the_hosts_thread_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED before T-DEV-2: a 4-core slice of a 128-core host must not get an 8-thread cap.

    This is the concrete defect. ``os.cpu_count()`` reports the **host's** cores, not the cgroup /
    affinity allowance the process actually has — on a rented pod ``nproc`` says 128 while the
    container owns 16 (or 4). Pre-fix the default resolved to ``min(8, 128) = 8``, i.e. 2x
    oversubscription on a 4-core slice, which is the same many-core thrash W9-scaling introduced the
    cap to prevent, just at the other end of the scale.
    """
    monkeypatch.setattr("os.cpu_count", lambda: 128)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(4)), raising=False)
    monkeypatch.setattr("os.process_cpu_count", lambda: 4, raising=False)

    assert effective_cpu_count() == 4
    assert resolve_intra_op_threads({}) == 4


def test_a_128_core_host_still_gets_the_pod_validated_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """And the 128-core box the cap was measured on is unchanged: 8."""
    monkeypatch.setattr("os.cpu_count", lambda: REFERENCE_CPU_CORES)
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(128)), raising=False)
    monkeypatch.setattr("os.process_cpu_count", lambda: REFERENCE_CPU_CORES, raising=False)

    assert resolve_intra_op_threads({}) == DEFAULT_INTRA_OP_THREADS


def test_derive_intra_op_threads_is_conservative_on_unknown_core_counts() -> None:
    """An unreadable core count keeps the reference cap rather than guessing a smaller one."""
    assert derive_intra_op_threads(None) == DEFAULT_INTRA_OP_THREADS
    assert derive_intra_op_threads(0) == 1
    assert derive_intra_op_threads(1) == 1


# --- The VRAM probe itself -------------------------------------------------------------------------


def test_vram_is_read_from_nvidia_smi_without_importing_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe is torch-free (ADR-0016) — ``latence-core`` must not pull torch to size a batch."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="24576\n"),
    )

    assert detect_total_vram_mib() == 24576
    assert "torch" not in sys.modules


def test_a_mixed_gpu_host_is_sized_by_its_SMALLEST_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bound is applied process-wide, so the smallest card is the one that must survive it."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="81920\n24576\n"),
    )

    assert detect_total_vram_mib() == 24576


def test_an_unreadable_vram_answer_is_unknown_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``[N/A]`` from a driver that cannot report it must not parse into "0 MiB of VRAM"."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="[N/A]\n"),
    )

    assert detect_total_vram_mib() is None


def test_vram_is_read_from_an_already_imported_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When torch is already in ``sys.modules`` it is the most accurate source — and it is free."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.device_count = lambda: 1  # type: ignore[attr-defined]
    cuda.get_device_properties = lambda _i: types.SimpleNamespace(  # type: ignore[attr-defined]
        total_memory=24 * 1024 * 1024 * 1024
    )
    torch.cuda = cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert detect_total_vram_mib() == 24 * _GIB


def test_a_torch_that_cannot_answer_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU wheel / a driver-less container reports unknown, never a fabricated size."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    torch = types.ModuleType("torch")

    def _boom(*_a: Any, **_k: Any) -> int:
        msg = "no CUDA driver"
        raise RuntimeError(msg)

    cuda = types.ModuleType("torch.cuda")
    cuda.device_count = _boom  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert detect_total_vram_mib() is None
