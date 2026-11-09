"""REGRESSION (b): ``redaction.gliner2`` caps torch's intra-op threads at load, like its twin.

The defect. ``fused_entity_relation.gliner2`` called ``cap_intra_op_threads`` (7 references in its
source); ``redaction.gliner2`` had **none**. Both run the same mdeberta-v3 backbone, in the same
process, on the same many-core host, in the blessed stack — and the failure the cap exists to
prevent is a *process-wide* torch thread pool: on the diagnosed 128-core Blackwell box torch's
default intra-op pool is ``os.cpu_count()``, gliner2's internal batching oversubscribes to ~129
threads, and the run sits at ~520% CPU with the GPU at 0% making no forward progress. Exporting
``OMP_NUM_THREADS`` does not help, because once torch is imported it owns its own pool.

So protecting the extract Provider and not the redact Provider protected neither: whichever loaded
second inherited a pool the first had capped, or — in a redact-only pipeline — no cap at all. The
cap now rides the shared load choreography, so it is not something a Provider can be wired without.

The assertion is on the OBSERVED effect (``torch.set_num_threads`` actually called, with the
configured value) — never on a config flag.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest
from latence_pii_gliner2 import Gliner2PIIRedactor


class _FakeGLiNER2:
    """A gliner2 stand-in exposing the batched API the Provider checks for at load."""

    def __init__(self) -> None:
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs: Any) -> _FakeGLiNER2:
        return cls()

    def to(self, target: str) -> _FakeGLiNER2:
        self.device = target
        return self

    def batch_extract_entities(self, *_a: Any, **_k: Any) -> list[Any]:
        return []


@pytest.fixture()
def fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _FakeGLiNER2  # type: ignore[attr-defined]
    module.__version__ = "1.3.2"  # type: ignore[attr-defined]  # the #192 version gate reads this
    monkeypatch.setitem(sys.modules, "gliner2", module)


@pytest.fixture()
def recording_torch(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """A ``torch`` that records ``set_num_threads`` — the only lever the cap actually pulls."""
    calls: list[int] = []
    torch = types.ModuleType("torch")
    torch.set_num_threads = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.fixture()
def cores(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Pin the host core count so a cap assertion is about the CAP, not about this machine.

    ``resolve_intra_op_threads`` clamps the requested cap to ``os.cpu_count()``. A test that asserts
    a bare literal is therefore only true on a host with at least that many cores: it is green on a
    10-core dev box and RED on the 2-4 vCPU ``ubuntu-latest`` runner CI runs the suite on. Pinning
    the count makes every assertion below host-independent — high to hold the clamp inert while the
    cap itself is under test, low in the one test that puts the clamp under test.

    Every probe ``effective_cpu_count`` consults is pinned, not just ``os.cpu_count``: the default
    cap derives from the affinity-aware ``process_cpu_count`` / ``sched_getaffinity`` first, which
    say 4 on a 4-vCPU runner whatever ``cpu_count`` claims. ``raising=False`` because neither probe
    exists on every platform.
    """

    def _pin(n: int) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: n)
        monkeypatch.setattr(os, "process_cpu_count", lambda: n, raising=False)
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(n)), raising=False)

    return _pin


def _config(**extra: Any) -> dict[str, Any]:
    return {"pii_labels": ["person"], "device": "cpu", **extra}


def test_redaction_gliner2_caps_intra_op_threads_at_load(
    fake_gliner2: None, recording_torch: list[int], cores: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default cap (8) is applied — the twin had this, this Provider did not."""
    monkeypatch.setenv("LATENCE_CUDA", "0")
    cores(64)  # a many-core host: the clamp is inert, so this asserts the cap and nothing else
    Gliner2PIIRedactor(_config())._load_model()
    assert recording_torch == [8], (
        "redaction.gliner2 must cap torch's intra-op threads at load; without it a many-core host "
        "oversubscribes to ~cpu_count threads and spins at ~520% CPU with the GPU idle"
    )


def test_the_cap_never_exceeds_the_host_core_count(
    fake_gliner2: None, recording_torch: list[int], cores: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the contract: on a small host the cap clamps down to the cores present.

    Asking torch for 8 intra-op threads on a 4-vCPU CI runner would itself be the oversubscription
    this guard exists to prevent, so the clamp is behaviour worth pinning — and pinning it here is
    what lets the test above assert a literal instead of restating ``min(...)`` back at the code.
    """
    monkeypatch.setenv("LATENCE_CUDA", "0")
    cores(4)
    Gliner2PIIRedactor(_config())._load_model()
    assert recording_torch == [4]


def test_the_cap_is_configurable(
    fake_gliner2: None, recording_torch: list[int], cores: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")
    cores(64)
    Gliner2PIIRedactor(_config(intra_op_threads=2))._load_model()
    assert recording_torch == [2]


def test_the_cap_can_be_explicitly_disabled(
    fake_gliner2: None, recording_torch: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``intra_op_threads: 0`` is the documented opt-out for a box that wants all cores."""
    monkeypatch.setenv("LATENCE_CUDA", "0")
    Gliner2PIIRedactor(_config(intra_op_threads=0))._load_model()
    assert recording_torch == []


def test_an_invalid_thread_cap_is_a_config_error_at_construction(fake_gliner2: None) -> None:
    """Fail-fast on a bad value, like ``batch_size`` — not a native fault buried in the lazy load."""
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="intra_op_threads"):
        Gliner2PIIRedactor(_config(intra_op_threads="lots"))
