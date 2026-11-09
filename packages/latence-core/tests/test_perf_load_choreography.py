"""The load choreography, tested THROUGH the interface with a recording fake (C4).

This is what replaced ~830 lines of AST linting. The old guard parsed each Provider's source to
assert that seven perf-seam calls were present and correctly sequenced; the sequence now lives
inside three deep entry points, so it is tested the way any behaviour is — by calling the interface
and observing what actually happened, in what order.

**Anti-false-green.** The fake ``torch`` here MIRRORS the real one where the seam touches it and
REJECTS what the real one rejects:

* ``torch.jit._state.disable()`` really flips ``_enabled.enabled``, so "the JIT is off" is read
  back, not assumed;
* the fake model is created resident on ``"cpu"`` — exactly what ``from_pretrained`` does, which is
  the whole reason placement exists — and only ``.to(device)`` moves it. A seam that forgets to
  place is therefore observable, not merely un-asserted;
* ``.to()`` on a fake that refuses to move leaves ``device`` unchanged, which is the #191 shape:
  the placement call "succeeded" and the model is still on the wrong device. The seam must raise.

Every assertion below is on OBSERVED EFFECTS (an ordered event log, a resident device string, the
kwargs a load actually received) — never on a flag saying an effect was requested.
"""

from __future__ import annotations

import os
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.errors import ConfigError, ProviderError
from latence_core.providers.device import DeviceDecision
from latence_core.providers.perf import (
    TORCH_JIT_ENV,
    load_gliner2_model,
    load_gliner_model,
    load_transformers_model,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# --- the recording fakes ----------------------------------------------------


class _Dtype:
    """Stands in for a real ``torch.dtype`` — an OBJECT, never a str (see :func:`_make_fake_torch`)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _Enabled:
    """Mirrors ``torch.jit._state._enabled`` — the proxy torch parses ``PYTORCH_JIT`` into."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled


class _JitState:
    def __init__(self, log: list[str]) -> None:
        self._enabled = _Enabled(enabled=True)
        self._log = log

    def disable(self) -> None:
        self._enabled.enabled = False
        self._log.append("jit-off")


class _FakeConfig:
    """A model config that records ``max_len`` writes — the W12 window bound is observable."""

    _log: list[str]
    max_len: int

    def __init__(self, log: list[str]) -> None:
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "max_len", 512)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "max_len":
            self._log.append(f"max_len={value}")


class _FakeModel:
    """A model that behaves like a freshly loaded ``nn.Module``: resident on CPU until moved.

    ``movable=False`` reproduces the #191 shape — ``.to()`` returns without error and the model is
    STILL on cpu. A seam that trusts the call instead of reading the device back cannot tell the
    difference, which is exactly how ``device: cuda`` ran an 11.5 s/chunk CPU job with no warning.
    """

    def __init__(self, log: list[str], *, movable: bool = True) -> None:
        self.log = log
        self.movable = movable
        self.device = "cpu"
        self.config = _FakeConfig(log)
        self.load_kwargs: dict[str, Any] = {}

    def to(self, target: Any) -> _FakeModel:
        if isinstance(target, str):
            self.log.append(f"to-device:{target}")
            if self.movable:
                self.device = target
        else:
            self.log.append(f"to-dtype:{target}")
        return self


def _make_fake_torch(log: list[str]) -> types.ModuleType:
    """A ``torch`` that mirrors every lever this seam pulls, and records each one."""
    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    jit._state = _JitState(log)  # type: ignore[attr-defined]
    torch.jit = jit  # type: ignore[attr-defined]
    # NOT strings: ``load_then_cast`` deliberately skips the cast when the resolved dtype came
    # back as a NAME string, which is how it stays importable on a torch-free host. A fake whose
    # dtypes are strings would silently disable the very cast under test.
    torch.bfloat16 = _Dtype("bfloat16")  # type: ignore[attr-defined]
    torch.float32 = _Dtype("float32")  # type: ignore[attr-defined]

    def set_num_threads(n: int) -> None:
        log.append(f"threads={n}")

    def compile_(model: Any, **_kw: Any) -> Any:
        log.append("compiled")
        return model

    torch.set_num_threads = set_num_threads  # type: ignore[attr-defined]
    torch.compile = compile_  # type: ignore[attr-defined]
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False  # type: ignore[attr-defined]
    cuda.empty_cache = lambda: None  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]
    return torch


@pytest.fixture()
def log() -> list[str]:
    return []


@pytest.fixture()
def fake_torch(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> Iterator[types.ModuleType]:
    torch = _make_fake_torch(log)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv(TORCH_JIT_ENV, raising=False)
    yield torch


@pytest.fixture()
def cores(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Pin the host core count so a ``threads=N`` assertion is about N, not about this machine.

    ``resolve_intra_op_threads`` clamps the requested cap to ``os.cpu_count()``, so an unpinned
    ``assert "threads=8" in log`` is only true on a host with at least 8 cores: green on the
    10-core dev box, RED on the 2-4 vCPU ``ubuntu-latest`` runner CI runs the whole suite on. None
    of the choreography tests are about the machine's core count — they are about the cap value and
    where in the sequence it lands. Pinning the count high holds the clamp inert so they assert
    exactly that; :func:`test_the_intra_op_cap_never_exceeds_the_host_core_count` pins it low and
    puts the clamp itself under test.

    Pinning ``os.cpu_count`` alone is not enough: the DEFAULT cap is derived from
    ``effective_cpu_count()``, which asks ``os.process_cpu_count`` and ``os.sched_getaffinity``
    first (the affinity-aware answer). On a 4-vCPU runner those say 4 whatever ``cpu_count``
    claims, so all three probes are pinned to the same count — and the real
    ``effective_cpu_count`` still runs.
    """

    def _pin(n: int) -> None:
        _pin_core_probes(monkeypatch, n)

    return _pin


def _pin_core_probes(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """Make every core-count probe ``effective_cpu_count`` consults answer ``n``.

    ``raising=False`` because ``process_cpu_count`` is 3.13+ and ``sched_getaffinity`` is
    Linux-only; setting the one a platform lacks just adds the attribute for this test.
    """
    monkeypatch.setattr(os, "cpu_count", lambda: n)
    monkeypatch.setattr(os, "process_cpu_count", lambda: n, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(n)), raising=False)


_MANY_CORES = 64  # the clamp is inert above the cap, so the cap alone is observable


def _factory(log: list[str], model: _FakeModel, *, family: str = "gliner") -> Any:
    """A backbone-import factory that records the JIT state AT THE MOMENT OF IMPORT.

    This is the #190 property, made observable without torch or a GPU: the real failure is that
    transformers scripts its deberta_v2 relative-position helpers *while the module is being
    imported*, so what matters is whether the JIT was already dead at that instant — not whether
    ``neutralize_torch_jit`` was called at all.
    """

    class _Cls:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            log.append(f"load:{model_id}:{sorted(kwargs.items())}")
            model.load_kwargs = dict(kwargs)
            return model

    def factory() -> Any:
        jit_live = sys.modules["torch"].jit._state._enabled.enabled  # noqa: SLF001
        log.append(f"import:{family}:jit_live={jit_live}")
        return _Cls

    return factory


_CUDA = DeviceDecision(device="cuda", skipped=False, reason="test: cuda")
_CPU = DeviceDecision(device="cpu", skipped=False, reason="test: cpu")
_SKIPPED = DeviceDecision(device=None, skipped=True, reason="no CUDA device is available (skipped)")


# --- the sequence -----------------------------------------------------------


def test_gliner_load_runs_the_whole_sequence_in_order(
    fake_torch: types.ModuleType, log: list[str], cores: Any
) -> None:
    """REGRESSION (C4): one call performs all seven steps, in the one order that is correct.

    The ordering the deleted AST linter existed to assert, now asserted on OBSERVED EFFECTS. Note
    what a caller supplied: a factory, an id, a config, a decision. Nothing about order.
    """
    cores(_MANY_CORES)  # the step is "the cap landed here", not "this box has >= 8 cores"
    model = _FakeModel(log)
    out = load_gliner_model(
        _factory(log, model),
        "acme/gliner",
        {"compile": True, "max_len": 640},
        _CUDA,
        provider_name="test.gliner",
    )

    assert out is model
    # The JIT was already dead when the backbone import evaluated — the #190 property.
    assert "import:gliner:jit_live=False" in log
    order = [
        log.index("jit-off"),
        log.index("threads=8"),
        log.index("import:gliner:jit_live=False"),
        next(i for i, e in enumerate(log) if e.startswith("load:")),
        log.index("to-dtype:torch.bfloat16"),
        log.index("to-device:cuda"),
        log.index("max_len=640"),
        log.index("compiled"),
    ]
    assert order == sorted(order), f"steps ran out of order: {log}"
    assert model.device == "cuda"


def test_gliner_load_passes_no_perf_kwargs_to_the_plain_library(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """The plain ``gliner`` library REJECTS ``torch_dtype``/``attn_implementation`` (pod-proven).

    So the bf16 win must come from the ``.to(dtype)`` cast, never from spreading perf kwargs — a
    property the old linter asserted lexically and this asserts on the kwargs the load received.
    """
    model = _FakeModel(log)
    load_gliner_model(
        _factory(log, model), "acme/gliner", {}, _CUDA, provider_name="test.gliner"
    )
    assert model.load_kwargs == {}
    assert "to-dtype:torch.bfloat16" in log


def test_cpu_load_skips_the_bf16_cast_and_still_places(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """bf16 on CPU is not a universal win, so the cast is cuda-only — placement is not."""
    model = _FakeModel(log)
    load_gliner_model(_factory(log, model), "acme/gliner", {}, _CPU, provider_name="test.gliner")
    assert not [e for e in log if e.startswith("to-dtype:")]
    assert model.device == "cpu"
    assert "to-device:cpu" in log


def test_a_skipped_device_decision_is_a_loud_config_error(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """REGRESSION (#191): honour the resolved device or fail loudly — never a silent CPU fallback.

    And the failure happens BEFORE the backbone import, so a skipped decision never pays the cost of
    importing torch/transformers to then do nothing useful.
    """
    model = _FakeModel(log)
    for entry in (load_gliner_model, load_gliner2_model, load_transformers_model):
        with pytest.raises(ConfigError, match="no CUDA device is available"):
            entry(
                _factory(log, model), "acme/m", {}, _SKIPPED, provider_name="test.p"
            )
    assert not [e for e in log if e.startswith("import:")]


def test_a_model_that_lands_on_the_wrong_device_is_refused(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """REGRESSION (#191): the placement is VERIFIED by read-back, not trusted.

    The fake's ``.to()`` succeeds and leaves the model on cpu — the shape a bare
    ``model = model.to(device)`` cannot distinguish from success, and the reason a ``device: cuda``
    run sat at ~11.5 s/chunk with the GPU idle and no warning.
    """
    model = _FakeModel(log, movable=False)
    with pytest.raises(ProviderError, match="reports device"):
        load_gliner_model(
            _factory(log, model), "acme/gliner", {}, _CUDA, provider_name="test.gliner"
        )


def test_a_native_load_fault_becomes_a_typed_error_carrying_no_native_message(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """ADR-0034: a native fault is typed, and its message (which can echo an input) is not leaked."""

    class _Boom:
        @staticmethod
        def from_pretrained(_model_id: str, **_kw: Any) -> Any:
            msg = "SECRET-CONTENT in the checkpoint path"
            raise RuntimeError(msg)

    with pytest.raises(ProviderError) as exc_info:
        load_gliner_model(
            lambda: _Boom, "acme/gliner", {}, _CPU, provider_name="test.gliner"
        )
    assert "SECRET-CONTENT" not in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)


def test_a_missing_dependency_import_error_is_never_relabelled(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """The install hint must survive: an ImportError from the factory is not a provider failure."""

    def _no_dep() -> Any:
        msg = "Install it with: pip install latence-ner-gliner"
        raise ImportError(msg)

    with pytest.raises(ImportError, match="pip install latence-ner-gliner"):
        load_gliner_model(_no_dep, "acme/gliner", {}, _CPU, provider_name="test.gliner")


def test_the_caller_s_typed_error_policy_is_applied_around_the_load_only(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """A supplied ``guard`` wraps the native load — and NOT the backbone import.

    That distinction is the whole reason ``guard`` is a parameter rather than something the seam
    hard-codes: an adapter's ``config_exceptions`` policy must decide "bad model id" versus
    "provider failure", while a missing optional dependency stays the operator's install hint.
    """
    import contextlib

    @contextlib.contextmanager
    def guard(op: str) -> Iterator[None]:
        try:
            yield
        except ValueError as exc:
            msg = f"test.p: config error in {op!r}"
            raise ConfigError(msg) from exc

    class _BadId:
        @staticmethod
        def from_pretrained(_model_id: str, **_kw: Any) -> Any:
            msg = "no such checkpoint"
            raise ValueError(msg)

    with pytest.raises(ConfigError, match="load_model"):
        load_transformers_model(
            lambda: _BadId, "nope", {}, _CPU, provider_name="test.p", guard=guard
        )

    def _no_dep() -> Any:
        msg = "pip install something"
        raise ImportError(msg)

    with pytest.raises(ImportError):
        load_transformers_model(
            _no_dep, "nope", {}, _CPU, provider_name="test.p", guard=guard
        )


# --- the gliner2 family -----------------------------------------------------


def test_gliner2_knobs_key_off_the_resolved_device_not_a_cuda_probe(
    fake_torch: types.ModuleType, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION (a): ``device: cpu`` on a CUDA box is HONOURED — no fp16, no move to cuda.

    The defect: ``fused_entity_relation.gliner2`` resolved ``quantize``/``compile`` against a bare
    ``cuda_available()`` probe and then called ``ensure_gliner2_on_cuda`` off the same probe, so an
    operator's ``device: cpu`` was silently overridden on any GPU host. The seam now takes the
    RESOLVED decision, so the probe cannot override it — pinned here with ``LATENCE_CUDA=1`` so the
    probe would say cuda if anything still consulted it.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    model = _FakeModel(log)
    load_gliner2_model(
        _factory(log, model, family="gliner2"),
        "acme/gliner2",
        {},
        _CPU,
        provider_name="test.gliner2",
    )
    assert model.load_kwargs == {}, "cpu must not resolve quantize/compile on (fp16 on CPU is a hang)"
    assert "to-device:cuda" not in log, "an explicit device: cpu must never be moved to cuda"
    assert model.device == "cpu"


def test_gliner2_on_cuda_resolves_the_native_knobs_and_places(
    fake_torch: types.ModuleType, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of (a): a RESOLVED cuda decision does turn the native knobs on and place."""
    monkeypatch.setenv("LATENCE_CUDA", "0")  # the probe says NO cuda; the decision says cuda
    model = _FakeModel(log)
    load_gliner2_model(
        _factory(log, model, family="gliner2"),
        "acme/gliner2",
        {},
        _CUDA,
        provider_name="test.gliner2",
    )
    assert model.load_kwargs == {"quantize": True, "compile": True}
    assert model.device == "cuda"


def test_gliner2_caps_intra_op_threads_before_importing_the_backbone(
    fake_torch: types.ModuleType, log: list[str], cores: Any
) -> None:
    """REGRESSION (b): the many-core oversubscription guard applies to the gliner2 family too.

    ``fused_entity_relation.gliner2`` had it and ``redaction.gliner2`` did not — yet the blessed
    stack runs both mdeberta backbones in ONE process on ONE many-core host, and the spin is a
    process-wide torch thread pool: protecting one of the two protects neither. The cap must also
    land BEFORE the backbone import, because capping threads is what first imports torch.
    """
    cores(_MANY_CORES)  # the many-core host this guard exists FOR — and it makes the 8 mean 8
    model = _FakeModel(log)
    load_gliner2_model(
        _factory(log, model, family="gliner2"),
        "acme/gliner2",
        {},
        _CPU,
        provider_name="test.gliner2",
    )
    assert "threads=8" in log
    assert log.index("threads=8") < log.index("import:gliner2:jit_live=False")


def test_the_intra_op_cap_never_exceeds_the_host_core_count(
    fake_torch: types.ModuleType, log: list[str], cores: Any
) -> None:
    """The other half of the contract, and the reason the two tests above pin the count.

    Asking torch for 8 intra-op threads on a 4-vCPU box would itself be the oversubscription this
    guard exists to prevent, so the clamp is behaviour worth observing through the seam — not a
    detail to be worked around by restating ``min(8, os.cpu_count())`` back at the implementation.
    """
    cores(4)  # a small host — the GitHub ``ubuntu-latest`` runner shape
    load_gliner2_model(
        _factory(log, _FakeModel(log), family="gliner2"),
        "acme/gliner2",
        {},
        _CPU,
        provider_name="test.gliner2",
    )
    assert "threads=4" in log
    assert "threads=8" not in log


def test_intra_op_thread_cap_is_configurable_and_can_be_opted_out(
    fake_torch: types.ModuleType, log: list[str], cores: Any
) -> None:
    """``intra_op_threads`` still belongs to the Provider's config — only the SEQUENCING moved."""
    cores(_MANY_CORES)
    load_gliner2_model(
        _factory(log, _FakeModel(log), family="gliner2"),
        "acme/gliner2",
        {"intra_op_threads": 3},
        _CPU,
        provider_name="test.gliner2",
    )
    assert "threads=3" in log

    off: list[str] = []
    load_gliner2_model(
        _factory(off, _FakeModel(off), family="gliner2"),
        "acme/gliner2",
        {"intra_op_threads": 0},
        _CPU,
        provider_name="test.gliner2",
    )
    assert not [e for e in off if e.startswith("threads=")]


def test_gliner2_optimized_load_degrades_to_plain_never_crashes(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """A quantize/compile fault costs throughput, never the run — the W9 guard, through the seam."""
    model = _FakeModel(log)

    class _RejectsKnobs:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            if kwargs:
                msg = "this gliner2 build does not accept quantize/compile"
                raise TypeError(msg)
            log.append(f"plain-load:{model_id}")
            return model

    out = load_gliner2_model(
        lambda: _RejectsKnobs, "acme/gliner2", {}, _CUDA, provider_name="test.gliner2"
    )
    assert out is model
    assert "plain-load:acme/gliner2" in log


# --- the transformers family ------------------------------------------------


def test_transformers_load_spreads_perf_kwargs_and_places(
    fake_torch: types.ModuleType, log: list[str]
) -> None:
    """REGRESSION (c / the still-open OCR bug): bf16 + sdpa AND a verified placement.

    ``sparse.splade`` loaded bare fp32 with no placement at all; the OCR parsers loaded with perf
    kwargs but placed with a bare ``model.to(device)``. One entry point closes both.
    """
    model = _FakeModel(log)
    load_transformers_model(
        _factory(log, model, family="transformers"),
        "acme/mlm",
        {},
        _CUDA,
        provider_name="test.tf",
    )
    assert repr(model.load_kwargs["torch_dtype"]) == "torch.bfloat16"
    assert model.load_kwargs["attn_implementation"] == "sdpa"
    assert model.device == "cuda"


def test_transformers_fa2_load_falls_back_to_sdpa(
    fake_torch: types.ModuleType, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FA2 load that raises is retried once on sdpa — never a crash."""
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    model = _FakeModel(log)

    class _NoFa2:
        @staticmethod
        def from_pretrained(_model_id: str, **kwargs: Any) -> _FakeModel:
            if kwargs.get("attn_implementation") == "flash_attention_2":
                msg = "no FA2 kernel for this arch"
                raise ValueError(msg)
            model.load_kwargs = dict(kwargs)
            return model

    load_transformers_model(
        lambda: _NoFa2,
        "acme/vlm",
        {"flash_attention": True},
        _CUDA,
        provider_name="test.tf",
    )
    assert model.load_kwargs["attn_implementation"] == "sdpa"
