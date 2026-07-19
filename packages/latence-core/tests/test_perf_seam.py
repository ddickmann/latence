"""The W8 performance seam — offline unit tests (no GPU, no torch, no flash_attn).

These pin the perf seam's *decisions* against device/config, torch-free (the offline env has neither
torch nor flash_attn installed — see ``verify-local``'s CPU-first discipline, ADR-0016). The seam
resolves dtype/attn as names/strings without importing torch, materialises a real ``torch.dtype``
only where torch is present (here it stays the name string, still a valid ``from_pretrained`` kwarg),
and guards ``torch.compile`` so a missing/failing compile degrades to the eager model — never a crash.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.providers.perf import (
    ATTN_FA2,
    ATTN_SDPA,
    DEFAULT_GLINER_MAX_LEN,
    DEFAULT_INTRA_OP_THREADS,
    DTYPE_BF16,
    DTYPE_FP32,
    TORCH_JIT_ENV,
    apply_gliner_max_len,
    cap_intra_op_threads,
    ensure_gliner2_on_cuda,
    flash_attn_available,
    is_compile_backend_failure,
    load_gliner2_optimized,
    load_perf_kwargs,
    load_then_cast,
    load_with_fa2_fallback,
    maybe_compile,
    neutralize_torch_jit,
    resolve_attn_impl,
    resolve_dtype_name,
    resolve_gliner2_knob,
    resolve_intra_op_threads,
    resolve_max_len,
)

# --- W12 model token-window bound ----------------------------------------------------------------


def test_resolve_max_len_default_is_768() -> None:
    assert resolve_max_len({}) == DEFAULT_GLINER_MAX_LEN == 768


def test_resolve_max_len_config_override_and_clamp() -> None:
    assert resolve_max_len({"max_len": 640}) == 640
    assert resolve_max_len({"max_len": 0}) == 1  # clamped to a >= 1 window, never empty/negative
    assert resolve_max_len({"max_len": -5}) == 1


def test_apply_gliner_max_len_sets_config_max_len() -> None:
    model = types.SimpleNamespace(config=types.SimpleNamespace(max_len=384))
    apply_gliner_max_len(model, 768)
    assert model.config.max_len == 768


def test_apply_gliner_max_len_is_a_guarded_noop_without_config() -> None:
    # The guard must never crash on a model/config that has no settable max_len (a fake, a future
    # arch): it is best-effort. No config at all, and a config lacking max_len, both no-op cleanly.
    apply_gliner_max_len(types.SimpleNamespace(), 768)  # no .config → no-op, no raise
    frozen = types.SimpleNamespace(config=types.SimpleNamespace())  # config, but no max_len
    apply_gliner_max_len(frozen, 768)
    assert not hasattr(frozen.config, "max_len")  # left unchanged, no crash


# --- dtype resolution: bf16 on cuda, fp32 on cpu, config override --------------------------------


def test_resolve_dtype_bf16_on_cuda_device() -> None:
    """An explicit cuda device resolves the default (auto) dtype to bf16 (the ~2× win)."""
    assert resolve_dtype_name("cuda", {}) == DTYPE_BF16


def test_resolve_dtype_fp32_on_cpu_device() -> None:
    """An explicit cpu device resolves the default (auto) dtype to fp32."""
    assert resolve_dtype_name("cpu", {}) == DTYPE_FP32


def test_resolve_dtype_none_device_follows_cuda_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=None (a provider that lets the lib auto-place) follows the LATENCE_CUDA probe."""
    monkeypatch.setenv("LATENCE_CUDA", "1")
    assert resolve_dtype_name(None, {}) == DTYPE_BF16
    monkeypatch.setenv("LATENCE_CUDA", "0")
    assert resolve_dtype_name(None, {}) == DTYPE_FP32


def test_resolve_dtype_config_override_pins_regardless_of_device() -> None:
    """An explicit ``dtype`` config pins the load dtype, overriding the device default."""
    assert resolve_dtype_name("cuda", {"dtype": "float32"}) == DTYPE_FP32
    assert resolve_dtype_name("cpu", {"dtype": "bfloat16"}) == DTYPE_BF16


def test_resolve_dtype_invalid_config_raises() -> None:
    """An unrecognised ``dtype`` value is a hard error (the caller maps it to a ConfigError)."""
    with pytest.raises(ValueError, match="invalid dtype"):
        resolve_dtype_name("cuda", {"dtype": "float8_nonsense"})


# --- attn resolution: sdpa default, FA2 only when flash_attn importable ---------------------------


def test_resolve_attn_defaults_to_sdpa() -> None:
    """With no config, the attention impl is sdpa (the always-available default)."""
    assert resolve_attn_impl({}) == ATTN_SDPA


def test_resolve_attn_fa2_requested_but_unavailable_degrades_to_sdpa() -> None:
    """FA2 requested on a host without ``flash_attn`` (this env) degrades gracefully to sdpa."""
    # flash_attn is NOT installed in the offline env, so a FA2 request must fall back to sdpa.
    assert flash_attn_available() is False
    assert resolve_attn_impl({"attn_implementation": ATTN_FA2}) == ATTN_SDPA
    assert resolve_attn_impl({"flash_attention": True}) == ATTN_SDPA


def test_resolve_attn_fa2_selected_when_flash_attn_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``flash_attn`` IS importable (monkeypatched present), a FA2 request selects FA2."""
    import importlib.machinery

    mod = types.ModuleType("flash_attn")
    # ``importlib.util.find_spec`` reads ``__spec__`` off an already-imported module; give the fake
    # a real spec so the probe sees it as importable (a bare ModuleType has ``__spec__ = None``).
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    monkeypatch.setitem(sys.modules, "flash_attn", mod)
    assert flash_attn_available() is True
    assert resolve_attn_impl({"attn_implementation": ATTN_FA2}) == ATTN_FA2


def test_resolve_attn_unknown_value_is_sdpa() -> None:
    """An unknown attn value is the safe sdpa default (a typo never fails the load)."""
    assert resolve_attn_impl({"attn_implementation": "made_up"}) == ATTN_SDPA


# --- load_perf_kwargs: the from_pretrained dict --------------------------------------------------


def test_load_perf_kwargs_shape_bf16_sdpa_on_cuda() -> None:
    """The dict spread into ``from_pretrained`` carries torch_dtype + attn_implementation."""
    kwargs = load_perf_kwargs("cuda", {})
    assert set(kwargs) == {"torch_dtype", "attn_implementation"}
    # torch is not installed offline, so torch_dtype degrades to the dtype NAME string (still a
    # valid from_pretrained kwarg) — asserting exactly the torch-free fallback contract.
    assert kwargs["torch_dtype"] == DTYPE_BF16
    assert kwargs["attn_implementation"] == ATTN_SDPA


def test_load_perf_kwargs_fp32_on_cpu() -> None:
    kwargs = load_perf_kwargs("cpu", {})
    assert kwargs["torch_dtype"] == DTYPE_FP32


def test_load_perf_kwargs_materialises_real_torch_dtype_when_torch_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where torch IS importable, torch_dtype is the real ``torch.dtype`` object, not the name."""
    fake_torch = types.ModuleType("torch")
    sentinel_bf16 = object()
    fake_torch.bfloat16 = sentinel_bf16  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    kwargs = load_perf_kwargs("cuda", {})
    assert kwargs["torch_dtype"] is sentinel_bf16


# --- maybe_compile: no-op off / off-on-failure, never crashes -------------------------------------


def test_maybe_compile_returns_model_unchanged_when_off() -> None:
    """compile off (default) → the exact same model object back (a pure no-op)."""
    model = object()
    assert maybe_compile(model, {}) is model
    assert maybe_compile(model, {"compile": False}) is model


def test_maybe_compile_returns_model_when_torch_compile_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """compile on but torch.compile raises → the UNCOMPILED model back, logged, never a crash."""
    fake_torch = types.ModuleType("torch")

    def _boom(_model: Any, **_kwargs: Any) -> Any:
        msg = "compile/warmup blew up"
        raise RuntimeError(msg)

    fake_torch.compile = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = object()
    assert maybe_compile(model, {"compile": True}) is model


def test_maybe_compile_returns_model_when_torch_missing() -> None:
    """compile on but torch not importable (offline env) → the uncompiled model, never a crash."""
    # torch is not installed in the offline env; a compile request degrades to the eager model.
    model = object()
    assert maybe_compile(model, {"compile": True}) is model


def test_maybe_compile_wraps_when_compile_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """compile on and torch.compile succeeds → the compiled (wrapped) module is returned."""
    fake_torch = types.ModuleType("torch")
    wrapped = object()

    def _compile(_model: Any, **_kwargs: Any) -> Any:
        return wrapped

    fake_torch.compile = _compile  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert maybe_compile(object(), {"compile": True}) is wrapped


# --- load_with_fa2_fallback: FA2 load raises → sdpa retry -----------------------------------------


def test_load_with_fa2_fallback_retries_on_sdpa_when_fa2_load_raises() -> None:
    """A FA2 load that raises is retried ONCE on sdpa (graceful degradation, never a crash)."""
    calls: list[dict[str, Any]] = []

    def loader(model_id: str, **kwargs: Any) -> str:
        calls.append(dict(kwargs))
        if kwargs.get("attn_implementation") == ATTN_FA2:
            msg = "no FA2 kernel for this arch"
            raise RuntimeError(msg)
        return f"loaded {model_id}"

    perf = {"torch_dtype": DTYPE_BF16, "attn_implementation": ATTN_FA2}
    out = load_with_fa2_fallback(loader, "some/model", perf)
    assert out == "loaded some/model"
    # Attempted FA2 first, then retried on sdpa.
    assert [c["attn_implementation"] for c in calls] == [ATTN_FA2, ATTN_SDPA]


def test_load_with_fa2_fallback_propagates_non_fa2_error() -> None:
    """A non-FA2 (sdpa) load error is NOT swallowed — it propagates (mapped by the adapter guard)."""

    def loader(_model_id: str, **_kwargs: Any) -> str:
        msg = "genuinely bad model id"
        raise RuntimeError(msg)

    perf = {"torch_dtype": DTYPE_FP32, "attn_implementation": ATTN_SDPA}
    with pytest.raises(RuntimeError, match="genuinely bad model id"):
        load_with_fa2_fallback(loader, "bad/model", perf)


# --- load_then_cast: plain load + .to(dtype) for libs that reject perf kwargs (gliner) -----------


class _FakeModel:
    """A fake model: records the loader kwargs it was built with and the dtype it was cast to."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.load_kwargs = dict(kwargs)
        self.cast_dtype: Any = None

    def to(self, dtype: Any) -> _FakeModel:
        self.cast_dtype = dtype
        return self


def _rejecting_loader(model_id: str, **kwargs: Any) -> _FakeModel:
    """A loader FAITHFUL to gliner 0.2.27: it REJECTS perf kwargs (like the real from_pretrained)."""
    for bad in ("torch_dtype", "attn_implementation"):
        if bad in kwargs:
            msg = f"unexpected keyword argument '{bad}'"
            raise TypeError(msg)
    return _FakeModel(model_id, **kwargs)


def test_load_then_cast_loads_plain_and_casts_bf16_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """On cuda with torch present, load_then_cast loads PLAIN (no perf kwargs) then .to(bf16)."""
    fake_torch = types.ModuleType("torch")
    bf16 = object()
    fake_torch.bfloat16 = bf16  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = load_then_cast(_rejecting_loader, "urchade/gliner", {}, "cuda")
    assert model.load_kwargs == {}  # loaded PLAIN — no torch_dtype/attn_implementation (no crash)
    assert model.cast_dtype is bf16  # cast applied via .to()


def test_load_then_cast_no_cast_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """On cpu (fp32 default) load_then_cast does NOT cast (bf16 CPU is not a universal win)."""
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    model = load_then_cast(_rejecting_loader, "urchade/gliner", {}, "cpu")
    assert model.cast_dtype is None


def test_load_then_cast_skips_cast_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """With torch NOT importable (offline env), a cuda host loads plain and skips the cast, no crash."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setenv("LATENCE_CUDA", "1")
    model = load_then_cast(_rejecting_loader, "urchade/gliner", {}, None)
    assert model.load_kwargs == {}  # loaded plain
    assert model.cast_dtype is None  # no real dtype to cast to → skipped, never a crash


def test_load_then_cast_never_passes_perf_kwargs_to_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_then_cast must NEVER spread perf kwargs into the loader (the gliner-crash guard)."""
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # A rejecting loader would raise TypeError if perf kwargs were passed — assert it does NOT raise.
    load_then_cast(_rejecting_loader, "urchade/gliner", {"dtype": "bfloat16"}, "cuda")


# --- W9 gliner2 native quantize/compile seam ---------------------------------------------------


def test_resolve_gliner2_knob_auto_follows_device() -> None:
    # 'auto' (the default) → True on cuda, False on cpu.
    assert resolve_gliner2_knob("quantize", "cuda", {}) is True
    assert resolve_gliner2_knob("compile", "cpu", {}) is False


def test_resolve_gliner2_knob_bool_and_string_overrides() -> None:
    assert resolve_gliner2_knob("quantize", "cpu", {"quantize": True}) is True
    assert resolve_gliner2_knob("compile", "cuda", {"compile": False}) is False
    assert resolve_gliner2_knob("quantize", "cpu", {"quantize": "yes"}) is True
    assert resolve_gliner2_knob("compile", "cuda", {"compile": "0"}) is False


def test_resolve_gliner2_knob_invalid_raises() -> None:
    with pytest.raises(ValueError, match="invalid quantize"):
        resolve_gliner2_knob("quantize", "cpu", {"quantize": "sometimes"})


def test_load_gliner2_optimized_cpu_default_loads_plain() -> None:
    # On cpu both knobs resolve False → the loader is called PLAIN (no quantize/compile kwargs).
    seen: dict[str, Any] = {}

    def loader(model_id: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        return f"model:{model_id}"

    out = load_gliner2_optimized(loader, "m", "cpu", {})
    assert out == "model:m"
    assert seen == {}


def test_load_gliner2_optimized_cuda_passes_native_kwargs() -> None:
    seen: dict[str, Any] = {}

    def loader(model_id: str, **kwargs: Any) -> str:  # noqa: ARG001
        seen.update(kwargs)
        return "ok"

    load_gliner2_optimized(loader, "m", "cuda", {})
    assert seen == {"quantize": True, "compile": True}


def test_load_gliner2_optimized_fault_degrades_to_plain() -> None:
    # An optimized load that raises degrades to a PLAIN load (never a crash).
    calls: list[dict[str, Any]] = []

    def loader(model_id: str, **kwargs: Any) -> str:  # noqa: ARG001
        calls.append(kwargs)
        if kwargs:  # the optimized attempt (quantize/compile) fails
            msg = "torch/CUDA rejects fp16"
            raise RuntimeError(msg)
        return "plain"

    out = load_gliner2_optimized(loader, "m", "cuda", {})
    assert out == "plain"
    assert calls[0] == {"quantize": True, "compile": True}  # optimized attempt
    assert calls[1] == {}  # plain fallback


class _FakeBackendCompilerFailed(RuntimeError):
    """A stand-in for torch._dynamo.exc.BackendCompilerFailed (matched by class name).

    The predicate matches on the concrete class NAME, so the ``__name__`` is set to torch's exact
    class name (the leading-underscore module-private test alias is only for the local reference).
    """


_FakeBackendCompilerFailed.__name__ = "BackendCompilerFailed"


def test_is_compile_backend_failure_matches_by_name() -> None:
    assert is_compile_backend_failure(_FakeBackendCompilerFailed("boom")) is True


def test_is_compile_backend_failure_matches_wrapped_cause() -> None:
    # A compile fault re-raised under a generic error (e.g. via the adapter guard) is still caught.
    try:
        try:
            raise _FakeBackendCompilerFailed("inner")
        except _FakeBackendCompilerFailed as inner:
            raise RuntimeError("wrapped by guard") from inner
    except RuntimeError as outer:
        assert is_compile_backend_failure(outer) is True


def test_is_compile_backend_failure_matches_by_module_prefix() -> None:
    # A concrete class defined in torch's compile stack is caught even if its name is not listed.
    class _InductorThing(RuntimeError):
        pass

    _InductorThing.__module__ = "torch._inductor.compile_fx"
    assert is_compile_backend_failure(_InductorThing("y")) is True


def test_is_compile_backend_failure_false_for_ordinary_error() -> None:
    # A genuine (non-compile) runtime fault is NOT swallowed by the guard.
    assert is_compile_backend_failure(RuntimeError("cuda oom")) is False
    assert is_compile_backend_failure(ValueError("bad shape")) is False


# --- W9-scaling: intra-op thread cap + fp16 cuda placement -------------------------------------


def test_resolve_intra_op_threads_default_and_clamp() -> None:
    import os

    cores = os.cpu_count() or DEFAULT_INTRA_OP_THREADS
    # Default 8, clamped to the core count (never more threads than cores).
    assert resolve_intra_op_threads({}) == min(DEFAULT_INTRA_OP_THREADS, cores)
    # A tiny explicit cap is honoured as-is.
    assert resolve_intra_op_threads({"intra_op_threads": 2}) == 2
    # A huge explicit cap is clamped to cores.
    assert resolve_intra_op_threads({"intra_op_threads": 100_000}) == cores


def test_resolve_intra_op_threads_disable_and_invalid() -> None:
    # <= 0 disables the cap (None = leave torch's default pool).
    assert resolve_intra_op_threads({"intra_op_threads": 0}) is None
    assert resolve_intra_op_threads({"intra_op_threads": -1}) is None
    for bad in ("lots", 3.5, True):
        with pytest.raises(ValueError, match="invalid intra_op_threads"):
            resolve_intra_op_threads({"intra_op_threads": bad})


def test_cap_intra_op_threads_applies_via_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cap_intra_op_threads(8) == 8
    assert calls == [8]


def test_cap_intra_op_threads_none_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert cap_intra_op_threads(None) is None
    assert calls == []


def test_cap_intra_op_threads_torch_free_is_guarded_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # No torch installed → a logged no-op returning None (best-effort, never a crash).
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cap_intra_op_threads(8) is None


class _MovableModel:
    """A model exposing ``.to`` + ``device`` — mirrors what ensure_gliner2_on_cuda moves/reads."""

    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.device = "cpu"

    def to(self, target: str) -> _MovableModel:
        self.to_calls.append(target)
        self.device = target
        return self


def test_ensure_gliner2_on_cuda_moves_and_detects() -> None:
    model = _MovableModel()
    assert ensure_gliner2_on_cuda(model, cuda=True) == "cuda"
    assert model.to_calls == ["cuda"]


def test_ensure_gliner2_on_cuda_cpu_does_not_move() -> None:
    model = _MovableModel()
    # cuda=False: no move, but the resident device is still read back (diagnostic).
    assert ensure_gliner2_on_cuda(model, cuda=False) == "cpu"
    assert model.to_calls == []


def test_ensure_gliner2_on_cuda_guarded_on_bare_model() -> None:
    # A model with no ``.to`` / no device (a test double, a frozen wrapper) never crashes.
    assert ensure_gliner2_on_cuda(object(), cuda=True) is None


# --- W9-jit: TorchScript JIT neutralization (the Blackwell/CUDA-13 nvrtc crash) -------------------


class _FakeScriptFunction:
    """What ``torch.jit.script`` returns while the JIT is ENABLED — and what dies on Blackwell.

    A real ``torch.jit.script`` builds a ScriptFunction whose fused kernels are compiled at CALL
    time by nvrtc; on the CUDA-13 / Blackwell pod that compile fails in native code with
    ``failed to open libnvrtc-builtins.so.13.0``. The fake raises exactly there — so a fake that
    "accepts" a JIT the real box rejects is impossible: with the JIT on, the forward dies.
    """

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        msg = (
            "The following operation failed in the TorchScript interpreter: Compilation error: "
            "nvrtc: failed to open libnvrtc-builtins.so.13.0"
        )
        raise RuntimeError(msg)


class _FakeEnabled:
    """Mirrors ``torch.jit._state._enabled`` — the proxy torch parses ``PYTORCH_JIT`` into."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake ``torch`` that mirrors the REAL TorchScript JIT switch semantics.

    Real torch parses ``PYTORCH_JIT`` **once, at import**, into ``torch.jit._state._enabled.enabled``
    and ``torch.jit.script`` then returns the function **unchanged** when that flag is False, or a
    ScriptFunction when it is True. ``torch.jit._state.disable()`` flips the flag at runtime. The
    fake reproduces all three, so the seam is exercised against the real contract.
    """
    import os

    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    state = types.ModuleType("torch.jit._state")
    enabled = _FakeEnabled(os.environ.get("PYTORCH_JIT", "1") != "0")
    state._enabled = enabled  # type: ignore[attr-defined]
    state.disable = lambda: setattr(enabled, "enabled", False)  # type: ignore[attr-defined]

    def script(fn: Any) -> Any:
        # The real torch.jit.script: a pass-through while the JIT is disabled, a ScriptFunction
        # otherwise. This IS the neutralization the seam relies on.
        return _FakeScriptFunction(fn) if enabled.enabled else fn

    jit.script = script  # type: ignore[attr-defined]
    jit._state = state  # type: ignore[attr-defined]
    torch.jit = jit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.jit", jit)
    return torch


def _build_relative_position(query_size: int, key_size: int) -> int:
    """Stands in for transformers' ``@torch.jit.script`` mdeberta relative-position helper."""
    return query_size - key_size


def test_torch_jit_enabled_reproduces_the_blackwell_nvrtc_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fault this seam exists for: with the JIT ON, the scripted helper dies in nvrtc."""
    monkeypatch.setenv("PYTORCH_JIT", "1")
    torch = _install_fake_torch(monkeypatch)
    scripted = torch.jit.script(_build_relative_position)
    with pytest.raises(RuntimeError, match="libnvrtc-builtins.so.13.0"):
        scripted(5, 2)


def test_neutralize_torch_jit_makes_scripting_a_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the seam runs, ``torch.jit.script`` returns the eager function — same value, no nvrtc.

    Asserts BEHAVIOUR (scripting is neutralized and the helper still computes the same answer), not
    a flag recording intent.
    """
    import os

    monkeypatch.setenv("PYTORCH_JIT", "1")
    torch = _install_fake_torch(monkeypatch)
    assert neutralize_torch_jit() is True
    assert os.environ[TORCH_JIT_ENV] == "0"
    scripted = torch.jit.script(_build_relative_position)
    assert scripted is _build_relative_position  # no ScriptFunction, so nothing reaches nvrtc
    assert scripted(5, 2) == _build_relative_position(5, 2) == 3  # byte-identical output


def test_neutralize_torch_jit_without_torch_still_sets_the_env_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch-free host: the env switch is still set (a later ``import torch`` comes up JIT-free)."""
    import os

    monkeypatch.setenv("PYTORCH_JIT", "1")
    monkeypatch.setitem(sys.modules, "torch", None)
    assert neutralize_torch_jit() is False  # nothing to disable in-process
    assert os.environ[TORCH_JIT_ENV] == "0"


def test_neutralize_torch_jit_is_guarded_on_a_torch_without_jit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torch that exposes no ``jit._state`` (a stub, a future rename) is a logged no-op."""
    monkeypatch.setenv("PYTORCH_JIT", "1")
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    assert neutralize_torch_jit() is False  # never raises


def _mdeberta_loader(torch: Any) -> Any:
    """A loader that SCRIPTS its relative-position helper at load and calls it in the forward.

    Scope (audit R1 / #190-4): this pins the *loader's* contract — a model loaded through the seam
    must come back un-scripted. It is deliberately NOT the full mdeberta shape: real transformers
    scripts at MODULE IMPORT (``from gliner2 import GLiNER2``), which happens before any of these
    loaders is entered, so nothing here can speak to that ordering. That property is covered by
    ``test_perf_jit_import_ordering.py`` and by the structural guard
    ``test_perf_fp32_regression_guard.py::
    test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone``.
    """

    def loader(model_id: str, **kwargs: Any) -> Any:
        rel_pos = torch.jit.script(_build_relative_position)
        return types.SimpleNamespace(
            model_id=model_id,
            kwargs=kwargs,
            forward=lambda q, k: rel_pos(q, k),
        )

    return loader


def test_load_then_cast_neutralizes_the_jit_for_the_gliner_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION (#190): the shared gliner-library load path must not leave a scripted forward.

    Pre-fix this raises ``RuntimeError: ... libnvrtc-builtins.so.13.0`` from ``model.forward``.
    """
    monkeypatch.setenv("PYTORCH_JIT", "1")
    torch = _install_fake_torch(monkeypatch)
    model = load_then_cast(_mdeberta_loader(torch), "mdeberta-model", {}, "cpu")
    assert model.forward(5, 2) == 3  # the forward runs eagerly, same value, no nvrtc


def test_load_gliner2_optimized_neutralizes_the_jit(monkeypatch: pytest.MonkeyPatch) -> None:
    """REGRESSION (#190): the shared gliner2 load path must not leave a scripted forward.

    Pre-fix this raises ``RuntimeError: ... libnvrtc-builtins.so.13.0`` from ``model.forward``.
    """
    monkeypatch.setenv("PYTORCH_JIT", "1")
    monkeypatch.setenv("LATENCE_CUDA", "0")
    torch = _install_fake_torch(monkeypatch)
    model = load_gliner2_optimized(_mdeberta_loader(torch), "gliner2-model", None, {})
    assert model.forward(5, 2) == 3


def test_load_gliner2_optimized_neutralizes_the_jit_on_the_optimized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The neutralization is unconditional — it applies to the quantize/compile load too, not only
    the plain one (a Blackwell box is exactly where quantize=True is on)."""
    monkeypatch.setenv("PYTORCH_JIT", "1")
    torch = _install_fake_torch(monkeypatch)
    model = load_gliner2_optimized(
        _mdeberta_loader(torch), "gliner2-model", "cuda", {"quantize": True}
    )
    assert model.kwargs == {"quantize": True, "compile": True}
    assert model.forward(5, 2) == 3
