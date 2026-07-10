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
    DTYPE_BF16,
    DTYPE_FP32,
    flash_attn_available,
    load_perf_kwargs,
    load_then_cast,
    load_with_fa2_fallback,
    maybe_compile,
    resolve_attn_impl,
    resolve_dtype_name,
)

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
