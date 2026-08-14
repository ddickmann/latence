"""TorchRuntime — the owner of process-global torch state — and the ST load composite (Wave 3-H).

Two properties, both previously unowned:

1. **Global torch state is recorded and restorable.** ``neutralize_torch_jit`` (the
   ``PYTORCH_JIT`` env switch + ``torch.jit._state``), ``cap_intra_op_threads``
   (``torch.set_num_threads``) and ``maybe_compile`` (TorchDynamo caches) mutate state that
   outlives any one Provider. FINDINGS F7 (an order-dependent ``prims.iota`` flake in the
   5,000-test single-interpreter roster run) and F9 (an interpreter-shutdown segfault with torch
   loaded) are symptoms of those mutations having no owner. The three functions now delegate to
   the current :class:`~latence_core.providers.perf.TorchRuntime`, which RECORDS the prior value
   of everything it changes; :func:`~latence_core.providers.perf.scoped_torch_runtime` restores on
   exit so a real-weights test boundary or a Stage boundary can scope the mutations.

2. **The ST family loads through a composite**, like every other learned family (C4):
   :func:`~latence_core.providers.perf.load_sentence_transformer` owns the ordered sequence
   (skip check → thread cap → deferred backbone import → perf kwargs → FA2→sdpa fallback →
   eager-model inspection → guarded compile) that ``embedding.sentence_transformers`` previously
   assembled piecemeal.

Anti-false-green, same discipline as ``test_perf_load_choreography``: the fake torch mirrors the
real levers (a jit state whose ``disable()``/``enable()`` really flip the flag; ``dtype`` objects,
not strings; a readable/writable thread count; a ``torch._dynamo.reset`` that records the call),
so every assertion is on an observed effect, never on a flag recording intent.
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
    cap_intra_op_threads,
    current_torch_runtime,
    load_sentence_transformer,
    maybe_compile,
    neutralize_torch_jit,
    scoped_torch_runtime,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# --- a fake torch that mirrors every lever the runtime records/restores ----------------------


class _Dtype:
    """A real OBJECT dtype (never a str — the seam treats a str dtype as the torch-free case)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


class _Enabled:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled


class _JitState:
    """Mirrors ``torch.jit._state``: ``disable()``/``enable()`` really flip the proxy flag."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = _Enabled(enabled=enabled)

    def disable(self) -> None:
        self._enabled.enabled = False

    def enable(self) -> None:
        self._enabled.enabled = True


def _make_fake_torch(log: list[str], *, threads: int = 4, jit_enabled: bool = True) -> Any:
    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    jit._state = _JitState(enabled=jit_enabled)  # type: ignore[attr-defined]
    torch.jit = jit  # type: ignore[attr-defined]
    torch.bfloat16 = _Dtype("bfloat16")  # type: ignore[attr-defined]
    torch.float32 = _Dtype("float32")  # type: ignore[attr-defined]
    state = {"threads": threads}

    def get_num_threads() -> int:
        return state["threads"]

    def set_num_threads(n: int) -> None:
        state["threads"] = n
        log.append(f"threads={n}")

    def compile_(model: Any, **_kw: Any) -> Any:
        log.append("compiled")
        return model

    torch.get_num_threads = get_num_threads  # type: ignore[attr-defined]
    torch.set_num_threads = set_num_threads  # type: ignore[attr-defined]
    torch.compile = compile_  # type: ignore[attr-defined]
    dynamo = types.ModuleType("torch._dynamo")
    dynamo.reset = lambda: log.append("dynamo-reset")  # type: ignore[attr-defined]
    torch._dynamo = dynamo  # type: ignore[attr-defined]
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False  # type: ignore[attr-defined]
    torch.cuda = cuda  # type: ignore[attr-defined]
    return torch


@pytest.fixture()
def log() -> list[str]:
    return []


@pytest.fixture()
def fake_torch(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> Any:
    torch = _make_fake_torch(log)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", torch._dynamo)  # noqa: SLF001
    monkeypatch.delenv(TORCH_JIT_ENV, raising=False)
    return torch


# --- record + restore -------------------------------------------------------------------------


def test_scoped_runtime_restores_the_jit_env_and_in_process_state(
    fake_torch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two JIT levers go back on exit: the env switch AND the runtime state (observed)."""
    monkeypatch.setenv(TORCH_JIT_ENV, "1")
    with scoped_torch_runtime() as runtime:
        assert neutralize_torch_jit() is True
        assert os.environ[TORCH_JIT_ENV] == "0"
        assert fake_torch.jit._state._enabled.enabled is False  # noqa: SLF001
        assert runtime.recorded == {"jit_env": "1", "jit_enabled": True}
    assert os.environ[TORCH_JIT_ENV] == "1"
    assert fake_torch.jit._state._enabled.enabled is True  # noqa: SLF001


def test_scoped_runtime_removes_an_env_switch_that_was_unset(
    fake_torch: Any,
) -> None:
    """A ``PYTORCH_JIT`` that did not exist before the scope does not exist after it."""
    assert TORCH_JIT_ENV not in os.environ
    with scoped_torch_runtime() as runtime:
        neutralize_torch_jit()
        assert os.environ[TORCH_JIT_ENV] == "0"
        assert runtime.recorded["jit_env"] is None  # None records "was unset"
    assert TORCH_JIT_ENV not in os.environ


def test_scoped_runtime_restores_the_thread_cap(fake_torch: Any, log: list[str]) -> None:
    """The intra-op pool goes back to its prior size — via a real ``set_num_threads`` call."""
    with scoped_torch_runtime() as runtime:
        assert cap_intra_op_threads(2) == 2
        assert fake_torch.get_num_threads() == 2
        assert runtime.recorded["num_threads"] == 4
    assert fake_torch.get_num_threads() == 4
    assert log == ["threads=2", "threads=4"]


def test_scoped_runtime_resets_dynamo_after_a_compile(fake_torch: Any, log: list[str]) -> None:
    """A compile inside the scope is followed by a ``torch._dynamo.reset()`` on exit."""
    model = object()
    with scoped_torch_runtime() as runtime:
        assert maybe_compile(model, {"compile": True}) is model  # the fake returns it unchanged
        assert runtime.recorded == {"dynamo": True}
    assert log == ["compiled", "dynamo-reset"]


def test_a_scope_that_mutated_nothing_restores_nothing(fake_torch: Any, log: list[str]) -> None:
    """No mutation inside the scope → no restore side effects on exit (compile off is a no-op)."""
    with scoped_torch_runtime() as runtime:
        maybe_compile(object(), {})  # off by default — must not touch dynamo
        assert runtime.recorded == {}
    assert log == []
    assert TORCH_JIT_ENV not in os.environ


def test_restore_is_idempotent(fake_torch: Any, log: list[str]) -> None:
    """The records are consumed: a second ``restore()`` is a no-op, never a re-apply."""
    with scoped_torch_runtime() as runtime:
        cap_intra_op_threads(2)
    assert log == ["threads=2", "threads=4"]
    runtime.restore()
    assert log == ["threads=2", "threads=4"]


def test_the_first_prior_value_wins_within_a_runtime(fake_torch: Any) -> None:
    """Two caps inside one scope restore to the value BEFORE the scope, not between the caps."""
    with scoped_torch_runtime() as runtime:
        cap_intra_op_threads(2)
        cap_intra_op_threads(3)
        assert runtime.recorded["num_threads"] == 4
    assert fake_torch.get_num_threads() == 4


def test_module_functions_delegate_to_the_innermost_scope(fake_torch: Any) -> None:
    """Nesting: each scope records and restores only its own mutations."""
    with scoped_torch_runtime() as outer:
        cap_intra_op_threads(3)
        with scoped_torch_runtime() as inner:
            assert current_torch_runtime() is inner
            cap_intra_op_threads(2)
            assert inner.recorded == {"num_threads": 3}  # the inner scope's prior is 3
        assert fake_torch.get_num_threads() == 3  # inner restored to the outer's value
        assert outer.recorded == {"num_threads": 4}
        assert current_torch_runtime() is outer
    assert fake_torch.get_num_threads() == 4


def test_restore_is_guarded_when_torch_disappears(
    fake_torch: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torch monkeypatched away between mutation and restore never crashes the restore."""
    with scoped_torch_runtime():
        cap_intra_op_threads(2)
        neutralize_torch_jit()
        maybe_compile(object(), {"compile": True})
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "torch._dynamo", None)
    # The env lever needs no torch, so it is still restored (removed — it was unset before).
    assert TORCH_JIT_ENV not in os.environ


def test_unscoped_calls_record_into_the_root_runtime(fake_torch: Any) -> None:
    """No scope → the root runtime records (diagnostics) but nothing is restored (old semantics)."""
    root = current_torch_runtime()
    before = root.recorded
    assert cap_intra_op_threads(2) == 2
    assert fake_torch.get_num_threads() == 2  # stays: the root runtime is never restored
    after = root.recorded
    assert "num_threads" in after
    # The root's FIRST recorded prior persists across the whole process, so on a re-entrant test
    # run the key may predate this test; assert the record exists rather than its exact value.
    assert before.keys() <= after.keys()


# --- the ST load composite --------------------------------------------------------------------


class _FakeST:
    """Stands in for ``SentenceTransformer``: records construction, exposes the width read."""

    def __init__(self, log: list[str], *, width: int = 6) -> None:
        self._log = log
        self._width = width
        self.constructed: list[tuple[str, str | None, dict[str, Any]]] = []

    def make_cls(self) -> Any:
        outer = self

        class _Cls:
            def __init__(
                self, model_id: str, device: str | None = None, model_kwargs: Any = None
            ) -> None:
                outer.constructed.append((model_id, device, dict(model_kwargs or {})))
                outer._log.append(f"st-load:{model_id}:{device}")  # noqa: SLF001
                self.width = outer._width  # noqa: SLF001

            def get_sentence_embedding_dimension(self) -> int:
                outer._log.append("width-read")  # noqa: SLF001
                return self.width

        return _Cls


_CUDA = DeviceDecision(device="cuda", skipped=False, reason="test: cuda")
_CPU = DeviceDecision(device="cpu", skipped=False, reason="test: cpu")
_SKIPPED = DeviceDecision(device=None, skipped=True, reason="no CUDA device is available (skipped)")


def _pin_cores(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """Pin the core count so ``threads=8`` is about the cap, not this machine (see choreography)."""
    monkeypatch.setattr(os, "cpu_count", lambda: n)


def test_st_load_runs_the_whole_sequence_in_order(
    fake_torch: Any, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One call: thread cap → import → perf-kwargs load → eager inspection → compile, in order."""
    _pin_cores(monkeypatch, 64)
    st = _FakeST(log)

    def factory() -> Any:
        log.append("import:st")
        return st.make_cls()

    out = load_sentence_transformer(
        factory,
        "acme/st",
        {"compile": True},
        _CUDA,
        provider_name="test.st",
        inspect_model=lambda _m: log.append("inspected"),
    )
    assert out is not None
    order = [
        log.index("threads=8"),
        log.index("import:st"),
        log.index("st-load:acme/st:cuda"),
        log.index("inspected"),
        log.index("compiled"),
    ]
    assert order == sorted(order), f"steps ran out of order: {log}"
    # The W8 perf kwargs thread through ST's model_kwargs: real dtype OBJECT + attn impl.
    (_mid, device, model_kwargs) = st.constructed[0]
    assert device == "cuda"
    assert repr(model_kwargs["torch_dtype"]) == "torch.bfloat16"
    assert model_kwargs["attn_implementation"] == "sdpa"


def test_st_load_does_not_touch_the_jit_levers(
    fake_torch: Any, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """DELIBERATE: no ``neutralize_torch_jit`` for the ST family (ModernBERT, no scripted helpers).

    The kill is a DeBERTa/nvrtc defence; pulling the process-wide ``PYTORCH_JIT=0`` lever from
    this family would add exactly the unowned cross-provider mutation Wave 3-H removes.
    """
    monkeypatch.delenv(TORCH_JIT_ENV, raising=False)
    st = _FakeST(log)
    load_sentence_transformer(
        st.make_cls, "acme/st", {}, _CPU, provider_name="test.st"
    )
    assert TORCH_JIT_ENV not in os.environ
    assert fake_torch.jit._state._enabled.enabled is True  # noqa: SLF001


def test_st_load_inspects_the_eager_model_before_any_compile_wrap(
    fake_torch: Any, log: list[str]
) -> None:
    """``inspect_model`` sees the EAGER model: a width check can never read a compiled wrapper."""
    st = _FakeST(log)
    seen: list[Any] = []
    wrapped = object()

    def compile_(_model: Any, **_kw: Any) -> Any:
        log.append("compiled")
        return wrapped

    fake_torch.compile = compile_

    out = load_sentence_transformer(
        st.make_cls,
        "acme/st",
        {"compile": True},
        _CPU,
        provider_name="test.st",
        inspect_model=seen.append,
    )
    assert out is wrapped
    assert seen and seen[0] is not wrapped  # the hook saw the eager model, not the wrapper
    assert log.index("compiled") > log.index("st-load:acme/st:cpu")


def test_st_load_skipped_decision_is_a_loud_config_error(fake_torch: Any, log: list[str]) -> None:
    """#191 discipline: a skipped DeviceDecision refuses loudly, before any import happens."""
    st = _FakeST(log)
    with pytest.raises(ConfigError, match="no CUDA device is available"):
        load_sentence_transformer(st.make_cls, "acme/st", {}, _SKIPPED, provider_name="test.st")
    assert not st.constructed


def test_st_load_fa2_falls_back_to_sdpa(
    fake_torch: Any, log: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FA2 construction that raises is retried once on sdpa — never a crash."""
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    st = _FakeST(log)
    inner_cls = st.make_cls()

    class _NoFa2:
        def __new__(cls, model_id: str, device: str | None = None, model_kwargs: Any = None) -> Any:
            if (model_kwargs or {}).get("attn_implementation") == "flash_attention_2":
                msg = "no FA2 kernel for this arch"
                raise ValueError(msg)
            return inner_cls(model_id, device=device, model_kwargs=model_kwargs)

    load_sentence_transformer(
        lambda: _NoFa2,
        "acme/st",
        {"flash_attention": True},
        _CUDA,
        provider_name="test.st",
    )
    assert st.constructed[-1][2]["attn_implementation"] == "sdpa"


def test_st_load_import_error_propagates_with_the_install_hint(fake_torch: Any) -> None:
    """A missing optional dep stays the operator's actionable message, never a provider fault."""

    def _no_dep() -> Any:
        msg = "Install it with: pip install latence-embedder-st"
        raise ImportError(msg)

    with pytest.raises(ImportError, match="pip install latence-embedder-st"):
        load_sentence_transformer(_no_dep, "acme/st", {}, _CPU, provider_name="test.st")


def test_st_load_guard_wraps_the_native_load_and_inspection_faults_propagate(
    fake_torch: Any, log: list[str]
) -> None:
    """The caller's guard maps the native load; an ``inspect_model`` fault passes through as-is."""
    import contextlib

    @contextlib.contextmanager
    def guard(op: str) -> Iterator[None]:
        try:
            yield
        except OSError as exc:
            msg = f"test.st: config error in {op!r}"
            raise ConfigError(msg) from exc

    class _BadId:
        def __init__(self, _model_id: str, device: str | None = None, model_kwargs: Any = None) -> None:
            msg = "model not found on the hub"
            raise OSError(msg)

    with pytest.raises(ConfigError, match="load_model"):
        load_sentence_transformer(
            lambda: _BadId, "nope", {}, _CPU, provider_name="test.st", guard=guard
        )

    st = _FakeST(log)

    def _bad_width(_model: Any) -> None:
        msg = "test.st: model produces 6-dim vectors, but config declared dimension 8."
        raise ProviderError(msg)

    with pytest.raises(ProviderError, match="declared dimension 8"):
        load_sentence_transformer(
            st.make_cls,
            "acme/st",
            {},
            _CPU,
            provider_name="test.st",
            guard=guard,
            inspect_model=_bad_width,
        )
