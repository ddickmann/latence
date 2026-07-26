"""REGRESSION (c): ``sparse.splade`` is a learned Provider and now loads like one.

The defect. ``TransformersSpladeModel`` did a bare
``AutoModelForMaskedLM.from_pretrained(model_id)`` — **fp32 on a GPU**, the ~2x miss the whole perf
seam exists to close — and never placed the model on any device, because ``sparse.splade`` was
simply not in the fp32 regression guard's curated ``_LEARNED_PROVIDER_SOURCES`` tuple. It is a
registered Provider running real transformer forwards over the corpus.

The second half was the forward itself: ``torch.no_grad()`` instead of
:func:`~latence_core.providers.perf.inference_context`. ``no_grad`` is STRICTLY WEAKER — it stops
gradients being computed, but the forward still builds an autograd graph and pins every intermediate
activation alive for a backward that never comes. Those accumulate across a long run (pod-measured
27.28 GiB resident on a ~2-3 GB model) until the next allocation CUDA-OOMs, and ``empty_cache()``
cannot reclaim any of it because the memory is held LIVE, not merely cached. ``inference_mode`` is
what killed that OOM.

Both are asserted on observed effects: the kwargs the load received, where the model ended up, and
which torch context manager the forward actually entered.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.providers.device import DeviceDecision
from latence_splade.model import TransformersSpladeModel


class _Arr:
    def __init__(self, rows: Any) -> None:
        self._rows = rows

    def tolist(self) -> Any:
        return self._rows

    def __getitem__(self, key: Any) -> Any:
        return self._rows[key]


class _Config:
    vocab_size = 16


class _Output:
    def __init__(self, logits: _Arr) -> None:
        self.logits = logits


class _FakeMaskedLM:
    """Faithful where it matters: resident on cpu after ``from_pretrained``, movable via ``.to``."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self) -> None:
        self.config = _Config()
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **kwargs: Any) -> _FakeMaskedLM:
        cls.last_kwargs = dict(kwargs)
        return cls()

    def to(self, target: Any) -> _FakeMaskedLM:
        if isinstance(target, str):
            self.device = target
        return self

    def eval(self) -> _FakeMaskedLM:
        return self

    def __call__(self, **_kw: Any) -> _Output:
        return _Output(_Arr([[[0.0] * 16]]))


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_id: str, **_kw: Any) -> _FakeTokenizer:
        return cls()

    def __call__(self, texts: Any, **_kw: Any) -> dict[str, Any]:
        return {"input_ids": _Arr([[1]]), "attention_mask": _Arr([[1]])}


@pytest.fixture()
def fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    module.AutoModelForMaskedLM = _FakeMaskedLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    _FakeMaskedLM.last_kwargs = {}


_CUDA = DeviceDecision(device="cuda", skipped=False, reason="test: cuda")


def test_splade_loads_bf16_on_gpu_not_bare_fp32(
    fake_transformers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The load spreads the perf kwargs instead of loading bare (fp32/eager) on a GPU."""
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    model = TransformersSpladeModel("acme/splade", device=_CUDA)
    assert model.vocab_size == 16  # triggers the lazy load
    assert _FakeMaskedLM.last_kwargs.get("torch_dtype") is torch.bfloat16, (
        "sparse.splade loaded bare fp32 on a GPU — the ~2x miss the perf seam closes"
    )
    assert _FakeMaskedLM.last_kwargs.get("attn_implementation") == "sdpa"


def test_splade_places_the_model_on_the_resolved_device(fake_transformers: None) -> None:
    """A freshly loaded model reports cpu; the seam must MOVE it and verify it landed."""
    model = TransformersSpladeModel("acme/splade", device=_CUDA)
    model._load()  # noqa: SLF001
    assert model._model.device == "cuda", (  # noqa: SLF001
        "sparse.splade never placed its model — it ran wherever from_pretrained left it (cpu)"
    )


def test_splade_forward_runs_under_inference_mode_not_no_grad(
    fake_transformers: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION: the forward enters ``torch.inference_mode()``, and never ``torch.no_grad()``.

    A fake that RECORDS which context was entered — and whose ``no_grad`` is present and usable, so
    the test cannot pass merely because ``no_grad`` was unavailable.
    """
    import contextlib

    entered: list[str] = []

    @contextlib.contextmanager
    def _ctx(name: str) -> Any:
        entered.append(name)
        yield

    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: _ctx("inference_mode")  # type: ignore[attr-defined]
    torch.no_grad = lambda: _ctx("no_grad")  # type: ignore[attr-defined]
    # A real torch has these; the load choreography touches them on the way to the forward.
    torch.float32 = object()  # type: ignore[attr-defined]
    torch.bfloat16 = object()  # type: ignore[attr-defined]
    torch.set_num_threads = lambda _n: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)

    TransformersSpladeModel("acme/splade").term_weights(["hello"])

    assert "inference_mode" in entered, (
        "the SPLADE forward must run under inference_mode: no_grad still builds an autograd graph "
        "and pins its activations alive, which is the 27.28 GiB accumulation that OOM'd"
    )
    assert "no_grad" not in entered


def test_no_device_decision_keeps_the_pre_existing_no_placement_behaviour(
    fake_transformers: None,
) -> None:
    """A caller that resolves no device (an offline test, an injected adapter) is still served."""
    model = TransformersSpladeModel("acme/splade")
    model._load()  # noqa: SLF001
    assert model._model.device == "cpu"  # noqa: SLF001
