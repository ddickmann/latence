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

import math
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


# --- The other half of placement: the ENCODING has to follow the model -----------------------
#
# ``_FakeMaskedLM`` above ignores the device of what it is called with, so it can prove the model
# MOVED but never that the forward is *runnable* afterwards. Torch's real contract is that a
# forward mixing a cuda module with cpu inputs raises ``RuntimeError: Expected all tensors to be on
# the same device``. The doubles below encode exactly that contract: a tensor carries a device and
# a ``.to``; the encoding is a ``BatchEncoding``-shaped mapping that moves its tensors; and the
# model REFUSES a forward whose inputs are not resident where it is.


class _DeviceArr(_Arr):
    """A tensor double that knows where it lives — ``.to`` returns a moved copy, like torch."""

    def __init__(self, rows: Any, device: str = "cpu") -> None:
        super().__init__(rows)
        self.device = device

    def to(self, target: Any) -> _DeviceArr:
        return _DeviceArr(self._rows, str(target))


class _Encoding(dict[str, Any]):
    """A ``BatchEncoding`` double: a mapping (so ``**encoding`` works) that can move its tensors."""

    def to(self, target: Any) -> _Encoding:
        return _Encoding({key: value.to(target) for key, value in self.items()})


class _DeviceStrictMaskedLM(_FakeMaskedLM):
    """Torch's device contract: a forward over inputs resident elsewhere is a ``RuntimeError``."""

    def __call__(self, **kwargs: Any) -> _Output:
        for name, tensor in kwargs.items():
            resident = getattr(tensor, "device", "cpu")
            if str(resident).split(":", 1)[0] != str(self.device).split(":", 1)[0]:
                msg = (
                    f"Expected all tensors to be on the same device, but got {name} is on "
                    f"{resident}, different from other tensors on {self.device}"
                )
                raise RuntimeError(msg)
        return _Output(_Arr([[[math.e - 1.0] * 16]]))


class _DeviceStrictTokenizer(_FakeTokenizer):
    """A real tokenizer returns CPU tensors regardless of where the model was placed."""

    def __call__(self, texts: Any, **kwargs: Any) -> Any:
        if kwargs.get("return_tensors") != "pt":
            return {"input_ids": [1]}
        return _Encoding(
            {"input_ids": _DeviceArr([[1]]), "attention_mask": _DeviceArr([[1]])}
        )


@pytest.fixture()
def device_strict_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _DeviceStrictTokenizer  # type: ignore[attr-defined]
    module.AutoModelForMaskedLM = _DeviceStrictMaskedLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_splade_moves_the_encoding_onto_the_device_it_placed_the_model_on(
    device_strict_transformers: None,
) -> None:
    """REGRESSION: placing the model without moving the encoding breaks every CUDA host.

    The tokenizer hands back CPU tensors no matter where the model was placed. With the model on
    ``cuda`` and the encoding left on ``cpu``, torch raises ``RuntimeError: Expected all tensors to
    be on the same device`` — which this adapter maps to a ``ProviderError`` and the Export's
    sparse-embedding stage fails outright. Before device placement existed the model stayed on CPU
    and the forward succeeded, so this is a *new* break of a provider that used to work (slowly).
    """
    model = TransformersSpladeModel("acme/splade", device=_CUDA)

    weights = model.term_weights(["hello"])

    assert model._model.device == "cuda"  # noqa: SLF001 - the model really was placed
    assert weights == [dict.fromkeys(range(16), 1.0)], (
        "the SPLADE forward must run with the encoding resident on the same device as the model"
    )


def test_splade_encoding_stays_on_cpu_when_no_device_was_resolved(
    device_strict_transformers: None,
) -> None:
    """The no-placement path is unchanged: nothing moves, and the forward still runs."""
    model = TransformersSpladeModel("acme/splade")

    weights = model.term_weights(["hello"])

    assert model._model.device == "cpu"  # noqa: SLF001
    assert weights == [dict.fromkeys(range(16), 1.0)]
