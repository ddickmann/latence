"""REGRESSION (a): ``fused_entity_relation.gliner2`` goes through the ADR-0036 device seam.

The defect. This Provider — the one carrying the heaviest Stage — had **zero** hits for
``select_device`` / ``AdapterBase`` / ``config["device"]``. It resolved the ``quantize``/``compile``
knobs by handing ``device=None`` to the resolver (which falls back to a bare ``cuda_available()``
probe) and then placed the model with ``ensure_gliner2_on_cuda(cuda=cuda_available())``. Two
consequences, both silent:

* ``device: cuda`` in the blessed pod stack was a **no-op** — nothing read the config key at all;
* ``device: cpu`` on a GPU box was **silently overridden**: fp16 was still resolved on, and the
  model was still moved to cuda. An operator pinning CPU (to leave the GPU for another stage, to
  reproduce a CPU result, to work around a driver fault) did not get CPU and was never told.

That is #191's failure class — a Provider deciding where to run from an availability probe rather
than from the routed decision — reappearing in the Provider that matters most. The old AST guard
could not see it: its placement check was scoped to ``_PLAIN_GLINER_SOURCES``.

These tests assert BEHAVIOUR through the Provider's own load path: what kwargs the native
``from_pretrained`` actually received, and where the model actually ended up.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any

import pytest
from latence_core.errors import ConfigError
from latence_extract_gliner2 import Gliner2FusedExtractor


class _FakeGLiNER2:
    """A gliner2 stand-in that records the load kwargs and every placement move.

    Faithful in the way that matters here: a freshly ``from_pretrained``-ed model reports ``cpu``
    (which is why placement exists at all), and ``.to()`` is what moves it. It also exposes the
    batched API the Provider checks for at load, so a device-seam test never fails for an unrelated
    reason.
    """

    last_kwargs: dict[str, Any] = {}
    moves: list[str] = []

    def __init__(self) -> None:
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **kwargs: Any) -> _FakeGLiNER2:
        cls.last_kwargs = dict(kwargs)
        return cls()

    def to(self, target: str) -> _FakeGLiNER2:
        type(self).moves.append(target)
        self.device = target
        return self

    def batch_extract_entities(self, *_a: Any, **_k: Any) -> list[Any]:
        return []

    def batch_extract_relations(self, *_a: Any, **_k: Any) -> list[Any]:
        return []


@pytest.fixture()
def fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER2]:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _FakeGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    _FakeGLiNER2.last_kwargs = {}
    _FakeGLiNER2.moves = []
    return _FakeGLiNER2


def _config(**extra: Any) -> dict[str, Any]:
    return {"labels": ["person"], "relation_labels": ["works_for"], **extra}


def test_device_cpu_is_honoured_on_a_cuda_box(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``device: cpu`` on a GPU host means CPU: no fp16 quantize, and no move to cuda.

    ``LATENCE_CUDA=1`` pins the availability probe to "a GPU is present", so anything still keying
    off the probe rather than the resolved decision will visibly override the operator here.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    model = Gliner2FusedExtractor(_config(device="cpu"))._load_model()

    assert fake_gliner2.last_kwargs.get("quantize") is not True, (
        "device: cpu must not resolve fp16 quantize on — fp16 eager on CPU is the pathological "
        "CPU-bound spin the W9-scaling diagnosis chased"
    )
    assert "cuda" not in fake_gliner2.moves, (
        f"device: cpu was silently overridden and the model was moved to {fake_gliner2.moves}"
    )
    assert model.device == "cpu"


def test_device_cuda_resolves_the_native_knobs_on_a_cuda_box(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other direction: a routed cuda decision does turn quantize/compile on and place."""
    monkeypatch.setenv("LATENCE_CUDA", "1")
    model = Gliner2FusedExtractor(_config(device="cuda"))._load_model()

    assert fake_gliner2.last_kwargs == {"quantize": True, "compile": True}
    assert model.device == "cuda"


def test_device_cuda_on_a_cpu_only_host_is_a_loud_config_error(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``device: cuda`` with no GPU is an operator error, not a silent CPU run (#191).

    Pre-fix this Provider ignored ``config["device"]`` entirely, so this configuration loaded
    happily on CPU and ran at CPU speed with no warning anywhere.
    """
    monkeypatch.setenv("LATENCE_CUDA", "0")
    with pytest.raises(ConfigError, match="cuda"):
        Gliner2FusedExtractor(_config(device="cuda"))._load_model()


def test_an_invalid_device_string_is_a_config_error_at_construction(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    """A typo'd device is caught before any weights are downloaded, like every other Provider."""
    with pytest.raises(ConfigError, match="invalid device config"):
        Gliner2FusedExtractor(_config(device="gpu"))


def test_intra_op_threads_are_capped_at_load(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The many-core oversubscription guard still applies after the choreography moved inward.

    ``os.cpu_count`` is pinned high on purpose: ``resolve_intra_op_threads`` clamps the requested
    cap to the core count, so an unpinned ``assert calls == [5]`` would be an assertion about the
    machine the suite runs on (green on a 10-core dev box, red on the 2-4 vCPU ``ubuntu-latest``
    runner). Pinned, this asserts the configured cap; the clamp is pinned separately below.
    """
    calls: list[int] = []
    torch = types.ModuleType("torch")
    torch.set_num_threads = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("LATENCE_CUDA", "0")
    monkeypatch.setattr(os, "cpu_count", lambda: 64)

    Gliner2FusedExtractor(_config(device="cpu", intra_op_threads=5))._load_model()
    assert calls == [5]


def test_the_intra_op_cap_never_exceeds_the_host_core_count(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cap larger than the box clamps down — asking for more threads than cores IS the bug."""
    calls: list[int] = []
    torch = types.ModuleType("torch")
    torch.set_num_threads = calls.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("LATENCE_CUDA", "0")
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    Gliner2FusedExtractor(_config(device="cpu", intra_op_threads=5))._load_model()
    assert calls == [4]
