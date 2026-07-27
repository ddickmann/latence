"""REGRESSION (the still-open #191 gap): ``parser.lighton`` PROVES its model landed.

The defect. #191 established the rule "honour the resolved device or fail loudly" and encoded it in
:func:`~latence_core.providers.perf.place_on_device`, which moves the model AND reads its resident
device back, raising when the two disagree. Three of the five model-placing Providers used it. The
two OCR parsers did not: they used a bare ``model = model.to(device)`` — the exact shape that cannot
distinguish "moved" from "call returned, model still on cpu", which is how a ``device: cuda`` run sat
at CPU speed with the GPU idle and no warning at all.

The fake below moves normally in the first test and REFUSES to move in the second, which is the
whole point: a fake that always moves cannot tell a verified placement from an unverified one.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.errors import ProviderError
from latence_parser_lighton import LightOnOcrParser


class _StubProcessor:
    @classmethod
    def from_pretrained(cls, _model_id: str, **_kw: Any) -> _StubProcessor:
        return cls()


class _StubModel:
    """A VLM stand-in that starts on cpu (as ``from_pretrained`` leaves it) and may refuse to move."""

    movable: bool = True

    def __init__(self) -> None:
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kw: Any) -> _StubModel:
        return cls()

    def to(self, target: str) -> _StubModel:
        if type(self).movable:
            self.device = target
        return self


@pytest.fixture()
def stub_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")
    module.LightOnOcrForConditionalGeneration = _StubModel  # type: ignore[attr-defined]
    module.LightOnOcrProcessor = _StubProcessor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)
    _StubModel.movable = True
    monkeypatch.setenv("LATENCE_CUDA", "1")


def test_the_model_is_actually_moved_to_the_resolved_device(stub_transformers: None) -> None:
    model, _processor = LightOnOcrParser({"model": "acme/ocr"})._load_model()
    assert model.device == "cuda"


def test_a_model_that_reports_the_wrong_device_is_refused(stub_transformers: None) -> None:
    """The bare ``model.to(device)`` this Provider used could not see this; the seam raises.

    Silence is the bug: running on cpu while the config says cuda is a ~10x throughput cliff that
    reports itself as a successful run.
    """
    _StubModel.movable = False
    with pytest.raises(ProviderError, match="refusing to run on the wrong device"):
        LightOnOcrParser({"model": "acme/ocr"})._load_model()
