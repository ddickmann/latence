"""#190 (audit R1) — the torch JIT must be OFF **before** the gliner import, not after it.

The #190 fix put :func:`latence_core.providers.perf.neutralize_torch_jit` inside the two shared
loaders (``load_then_cast`` / ``load_gliner2_optimized``). That is too late: every gliner-family
Provider evaluates ``from gliner import GLiNER`` (``from gliner2 import GLiNER2``) *before* it
enters the loader, and that import transitively pulls in transformers'
``models.deberta_v2.modeling_deberta_v2``, whose six relative-position helpers are decorated with
``@torch.jit.script`` and are therefore compiled into real ``ScriptFunction``s **at import time**,
while the JIT is still live. Neither of the seam's levers can undo that afterwards: ``PYTORCH_JIT``
is read by torch only at *torch* import, and ``torch.jit._state.disable()`` only affects *future*
``script()`` calls. So the seam logged "torch JIT neutralized" while the encoder stayed scripted and
the Blackwell/CUDA-13 ``libnvrtc-builtins.so.13.0`` crash was NOT fixed.

Anti-false-green, and the reason this suite exists next to
``latence-extract-gliner2/tests/test_gliner2_jit_neutralization.py``: that suite's fake scripts the
helper inside ``from_pretrained`` (i.e. *after* the seam call), so it passes either way. Here the
fake ``gliner``/``gliner2`` module scripts at **module import**, which is the real transformers
shape — the only shape that can distinguish "neutralized before the import" from "after". The fake
torch mirrors the real switch semantics: ``torch.jit.script`` is a pass-through only while
``torch.jit._state._enabled.enabled`` is False, and otherwise returns a ScriptFunction that dies
with the exact nvrtc message when CALLED (the field behaviour: the fault surfaces on the first
forward, which is why no load-time try/except can see it).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest

NVRTC_MESSAGE = "nvrtc: failed to open libnvrtc-builtins.so.13.0"


def build_relative_position(query_size: int, key_size: int) -> int:
    """Stands in for a transformers deberta_v2 ``@torch.jit.script``-ed module-level helper."""
    return query_size - key_size


class _ScriptFunction:
    """What ``torch.jit.script`` returns with the JIT live — and what dies on Blackwell/CUDA-13."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        msg = (
            "The following operation failed in the TorchScript interpreter: Compilation error: "
            + NVRTC_MESSAGE
        )
        raise RuntimeError(msg)


class _Enabled:
    """Mirrors ``torch.jit._state._enabled`` — the proxy torch parses ``PYTORCH_JIT`` into."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled


@pytest.fixture()
def live_jit_torch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A ``torch`` whose JIT starts LIVE (``PYTORCH_JIT=1``), mirroring the real switch."""
    monkeypatch.setenv("PYTORCH_JIT", "1")
    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    state = types.ModuleType("torch.jit._state")
    enabled = _Enabled(enabled=True)

    def disable() -> None:
        enabled.enabled = False

    def script(fn: Callable[..., Any]) -> Any:
        return _ScriptFunction(fn) if enabled.enabled else fn

    def set_float32_matmul_precision(precision: str) -> None:  # noqa: ARG001 - throughput-only knob
        """``disambiguation.glinker`` sets this right after the (post-neutralization) torch import."""

    torch.set_float32_matmul_precision = set_float32_matmul_precision  # type: ignore[attr-defined]
    state._enabled = enabled  # type: ignore[attr-defined]
    state.disable = disable  # type: ignore[attr-defined]
    jit._state = state  # type: ignore[attr-defined]
    jit.script = script  # type: ignore[attr-defined]
    torch.jit = jit  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.jit", jit)
    return torch


class _MdebertaModel:
    """The minimum a loaded gliner/gliner2 model must expose for the shared load path.

    ``to`` (device move + dtype cast), a ``config.max_len`` (the W12 window bound) and the scripted
    relative-position helper the encoder forward consults — the fault surface under test.
    """

    def __init__(self, relative_position: Any) -> None:
        self.config = types.SimpleNamespace(max_len=384)
        self.relative_position = relative_position

    @staticmethod
    def _from_pretrained_factory(relative_position: Any) -> Callable[..., _MdebertaModel]:
        def from_pretrained(model_id: str, **kwargs: Any) -> _MdebertaModel:  # noqa: ARG001
            return _MdebertaModel(relative_position)

        return from_pretrained

    def to(self, arg: Any) -> _MdebertaModel:  # noqa: ARG002 - device/dtype both no-op here
        return self

    def batch_extract_entities(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ARG002
        """Inert: ``Gliner2PIIRedactor._load_model`` asserts this API exists (#192) at LOAD.

        This suite asserts on the JIT state at the backbone IMPORT, never on inference, so the
        method only has to be callable for the load path to complete.
        """
        return {"entities": {}}

    def forward(self, text: str) -> None:
        """The encoder forward — consults the relative-position helper, like mdeberta does."""
        self.relative_position(len(text), 0)


class _ImportTimeScriptingFinder:
    """A ``gliner``/``gliner2`` whose **import** scripts the helper — the transformers shape.

    Installed on ``sys.meta_path``, so the scripting happens exactly when the Provider evaluates its
    ``from gliner import GLiNER`` statement. ``jit_enabled_at_import`` records the JIT state at that
    instant: the property under test.
    """

    def __init__(
        self, module_name: str, attr: str, extra: dict[str, Any] | None = None
    ) -> None:
        self.module_name = module_name
        self.attr = attr
        #: Additional module attributes the importer needs. ``glinker`` is consumed as
        #: ``from glinker import ProcessorFactory`` (a DAG builder), not as a ``from_pretrained``
        #: model class, so its shim is injected here rather than synthesised as ``attr``.
        self.extra = extra or {}
        self.jit_enabled_at_import: bool | None = None
        self.scripted: Any = None

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:  # noqa: ARG002
        if fullname != self.module_name:
            return None
        # ``self`` is a duck-typed loader (create_module/exec_module); mypy's LoaderProtocol also
        # wants the legacy ``load_module``, which importlib no longer calls.
        return importlib.util.spec_from_loader(fullname, cast("Any", self))

    def create_module(self, spec: Any) -> None:  # noqa: ARG002 - default module creation
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        import torch

        self.jit_enabled_at_import = torch.jit._state._enabled.enabled  # noqa: SLF001
        # What transformers' deberta_v2 modeling module does at import: evaluate @torch.jit.script.
        self.scripted = torch.jit.script(build_relative_position)
        model_cls = type(
            self.attr,
            (),
            {"from_pretrained": staticmethod(_MdebertaModel._from_pretrained_factory(self.scripted))},  # noqa: SLF001
        )
        setattr(module, self.attr, model_cls)
        for name, value in self.extra.items():
            setattr(module, name, value)


@pytest.fixture()
def scripting_gliner(
    request: pytest.FixtureRequest, live_jit_torch: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Iterator[_ImportTimeScriptingFinder]:
    """Install a ``gliner``/``gliner2`` that scripts the mdeberta helper at IMPORT time."""
    module_name, attr, *rest = request.param
    finder = _ImportTimeScriptingFinder(module_name, attr, rest[0] if rest else None)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    try:
        yield finder
    finally:
        sys.modules.pop(module_name, None)


class _FakeProcessorFactory:
    """The one thing ``disambiguation.glinker`` consumes from the ``glinker`` library."""

    @staticmethod
    def create_from_dict(config: dict[str, Any]) -> Any:
        assert config["nodes"], "the DAG must carry its L2/L3/L0 nodes"
        return types.SimpleNamespace(config=config)


# Each entry LOADS its Provider: constructing is not enough — the backbone import happens inside the
# lazy load path (``_load_model`` for the gliner family, ``_init_executor`` for glinker).
def _load_entity_extractor(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    from latence_ner_gliner import GlinerEntityExtractor

    GlinerEntityExtractor({"labels": ["person"], "device": "cpu"})._load_model()  # noqa: SLF001


def _load_pii_redactor(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    from latence_pii_gliner import GlinerPIIRedactor

    GlinerPIIRedactor({"pii_labels": ["person"], "device": "cpu"})._load_model()  # noqa: SLF001


def _load_relation_extractor(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    from latence_relation_gliner import GlinerRelexExtractor

    GlinerRelexExtractor(
        {"labels": ["person"], "relation_labels": ["employs"], "device": "cpu"}
    )._load_model()  # noqa: SLF001


def _load_fused_extractor(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    from latence_extract_gliner2 import Gliner2FusedExtractor

    Gliner2FusedExtractor(
        {"labels": ["person"], "relation_labels": ["employs"]}
    )._load_model()  # noqa: SLF001


def _load_pii_gliner2_redactor(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    """R4: the FIFTH mdeberta Provider — and the one #197 made the wizard's blessed default.

    ``redaction.gliner2`` is emitted by the wizard in BOTH profiles
    (``BLESSED_REDACT_PROVIDER``), yet it was the only mdeberta Provider whose W9-jit ordering had
    no behavioural proof: its sole guard was the AST check in
    ``test_perf_fp32_regression_guard.py``, which is purely lexical (min-lineno of any
    ``neutralize_torch_jit`` call vs min-lineno of any backbone import) and therefore blind to any
    call that is textually first but does not take effect — e.g. one wrapped in an env-gated
    ``if``. ``Gliner2PIIRedactor._gliner2_cls`` (entered from ``_load_model``) is the load path
    that evaluates ``from gliner2 import GLiNER2``.
    """
    from latence_pii_gliner2 import Gliner2PIIRedactor

    Gliner2PIIRedactor({"pii_labels": ["person"], "device": "cpu"})._load_model()  # noqa: SLF001


def _load_glinker_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2/#190-5: the sixth DeBERTa-backed Provider, previously absent from every #190 suite.

    ``GLinkerBackend._init_executor`` is the load path: it calls the shared
    ``neutralize_torch_jit`` seam and only then evaluates ``import torch`` /
    ``from glinker import ProcessorFactory``. The tokenizer-config patch is stubbed out because it
    reaches the Hugging Face hub cache, which is orthogonal to the ordering property under test.
    """
    from latence_disambig_glinker.backends import GLinkerBackend

    monkeypatch.setattr(GLinkerBackend, "_patch_tokenizer_config", staticmethod(lambda _name: None))
    GLinkerBackend(device="cpu", reranker_model=None)._init_executor()  # noqa: SLF001


_PROVIDERS = [
    pytest.param(("gliner", "GLiNER"), _load_entity_extractor, id="entity.gliner"),
    pytest.param(("gliner", "GLiNER"), _load_pii_redactor, id="redaction.gliner_pii"),
    pytest.param(("gliner", "GLiNER"), _load_relation_extractor, id="relation.gliner_relex"),
    pytest.param(("gliner2", "GLiNER2"), _load_fused_extractor, id="fused_entity_relation.gliner2"),
    pytest.param(("gliner2", "GLiNER2"), _load_pii_gliner2_redactor, id="redaction.gliner2"),
    pytest.param(
        ("glinker", "GLinkerModel", {"ProcessorFactory": _FakeProcessorFactory}),
        _load_glinker_backend,
        id="disambiguation.glinker",
    ),
]


@pytest.mark.parametrize(
    ("scripting_gliner", "load_provider"), _PROVIDERS, indirect=["scripting_gliner"]
)
def test_jit_is_neutralized_before_the_gliner_import(
    scripting_gliner: _ImportTimeScriptingFinder,
    load_provider: Callable[[pytest.MonkeyPatch], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION (audit R1 / #190): the JIT is already OFF when gliner is imported.

    Pre-fix the seam ran *after* ``from gliner import GLiNER``, so the mdeberta helper was scripted
    at import and stayed scripted: ``jit_enabled_at_import`` is True, the helper is a
    ``ScriptFunction``, and the first forward raises the nvrtc error — the #190 crash, unfixed.
    """
    monkeypatch.setenv("LATENCE_CUDA", "0")

    load_provider(monkeypatch)  # the lazy load path is the unit under test

    assert scripting_gliner.jit_enabled_at_import is False, (
        "the torch JIT was still LIVE when the gliner module was imported — every "
        "@torch.jit.script helper in transformers' deberta_v2 module is a ScriptFunction and "
        "neutralize_torch_jit() can no longer undo it (Blackwell/CUDA-13 nvrtc crash, #190)"
    )
    # The observable consequence: the helper is the plain eager function, so the encoder forward
    # never reaches nvrtc.
    assert scripting_gliner.scripted is build_relative_position
    _MdebertaModel(scripting_gliner.scripted).forward("Ada Lovelace")


@pytest.mark.parametrize(
    ("scripting_gliner", "load_provider"),
    [_PROVIDERS[0], _PROVIDERS[-1]],
    indirect=["scripting_gliner"],
)
def test_a_live_jit_at_import_reproduces_the_blackwell_crash(
    scripting_gliner: _ImportTimeScriptingFinder,
    load_provider: Callable[..., None],  # noqa: ARG001 - not loaded: this is the fake's own guard
    live_jit_torch: types.ModuleType,
) -> None:
    """Anti-false-green: with the JIT live at import, the fake DOES reproduce the nvrtc crash.

    Without this, the assertions above could pass against a fake that never fails.
    """
    # The finder's module, imported here with the JIT untouched. Parametrized over the gliner AND
    # the glinker shapes, so the glinker fake is proven capable of failing too (R2/#190-5).
    backbone = importlib.import_module(scripting_gliner.module_name)

    assert scripting_gliner.jit_enabled_at_import is True
    model = getattr(backbone, scripting_gliner.attr).from_pretrained("m")
    with pytest.raises(RuntimeError, match=NVRTC_MESSAGE):
        model.forward("Ada Lovelace")
    assert live_jit_torch.jit._state._enabled.enabled is True  # noqa: SLF001
