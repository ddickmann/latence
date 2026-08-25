"""The ``gliner2`` 2.x compatibility gate — the second gate, for an env the resolver never saw.

The package pin (``gliner2[local]>=2.0,<3``) is the primary gate and fails at RESOLUTION. These
tests cover the case that produced the #192 field failure on the 1.x redactor: a pod with a
hand-``pip install``ed ``gliner2``, where nothing resolved anything. Every failure below must be a
typed :class:`~latence_core.errors.ProviderError` naming the supported range — never a raw native
``AttributeError``/``TypeError``.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from _gliner25_fake import install_gliner2
from latence_core.errors import ConfigError, ProviderError
from latence_gliner25 import Gliner25FusedExtractor, Gliner25PIIRedactor
from latence_gliner25._backbone import DEFAULT_MODEL, MAX_SUPPORTED_MAX_LEN
from latence_gliner25._compat import SUPPORTED_SPEC

_EXTRACT_CONFIG = {"labels": ["person"], "relation_labels": ["works for"], "device": "cpu"}
_REDACT_CONFIG = {"pii_labels": ["person", "email"], "device": "cpu"}


def _pyproject() -> dict[str, object]:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with path.open("rb") as handle:
        return tomllib.load(handle)


def test_supported_spec_matches_the_declared_dependency_pin() -> None:
    """``_compat.SUPPORTED_SPEC`` must be byte-identical to the ``pyproject.toml`` specifier.

    The two gates say the same thing to two different audiences (the resolver and the operator),
    and a drift between them is how an error message starts recommending an install that the
    package itself refuses. Asserted against the file, not against a mirrored constant.
    """
    project = _pyproject()["project"]
    assert isinstance(project, dict)
    deps = project["dependencies"]
    assert isinstance(deps, list)
    pins = [d for d in deps if str(d).startswith("gliner2")]
    assert pins == [f"gliner2[local]{SUPPORTED_SPEC}"]


def test_the_pin_is_disjoint_from_the_gliner2_1x_packages() -> None:
    """The whole architectural crux, asserted from the two committed manifests.

    ``latence-extract-gliner2`` / ``latence-pii-gliner2`` pin ``>=1.3,<2``; this package pins
    ``>=2.0,<3``. If some future edit widened either, they would silently become co-installable and
    whichever ``gliner2`` won the resolve would break the other half at model load. This test is
    the cheap, offline statement of that; ``deploy/env/`` is the enforced one.
    """
    packages = Path(__file__).resolve().parents[2]
    for sibling in ("latence-extract-gliner2", "latence-pii-gliner2"):
        with (packages / sibling / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        deps = [str(d) for d in project["dependencies"] if str(d).startswith("gliner2")]
        assert deps == ["gliner2[local]>=1.3,<2"], sibling


@pytest.mark.parametrize("version", ["1.3.2", "1.9.9", "3.0.0", "not-a-version"])
def test_an_unsupported_gliner2_is_a_typed_provider_error(version: str) -> None:
    """An out-of-range (or unparseable) install fails BEFORE any weight download, typed."""
    provider = Gliner25FusedExtractor(dict(_EXTRACT_CONFIG))
    with install_gliner2(version=version), pytest.raises(ProviderError) as excinfo:
        provider._load_model()  # noqa: SLF001 - the load is the gate under test
    assert SUPPORTED_SPEC in str(excinfo.value)


def test_an_unreadable_version_is_allowed_with_the_api_checks_still_binding() -> None:
    """A version that cannot be determined is logged and allowed — the API checks still bite.

    The metadata pin already gated resolution, so refusing an unversioned source checkout would
    block a legitimate development install. What must NOT happen is the check being skipped
    silently: with no ``AutoExtractor`` the load is still a typed error.
    """
    provider = Gliner25FusedExtractor(dict(_EXTRACT_CONFIG))
    with install_gliner2(version="", with_auto=False), pytest.raises(ProviderError) as excinfo:
        provider._load_model()  # noqa: SLF001 - the load is the gate under test
    assert "AutoExtractor" in str(excinfo.value)


def test_a_pre_2x_install_without_autoextractor_is_typed_not_an_attribute_error() -> None:
    """``gliner2`` < 2 has no ``gliner2.auto``; the operator gets a range, not an AttributeError."""
    provider = Gliner25PIIRedactor(dict(_REDACT_CONFIG))
    with install_gliner2(version="2.0.0", with_auto=False), pytest.raises(ProviderError) as exc:
        provider._load_model()  # noqa: SLF001 - the load is the gate under test
    assert "AutoExtractor" in str(exc.value)
    assert SUPPORTED_SPEC in str(exc.value)


def test_a_span_checkpoint_is_refused_and_names_the_provider_that_loads_it() -> None:
    """Pointing ``model`` at a gliner2 1.x checkpoint must FAIL, not quietly run a different model.

    ``AutoExtractor`` dispatches on the checkpoint, so a span checkpoint loads *successfully* — the
    same method names, a different head, different quality, none of this package's tuning. That is
    a silent wrong-model run, which is worse than a crash.
    """
    provider = Gliner25FusedExtractor(
        {**_EXTRACT_CONFIG, "model": "fastino/gliner2-multi-v1"}
    )
    with install_gliner2(), pytest.raises(ProviderError) as excinfo:
        provider._load_model()  # noqa: SLF001 - the load is the gate under test
    message = str(excinfo.value)
    assert "span" in message
    assert "fused_entity_relation.gliner2" in message


def test_the_default_checkpoint_loads_as_boundary() -> None:
    """The positive control: the declared default really is a boundary checkpoint to the loader."""
    provider = Gliner25FusedExtractor(dict(_EXTRACT_CONFIG))
    with install_gliner2() as fake:
        model = provider._load_model()  # noqa: SLF001 - the load is the subject
    assert model.architecture == "boundary"
    assert fake.last_model_id == DEFAULT_MODEL


def test_a_model_without_the_batched_api_is_a_typed_error_naming_the_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The #192 backstop: a wrapper missing a batched entry point is typed, at LOAD.

    A vendored subset or a fork that kept only the entity half is indistinguishable, to
    ``getattr``, from the attribute being gone — so that is exactly how it is simulated. The fused
    extractor drives BOTH methods, so it must name the one it lost.
    """
    provider = Gliner25FusedExtractor(dict(_EXTRACT_CONFIG))
    with install_gliner2() as fake:
        monkeypatch.delattr(fake, "batch_extract_relations")
        with pytest.raises(ProviderError) as excinfo:
            provider._load_model()  # noqa: SLF001 - the load is the gate under test
    assert "batch_extract_relations" in str(excinfo.value)


def test_the_redactor_does_not_fail_on_a_method_it_never_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Redactor asks no relation question, so a missing relation method must not fail its load.

    The negative control for the check above: ``require_batched_api`` takes the caller's own slice
    of the batched surface, and a Provider that over-declares would turn an irrelevant library
    change into an outage.
    """
    provider = Gliner25PIIRedactor(dict(_REDACT_CONFIG))
    with install_gliner2() as fake:
        monkeypatch.delattr(fake, "batch_extract_relations")
        model = provider._load_model()  # noqa: SLF001 - the load is the subject
    assert model.architecture == "boundary"


def test_max_len_above_the_checkpoint_window_is_a_config_error() -> None:
    """A window wider than the checkpoint can attend to is a boundary that is not there."""
    with pytest.raises(ConfigError) as excinfo:
        Gliner25FusedExtractor({**_EXTRACT_CONFIG, "max_len": MAX_SUPPORTED_MAX_LEN + 1})
    assert str(MAX_SUPPORTED_MAX_LEN) in str(excinfo.value)


# -- the tokenizer-config shape shim (MEASURED on a real load, 2026-08-25) ----------------------


def test_the_shim_is_applied_before_the_backbone_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the shim the REAL load dies; the fake refuses the same way, so absence is RED.

    Measured on 2026-08-25: ``AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1")`` raises
    ``AttributeError: 'list' object has no attribute 'keys'`` inside transformers'
    ``_set_model_specific_special_tokens``, because the checkpoint's ``tokenizer_config.json``
    stores ``extra_special_tokens`` as a 10-element LIST. That is the same field failure ADR-0034
    records for the 1.x PII checkpoint — and it contradicts the old claim that only that one
    checkpoint was affected.

    The shim rewrites a file in the HF cache, which does not exist offline, so what this test can
    honestly assert is the half that a code change can break: that the Provider CALLS it, with the
    right model id, BEFORE the backbone load. Deleting the call turns this red rather than shipping
    a package that dies on its first real load.
    """
    calls: list[str] = []

    def spy(model_id: str, *, provider_name: str) -> bool:  # noqa: ARG001 - name is not asserted
        calls.append(model_id)
        fake.shim_applied = True
        return True

    provider = Gliner25FusedExtractor(dict(_EXTRACT_CONFIG))
    with install_gliner2(tokenizer_config_is_list=True) as fake:
        fake.shim_applied = False
        monkeypatch.setattr("latence_gliner25._backbone.coerce_list_extra_special_tokens", spy)
        provider._load_model()  # noqa: SLF001 - the load ORDER is the subject
    assert calls == [DEFAULT_MODEL]


def test_without_the_shim_the_load_fails_the_way_the_real_stack_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control that makes the test above mean something.

    A fake that cannot refuse proves nothing. With the shim stubbed out to a no-op the load must
    fail — and it must fail as a TYPED ProviderError (the shared load seam maps the native
    AttributeError, ADR-0034), never as the raw native error an operator cannot act on.
    """
    provider = Gliner25PIIRedactor(dict(_REDACT_CONFIG))
    with install_gliner2(tokenizer_config_is_list=True) as fake:
        fake.shim_applied = False
        monkeypatch.setattr(
            "latence_gliner25._backbone.coerce_list_extra_special_tokens",
            lambda model_id, *, provider_name: False,  # noqa: ARG005 - a deliberate no-op shim
        )
        with pytest.raises(ProviderError) as excinfo:
            provider._load_model()  # noqa: SLF001 - the load is the subject
    assert "AttributeError" in str(excinfo.value)


def test_the_shim_coerces_a_list_config_and_no_ops_on_a_dict_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shim itself, over a real on-disk config — the half the spy above cannot cover.

    Driven through the public function with ``hf_hub_download`` pointed at a real file, so this
    exercises the actual read → coerce → atomic-rename path rather than a re-implementation of it.
    """
    import json

    from latence_core.providers import perf

    tokens = ["[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]"]
    config = tmp_path / "tokenizer_config.json"

    def fake_download(model_id: str, filename: str, **_kwargs: object) -> str:  # noqa: ARG001
        return str(config)

    hub = type(sys)("huggingface_hub")
    hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    # A LIST is coerced to the {token: token} dict shape transformers >= 4.57 iterates.
    config.write_text(json.dumps({"extra_special_tokens": tokens, "other": 1}), encoding="utf-8")
    assert perf.coerce_list_extra_special_tokens("acme/ckpt", provider_name="t") is True
    patched = json.loads(config.read_text(encoding="utf-8"))
    assert patched["extra_special_tokens"] == {t: t for t in tokens}
    assert patched["other"] == 1  # nothing else in the config is touched

    # Idempotent: a second run (now dict-shaped) rewrites nothing.
    assert perf.coerce_list_extra_special_tokens("acme/ckpt", provider_name="t") is False
    assert json.loads(config.read_text(encoding="utf-8")) == patched

    # And a config that never had the field is a no-op too.
    config.write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert perf.coerce_list_extra_special_tokens("acme/ckpt", provider_name="t") is False
