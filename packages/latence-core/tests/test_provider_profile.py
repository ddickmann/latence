"""P3-F1 — the ProviderProfile descriptor contract + declaration reader (ADR-0036 §2).

Deterministic, seeded (no I/O, no model download): profile validation (incl. the
UNVERIFIED invariant), frozen/immutable enforcement, and the two declaration forms
(``profile`` classmethod / ``PROFILE`` ClassVar) the registry detects via
``profile_of``.
"""

from __future__ import annotations

import pytest
from latence_core import ProviderProfile
from latence_core.providers.profile import (
    REPO_LICENSE_CODE,
    UNVERIFIED_LICENSE,
    HasProfile,
    profile_of,
)
from pydantic import ValidationError


def _valid(**overrides: object) -> ProviderProfile:
    base: dict[str, object] = {
        "compute": "cpu",
        "memory_mb": 64,
        "model_id": None,
        "license": "Apache-2.0",
        "license_verified": True,
        "license_code": "Apache-2.0",
        "license_weights": None,
        "license_source": "LICENSE",
        "license_verified_on": "2026-07-06",
        "deterministic": True,
        "batch": False,
        "cost_per_1k": None,
    }
    base.update(overrides)
    return ProviderProfile(**base)  # type: ignore[arg-type]


def test_happy_path_cpu_pure_code() -> None:
    p = _valid()
    assert p.compute == "cpu"
    assert p.model_id is None
    assert p.license == "Apache-2.0"
    assert p.license_verified is True
    assert p.cost_per_1k is None


def test_frozen_is_immutable() -> None:
    p = _valid()
    with pytest.raises(ValidationError):
        p.memory_mb = 999


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ProviderProfile(  # type: ignore[call-arg]
            compute="cpu",
            memory_mb=1,
            license="MIT",
            license_verified=True,
            deterministic=True,
            batch=False,
            surprise="nope",
        )


def test_negative_memory_rejected() -> None:
    with pytest.raises(ValidationError):
        _valid(memory_mb=-1)


def test_negative_cost_rejected() -> None:
    with pytest.raises(ValidationError):
        _valid(cost_per_1k=-0.01)


def test_endpoint_cost_allowed() -> None:
    p = _valid(
        compute="either",
        cost_per_1k=0.5,
        model_id="vendor/model",
        license_weights="Apache-2.0",
        license_source="https://vendor/model-card",
    )
    assert p.cost_per_1k == 0.5


def test_unverified_invariant_rejects_verified_unknown() -> None:
    # The research-diligence directive (ADR-0012): an UNVERIFIED license can NEVER be
    # silently verified. license=UNVERIFIED + license_verified=True must fail.
    with pytest.raises(ValidationError, match="license_verified=False"):
        _valid(license=UNVERIFIED_LICENSE, license_verified=True)


def test_unverified_with_false_is_allowed() -> None:
    p = _valid(license=UNVERIFIED_LICENSE, license_verified=False, model_id="x/y", deterministic=False)
    assert p.license == UNVERIFIED_LICENSE
    assert p.license_verified is False


# -- H-B1: evidence-bearing verification (the theater is now impossible) -------


def test_verified_pure_code_carries_evidence() -> None:
    # The reconciled happy path: a pure-code provider's verified claim names code SPDX, a
    # citation, and a date; weights is None (ships no model).
    p = _valid()
    assert p.license_verified is True
    assert p.license_code == "Apache-2.0"
    assert p.license_weights is None
    assert p.license_source == "LICENSE"
    assert p.license_verified_on == "2026-07-06"


def test_verified_without_source_raises() -> None:
    # The exact audit finding: license="Apache-2.0", license_verified=True with NO evidence.
    with pytest.raises(ValidationError, match="license_source"):
        _valid(license_source=None)


def test_verified_with_whitespace_only_source_raises() -> None:
    # A blank/whitespace citation must not satisfy the letter of the evidence requirement
    # while carrying no actual proof — the citation has to be substantive.
    with pytest.raises(ValidationError, match="license_source"):
        _valid(license_source="   ")


def test_verified_without_date_raises() -> None:
    with pytest.raises(ValidationError, match="license_verified_on"):
        _valid(license_verified_on=None)


def test_verified_with_whitespace_only_date_raises() -> None:
    with pytest.raises(ValidationError, match="license_verified_on"):
        _valid(license_verified_on="  ")


def test_verified_without_code_spdx_raises() -> None:
    with pytest.raises(ValidationError, match="license_code"):
        _valid(license_code=UNVERIFIED_LICENSE)


def test_model_bearing_verified_requires_weights_spdx() -> None:
    # ADR-0012: weights recorded SEPARATELY. A model-bearing verified claim without a weights
    # SPDX is un-constructable.
    with pytest.raises(ValidationError, match="license_weights"):
        _valid(
            model_id="urchade/gliner_multi-v2.1",
            license_source="https://huggingface.co/urchade/gliner_multi-v2.1",
            deterministic=False,
        )


def test_model_bearing_headline_must_equal_weights() -> None:
    # license / license_code / license_weights cannot silently disagree.
    with pytest.raises(ValidationError, match="disagree|must equal"):
        _valid(
            model_id="x/y",
            license="MIT",
            license_weights="Apache-2.0",
            license_source="https://hf.co/x/y",
            deterministic=False,
        )


def test_pure_code_headline_must_equal_code() -> None:
    with pytest.raises(ValidationError, match="disagree|must equal"):
        _valid(license="MIT", license_code="Apache-2.0")


def test_pure_code_must_not_declare_weights() -> None:
    # A code-only provider (model_id None) declaring a weights SPDX is inconsistent.
    with pytest.raises(ValidationError, match="license_weights"):
        _valid(license_weights="Apache-2.0")


def test_malformed_verified_on_date_raises() -> None:
    # A non-ISO date is rejected even structurally — a real YYYY-MM-DD, not a free-form note.
    with pytest.raises(ValidationError, match="ISO date"):
        _valid(license_verified_on="July 6 2026")


def test_model_bearing_verified_happy_path() -> None:
    p = _valid(
        model_id="urchade/gliner_multi-v2.1",
        license="Apache-2.0",
        license_weights="Apache-2.0",
        license_source="https://huggingface.co/urchade/gliner_multi-v2.1",
        deterministic=False,
    )
    assert p.license_weights == "Apache-2.0"
    assert p.license == p.license_weights


def test_unverified_needs_no_evidence() -> None:
    # An honest gap: UNVERIFIED + not verified needs no source/date (nothing was checked).
    p = _valid(
        license=UNVERIFIED_LICENSE,
        license_verified=False,
        license_source=None,
        license_verified_on=None,
        model_id="x/y",
        deterministic=False,
    )
    assert p.license_verified is False
    assert p.license_source is None


def test_empty_license_rejected() -> None:
    with pytest.raises(ValidationError):
        _valid(license="")


def test_invalid_compute_rejected() -> None:
    with pytest.raises(ValidationError):
        _valid(compute="tpu")


# -- declaration reader (profile_of + HasProfile) -----------------------------


class _ClassmethodProvider:
    @classmethod
    def profile(cls) -> ProviderProfile:
        return _valid(memory_mb=32)


class _ClassVarProvider:
    PROFILE = _valid(
        memory_mb=48,
        compute="gpu",
        model_id="m",
        license="MIT",
        license_weights="MIT",
        license_source="https://hf.co/m",
    )


class _NoProfileProvider:
    pass


class _BadClassmethodProvider:
    @classmethod
    def profile(cls) -> object:
        return "not a profile"


class _BadClassVarProvider:
    PROFILE = "not a profile"


def test_profile_of_reads_classmethod() -> None:
    prof = profile_of(_ClassmethodProvider)
    assert prof is not None
    assert prof.memory_mb == 32


def test_profile_of_reads_classvar() -> None:
    prof = profile_of(_ClassVarProvider)
    assert prof is not None
    assert prof.compute == "gpu"
    assert prof.model_id == "m"


def test_profile_of_none_when_undeclared() -> None:
    # A provider that declares no profile returns None — never a fabricated default.
    assert profile_of(_NoProfileProvider) is None


class _ProfilerCapabilityShaped:
    """A Provider whose Capability method is *named* ``profile`` (like the Profiler seam).

    The ``Profiler`` Capability's method is ``profile(self, documents, mentions)`` — an
    INSTANCE method, not a ProviderProfile declaration. ``profile_of`` must NOT invoke it
    (which would raise a spurious TypeError) and must read it as "no profile declared"
    (``None``) — a Capability method named ``profile`` is not a malformed declaration.
    """

    def profile(self, documents: object, mentions: object) -> object:  # noqa: ARG002
        return []


class _ProfilerCapabilityWithProfileClassVar:
    """A Profiler-shaped Provider that ALSO declares a real profile via the ``PROFILE`` ClassVar.

    The instance ``profile`` method (the Capability) must not shadow the ``PROFILE`` ClassVar:
    ``profile_of`` reads the declared profile, not "no profile".
    """

    PROFILE = _valid(compute="cpu", memory_mb=8)

    def profile(self, documents: object, mentions: object) -> object:  # noqa: ARG002
        return []


def test_profile_of_ignores_capability_profile_instance_method() -> None:
    # The Profiler-name-collision regression (P3-F2): a Capability method named ``profile``
    # is NOT a profile declaration — read as None, never called-with-no-args → spurious error.
    assert profile_of(_ProfilerCapabilityShaped) is None


def test_profile_of_reads_classvar_despite_capability_profile_method() -> None:
    prof = profile_of(_ProfilerCapabilityWithProfileClassVar)
    assert prof is not None
    assert prof.memory_mb == 8


def test_profile_of_raises_on_bad_classmethod() -> None:
    with pytest.raises(TypeError, match="did not return a ProviderProfile"):
        profile_of(_BadClassmethodProvider)


def test_profile_of_raises_on_bad_classvar() -> None:
    with pytest.raises(TypeError, match="not a ProviderProfile"):
        profile_of(_BadClassVarProvider)


def test_hasprofile_protocol_is_structural() -> None:
    # runtime_checkable: the classmethod provider satisfies HasProfile structurally.
    assert isinstance(_ClassmethodProvider, HasProfile)
    assert not isinstance(_NoProfileProvider, HasProfile)


def test_in_core_reference_providers_declare_honest_profiles() -> None:
    # The in-core references declare CPU, pure-code, verified profiles carrying the REPO's
    # own licence (they ship no weights, so their headline SPDX IS the framework's).
    from latence_core.stages.chunk import MarkdownChunker
    from latence_core.stages.entity import GazetteerEntityExtractor
    from latence_core.stages.relation import PatternRelationExtractor
    from latence_core.stages.screening import SignatureSizeIntakeScreener

    for cls in (
        GazetteerEntityExtractor,
        PatternRelationExtractor,
        MarkdownChunker,
        SignatureSizeIntakeScreener,
    ):
        prof = profile_of(cls)
        assert prof is not None, cls.__name__
        assert prof.compute == "cpu"
        assert prof.model_id is None
        assert prof.license == REPO_LICENSE_CODE
        assert prof.license_verified is True
        assert prof.deterministic is True
        assert prof.cost_per_1k is None
