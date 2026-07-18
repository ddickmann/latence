"""The license gate — the behaviour that stops a restricted corpus being used by accident."""

from __future__ import annotations

import pytest

from latence_benchmark.licensing import (
    UNVERIFIED,
    DatasetLicense,
    LicenseGateError,
    require_usable,
)

PERMISSIVE = DatasetLicense(
    dataset_license="ODC-BY-1.0",
    code_license="Apache-2.0",
    commercial_use=True,
    attribution_required=True,
    verified_on="2026-07-18",
    source_url="https://example.invalid/licence",
)
RESEARCH_ONLY = DatasetLicense(
    dataset_license="research-only (non-commercial)",
    code_license="Apache-2.0",
    commercial_use=False,
    attribution_required=True,
    verified_on="2026-07-18",
    source_url="https://example.invalid/copyright",
)
UNVERIFIED_CORPUS = DatasetLicense(
    dataset_license=UNVERIFIED,
    code_license="Apache-2.0",
    commercial_use=True,
    attribution_required=False,
    verified_on="2026-07-18",
    source_url="https://example.invalid/unknown",
)
UNVERIFIED_TOOLKIT = DatasetLicense(
    dataset_license="ODC-BY-1.0",
    code_license=UNVERIFIED,
    commercial_use=True,
    attribution_required=False,
    verified_on="2026-07-18",
    source_url="https://example.invalid/unknown",
)


def test_permissive_dataset_passes_the_gate() -> None:
    require_usable("permissive", PERMISSIVE)


def test_non_commercial_dataset_is_refused_by_default() -> None:
    with pytest.raises(LicenseGateError, match="does NOT permit commercial use"):
        require_usable("research-set", RESEARCH_ONLY)


def test_non_commercial_dataset_passes_with_an_explicit_acknowledgement() -> None:
    require_usable("research-set", RESEARCH_ONLY, accept_non_commercial=True)


@pytest.mark.parametrize("licence", [UNVERIFIED_CORPUS, UNVERIFIED_TOOLKIT])
def test_unverified_license_is_never_overridable(licence: DatasetLicense) -> None:
    """ADR-0012: corpus and toolkit are verified SEPARATELY; either gap is a hard stop.

    The acknowledgement flag lifts the non-commercial stop only. If it also lifted this one,
    "we could not confirm the license" would quietly become "assume it is fine".
    """
    for accept in (False, True):
        with pytest.raises(LicenseGateError, match="UNVERIFIED"):
            require_usable("mystery-set", licence, accept_non_commercial=accept)


def test_verified_property_requires_both_licenses() -> None:
    assert PERMISSIVE.verified
    assert not UNVERIFIED_CORPUS.verified
    assert not UNVERIFIED_TOOLKIT.verified
