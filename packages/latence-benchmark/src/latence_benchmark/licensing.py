"""The dataset license gate (ADR-0012, applied to DATA rather than to weights).

ADR-0012 makes the framework verify a model's *weights* license and its *code* license
**separately** and cite both. A public benchmark dataset has exactly the same split and it
bites harder, because a benchmark is usually shipped as two artifacts under two different
licenses: a permissive evaluation *toolkit* and a much more restricted *corpus*. Reading the
repository badge and stopping there is the failure mode this module exists to prevent —
OmniDocBench, the obvious choice for a document-parsing benchmark, ships Apache-2.0 **code**
next to a corpus whose copyright statement says it "is for research purposes only and not for
commercial use" (verified 2026-07-18, see :mod:`latence_benchmark.datasets`).

So every :class:`~latence_benchmark.datasets.DatasetSpec` carries a :class:`DatasetLicense`
recording *both* licenses, whether commercial use is permitted, and where that was verified —
and :func:`require_usable` is called **before any acquisition happens**, so a restricted corpus
cannot be silently downloaded, vendored or published against.

Two rules, both structural rather than advisory:

* An **unverified** license is a hard stop with **no override**. "We could not confirm it" never
  degrades into "assume it is fine".
* A **non-commercial** corpus is a stop that the operator can lift only by opting in explicitly
  (``--accept-non-commercial``), which records the acknowledgement in the results artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Sentinel for a license we could NOT verify. Never a guess, never overridable (ADR-0012).
UNVERIFIED = "UNVERIFIED"


class LicenseGateError(RuntimeError):
    """A dataset may not be acquired or scored under its recorded license."""


@dataclass(frozen=True)
class DatasetLicense:
    """The separately verified license record of one public benchmark dataset.

    :param dataset_license: License of the CORPUS itself (SPDX id, a plain-language term such as
        ``"research-only"``, or :data:`UNVERIFIED`).
    :param code_license: License of the benchmark's evaluation TOOLKIT, verified separately —
        the two routinely differ (ADR-0012).
    :param commercial_use: Whether the corpus license permits commercial use. ``False`` gates
        acquisition behind an explicit operator acknowledgement.
    :param attribution_required: Whether the corpus license requires attribution (e.g. ODC-BY).
    :param verified_on: ISO date on which a human verified both licenses at ``source_url``.
    :param source_url: Where the license text/statement was read.
    :param notes: Verbatim license wording or caveats worth carrying into the results page.
    """

    dataset_license: str
    code_license: str
    commercial_use: bool
    attribution_required: bool
    verified_on: str
    source_url: str
    notes: str = ""

    @property
    def verified(self) -> bool:
        """True only when BOTH the corpus and the toolkit license were verified."""
        return UNVERIFIED not in (self.dataset_license, self.code_license)


def require_usable(
    dataset_name: str,
    license_: DatasetLicense,
    *,
    accept_non_commercial: bool = False,
) -> None:
    """Raise :class:`LicenseGateError` unless ``dataset_name`` may be acquired and scored.

    Call this *before* acquisition, not after — the point is that restricted bytes never land
    on disk. ``accept_non_commercial`` lifts the non-commercial stop **only**; an unverified
    license is not overridable by any flag.
    """
    if not license_.verified:
        msg = (
            f"Dataset {dataset_name!r} has an UNVERIFIED license "
            f"(corpus={license_.dataset_license!r}, code={license_.code_license!r}). "
            "ADR-0012 requires the corpus license and the evaluation-toolkit license to be "
            "verified SEPARATELY and cited before use; an unverified license is never assumed "
            "permissive and cannot be overridden by a flag. Verify it at "
            f"{license_.source_url} and record the result on the DatasetSpec."
        )
        raise LicenseGateError(msg)
    if not license_.commercial_use and not accept_non_commercial:
        msg = (
            f"Dataset {dataset_name!r} is licensed {license_.dataset_license!r}, which does NOT "
            "permit commercial use, so it is not acquired by default and its numbers are not "
            "published as part of the framework's headline results. "
            f"Source: {license_.source_url}. {license_.notes} "
            "If your use is genuinely research-only, re-run with --accept-non-commercial; the "
            "acknowledgement is recorded in the results artifact."
        )
        raise LicenseGateError(msg)
