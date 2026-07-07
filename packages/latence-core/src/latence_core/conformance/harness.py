"""The conformance harness — run C1–C6 for one Provider and roll results into a report (P3-F3).

:func:`run_conformance` constructs a Provider (via the registry), runs the six checks against
its Capability's :class:`~latence_core.conformance.cases.ConformanceCase`, and returns a typed
:class:`ProviderConformance`. The checks (ADR-0036 §5) are:

* **C1** valid typed records — the canonical input emits contract-valid records
  (Provenance/Classification present, offsets in-range), or a legitimately-empty result.
* **C2** graceful failure — the adversarial input raises a **typed**
  :class:`~latence_core.errors.LatenceError` or takes a defined quarantine / PARSE_ERROR path;
  and the raised message / reason carries **no PII** (proven with a PII-bearing input).
* **C3** license recorded + **evidence-bearing** + permissive-or-opt-in (H-B1) — the Provider
  declares a :class:`~latence_core.providers.profile.ProviderProfile`; ``UNVERIFIED`` never
  appears with ``license_verified=True``; a ``license_verified=True`` claim carries its complete
  evidence (source + ISO date + code SPDX, + a weights SPDX when ``model_id`` is set) and its
  license fields are mutually consistent; a restricted (non-permissive) license is flagged opt-in.
* **C4** determinism-or-documented — a ``deterministic`` Provider is byte-identical over two
  runs; a non-deterministic one records its declaration (no assertion).
* **C5** resource/device honored — a ``compute="gpu"`` Provider on a CPU-only host is
  skipped-with-flag (recorded, not run); a CPU/either Provider runs.
* **C6** no secret/PII leak — a non-Redactor Provider does not surface raw PII in its output.

The result models (:class:`CheckOutcome`, :class:`ProviderConformance`, :class:`ConformanceReport`)
are counts-only Pydantic contracts safe to fold into the Quality Report — never record content.
"""

from __future__ import annotations

from latence_core.conformance import fixtures
from latence_core.conformance.cases import Capability, capability_of, case_for_provider
from latence_core.conformance.report import (
    CheckOutcome,
    CheckStatus,
    ConformanceReport,
    ProviderConformance,
)
from latence_core.providers.device import select_device
from latence_core.providers.profile import UNVERIFIED_LICENSE, ProviderProfile, profile_of

# SPDX licenses considered permissive defaults (ADR-0012): commercial use, redistribution, and
# modification with no field-of-use restriction. Anything else (a restricted/Gemma-family
# license) must be flagged opt-in, and UNVERIFIED is an honest gap (never silently verified).
_PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "MPL-2.0",
        "Unlicense",
        "CC0-1.0",
    }
)


def run_conformance(registry: object, provider_name: str) -> ProviderConformance:
    """Run C1–C6 for one registered Provider and return its typed conformance verdict.

    ``registry`` is a :class:`~latence_core.capability.ProviderRegistry` (typed as ``object`` to
    keep this module import-light). The Provider is loaded, its profile read, and its Capability's
    :class:`~latence_core.conformance.cases.ConformanceCase` run. A Provider with **no** case (its
    Capability is not covered) raises — a loud failure, never a silent skip.
    """
    from latence_core.capability import ProviderRegistry

    assert isinstance(registry, ProviderRegistry)
    capability = capability_of(provider_name)
    provider_cls = registry.load(provider_name)
    return _run_conformance_checks_for(
        provider_name=provider_name, capability=capability, provider_cls=provider_cls
    )


def _run_conformance_checks_for(
    *,
    provider_name: str,
    capability: Capability,
    provider_cls: type,
) -> ProviderConformance:
    """Run C1–C6 for a Provider CLASS + its Capability (registry-independent core of the harness).

    Split out so the harness can run the checks on a Provider not in the registry (a synthetic
    GPU-only Provider proving C5 bites, a throwaway broken Provider proving C2/C4 bite) exactly
    as it runs a registered one — the checks are identical, only the class source differs. The
    :class:`~latence_core.conformance.cases.ConformanceCase` is resolved by Provider name (a
    stub-scoped override) else by Capability; a Capability with no case raises (never a silent
    skip).
    """
    case = case_for_provider(provider_name)
    profile = profile_of(provider_cls)

    checks: list[CheckOutcome] = []

    # -- C3 license (does not need the Provider to run) --------------------
    checks.append(_check_c3_license(profile))

    # -- C5 device: decide whether this Provider is skipped-with-flag ------
    device_skipped = False
    skip_reason: str | None = None
    c5, device_skipped, skip_reason = _check_c5_device(profile)
    checks.append(c5)

    if device_skipped:
        # A GPU-only Provider on a CPU-only host: skipped-with-flag, not run and not faked. C1/C2/
        # C4/C6 (which require running it) are recorded as SKIPPED, never fabricated.
        for name in ("C1_valid_records", "C2_graceful_failure", "C4_determinism", "C6_no_pii_leak"):
            checks.append(
                CheckOutcome(
                    check=name,
                    status=CheckStatus.SKIPPED,
                    detail="GPU provider skipped on CPU-only host (needs a GPU rig)",
                )
            )
        return ProviderConformance(
            provider=provider_name,
            capability=capability.value,
            stub_scoped=case.stub_scoped,
            device_skipped=True,
            skip_reason=skip_reason,
            checks=checks,
        )

    # -- C1 valid typed records -------------------------------------------
    produced: list[object] = []
    c1, produced = _check_c1_valid(case, provider_cls)
    checks.append(c1)

    # -- C2 graceful failure (typed error / defined path + no PII) --------
    checks.append(_check_c2_graceful(case, provider_cls))

    # -- C4 determinism-or-documented -------------------------------------
    checks.append(_check_c4_determinism(case, provider_cls, profile))

    # -- C6 no secret/PII leak (non-Redactor) -----------------------------
    checks.append(_check_c6_no_leak(capability, produced))

    return ProviderConformance(
        provider=provider_name,
        capability=capability.value,
        stub_scoped=case.stub_scoped,
        device_skipped=False,
        skip_reason=None,
        checks=checks,
    )


def _check_c3_license(profile: ProviderProfile | None) -> CheckOutcome:
    """C3: a declared profile with a real SPDX id or UNVERIFIED; restricted flagged opt-in.

    H-B1 turns C3 into the real enforcement point the audit wanted: beyond "a profile is
    declared" and the UNVERIFIED invariant, C3 FAILs a provider that claims
    ``license_verified=True`` **without a complete evidence trail** (source + ISO date +
    code SPDX, plus a weights SPDX when ``model_id`` is set), or whose
    ``license``/``license_code``/``license_weights`` are mutually **inconsistent**. The
    :class:`ProviderProfile` validator makes such a profile un-constructable, but C3
    re-asserts the same rule at the provider level (defense-in-depth) so the gate *bites*
    on a fake profile-holder that bypasses the model validator (a bare object carrying the
    raw fields) — the evidence-bearing verification is enforced across every provider, not
    merely relied on to have been enforced at construction.
    """
    if profile is None:
        return CheckOutcome(
            check="C3_license",
            status=CheckStatus.FAIL,
            detail="provider declares no ProviderProfile (every provider must declare one)",
        )
    # The UNVERIFIED invariant is enforced by the ProviderProfile validator, but re-assert at the
    # provider level (defense-in-depth) — an UNVERIFIED license must never be verified.
    if profile.license == UNVERIFIED_LICENSE and profile.license_verified:
        return CheckOutcome(
            check="C3_license",
            status=CheckStatus.FAIL,
            detail="UNVERIFIED license marked verified (forbidden)",
        )
    # H-B1: a verified claim must carry its evidence + be internally consistent, else FAIL —
    # the theater ("license_verified=True" with nothing checked) is caught here too.
    if profile.license_verified:
        evidence_gap = _verified_evidence_gap(profile)
        if evidence_gap is not None:
            return CheckOutcome(
                check="C3_license",
                status=CheckStatus.FAIL,
                detail=f"verified without complete evidence: {evidence_gap}",
            )
    if profile.license == UNVERIFIED_LICENSE:
        # An honest gap — recorded, not a fail (the directive: flag, never fabricate).
        return CheckOutcome(
            check="C3_license",
            status=CheckStatus.SKIPPED,
            detail="license UNVERIFIED (honest gap; not a permissive-or-opt-in claim)",
        )
    if profile.license in _PERMISSIVE_LICENSES:
        return CheckOutcome(
            check="C3_license",
            status=CheckStatus.PASS,
            detail=f"permissive: {profile.license} (verified={profile.license_verified})",
        )
    # A restricted (non-permissive) license: acceptable only if verified (flagged opt-in), which
    # the profile records via license_verified. A restricted + unverified license is a fail.
    if profile.license_verified:
        return CheckOutcome(
            check="C3_license",
            status=CheckStatus.PASS,
            detail=f"restricted opt-in: {profile.license} (verified, flagged)",
        )
    return CheckOutcome(
        check="C3_license",
        status=CheckStatus.FAIL,
        detail=f"restricted license {profile.license} not verified (must be opt-in-flagged)",
    )


def _verified_evidence_gap(profile: object) -> str | None:
    """Return the first missing/inconsistent evidence element for a verified claim, else ``None``.

    Mirrors the :meth:`ProviderProfile._verified_requires_evidence` validator so C3 bites on a
    provider that claims verified without its proof — including a fake profile-holder that
    bypassed the model validator (a bare object carrying the raw fields). Reads the evidence
    fields defensively (a stub may omit them), so a holder that carries only ``license`` +
    ``license_verified`` (the exact audit-flagged shape) is caught as a missing evidence trail.
    """
    source = getattr(profile, "license_source", None)
    verified_on = getattr(profile, "license_verified_on", None)
    code = getattr(profile, "license_code", None)
    weights = getattr(profile, "license_weights", None)
    model_id = getattr(profile, "model_id", None)
    headline = getattr(profile, "license", None)

    if not (source or "").strip():
        return "missing license_source (a citation for where the license was checked)"
    if not (verified_on or "").strip():
        return "missing license_verified_on (ISO date of the check)"
    if not code or code == UNVERIFIED_LICENSE:
        return "missing/UNVERIFIED license_code (the code/library SPDX)"
    if model_id is not None:
        if not weights or weights == UNVERIFIED_LICENSE:
            return "model-bearing provider missing license_weights (ADR-0012: recorded separately)"
        if headline != weights:
            return f"headline license {headline!r} disagrees with license_weights {weights!r}"
    else:
        if weights is not None:
            return f"pure-code provider must not declare license_weights (got {weights!r})"
        if headline != code:
            return f"headline license {headline!r} disagrees with license_code {code!r}"
    return None


def _check_c5_device(
    profile: ProviderProfile | None,
) -> tuple[CheckOutcome, bool, str | None]:
    """C5: route the profile through the device seam; a GPU provider on a CPU host is skipped."""
    decision = select_device("auto", profile)
    if decision.skipped:
        return (
            CheckOutcome(
                check="C5_device",
                status=CheckStatus.SKIPPED,
                detail=f"skipped-with-flag: {decision.reason}",
            ),
            True,
            decision.reason,
        )
    return (
        CheckOutcome(
            check="C5_device",
            status=CheckStatus.PASS,
            detail=f"device honored: {decision.device} ({decision.reason})",
        ),
        False,
        None,
    )


def _check_c1_valid(case: object, provider_cls: type) -> tuple[CheckOutcome, list[object]]:
    """C1: run the Provider on the canonical input and validate the produced records."""
    from latence_core.conformance.cases import ConformanceCase

    assert isinstance(case, ConformanceCase)
    try:
        provider = provider_cls(case.config)
        produced = list(case.invoke(provider))
    except Exception as exc:  # noqa: BLE001 - a C1 failure is a conformance FAIL, not a crash
        return (
            CheckOutcome(
                check="C1_valid_records",
                status=CheckStatus.FAIL,
                detail=f"invoke raised {type(exc).__name__}",
            ),
            [],
        )
    if case.valid_output(produced):
        note = " (stub)" if case.stub_scoped else ""
        return (
            CheckOutcome(
                check="C1_valid_records",
                status=CheckStatus.PASS,
                detail=f"{len(produced)} contract-valid records{note}",
            ),
            list(produced),
        )
    return (
        CheckOutcome(
            check="C1_valid_records",
            status=CheckStatus.FAIL,
            detail=f"{len(produced)} records failed the validity predicate",
        ),
        list(produced),
    )


def _check_c2_graceful(case: object, provider_cls: type) -> CheckOutcome:
    """C2: adversarial input → typed error / defined graceful path, and no PII in the message."""
    from latence_core.conformance.cases import AdversarialOutcome, ConformanceCase

    assert isinstance(case, ConformanceCase)
    try:
        provider = provider_cls(case.config)
        outcome = case.adversarial(provider)
    except Exception as exc:  # noqa: BLE001 - an untyped crash on adversarial input is a C2 FAIL
        # A LatenceError is caught inside the case's _capture_raise and returned as an outcome;
        # reaching here means a NON-typed exception escaped — the exact thing C2 forbids.
        return CheckOutcome(
            check="C2_graceful_failure",
            status=CheckStatus.FAIL,
            detail=f"untyped {type(exc).__name__} escaped (must be a typed LatenceError)",
        )
    assert isinstance(outcome, AdversarialOutcome)
    if not outcome.graceful:
        return CheckOutcome(
            check="C2_graceful_failure",
            status=CheckStatus.FAIL,
            detail="no typed error and no defined graceful path taken",
        )
    # No-PII assertion: the surfaced text (error message / quarantine reason) must not contain the
    # adversarial payload's PII (proven against a PII-bearing input, the P3-F2 discipline).
    leaked = _pii_in_text(outcome.text)
    if leaked:
        return CheckOutcome(
            check="C2_graceful_failure",
            status=CheckStatus.FAIL,
            detail="raised message / reason leaked adversarial PII",
        )
    kind = (
        "typed-error"
        if outcome.raised
        else "quarantine"
        if outcome.quarantined
        else "parse_error"
        if outcome.parse_error
        else "empty"
        if outcome.empty
        else "noop"
    )
    return CheckOutcome(
        check="C2_graceful_failure",
        status=CheckStatus.PASS,
        detail=f"graceful ({kind}), no PII in surfaced text",
    )


def _check_c4_determinism(
    case: object, provider_cls: type, profile: ProviderProfile | None
) -> CheckOutcome:
    """C4: a deterministic Provider is byte-identical over two runs; else record the declaration."""
    from latence_core.conformance.cases import ConformanceCase

    assert isinstance(case, ConformanceCase)
    if profile is None or not profile.deterministic:
        return CheckOutcome(
            check="C4_determinism",
            status=CheckStatus.SKIPPED,
            detail="declared non-deterministic (recorded, no assertion)",
        )
    try:
        first = _dump(case.invoke(provider_cls(case.config)))
        second = _dump(case.invoke(provider_cls(case.config)))
    except Exception as exc:  # noqa: BLE001 - a C4 crash is a conformance FAIL
        return CheckOutcome(
            check="C4_determinism",
            status=CheckStatus.FAIL,
            detail=f"determinism run raised {type(exc).__name__}",
        )
    if first == second:
        return CheckOutcome(
            check="C4_determinism",
            status=CheckStatus.PASS,
            detail="byte-identical over two runs",
        )
    return CheckOutcome(
        check="C4_determinism",
        status=CheckStatus.FAIL,
        detail="declared deterministic but two runs differ",
    )


def _check_c6_no_leak(capability: Capability, produced: list[object]) -> CheckOutcome:
    """C6: a non-Redactor Provider must not surface PII in NON-payload (non-content) record fields.

    The canonical document text legitimately *contains* PII (the whole point of a corpus is to
    carry content) — a Chunker/Parser passing that text through in its ``content`` payload is not
    a leak. C6 targets the repo's no-PII-in-report discipline: PII must not surface in a record's
    **non-content** fields (a ``record_id``, a ``label``, a Provenance field, an Evidence field)
    or in logs/spans, where a downstream consumer that reads only metadata would be exposed. So
    the scan excludes the ``content``/``masked_text`` payload fields and asserts no PII in the
    rest. The Redactor is exempt (it handles PII by design — the S6 AC covers its discipline).
    """
    if capability is Capability.REDACTION:
        return CheckOutcome(
            check="C6_no_pii_leak",
            status=CheckStatus.SKIPPED,
            detail="Redactor handles PII by design (C6 targets non-Redactors)",
        )
    dumped = _dump_non_content(produced)
    if fixtures.CANONICAL_PII_VALUE in dumped:
        return CheckOutcome(
            check="C6_no_pii_leak",
            status=CheckStatus.FAIL,
            detail="a non-content record field contains a raw PII value",
        )
    return CheckOutcome(
        check="C6_no_pii_leak",
        status=CheckStatus.PASS,
        detail="no raw PII in non-content record fields",
    )


def _pii_in_text(text: str) -> bool:
    """True if the surfaced text contains any of the adversarial PII payload's secrets."""
    return (
        fixtures.PII_ADVERSARIAL_SECRET in text
        or fixtures.PII_ADVERSARIAL_EMAIL in text
        or fixtures.CANONICAL_PII_VALUE in text
    )


def _dump(records: object) -> str:
    """Serialize produced records to a string for byte-comparison (Pydantic-aware, all fields)."""
    parts: list[str] = []
    for r in list(records):  # type: ignore[call-overload]
        dumper = getattr(r, "model_dump_json", None)
        parts.append(dumper() if callable(dumper) else str(r))
    return "\n".join(parts)


# Payload text fields a record legitimately carries the document content in — excluded from the
# C6 non-content PII scan (content carrying PII is the corpus, not a leak).
_CONTENT_FIELDS: frozenset[str] = frozenset({"content", "masked_text", "text"})


def _dump_non_content(records: object) -> str:
    """Serialize records EXCLUDING the content payload fields — for the C6 non-content PII scan."""
    parts: list[str] = []
    for r in list(records):  # type: ignore[call-overload]
        dumper = getattr(r, "model_dump", None)
        if callable(dumper):
            data = dumper(mode="json", exclude=_CONTENT_FIELDS)
            parts.append(str(data))
        else:
            parts.append(str(r))
    return "\n".join(parts)


def build_conformance_report(
    registry: object, provider_names: list[str] | None = None
) -> ConformanceReport:
    """Run conformance for every registered Provider (or the given subset) and roll up the report.

    The default sweep is every Provider the registry discovers — the parametrized-over-every-
    Provider gate (ADR-0036 §5). A Provider whose Capability has no case raises (loud, not
    skipped). The returned :class:`ConformanceReport` is counts-only and safe to fold into the
    Quality Report.
    """
    from latence_core.capability import ProviderRegistry

    assert isinstance(registry, ProviderRegistry)
    names = provider_names if provider_names is not None else registry.names()
    results = [run_conformance(registry, name) for name in names]
    passed = sum(1 for r in results if r.passed)
    return ConformanceReport(
        providers_total=len(results),
        providers_passed=passed,
        providers_failed=len(results) - passed,
        providers_stub_scoped=sum(1 for r in results if r.stub_scoped),
        providers_device_skipped=sum(1 for r in results if r.device_skipped),
        results=results,
    )
