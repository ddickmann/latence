"""Provider Conformance suite — the enterprise-readiness gate (P3-F3, ADR-0036 §5).

"Passes conformance" *is* the definition of enterprise-ready (ADR-0036 goal C). This
package is a reusable, **parametrized-over-every-registered-Provider** harness that
asserts a fixed contract every Provider must satisfy to be called production-ready —
the check that makes the model-agnostic claim *auditable* instead of asserted, and the
gate every Wave-1 adapter package must pass to land.

It is dependency-light (ADR-0016): it lives in ``latence-core``, pulls in **no** heavy
model/GPU deps, runs the in-core CPU reference Providers for real, and conformance-checks
GPU/endpoint Providers against a deterministic **stub** (``stub`` module) — their real
quality/throughput is out of scope (needs a rig, #64), so the harness flags them
``conformance-verified against stub; live numbers need a rig`` rather than fabricating a
number.

The moving parts:

* :mod:`~latence_core.conformance.report` — the pure, stage-free counts-only verdict models
  (:class:`ConformanceReport`, :class:`ProviderConformance`, :class:`CheckOutcome`), so the
  Quality Report can carry a conformance roll-up without importing the harness/fixtures.
* :mod:`~latence_core.conformance.fixtures` — the canonical valid inputs, built by running
  the CPU reference Providers over one tiny in-repo document, so every derived record is
  genuinely contract-valid (not a hand-rolled weaker fixture).
* :mod:`~latence_core.conformance.stub` — the deterministic endpoint stub + the openai/gliner
  monkeypatch context managers a stub-scoped Provider is checked against.
* :mod:`~latence_core.conformance.cases` — the per-Capability :class:`ConformanceCase`
  registry: a canonical valid input, an output-validity predicate, and an adversarial input
  with its expected graceful outcome. A Provider is mapped to its Capability by name prefix.
* :mod:`~latence_core.conformance.harness` — :func:`run_conformance` runs C1–C6 for one Provider
  and returns a typed :class:`ProviderConformance`; :func:`build_conformance_report` rolls the
  per-Provider results into a :class:`ConformanceReport` the Quality Report can carry.

The report models are imported eagerly (pure, stage-free); the harness + cases (which pull the
reference-Provider fixtures) are imported **lazily** via module ``__getattr__`` so a bare
``import latence_core.conformance.report`` (what the lean Quality Report needs) does not drag the
whole harness/fixtures chain in.

The C1–C6 checks (ADR-0036 §5):

* **C1** valid typed records — contract-complete, Provenance/Classification + in-range offsets.
* **C2** graceful failure — an adversarial input raises a **typed** ``LatenceError`` (or takes a
  Screening quarantine / PARSE_ERROR path), never a bare crash; **no PII in the error**.
* **C3** license recorded + permissive-or-opt-in (ADR-0012) — every Provider declares a profile;
  ``UNVERIFIED`` never appears with ``license_verified=True``; a restricted license is opt-in.
* **C4** determinism-or-documented — a ``deterministic`` Provider is byte-identical over two runs.
* **C5** resource/device honored — a ``compute="gpu"`` Provider on a CPU host is skipped-with-flag.
* **C6** no secret/PII leak — a non-Redactor does not surface raw PII in non-content fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.conformance.report import (
    CheckOutcome,
    CheckStatus,
    ConformanceReport,
    ProviderConformance,
)

if TYPE_CHECKING:
    from latence_core.conformance.cases import (
        CONFORMANCE_CASES,
        Capability,
        ConformanceCase,
        capability_of,
        case_for_provider,
    )
    from latence_core.conformance.harness import (
        build_conformance_report,
        run_conformance,
    )

# The harness + cases symbols are resolved lazily (they import the reference-Provider fixtures);
# importing only :mod:`~latence_core.conformance.report` (the lean Quality-Report dependency)
# must not drag that chain in.
_LAZY_CASES = frozenset(
    {"CONFORMANCE_CASES", "Capability", "ConformanceCase", "capability_of", "case_for_provider"}
)
_LAZY_HARNESS = frozenset({"build_conformance_report", "run_conformance"})


def __getattr__(name: str) -> Any:
    """Lazily resolve the harness/cases symbols on first access (PEP 562)."""
    if name in _LAZY_CASES:
        from latence_core.conformance import cases

        return getattr(cases, name)
    if name in _LAZY_HARNESS:
        from latence_core.conformance import harness

        return getattr(harness, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "CONFORMANCE_CASES",
    "Capability",
    "CheckOutcome",
    "CheckStatus",
    "ConformanceCase",
    "ConformanceReport",
    "ProviderConformance",
    "build_conformance_report",
    "capability_of",
    "case_for_provider",
    "run_conformance",
]
