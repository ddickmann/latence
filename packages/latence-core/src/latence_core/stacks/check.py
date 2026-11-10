"""``check_stack`` — the OFFLINE stack check: no corpus, no models, no run, CI-gateable (C8).

The framework already had good static checks. What it did not have was a way for an operator to
*reach* them. Pydantic + the DAG validator (:mod:`latence_core.pipeline`), the capability guard
(:mod:`latence_core.stacks.capability_guard`) and the phase-boundary invariant
(:mod:`latence_core.stacks.phase_boundary`) were composed in exactly one function — the setup
wizard's ``validate_config_text`` — whose only caller was ``latence setup``. So:

* ``latence stack validate`` never ran the capability guard at all. It staged a bundled corpus and
  ran the pipeline TWICE. The guard exists precisely because a CPU host device-skips GPU Stages
  and therefore never reaches the Runner's dispatch-time ``isinstance`` check (ADR-0036 §3) — and
  it was absent from the harness meant to catch that class of bug on CPU.
* ``latence run`` / ``latence process`` did neither, going straight to execution.
* The emitted stack header tells the operator "Edit by hand freely — it is a plain, diffable stack
  config", and there was no cheap command to re-check a hand-edited file. The only option was a
  full local run.

This module is the single composition point, and :func:`check_stack` is its whole interface: one
call, one :class:`StackCheck`. Everything a caller used to have to know — which three checkers to
call, in which order, which exception each raises, how to turn three different finding types into
one message, and which nested sub-Provider keys exist — now lives here. The enterprise value is
the reason it is a *command*: a fleet of committed ``latence.stack.yaml`` files across business
units could not be lint-gated in CI without running each one.

What it checks, in one pass over the DAG:

``structure``
    the document parses into a :class:`~latence_core.pipeline.Pipeline` — Pydantic types, unknown
    fields, duplicate Stage names, unknown/self ``depends_on``, and the ``extract_on: masked``
    ordering rule.
``phase_boundary``
    no chunk-level Stage transitively depends on a corpus-level one (T1).
``capability``
    every installed Stage Provider structurally satisfies the Protocol its Capability declares
    (:func:`~latence_core.stacks.capability_guard.check_stack_capabilities`) — class-only, no
    instantiation, torch-free.
``provider``
    every named Provider resolves in the ``latence.providers`` registry. A name that is merely not
    installed on THIS host is a WARNING (ADR-0016 thin core: the GPU Providers are legitimately
    absent from a CPU CI box); a name that is a near-miss for a registered one is an ERROR with
    the suggestion, because that is a typo, not a missing package.
``sub_provider``
    every nested ``{provider, config}`` block (:mod:`latence_core.subprovider`) — shape, Provider
    resolution, and a class-only check against the slot's declared Protocol.
``config_key``
    a config key ADR-0068 removed (the closed list in :mod:`latence_core.pipeline`, which a run
    refuses), and a ``{provider, ...}`` block under a Capability that declares no such slot
    (silently ignored at run time).
``wiring``
    a Stage whose parents cannot supply a carrier its Capability REQUIRES (a Relation-Extraction
    Stage with no Entity-Extraction parent). Both halves are declarations — what a Capability's
    Stage must receive
    (:attr:`~latence_core.capability_descriptor.CapabilityDescriptor.input_plan`) and what a Stage
    checkpoints — so the run-time ``ContractError`` a full extraction would eventually raise is
    decidable here, with the same message.

Severity, and why a WARNING exists at all: this check runs on hosts that deliberately do not have
the heavy Provider packages installed. Treating "not installed" as an error would make the command
unusable in exactly the CI environment it is for, and treating it as silence would hide typos. The
near-miss split gives both — a typo is an ERROR everywhere, an absent package is a WARNING
everywhere — with no host-dependent verdict flapping. ``ok`` is "no ERRORs".

Offline by construction: nothing here reads a corpus, constructs a Provider, opens a socket or
writes a file. A Provider class is *imported* to inspect its members (the same cheap import
``latence stack validate``'s device pre-flight already does); an import that fails is skipped, not
a finding, because an un-importable class cannot be checked.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from latence_core.capability_descriptor import input_plan_for
from latence_core.errors import ConfigError
from latence_core.pipeline import Pipeline, removed_config_keys
from latence_core.reuse_overlay import stage_produces
from latence_core.run_store import stage_output_type
from latence_core.subprovider import (
    class_satisfies_capability,
    slots_for,
    sub_provider_ref,
    undeclared_sub_provider_keys,
)

if TYPE_CHECKING:
    from latence_core.capability import ProviderRegistry
    from latence_core.pipeline import Stage

__all__ = ["Finding", "Severity", "StackCheck", "check_stack"]

# How close a mistyped Provider name has to be to a registered one before we call it a typo rather
# than an uninstalled package. Measured against the real registry (``disambiguation.cascad`` vs
# ``disambiguation.cascade`` = 0.98, ``embedding.hashng`` vs ``embedding.hashing`` = 0.97) and
# against genuinely-absent heavy Providers (``embedding.granite_r2`` vs its nearest in-core
# neighbour = 0.60, ``parser.lighton_vllm`` = 0.49). 0.9 separates the two populations with a wide
# margin in both directions, so neither verdict is a coin flip.
_TYPO_RATIO = 0.9


class Severity(StrEnum):
    """Whether a finding blocks (``error``) or merely informs (``warning``).

    Only ``error`` fails :attr:`StackCheck.ok`. A ``warning`` is for something this HOST cannot
    decide — the one case being a Provider package that is legitimately not installed here.
    """

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One thing wrong with a stack, addressed to the operator who has to fix it.

    ``check`` names the family (``structure`` / ``phase_boundary`` / ``capability`` / ``provider``
    / ``sub_provider`` / ``config_key`` / ``wiring``) so CI can filter; ``stage`` is the Stage it
    belongs to (``None`` for a whole-document finding); ``message`` says what to change.
    """

    check: str
    severity: Severity
    message: str
    stage: str | None = None

    def describe(self) -> str:
        """A single, greppable line: ``error [capability] stage 'extract': …``."""
        where = f" stage {self.stage!r}:" if self.stage else ""
        return f"{self.severity.value} [{self.check}]{where} {self.message}"


@dataclass(frozen=True)
class StackCheck:
    """The result of one offline check: every finding, and the single gate ``ok``.

    ``ok`` is False iff at least one ERROR was found — the exit code ``latence stack check``
    returns and the pre-flight ``latence stack validate`` / ``latence process`` gate on.
    """

    findings: tuple[Finding, ...]

    @property
    def ok(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    def render(self) -> str:
        """A deterministic, human- and CI-readable report (one line per finding + a verdict)."""
        lines = [f.describe() for f in self.findings]
        verdict = "OK" if self.ok else "FAILED"
        lines.append(
            f"stack check {verdict} — {len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        return "\n".join(lines)


def check_stack(
    stack: Pipeline | str | Path,
    *,
    registry: ProviderRegistry | None = None,
) -> StackCheck:
    """Check a stack offline — no corpus, no models, no run — and return every finding.

    ``stack`` is a already-parsed :class:`~latence_core.pipeline.Pipeline`, the YAML **text**, or a
    :class:`~pathlib.Path` to the file. Given text or a path the structural/DAG parse is itself a
    check, so a malformed document comes back as a :class:`Finding` rather than an exception — a
    linter that raises on the input it exists to lint is not a linter.

    Findings are ordered deterministically: whole-document checks first, then each Stage in
    declaration order with its checks in a fixed order, so two runs over the same file produce
    byte-identical output (the determinism invariant) and a CI diff is meaningful.
    """
    from latence_core.capability import ProviderRegistry

    if isinstance(stack, Path):
        return check_stack(stack.read_text(encoding="utf-8"), registry=registry)
    if isinstance(stack, str):
        try:
            pipeline = Pipeline.from_yaml_text(stack)
        except Exception as exc:  # noqa: BLE001 - a malformed stack is a FINDING, not a crash
            return StackCheck(
                (
                    Finding(
                        check="structure",
                        severity=Severity.ERROR,
                        message=f"stack config is structurally invalid: {exc}",
                    ),
                )
            )
        return check_stack(pipeline, registry=registry)

    pipeline = stack
    registry = registry or ProviderRegistry()
    installed = sorted(registry.names())

    findings: list[Finding] = [*_phase_boundary_findings(pipeline)]
    capability_findings = _capability_findings(pipeline, registry)
    for stage in pipeline.stages:
        findings.extend(_provider_findings(stage.provider, stage, "provider", installed, registry))
        findings.extend(capability_findings.get(stage.name, ()))
        findings.extend(_sub_provider_findings(stage, installed, registry))
        findings.extend(_config_key_findings(stage))
        findings.extend(_wiring_findings(stage, pipeline))
    return StackCheck(tuple(findings))


# -- the individual checks -----------------------------------------------------


def _phase_boundary_findings(pipeline: Pipeline) -> tuple[Finding, ...]:
    """T1: no chunk-level Stage transitively depends on a corpus-level one."""
    from latence_core.stacks.phase_boundary import check_phase_boundary

    return tuple(
        Finding(
            check="phase_boundary",
            severity=Severity.ERROR,
            message=violation.describe(),
            stage=violation.stage,
        )
        for violation in check_phase_boundary(pipeline)
    )


def _capability_findings(
    pipeline: Pipeline, registry: ProviderRegistry
) -> dict[str, tuple[Finding, ...]]:
    """Stage-level capability wiring, from the guard, grouped by Stage name.

    The guard's own nested-``embedder`` findings are dropped here: this module checks EVERY nested
    slot (not just ``embedder``, and not just on Export) in :func:`_sub_provider_findings`, from
    the one slot declaration, so keeping the guard's narrower copy would report the same mis-wired
    embedder twice with two different wordings.
    """
    from latence_core.stacks.capability_guard import check_stack_capabilities

    grouped: dict[str, list[Finding]] = {}
    for finding in check_stack_capabilities(pipeline, registry):
        if finding.embedder:
            continue
        grouped.setdefault(finding.stage_name, []).append(
            Finding(
                check="capability",
                severity=Severity.ERROR,
                message=finding.describe(),
                stage=finding.stage_name,
            )
        )
    return {name: tuple(items) for name, items in grouped.items()}


def _provider_findings(
    provider: str,
    stage: Stage,
    check: str,
    installed: list[str],
    registry: ProviderRegistry,
    *,
    slot: str | None = None,
) -> tuple[Finding, ...]:
    """Does this Provider name resolve? A near-miss is a typo (ERROR); an absence is a WARNING.

    Shared by the Stage-level Provider and every nested sub-Provider, so a mistyped
    ``embedding.hashng`` inside a config block gets the same "did you mean" a mistyped
    Stage Provider gets.
    """
    if provider in registry.names():
        return ()
    where = f"sub-Provider slot {slot!r}: " if slot else ""
    suggestion = difflib.get_close_matches(provider, installed, n=1, cutoff=_TYPO_RATIO)
    if suggestion:
        return (
            Finding(
                check=check,
                severity=Severity.ERROR,
                message=(
                    f"{where}unknown Provider {provider!r} — did you mean {suggestion[0]!r}? "
                    f"(no entry point of that name is registered)"
                ),
                stage=stage.name,
            ),
        )
    return (
        Finding(
            check=check,
            severity=Severity.WARNING,
            message=(
                f"{where}Provider {provider!r} is not installed on this host, so its wiring "
                f"cannot be checked offline (install its package to check it here)"
            ),
            stage=stage.name,
        ),
    )


def _sub_provider_findings(
    stage: Stage, installed: list[str], registry: ProviderRegistry
) -> tuple[Finding, ...]:
    """Every nested ``{provider, config}`` block: shape, resolution, and the Protocol it must fit.

    Iterates the slots the Stage's Capability *declares* (:func:`~latence_core.subprovider.
    slots_for`), so a resolver's ``embedder`` and the universal Parser's ``ocr`` are checked
    exactly like the Export's ``embedder``. The Protocol check is class-only: the Provider is
    imported but never constructed, so this stays torch-free and offline.
    """
    findings: list[Finding] = []
    for slot in slots_for(stage.capability):
        spec = stage.config.get(slot.key)
        if spec is None:
            continue
        try:
            ref = sub_provider_ref(spec, capability=stage.capability, key=slot.key)
        except ConfigError as exc:
            findings.append(
                Finding(
                    check="sub_provider",
                    severity=Severity.ERROR,
                    message=str(exc),
                    stage=stage.name,
                )
            )
            continue
        if ref is None:  # pragma: no cover - spec is not None, so the ref is not either
            continue
        resolution = _provider_findings(
            ref.provider, stage, "sub_provider", installed, registry, slot=slot.key
        )
        findings.extend(resolution)
        if resolution:
            continue
        try:
            provider_cls = registry.load(ref.provider)
        except Exception:  # noqa: BLE001 - an un-importable class cannot be inspected, only skipped
            continue
        if not class_satisfies_capability(provider_cls, ref.protocol):
            findings.append(
                Finding(
                    check="sub_provider",
                    severity=Severity.ERROR,
                    message=(
                        f"sub-Provider slot {slot.key!r}: {ref.provider!r} does not satisfy the "
                        f"{ref.protocol.__name__} capability this slot requires"
                    ),
                    stage=stage.name,
                )
            )
    return tuple(findings)


def _wiring_findings(stage: Stage, pipeline: Pipeline) -> tuple[Finding, ...]:
    """A Stage wired to parents that cannot supply a carrier its Capability REQUIRES.

    The static half of the run-time wiring check. A Capability declares what its Stage must receive
    (:attr:`~latence_core.capability_descriptor.CapabilityDescriptor.input_plan`), and a Stage's
    carrier is likewise declared, so "no parent of this Stage produces the DocumentRecords a
    Profiling Stage profiles" is decidable with no corpus and no run — and it is the mistake that
    otherwise surfaces only after a full extraction, as a typed ``ContractError`` from the
    Runner's dispatch. The message IS the Runner's, formatted from the same declaration, so an
    operator reads one wording whichever end catches it (``got:`` naming the carriers the wiring
    actually delivers rather than the ones a run observed).

    Only the "required carrier" half is checked, and only for a Stage that HAS parents. The
    exclusive rules (Graph Assembly / Graph Completion reject *any other* carrier) are deliberately
    left to the run: a mixed-carrier ancestor's contribution is a per-record fact, and a linter
    that guesses would fail stacks that run correctly. A parentless Stage is not a finding either —
    an empty input is legitimately empty, exactly as at run time.
    """
    plan = input_plan_for(stage.capability)
    by_name = {s.name: s for s in pipeline.stages}
    parents = [by_name[name] for name in stage.depends_on if name in by_name]
    if not plan.requires or not parents:
        return ()
    delivered = sorted({stage_output_type(parent, pipeline).__name__ for parent in parents})
    return tuple(
        Finding(
            check="wiring",
            severity=Severity.ERROR,
            message=str(requirement.error(stage, delivered)),
            stage=stage.name,
        )
        for requirement in plan.requires
        if not any(stage_produces(parent, pipeline, requirement.carrier) for parent in parents)
    )


def _config_key_findings(stage: Stage) -> tuple[Finding, ...]:
    """Config keys that go nowhere: removed keys first, then sub-Provider blocks nobody reads.

    A key ADR-0068 removed (:data:`~latence_core.pipeline.REMOVED_CONFIG_KEYS`, the one list the
    run-time refusal reads too) is reported whatever its value. An undeclared sub-Provider block is
    a wiring mistake that otherwise never surfaces at all: it is read by nobody. A removed key
    written as a block is reported once, as removed, because that says what to do about it.
    """
    removed = removed_config_keys(stage)
    removed_findings = tuple(
        Finding(
            check="config_key",
            severity=Severity.ERROR,
            message=(
                f"config key {key!r} was removed by ADR-0068 with the query-time retrieval "
                f"tooling; nothing reads it any more, and a run refuses a Pipeline that carries "
                f"it. Delete the key."
            ),
            stage=stage.name,
        )
        for key in removed
    )
    return removed_findings + tuple(
        Finding(
            check="config_key",
            severity=Severity.ERROR,
            message=(
                f"config key {key!r} looks like a sub-Provider block, but capability "
                f"{stage.capability} declares no such slot — it is IGNORED at run time "
                f"(declared slots: {[s.key for s in slots_for(stage.capability)] or 'none'})"
            ),
            stage=stage.name,
        )
        for key in undeclared_sub_provider_keys(stage)
        if key not in removed
    )
