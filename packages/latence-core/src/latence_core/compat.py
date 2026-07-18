"""Blackwell / CUDA-13 compatibility matrix — a matrix the build checks (issue #199).

**The problem.** The live 519-page German-corpus run on a Blackwell (``sm_120``) / CUDA-13 pod on
2026-07-17 burned hours on failures that were nobody's bug and everybody's surprise: FlashInfer
refusing a capable device because its check misparses compute capability 12.0, a native
``nvrtc``/TorchScript crash inside the mdeberta encoder, ABI-broken prebuilt wheels, a multimodal
OOM at the documented concurrency, and a ``transformers`` major split with no in-process
resolution. #195/#196/#198 turned each *fix* into a pinned environment. What was still missing is
the **map**: which Provider and which serving dependency is known-good on which CUDA / torch /
``transformers`` combination — and, just as importantly, which combination nobody has actually
measured.

**The rule this module exists to enforce: every cell is MEASURED or explicitly ``UNVERIFIED``.**
A matrix whose cells can be filled in from plausible inference is worse than no matrix, because
an operator trusts it. So the shape is a specification, not prose:

* ``deploy/env/compatibility-matrix.yaml`` is the SINGLE SOURCE OF TRUTH, parsed by
  :func:`load_matrix`;
* ``docs/compatibility-matrix.md`` is **generated** from it by :func:`render_matrix` and
  byte-compared, so the published table can never drift from the checked data;
* :func:`verify_matrix` reads the matrix *together with the committed environments* and returns
  :class:`~latence_core.envspec.Finding` s.

What :func:`verify_matrix` makes impossible:

* a learned Provider or a serving dependency of a committed environment with **no cell**
  (``E-MATRIX-CELL-UNCOVERED``) — coverage is derived from ``deploy/env/*/environment.yaml``, so
  adding a Provider or a boundary to an environment turns the matrix red until it is mapped;
* a cell that claims ``verified`` without the four things a measurement needs — date, hardware,
  the exact command and the observed output (``E-MATRIX-EVIDENCE-MISSING``);
* a cell that says ``UNVERIFIED`` without the exact command the next operator must run, pointing
  at a script that actually exists (``E-MATRIX-REPRODUCE-MISSING`` /
  ``E-MATRIX-REPRODUCE-UNRUNNABLE``) — "unknown" must still be actionable;
* a distribution cell whose version disagrees with that environment's committed lock
  (``E-MATRIX-VERSION-DRIFT``) — the matrix describes the pinned environments or it describes
  nothing;
* dropping one of the live-run failure modes (:data:`REQUIRED_FAILURE_MODES`) or recording one
  without a resolution that is enforced by a file in the repo
  (``E-MATRIX-FAILURE-MODE-MISSING`` / ``-UNRESOLVED`` / ``E-MATRIX-ENFORCEMENT-MISSING``);
* a named served model without its **weights** and **code** licences cited separately
  (``E-MATRIX-LICENCE-UNCITED``, ADR-0012).

**Two axes per cell, because they fail differently.** ``resolution`` is whether the combination
can be *installed* — offline, machine-checkable anywhere, and already gated by ``latence env
verify``. ``runtime`` is whether it *runs on Blackwell*, which needs the hardware. Collapsing them
into one status is how a green "resolves fine" gets read as "runs fine on sm_120".

Governing: ADR-0016 (heavy deps live in per-Provider packages), ADR-0036 (device profiles — never
fabricate a device fact), ADR-0030 (the endpoint seam, which is what resolves the
``transformers`` split), ADR-0012 (weights and code licences verified separately).

Like :mod:`latence_core.envspec` this module is offline and import-free: it reads YAML, the
committed locks and the filesystem. It never installs, imports a Provider or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from latence_core.envspec import (
    EnvironmentSpec,
    Finding,
    distribution_name,
    load_environment,
    parse_lock,
)
from latence_core.errors import ConfigError

VERIFIED = "verified"
"""A cell backed by a real measurement, with the evidence recorded."""

UNVERIFIED = "UNVERIFIED"
"""A cell nobody has measured. Shouted in caps on purpose — it is never a soft "probably fine"."""

_STATUSES = (VERIFIED, UNVERIFIED)

_KIND_DISTRIBUTION = "distribution"
_KIND_EXCLUDED = "excluded_distribution"
_KIND_LEARNED_PROVIDER = "learned_provider"
_KIND_SERVED_MODEL = "served_model"
_KINDS = (_KIND_DISTRIBUTION, _KIND_EXCLUDED, _KIND_LEARNED_PROVIDER, _KIND_SERVED_MODEL)

REQUIRED_FAILURE_MODES: tuple[str, ...] = (
    "FM-FLASHINFER-SM120",
    "FM-NVRTC-JIT",
    "FM-WHEEL-ABI",
    "FM-MULTIMODAL-OOM",
    "FM-TRANSFORMERS-SPLIT",
)
"""The failure modes the live 2026-07-17 Blackwell/CUDA-13 run produced (issue #199).

Hard-coded rather than derived, and that is the point: these five cost a real operator real hours,
and the ticket's requirement is that the *next* operator hits documentation instead of breakage.
A matrix that quietly drops one fails :func:`verify_matrix`.
"""

_GENERATED_HEADER = (
    "<!-- GENERATED FILE — do not edit by hand.\n"
    "     Source of truth: {source}\n"
    "     Regenerate:      latence env matrix {source} --render\n"
    "     Verify:          latence env matrix {source}                       -->\n"
)


# --------------------------------------------------------------------------------------------
# The data model.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """What a ``verified`` claim has to carry: when, on what, run how, and what came out."""

    date: str
    hardware: str
    command: str
    observed: str
    source: str = ""
    """Where the observation is recorded (an ADR section, a run log) — provenance, not proof."""

    def missing(self) -> tuple[str, ...]:
        """The required fields left blank. Empty means the claim is complete."""
        return tuple(
            key
            for key, value in (
                ("date", self.date),
                ("hardware", self.hardware),
                ("command", self.command),
                ("observed", self.observed),
            )
            if not value.strip()
        )


@dataclass(frozen=True)
class Axis:
    """One measurable axis of a cell: ``resolution`` (installs) or ``runtime`` (runs on sm_120)."""

    name: str
    status: str
    evidence: Evidence | None = None
    reproduce: str = ""
    """The exact command the next operator runs to turn ``UNVERIFIED`` into a measurement."""
    blocked_by: str = ""
    """Why it is not measured yet — honest, and the thing a maintainer schedules against."""

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED


@dataclass(frozen=True)
class Cell:
    """One (component, environment) pair of the matrix."""

    component: str
    kind: str
    environment: str
    resolution: Axis
    runtime: Axis
    version: str = ""
    """For a distribution: the version the environment's committed lock pins it to."""
    notes: str = ""
    weights_license: str = ""
    code_license: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.environment, self.component)


@dataclass(frozen=True)
class FailureMode:
    """A failure the live run produced, with the resolution and the file that enforces it.

    TWO axes again, and for the same reason a cell has two: ``observation`` is that the failure
    really happened (measured on the pod, on the day, at the versions of the day) and
    ``resolution_state`` is whether the fix has been confirmed *on the pinned environments this
    matrix describes*. They are routinely not the same, and collapsing them would let "we know
    about this" read as "this is fixed for you".
    """

    id: str
    title: str
    environment: str
    component: str
    symptom: str
    cause: str
    resolution: str
    enforced_by: tuple[str, ...]
    observation: Axis
    resolution_state: Axis


@dataclass(frozen=True)
class FullStackRun:
    """The one documented end-to-end verification: corpus in, Quality Report + Exports out."""

    status: str
    command: str
    script: str
    stack: str
    artifacts: tuple[str, ...]
    evidence: Evidence | None = None
    blocked_by: str = ""


@dataclass(frozen=True)
class CompatibilityMatrix:
    """``deploy/env/compatibility-matrix.yaml``, parsed."""

    name: str
    summary: str
    hardware: str
    verified_on: str
    doc_path: str
    """Repo-relative path of the generated document (byte-compared by :func:`verify_matrix`)."""
    source_path: Path
    cells: tuple[Cell, ...]
    failure_modes: tuple[FailureMode, ...]
    full_stack_run: FullStackRun

    @property
    def environments(self) -> tuple[str, ...]:
        seen: list[str] = []
        for cell in self.cells:
            if cell.environment not in seen:
                seen.append(cell.environment)
        return tuple(seen)


@dataclass(frozen=True)
class MatrixReport:
    """The result of :func:`verify_matrix` — findings only; empty means green."""

    name: str
    findings: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings

    def describe(self) -> str:
        if self.ok:
            return f"compatibility matrix '{self.name}': OK (no findings)"
        lines = [f"compatibility matrix '{self.name}': {len(self.findings)} finding(s)"]
        lines.extend(f"  - {f}" for f in self.findings)
        return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------------------------


def _mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"Compatibility matrix {where}: expected a mapping.")
    return raw


def _text(raw: dict[str, Any], key: str, where: str, *, required: bool = True) -> str:
    value = raw.get(key, "")
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ConfigError(f"Compatibility matrix {where}: '{key}' must be a string.")
    value = value.strip()
    if required and not value:
        raise ConfigError(f"Compatibility matrix {where}: missing required key '{key}'.")
    return value


def _texts(raw: dict[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = raw.get(key, []) or []
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise ConfigError(f"Compatibility matrix {where}: '{key}' must be a list of strings.")
    return tuple(v.strip() for v in value)


def _status(raw: dict[str, Any], where: str) -> str:
    value = _text(raw, "status", where)
    if value not in _STATUSES:
        raise ConfigError(
            f"Compatibility matrix {where}: status must be exactly one of {_STATUSES!r} "
            f"(got {value!r}). There is deliberately no third, softer state — a cell is a "
            "measurement or it is UNVERIFIED."
        )
    return value


def _evidence(raw: dict[str, Any], where: str) -> Evidence | None:
    node = raw.get("evidence")
    if node is None:
        return None
    data = _mapping(node, f"{where}.evidence")
    return Evidence(
        date=_text(data, "date", f"{where}.evidence", required=False),
        hardware=_text(data, "hardware", f"{where}.evidence", required=False),
        command=_text(data, "command", f"{where}.evidence", required=False),
        observed=_text(data, "observed", f"{where}.evidence", required=False),
        source=_text(data, "source", f"{where}.evidence", required=False),
    )


def _axis(raw: dict[str, Any], name: str, where: str) -> Axis:
    node = _mapping(raw.get(name), f"{where}.{name}")
    return Axis(
        name=name,
        status=_status(node, f"{where}.{name}"),
        evidence=_evidence(node, f"{where}.{name}"),
        reproduce=_text(node, "reproduce", f"{where}.{name}", required=False),
        blocked_by=_text(node, "blocked_by", f"{where}.{name}", required=False),
    )


def load_matrix(path: Path) -> CompatibilityMatrix:
    """Parse ``compatibility-matrix.yaml``.

    Raises :class:`~latence_core.errors.ConfigError` on a malformed matrix — a matrix the tooling
    cannot read is a broken matrix, never a skipped check.
    """
    path = path.resolve()
    try:
        raw_any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Compatibility matrix not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Compatibility matrix is not valid YAML: {path}: {exc}") from exc
    raw = _mapping(raw_any, str(path))

    cells: list[Cell] = []
    raw_cells = raw.get("cells", []) or []
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ConfigError(f"Compatibility matrix {path}: 'cells' must be a non-empty list.")
    for entry in raw_cells:
        data = _mapping(entry, f"{path}: cell")
        component = _text(data, "component", f"{path}: cell")
        where = f"{path}: cell {component!r}"
        kind = _text(data, "kind", where)
        if kind not in _KINDS:
            raise ConfigError(f"Compatibility matrix {where}: 'kind' must be one of {_KINDS!r}.")
        cells.append(
            Cell(
                component=component,
                kind=kind,
                environment=_text(data, "environment", where),
                resolution=_axis(data, "resolution", where),
                runtime=_axis(data, "runtime", where),
                version=_text(data, "version", where, required=False),
                notes=_text(data, "notes", where, required=False),
                weights_license=_text(data, "weights_license", where, required=False),
                code_license=_text(data, "code_license", where, required=False),
            )
        )

    failure_modes: list[FailureMode] = []
    for entry in raw.get("failure_modes", []) or []:
        data = _mapping(entry, f"{path}: failure mode")
        fm_id = _text(data, "id", f"{path}: failure mode")
        where = f"{path}: failure mode {fm_id}"
        failure_modes.append(
            FailureMode(
                id=fm_id,
                title=_text(data, "title", where),
                environment=_text(data, "environment", where),
                component=_text(data, "component", where),
                symptom=_text(data, "symptom", where),
                cause=_text(data, "cause", where),
                resolution=_text(data, "resolution", where, required=False),
                enforced_by=_texts(data, "enforced_by", where),
                observation=_axis(data, "observation", where),
                resolution_state=_axis(data, "resolution_state", where),
            )
        )

    run_where = f"{path}: full_stack_run"
    run_raw = _mapping(raw.get("full_stack_run"), run_where)
    run = FullStackRun(
        status=_status(run_raw, run_where),
        command=_text(run_raw, "command", run_where),
        script=_text(run_raw, "script", run_where),
        stack=_text(run_raw, "stack", run_where, required=False),
        artifacts=_texts(run_raw, "artifacts", run_where),
        evidence=_evidence(run_raw, run_where),
        blocked_by=_text(run_raw, "blocked_by", run_where, required=False),
    )

    return CompatibilityMatrix(
        name=_text(raw, "name", str(path)),
        summary=_text(raw, "summary", str(path)),
        hardware=_text(raw, "hardware", str(path)),
        verified_on=_text(raw, "verified_on", str(path)),
        doc_path=_text(raw, "doc_path", str(path)),
        source_path=path,
        cells=tuple(cells),
        failure_modes=tuple(failure_modes),
        full_stack_run=run,
    )


# --------------------------------------------------------------------------------------------
# Coverage — derived from the committed environments, never from the matrix's own opinion.
# --------------------------------------------------------------------------------------------


def required_components(spec: EnvironmentSpec) -> dict[str, str]:
    """The components an environment obliges the matrix to have a cell for: ``{name: kind}``.

    Derived from the committed ``environment.yaml``: every learned Provider it promises to place on
    the GPU, and every distribution it names in an enforced boundary, an exclusion or its own
    ``extra_requirements`` — which is exactly the serving surface (``vllm``,
    ``flashinfer-python``, ``torch``, ``transformers``, the ABI-broken wheels). Because the set is
    derived, *adding* a boundary or a learned Provider to an environment turns the matrix red until
    somebody maps the new combination.
    """
    components: dict[str, str] = {}
    for constraint in spec.constraints:
        components[distribution_name(constraint.requirement)] = _KIND_DISTRIBUTION
    for exclusion in spec.excluded_distributions:
        components[distribution_name(exclusion.package)] = _KIND_EXCLUDED
    for requirement in spec.extra_requirements:
        components[distribution_name(requirement)] = _KIND_DISTRIBUTION
    for provider in sorted(spec.learned_providers):
        components[provider] = _KIND_LEARNED_PROVIDER
    return components


def _environment_specs(repo_root: Path) -> dict[str, EnvironmentSpec]:
    root = repo_root / "deploy" / "env"
    return {
        spec.name: spec
        for spec in (load_environment(p) for p in sorted(root.glob("*/environment.yaml")))
    }


# --------------------------------------------------------------------------------------------
# Verification.
# --------------------------------------------------------------------------------------------


def _check_axis(subject: str, axis: Axis, repo_root: Path) -> list[Finding]:
    """The one rule the whole matrix rests on, applied to one axis of one subject.

    ``verified`` obliges a complete measurement; ``UNVERIFIED`` obliges a runnable command and a
    reason. There is no third path, which is what makes "no inferred cells" enforceable rather
    than aspirational.
    """
    label = f"{subject} [{axis.name}]"
    findings: list[Finding] = []
    if axis.verified:
        if axis.evidence is None:
            findings.append(
                Finding(
                    "E-MATRIX-EVIDENCE-MISSING",
                    f"{label} claims '{VERIFIED}' with no evidence block. A verified cell records "
                    "date, hardware, the exact command and the observed output — otherwise it is "
                    f"an inference, and this matrix only holds measurements or {UNVERIFIED}.",
                )
            )
        elif missing := axis.evidence.missing():
            findings.append(
                Finding(
                    "E-MATRIX-EVIDENCE-MISSING",
                    f"{label} claims '{VERIFIED}' but its evidence is missing "
                    f"{', '.join(missing)}.",
                )
            )
        return findings

    if not axis.reproduce:
        findings.append(
            Finding(
                "E-MATRIX-REPRODUCE-MISSING",
                f"{label} is {UNVERIFIED} without a 'reproduce' command. An unmeasured cell must "
                "still tell the next operator exactly what to run.",
            )
        )
    if not axis.blocked_by:
        findings.append(
            Finding(
                "E-MATRIX-REPRODUCE-MISSING",
                f"{label} is {UNVERIFIED} without 'blocked_by'. Why it is unmeasured is what a "
                "maintainer schedules against.",
            )
        )
    findings.extend(_check_referenced_paths(axis.reproduce, repo_root, label))
    return findings


def _check_referenced_paths(command: str, repo_root: Path, label: str) -> list[Finding]:
    """Every repo-relative path a command names must exist — a dead runbook is not a runbook."""
    findings: list[Finding] = []
    for token in command.replace("'", " ").replace('"', " ").split():
        if not token.startswith(("deploy/", "scripts/", "stacks/", "packages/", "docs/")):
            continue
        if not (repo_root / token).exists():
            findings.append(
                Finding(
                    "E-MATRIX-REPRODUCE-UNRUNNABLE",
                    f"{label} points at '{token}', which does not exist in the repo.",
                )
            )
    return findings


def _check_coverage(
    matrix: CompatibilityMatrix, specs: dict[str, EnvironmentSpec]
) -> list[Finding]:
    findings: list[Finding] = []
    by_key = {cell.key: cell for cell in matrix.cells}

    seen: set[tuple[str, str]] = set()
    for cell in matrix.cells:
        if cell.key in seen:
            findings.append(
                Finding(
                    "E-MATRIX-CELL-DUPLICATE",
                    f"cell {cell.environment}/{cell.component} is declared twice; two rows for one "
                    "combination is how two different answers ship.",
                )
            )
        seen.add(cell.key)
        if cell.environment not in specs:
            findings.append(
                Finding(
                    "E-MATRIX-CELL-UNKNOWN",
                    f"cell {cell.environment}/{cell.component} names environment "
                    f"'{cell.environment}', which is not a committed deploy/env/<name>/.",
                )
            )

    for name, spec in specs.items():
        if not any(cell.environment == name for cell in matrix.cells):
            findings.append(
                Finding(
                    "E-MATRIX-ENVIRONMENT-UNCOVERED",
                    f"environment '{name}' is committed under deploy/env/ but has no matrix cell "
                    "at all. Every shipped environment is somebody's install target.",
                )
            )
        for component, kind in required_components(spec).items():
            cell = by_key.get((name, component))
            if cell is None:
                findings.append(
                    Finding(
                        "E-MATRIX-CELL-UNCOVERED",
                        f"environment '{name}' declares {kind} '{component}' but the matrix has no "
                        "cell for it. Coverage is derived from the committed environment specs, so "
                        "a new boundary or learned Provider is mapped or the gate stays red.",
                    )
                )
            elif cell.kind != kind:
                findings.append(
                    Finding(
                        "E-MATRIX-CELL-UNKNOWN",
                        f"cell {name}/{component} is filed as '{cell.kind}' but the environment "
                        f"declares it as '{kind}'.",
                    )
                )

    for cell in matrix.cells:
        spec = specs.get(cell.environment) if cell.environment in specs else None
        if spec is None or cell.kind == _KIND_SERVED_MODEL:
            continue
        if cell.component not in required_components(spec):
            findings.append(
                Finding(
                    "E-MATRIX-CELL-UNKNOWN",
                    f"cell {cell.environment}/{cell.component} describes a component that "
                    "environment does not declare. A matrix that invents rows stops describing the "
                    "thing it claims to describe.",
                )
            )
    return findings


def _check_versions(
    matrix: CompatibilityMatrix, specs: dict[str, EnvironmentSpec]
) -> list[Finding]:
    """A distribution cell's version must be the version that environment's lock actually pins."""
    findings: list[Finding] = []
    locks: dict[str, dict[str, str]] = {}
    for cell in matrix.cells:
        if cell.kind not in (_KIND_DISTRIBUTION, _KIND_EXCLUDED):
            continue
        spec = specs.get(cell.environment)
        if spec is None:
            continue
        if cell.environment not in locks:
            locks[cell.environment] = parse_lock(spec.lock_path.read_text(encoding="utf-8"))
        if cell.kind == _KIND_EXCLUDED:
            # An excluded distribution's whole claim is that it is NOT installable here, so the
            # honest cell has no version — and the lock had better agree, which is the same fact
            # `latence env verify` reports as E-EXCLUSION-VIOLATED.
            if cell.version:
                findings.append(
                    Finding(
                        "E-MATRIX-VERSION-DRIFT",
                        f"cell {cell.environment}/{cell.component} is an excluded distribution but "
                        f"records version '{cell.version}'. An exclusion has no version: it is the "
                        "claim that no version can be installed.",
                    )
                )
            if cell.component in locks[cell.environment]:
                findings.append(
                    Finding(
                        "E-MATRIX-VERSION-DRIFT",
                        f"cell {cell.environment}/{cell.component} says the distribution is "
                        f"excluded, but {spec.lock_path.name} pins it.",
                    )
                )
            continue
        if not cell.version:
            findings.append(
                Finding(
                    "E-MATRIX-VERSION-MISSING",
                    f"cell {cell.environment}/{cell.component} records no version; a distribution "
                    "row with no version describes no combination.",
                )
            )
            continue
        pinned = locks[cell.environment].get(cell.component)
        if pinned is None:
            findings.append(
                Finding(
                    "E-MATRIX-VERSION-DRIFT",
                    f"cell {cell.environment}/{cell.component} records version "
                    f"'{cell.version}' but that distribution is not in "
                    f"{spec.lock_path.name}.",
                )
            )
        elif pinned != f"=={cell.version}":
            findings.append(
                Finding(
                    "E-MATRIX-VERSION-DRIFT",
                    f"cell {cell.environment}/{cell.component} records version "
                    f"'{cell.version}' but the committed lock pins '{pinned.lstrip('=')}'. The "
                    "matrix describes the PINNED environments or it describes nothing.",
                )
            )
    return findings


def _check_licenses(matrix: CompatibilityMatrix) -> list[Finding]:
    findings: list[Finding] = []
    for cell in matrix.cells:
        if cell.kind != _KIND_SERVED_MODEL:
            continue
        missing = [
            field_name
            for field_name, value in (
                ("weights_license", cell.weights_license),
                ("code_license", cell.code_license),
            )
            if not value
        ]
        if missing:
            findings.append(
                Finding(
                    "E-MATRIX-LICENCE-UNCITED",
                    f"served model '{cell.component}' is missing {', '.join(missing)}. ADR-0012: "
                    "weights and code licences are verified SEPARATELY before a model is named.",
                )
            )
    return findings


def _check_failure_modes(matrix: CompatibilityMatrix, repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    recorded = {fm.id: fm for fm in matrix.failure_modes}
    for required in REQUIRED_FAILURE_MODES:
        if required not in recorded:
            findings.append(
                Finding(
                    "E-MATRIX-FAILURE-MODE-MISSING",
                    f"failure mode '{required}' from the live Blackwell/CUDA-13 run is not in the "
                    "matrix. Each of these cost a real operator real hours; the next one must hit "
                    "documentation, not breakage.",
                )
            )
    for fm in matrix.failure_modes:
        label = f"failure mode {fm.id}"
        if not fm.resolution:
            findings.append(
                Finding(
                    "E-MATRIX-FAILURE-MODE-UNRESOLVED",
                    f"{label} records a symptom with no resolution.",
                )
            )
        if not fm.enforced_by:
            findings.append(
                Finding(
                    "E-MATRIX-FAILURE-MODE-UNRESOLVED",
                    f"{label} names no file that enforces its resolution. A fix that lives only in "
                    "a table is folklore with better typography.",
                )
            )
        for path in fm.enforced_by:
            if not (repo_root / path).exists():
                findings.append(
                    Finding(
                        "E-MATRIX-ENFORCEMENT-MISSING",
                        f"{label} is enforced by '{path}', which does not exist in the repo.",
                    )
                )
        findings.extend(_check_axis(label, fm.observation, repo_root))
        findings.extend(_check_axis(label, fm.resolution_state, repo_root))
    return findings


def _check_full_stack_run(
    matrix: CompatibilityMatrix, repo_root: Path, *, require_verified: bool
) -> list[Finding]:
    run = matrix.full_stack_run
    findings: list[Finding] = []
    if not (repo_root / run.script).exists():
        findings.append(
            Finding(
                "E-MATRIX-RUN-UNRECORDED",
                f"the full-stack verification names script '{run.script}', which does not exist. "
                "The one documented end-to-end run must be one runnable command.",
            )
        )
    if not run.artifacts:
        findings.append(
            Finding(
                "E-MATRIX-RUN-UNRECORDED",
                "the full-stack verification names no artifacts. 'Quality Report + Exports out' is "
                "the acceptance criterion, so the artifacts are part of the record.",
            )
        )
    if run.status == VERIFIED:
        if run.evidence is None or run.evidence.missing():
            missing = ", ".join(run.evidence.missing()) if run.evidence else "the whole block"
            findings.append(
                Finding(
                    "E-MATRIX-EVIDENCE-MISSING",
                    f"the full-stack verification claims '{VERIFIED}' but its evidence is missing "
                    f"{missing}.",
                )
            )
    else:
        if not run.blocked_by:
            findings.append(
                Finding(
                    "E-MATRIX-RUN-UNRECORDED",
                    f"the full-stack verification is {UNVERIFIED} without 'blocked_by'.",
                )
            )
        if require_verified:
            findings.append(
                Finding(
                    "E-MATRIX-RUN-UNVERIFIED",
                    f"the full-stack verification is still {UNVERIFIED}: run `{run.command}` on "
                    "the Blackwell/CUDA-13 host and record the evidence.",
                )
            )
    findings.extend(_check_referenced_paths(run.command, repo_root, "full-stack verification"))
    return findings


def _check_document(matrix: CompatibilityMatrix, repo_root: Path) -> list[Finding]:
    doc = repo_root / matrix.doc_path
    expected = render_matrix(matrix)
    if not doc.exists():
        return [
            Finding(
                "E-MATRIX-DOC-STALE",
                f"the generated document '{matrix.doc_path}' is missing. Regenerate it with "
                f"`latence env matrix {_rel(matrix.source_path, repo_root)} --render`.",
            )
        ]
    if doc.read_text(encoding="utf-8") != expected:
        return [
            Finding(
                "E-MATRIX-DOC-STALE",
                f"the generated document '{matrix.doc_path}' differs from the render of "
                f"{_rel(matrix.source_path, repo_root)}. It is GENERATED — edit the YAML and "
                "re-render, never the Markdown.",
            )
        ]
    return []


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:  # pragma: no cover - only for a matrix outside the repo
        return str(path)


def verify_matrix(
    matrix: CompatibilityMatrix, repo_root: Path, *, require_verified: bool = False
) -> MatrixReport:
    """Check the matrix against the committed environments, the locks and the filesystem.

    Offline and import-free. Returns every finding (never short-circuits) so one run shows the
    whole picture. ``require_verified`` additionally fails while anything is still ``UNVERIFIED``
    — the switch a maintainer flips *on the Blackwell host* to prove the map is fully measured;
    it is off in the ordinary merge gate, where an honest ``UNVERIFIED`` is a legitimate state.
    """
    specs = _environment_specs(repo_root)
    findings: list[Finding] = []
    findings.extend(_check_coverage(matrix, specs))
    findings.extend(_check_versions(matrix, specs))
    findings.extend(_check_licenses(matrix))
    for cell in matrix.cells:
        subject = f"cell {cell.environment}/{cell.component}"
        findings.extend(_check_axis(subject, cell.resolution, repo_root))
        findings.extend(_check_axis(subject, cell.runtime, repo_root))
        if require_verified and not cell.runtime.verified:
            findings.append(
                Finding(
                    "E-MATRIX-CELL-UNVERIFIED",
                    f"cell {cell.environment}/{cell.component} [runtime] is still {UNVERIFIED}: "
                    f"run `{cell.runtime.reproduce}`.",
                )
            )
    findings.extend(_check_failure_modes(matrix, repo_root))
    findings.extend(_check_full_stack_run(matrix, repo_root, require_verified=require_verified))
    findings.extend(_check_document(matrix, repo_root))
    return MatrixReport(name=matrix.name, findings=tuple(findings))


# --------------------------------------------------------------------------------------------
# Rendering — the published document is generated, so it cannot drift from the checked data.
# --------------------------------------------------------------------------------------------


def _cell_status(axis: Axis) -> str:
    if axis.verified:
        return "**verified**"
    return f"`{UNVERIFIED}`"


def _render_axis(label: str, axis: Axis) -> list[str]:
    """One axis as a document bullet: the status, then either the measurement or the way to make
    one. The two branches are exactly the two the verifier enforces."""
    lines = [f"* **{label}:** {_cell_status(axis)}"]
    if axis.evidence is not None:
        ev = axis.evidence
        lines.append(
            f"    * {_md(ev.date)}, {_md(ev.hardware)} — `{_md(ev.command)}` observed: "
            f"{_md(ev.observed)}" + (f" ({_md(ev.source)})" if ev.source else "")
        )
    if not axis.verified:
        lines.append(f"    * Reproduce: `{_md(axis.reproduce)}`")
        if axis.blocked_by:
            lines.append(f"    * Blocked by: {_md(axis.blocked_by)}")
    return lines


def _md(value: str) -> str:
    """One table cell: no newlines, no pipe that would split the column."""
    return " ".join(value.split()).replace("|", "\\|")


def render_matrix(matrix: CompatibilityMatrix) -> str:
    """Render the published Markdown document from the matrix. Deterministic."""
    source = _md(str(matrix.source_path.name))
    source_rel = f"deploy/env/{source}"
    lines: list[str] = [_GENERATED_HEADER.format(source=source_rel), ""]
    lines.append(f"# Blackwell / CUDA-13 compatibility matrix ({matrix.name})")
    lines.append("")
    lines.append(_md(matrix.summary))
    lines.append("")
    lines.append(f"* **Reference hardware:** {_md(matrix.hardware)}")
    lines.append(f"* **Verification date:** {_md(matrix.verified_on)}")
    lines.append(f"* **Source of truth:** `{source_rel}` (this page is generated from it)")
    lines.append(
        "* **Check it:** `latence env matrix "
        f"{source_rel}` — every cell is a measurement or `{UNVERIFIED}`; "
        "the verifier refuses a `verified` cell with no evidence and an `UNVERIFIED` cell with no "
        "reproduce command."
    )
    lines.append("")
    lines.append(
        f"Two axes per cell. **resolution** — does the combination install from the pinned "
        "environment (offline, machine-checked on every commit). **runtime** — does it *run* on "
        "Blackwell `sm_120`, which needs the hardware. A green resolution is not a runtime claim."
    )
    lines.append("")

    for env in matrix.environments:
        lines.append(f"## `deploy/env/{env}/`")
        lines.append("")
        lines.append("| Component | Kind | Version | Resolution | Runtime | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for cell in matrix.cells:
            if cell.environment != env:
                continue
            note = cell.notes
            if cell.kind == _KIND_SERVED_MODEL:
                note = (
                    f"{note} Weights: {cell.weights_license}. Code: {cell.code_license}."
                ).strip()
            if not cell.runtime.verified and cell.runtime.reproduce:
                note = f"{note} Measure with `{cell.runtime.reproduce}`.".strip()
            lines.append(
                f"| `{_md(cell.component)}` | {_md(cell.kind)} | "
                f"{('`' + _md(cell.version) + '`') if cell.version else '—'} | "
                f"{_cell_status(cell.resolution)} | {_cell_status(cell.runtime)} | {_md(note)} |"
            )
        lines.append("")

    lines.append("## Failure modes from the live run, and their resolutions")
    lines.append("")
    lines.append(
        "Every row below was produced by a real run, not predicted. The resolution column names "
        "the file that ENFORCES the fix, so the next operator inherits it without knowing it "
        "exists."
    )
    lines.append("")
    for fm in matrix.failure_modes:
        lines.append(f"### {fm.id} — {_md(fm.title)}")
        lines.append("")
        lines.append(f"* **Where:** `{_md(fm.environment)}` / `{_md(fm.component)}`")
        lines.append(f"* **Symptom:** {_md(fm.symptom)}")
        lines.append(f"* **Cause:** {_md(fm.cause)}")
        lines.append(f"* **Resolution:** {_md(fm.resolution)}")
        enforced = ", ".join(f"`{_md(p)}`" for p in fm.enforced_by)
        lines.append(f"* **Enforced by:** {enforced}")
        lines.extend(_render_axis("Observed", fm.observation))
        lines.extend(_render_axis("Resolution confirmed", fm.resolution_state))
        lines.append("")

    run = matrix.full_stack_run
    lines.append("## Full-stack end-to-end verification")
    lines.append("")
    lines.append(
        f"* **Status:** {_cell_status(Axis(name='status', status=run.status))}"
    )
    lines.append(f"* **Command:** `{_md(run.command)}`")
    lines.append(f"* **Script:** `{_md(run.script)}`")
    if run.stack:
        lines.append(f"* **Stack:** `{_md(run.stack)}`")
    lines.append("* **Artifacts the run must produce:**")
    for artifact in run.artifacts:
        lines.append(f"    * `{_md(artifact)}`")
    if run.evidence is not None:
        lines.append(
            f"* **Evidence:** {_md(run.evidence.date)}, {_md(run.evidence.hardware)} — "
            f"`{_md(run.evidence.command)}` observed: {_md(run.evidence.observed)}"
        )
    if run.status != VERIFIED:
        lines.append(f"* **Blocked by:** {_md(run.blocked_by)}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"
