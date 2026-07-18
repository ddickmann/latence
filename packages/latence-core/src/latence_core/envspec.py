"""Reproducible-environment specification + verifier (issue #195, ADR-0016/0007/0041).

**The problem.** Standing up a heavy Provider set is per-session trial-and-error: torch version
churn ABI-breaks prebuilt wheels, the GLiNER family and the OCR model disagree on the
``transformers`` major, PEP 668 blocks naive installs. The cure is not a wiki page — it is a
*specification the build can check*.

**The shape.** An *environment* is a directory under ``deploy/env/<name>/`` holding

* ``environment.yaml`` — the SINGLE SOURCE OF TRUTH parsed by :func:`load_environment`: which
  workspace Provider packages the environment stands up, which conflict boundaries are
  ENFORCED (with the reason each exists), which Providers the environment must expose, and
  which stack the smoke test drives;
* ``requirements.in`` + ``constraints.txt`` — **generated** from that spec by
  :func:`render_requirements` / :func:`render_constraints` (never hand-edited);
* ``requirements.lock`` — the solved, exactly-pinned resolution;
* ``install.sh`` + ``smoke.sh`` — the one documented install path and the one-command guard.

**The guard.** :func:`verify_environment` reads the spec, the workspace packages' real
``pyproject.toml`` metadata and the committed lock, and returns a list of :class:`Finding`.
It is deliberately OFFLINE and import-free — it never installs, never imports a Provider and
never touches the network — so it runs in the ordinary test suite and turns **red on dependency
drift**:

* a Provider package gains/loses/retightens a dependency  → the rendered ``requirements.in`` no
  longer matches the committed one (``E-REQUIREMENTS-STALE``);
* a Provider package raises a floor or drops a cap the lock no longer satisfies
  (``E-LOCK-UNSATISFIED``);
* an enforced conflict boundary stops holding (``E-CONSTRAINT-VIOLATED``);
* a Provider is renamed out from under the recipe (``E-PROVIDER-UNREGISTERED``) or the smoke
  stack drifts away from the declared Provider set (``E-STACK-PROVIDER-UNDECLARED``).

Every finding is a *behavioural* fact about the committed environment, never a flag recording
intent.

The module is reusable by construction: nothing here knows the word "cpu-text". The served-OCR
(#196) and GPU learned-Provider (#198) environments are additional ``deploy/env/<name>/``
directories with the same six files and are verified by the same code path.

``packaging`` is the only dependency beyond the core spine and it is imported LAZILY: it ships in
the optional ``latence-core[envspec]`` extra (ADR-0016 — the core stays near-zero-dep), exactly
like the ``observability`` extra. Absent, :func:`verify_environment` raises a typed, actionable
error rather than silently degrading.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from latence_core.errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from packaging.requirements import Requirement
    from packaging.version import Version

_GENERATED_HEADER = (
    "# GENERATED FILE — do not edit by hand.\n"
    "# Source of truth: {spec} + the workspace packages' pyproject.toml.\n"
    "# Regenerate:      latence env render {spec}\n"
    "# Verify:          latence env verify {spec}\n"
)

# Workspace-internal distributions are installed from source with ``--no-deps`` by the install
# recipe, so they never appear in the third-party requirement set.
_WORKSPACE_PREFIX = "latence-"


def _require_packaging() -> Any:
    """Import ``packaging`` lazily, with an actionable error when the extra is not installed."""
    try:
        import packaging.requirements as _req
        import packaging.utils as _utils
        import packaging.version as _ver
    except ModuleNotFoundError as exc:  # pragma: no cover - only without the optional extra
        msg = (
            "The environment verifier needs PEP 440/508 parsing. Install the optional extra: "
            "`pip install 'latence-core[envspec]'` (or `uv sync` in the workspace, where the dev "
            "group pins it)."
        )
        raise ConfigError(msg) from exc
    return _req, _utils, _ver


@dataclass(frozen=True)
class Constraint:
    """One ENFORCED conflict boundary: a PEP 508 requirement plus why it exists.

    The ``requirement`` is emitted verbatim into ``constraints.txt`` (so the installer applies it)
    AND asserted against the committed lock (so it cannot rot into folklore). The ``reason`` is
    carried into the generated file as a comment — a boundary nobody can explain is a boundary
    nobody will maintain.
    """

    requirement: str
    reason: str


@dataclass(frozen=True)
class ExcludedPackage:
    """One workspace package that MUST NOT enter this environment, plus why (issue #198).

    The mechanism that turns an incompatible-by-construction Provider into a **resolution-time**
    refusal instead of a runtime surprise. The motivating case is the ``transformers`` major split:
    the GLiNER family is written against ``transformers`` 4.x, while the in-process OCR VLM needs
    5.x. That conflict is not expressible as a version range on either side — the OCR package's own
    metadata declares an *uncapped* ``transformers>=4.45``, so a resolver happily installs it into a
    ``transformers<5`` environment and the incompatibility only detonates at model load.

    Naming the package here closes that hole: :func:`verify_environment` refuses the environment
    the moment such a package appears in ``workspace_packages`` **or is reachable through another
    installed package's workspace dependencies** (``latence-parser-lighton-vllm`` depends on
    ``latence-parser-lighton``, so pulling the served wrapper in drags the in-process VLM with it).
    Gate 1 of ``smoke.sh`` runs before ``install.sh``, so the unsupported combination fails at
    dependency resolution, exactly as the ADR-0030 endpoint boundary requires.
    """

    package: str
    """Repo-relative path of the workspace package (e.g. ``packages/latence-parser-lighton``)."""
    reason: str


@dataclass(frozen=True)
class EnvironmentSpec:
    """A reproducible environment, as declared by ``deploy/env/<name>/environment.yaml``."""

    name: str
    summary: str
    python: str
    platform: str
    directory: Path
    """Absolute path of the environment directory (where the generated files live)."""
    spec_path: Path
    workspace_packages: tuple[str, ...]
    """Repo-relative paths of the workspace packages this environment installs, in install order."""
    extra_requirements: tuple[str, ...]
    """Third-party requirements no workspace package declares (transitive pins the recipe owns)."""
    constraints: tuple[Constraint, ...]
    excluded_packages: tuple[ExcludedPackage, ...]
    """Workspace packages this environment REFUSES (issue #198) — see :class:`ExcludedPackage`."""
    providers: tuple[str, ...]
    """Provider names the environment must expose (the smoke test's strict assertion)."""
    gpu_only_providers: frozenset[str]
    """Providers whose *runtime* needs a GPU: installed + registered here, exercised by #198."""
    learned_providers: frozenset[str]
    """Providers that load a MODEL, and must therefore prove real GPU placement (issue #198).

    The probe set of :func:`latence_core.placement.probe_placement`. A Provider named here is
    driven through one real inference and must allocate CUDA memory doing it; a Provider that owns
    no model (source, chunker, export) is not, because demanding GPU work of it would be noise.
    Declared rather than inferred, so *adding* a learned Provider to a GPU environment without
    adding it here is a visible omission in one file rather than a silent hole in the gate.
    """
    extra_index_urls: tuple[str, ...]
    index_strategy: str
    requirements_file: str = "requirements.in"
    constraints_file: str = "constraints.txt"
    lock_file: str = "requirements.lock"
    stack_file: str = "stack.yaml"

    @property
    def requirements_path(self) -> Path:
        return self.directory / self.requirements_file

    @property
    def constraints_path(self) -> Path:
        return self.directory / self.constraints_file

    @property
    def lock_path(self) -> Path:
        return self.directory / self.lock_file

    @property
    def stack_path(self) -> Path:
        return self.directory / self.stack_file


@dataclass(frozen=True)
class Finding:
    """One verification failure: a stable machine code plus a human-actionable message."""

    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


@dataclass(frozen=True)
class EnvironmentReport:
    """The result of :func:`verify_environment` — findings only; empty means green."""

    name: str
    findings: tuple[Finding, ...] = field(default=())

    @property
    def ok(self) -> bool:
        return not self.findings

    def describe(self) -> str:
        if self.ok:
            return f"environment '{self.name}': OK (no findings)"
        lines = [f"environment '{self.name}': {len(self.findings)} finding(s)"]
        lines.extend(f"  - {f}" for f in self.findings)
        return "\n".join(lines)


def load_environment(spec_path: Path) -> EnvironmentSpec:
    """Parse ``environment.yaml`` into an :class:`EnvironmentSpec`.

    Raises :class:`~latence_core.errors.ConfigError` on a malformed spec — an environment the
    tooling cannot read is a broken environment, never a skipped check.
    """
    spec_path = spec_path.resolve()
    try:
        raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Environment spec not found: {spec_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Environment spec is not valid YAML: {spec_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Environment spec must be a YAML mapping: {spec_path}")

    def _str(key: str, *, required: bool = True, default: str = "") -> str:
        value = raw.get(key, default)
        if required and not value:
            raise ConfigError(f"Environment spec {spec_path}: missing required key '{key}'.")
        if not isinstance(value, str):
            raise ConfigError(f"Environment spec {spec_path}: '{key}' must be a string.")
        return value

    def _strs(key: str) -> tuple[str, ...]:
        value = raw.get(key, []) or []
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ConfigError(f"Environment spec {spec_path}: '{key}' must be a list of strings.")
        return tuple(value)

    constraints: list[Constraint] = []
    raw_constraints = raw.get("constraints", []) or []
    if not isinstance(raw_constraints, list):
        raise ConfigError(f"Environment spec {spec_path}: 'constraints' must be a list.")
    for entry in raw_constraints:
        if not isinstance(entry, dict) or "requirement" not in entry or "reason" not in entry:
            raise ConfigError(
                f"Environment spec {spec_path}: every constraint needs 'requirement' + 'reason' "
                "(an enforced boundary nobody can explain is folklore)."
            )
        constraints.append(
            Constraint(requirement=str(entry["requirement"]), reason=str(entry["reason"]))
        )

    excluded: list[ExcludedPackage] = []
    raw_excluded = raw.get("excluded_packages", []) or []
    if not isinstance(raw_excluded, list):
        raise ConfigError(f"Environment spec {spec_path}: 'excluded_packages' must be a list.")
    for entry in raw_excluded:
        if not isinstance(entry, dict) or "package" not in entry or "reason" not in entry:
            raise ConfigError(
                f"Environment spec {spec_path}: every excluded package needs 'package' + 'reason' "
                "(a refusal nobody can explain is folklore)."
            )
        excluded.append(
            ExcludedPackage(package=str(entry["package"]), reason=str(entry["reason"]))
        )

    workspace_packages = _strs("workspace_packages")
    providers = _strs("providers")
    if not workspace_packages:
        raise ConfigError(f"Environment spec {spec_path}: 'workspace_packages' must be non-empty.")
    if not providers:
        raise ConfigError(f"Environment spec {spec_path}: 'providers' must be non-empty.")

    return EnvironmentSpec(
        name=_str("name"),
        summary=_str("summary"),
        python=_str("python"),
        platform=_str("platform"),
        directory=spec_path.parent,
        spec_path=spec_path,
        workspace_packages=workspace_packages,
        extra_requirements=_strs("extra_requirements"),
        constraints=tuple(constraints),
        excluded_packages=tuple(excluded),
        providers=providers,
        gpu_only_providers=frozenset(_strs("gpu_only_providers")),
        learned_providers=frozenset(_strs("learned_providers")),
        extra_index_urls=_strs("extra_index_urls"),
        index_strategy=_str("index_strategy", required=False, default="unsafe-best-match"),
        requirements_file=_str("requirements_file", required=False, default="requirements.in"),
        constraints_file=_str("constraints_file", required=False, default="constraints.txt"),
        lock_file=_str("lock_file", required=False, default="requirements.lock"),
        stack_file=_str("stack_file", required=False, default="stack.yaml"),
    )


# --------------------------------------------------------------------------------------------
# Workspace package metadata (offline; read from pyproject.toml, never from an installed dist).
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _PackageMeta:
    rel_path: str
    name: str
    dependencies: tuple[str, ...]
    provider_entry_points: frozenset[str]

    @property
    def workspace_dependencies(self) -> frozenset[str]:
        """The ``latence-*`` distributions this package requires, by *distribution* name.

        Installing this package installs those too, so an environment's real package set is the
        transitive closure over them — which is what makes an excluded package reachable
        indirectly (``latence-parser-lighton-vllm`` → ``latence-parser-lighton``).
        """
        names: set[str] = set()
        for dep in self.dependencies:
            # Strip the version specifier / extras / marker without needing ``packaging``: the
            # distribution name is the leading run of name characters.
            head = dep.strip().split(";")[0].split("[")[0]
            for stop in ("<", ">", "=", "!", "~", " ", "("):
                head = head.split(stop)[0]
            head = head.strip().lower().replace("_", "-")
            if head.startswith(_WORKSPACE_PREFIX):
                names.add(head)
        return frozenset(names)


def _read_package(repo_root: Path, rel_path: str) -> _PackageMeta:
    pyproject = repo_root / rel_path / "pyproject.toml"
    if not pyproject.is_file():
        # A render over a workspace that is missing one of the environment's packages would be a
        # SILENTLY partial file, so it is refused. `verify_environment` reports the missing package
        # as E-PACKAGE-MISSING and skips the (meaningless) render comparison.
        raise ConfigError(f"workspace package '{rel_path}' has no pyproject.toml at {pyproject}")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    entry_points = (
        data.get("project", {}).get("entry-points", {}).get("latence.providers", {}) or {}
    )
    return _PackageMeta(
        rel_path=rel_path,
        name=str(project.get("name", "")),
        dependencies=tuple(str(d) for d in project.get("dependencies", []) or []),
        provider_entry_points=frozenset(str(k) for k in entry_points),
    )


def _third_party_requirements(
    packages: list[_PackageMeta],
) -> list[tuple[str, tuple[str, ...]]]:
    """The de-duplicated third-party requirement strings, each with its declaring packages.

    Ordered by canonical distribution name then requirement string so the render is deterministic
    (the byte-comparison guard depends on it). Workspace-internal ``latence-*`` requirements are
    dropped: the recipe installs those from source with ``--no-deps``.
    """
    _req, _utils, _ = _require_packaging()
    by_requirement: dict[str, list[str]] = {}
    for pkg in packages:
        for dep in pkg.dependencies:
            parsed: Requirement = _req.Requirement(dep)
            if _utils.canonicalize_name(parsed.name).startswith(_WORKSPACE_PREFIX):
                continue
            by_requirement.setdefault(dep, []).append(pkg.name)
    return sorted(
        ((dep, tuple(sorted(set(owners)))) for dep, owners in by_requirement.items()),
        key=lambda item: (_utils.canonicalize_name(_req.Requirement(item[0]).name), item[0]),
    )


# --------------------------------------------------------------------------------------------
# Rendering the generated files.
# --------------------------------------------------------------------------------------------


def render_requirements(spec: EnvironmentSpec, repo_root: Path) -> str:
    """Render ``requirements.in`` from the spec + the workspace packages' real dependencies.

    This is what makes dependency drift *visible*: the file is derived, so a dependency added to
    any Provider package in the set changes the render and the committed file goes stale.
    """
    packages = [_read_package(repo_root, p) for p in spec.workspace_packages]
    lines = [_GENERATED_HEADER.format(spec=_spec_label(spec, repo_root))]
    lines.append(
        "#\n"
        f"# Direct third-party requirements of the '{spec.name}' environment.\n"
        "# The workspace latence-* packages are installed from source with --no-deps by\n"
        "# install.sh, so their intra-workspace requirements never appear here — this file is\n"
        "# what actually pins their third-party surface.\n"
    )
    for requirement, owners in _third_party_requirements(packages):
        lines.append(requirement)
        lines.append(f"    # via {', '.join(owners)}")
    for requirement in spec.extra_requirements:
        lines.append(requirement)
        lines.append(f"    # via {spec.name} environment (extra_requirements)")
    return "\n".join(lines).rstrip("\n") + "\n"


def render_constraints(spec: EnvironmentSpec, repo_root: Path) -> str:
    """Render ``constraints.txt`` — the enforced conflict boundaries, each with its reason."""
    lines = [_GENERATED_HEADER.format(spec=_spec_label(spec, repo_root))]
    lines.append(
        "#\n"
        f"# ENFORCED conflict boundaries for the '{spec.name}' environment.\n"
        "# Passed to the installer as `-c constraints.txt`, so they bind the resolution — they\n"
        "# are not documentation. Every one is also asserted against requirements.lock by\n"
        "# `latence env verify`, so a boundary that stops holding fails the build.\n"
    )
    for constraint in spec.constraints:
        for reason_line in constraint.reason.strip().splitlines():
            lines.append(f"# {reason_line}".rstrip())
        lines.append(constraint.requirement)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _spec_label(spec: EnvironmentSpec, repo_root: Path) -> str:
    try:
        return spec.spec_path.relative_to(repo_root).as_posix()
    except ValueError:  # pragma: no cover - spec outside the repo
        return spec.spec_path.as_posix()


# --------------------------------------------------------------------------------------------
# The lock.
# --------------------------------------------------------------------------------------------


def parse_lock(text: str) -> dict[str, str]:
    """Parse a pinned ``requirements.lock`` into ``{canonical name: version}``.

    Accepts the ``uv pip compile`` / ``pip-compile`` output shape: ``name==version`` lines,
    ``# ...`` comments, indented ``# via`` provenance lines, ``--index-url``-style options and
    environment markers (``; python_version < '3.13'``). Anything that is not an exact ``==``
    pin is reported by :func:`verify_environment` rather than silently dropped, so a range
    sneaking into a lockfile cannot pass as a pin.
    """
    _req, _utils, _ = _require_packaging()
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        line = line.split(" ;")[0].split(";")[0].split(" #")[0].strip()
        if not line:
            continue
        parsed: Requirement = _req.Requirement(line)
        pins[_utils.canonicalize_name(parsed.name)] = str(parsed.specifier)
    return pins


def _exact_version(specifier: str) -> str | None:
    """The pinned version of an exact ``==x.y.z`` specifier string, else ``None``."""
    if not specifier.startswith("==") or "," in specifier:
        return None
    return specifier[2:]


def _satisfies(version: Version, requirement: Requirement) -> bool:
    """Does ``version`` satisfy ``requirement``'s specifier (pre-releases allowed)?"""
    return requirement.specifier.contains(version, prereleases=True)


# --------------------------------------------------------------------------------------------
# Verification.
# --------------------------------------------------------------------------------------------


def verify_environment(spec: EnvironmentSpec, repo_root: Path) -> EnvironmentReport:
    """Check the committed environment against the workspace it claims to stand up.

    Offline and import-free: reads ``pyproject.toml`` metadata, the generated files and the lock.
    Returns every finding (never short-circuits) so one run shows the whole picture.
    """
    _req, _utils, _ver = _require_packaging()
    findings: list[Finding] = []

    packages: list[_PackageMeta] = []
    for rel_path in spec.workspace_packages:
        pyproject = repo_root / rel_path / "pyproject.toml"
        if not pyproject.is_file():
            findings.append(
                Finding(
                    "E-PACKAGE-MISSING",
                    f"workspace package '{rel_path}' has no pyproject.toml at {pyproject}",
                )
            )
            continue
        meta = _read_package(repo_root, rel_path)
        if meta.name != Path(rel_path).name:
            findings.append(
                Finding(
                    "E-PACKAGE-NAME-MISMATCH",
                    f"workspace package '{rel_path}' declares project name '{meta.name}'",
                )
            )
        packages.append(meta)

    # Resolution-time refusals run FIRST and are lock-independent: an environment whose package
    # set cannot resolve is broken whether or not it has ever been solved (issue #198).
    findings.extend(check_resolution_conflicts(spec, repo_root))

    findings.extend(_check_generated(spec, repo_root, "requirements", render_requirements))
    findings.extend(_check_generated(spec, repo_root, "constraints", render_constraints))

    if not spec.lock_path.is_file():
        findings.append(
            Finding("E-LOCK-MISSING", f"lockfile not found: {spec.lock_path} (run install.sh -l)")
        )
        return EnvironmentReport(spec.name, tuple(findings))

    pins = parse_lock(spec.lock_path.read_text(encoding="utf-8"))
    versions: dict[str, Version] = {}
    for dist, specifier in sorted(pins.items()):
        exact = _exact_version(specifier)
        if exact is None:
            findings.append(
                Finding(
                    "E-LOCK-NOT-PINNED",
                    f"lockfile entry '{dist}{specifier}' is not an exact '==' pin; a lockfile "
                    "records a solved version, never a range",
                )
            )
            continue
        versions[dist] = _ver.Version(exact)

    # Every declared third-party requirement of every package in the set must be (a) present in
    # the lock and (b) satisfied by the locked version. This is the drift alarm: raise a floor or
    # drop a cap in any Provider package and the committed lock stops satisfying it.
    for requirement_str, owners in _third_party_requirements(packages):
        requirement: Requirement = _req.Requirement(requirement_str)
        dist = _utils.canonicalize_name(requirement.name)
        locked = versions.get(dist)
        if locked is None:
            findings.append(
                Finding(
                    "E-LOCK-INCOMPLETE",
                    f"'{requirement_str}' (declared by {', '.join(owners)}) has no pin in "
                    f"{spec.lock_file}",
                )
            )
        elif not _satisfies(locked, requirement):
            findings.append(
                Finding(
                    "E-LOCK-UNSATISFIED",
                    f"locked {dist}=={locked} does not satisfy '{requirement_str}' declared by "
                    f"{', '.join(owners)}",
                )
            )
        for extra in sorted(requirement.extras):
            del extra  # extras pull transitive dists the lock covers; nothing to assert per-extra

    for constraint in spec.constraints:
        constraint_req: Requirement = _req.Requirement(constraint.requirement)
        dist = _utils.canonicalize_name(constraint_req.name)
        locked = versions.get(dist)
        if locked is None:
            findings.append(
                Finding(
                    "E-CONSTRAINT-UNUSED",
                    f"enforced boundary '{constraint.requirement}' names a distribution absent "
                    f"from {spec.lock_file}; a constraint nothing binds is folklore — drop it or "
                    "fix the lock",
                )
            )
        elif not _satisfies(locked, constraint_req):
            findings.append(
                Finding(
                    "E-CONSTRAINT-VIOLATED",
                    f"locked {dist}=={locked} violates the enforced boundary "
                    f"'{constraint.requirement}' ({constraint.reason.strip().splitlines()[0]})",
                )
            )

    findings.extend(_check_providers(spec, packages))
    return EnvironmentReport(spec.name, tuple(findings))


# --------------------------------------------------------------------------------------------
# Resolution-time refusals (issue #198): incompatibilities that must NEVER reach a model load.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Interval:
    """A version specifier reduced to a closed/open interval, for exact disjointness math."""

    low: Version | None
    low_inclusive: bool
    high: Version | None
    high_inclusive: bool

    def intersect(self, other: _Interval) -> _Interval:
        low, low_inc = self.low, self.low_inclusive
        if other.low is not None and (low is None or other.low > low):
            low, low_inc = other.low, other.low_inclusive
        elif other.low is not None and other.low == low:
            low_inc = low_inc and other.low_inclusive
        high, high_inc = self.high, self.high_inclusive
        if other.high is not None and (high is None or other.high < high):
            high, high_inc = other.high, other.high_inclusive
        elif other.high is not None and other.high == high:
            high_inc = high_inc and other.high_inclusive
        return _Interval(low, low_inc, high, high_inc)

    @property
    def empty(self) -> bool:
        if self.low is None or self.high is None:
            return False
        if self.low > self.high:
            return True
        return self.low == self.high and not (self.low_inclusive and self.high_inclusive)


def _as_interval(specifier: Any) -> _Interval | None:
    """Reduce a :class:`SpecifierSet` to an interval, or ``None`` when it is not interval-shaped.

    Deliberately **sound, not complete**: only the bound operators (``>=``/``>``/``<``/``<=``/
    ``==`` with no wildcard) reduce to an interval. ``!=``/``~=``/``===``/wildcards make the set
    non-convex, and rather than guess, this returns ``None`` and the caller skips the check. The
    consequence is that a conflict can go *unreported* (a false negative), never that a satisfiable
    pair is reported as conflicting (a false positive) — a verifier that cries wolf gets disabled.
    """
    _, _, _ver = _require_packaging()
    interval = _Interval(None, True, None, True)
    for spec in specifier:
        if spec.operator not in {">=", ">", "<", "<=", "=="} or spec.version.endswith("*"):
            return None
        version = _ver.Version(spec.version)
        if spec.operator == "==":
            bound = _Interval(version, True, version, True)
        elif spec.operator == ">=":
            bound = _Interval(version, True, None, True)
        elif spec.operator == ">":
            bound = _Interval(version, False, None, True)
        elif spec.operator == "<=":
            bound = _Interval(None, True, version, True)
        else:  # "<"
            bound = _Interval(None, True, version, False)
        interval = interval.intersect(bound)
    return interval


def check_resolution_conflicts(spec: EnvironmentSpec, repo_root: Path) -> tuple[Finding, ...]:
    """Refuse an environment whose package set cannot resolve — **before** anything installs.

    Two refusals, both lock-independent (they hold even for an environment that has never been
    solved), which is what makes them *dependency-resolution* failures rather than runtime ones:

    * ``E-CONSTRAINT-CONFLICT`` — a package in the set declares a requirement whose version window
      is **disjoint** from an enforced conflict boundary. Adding a ``transformers>=5`` package to
      an environment that enforces ``transformers<5`` is refused here, naming both sides, instead
      of resolving to something and exploding at model load.
    * ``E-PACKAGE-EXCLUDED`` — a package the spec names as incompatible-by-construction is in the
      set, either directly or transitively through another package's workspace dependencies. This
      covers the incompatibilities that version metadata cannot express (see
      :class:`ExcludedPackage`).
    """
    _req, _utils, _ = _require_packaging()
    findings: list[Finding] = []

    boundaries: list[tuple[Constraint, Requirement, _Interval]] = []
    for constraint in spec.constraints:
        parsed: Requirement = _req.Requirement(constraint.requirement)
        interval = _as_interval(parsed.specifier)
        if interval is not None:
            boundaries.append((constraint, parsed, interval))

    packages: list[_PackageMeta] = []
    for rel_path in spec.workspace_packages:
        if (repo_root / rel_path / "pyproject.toml").is_file():
            packages.append(_read_package(repo_root, rel_path))

    for requirement_str, owners in _third_party_requirements(packages):
        requirement: Requirement = _req.Requirement(requirement_str)
        declared = _as_interval(requirement.specifier)
        if declared is None:
            continue
        dist = _utils.canonicalize_name(requirement.name)
        for constraint, parsed, boundary in boundaries:
            if _utils.canonicalize_name(parsed.name) != dist:
                continue
            if declared.intersect(boundary).empty:
                findings.append(
                    Finding(
                        "E-CONSTRAINT-CONFLICT",
                        f"'{requirement_str}' (declared by {', '.join(owners)}) is disjoint from "
                        f"the enforced boundary '{constraint.requirement}' — this environment "
                        "cannot resolve. "
                        f"{constraint.reason.strip().splitlines()[0]}",
                    )
                )

    findings.extend(_check_excluded(spec, repo_root, packages))
    return tuple(findings)


def _check_excluded(
    spec: EnvironmentSpec, repo_root: Path, packages: list[_PackageMeta]
) -> list[Finding]:
    """Refuse an excluded package, whether named directly or dragged in as a workspace dependency."""
    if not spec.excluded_packages:
        return []
    excluded_by_path = {e.package: e for e in spec.excluded_packages}
    excluded_by_dist: dict[str, ExcludedPackage] = {}
    for entry in spec.excluded_packages:
        pyproject = repo_root / entry.package / "pyproject.toml"
        dist = Path(entry.package).name
        if pyproject.is_file():
            dist = str(tomllib.loads(pyproject.read_text(encoding="utf-8"))
                       .get("project", {}).get("name", dist))
        excluded_by_dist[dist.lower().replace("_", "-")] = entry

    findings: list[Finding] = []
    for rel_path in spec.workspace_packages:
        entry = excluded_by_path.get(rel_path)
        if entry is not None:
            findings.append(
                Finding(
                    "E-PACKAGE-EXCLUDED",
                    f"'{rel_path}' is in workspace_packages but this environment excludes it: "
                    f"{entry.reason.strip().splitlines()[0]}",
                )
            )
    installed = {p.rel_path: p for p in packages}
    for rel_path, meta in sorted(installed.items()):
        for dep_dist in sorted(meta.workspace_dependencies):
            entry = excluded_by_dist.get(dep_dist)
            if entry is None:
                continue
            findings.append(
                Finding(
                    "E-PACKAGE-EXCLUDED",
                    f"'{rel_path}' requires '{dep_dist}', which this environment excludes "
                    f"('{entry.package}') — installing it drags the excluded package in. "
                    f"{entry.reason.strip().splitlines()[0]}",
                )
            )
    return findings


def _check_generated(
    spec: EnvironmentSpec,
    repo_root: Path,
    kind: str,
    render: Any,
) -> list[Finding]:
    path = spec.requirements_path if kind == "requirements" else spec.constraints_path
    try:
        expected = render(spec, repo_root)
    except ConfigError:
        # A package the environment names is missing; E-PACKAGE-MISSING already says so and the
        # render comparison would only add noise.
        return []
    if not path.is_file():
        return [Finding(f"E-{kind.upper()}-MISSING", f"generated file not found: {path}")]
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        return [
            Finding(
                f"E-{kind.upper()}-STALE",
                f"{path.name} does not match the render of {spec.spec_path.name} + the workspace "
                f"packages' declared dependencies — a dependency changed. Regenerate with "
                f"`latence env render {_spec_label(spec, repo_root)}` and re-solve the lock.",
            )
        ]
    return []


def _check_providers(spec: EnvironmentSpec, packages: list[_PackageMeta]) -> list[Finding]:
    """The Provider set the recipe promises must exist, and the smoke stack must stay inside it."""
    findings: list[Finding] = []
    registered: set[str] = set()
    for pkg in packages:
        registered |= pkg.provider_entry_points

    for provider in spec.providers:
        if provider not in registered:
            findings.append(
                Finding(
                    "E-PROVIDER-UNREGISTERED",
                    f"provider '{provider}' is not declared by any 'latence.providers' "
                    "entry-point of this environment's workspace packages (renamed, moved, or "
                    "the package is missing from workspace_packages)",
                )
            )

    unknown = spec.gpu_only_providers - set(spec.providers)
    for provider in sorted(unknown):
        findings.append(
            Finding(
                "E-GPU-ONLY-UNKNOWN",
                f"gpu_only_providers lists '{provider}', which is not in this environment's "
                "providers",
            )
        )

    for provider in sorted(spec.learned_providers - set(spec.providers)):
        findings.append(
            Finding(
                "E-LEARNED-UNKNOWN",
                f"learned_providers lists '{provider}', which is not in this environment's "
                "providers",
            )
        )

    if not spec.stack_path.is_file():
        findings.append(Finding("E-STACK-MISSING", f"smoke stack not found: {spec.stack_path}"))
        return findings

    # A learned Provider the environment declares but the stack never instantiates is never
    # driven, so the GPU-placement gate would silently have nothing to say about it (#198).
    in_stack = stack_providers(spec.stack_path)
    for provider in sorted(spec.learned_providers - in_stack):
        findings.append(
            Finding(
                "E-LEARNED-UNDRIVEN",
                f"learned provider '{provider}' never appears in {spec.stack_file}, so the "
                "GPU-placement probe can never exercise it — an unexercised learned Provider is "
                "exactly where a silent CPU fallback hides (#191)",
            )
        )

    for provider in sorted(in_stack):
        if provider not in spec.providers:
            findings.append(
                Finding(
                    "E-STACK-PROVIDER-UNDECLARED",
                    f"{spec.stack_file} uses provider '{provider}', which the environment does "
                    "not declare — the recipe and the stack it is meant to stand up have drifted",
                )
            )
    return findings


def probe_providers(spec: EnvironmentSpec) -> EnvironmentReport:
    """LIVE check: every declared Provider resolves from the registry *in this interpreter*.

    Unlike :func:`verify_environment` this one IMPORTS — ``ProviderRegistry.load`` executes the
    Provider module, and therefore the heavy stack behind it. That import is precisely where a
    ``transformers``/``torch``/``huggingface-hub`` drift detonates (a mismatched major raises at
    module import, long before any weights are fetched), which makes it the sharpest cheap proof
    that the installed environment actually stands up. It is meaningless outside an installed
    environment, so it is opt-in (``latence env verify --probe``) rather than part of the offline
    verification.
    """
    from latence_core.capability import ProviderRegistry

    registry = ProviderRegistry()
    findings: list[Finding] = []
    for provider in spec.providers:
        try:
            registry.load(provider)
        except Exception as exc:  # noqa: BLE001 - any import failure is the finding
            findings.append(
                Finding(
                    "E-PROVIDER-IMPORT",
                    f"provider '{provider}' did not load in this environment: "
                    f"{type(exc).__name__}: {exc}",
                )
            )
    return EnvironmentReport(spec.name, tuple(findings))


@dataclass(frozen=True)
class StackComponent:
    """One ``provider`` + its ``config`` as a stack YAML declares it (issue #198).

    Covers Stage-level components AND the nested signal components (``embedder`` /
    ``sparse_embedder`` / ``multivector_embedder``): both are real, separately-loaded models, and
    the GPU-placement probe must exercise the nested Granite embedder exactly as it exercises the
    Stage-level extractor. ``capability`` is the Stage's declared capability, or the nesting key
    for a nested component — either way it is what selects the driver that runs the model.
    """

    capability: str
    provider: str
    config: dict[str, Any]


# The nesting keys under which a stack declares a *signal* component (a separately-loaded model
# inside a Stage's config). The key doubles as the capability name for driver selection.
_NESTED_COMPONENT_KEYS = ("embedder", "sparse_embedder", "multivector_embedder")


def stack_components(stack_path: Path) -> tuple[StackComponent, ...]:
    """Every provider a stack YAML instantiates, with its capability and config, in file order."""
    raw = yaml.safe_load(stack_path.read_text(encoding="utf-8"))
    found: list[StackComponent] = []

    def walk(node: Any, capability: str) -> None:
        if not isinstance(node, dict):
            if isinstance(node, list):
                for item in node:
                    walk(item, capability)
            return
        provider = node.get("provider")
        config = node.get("config") if isinstance(node.get("config"), dict) else {}
        own_capability = node.get("capability") if isinstance(node.get("capability"), str) else None
        if isinstance(provider, str):
            found.append(
                StackComponent(
                    capability=own_capability or capability,
                    provider=provider,
                    config=dict(config),
                )
            )
        for key, value in node.items():
            if key == "config" and isinstance(value, dict):
                for nested_key in _NESTED_COMPONENT_KEYS:
                    if nested_key in value:
                        walk(value[nested_key], nested_key)
            elif key not in {"provider", "capability"}:
                walk(value, own_capability or capability)

    walk(raw, "")
    return tuple(found)


def stack_providers(stack_path: Path) -> set[str]:
    """Every provider name a stack YAML references, including nested signal Providers.

    Nested (``embedder`` / ``sparse_embedder`` / ``multivector_embedder``) Providers are as real a
    part of the environment as the Stage ones — the pod post-mortems were full of nested-embedder
    surprises — so the walk is structural rather than a fixed key list.
    """
    raw = yaml.safe_load(stack_path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            provider = node.get("provider")
            if isinstance(provider, str):
                found.add(provider)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    return found
