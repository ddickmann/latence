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
    providers: tuple[str, ...]
    """Provider names the environment must expose (the smoke test's strict assertion)."""
    gpu_only_providers: frozenset[str]
    """Providers whose *runtime* needs a GPU: installed + registered here, exercised by #198."""
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
        providers=providers,
        gpu_only_providers=frozenset(_strs("gpu_only_providers")),
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
            parsed = _req.Requirement(dep)
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
        parsed = _req.Requirement(line)
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
        requirement = _req.Requirement(requirement_str)
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
        requirement = _req.Requirement(constraint.requirement)
        dist = _utils.canonicalize_name(requirement.name)
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
        elif not _satisfies(locked, requirement):
            findings.append(
                Finding(
                    "E-CONSTRAINT-VIOLATED",
                    f"locked {dist}=={locked} violates the enforced boundary "
                    f"'{constraint.requirement}' ({constraint.reason.strip().splitlines()[0]})",
                )
            )

    findings.extend(_check_providers(spec, packages))
    return EnvironmentReport(spec.name, tuple(findings))


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

    if not spec.stack_path.is_file():
        findings.append(Finding("E-STACK-MISSING", f"smoke stack not found: {spec.stack_path}"))
        return findings
    for provider in sorted(stack_providers(spec.stack_path)):
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
