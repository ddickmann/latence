"""The reproducible-environment regression guard (issue #195).

The gap this closes: standing up the enterprise-sota Provider set was per-session trial-and-error,
and every failure — a ``transformers`` major bump, a ``huggingface-hub`` 1.x, a
``tokenizers`` patch outside the window, a torch cap dropped — was discovered on a pod, at
runtime, hours in. :mod:`latence_core.envspec` turns the environment into a *specification the
build checks*, and this module is the check.

Two things are proved here, and the second is the one that matters:

* every committed environment under ``deploy/env/*/`` verifies clean, and
* the verifier **genuinely bites** — each drift the ticket names is injected into a real copy of
  a real environment and MUST be reported. A guard that cannot be made to fail is not a guard.

Everything runs offline and import-free: the verifier reads ``pyproject.toml`` metadata, the
generated files and the lock. No install, no network, no Provider import.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from latence_core.envspec import (
    load_environment,
    render_constraints,
    render_requirements,
    stack_providers,
    verify_environment,
)
from latence_core.errors import ConfigError

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_ROOT = _REPO_ROOT / "deploy" / "env"

_ENV_SPECS = sorted(_ENV_ROOT.glob("*/environment.yaml"))


def _ids(paths: list[Path]) -> list[str]:
    return [p.parent.name for p in paths]


# --------------------------------------------------------------------------------------------
# The committed environments are green (and there IS at least one).
# --------------------------------------------------------------------------------------------


def test_at_least_one_environment_is_committed() -> None:
    """A parametrised suite over an empty glob passes vacuously; this makes that impossible."""
    assert _ENV_SPECS, f"no deploy/env/*/environment.yaml found under {_ENV_ROOT}"


@pytest.mark.parametrize("spec_path", _ENV_SPECS, ids=_ids(_ENV_SPECS))
def test_committed_environment_verifies_clean(spec_path: Path) -> None:
    """Every committed environment's pins, boundaries, providers and stack agree.

    This is the assertion that turns red on dependency drift: change a Provider package's
    declared dependency and the generated ``requirements.in`` goes stale; raise a floor or drop a
    cap the lock no longer satisfies and the lock check fires.
    """
    report = verify_environment(load_environment(spec_path), _REPO_ROOT)
    assert report.ok, report.describe()


@pytest.mark.parametrize("spec_path", _ENV_SPECS, ids=_ids(_ENV_SPECS))
def test_generated_files_are_reproducible(spec_path: Path) -> None:
    """Rendering twice yields the same bytes — the byte-comparison guard needs determinism."""
    spec = load_environment(spec_path)
    assert render_requirements(spec, _REPO_ROOT) == render_requirements(spec, _REPO_ROOT)
    assert render_constraints(spec, _REPO_ROOT) == render_constraints(spec, _REPO_ROOT)


@pytest.mark.parametrize("spec_path", _ENV_SPECS, ids=_ids(_ENV_SPECS))
def test_every_enforced_boundary_carries_a_reason(spec_path: Path) -> None:
    """A boundary nobody can explain is folklore; the reason is rendered into constraints.txt."""
    spec = load_environment(spec_path)
    assert spec.constraints, "an environment with no enforced boundary is not pinning anything"
    rendered = render_constraints(spec, _REPO_ROOT)
    for constraint in spec.constraints:
        assert constraint.reason.strip(), constraint.requirement
        for line in constraint.reason.strip().splitlines():
            assert f"# {line}".rstrip() in rendered, constraint.requirement


@pytest.mark.parametrize("spec_path", _ENV_SPECS, ids=_ids(_ENV_SPECS))
def test_install_and_smoke_scripts_ship_with_every_environment(spec_path: Path) -> None:
    """An environment without a one-command install + smoke is a wiki page, not a recipe."""
    env_dir = spec_path.parent
    for script in ("install.sh", "smoke.sh"):
        assert (env_dir / script).is_file(), f"{env_dir.name} is missing {script}"


# --------------------------------------------------------------------------------------------
# The guard genuinely bites. Each case mutates a real copy of a real environment.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A writable copy of the whole repo skeleton the verifier reads (packages/ + deploy/env/).

    Copying rather than monkeypatching keeps the tests on the PUBLIC surface: they drive
    ``verify_environment`` over a real directory tree exactly as the CLI does.
    """
    root = tmp_path / "repo"
    (root / "deploy").mkdir(parents=True)
    shutil.copytree(_ENV_ROOT, root / "deploy" / "env")
    packages = root / "packages"
    packages.mkdir()
    for pkg in sorted((_REPO_ROOT / "packages").iterdir()):
        if (pkg / "pyproject.toml").is_file():
            (packages / pkg.name).mkdir()
            shutil.copy(pkg / "pyproject.toml", packages / pkg.name / "pyproject.toml")
    return root


def _spec_in(sandbox: Path, name: str = "cpu-text") -> Path:
    return sandbox / "deploy" / "env" / name / "environment.yaml"


def _codes(sandbox: Path, name: str = "cpu-text") -> set[str]:
    report = verify_environment(load_environment(_spec_in(sandbox, name)), sandbox)
    return {f.code for f in report.findings}


def test_sandbox_copy_is_green_before_mutation(sandbox: Path) -> None:
    """The baseline: without a mutation the copy verifies clean, so each finding below is causal."""
    assert _codes(sandbox) == set()


def test_a_new_dependency_on_a_provider_package_is_caught(sandbox: Path) -> None:
    """THE drift alarm: a heavy dep added to a Provider package with no pin in the recipe.

    This is the ticket's failure mode in miniature — ``torchcodec`` appears in a Provider's
    dependencies, nothing pins it, and the environment silently stops being reproducible.
    """
    pyproject = sandbox / "packages" / "latence-extract-gliner2" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '"peft>=0.10",', '"peft>=0.10",\n    "torchcodec>=0.2",'
        ),
        encoding="utf-8",
    )
    assert "E-REQUIREMENTS-STALE" in _codes(sandbox)


def test_a_raised_floor_the_lock_cannot_satisfy_is_caught(sandbox: Path) -> None:
    """A Provider package moves to a transformers major the solved lock does not carry."""
    pyproject = sandbox / "packages" / "latence-disambig-glinker" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '"transformers>=4.40,<5",', '"transformers>=5.0,<6",'
        ),
        encoding="utf-8",
    )
    codes = _codes(sandbox)
    # The declared range changed, so the generated file is stale AND the lock no longer satisfies
    # the declaration. Both are true and both must be reported.
    assert "E-LOCK-UNSATISFIED" in codes
    assert "E-REQUIREMENTS-STALE" in codes


def test_a_violated_conflict_boundary_is_caught(sandbox: Path) -> None:
    """The lock drifts past an ENFORCED boundary — the exact transformers-5.x break, simulated.

    Constraints are enforced, not documentation: a lock that violates one is a finding even
    though every *package-declared* range still holds.
    """
    lock = sandbox / "deploy" / "env" / "cpu-text" / "requirements.lock"
    text = lock.read_text(encoding="utf-8")
    assert "transformers==4.57.6" in text
    lock.write_text(text.replace("transformers==4.57.6", "transformers==4.50.0"), encoding="utf-8")
    assert "E-CONSTRAINT-VIOLATED" in _codes(sandbox)


def test_a_range_masquerading_as_a_lock_pin_is_caught(sandbox: Path) -> None:
    """A lockfile records solved versions; a range in it is a lock that does not lock."""
    lock = sandbox / "deploy" / "env" / "cpu-text" / "requirements.lock"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace("torch==2.13.0+cpu", "torch>=2.1"),
        encoding="utf-8",
    )
    assert "E-LOCK-NOT-PINNED" in _codes(sandbox)


def test_a_missing_lock_pin_is_caught(sandbox: Path) -> None:
    """A declared dependency with no entry in the lock is an unreproducible install."""
    lock = sandbox / "deploy" / "env" / "cpu-text" / "requirements.lock"
    kept = [
        line
        for line in lock.read_text(encoding="utf-8").splitlines()
        if not line.startswith("pdfplumber==")
    ]
    lock.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert "E-LOCK-INCOMPLETE" in _codes(sandbox)


def test_a_renamed_provider_is_caught(sandbox: Path) -> None:
    """The recipe promises a Provider set; a rename under it must not pass silently."""
    pyproject = sandbox / "packages" / "latence-splade" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('"sparse.splade"', '"sparse.splade_v2"'),
        encoding="utf-8",
    )
    codes = _codes(sandbox)
    assert "E-PROVIDER-UNREGISTERED" in codes


def test_a_stack_using_an_undeclared_provider_is_caught(sandbox: Path) -> None:
    """The smoke stack may not drift outside the Provider set the environment installs."""
    stack = sandbox / "deploy" / "env" / "cpu-text" / "stack.yaml"
    stack.write_text(
        stack.read_text(encoding="utf-8").replace(
            "provider: chunk.markdown", "provider: chunk.not_installed"
        ),
        encoding="utf-8",
    )
    assert "E-STACK-PROVIDER-UNDECLARED" in _codes(sandbox)


def test_a_hand_edited_constraints_file_is_caught(sandbox: Path) -> None:
    """constraints.txt is generated; a hand edit (adding OR relaxing a boundary) is drift."""
    constraints = sandbox / "deploy" / "env" / "cpu-text" / "constraints.txt"
    constraints.write_text(
        constraints.read_text(encoding="utf-8").replace(
            "transformers>=4.51.3,<5", "transformers>=4.51.3"
        ),
        encoding="utf-8",
    )
    assert "E-CONSTRAINTS-STALE" in _codes(sandbox)


def test_a_missing_workspace_package_is_caught(sandbox: Path) -> None:
    """An environment that names a package the workspace does not have cannot stand up."""
    shutil.rmtree(sandbox / "packages" / "latence-embedder-st")
    assert "E-PACKAGE-MISSING" in _codes(sandbox)


# --------------------------------------------------------------------------------------------
# Spec parsing rejects what a real recipe would reject.
# --------------------------------------------------------------------------------------------


def test_a_constraint_without_a_reason_is_rejected(tmp_path: Path) -> None:
    """The reason is not decoration: an unexplained boundary is exactly what rots into folklore."""
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        "name: x\nsummary: x\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: [packages/latence-core]\nproviders: [source.local_folder]\n"
        "constraints:\n  - requirement: 'transformers<5'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="reason"):
        load_environment(spec)


def test_an_environment_without_providers_is_rejected(tmp_path: Path) -> None:
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        "name: x\nsummary: x\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: [packages/latence-core]\nproviders: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="providers"):
        load_environment(spec)


def test_a_missing_spec_is_a_config_error_not_a_skip(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_environment(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------------------------
# The stack walk sees nested signal Providers, not just Stage ones.
# --------------------------------------------------------------------------------------------


def test_stack_walk_finds_nested_signal_providers(tmp_path: Path) -> None:
    """Nested embedders were a repeated source of pod surprises; they are part of the environment."""
    stack = tmp_path / "stack.yaml"
    stack.write_text(
        "name: s\nstages:\n"
        "  - name: export\n    capability: export\n    provider: export.jsonl_parquet\n"
        "    config:\n"
        "      embedder: {provider: embedding.sentence_transformers}\n"
        "      sparse_embedder: {provider: sparse.splade}\n",
        encoding="utf-8",
    )
    assert stack_providers(stack) == {
        "export.jsonl_parquet",
        "embedding.sentence_transformers",
        "sparse.splade",
    }


# --------------------------------------------------------------------------------------------
# The transformers split fails at DEPENDENCY RESOLUTION, not at runtime (issue #198).
#
# Both refusals below are lock-independent: they hold for an environment that has never been
# solved, which is what makes them resolution-time failures. Gate 1 of ``smoke.sh`` runs them
# before ``install.sh``, so the unsupported combination cannot reach a model load.
# --------------------------------------------------------------------------------------------

_GPU_ENV = "gpu-learned"


def test_gpu_environment_enforces_the_transformers_four_window(sandbox: Path) -> None:
    """The committed resolution is ONE window, not a menu: 4.51.3 <= transformers < 5."""
    spec = load_environment(_spec_in(sandbox, _GPU_ENV))
    windows = [c.requirement for c in spec.constraints if c.requirement.startswith("transformers")]
    assert windows == ["transformers>=4.51.3,<5"]


def test_a_transformers_five_package_cannot_resolve_into_the_gpu_environment(sandbox: Path) -> None:
    """A package on the >=5 side of the split is refused by the BOUNDARY, before anything installs.

    The disjointness is computed from the declared window against the enforced one — no lock, no
    install, no import. This is the mechanism that makes the split a resolution failure.
    """
    pyproject = sandbox / "packages" / "latence-extract-gliner2" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '"peft>=0.10",', '"peft>=0.10",\n    "transformers>=5.0",'
        ),
        encoding="utf-8",
    )
    codes = _codes(sandbox, _GPU_ENV)
    assert "E-CONSTRAINT-CONFLICT" in codes


def test_a_satisfiable_window_is_not_reported_as_a_conflict(sandbox: Path) -> None:
    """The conflict check must never cry wolf: an overlapping window is not a conflict.

    A verifier that reports satisfiable pairs gets switched off, and then it guards nothing.
    """
    pyproject = sandbox / "packages" / "latence-extract-gliner2" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '"peft>=0.10",', '"peft>=0.10",\n    "transformers>=4.52,<4.99",'
        ),
        encoding="utf-8",
    )
    assert "E-CONSTRAINT-CONFLICT" not in _codes(sandbox, _GPU_ENV)


def test_the_in_process_ocr_vlm_is_refused_by_name(sandbox: Path) -> None:
    """The half of the split version metadata cannot express.

    ``latence-parser-lighton`` declares an UNCAPPED ``transformers>=4.45``, so no version math
    refuses it — a resolver installs it happily next to transformers 4.57 and the incompatibility
    detonates at ``from_pretrained``. The named exclusion is what turns that into a gate-1 refusal.
    """
    spec_path = _spec_in(sandbox, _GPU_ENV)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            "  - packages/latence-parser-pdfplumber\n",
            "  - packages/latence-parser-pdfplumber\n  - packages/latence-parser-lighton\n",
        ),
        encoding="utf-8",
    )
    assert "E-PACKAGE-EXCLUDED" in _codes(sandbox, _GPU_ENV)


def test_an_excluded_package_dragged_in_transitively_is_refused(sandbox: Path) -> None:
    """An excluded package can arrive as somebody else's dependency, never named in the spec.

    That is exactly the shape ``latence-parser-lighton-vllm`` has in the real workspace: it
    depends on ``latence-parser-lighton``. Here the same shape is induced on a package the GPU
    environment DOES install, so the transitive rule is what has to catch it — checking only the
    names listed in ``workspace_packages`` would see nothing at all.
    """
    pyproject = sandbox / "packages" / "latence-parser-pdfplumber" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            '"latence-core"', '"latence-core", "latence-parser-lighton"'
        ),
        encoding="utf-8",
    )
    findings = verify_environment(load_environment(_spec_in(sandbox, _GPU_ENV)), sandbox).findings
    dragged = [
        f
        for f in findings
        if f.code == "E-PACKAGE-EXCLUDED" and "latence-parser-pdfplumber" in f.message
    ]
    assert dragged, [str(f) for f in findings]
    assert "drags the excluded package in" in dragged[0].message


def test_every_excluded_package_carries_a_reason(sandbox: Path) -> None:
    """A refusal nobody can explain is folklore, exactly like an unexplained boundary."""
    spec = load_environment(_spec_in(sandbox, _GPU_ENV))
    assert spec.excluded_packages
    for excluded in spec.excluded_packages:
        assert excluded.reason.strip(), excluded.package


# --------------------------------------------------------------------------------------------
# A learned Provider the GPU gate can never speak for is a hole in the gate (issue #198).
# --------------------------------------------------------------------------------------------


def test_a_learned_provider_missing_from_the_stack_is_caught(sandbox: Path) -> None:
    """Declared-but-never-instantiated means the placement probe silently never drives it."""
    stack = sandbox / "deploy" / "env" / _GPU_ENV / "stack.yaml"
    kept = [
        line
        for line in stack.read_text(encoding="utf-8").splitlines()
        if "provider: redaction.gliner_pii" not in line
    ]
    stack.write_text("\n".join(kept) + "\n", encoding="utf-8")
    codes = _codes(sandbox, _GPU_ENV)
    assert "E-LEARNED-UNDRIVEN" in codes


def test_a_learned_provider_outside_the_environment_is_caught(sandbox: Path) -> None:
    """``learned_providers`` cannot name a Provider the environment does not even install."""
    spec_path = _spec_in(sandbox, _GPU_ENV)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            "learned_providers:\n", "learned_providers:\n  - redaction.does_not_exist\n"
        ),
        encoding="utf-8",
    )
    assert "E-LEARNED-UNKNOWN" in _codes(sandbox, _GPU_ENV)


def test_stack_components_carry_capability_and_config_for_nested_providers() -> None:
    """The placement probe drives components, so it needs the capability AND the config of each."""
    from latence_core.envspec import stack_components

    spec = load_environment(_ENV_ROOT / _GPU_ENV / "environment.yaml")
    by_provider = {c.provider: c for c in stack_components(spec.stack_path)}
    # A nested signal Provider is as real a model as a Stage-level one; its nesting key is its
    # capability, and its config is what the probe constructs it with.
    nested = by_provider["embedding.sentence_transformers"]
    assert nested.capability == "embedder"
    assert nested.config["device"] == "cuda"
    stage = by_provider["redaction.gliner_pii"]
    assert stage.capability == "redaction"
    assert stage.config["device"] == "cuda"


def test_every_learned_provider_in_the_gpu_stack_is_pinned_to_cuda() -> None:
    """``device: auto`` would resolve to cpu on a CPU host and quietly pass a GPU smoke test."""
    from latence_core.envspec import stack_components

    spec = load_environment(_ENV_ROOT / _GPU_ENV / "environment.yaml")
    for component in stack_components(spec.stack_path):
        if component.provider in spec.learned_providers:
            assert component.config.get("device") == "cuda", component.provider
