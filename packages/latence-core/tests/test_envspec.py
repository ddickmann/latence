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
            '"peft>=0.10,<1",', '"peft>=0.10",\n    "torchcodec>=0.2",'
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
    pyproject = sandbox / "packages" / "latence-embedder-st" / "pyproject.toml"
    original = pyproject.read_text(encoding="utf-8")
    renamed = original.replace(
        '"embedding.sentence_transformers"', '"embedding.sentence_transformers_v2"'
    )
    assert renamed != original, "the mutation must actually rename the registered Provider"
    pyproject.write_text(renamed, encoding="utf-8")
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


def test_an_environment_that_installs_packages_without_providers_is_rejected(
    tmp_path: Path,
) -> None:
    """The Provider set is what a CLIENT environment promises; omitting it promises nothing."""
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        "name: x\nsummary: x\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: [packages/latence-core]\nproviders: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="providers"):
        load_environment(spec)


def test_an_environment_that_installs_nothing_at_all_is_rejected(tmp_path: Path) -> None:
    """Service environments may skip workspace packages — but not by installing nothing."""
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        "name: x\nsummary: x\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: []\nproviders: []\nextra_requirements: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="installs nothing"):
        load_environment(spec)


def test_providers_without_a_package_to_register_them_is_rejected(tmp_path: Path) -> None:
    """A service environment cannot promise Providers: nothing there declares an entry-point."""
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        "name: x\nsummary: x\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: []\nproviders: [source.local_folder]\n"
        "extra_requirements: ['vllm==0.24.0']\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="no workspace package"):
        load_environment(spec)


def test_an_exclusion_without_a_reason_is_rejected(tmp_path: Path) -> None:
    """An unexplained exclusion is the wheel the next person quietly re-adds."""
    spec = tmp_path / "environment.yaml"
    spec.write_text(
        "name: x\nsummary: x\npython: '>=3.11'\nplatform: linux-x86_64\n"
        "workspace_packages: [packages/latence-core]\nproviders: [source.local_folder]\n"
        "excluded_distributions:\n  - package: flash-attn\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="reason"):
        load_environment(spec)


def test_a_missing_spec_is_a_config_error_not_a_skip(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_environment(tmp_path / "nope.yaml")


# --------------------------------------------------------------------------------------------
# EXCLUSIONS — the wheels that must be IMPOSSIBLE to install (issue #196).
#
# The live Blackwell/CUDA-13 run lost hours to prebuilt `flash-attn` and `torchcodec` wheels that
# the serving layer's torch churn (2.8+cu128 -> 2.11+cu130) had ABI-broken. The fix the ticket
# demands is that they cannot end up installed AT ALL — enforced by pins, never by an instruction
# to uninstall them afterwards. These tests are what makes that claim checkable.
# --------------------------------------------------------------------------------------------

_OCR_ENVIRONMENTS = ("served-ocr-server", "served-ocr-client")
_ABI_BROKEN_WHEELS = ("flash-attn", "torchcodec")


@pytest.mark.parametrize("name", _OCR_ENVIRONMENTS)
def test_the_abi_broken_wheels_are_excluded_from_both_ocr_environments(name: str) -> None:
    """Both halves of the split exclude both wheels — 'it can't reach us' is not a mechanism."""
    spec = load_environment(_ENV_ROOT / name / "environment.yaml")
    excluded = {exclusion.package for exclusion in spec.excluded_distributions}
    assert set(_ABI_BROKEN_WHEELS) <= excluded, f"{name} excludes only {sorted(excluded)}"
    for exclusion in spec.excluded_distributions:
        assert exclusion.reason.strip(), exclusion.package


@pytest.mark.parametrize("name", _OCR_ENVIRONMENTS)
def test_an_exclusion_is_rendered_as_a_specifier_nothing_can_satisfy(name: str) -> None:
    """The mechanism, asserted as behaviour: no released version satisfies the rendered pin.

    A specifier that some version DOES satisfy would make the exclusion decorative — the resolver
    would happily install that version. The check is over the specifier itself rather than over
    the literal string ``<0``, so a future change of mechanism stays honest.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    spec = load_environment(_ENV_ROOT / name / "environment.yaml")
    rendered = render_constraints(spec, _REPO_ROOT)
    for exclusion in spec.excluded_distributions:
        assert exclusion.requirement in rendered.splitlines(), exclusion.package
        specifier = Requirement(exclusion.requirement).specifier
        for candidate in ("0", "0.0.1", "1.0", "2.8.3", "99999.0", "1.0.0rc1", "0.14"):
            assert not specifier.contains(Version(candidate), prereleases=True), (
                f"{exclusion.package}: version {candidate} satisfies '{exclusion.requirement}', "
                "so the exclusion does not exclude"
            )


def test_an_excluded_wheel_that_reached_the_lock_is_caught(sandbox: Path) -> None:
    """THE regression: the exclusion holds only because the lock is checked against it.

    Simulates a lock solved WITHOUT `-c constraints.txt` — the exact mistake that put an
    ABI-broken flash-attn on the pod. Before this check existed the environment verified clean
    with the excluded wheel sitting in its lockfile.
    """
    lock = sandbox / "deploy" / "env" / "served-ocr-server" / "requirements.lock"
    lock.write_text(
        lock.read_text(encoding="utf-8") + "flash-attn==2.8.3\n    # via vllm\n", encoding="utf-8"
    )
    assert "E-EXCLUSION-VIOLATED" in _codes(sandbox, "served-ocr-server")


def test_an_exclusion_contradicted_by_a_package_dependency_is_caught(sandbox: Path) -> None:
    """An environment that both excludes and requires a distribution cannot be installed.

    Shipping it would hand an adopter a resolve that fails on their machine and 'passed' on ours,
    because our own check never looked at whether the two claims could both hold.
    """
    pyproject = sandbox / "packages" / "latence-parser-lighton" / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace('"torch>=2.1,<3",', '"torch>=2.1,<3",\n    "flash-attn>=2.8",'),
        encoding="utf-8",
    )
    assert "E-EXCLUSION-CONTRADICTED" in _codes(sandbox, "served-ocr-client")


def test_an_extra_requirement_absent_from_the_lock_is_caught(sandbox: Path) -> None:
    """THE regression for a SERVICE environment, whose extras are its whole dependency set.

    ``extra_requirements`` were rendered into requirements.in but never checked against the lock,
    so ``served-ocr-server`` — which declares nothing else — had a vacuously green lock check.
    """
    lock = sandbox / "deploy" / "env" / "served-ocr-server" / "requirements.lock"
    kept = [
        line
        for line in lock.read_text(encoding="utf-8").splitlines()
        if not line.startswith("vllm==")
    ]
    lock.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert "E-LOCK-INCOMPLETE" in _codes(sandbox, "served-ocr-server")


def test_an_extra_requirement_the_lock_contradicts_is_caught(sandbox: Path) -> None:
    """The serving environment's identity IS its vLLM pin; a lock that drifts off it is a lie."""
    env = sandbox / "deploy" / "env" / "served-ocr-server"
    spec = env / "environment.yaml"
    spec.write_text(
        spec.read_text(encoding="utf-8").replace('"vllm==0.24.0"', '"vllm==0.25.1"'),
        encoding="utf-8",
    )
    codes = _codes(sandbox, "served-ocr-server")
    assert "E-LOCK-UNSATISFIED" in codes
    assert "E-REQUIREMENTS-STALE" in codes


# --------------------------------------------------------------------------------------------
# The two-environment split (issue #196) is structural, not prose.
# --------------------------------------------------------------------------------------------


def test_the_ocr_split_is_a_genuine_dependency_conflict() -> None:
    """The serving and client environments' transformers boundaries CANNOT both be satisfied.

    This is the whole justification for serving OCR behind the ADR-0030 endpoint seam rather than
    importing it. Asserted from the committed specs, so if somebody ever 'simplifies' the two
    environments back into one, this test states exactly why that cannot work — and if the
    conflict genuinely dissolves upstream, this test is the prompt to revisit the topology
    deliberately rather than discovering it on a pod.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    def transformers_specifier(name: str) -> object:
        spec = load_environment(_ENV_ROOT / name / "environment.yaml")
        for constraint in spec.constraints:
            requirement = Requirement(constraint.requirement)
            if requirement.name == "transformers":
                return requirement.specifier
        raise AssertionError(f"{name} declares no enforced transformers boundary")

    server = transformers_specifier("served-ocr-server")
    client = transformers_specifier("served-ocr-client")
    # Sweep the real release line either side of the 4/5 break. No version satisfies both.
    candidates = ["4.45.0", "4.51.3", "4.57.6", "4.99.0", "5.0.0", "5.14.1", "5.99.0"]
    both = [
        version
        for version in candidates
        if server.contains(Version(version), prereleases=True)  # type: ignore[attr-defined]
        and client.contains(Version(version), prereleases=True)  # type: ignore[attr-defined]
    ]
    assert not both, (
        f"transformers {both} satisfies BOTH the served-OCR server and client boundaries — the "
        "conflict that justifies the endpoint seam no longer exists as declared"
    )
    # ...and each side is satisfiable on its own, so the split is a real split and not two
    # environments that are both simply broken.
    assert server.contains(Version("5.14.1"), prereleases=True)  # type: ignore[attr-defined]
    assert client.contains(Version("4.57.6"), prereleases=True)  # type: ignore[attr-defined]


def test_the_client_never_installs_the_in_process_ocr_provider() -> None:
    """The served client must not promise ``parser.lighton`` — that would defeat the split.

    The package is installed (for its rasteriser), so nothing structurally stops a stack from
    naming the in-process 1B vision-language Provider. What the environment PROMISES is the
    Provider set, and loading a VLM into the CPU client is precisely what the served architecture
    exists to avoid.
    """
    spec = load_environment(_ENV_ROOT / "served-ocr-client" / "environment.yaml")
    assert "parser.lighton_vllm" in spec.providers
    assert "parser.lighton" not in spec.providers
    assert "packages/latence-parser-lighton" in spec.workspace_packages


def test_the_served_smoke_stack_cannot_be_satisfied_without_the_endpoint() -> None:
    """The client's smoke stack has exactly ONE Parse Provider, and it is the served one.

    A second, local Parse Provider (or a text extension on the Source) would let gate 3 go green
    without the endpoint ever being called — the false green this stack is shaped to prevent.
    """
    import yaml

    stack_path = _ENV_ROOT / "served-ocr-client" / "stack.yaml"
    raw = yaml.safe_load(stack_path.read_text(encoding="utf-8"))
    parse_stages = [stage for stage in raw["stages"] if stage["capability"] == "parse"]
    assert [stage["provider"] for stage in parse_stages] == ["parser.lighton_vllm"]
    assert parse_stages[0]["config"]["base_url"] == "{OCR_BASE_URL}", (
        "the endpoint must be a placeholder the smoke fills from the environment; a hard-coded "
        "localhost default would let the smoke pass against whatever is listening"
    )
    source = next(stage for stage in raw["stages"] if stage["capability"] == "source")
    text_extensions = {"txt", "md", "markdown"} & set(source["config"]["extensions"])
    assert not text_extensions, (
        f"the served-OCR smoke corpus accepts text extensions {sorted(text_extensions)}, so a "
        "text Provider could satisfy it without the OCR endpoint doing any work"
    )


# --------------------------------------------------------------------------------------------
# The serving image enforces the exclusions the way the ticket demands.
# --------------------------------------------------------------------------------------------


def test_the_serving_image_bakes_the_constraints_in_rather_than_uninstalling() -> None:
    """Pins, not "uninstall these afterwards" — the acceptance criterion, read off the Dockerfile.

    Three claims, each a real failure mode: the constraints must bind the install; they must
    OUTLIVE it (a later ``pip install`` in the running container is bound too); and there must be
    no uninstall step anywhere, because a step like that runs only when somebody remembers.
    """
    dockerfile = (_ENV_ROOT / "served-ocr-server" / "Dockerfile").read_text(encoding="utf-8")
    assert "--constraint /etc/latence/ocr-constraints.txt" in dockerfile
    assert "PIP_CONSTRAINT=/etc/latence/ocr-constraints.txt" in dockerfile
    assert "UV_CONSTRAINT=/etc/latence/ocr-constraints.txt" in dockerfile
    lowered = dockerfile.lower()
    for forbidden in ("pip uninstall", "uv pip uninstall"):
        assert forbidden not in lowered, (
            f"the serving image contains '{forbidden}': an exclusion enforced by removing the "
            "wheel after installing it is exactly what this environment must not do"
        )
    # A mutable tag is not a reproducible base. The digest is the artifact.
    base = next(line for line in dockerfile.splitlines() if line.startswith("FROM "))
    assert "@sha256:" in base, f"base image is not digest-pinned: {base}"
    assert ":latest" not in base


def test_the_image_side_exclusion_check_reads_the_generated_constraints() -> None:
    """The in-image check derives its list from constraints.txt, so the two cannot drift.

    Drive the real parser over the real generated file: a change to how exclusions are rendered
    that the script stopped recognising would silently turn the image's build-time proof into a
    check of nothing.
    """
    import importlib.util

    script = _ENV_ROOT / "served-ocr-server" / "verify-exclusions.py"
    module_spec = importlib.util.spec_from_file_location("_verify_exclusions", script)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    constraints = (_ENV_ROOT / "served-ocr-server" / "constraints.txt").read_text(encoding="utf-8")
    found = module.excluded_distributions(constraints)
    assert set(_ABI_BROKEN_WHEELS) <= set(found), found
    # And it must not mistake an ENFORCED BOUNDARY for an exclusion — that would make the image
    # refuse to build over a distribution it is supposed to install.
    assert "transformers" not in found
    assert "torch" not in found


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
        "  - name: parse\n    capability: parse\n    provider: parser.render\n"
        "    config:\n"
        "      ocr: {provider: parser.lighton}\n",
        encoding="utf-8",
    )
    assert stack_providers(stack) == {
        "export.jsonl_parquet",
        "embedding.sentence_transformers",
        "parser.render",
        "parser.lighton",
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
            '"peft>=0.10,<1",', '"peft>=0.10",\n    "transformers>=5.0",'
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
            '"peft>=0.10,<1",', '"peft>=0.10",\n    "transformers>=4.52,<4.99",'
        ),
        encoding="utf-8",
    )
    assert "E-CONSTRAINT-CONFLICT" not in _codes(sandbox, _GPU_ENV)


# --------------------------------------------------------------------------------------------
# The gliner2 MAJOR boundary (ADR-0064) — the second enforced conflict, and the one it exists for
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", ["cpu-text", _GPU_ENV])
def test_every_gliner2_environment_enforces_the_major_boundary(
    sandbox: Path, env_name: str
) -> None:
    """An environment that installs a 1.x gliner2 Provider must pin the major, not hope.

    Parametrised over both, because the failure this prevents is asymmetric: it is the environment
    that FORGOT the boundary that silently resolves a 2.x.
    """
    spec = load_environment(_spec_in(sandbox, env_name))
    windows = [c.requirement for c in spec.constraints if c.requirement.startswith("gliner2")]
    assert windows == ["gliner2>=1.3,<2"]


@pytest.mark.parametrize("env_name", ["cpu-text", _GPU_ENV])
def test_the_gliner_2_5_package_cannot_resolve_into_a_gliner2_1x_environment(
    sandbox: Path, env_name: str
) -> None:
    """THE architectural crux of ADR-0064, as a machine-checked refusal rather than a warning.

    ``latence-gliner25`` declares ``gliner2[local]>=2.0,<3``; these environments enforce
    ``gliner2>=1.3,<2``. The windows are DISJOINT, so adding the package is refused here — offline,
    import-free, lock-independent, before anything installs — instead of resolving to something and
    exploding at ``AutoExtractor.from_pretrained`` on a pod.

    The finding is asserted by CODE and the offending requirement by NAME, so a future edit that
    widened either pin would turn this red rather than quietly making the two co-installable.
    """
    spec_path = _spec_in(sandbox, env_name)
    spec_path.write_text(
        spec_path.read_text(encoding="utf-8").replace(
            "  - packages/latence-extract-gliner2\n",
            "  - packages/latence-extract-gliner2\n  - packages/latence-gliner25\n",
        ),
        encoding="utf-8",
    )
    report = verify_environment(load_environment(spec_path), sandbox)
    conflicts = [f for f in report.findings if f.code == "E-CONSTRAINT-CONFLICT"]
    assert conflicts, {f.code for f in report.findings}
    assert any("gliner2[local]>=2.0,<3" in f.message for f in conflicts)
    assert any("latence-gliner25" in f.message for f in conflicts)


def test_the_1x_packages_are_not_reported_as_conflicting_with_their_own_boundary(
    sandbox: Path,
) -> None:
    """The negative control: the boundary must refuse the 2.x package and NOTHING else.

    A boundary that also flags the packages it was written for is a boundary somebody deletes.
    """
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


# --------------------------------------------------------------------------------------------
# R3-seams_real_not_hypothetical-0 — the two walkers in this module must not disagree.
# --------------------------------------------------------------------------------------------

_COMMITTED_STACKS = sorted((_REPO_ROOT / "stacks").glob("*.yaml")) + sorted(
    _ENV_ROOT.glob("*/stack.yaml")
)


@pytest.mark.parametrize(
    "stack_path",
    _COMMITTED_STACKS,
    ids=[str(p.relative_to(_REPO_ROOT)) for p in _COMMITTED_STACKS],
)
def test_the_two_stack_walkers_agree_on_every_committed_stack(stack_path: Path) -> None:
    """``stack_components`` must find every Provider the structural ``stack_providers`` walk finds.

    They answer the same question about the same file — one returns names, the other returns names
    WITH the capability and config the placement probe needs to drive them — so a Provider visible
    to one and invisible to the other is a hole in whichever gate reads the poorer walker.

    It was a real hole: ``stack_components`` used a hard-coded 3-key tuple
    (``embedder``/``sparse_embedder``/``multivector_embedder``) while the framework declares SEVEN
    nested slots, so ``parser.lighton`` (the GPU-gated OCR of ``universal-parse``),
    ``fde.reference`` and ``graph_features.degree`` were invisible to the #191 placement gate and
    to the env-stack constructibility gate.
    """
    from latence_core.envspec import stack_components

    components = {c.provider for c in stack_components(stack_path)}
    structural = stack_providers(stack_path)
    assert components == structural, (
        f"{stack_path.relative_to(_REPO_ROOT)}: the walkers disagree — "
        f"stack_components misses {sorted(structural - components)}, "
        f"invents {sorted(components - structural)}"
    )


def test_stack_components_finds_the_nested_ocr_sub_provider() -> None:
    """The universal-parse stack's OCR model is nested under ``parse.config.ocr``.

    ``subprovider.py`` calls it "the GPU-gated node of that stack"; before this fix no static
    consumer saw it, so ``latence env placement`` could report ok=True for a stack shaped like this
    without ever driving a forward pass through the one model on the GPU.
    """
    from latence_core.envspec import stack_components

    stack = _REPO_ROOT / "stacks" / "universal-parse.yaml"
    by_provider = {c.provider: c for c in stack_components(stack)}
    ocr = by_provider["parser.lighton"]
    assert ocr.capability == "ocr"  # the nesting key doubles as the driver-selection capability
    assert ocr.config["device"] == "cuda"


def test_stack_component_order_is_deterministic() -> None:
    """The walk reads a ``frozenset`` of keys; order must come from the FILE, not from hashing."""
    from latence_core.envspec import stack_components

    stack = _REPO_ROOT / "stacks" / "gpu-sota.yaml"
    first = [(c.capability, c.provider) for c in stack_components(stack)]
    for _ in range(5):
        assert [(c.capability, c.provider) for c in stack_components(stack)] == first
