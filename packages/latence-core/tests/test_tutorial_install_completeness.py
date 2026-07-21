"""The go-live tutorial may not promise a stack it does not install (R1-wizard_profile-17).

``docs/TUTORIAL-fresh-pod-walkthrough.md`` is the one document an operator runs *verbatim* on a
fresh GPU pod, and its §3 fast path is ``latence setup --profile enterprise-sota`` →
``latence process``. That makes the §1.2 editable-install block a machine-checkable claim: every
Provider the wizard emits for that profile must be *registered* by a distribution §1.2 actually
installs, because an unregistered ``provider:`` aborts the run at that Stage with a bare
``KeyError`` from the registry.

#197 broke exactly that. It blessed ``redaction.gliner2`` in BOTH wizard profiles, but the entry
point lives solely in ``packages/latence-pii-gliner2`` while §1.2 installs only
``packages/latence-pii-gliner`` (which registers the *twin*, ``redaction.gliner_pii``). The prose
even claimed the blessed redactor "needs no package beyond the ``gliner2`` above", conflating the
third-party PyPI wheel with the Provider distribution. On ``main`` the doc was self-consistent, so
the regression was invisible to a fully green suite — nothing else in the tree reads the tutorial.

Both guards are derived from the repository — the install block is parsed out of the markdown, the
Provider→distribution map out of every ``packages/*/pyproject.toml``, and the Provider set out of
the real wizard emission. Nothing here restates a literal, so re-blessing a different Provider
without updating the tutorial fails again.

Scope note: the guards cover the ``enterprise-sota`` profile only. §1.2 says so in its own heading
("the packages the enterprise-SOTA stack uses"), and the ``lite`` profile deliberately composes
different Providers (``parser.render``, ``disambiguation.embedding``) that this pod walkthrough
does not claim to install.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from latence_core.pipeline import Pipeline
from latence_core.setup.wizard import (
    PARSER_OCR_VLLM,
    PROFILE_SOTA,
    SetupAnswers,
    build_stack_config,
)

# tests/  ->  latence-core/  ->  packages/  ->  repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = _REPO_ROOT / "packages"
_TUTORIAL = _REPO_ROOT / "docs" / "TUTORIAL-fresh-pod-walkthrough.md"

#: The distribution whose Providers are always present — it is the package the CLI itself ships in.
_CORE = "packages/latence-core"

#: An editable install line in the tutorial's §1.2 block: ``-e packages/latence-<something>``.
_EDITABLE = re.compile(r"-e\s+(packages/[a-z0-9][a-z0-9\-]*)")

#: The §1.4 probe iterates a tuple of double-quoted ``group.name`` Provider specs.
_PROBE_NAME = re.compile(r'"([a-z_]+\.[a-z_0-9]+)"')


def _tutorial_text() -> str:
    assert _TUTORIAL.is_file(), f"the walkthrough this gate polices is missing: {_TUTORIAL}"
    return _TUTORIAL.read_text(encoding="utf-8")


def _installed_distributions() -> frozenset[str]:
    """Repo-relative package paths the tutorial tells the operator to ``pip install -e``."""
    found = frozenset(_EDITABLE.findall(_tutorial_text()))
    assert _CORE in found, "§1.2 must install latence-core — the parse found no editable block"
    for path in found:
        assert (_REPO_ROOT / path).is_dir(), f"§1.2 installs a package that does not exist: {path}"
    return found


def _provider_owner() -> dict[str, str]:
    """Map every ``latence.providers`` entry-point name to the package that registers it."""
    owner: dict[str, str] = {}
    for pyproject in sorted(_PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        groups = data.get("project", {}).get("entry-points", {})
        for name in groups.get("latence.providers", {}):
            owner[name] = f"packages/{pyproject.parent.name}"
    assert "redaction.gliner2" in owner, "no package registers the blessed redactor at all"
    return owner


def _sota_providers(tmp_path: Path) -> list[str]:
    """Every ``provider:`` in the config `latence setup --profile enterprise-sota` emits."""
    source = tmp_path / "corpus"
    source.mkdir(exist_ok=True)
    (source / "a.pdf").write_text("a", encoding="utf-8")
    answers = SetupAnswers(
        # The walkthrough's §3 fast path is the served-vLLM OCR parse path for a scanned/PDF corpus
        # (parser.lighton_vllm) — that is the parser this install-completeness gate must cover. (A
        # born-digital text corpus instead routes to parser.plaintext; see recommend_sota_parser.)
        source=f"file://{source}",
        storage=f"file://{tmp_path}/out",
        parser=PARSER_OCR_VLLM,
        device="auto",
        pii=True,
        profile=PROFILE_SOTA,
        extensions=("pdf", "txt"),
    )
    pipeline = Pipeline.from_yaml_text(build_stack_config(answers))
    return [stage.provider for stage in pipeline.stages]


def _uninstalled(specs: list[str]) -> list[str]:
    """The subset of ``specs`` whose registering distribution §1.2 never installs."""
    installed = _installed_distributions()
    owner = _provider_owner()
    return [
        f"{spec} (registered by {owner[spec]})"
        for spec in specs
        # A spec with no owning package is built into core's own registry, not a plugin.
        if spec in owner and owner[spec] not in installed
    ]


# ---------------------------------------------------------------------------
# AC1 — the documented fast path's own Providers are all installed by §1.2
# ---------------------------------------------------------------------------


def test_tutorial_installs_every_provider_the_sota_wizard_emits(tmp_path: Path) -> None:
    missing = _uninstalled(_sota_providers(tmp_path))
    assert not missing, (
        "§3 tells the operator to run `latence setup --profile enterprise-sota` then "
        "`latence process`, but §1.2 never installs the package(s) registering: "
        + ", ".join(missing)
        + " — the run aborts at that Stage with a bare registry KeyError."
    )


def test_blessed_redactor_is_installed_by_the_tutorial(tmp_path: Path) -> None:
    """The #197 regression itself, pinned at the Stage that broke."""
    redact = next(
        stage
        for stage in Pipeline.from_yaml_text(
            build_stack_config(
                SetupAnswers(
                    source=f"file://{tmp_path}/corpus",
                    storage=f"file://{tmp_path}/out",
                    parser="parser.render",
                    pii=True,
                    profile=PROFILE_SOTA,
                    extensions=("pdf",),
                )
            )
        ).stages
        if stage.name == "redact"
    )
    assert not _uninstalled([redact.provider]), (
        f"the wizard blesses {redact.provider!r} but §1.2 does not install the distribution that "
        "registers it (the PyPI wheel `gliner2` is NOT the Provider package)"
    )


# ---------------------------------------------------------------------------
# AC2 — §1.4's probe cannot print MISSING for a package §1.2 never lists
# ---------------------------------------------------------------------------


def test_verify_probe_names_only_providers_the_tutorial_installs() -> None:
    text = _tutorial_text()
    assert "### 1.4" in text, "the verify section this gate reads has moved or been renamed"
    probe = text.split("### 1.4", 1)[1].split("### ", 1)[0]
    names = [n for n in dict.fromkeys(_PROBE_NAME.findall(probe)) if n in _provider_owner()]
    assert len(names) >= 5, f"failed to parse the §1.4 probe's Provider list (got {names})"

    missing = _uninstalled(names)
    assert not missing, (
        "§1.4 tells the reader a MISSING Provider means 'its package wasn't installed in §1.2', "
        "but §1.2 lists no package providing: " + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# AC3 — the guard is derived, not restated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["redaction.gliner2", "redaction.gliner_pii"])
def test_provider_owner_map_is_read_from_the_packages_tree(spec: str) -> None:
    owner = _provider_owner()
    pyproject = _REPO_ROOT / owner[spec] / "pyproject.toml"
    entry_points = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["entry-points"]
    assert spec in entry_points["latence.providers"]
