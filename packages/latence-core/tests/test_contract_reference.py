"""Drift gate for the generated Stage-contract reference (the agent-facing discovery format).

The committed ``docs/reference/stage-contracts.{json,md}`` must be **byte-identical** to a fresh
render from the live sources — the same regenerate-and-compare pattern the repo's other generated
documents use. With this gate the reference can never lie: a ninth descriptor fact, a new
Capability, a changed carrier field, a new entry point — any of them turns CI red until
``PYTHONHASHSEED=0 .venv/bin/latence contracts --write --repo-root .`` is re-run.

Beyond byte-identity the gate pins the coverage claims the discovery format makes:

* every :class:`~latence_core.pipeline.CapabilityKind` appears in the reference;
* every ``latence.providers`` entry point declared across ``packages/*/pyproject.toml`` is listed
  under exactly one documented Capability/seam (an unclassifiable name refuses the render);
* every conformance case the reference cites exists in ``conformance/cases.py``, and the
  signal-seam prefix→case convention agrees with :func:`~latence_core.conformance.cases.
  capability_of` — the map the suite actually routes Providers with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from latence_core.contract_reference import (
    REGENERATE_COMMAND,
    build_reference,
    render_json,
    render_markdown,
    render_reference_files,
    workspace_provider_packages,
)
from latence_core.errors import ContractError
from latence_core.pipeline import CapabilityKind

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _committed_and_fresh() -> list[tuple[Path, str]]:
    """Each committed artifact path paired with its freshly rendered content."""
    return sorted(render_reference_files(_REPO_ROOT).items())


def test_committed_artifacts_are_byte_identical_to_a_fresh_render() -> None:
    """The core gate: the committed reference IS a render of the live sources, byte for byte."""
    for path, fresh in _committed_and_fresh():
        assert path.is_file(), (
            f"{path.relative_to(_REPO_ROOT)} is missing; regenerate with `{REGENERATE_COMMAND}`"
        )
        committed = path.read_text(encoding="utf-8")
        assert committed == fresh, (
            f"{path.relative_to(_REPO_ROOT)} is stale against the descriptor table / Protocols / "
            f"carriers / workspace entry points; regenerate with `{REGENERATE_COMMAND}`"
        )


def test_render_is_deterministic() -> None:
    """Two builds + renders in one process are identical — the byte-identity gate's precondition."""
    packages = workspace_provider_packages(_REPO_ROOT)
    first = build_reference(packages)
    second = build_reference(packages)
    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)


def test_every_capability_kind_appears_exactly_once() -> None:
    """A new CapabilityKind cannot ship undocumented: the reference covers the enum exactly."""
    reference = build_reference(workspace_provider_packages(_REPO_ROOT))
    kinds = [entry["kind"] for entry in reference["capabilities"]]
    assert kinds == [kind.value for kind in CapabilityKind]


def test_every_workspace_entry_point_is_documented() -> None:
    """Every declared Provider name appears under exactly one Capability/seam entry."""
    packages = workspace_provider_packages(_REPO_ROOT)
    assert packages, "no latence.providers entry points discovered in packages/*/pyproject.toml"
    reference = build_reference(packages)
    listed: list[str] = []
    for entry in [*reference["capabilities"], *reference["signal_seams"]]:
        listed += [provider["name"] for provider in entry["providers"]]
    assert sorted(listed) == sorted(packages)
    assert len(listed) == len(set(listed)), "an entry point is listed under two seams"


def test_an_unclassifiable_entry_point_refuses_the_render() -> None:
    """The reference fails loudly rather than silently omitting an undocumented seam."""
    with pytest.raises(ContractError, match="bogus.thing"):
        build_reference({"bogus.thing": "some-package"})


def test_every_cited_conformance_case_exists_and_routes_the_same() -> None:
    """Cited case ids exist in conformance/cases.py, and prefix routing agrees with the suite."""
    from latence_core.conformance.cases import Capability, capability_of

    reference = build_reference(workspace_provider_packages(_REPO_ROOT))
    known_cases = {case.value for case in Capability}
    for entry in reference["capabilities"]:
        assert entry["conformance_case"] in known_cases, entry["kind"]
        # The suite must route a Provider of this seam to exactly the cited case. A dotted
        # prefix (screening.intake) is exercised with its discriminating second segment.
        for prefix in entry["entry_point"]["prefixes"]:
            probe = f"{prefix}_probe" if "." in prefix else f"{prefix}.probe"
            assert capability_of(probe).value == entry["conformance_case"], probe
    for seam in reference["signal_seams"]:
        assert seam["conformance_case"] in known_cases, seam["prefix"]
        assert capability_of(f"{seam['prefix']}.probe").value == seam["conformance_case"]


def test_committed_json_declares_the_schema_version() -> None:
    """A consumer keys off the schema field; the committed artifact carries it."""
    committed = json.loads(
        (_REPO_ROOT / "docs" / "reference" / "stage-contracts.json").read_text(encoding="utf-8")
    )
    assert committed["schema"] == 1
    assert committed["entry_point_group"] == "latence.providers"
