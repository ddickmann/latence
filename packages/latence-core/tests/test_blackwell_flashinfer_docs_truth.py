"""R2-docs_truth-8 — the deployment docs must carry FM-FLASHINFER-SM120's RECORDED resolution.

``deploy/env/compatibility-matrix.yaml`` § ``FM-FLASHINFER-SM120`` is a ``verified`` field report: on
a Blackwell (``sm_120``) pod, vLLM aborted with ``FlashInfer requires sm75+`` because FlashInfer's
capability check misparses ``12.0`` as ``12``. Its recorded resolution is
``VLLM_USE_FLASHINFER_SAMPLER=0`` — it disables only the *sampler*, so cudagraphs stay ON — and the
record says explicitly that this is **deliberately NOT** ``--enforce-eager``, which would floor the
throughput the served path exists for.

Every operator-facing deployment surface used to say the opposite: add ``--enforce-eager``. An
operator on the Helm path (Option B) followed ``docs/deployment.md`` § "GPU requirements + the
Blackwell note" into the ``vllmOcr.extraArgs`` row, set the flag, and ran the served path
permanently without cudagraphs — while the one-line fix appeared in no deployment doc and could not
be expressed through the chart at all.

Anti-tautology: nothing here is hard-coded. :func:`_recorded` reads the failure mode out of the
committed matrix YAML and derives, from its own prose, (a) the resolution's env assignment, (b) the
remedy the record REFUTES, and (c) the failure signature an operator greps for. If the matrix's
recorded resolution ever changes, these assertions change with it — and the docs must follow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MATRIX_PATH = _REPO_ROOT / "deploy" / "env" / "compatibility-matrix.yaml"
_FAILURE_MODE_ID = "FM-FLASHINFER-SM120"

#: The surfaces an operator standing up a deployment actually reads. Not the ADRs (history) and not
#: ``docs/compatibility-matrix.md`` (generated from the YAML this test derives its truth from).
_OPERATOR_SURFACES = (
    "docs/deployment.md",
    "docs/serving-lighton-vllm.md",
    "docs/TUTORIAL-fresh-pod-walkthrough.md",
    "deploy/docker-compose.yml",
    "deploy/helm/latence/values.yaml",
)

_HELM_VALUES = "deploy/helm/latence/values.yaml"
_HELM_OCR_TEMPLATE = "deploy/helm/latence/templates/vllm-ocr.yaml"

#: Verbs that turn a mention of a flag into a RECOMMENDATION of it.
_PRESCRIPTIVE = re.compile(r"\b(add|adds|adding|set|sets|use|uses|pass|append|specify)\b", re.I)
#: Markers that make the same sentence a WARNING against it instead.
_NEGATED = re.compile(
    r"(\bnot\b|\bnever\b|\bdo not\b|\bdon't\b|\brather than\b|\binstead of\b|\bno\s+`?--|"
    r"\bwithout\b|\bsuperseded\b)",
    re.I,
)


def _recorded() -> dict[str, str]:
    """Derive the required doc content from the committed matrix — never from constants here."""
    raw = yaml.safe_load(_MATRIX_PATH.read_text(encoding="utf-8"))
    modes = [m for m in raw["failure_modes"] if m["id"] == _FAILURE_MODE_ID]
    assert modes, f"{_FAILURE_MODE_ID} is gone from the matrix — this guard has lost its subject"
    mode = modes[0]

    resolution = " ".join(mode["resolution"].split())
    env_assignments = re.findall(r"\b([A-Z][A-Z0-9_]{4,})=(\S+?)(?=[\s.,]|$)", resolution)
    assert len(env_assignments) == 1, (
        f"expected exactly one env assignment in {_FAILURE_MODE_ID}'s resolution, "
        f"got {env_assignments!r}"
    )
    name, value = env_assignments[0]

    refuted = re.findall(r"\bNOT\s+(--[a-z][a-z0-9-]*)", resolution)
    assert len(refuted) == 1, (
        f"expected exactly one refuted flag in {_FAILURE_MODE_ID}'s resolution, got {refuted!r}"
    )

    symptom = " ".join(mode["symptom"].split())
    signature = re.search(r"\b(sm\d{2,3})\+", symptom)
    assert signature, f"no failure signature found in {_FAILURE_MODE_ID}'s symptom: {symptom!r}"

    return {
        "env": f"{name}={value}",
        "env_name": name,
        "refuted": refuted[0],
        "signature": signature.group(1),
    }


@pytest.fixture(scope="module")
def recorded() -> dict[str, str]:
    return _recorded()


def _read(relative: str) -> str:
    path = _REPO_ROOT / relative
    assert path.is_file(), f"{relative} is missing — the doc-truth guard points at nothing"
    return path.read_text(encoding="utf-8")


def _surfaces_naming_signature(signature: str) -> list[str]:
    return [rel for rel in _OPERATOR_SURFACES if signature in _read(rel)]


def test_the_signature_is_actually_documented_somewhere(recorded: dict[str, str]) -> None:
    """Guard the guard: if no surface mentions the failure, the rules below are vacuous."""
    naming = _surfaces_naming_signature(recorded["signature"])
    assert naming, (
        f"no operator-facing surface mentions {recorded['signature']!r} — either the docs stopped "
        "routing Blackwell operators anywhere, or this guard is checking nothing"
    )


@pytest.mark.parametrize("relative", _OPERATOR_SURFACES)
def test_surface_naming_the_failure_also_names_the_recorded_fix(
    relative: str, recorded: dict[str, str]
) -> None:
    """A page that tells an operator about the abort must tell them the resolution on record."""
    text = _read(relative)
    if recorded["signature"] not in text:
        pytest.skip(f"{relative} does not mention {recorded['signature']}")
    assert recorded["env"] in text, (
        f"{relative} routes a Blackwell operator at the {recorded['signature']!r} abort but never "
        f"names {_FAILURE_MODE_ID}'s recorded resolution {recorded['env']!r} — the one setting that "
        "fixes the abort while keeping cudagraphs on"
    )


@pytest.mark.parametrize("relative", _OPERATOR_SURFACES)
def test_no_surface_prescribes_the_refuted_remedy(
    relative: str, recorded: dict[str, str]
) -> None:
    """``--enforce-eager`` may be named only to warn against it, never to prescribe it."""
    refuted = recorded["refuted"]
    offenders = [
        (number, line.strip())
        for number, line in enumerate(_read(relative).splitlines(), start=1)
        if refuted in line and _PRESCRIPTIVE.search(line) and not _NEGATED.search(line)
    ]
    assert not offenders, (
        f"{relative} prescribes {refuted!r}, which {_FAILURE_MODE_ID} records as deliberately NOT "
        f"the fix (it floors throughput). Offending lines: {offenders!r}"
    )


def test_the_helm_chart_can_express_the_recorded_fix(recorded: dict[str, str]) -> None:
    """The resolution is an ENV VAR: the Option B path must be able to set one without a fork."""
    values = yaml.safe_load(_read(_HELM_VALUES))
    extra_env = values["vllmOcr"].get("extraEnv")
    assert extra_env is not None, (
        f"{_HELM_VALUES} has no `vllmOcr.extraEnv`, so {recorded['env']!r} — the recorded fix for "
        f"{_FAILURE_MODE_ID} — is unsettable on the Helm path without editing chart templates"
    )
    assert isinstance(extra_env, list), f"vllmOcr.extraEnv must be a list, got {type(extra_env)}"
    assert extra_env == [], "vllmOcr.extraEnv must default to empty (off-by-default is a no-op)"

    template = _read(_HELM_OCR_TEMPLATE)
    assert ".Values.vllmOcr.extraEnv" in template, (
        f"{_HELM_OCR_TEMPLATE} never renders `vllmOcr.extraEnv` — the value would be inert"
    )
    assert recorded["env_name"] in _read(_HELM_VALUES), (
        f"{_HELM_VALUES} never names {recorded['env_name']} — an operator hitting the "
        f"{recorded['signature']!r} abort has no worked example of the knob that fixes it"
    )


def test_the_matrix_lists_the_deployment_surfaces_that_carry_the_fix(
    recorded: dict[str, str]
) -> None:
    """`enforced_by` must reach the operator surfaces, so deleting one fails `latence env matrix`."""
    raw = yaml.safe_load(_MATRIX_PATH.read_text(encoding="utf-8"))
    mode = next(m for m in raw["failure_modes"] if m["id"] == _FAILURE_MODE_ID)
    enforced_by = set(mode["enforced_by"])
    carriers = {
        rel
        for rel in _OPERATOR_SURFACES
        if recorded["env"] in _read(rel)
    }
    missing = carriers - enforced_by
    assert not missing, (
        f"{sorted(missing)} carry {recorded['env']!r} but are not in {_FAILURE_MODE_ID}'s "
        "`enforced_by`, so the matrix verifier would not notice them being deleted"
    )
