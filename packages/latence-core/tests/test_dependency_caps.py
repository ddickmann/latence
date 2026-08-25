"""The H-G1 dependency cap, ENFORCED rather than commented (ADR-0065).

Eleven package manifests carried a prose comment explaining that a third-party pin has a
major-boundary upper cap "so a downstream install can't silently pull an API-breaking 2.x
release" — and 42 pins across 14 packages had no upper bound at all, because the convention
lived only in comments and nothing read them. A convention no gate reads is a convention that is already
half-abandoned: ``pypdfium2>=4.0`` had silently resolved to a 5.x, ``neo4j>=5.0`` to a 6.x,
``numpy>=1.24`` to a 2.x and ``apache-airflow>=2.7`` to a 3.x, each one exactly the crossing the
comment says the cap exists to prevent.

The rule this module enforces (ADR-0065): **every third-party requirement declared by a package
under ``packages/`` carries an upper bound** — in ``[project].dependencies`` and in every
``[project.optional-dependencies]`` extra alike, because ``pip install latence-retrieval[graph-neo4j]``
is a downstream install with exactly the same exposure — *unless* the pair
``(package, distribution)`` appears in :data:`EXEMPT` with a reason.

Default-cap-with-named-exemptions is deliberate, and it is the whole point. The alternative
considered and rejected was a gate that knows a list of fast-moving model libraries
(``gliner``/``transformers``/``torch``/…) and checks only those: it passes vacuously on the next
heavy dependency someone adds, which is precisely the failure mode that produced this ticket. A
default-cap gate fails closed — a new uncapped pin is red until somebody either caps it or writes
down why it cannot be capped.

First-party ``latence-*`` requirements are out of scope: they are workspace members resolved from
source, not a distribution a resolver picks a major for.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import NamedTuple

import pytest
from _workspace_providers import workspace_root

#: Specifier operators that bound a requirement from above. ``==``/``===`` are exact pins, which
#: bound it more tightly than a cap does; ``~=`` is a compatible-release clause, which implies an
#: upper bound on its own (``~=1.4.2`` means ``>=1.4.2,<1.5``).
_UPPER_BOUND_OPERATORS = frozenset({"<", "<=", "==", "===", "~="})

#: ``(package directory, canonical distribution name) -> why this one stays uncapped``.
#:
#: Every entry is a claim about the distribution that a reviewer can check, not a preference. The
#: list is gated both ways: an uncapped pin missing from here fails, and an entry here whose pin is
#: capped (or gone) ALSO fails, so the exemptions cannot outlive their reason.
EXEMPT: dict[tuple[str, str], str] = {
    ("latence-core", "fsspec"): (
        "CalVer. fsspec versions are dates (2026.6.0 in uv.lock, against a >=2024.2 floor); its "
        "leading component is a year, not an API generation, so a '<N' bound would encode a "
        "calendar date and would have to be bumped every January to mean nothing."
    ),
    ("latence-core", "pyarrow"): (
        "A time-based release train, and the workspace has the evidence: the >=15.0 floor has "
        "spanned majors 15 -> 24 (uv.lock) -> 25 (deploy/env/*/requirements.lock) with no source "
        "change in latence-core, which touches only the stable Table/Parquet surface. Capping it "
        "in the near-zero-dep base package (ADR-0016) that every adopter installs alongside their "
        "own Arrow-using stack buys a boundary nobody has seen break and costs a resolution "
        "conflict in every such environment."
    ),
    ("latence-core", "packaging"): (
        "The same shape (>=23.0 floor, 26.2 in uv.lock) with a sharper cost: `packaging` is the "
        "most ubiquitous build-time dependency in the ecosystem, so a cap in a published library "
        "conflicts with pip/build/setuptools rather than protecting anything. latence-core imports "
        "it lazily, from one module, through Requirement/Version/SpecifierSet only."
    ),
    ("latence-parser-lighton", "transformers"): (
        "H-G1's own escape clause: a cap here would be guesswork, so it is noted and skipped. The "
        "compatible major is NOT settled — deploy/env/gpu-learned/environment.yaml records that "
        "the >=5 floor for this VLM comes from pod field evidence rather than verified package "
        "metadata, while served-ocr-client resolves it next to transformers 4.57.6. '<5' would "
        "contradict the field evidence and '<6' would invent a boundary nobody has tested. That "
        "spec's `excluded_packages` block ALSO names this package's uncapped pin three times as "
        "the reason the refusal has to be by name ('the incompatibility lives in the model code, "
        "not the package metadata'), so a cap here would falsify a recorded reason as well as "
        "guess. Resolution-time safety is delivered by that named refusal, not by version math."
    ),
    ("latence-parser-glm", "transformers"): (
        "Same unsettled split, same named refusal in deploy/env/gpu-learned/environment.yaml "
        "('It declares an uncapped transformers>=4.45'). A second in-process VLM Parser on the "
        "recent-transformers path with no environment that pins its major; see the "
        "latence-parser-lighton entry."
    ),
}


class Pin(NamedTuple):
    """One third-party requirement, and where it is declared."""

    package: str
    distribution: str
    requirement: str
    #: ``"dependencies"`` or ``"optional-dependencies[<extra>]"`` — named in failure messages so
    #: the reader does not have to search a 90-line manifest for the offending string.
    section: str

    def __str__(self) -> str:
        return f"{self.package}: {self.section}: {self.requirement!r}"


def _canonical(name: str) -> str:
    """PEP 503 normalisation, so ``python_dotenv`` and ``python-dotenv`` are one key."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared_pins(root: Path) -> list[Pin]:
    """Every third-party requirement declared by every package manifest under ``root``.

    Refuses an empty scan rather than reporting "nothing to check" — a gate that passes because it
    found no manifests is the vacuous-green failure this module exists to prevent.
    """
    from packaging.requirements import Requirement

    manifests = sorted((root / "packages").glob("*/pyproject.toml"))
    if not manifests:
        msg = (
            f"no package manifests under {root / 'packages'} — refusing to report zero uncapped "
            f"pins, which would let the dependency-cap gate pass vacuously."
        )
        raise RuntimeError(msg)
    pins: list[Pin] = []
    for manifest in manifests:
        package = manifest.parent.name
        project = tomllib.loads(manifest.read_text(encoding="utf-8")).get("project", {})
        sections: list[tuple[str, list[str]]] = [
            ("dependencies", project.get("dependencies", []))
        ]
        for extra, requirements in project.get("optional-dependencies", {}).items():
            sections.append((f"optional-dependencies[{extra}]", requirements))
        for section, requirements in sections:
            for text in requirements:
                distribution = _canonical(Requirement(text).name)
                if distribution.startswith("latence-"):
                    continue  # a workspace member, not a resolved third-party distribution
                pins.append(Pin(package, distribution, text, section))
    if not pins:
        msg = (
            f"scanned {len(manifests)} manifests under {root / 'packages'} and found no "
            f"third-party requirements at all — that is a broken scan, not a clean tree."
        )
        raise RuntimeError(msg)
    return pins


def uncapped(pins: list[Pin]) -> list[Pin]:
    """The pins with no upper bound — the gate's finding, factored out so it can be tested."""
    from packaging.requirements import Requirement

    return [
        pin
        for pin in pins
        if not any(
            clause.operator in _UPPER_BOUND_OPERATORS
            for clause in Requirement(pin.requirement).specifier
        )
    ]


@pytest.fixture(scope="module")
def pins() -> list[Pin]:
    return declared_pins(workspace_root())


def test_every_third_party_requirement_carries_an_upper_bound(pins: list[Pin]) -> None:
    """The H-G1 rule itself, over every manifest — dependencies and extras alike."""
    offenders = [pin for pin in uncapped(pins) if (pin.package, pin.distribution) not in EXEMPT]
    assert not offenders, (
        "uncapped third-party pins (ADR-0065: every third-party requirement carries an upper "
        "bound). Add the major boundary the distribution's own release history supports, or add "
        "an EXEMPT entry in this module saying why it cannot be capped:\n  "
        + "\n  ".join(str(pin) for pin in sorted(offenders))
    )


def test_no_exemption_outlives_its_reason(pins: list[Pin]) -> None:
    """An EXEMPT entry whose pin is now capped, or gone, is a stale claim — so it fails.

    Without this the list only ever grows: an exemption written for a pin that someone later
    capped (or deleted) would sit there forever, still asserting that the cap is impossible.
    """
    still_uncapped = {(pin.package, pin.distribution) for pin in uncapped(pins)}
    stale = sorted(key for key in EXEMPT if key not in still_uncapped)
    assert not stale, (
        "EXEMPT entries that no longer describe an uncapped pin — the pin was capped, moved or "
        "removed, so delete the exemption:\n  "
        + "\n  ".join(f"{package}: {distribution}" for package, distribution in stale)
    )


def test_every_exemption_states_a_reason() -> None:
    """A reason is the entire content of an exemption; a blank one is an unexplained hole."""
    for (package, distribution), reason in EXEMPT.items():
        assert len(reason.split()) >= 12, (
            f"{package}: {distribution}: an exemption from the dependency cap needs a reason a "
            f"reviewer can check, not a label — got {reason!r}"
        )


def test_the_gate_rejects_an_uncapped_manifest(tmp_path: Path) -> None:
    """The gate BITES: a manifest with a lower-bound-only pin is found, a capped one is not.

    Asserted against an injected violation rather than against the real tree, so this stays a
    proof of the mechanism on the day the real tree is clean — which is every day after this one.
    """
    packages = tmp_path / "packages" / "latence-fake"
    packages.mkdir(parents=True)
    manifest = packages / "pyproject.toml"
    manifest.write_text(
        '[project]\nname = "latence-fake"\n'
        'dependencies = ["latence-core", "pydantic>=2.5,<3", "widgetlib>=1.2"]\n'
        '[project.optional-dependencies]\nextra = ["gadgetlib>=0.4"]\n',
        encoding="utf-8",
    )

    found = {(pin.distribution, pin.section) for pin in uncapped(declared_pins(tmp_path))}

    assert found == {("widgetlib", "dependencies"), ("gadgetlib", "optional-dependencies[extra]")}, (
        f"the cap gate missed an injected violation (or invented one): {found}"
    )
