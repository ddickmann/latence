"""The BLESSED redact Provider of the setup profiles (#197) — emission and design must agree.

ADR-0041 says `latence setup` emits ONE opinionated, quality-baked config: the wizard's output must
BE the profile's documented design, not drift from it. It had drifted. The generator's own header
docstring described the spine as ``redact[gliner2, financial-PII]`` and the
``FINANCIAL_PII_LABELS`` constant documented the floor as the one ``redaction.gliner2`` runs, while
``_build_sota_stages`` emitted ``redaction.gliner_pii`` — a different package, a different library,
a different checkpoint. ADR-0041 carried the same split (§2 named ``redaction.gliner2``, §7 named
``redaction.gliner_pii``).

#197 decides ONE blessed redactor for both profiles — ``redaction.gliner2``, ADR-0036 W9's unified
extraction+PII engine, now that #192 made it loadable and version-pinned — and pins the decision
here so emission and documentation cannot drift apart again:

* **AC1/AC2** — both profiles emit :data:`BLESSED_REDACT_PROVIDER`, the generated file header names
  the same Provider it emits, and ADR-0041 records the decision in a machine-checked line.
* **AC4** — the ADR-0044 universal financial-PII floor still masks an IBAN the model missed, driven
  through the REAL blessed Provider with the config the wizard actually emitted.

Offline scope note: the enterprise-sota profile is a GPU-pod stack. The AC4 run resolves the real
``redaction.gliner2`` Provider class from the entry-point registry, installs the faithful fake
``gliner2`` module (the per-label-dict API, seeded to detect NOTHING), and substitutes ``device:
cpu`` for the profile's ``cuda`` so the run happens for real on a CPU-only host. Every other key —
``pii_labels`` (the ADR-0044 financial floor), ``max_len``, ``threshold`` — is used verbatim from
the emitted config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.capability import ProviderRegistry
from latence_core.conformance import stub as stubs
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    InducedLabels,
    OffsetMap,
    OffsetSegment,
    Provenance,
)
from latence_core.pipeline import Pipeline
from latence_core.setup import SetupAnswers, build_stack_config
from latence_core.setup.wizard import (
    BLESSED_REDACT_PROVIDER,
    FINANCIAL_PII_LABELS,
    PROFILE_LITE,
    PROFILE_SOTA,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_ADR_0041 = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "adr"
    / "0041-guided-setup-and-process-cli-opinionated-quality-baked-config-and-one-toggle.md"
)

#: The line ADR-0041 carries so the documented decision is machine-checkable against the emission.
_BLESSED_MARKER = re.compile(r"Blessed redact Provider \(both profiles\): `([a-z0-9_.]+)`")


def _answers(tmp_path: Path, *, profile: str) -> SetupAnswers:
    src = tmp_path / "corpus"
    src.mkdir(exist_ok=True)
    (src / "a.pdf").write_text("a", encoding="utf-8")
    return SetupAnswers(
        source=f"file://{src}",
        storage=f"file://{tmp_path}/out",
        parser="parser.render",
        device="auto",
        pii=True,
        extract_on="unmasked",
        profile=profile,
        extensions=("pdf", "txt"),
    )


def _redact_stage(tmp_path: Path, *, profile: str) -> tuple[str, dict[str, Any]]:
    """The emitted ``redact`` Stage's (provider, config) — read out of the generated YAML."""
    text = build_stack_config(_answers(tmp_path, profile=profile))
    stage = next(s for s in Pipeline.from_yaml_text(text).stages if s.name == "redact")
    return stage.provider, dict(stage.config)


# ---------------------------------------------------------------------------
# AC1 + AC2 — one decided Provider; emission and documented design agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", [PROFILE_LITE, PROFILE_SOTA])
def test_every_profile_emits_the_blessed_redactor(tmp_path: Path, profile: str) -> None:
    """AC2 (the regression): the emitted redact Provider IS the blessed one, in EVERY profile.

    Pre-#197 ``enterprise-sota`` emitted ``redaction.gliner_pii`` while ``lite`` emitted
    ``redaction.gliner2`` and the generator's own documented spine named ``gliner2`` — so this fails
    on the pre-fix generator for ``profile=enterprise-sota``.
    """
    provider, _ = _redact_stage(tmp_path, profile=profile)
    assert provider == BLESSED_REDACT_PROVIDER


def test_the_two_profiles_do_not_disagree_on_the_redactor(tmp_path: Path) -> None:
    """One decision, not one per profile: swapping the tier must not swap the PII engine."""
    lite, _ = _redact_stage(tmp_path, profile=PROFILE_LITE)
    sota, _ = _redact_stage(tmp_path, profile=PROFILE_SOTA)
    assert lite == sota == BLESSED_REDACT_PROVIDER


@pytest.mark.parametrize("profile", [PROFILE_LITE, PROFILE_SOTA])
def test_the_generated_header_names_the_provider_it_emits(tmp_path: Path, profile: str) -> None:
    """The file the operator reads must name the redactor the file actually wires."""
    text = build_stack_config(_answers(tmp_path, profile=profile))
    header = text.split("\nname:", 1)[0]
    named = {m for m in re.findall(r"redaction\.[a-z0-9_]+", header)}
    assert named == {BLESSED_REDACT_PROVIDER}


def test_the_blessed_provider_is_a_real_registered_provider() -> None:
    """The decision names something installable — not a Provider that does not exist."""
    assert BLESSED_REDACT_PROVIDER in ProviderRegistry().names()


# ---------------------------------------------------------------------------
# AC1 — the decision + reasoning are recorded where the profile design lives
# ---------------------------------------------------------------------------


def test_adr_0041_records_the_blessed_redactor(tmp_path: Path) -> None:
    """AC1 (the regression): ADR-0041 declares the blessed redactor, and it is what we emit.

    Pre-#197 the ADR carried BOTH candidates in design positions (§2 ``redaction.gliner2``, §7
    ``redaction.gliner_pii``) and no decision line at all — so this fails on the pre-fix docs.
    """
    adr = _ADR_0041.read_text(encoding="utf-8")
    declared = _BLESSED_MARKER.findall(adr)
    assert len(declared) == 1, f"ADR-0041 must declare the blessed redactor exactly once: {declared}"
    emitted, _ = _redact_stage(tmp_path, profile=PROFILE_SOTA)
    assert declared[0] == emitted == BLESSED_REDACT_PROVIDER


def test_adr_0041_stage_wiring_names_only_the_blessed_redactor() -> None:
    """No stale second candidate left in a *wiring* position (`provider: …` / a `redact → …` bullet).

    The reasoning prose may name the rejected candidate (that is the point of recording a decision);
    what must not survive is a second Provider presented as what the profile wires.
    """
    adr = _ADR_0041.read_text(encoding="utf-8")
    wired = set(re.findall(r"redact\s*→\s*\n?\s*`(redaction\.[a-z0-9_]+)`", adr))
    assert wired == {BLESSED_REDACT_PROVIDER}


# ---------------------------------------------------------------------------
# AC4 — the ADR-0044 financial-PII floor survives the blessed Provider
# ---------------------------------------------------------------------------

_IBAN = "DE89370400440532013000"
_TEXT = f"Rechnung an Alice Johnson, iban {_IBAN}, Betrag 1200 EUR."


def _chunk() -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=len(_TEXT),
        document_id="sha256:d1",
        char_start=0,
        char_end=len(_TEXT),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id="d1#chunk:0",
        document_record_id="d1",
        index=0,
        provenance=prov,
        classification=Classification(language="de", sensitivity="public"),
        content=_TEXT,
        token_count=len(_TEXT.split()),
        offset_map=OffsetMap(
            segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(_TEXT)
        ),
    )


def test_emitted_redact_config_carries_the_financial_pii_floor(tmp_path: Path) -> None:
    """The ADR-0044 financial label floor rides the blessed Provider's emitted config, both tiers."""
    for profile in (PROFILE_LITE, PROFILE_SOTA):
        _, config = _redact_stage(tmp_path, profile=profile)
        assert tuple(config["pii_labels"]) == FINANCIAL_PII_LABELS


def test_financial_floor_masks_the_iban_the_blessed_model_missed(tmp_path: Path) -> None:
    """AC4, as BEHAVIOUR: the real blessed Provider, the emitted config, a model that finds NOTHING.

    The IBAN must still be absent from ``masked_content`` and present as a ``floor:iban`` span — the
    ADR-0044 guarantee, proven through whichever Provider #197 blessed rather than assumed from it.
    """
    _, config = _redact_stage(tmp_path, profile=PROFILE_SOTA)
    config["device"] = "cpu"  # offline: the profile pins cuda; the floor is device-independent
    cls = ProviderRegistry().load(BLESSED_REDACT_PROVIDER)

    with stubs.install_gliner2(entities=[]):  # the model detects NOTHING
        records: Sequence[object] = list(cls(config).redact([_chunk()]))

    assert len(records) == 1
    rec = records[0]
    # W16 (ADR-0042): redaction is a chunk→chunk transform — the masked variant rides the chunk.
    assert isinstance(rec, ChunkRecord)
    assert rec.masked_content is not None
    assert _IBAN not in rec.masked_content
    assert [(s.pii_type, s.detector) for s in rec.pii_spans] == [("iban", "floor:iban")]


# ---------------------------------------------------------------------------
# T2 — the UNIVERSAL guardrail floor rides the blessed Provider's emitted config
# ---------------------------------------------------------------------------


def test_emitted_redact_config_turns_the_learned_guardrails_on(tmp_path: Path) -> None:
    """Both tiers wire the guardrails half of the guardrails+PII checkpoint (T2)."""
    for profile in (PROFILE_LITE, PROFILE_SOTA):
        _, config = _redact_stage(tmp_path, profile=profile)
        assert config["guardrails"] is True


def test_guard_floor_flags_an_injection_in_a_chunk_that_induced_no_guard_types(
    tmp_path: Path,
) -> None:
    """T2 as BEHAVIOUR through the blessed Provider + the emitted config, not from a flag.

    The chunk's induced schema names PII types and NO guard categories at all — under a purely
    per-chunk guard screen the model would never be asked about prompt injection here. The
    UNIVERSAL guard floor asks anyway, and the chunk comes back carrying a ``prompt_injection``
    RiskMarker with its text UNMODIFIED (a guard hit is flagged, never masked).
    """
    _, config = _redact_stage(tmp_path, profile=PROFILE_SOTA)
    config["device"] = "cpu"  # offline: the profile pins cuda; the floor is device-independent
    cls = ProviderRegistry().load(BLESSED_REDACT_PROVIDER)
    chunk = _chunk().model_copy(
        update={"induced_labels": InducedLabels(model_id="m", pii_types=["invoice number"])}
    )
    rows = [{"start": 0, "end": 8, "label": "prompt injection", "confidence": 0.9}]

    with stubs.install_gliner2(entities=rows):
        rec = next(iter(cls(config).redact([chunk])))

    assert isinstance(rec, ChunkRecord)
    assert [m.category for m in rec.risk_markers] == ["prompt_injection"]
    assert rec.masked_content is not None
    assert rec.masked_content.startswith("Rechnung")  # flagged, not masked
