"""The go-live tutorial's §4 stack must BE the enterprise-sota spine (audit R6-wizard_profile-15).

``docs/TUTORIAL-fresh-pod-walkthrough.md`` §3 tells the operator "The generated stack is exactly
the SOTA stack of §4", and §4 opens "The wizard above generates this file for you… This is the
**exact** enterprise-sota spine". That is a *checkable* claim, and nothing in the tree checked it:
the tutorial drifted to an un-blessed redactor and silently dropped BOTH the ADR-0051
``graph_features`` block (#193) and the ADR-0049 ``bm25`` block (#194), so an operator who
hand-authored §4 ran a stack with a dead sparse leg and all-zero graph bonuses.

These tests parse the §4 YAML block and diff its stage wiring against the LIVE
``build_stack_config(profile="enterprise-sota")`` output — so any future wizard change the
tutorial does not track fails the gate, not just the three drifts found in the audit.
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from latence_core.setup.wizard import (
    BLESSED_REDACT_PROVIDER,
    SetupAnswers,
    build_pipeline_dict,
)

# tests/ -> latence-core/ -> packages/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_TUTORIAL = _REPO_ROOT / "docs" / "TUTORIAL-fresh-pod-walkthrough.md"

# Keys whose values are operator-specific placeholders in the tutorial ({CORPUS_DIR}, the OCR
# endpoint, the stack name) rather than wiring. The wizard takes them from the operator's answers,
# so they are legitimately different and are compared structurally (present/absent), not by value.
#
# `model` is deliberately NOT in this set (audit R7-wizard_profile-11): `_wizard_stages()` feeds
# the tutorial's own `ocr_model`/`llm_model` into the fixture, so `parse.config.model` and
# `induce.config.model` compare byte-for-byte — while the wizard's HARD-CODED Granite embedder
# checkpoint (`_sota_embedder_block()`) is not operator-supplied at all. Exempting `model` blinded
# the gate to exactly the drift class it exists to catch.
_OPERATOR_SPECIFIC = frozenset({"uri", "path", "base_url", "storage_uri", "corpus_id", "name"})


def _tutorial_text() -> str:
    assert _TUTORIAL.is_file(), "the go-live tutorial must exist"
    return _TUTORIAL.read_text(encoding="utf-8")


def _tutorial_stack() -> dict[str, Any]:
    """Parse the §4 hand-authored stack YAML out of the tutorial."""
    text = _tutorial_text()
    blocks = [
        m.group(1)
        for m in re.finditer(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
        if "stacks/my-corpus.yaml" in m.group(1)
    ]
    assert blocks, "§4's hand-authored `stacks/my-corpus.yaml` YAML block must exist"
    assert len(blocks) == 1, f"exactly one §4 stack block expected, found {len(blocks)}"
    parsed = yaml.safe_load(blocks[0])
    assert isinstance(parsed, dict), "§4's stack block must parse as a YAML mapping"
    return parsed


def _stages_by_name(stack: dict[str, Any], what: str) -> dict[str, dict[str, Any]]:
    """Index a stack mapping's ``stages`` list by Stage name."""
    stages = stack.get("stages")
    assert isinstance(stages, list) and stages, f"{what} must declare a non-empty stage list"
    return {str(s["name"]): dict(s) for s in stages}


def _tutorial_stages() -> dict[str, dict[str, Any]]:
    return _stages_by_name(_tutorial_stack(), "§4's stack")


def _wizard_stages() -> dict[str, dict[str, Any]]:
    """The stages the wizard ACTUALLY emits for the profile the tutorial claims to reproduce."""
    answers = SetupAnswers(
        source="file:///workspace/corpus",
        storage="file:///workspace/latence-out",
        parser="parser.lighton_vllm",
        device="cuda",
        profile="enterprise-sota",
        ocr_base_url="http://localhost:8000/v1",
        ocr_model="lightonai/LightOnOCR-2-1B",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model="openai/gpt-4.1-mini",
        pii=True,
    )
    return _stages_by_name(dict(build_pipeline_dict(answers)), "the wizard's enterprise-sota stack")


def _strip_operator_specific(value: Any) -> Any:
    """Neutralise operator-supplied placeholder VALUES at the TOP level of a stage's config.

    Deliberately NOT recursive (audit R7-wizard_profile-11): every operator answer the wizard
    threads into a stack lands as a top-level key of a stage's ``config`` (``path``, ``base_url``,
    ``corpus_id``, …). Nested mappings are *nested Provider blocks* — the embedder, the
    ``graph_features`` block, the ``bm25`` block, the universal parser's ``ocr`` sub-block — whose
    pins the wizard hard-codes. Recursing neutralised those pins too, so §4 could name any
    embedder checkpoint it liked and the gate stayed green.
    """
    if isinstance(value, dict):
        return {k: (None if k in _OPERATOR_SPECIFIC else v) for k, v in value.items()}
    return value


def _wiring_drift(
    tutorial: dict[str, dict[str, Any]], wizard: dict[str, dict[str, Any]]
) -> list[str]:
    """Every wiring difference between §4's stages and the wizard's, one line each.

    The single comparison implementation, so a regression test can drive it against a mutated
    tutorial instead of re-implementing (and thereby not testing) the gate.
    """
    drift: list[str] = []
    for name, want in wizard.items():
        got = tutorial.get(name, {})
        for key in ("capability", "provider", "depends_on", "config"):
            want_v = _strip_operator_specific(want.get(key))
            got_v = _strip_operator_specific(got.get(key))
            if want_v != got_v:
                drift.append(f"  {name}.{key}: tutorial={got_v!r} wizard={want_v!r}")
    return drift


def test_tutorial_section4_names_the_blessed_redactor() -> None:
    """§4 must wire the redactor the wizard blesses, not its de-blessed twin (#197)."""
    redact = _tutorial_stages().get("redact")
    assert redact is not None, "§4's PII-ON spine must have a `redact` stage"
    assert redact["provider"] == BLESSED_REDACT_PROVIDER, (
        "§4 claims to be the exact enterprise-sota spine but wires "
        f"{redact['provider']!r}; the wizard emits {BLESSED_REDACT_PROVIDER!r}"
    )


def test_tutorial_section4_carries_the_graph_feature_block() -> None:
    """Without `graph_features` (#193) every centrality/community signal is silently absent."""
    tutorial = _tutorial_stages()["graph"].get("config") or {}
    wizard = _wizard_stages()["graph"].get("config") or {}
    assert "graph_features" in wizard, "wizard fixture drift: enterprise-sota must emit it"
    assert tutorial.get("graph_features") == wizard["graph_features"], (
        "§4's `graph` stage must carry the ADR-0051 graph_features block the wizard emits "
        f"({wizard['graph_features']!r}); got {tutorial.get('graph_features')!r}"
    )


def test_tutorial_section4_carries_the_bm25_block() -> None:
    """Without `bm25` (#194) no term statistics are written and the sparse leg is dead."""
    tutorial = _tutorial_stages()["export_corpus"].get("config") or {}
    wizard = _wizard_stages()["export_corpus"].get("config") or {}
    assert "bm25" in wizard, "wizard fixture drift: enterprise-sota must emit it"
    assert tutorial.get("bm25") == wizard["bm25"], (
        "§4's `export_corpus` stage must carry the ADR-0049 bm25 block the wizard emits "
        f"({wizard['bm25']!r}); got {tutorial.get('bm25')!r}"
    )


def test_tutorial_section4_stage_wiring_equals_the_wizard_output() -> None:
    """The whole claim, checked: same stages, same providers, same deps, same config wiring.

    Operator-specific placeholder VALUES (corpus URI, OCR endpoint, stack name) are neutralised;
    everything that is *wiring* must match exactly, so any future wizard change the tutorial does
    not track fails here.
    """
    tutorial = _tutorial_stages()
    wizard = _wizard_stages()

    assert list(tutorial) == list(wizard), (
        "§4's stage list must be the wizard's stage list, in order: "
        f"{list(tutorial)} != {list(wizard)}"
    )

    drift = _wiring_drift(tutorial, wizard)
    assert not drift, "§4 is NOT the exact enterprise-sota spine:\n" + "\n".join(drift)


@pytest.mark.parametrize(
    ("stage", "path", "replacement"),
    [
        # The hard-coded Granite embedder pin, in BOTH stages that carry it (R7-wizard_profile-11).
        (
            "disambiguate",
            ("embedder", "config", "model"),
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        (
            "export_corpus",
            ("embedder", "config", "model"),
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        # The other nested Provider pins recursion used to be able to hide.
        ("export_corpus", ("embedder", "provider"), "embedding.not_blessed"),
        ("graph", ("graph_features", "provider"), "graph_features.not_blessed"),
        ("export_corpus", ("bm25", "provider"), "tokenizer.not_blessed"),
    ],
)
def test_the_gate_catches_drift_in_a_NESTED_provider_block(
    stage: str, path: tuple[str, ...], replacement: str
) -> None:
    """Meta-test: a §4 that re-pins a nested Provider block MUST show up as drift.

    The R6 gate neutralised `model` recursively, so a tutorial naming a 384-dim English-only
    MiniLM behind the block's own `dimension: 768` diffed clean — the exact drift class the gate
    was written to make impossible. This drives the gate's own comparison logic against a mutated
    tutorial and asserts it reports the change.
    """
    wizard = _wizard_stages()
    tutorial = deepcopy(wizard)  # start from a perfect match, then plant ONE drift

    node = tutorial[stage]["config"]
    for key in path[:-1]:
        node = node[key]
    assert node[path[-1]] != replacement, "fixture drift: the mutation must change something"
    node[path[-1]] = replacement

    assert not _wiring_drift(deepcopy(wizard), wizard), (
        "control: an unmutated copy of the wizard output must diff clean"
    )
    drift = _wiring_drift(tutorial, wizard)
    assert drift, (
        f"the gate did not notice §4 re-pinning {stage}.config.{'.'.join(path)} to "
        f"{replacement!r} — a nested Provider block is wiring, not an operator answer"
    )


@pytest.mark.parametrize("token", ["graph_features", "bm25"])
def test_the_go_live_tutorial_mentions_the_signals_it_ships(token: str) -> None:
    """A prose-level backstop: the go-live guide cannot be silent about these two signals."""
    assert token in _tutorial_text(), (
        f"the go-live tutorial never mentions {token!r} — an operator reading it end-to-end "
        "would never learn the signal exists"
    )
