"""The shipped example RAG-corpus Export is PII-handled by default (#37, regression).

ADR-0027 and the `graph-slice.yaml` / `demo-pipeline.yaml` headers describe the RAG-ready
corpus as "Screened, PII-handled". The debt in #37 was that the shipped example wired the
corpus Export (`export.jsonl_parquet`) straight off `chunk`, bypassing Redaction (S6) — so
the *default* output was NOT PII-handled, contradicting the claim. The fix routes the corpus
Export off a Redaction Stage in the shipped examples.

This test loads the ACTUAL shipped example YAML (not an inline copy) and asserts every
corpus Export transitively depends on a Redaction Stage. It fails if the fix is reverted
(corpus Export back off `chunk`), closing the doc-vs-wiring gap machine-side so it cannot
regress silently again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from latence_core.pipeline import CapabilityKind, Pipeline, Stage

# packages/latence-core/tests/ -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES = _REPO_ROOT / "examples"

# The examples whose header claims a "Screened, PII-handled" RAG corpus: their
# corpus Export must be PII-handled by default.
_PII_HANDLED_CORPUS_EXAMPLES = ["graph-slice.yaml", "demo-pipeline.yaml"]

# The Provider that materializes the RAG-ready corpus (JSONL + Parquet).
_CORPUS_EXPORT_PROVIDER = "export.jsonl_parquet"


def _has_redaction_ancestor(stage: Stage, by_name: dict[str, Stage]) -> bool:
    """Walk ``depends_on`` transitively; True if any ancestor is a Redaction Stage."""
    seen: set[str] = set()
    frontier = list(stage.depends_on)
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        parent = by_name.get(name)
        if parent is None:
            continue
        if parent.capability is CapabilityKind.REDACTION:
            return True
        frontier.extend(parent.depends_on)
    return False


@pytest.mark.parametrize("example", _PII_HANDLED_CORPUS_EXAMPLES)
def test_shipped_corpus_export_is_pii_handled_by_default(example: str) -> None:
    path = _EXAMPLES / example
    pipeline = Pipeline.from_yaml_text(path.read_text(encoding="utf-8"))
    by_name = {s.name: s for s in pipeline.stages}

    corpus_exports = [s for s in pipeline.stages if s.provider == _CORPUS_EXPORT_PROVIDER]
    assert corpus_exports, (
        f"{example} declares no {_CORPUS_EXPORT_PROVIDER!r} corpus Export to check."
    )

    for stage in corpus_exports:
        assert _has_redaction_ancestor(stage, by_name), (
            f"{example}: corpus Export {stage.name!r} has no Redaction Stage upstream, so the "
            f"shipped corpus is NOT PII-handled — contradicting the 'Screened, PII-handled' "
            f"claim (#37). Wire it off a redaction Stage."
        )


def test_graph_slice_defines_a_redaction_stage() -> None:
    """The graph-slice example ships an actual Redaction Stage (not just a commented hint)."""
    pipeline = Pipeline.from_yaml_text(
        (_EXAMPLES / "graph-slice.yaml").read_text(encoding="utf-8")
    )
    redactions = [s for s in pipeline.stages if s.capability is CapabilityKind.REDACTION]
    assert redactions, "graph-slice.yaml ships no Redaction Stage; the corpus cannot be PII-handled."
