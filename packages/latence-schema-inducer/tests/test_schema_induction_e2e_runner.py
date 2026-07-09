"""End-to-end: schema_induction through the local Runner feeds induced labels to the extractors (W4).

The Baseline bar's ≥1-end-to-end-through-the-Runner proof, plus the two invariants the slice turns
on:

1. **The induced labels reach the extractor.** A real Source → Parse → Chunk → Content-Screen →
   ``schema_induction`` → Entity-Extraction pipeline on :class:`~latence_core.runner.LocalRunner`,
   where ``schema_induction`` (``label_inducer.llm``, driven by the faithful ``openai`` mock) induces
   ``entity_types`` the ``entity.gliner`` extractor is NOT configured with. We capture the labels the
   (faked) GLiNER model is asked to predict and assert they are the UNION of the config labels + the
   induced labels — the additive seam actually altering extraction, end to end through checkpoints.

2. **Off by default is byte-identical.** The SAME pipeline with the ``schema_induction`` Stage removed
   asks GLiNER for EXACTLY the config labels — the backward-compat proof.

Both use the offline fakes (no network, no torch): a scripted ``openai`` mock for the inducer and a
label-capturing ``gliner`` fake for the extractor, injected into ``sys.modules`` (both imports are
deferred to first use, so the in-process Runner picks them up).
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import _fake_openai as fake
import pytest
from latence_core.contracts import ChunkRecord
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner
from latence_core.storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

# Captures the label list each predict_entities call receives, so the test asserts the effective set.
_CAPTURED_LABELS: list[list[str]] = []


class _CapturingGLiNER:
    """A gliner.GLiNER fake that records the labels it is asked to predict and returns none."""

    @classmethod
    def from_pretrained(cls, model_id: str) -> _CapturingGLiNER:  # noqa: ARG003
        return cls()

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        _CAPTURED_LABELS.append(list(labels))
        return []  # no mentions needed — the test asserts the LABELS, not the extraction result


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    _CAPTURED_LABELS.clear()
    module = types.ModuleType("gliner")
    module.GLiNER = _CapturingGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)


def _induced_reply() -> str:
    # Induce entity types the extractor is NOT configured with — so the union is observable.
    return json.dumps(
        {
            "entity_types": ["location", "product"],
            "relation_types": [],
            "pii_types": [],
        }
    )


def _pipeline(store: str, docs: Path, *, with_induction: bool) -> Pipeline:
    induction_stage = (
        """
  - name: induce
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [content_screen]
    config:
      base_url: 'http://stub/v1'
"""
        if with_induction
        else ""
    )
    entities_dep = "induce" if with_induction else "content_screen"
    return Pipeline.from_yaml_text(
        f"""
name: schema-induction-e2e
storage_uri: {store}
stages:
  - name: source
    capability: source
    provider: source.local_folder
    config:
      path: 'file://{docs}'
      sensitivity: internal
  - name: parse
    capability: parse
    provider: parser.plaintext
    depends_on: [source]
  - name: chunk
    capability: chunk
    provider: chunk.markdown
    depends_on: [parse]
  - name: content_screen
    capability: content_screening
    provider: screening.content_keyword
    depends_on: [chunk]
{induction_stage}  - name: entities
    capability: entity_extraction
    provider: entity.gliner
    depends_on: [{entities_dep}]
    config:
      labels: [person, organization]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [entities]
    config:
      basename: corpus
"""
    )


def test_induced_labels_reach_the_extractor(fake_gliner: None, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Alice works for Acme Corp in Berlin.\n", encoding="utf-8")
    store = f"file://{tmp_path}/state"
    pipeline = _pipeline(store, docs, with_induction=True)

    with fake.install([fake.reply(_induced_reply())]):
        report = LocalRunner().run(pipeline, run_id="si-e2e")

    # The extractor was asked for the UNION: config floor + induced, deduped + order-stable.
    assert _CAPTURED_LABELS, "the extractor was never invoked"
    for labels in _CAPTURED_LABELS:
        assert labels == ["person", "organization", "location", "product"]

    # The chunk checkpoint carries the induced labels (survived serialize/deserialize round-trip).
    storage = Storage.for_uri(store)
    induce_ckpt = f"{store}/_latence/runs/si-e2e/checkpoints/induce.jsonl"
    assert storage.exists(induce_ckpt)
    rows = [
        ChunkRecord.model_validate_json(line)
        for line in storage.read_text(induce_ckpt).splitlines()
        if line.strip()
    ]
    assert rows
    for chunk in rows:
        assert chunk.induced_labels is not None
        assert chunk.induced_labels.entity_types == ["location", "product"]
        assert chunk.induced_labels.model_id == "gpt-4.1-mini"

    # The Quality Report records the induction Stage with zero fail-open (a clean run).
    induce_metrics = next(m for m in report.stages if m.name == "induce")
    assert induce_metrics.capability == "schema_induction"
    assert induce_metrics.induction_fallbacks == 0
    assert induce_metrics.induction_documents == 1


def test_without_induction_stage_is_byte_identical_labels(fake_gliner: None, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Alice works for Acme Corp in Berlin.\n", encoding="utf-8")
    store = f"file://{tmp_path}/state"
    pipeline = _pipeline(store, docs, with_induction=False)

    # No induction Stage ⇒ no openai call at all; the extractor gets EXACTLY the config labels.
    LocalRunner().run(pipeline, run_id="no-si-e2e")
    assert _CAPTURED_LABELS
    for labels in _CAPTURED_LABELS:
        assert labels == ["person", "organization"]
