"""End-to-end: schema_induction feeds induced pii_types to the Redaction Stage through the Runner (W4).

The redaction quarter of the "four extractor seams" the slice claims. The subtle part is the carrier
mismatch: schema_induction is a chunk→chunk transform (it populates ``ChunkRecord.induced_labels``),
but Redaction reads a per-document ``DocumentRecord.induced_labels``. The Runner bridges the two — when
the ``redact`` Stage ``depends_on`` BOTH ``parse`` (the DocumentRecords) AND ``induce`` (the induced
chunk stream), the Runner projects each document's chunk-carried induced pii_types onto its
DocumentRecord (``latence_core.induced.project_induced_onto_documents``) so ``redaction.gliner_pii``
UNIONs the induced pii_types onto its config PII floor.

Two invariants proven here (mirroring the entity e2e), both on the real ``LocalRunner`` through
checkpoints, with the offline fakes (a scripted ``openai`` mock for the inducer + a label-capturing
``gliner`` fake for the PII detector):

1. **Induced pii_types reach the redactor.** The inducer returns a ``pii_type`` the ``redact`` Stage
   is NOT configured with (``iban``); the (faked) GLiNER-PII model is asked for the UNION of the
   config PII floor + the induced pii_types — the seam actually altering redaction, end to end.
2. **Off/undwired is byte-identical.** The SAME pipeline with ``redact depends_on: [parse]`` only (no
   induced chunk stream) asks the PII detector for EXACTLY the config PII floor — the backward-compat
   proof that dropping the ``induce`` edge (or the whole Stage) leaves Redaction unchanged.
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import _fake_openai as fake
import pytest
from latence_core.pipeline import Pipeline
from latence_core.runner import LocalRunner

if TYPE_CHECKING:
    from pathlib import Path

# Captures the label list each predict_entities call receives, so the test asserts the effective PII
# set the redactor was asked for.
_CAPTURED_PII_LABELS: list[list[str]] = []


class _CapturingGLiNER:
    """A gliner.GLiNER fake that records the PII labels it is asked to predict and returns none."""

    @classmethod
    def from_pretrained(cls, model_id: str) -> _CapturingGLiNER:  # noqa: ARG003
        return cls()

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        _CAPTURED_PII_LABELS.append(list(labels))
        return []  # no spans needed — the test asserts the LABELS, not the redaction result


@pytest.fixture(autouse=True)
def _pin_cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LATENCE_CUDA", "0")


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> None:
    _CAPTURED_PII_LABELS.clear()
    module = types.ModuleType("gliner")
    module.GLiNER = _CapturingGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)


def _induced_reply() -> str:
    # Induce a pii_type the redactor is NOT configured with — so the union is observable.
    return json.dumps(
        {
            "entity_types": [],
            "relation_types": [],
            "pii_types": ["iban"],
        }
    )


def _pipeline(store: str, docs: Path, *, wire_induced_pii: bool) -> Pipeline:
    # When wire_induced_pii, redact depends on BOTH parse and the induce chunk stream (the runner
    # projects induced pii_types onto the DocumentRecords). Otherwise it depends on parse alone.
    redact_deps = "[parse, induce]" if wire_induced_pii else "[parse]"
    return Pipeline.from_yaml_text(
        f"""
name: schema-induction-redaction-e2e
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
  - name: induce
    capability: schema_induction
    provider: label_inducer.llm
    depends_on: [content_screen]
    config:
      base_url: 'http://stub/v1'
  - name: redact
    capability: redaction
    provider: redaction.gliner_pii
    depends_on: {redact_deps}
    config:
      pii_labels: [email, phone number]
  - name: export_corpus
    capability: export
    provider: export.jsonl_parquet
    depends_on: [redact]
    config:
      basename: corpus
"""
    )


def test_induced_pii_types_reach_the_redactor(fake_gliner: None, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "Wire the payment to DE89370400440532013000 before Friday.\n", encoding="utf-8"
    )
    store = f"file://{tmp_path}/state"
    pipeline = _pipeline(store, docs, wire_induced_pii=True)

    with fake.install([fake.reply(_induced_reply())]):
        LocalRunner().run(pipeline, run_id="si-redact-e2e")

    # The redactor was asked for the UNION: config PII floor + induced pii_types, deduped + ordered.
    assert _CAPTURED_PII_LABELS, "the PII detector was never invoked"
    for labels in _CAPTURED_PII_LABELS:
        assert labels == ["email", "phone number", "iban"]


def test_without_induced_edge_redactor_gets_config_pii_floor(
    fake_gliner: None, tmp_path: Path
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        "Wire the payment to DE89370400440532013000 before Friday.\n", encoding="utf-8"
    )
    store = f"file://{tmp_path}/state"
    # redact depends on parse ONLY: even though the induce Stage still runs (and annotates chunks),
    # the induced pii_types are NOT projected onto the DocumentRecords, so the redactor gets exactly
    # the config PII floor — the byte-identical backward-compat path.
    pipeline = _pipeline(store, docs, wire_induced_pii=False)

    with fake.install([fake.reply(_induced_reply())]):
        LocalRunner().run(pipeline, run_id="si-redact-noedge-e2e")

    assert _CAPTURED_PII_LABELS, "the PII detector was never invoked"
    for labels in _CAPTURED_PII_LABELS:
        assert labels == ["email", "phone number"]
