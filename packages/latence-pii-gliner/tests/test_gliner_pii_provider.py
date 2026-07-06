"""GlinerPIIRedactor: learned zero-shot PII Redaction (S6, ADR-0024).

The real ``gliner`` stack (torch/transformers) is intentionally NOT installed in the
dev/test env (ADR-0007/0016). These tests inject a tiny fake ``gliner`` module whose
model returns fixed document-offset PII spans, so the Provider's redaction output,
span stamping, offset→page math, sensitivity-keyed policy, and "raw PII never stored"
invariant are exercised deterministically and offline. One test covers the
missing-dependency error path and the zero-shot config validation (pii_labels
required).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from latence_core.contracts import (
    Classification,
    DocumentRecord,
    PageMap,
    Provenance,
    RedactionAction,
    RedactionRecord,
)
from latence_pii_gliner import GlinerPIIRedactor


def _doc(
    content: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
) -> DocumentRecord:
    """A parsed DocumentRecord — Redaction scans this (original markdown), not chunks."""
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=0,
        char_end=len(content),
        page_start=1,
        page_end=1,
    )
    return DocumentRecord(
        record_id="doc1",
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=content,
        page_map=page_map,
    )


class _FakeGLiNER:
    """Stands in for gliner.GLiNER with a predict_entities method."""

    entities: list[dict[str, Any]] = []
    model_id: str = ""

    @classmethod
    def from_pretrained(cls, model_id: str) -> _FakeGLiNER:
        inst = cls()
        inst.model_id = model_id
        return inst

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        return [
            e for e in type(self).entities if e["label"] in labels and e["score"] >= threshold
        ]


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _FakeGLiNER.entities = []
    return _FakeGLiNER


def _config() -> dict[str, Any]:
    return {"pii_labels": ["person", "email"]}


_TEXT = "Reach Alice at alice@acme.com."


def test_requires_pii_labels() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'pii_labels' list"):
        GlinerPIIRedactor({})


def test_default_model_is_permissive_apache() -> None:
    assert GlinerPIIRedactor(_config())._model_id == "urchade/gliner_multi_pii-v1"


def test_masks_detected_pii(fake_gliner: type[_FakeGLiNER]) -> None:
    a_start = _TEXT.index("Alice")
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "score": 0.95},
        {"start": e_start, "end": e_start + 14, "label": "email", "score": 0.9},
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_doc(_TEXT)]))[0]
    assert isinstance(rec, RedactionRecord)
    assert "Alice" not in rec.masked_content
    assert "alice@acme.com" not in rec.masked_content
    assert "[PERSON]" in rec.masked_content
    assert "[EMAIL]" in rec.masked_content
    assert {s.pii_type for s in rec.pii_spans} == {"person", "email"}
    assert all(s.detector == "gliner-pii" for s in rec.pii_spans)


def test_span_offsets_are_document_level(fake_gliner: type[_FakeGLiNER]) -> None:
    # The Provider runs the model over the whole parsed-document markdown, so the
    # model's offsets ARE document offsets: the stamped span offset matches, and the
    # slice round-trips to the raw value (the coordinate system the spans use).
    doc_e_start = _TEXT.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": doc_e_start, "end": doc_e_start + 14, "label": "email", "score": 0.9}
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_doc(_TEXT)]))[0]
    span = rec.pii_spans[0]
    assert span.char_start == doc_e_start
    assert _TEXT[span.char_start : span.char_end] == "alice@acme.com"


def test_threshold_drops_low_scoring_spans(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.2}]
    cfg = {"pii_labels": ["person"], "threshold": 0.5}
    rec = list(GlinerPIIRedactor(cfg).redact([_doc("Alice waved at the crowd.")]))[0]
    assert rec.pii_count == 0


def test_policy_keys_off_sensitivity(fake_gliner: type[_FakeGLiNER]) -> None:
    a_start = _TEXT.index("Alice")
    fake_gliner.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "score": 0.9}
    ]
    cfg = {
        "pii_labels": ["person"],
        "policies": {"public": {"action": "mask"}, "internal": {"action": "tag"}},
    }
    provider = GlinerPIIRedactor(cfg)
    public = list(provider.redact([_doc(_TEXT, sensitivity="public")]))[0]
    internal = list(provider.redact([_doc(_TEXT, sensitivity="internal")]))[0]
    assert public.policy == "public"
    assert "[PERSON]" in public.masked_content
    assert internal.policy == "internal"
    assert internal.masked_content == _TEXT  # tagged, not masked
    assert internal.pii_spans[0].action is RedactionAction.TAG


def test_span_page_resolves_through_page_map(fake_gliner: type[_FakeGLiNER]) -> None:
    assembly = PageMap.from_page_texts(["Nothing here.", "Reach alice@acme.com."])
    text = assembly.text
    e_start = text.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "score": 0.9}
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_doc(text, page_map=assembly.page_map)]))[0]
    assert rec.pii_spans[0].page_start == 2


def test_raw_pii_never_stored(fake_gliner: type[_FakeGLiNER]) -> None:
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "score": 0.9}
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_doc(_TEXT)]))[0]
    assert "alice@acme.com" not in rec.pii_spans[0].model_dump_json()


def test_empty_chunk_skipped_without_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    rec = list(GlinerPIIRedactor(_config()).redact([_doc("")]))[0]
    assert rec.pii_count == 0
    assert rec.masked_content == ""


def test_missing_gliner_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner", None)
    provider = GlinerPIIRedactor(_config())
    with pytest.raises(ImportError, match="pip install latence-pii-gliner"):
        list(provider.redact([_doc(_TEXT)]))
