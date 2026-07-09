"""Gliner2PIIRedactor: learned PII Redaction over the fastino GLiNER2 framework (W3, ADR-0024).

The real ``gliner2`` stack (torch/transformers, mdeberta-v3) is intentionally NOT installed in
the dev/test env (ADR-0007/0016). These tests inject a tiny fake ``gliner2`` module whose model's
``extract_entities`` returns the **REAL** GLiNER2 output shape —
``{"entities": {label: [{text, confidence, start, end}]}}`` — so the Provider's redaction output,
span stamping, offset→page math, sensitivity-keyed policy, "raw PII never stored" invariant, and
typed-error-no-PII-leak path are exercised deterministically and offline.

Anti-false-green (the PaddleOCR/relex lesson): the stub is faithful to the per-label-dict shape,
and a dedicated anchor test proves a provider that mis-parses that shape as a flat list (the
``gliner``-library shape) would detect ZERO spans — so the false-green trap is caught.
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
from latence_core.errors import ProviderError
from latence_pii_gliner2 import Gliner2PIIRedactor


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


class _FakeGLiNER2:
    """Stands in for ``gliner2.GLiNER2`` with the REAL per-label-dict ``extract_entities``.

    ``entities`` is a flat list of ``{start, end, label, confidence}`` rows the test seeds;
    ``extract_entities`` reshapes it into the real
    ``{"entities": {label: [{text, confidence, start, end}]}}`` dict — the exact shape the
    fastino GLiNER2 framework returns (introspected + run on the pod). Threshold + label filters
    are applied on the way (as the real model does), then the surviving rows are nested per-label.
    """

    entities: list[dict[str, Any]] = []
    #: an optional per-text map (text -> flat entity rows) so batched inputs get DISTINCT results;
    #: the equivalence test needs distinct per-doc results to catch an off-by-one / wrong-order map.
    per_text: dict[str, list[dict[str, Any]]] | None = None
    model_id: str = ""
    raise_on_extract: bool = False

    def __init__(self) -> None:
        self.batch_calls = 0

    @classmethod
    def from_pretrained(cls, model_id: str) -> _FakeGLiNER2:
        inst = cls()
        inst.model_id = model_id
        return inst

    def _rows_for(self, text: str) -> list[dict[str, Any]]:
        per_text = type(self).per_text
        if per_text is not None:
            return per_text.get(text, [])
        return type(self).entities

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,
        include_confidence: bool = False,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        if type(self).raise_on_extract:
            # A native gliner2 fault echoing the input text — the Provider must NOT leak it.
            raise RuntimeError(f"cuda oom while extracting {text!r}")
        by_label: dict[str, list[dict[str, Any]]] = {}
        for e in self._rows_for(text):
            label = e["label"]
            if label not in entity_types or e["confidence"] < threshold:
                continue
            if e["end"] > len(text):
                continue
            by_label.setdefault(label, []).append(
                {
                    "text": text[e["start"] : e["end"]],
                    "confidence": e["confidence"],
                    "start": e["start"],
                    "end": e["end"],
                }
            )
        return {"entities": by_label}

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,
        include_confidence: bool = False,
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        # The real batched API: one forward pass over all texts; result[i] is texts[i]'s per-label
        # dict — the exact single-text shape, one per input.
        self.batch_calls += 1
        return [
            self.extract_entities(
                text,
                entity_types,
                threshold=threshold,
                include_spans=include_spans,
                include_confidence=include_confidence,
            )
            for text in texts
        ]


@pytest.fixture()
def fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER2]:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _FakeGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    _FakeGLiNER2.entities = []
    _FakeGLiNER2.per_text = None
    _FakeGLiNER2.raise_on_extract = False
    return _FakeGLiNER2


def _config() -> dict[str, Any]:
    return {"pii_labels": ["person", "email"]}


_TEXT = "Reach Alice at alice@acme.com."


def test_requires_pii_labels() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'pii_labels' list"):
        Gliner2PIIRedactor({})


def test_default_model_is_permissive_apache() -> None:
    assert Gliner2PIIRedactor(_config())._model_id == "fastino/GLiNER2-Guardrails-PII-Multi"


def test_profile_is_verified_apache_either() -> None:
    prof = Gliner2PIIRedactor.profile()
    assert prof.compute == "either"
    assert prof.license == "Apache-2.0"
    assert prof.license_verified is True
    assert prof.license_weights == "Apache-2.0"
    assert prof.license_code == "Apache-2.0"
    assert prof.model_id == "fastino/GLiNER2-Guardrails-PII-Multi"


def test_masks_detected_pii(fake_gliner2: type[_FakeGLiNER2]) -> None:
    a_start = _TEXT.index("Alice")
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95},
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9},
    ]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_doc(_TEXT)])))
    assert isinstance(rec, RedactionRecord)
    assert "Alice" not in rec.masked_content
    assert "alice@acme.com" not in rec.masked_content
    assert "[PERSON]" in rec.masked_content
    assert "[EMAIL]" in rec.masked_content
    assert {s.pii_type for s in rec.pii_spans} == {"person", "email"}
    assert all(s.detector.startswith("gliner2:") for s in rec.pii_spans)


def test_span_offsets_are_document_level(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # The Provider runs the model over the whole parsed-document markdown, so the model's
    # offsets ARE document offsets: the stamped span offset matches, and the slice round-trips.
    doc_e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": doc_e_start, "end": doc_e_start + 14, "label": "email", "confidence": 0.9}
    ]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_doc(_TEXT)])))
    span = rec.pii_spans[0]
    assert span.char_start == doc_e_start
    assert _TEXT[span.char_start : span.char_end] == "alice@acme.com"


def test_flat_list_misparse_fails_anchor(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """ANCHOR (anti-false-green): the REAL per-label-dict shape is required, not a flat list.

    This proves the stub is faithful to the ``{"entities": {label: [rows]}}`` shape and that a
    provider treating the output as a flat list of rows (the ``gliner``-library shape) detects
    ZERO spans. We simulate the mis-parse by reading ``out["entities"]`` and iterating it as if
    it were a list of row-dicts — which yields the label STRINGS (dict keys), not rows, so no
    span offsets are recoverable. If ``extract_entities`` returned a flat list, this loop would
    succeed; because it returns a per-label dict, it recovers nothing — the trap bites.
    """
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9}
    ]
    model = fake_gliner2.from_pretrained("x")
    out = model.extract_entities(
        _TEXT, ["email"], threshold=0.5, include_spans=True, include_confidence=True
    )
    # The correct (nested) shape the Provider relies on:
    assert isinstance(out["entities"], dict)
    assert out["entities"]["email"][0]["start"] == e_start
    # A naive flat-list mis-parse: iterating out["entities"] yields label keys (str), never rows.
    flat_misparse = list(out["entities"])
    assert flat_misparse == ["email"]
    assert all(isinstance(item, str) for item in flat_misparse)  # not dict-rows -> zero spans

    # And end-to-end: the FAITHFUL provider DOES recover the span (proving it parses the nesting).
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_doc(_TEXT)])))
    assert rec.pii_count == 1
    assert rec.pii_spans[0].char_start == e_start


def test_threshold_drops_low_scoring_spans(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.2}]
    cfg = {"pii_labels": ["person"], "threshold": 0.5}
    rec = next(iter(Gliner2PIIRedactor(cfg).redact([_doc("Alice waved at the crowd.")])))
    assert rec.pii_count == 0


def test_policy_keys_off_sensitivity(fake_gliner2: type[_FakeGLiNER2]) -> None:
    a_start = _TEXT.index("Alice")
    fake_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.9}
    ]
    cfg = {
        "pii_labels": ["person"],
        "policies": {"public": {"action": "mask"}, "internal": {"action": "tag"}},
    }
    provider = Gliner2PIIRedactor(cfg)
    public = next(iter(provider.redact([_doc(_TEXT, sensitivity="public")])))
    internal = next(iter(provider.redact([_doc(_TEXT, sensitivity="internal")])))
    assert public.policy == "public"
    assert "[PERSON]" in public.masked_content
    assert internal.policy == "internal"
    assert internal.masked_content == _TEXT  # tagged, not masked
    assert internal.pii_spans[0].action is RedactionAction.TAG


def test_policy_allow_list_drops_unlisted_type(fake_gliner2: type[_FakeGLiNER2]) -> None:
    a_start = _TEXT.index("Alice")
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.9},
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9},
    ]
    cfg = {
        "pii_labels": ["person", "email"],
        "policies": {"public": {"action": "mask", "types": ["email"]}},
    }
    rec = next(iter(Gliner2PIIRedactor(cfg).redact([_doc(_TEXT)])))
    # Only 'email' is on the allow-list; 'person' is dropped by the policy.
    assert {s.pii_type for s in rec.pii_spans} == {"email"}
    assert "Alice" in rec.masked_content


def test_span_page_resolves_through_page_map(fake_gliner2: type[_FakeGLiNER2]) -> None:
    assembly = PageMap.from_page_texts(["Nothing here.", "Reach alice@acme.com."])
    text = assembly.text
    e_start = text.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9}
    ]
    rec = next(
        iter(Gliner2PIIRedactor(_config()).redact([_doc(text, page_map=assembly.page_map)]))
    )
    assert rec.pii_spans[0].page_start == 2


def test_deterministic_span_order(fake_gliner2: type[_FakeGLiNER2]) -> None:
    a_start = _TEXT.index("Alice")
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9},
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95},
    ]
    provider = Gliner2PIIRedactor(_config())
    starts = [
        [s.char_start for s in next(iter(provider.redact([_doc(_TEXT)]))).pii_spans]
        for _ in range(3)
    ]
    assert starts[0] == starts[1] == starts[2] == sorted(starts[0])


def test_raw_pii_never_stored(fake_gliner2: type[_FakeGLiNER2]) -> None:
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9}
    ]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_doc(_TEXT)])))
    assert "alice@acme.com" not in rec.pii_spans[0].model_dump_json()


def test_empty_document_skipped_without_model_call(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_doc("")])))
    assert rec.pii_count == 0
    assert rec.masked_content == ""


def test_native_fault_becomes_typed_error_no_pii_leak(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # A native gliner2 exception (echoing the input text) → typed ProviderError carrying NO
    # record content / PII: only the provider + op + native type name (C2/C6).
    fake_gliner2.raise_on_extract = True
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    with pytest.raises(ProviderError) as excinfo:
        list(Gliner2PIIRedactor(_config()).redact([_doc(_TEXT)]))
    message = str(excinfo.value)
    assert "redaction.gliner2" in message
    assert "extract_entities" in message
    assert "alice@acme.com" not in message
    assert "Alice" not in message


def test_missing_gliner2_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner2", None)
    provider = Gliner2PIIRedactor(_config())
    with pytest.raises(ImportError, match="pip install latence-pii-gliner2"):
        list(provider.redact([_doc(_TEXT)]))


# --- W7 encoder batching (ADR-0036 §1 / ADR-0033) --------------------------------------------


def _distinct_docs(n: int) -> list[DocumentRecord]:
    # N docs each with a DISTINCT person name AND a DISTINCT span (doc i is prefixed with i filler
    # chars so its name sits at offset i, span [i, i+9)). Distinct spans give the equivalence test
    # teeth: a mis-mapped batch hands doc i another doc's raw PII dict (start=j != i), so doc i
    # masks the WRONG offsets and its masked_content diverges from the single-item path — the
    # assertions FAIL. (A shared [0,9) span would let a reversed map slip through, since every
    # record still masks its OWN content at [0,9); H-E1/E2 PII false-green hardening.)
    docs: list[DocumentRecord] = []
    for i in range(n):
        content = ("." * i) + f"Person{i:03d} was here."
        prov = Provenance(
            source_uri=f"file:///c/{i}.txt",
            file_name=f"{i}.txt",
            file_type="txt",
            file_size=100,
            document_id=f"sha256:doc{i}",
            char_start=0,
            char_end=len(content),
            page_start=1,
            page_end=1,
        )
        docs.append(
            DocumentRecord(
                record_id=f"doc{i}",
                provenance=prov,
                classification=Classification(language="en", sensitivity="public"),
                content=content,
            )
        )
    return docs


def _per_text_person_map(docs: list[DocumentRecord]) -> dict[str, list[dict[str, Any]]]:
    # Each doc "<i dots>PersonNNN was here." -> a person entity spanning the name at its OWN doc
    # offset i (span [i, i+9)). Keyed by index so doc i carries start=i; a batch that maps results
    # back out of order gives a doc a DIFFERENT start than i — caught by the record-for-record test.
    return {
        d.content: [{"start": i, "end": i + 9, "label": "person", "confidence": 0.9}]
        for i, d in enumerate(docs)
    }


def test_batched_equals_unbatched_record_for_record(fake_gliner2: type[_FakeGLiNER2]) -> None:
    docs = _distinct_docs(7)
    fake_gliner2.per_text = _per_text_person_map(docs)
    cfg = {"pii_labels": ["person"]}

    unbatched = [
        r for d in docs for r in Gliner2PIIRedactor({**cfg, "batch_size": 1}).redact([d])
    ]
    batched = list(Gliner2PIIRedactor({**cfg, "batch_size": 4}).redact(docs))

    assert len(batched) == len(unbatched) == 7
    for b, u in zip(batched, unbatched, strict=True):
        assert b.record_id == u.record_id
        assert b.masked_content == u.masked_content
        assert [(s.char_start, s.char_end, s.pii_type) for s in b.pii_spans] == [
            (s.char_start, s.char_end, s.pii_type) for s in u.pii_spans
        ]
    for i, rec in enumerate(batched):
        assert f"Person{i:03d}" not in rec.masked_content
        assert "[PERSON]" in rec.masked_content


def test_model_called_ceil_n_over_batch_size_times(fake_gliner2: type[_FakeGLiNER2]) -> None:
    docs = _distinct_docs(7)
    fake_gliner2.per_text = _per_text_person_map(docs)
    provider = Gliner2PIIRedactor({"pii_labels": ["person"], "batch_size": 3})
    list(provider.redact(docs))
    # 7 docs at batch_size=3 => ceil(7/3)=3 forward passes, not 7.
    assert provider._model.batch_calls == 3


def test_batch_size_must_be_positive_int() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="batch_size must be >= 1"):
        Gliner2PIIRedactor({"pii_labels": ["person"], "batch_size": 0})


class _NoBatchGLiNER2:
    """A checkpoint that exposes only the single-text ``extract_entities``, NOT the W7 batched
    ``batch_extract_entities`` — the false-green class the provider guard must catch legibly.

    Does NOT subclass ``_FakeGLiNER2`` (which defines the batched method): a bare class genuinely
    lacking it is what makes ``hasattr(model, "batch_extract_entities")`` False at runtime, unlike
    an ``= None`` shadow (still an attribute — ``hasattr`` stays True).
    """

    @classmethod
    def from_pretrained(cls, model_id: str) -> _NoBatchGLiNER2:  # noqa: ARG003
        return cls()

    def extract_entities(
        self,
        text: str,  # noqa: ARG002
        entity_types: list[str],  # noqa: ARG002
        threshold: float = 0.5,  # noqa: ARG002
        include_spans: bool = False,  # noqa: ARG002
        include_confidence: bool = False,  # noqa: ARG002
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {"entities": {}}


def test_missing_batch_extract_entities_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The batched entry point is offline-unverifiable, so if the real checkpoint lacks it the
    # provider must fail with an actionable message naming the batched API, not a bare
    # AttributeError buried in the guard. Offline stand-in for the pod script's Gate 0.
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _NoBatchGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    with pytest.raises(AttributeError, match="batch_extract_entities"):
        list(Gliner2PIIRedactor(_config()).redact([_doc(_TEXT)]))
