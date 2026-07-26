"""PresidioRedactor: real Microsoft Presidio PII Redaction (W1-pii, #95; W16/ADR-0042 chunk-level).

The real Presidio stack (presidio-analyzer -> spaCy + a spaCy model) is intentionally NOT
installed in the dev/test env (ADR-0007/0016: CPU-first, offline, deterministic). These tests
inject a tiny fake ``presidio_analyzer`` module (+ its ``nlp_engine`` submodule) whose
``AnalyzerEngine`` returns fixed CHUNK-offset PII spans, so the Provider's redaction output,
span stamping, offset->page math, Presidio-type->pii_type mapping, sensitivity-keyed policy, the
H-C1 sensitive-doc no-op floor, the drop-in contract, and the "raw PII never stored / never
leaked into an error" invariants are exercised deterministically and offline.

W16 (ADR-0042): Redaction is a **chunk->chunk transform** — the Provider consumes ChunkRecords and
yields the SAME chunks with ``masked_content`` + ``pii_spans`` + ``redaction_disabled_for_sensitive``
populated (the clean ``content`` stays UNMASKED). Since each test chunk's ``content`` == the analysed
text, the analyzer's offsets ARE chunk-content offsets and a span slice still round-trips to the raw
value; an identity ``offset_map`` (+ optional ``page_map``) lets a chunk-local span still resolve to
its source page. Failure-mode tests cover the missing-dependency error, the config validation, and
the native-failure -> typed ProviderError mapping (no raw text / PII in the message).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
    RedactionAction,
)
from latence_core.errors import ConfigError, ProviderError
from latence_pii_presidio import PresidioRedactor
from latence_pii_presidio.provider import _DEFAULT_MODEL


def _chunk(
    content: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
    record_id: str = "doc1#chunk:0",
    index: int = 0,
    char_start: int = 0,
) -> ChunkRecord:
    """A ChunkRecord — Redaction scans this chunk's ``content`` (W16, ADR-0042).

    Spans are chunk-local (offsets into ``content``); an ``offset_map`` anchored at ``char_start``
    + optional ``page_map`` let a span still resolve to its source page (the shared chunk
    page-resolution seam).
    """
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=char_start,
        char_end=char_start + len(content),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=record_id,
        document_record_id="doc1",
        index=index,
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=content,
        token_count=max(len(content.split()), 0),
        page_map=page_map,
        offset_map=OffsetMap(
            segments=[OffsetSegment(local_start=0, doc_start=char_start)], length=len(content)
        )
        if content
        else None,
    )


class _FakeRecognizerResult:
    """Stands in for presidio_analyzer.RecognizerResult."""

    def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class _FakeAnalyzerEngine:
    """Stands in for presidio_analyzer.AnalyzerEngine with an analyze() method."""

    spans: list[dict[str, Any]] = []
    error: Exception | None = None
    load_error: Exception | None = None
    instances: int = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).instances += 1
        load_error = type(self).load_error
        if load_error is not None:
            raise load_error

    def analyze(
        self,
        text: str,
        language: str,  # noqa: ARG002
        entities: list[str] | None = None,
        score_threshold: float = 0.0,
        **_kwargs: Any,
    ) -> list[_FakeRecognizerResult]:
        error = type(self).error
        if error is not None:
            raise error
        wanted = set(entities) if entities is not None else None
        out: list[_FakeRecognizerResult] = []
        for s in type(self).spans:
            if wanted is not None and s["entity_type"] not in wanted:
                continue
            if float(s["score"]) < score_threshold:
                continue
            if int(s["end"]) > len(text):
                continue
            out.append(
                _FakeRecognizerResult(str(s["entity_type"]), int(s["start"]), int(s["end"]),
                                      float(s["score"]))
            )
        return out


class _FakeNlpEngineProvider:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def create_engine(self) -> object:
        return object()


@pytest.fixture()
def fake_presidio(monkeypatch: pytest.MonkeyPatch) -> type[_FakeAnalyzerEngine]:
    module = types.ModuleType("presidio_analyzer")
    module.AnalyzerEngine = _FakeAnalyzerEngine  # type: ignore[attr-defined]
    module.RecognizerResult = _FakeRecognizerResult  # type: ignore[attr-defined]
    nlp_module = types.ModuleType("presidio_analyzer.nlp_engine")
    nlp_module.NlpEngineProvider = _FakeNlpEngineProvider  # type: ignore[attr-defined]
    module.nlp_engine = nlp_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "presidio_analyzer", module)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", nlp_module)
    _FakeAnalyzerEngine.spans = []
    _FakeAnalyzerEngine.error = None
    _FakeAnalyzerEngine.load_error = None
    _FakeAnalyzerEngine.instances = 0
    return _FakeAnalyzerEngine


# The canonical PII-bearing text (email at [45, 62), person "Alice" at [0, 5)).
_TEXT = "Alice works for Acme Corp in Berlin. Contact alice@example.com about Acme Corp."
_EMAIL = "alice@example.com"
_EMAIL_START = _TEXT.index(_EMAIL)
_EMAIL_END = _EMAIL_START + len(_EMAIL)


def _config() -> dict[str, Any]:
    return {"entities": ["EMAIL_ADDRESS", "PERSON"]}


# -- config / profile ---------------------------------------------------------


def test_default_model_is_permissive_spacy_small() -> None:
    provider = PresidioRedactor({})
    assert provider._model_id == _DEFAULT_MODEL == "en_core_web_sm"  # noqa: SLF001


def test_rejects_non_list_entities() -> None:
    with pytest.raises(ValueError, match="'entities' config must be a list"):
        PresidioRedactor({"entities": "EMAIL_ADDRESS"})


def test_declares_profile_either_verified_mit() -> None:
    from latence_core.providers.profile import ProviderProfile, profile_of

    prof = profile_of(PresidioRedactor)
    assert isinstance(prof, ProviderProfile)
    assert prof.compute == "either"
    assert prof.model_id == "en_core_web_sm"
    assert prof.license == "MIT"  # Presidio's own license / effective across the stack
    assert prof.license_code == "MIT"  # presidio-analyzer + spaCy libraries
    assert prof.license_weights == "MIT"  # en_core_web_sm model weights
    assert prof.license_verified is True
    assert prof.deterministic is False  # cross-env byte-identity not promised — declared, not assumed
    assert prof.batch is True
    assert prof.cost_per_1k is None


# -- redaction happy path -----------------------------------------------------


def test_masks_detected_pii(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    a_start = _TEXT.index("Alice")
    fake_presidio.spans = [
        {"entity_type": "PERSON", "start": a_start, "end": a_start + 5, "score": 0.95},
        {"entity_type": "EMAIL_ADDRESS", "start": _EMAIL_START, "end": _EMAIL_END, "score": 0.9},
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(_TEXT)])))
    assert isinstance(rec, ChunkRecord)
    # The clean content is UNCHANGED (extraction still sees the entities).
    assert rec.content == _TEXT
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content
    assert _EMAIL not in rec.masked_content
    assert "[PERSON]" in rec.masked_content
    assert "[EMAIL]" in rec.masked_content  # EMAIL_ADDRESS mapped to the framework 'email' type
    # Presidio's UPPER_SNAKE types are mapped to the framework's lower-case pii_type vocabulary.
    assert {s.pii_type for s in rec.pii_spans} == {"person", "email"}
    assert all(s.detector == "presidio" for s in rec.pii_spans)


def test_span_offsets_land_on_chunk_coordinate_system(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # The Provider runs the analyzer over the chunk ``content``, so Presidio's offsets ARE
    # chunk-content offsets: the stamped span offset matches, and the slice round-trips to the raw
    # value (the coordinate system the spans + masked_content use).
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": _EMAIL_START, "end": _EMAIL_END, "score": 0.9}
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(_TEXT)])))
    span = rec.pii_spans[0]
    assert span.char_start == _EMAIL_START
    assert span.char_end == _EMAIL_END
    assert _TEXT[span.char_start : span.char_end] == _EMAIL


def test_chunk_carrier_preserved_content_unmasked(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # W16 (ADR-0042) the chunk carrier is preserved: the SAME chunk is re-emitted with its own
    # provenance intact (the chunk keeps its page range — per-span pages live on each PIISpan) and
    # ``content`` byte-UNCHANGED, while only ``masked_content`` carries the redaction.
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": _EMAIL_START, "end": _EMAIL_END, "score": 0.9}
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(_TEXT)])))  # _chunk has page_start=1
    assert rec.record_id == "doc1#chunk:0"
    assert rec.provenance.page_start == 1  # the chunk keeps its own page range
    assert rec.content == _TEXT  # content unmasked
    assert rec.masked_content is not None and rec.masked_content != _TEXT  # only masked_content masked


def test_out_of_bounds_span_is_dropped(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    # A span the analyzer returns past the end of the analysed text is defensively dropped (never a
    # crash, never an invalid record) — the offset invariant is enforced by the Provider, not trusted.
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": 5, "end": len(_TEXT) + 50, "score": 0.9}
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(_TEXT)])))
    assert len(rec.pii_spans) == 0
    assert rec.masked_content == _TEXT


def test_threshold_drops_low_scoring_spans(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    fake_presidio.spans = [
        {"entity_type": "PERSON", "start": 0, "end": 5, "score": 0.2}
    ]
    cfg = {"entities": ["PERSON"], "threshold": 0.5}
    rec = next(iter(PresidioRedactor(cfg).redact([_chunk(_TEXT)])))
    assert len(rec.pii_spans) == 0


def test_overlapping_spans_resolved_non_overlapping(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # Two overlapping analyzer results (a longer PERSON containing a shorter one) resolve to a
    # single, non-overlapping span — the record invariant (earlier start, longer span).
    fake_presidio.spans = [
        {"entity_type": "PERSON", "start": 0, "end": 5, "score": 0.9},
        {"entity_type": "PERSON", "start": 0, "end": 3, "score": 0.9},
    ]
    rec = next(iter(PresidioRedactor({"entities": ["PERSON"]}).redact([_chunk(_TEXT)])))
    assert len(rec.pii_spans) == 1
    assert rec.pii_spans[0].char_start == 0
    assert rec.pii_spans[0].char_end == 5


def test_types_narrowing_via_policy(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    # A policy that restricts to 'email' drops the person span even though the analyzer returns it.
    a_start = _TEXT.index("Alice")
    fake_presidio.spans = [
        {"entity_type": "PERSON", "start": a_start, "end": a_start + 5, "score": 0.95},
        {"entity_type": "EMAIL_ADDRESS", "start": _EMAIL_START, "end": _EMAIL_END, "score": 0.9},
    ]
    cfg = {"entities": ["EMAIL_ADDRESS", "PERSON"], "policies": {"public": {"types": ["email"]}}}
    rec = next(iter(PresidioRedactor(cfg).redact([_chunk(_TEXT)])))
    assert {s.pii_type for s in rec.pii_spans} == {"email"}


# -- metadata-aware policy (drop-in parity with hybrid_rule) -------------------


def test_policy_keys_off_sensitivity(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    a_start = _TEXT.index("Alice")
    fake_presidio.spans = [
        {"entity_type": "PERSON", "start": a_start, "end": a_start + 5, "score": 0.9}
    ]
    cfg = {
        "entities": ["PERSON"],
        "policies": {"public": {"action": "mask"}, "internal": {"action": "tag"}},
    }
    provider = PresidioRedactor(cfg)
    public = next(iter(provider.redact([_chunk(_TEXT, sensitivity="public")])))
    internal = next(iter(provider.redact([_chunk(_TEXT, sensitivity="internal")])))
    # Same text, different disposition driven purely by Classification.sensitivity.
    assert public.masked_content is not None and "[PERSON]" in public.masked_content
    assert internal.masked_content == _TEXT  # tagged, not masked
    assert internal.pii_spans[0].action is RedactionAction.TAG


def test_span_page_resolves_through_page_map(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    # The ADR-0031/0042 page-resolution seam resolves a span's own source page from the chunk's
    # offset_map + page_map — a span on page 2 cites page 2, not the inherited parent range.
    assembly = PageMap.from_page_texts(["Nothing here.", "Reach alice@example.com."])
    text = assembly.text
    e_start = text.index(_EMAIL)
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": e_start, "end": e_start + len(_EMAIL), "score": 0.9}
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(text, page_map=assembly.page_map)])))
    assert rec.pii_spans[0].page_start == 2


def test_span_page_resolves_on_a_later_chunk_that_carries_no_page_map(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    """REGRESSION (audit R1): the ONE resolver must span the whole ``redact`` call.

    Schema v18 stamps a document's ``page_map`` on its **first** chunk only, so a Provider that
    lets :func:`plan_chunk_redaction` mint a throwaway ``PageIndexResolver`` per chunk has no map
    for chunk 1..N and every PIISpan there is emitted with ``page_start=None`` — provenance lost
    for ~every chunk of every real document (ADR-0021/0031).
    """
    assembly = PageMap.from_page_texts(["Nothing here.", "Reach alice@example.com."])
    text = assembly.text
    boundary = assembly.page_map.pages[1].char_start  # doc offset where page 2 starts
    head, tail = text[:boundary], text[boundary:]
    e_local = tail.index(_EMAIL)
    # The fake drops any span running past the analysed text, so only the tail chunk detects it.
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": e_local, "end": e_local + len(_EMAIL),
         "score": 0.9}
    ]
    chunks = [
        # v18 carriage: the map rides chunk 0 ONLY.
        _chunk(head, page_map=assembly.page_map, record_id="doc1#chunk:0", index=0),
        _chunk(tail, record_id="doc1#chunk:1", index=1, char_start=boundary),
    ]
    recs = list(PresidioRedactor(_config()).redact(chunks))
    assert recs[0].pii_spans == []
    span = recs[1].pii_spans[0]
    assert (span.page_start, span.page_end) == (2, 2), (
        "a PII span on a non-first v18 chunk must resolve its OWN source page through the "
        "document-scoped resolver, not degrade to None"
    )


# -- H-C1 sensitive-doc no-op floor (must not regress on the drop-in) ----------


def test_no_op_floor_fires_on_sensitive_chunk_with_no_detectors(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # A confidential chunk whose policy skips redaction runs NO detectors: the chunk's
    # redaction_disabled_for_sensitive flag is set (control OFF, not "clean"). The analyzer is
    # never even invoked (skip short-circuits before load).
    cfg = {"entities": ["EMAIL_ADDRESS"], "policies": {"confidential": {"skip": True}}}
    rec = next(iter(PresidioRedactor(cfg).redact([_chunk(_TEXT, sensitivity="confidential")])))
    assert len(rec.pii_spans) == 0
    assert rec.redaction_disabled_for_sensitive is True
    assert fake_presidio.instances == 0  # skipped chunk never loads the analyzer


def test_no_op_floor_does_not_fire_on_public_chunk(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    cfg = {"entities": ["EMAIL_ADDRESS"], "policies": {"public": {"skip": True}}}
    rec = next(iter(PresidioRedactor(cfg).redact([_chunk(_TEXT, sensitivity="public")])))
    assert rec.redaction_disabled_for_sensitive is False  # public is not in the sensitive set


def test_no_op_floor_fires_when_types_intersect_nothing(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # A restricted chunk whose policy allows only a type NONE of the enabled entities can produce:
    # the control runs no detectors -> the floor fires (empty pii_spans means OFF, not clean).
    cfg = {
        "entities": ["EMAIL_ADDRESS"],
        "policies": {"restricted": {"types": ["ssn"]}},  # no enabled entity maps to 'ssn'
    }
    rec = next(iter(PresidioRedactor(cfg).redact([_chunk(_TEXT, sensitivity="restricted")])))
    assert rec.redaction_disabled_for_sensitive is True


# -- no-PII invariants (the load-bearing security checks) ---------------------


def test_financial_floor_masks_when_analyzer_detects_nothing(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    """W18 (ADR-0044): the universal financial-PII floor is non-bypassable through the presidio path.

    The fake analyzer returns NO spans, so pre-W18 the IBAN would have leaked. The shared
    ``finalize_redacted_chunk`` floor masks it regardless (``detector == 'floor:iban'``).
    """
    iban = "DE89370400440532013000"
    text = f"Alice holds account {iban} today."
    fake_presidio.spans = []  # the analyzer finds nothing
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(text)])))
    assert rec.masked_content is not None
    assert iban not in rec.masked_content
    assert [(s.pii_type, s.detector) for s in rec.pii_spans] == [("iban", "floor:iban")]


def test_raw_pii_never_stored_on_span(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": _EMAIL_START, "end": _EMAIL_END, "score": 0.9}
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(_TEXT)])))
    assert _EMAIL not in rec.pii_spans[0].model_dump_json()


def test_clean_chunk_is_no_op_not_error(fake_presidio: type[_FakeAnalyzerEngine]) -> None:
    fake_presidio.spans = []
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk(_TEXT)])))
    assert len(rec.pii_spans) == 0
    assert rec.masked_content == _TEXT  # no PII → masked == source, empty spans (the clean case)


def test_empty_chunk_skipped_without_analyzer_call(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    fake_presidio.spans = [
        {"entity_type": "EMAIL_ADDRESS", "start": _EMAIL_START, "end": _EMAIL_END, "score": 0.9}
    ]
    rec = next(iter(PresidioRedactor(_config()).redact([_chunk("")])))
    assert len(rec.pii_spans) == 0
    assert rec.masked_content == ""
    assert fake_presidio.instances == 0  # short text below min_chars never loads the analyzer


# -- failure modes ------------------------------------------------------------


def test_native_analyze_failure_becomes_typed_provider_error_without_pii(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # A native RuntimeError inside analyze() → typed ProviderError, and the raised message must NOT
    # echo the offending input text (no PII/content leak — the load-bearing C2/C6 check for a PII
    # provider: a leaked detected PII in an error would be the worst kind of failure).
    fake_presidio.error = RuntimeError(f"spaCy blew up on {_TEXT} with {_EMAIL}")
    provider = PresidioRedactor(_config())
    with pytest.raises(ProviderError) as exc_info:
        list(provider.redact([_chunk(_TEXT)]))
    message = str(exc_info.value)
    assert "provider failure in 'analyze'" in message
    assert "RuntimeError" in message  # the native type name is safe to surface
    assert _EMAIL not in message  # the detected PII is NOT leaked
    assert _TEXT not in message
    assert "spaCy blew up" not in message  # the native message (which echoed the input) is dropped


def test_missing_spacy_model_maps_to_config_error(
    fake_presidio: type[_FakeAnalyzerEngine],
) -> None:
    # A missing/uninstalled spaCy model makes engine construction raise OSError — classified as an
    # operator ConfigError ("download the model"), never a silent wrong result.
    fake_presidio.load_error = OSError("[E050] Can't find model 'en_core_web_sm'")
    provider = PresidioRedactor(_config())
    with pytest.raises(ConfigError, match="config error in 'load_model'"):
        list(provider.redact([_chunk(_TEXT)]))


def test_missing_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "presidio_analyzer", None)
    provider = PresidioRedactor(_config())
    with pytest.raises(ImportError, match="pip install latence-pii-presidio"):
        list(provider.redact([_chunk(_TEXT)]))


# -- real-weights test (gated on the dep; skipped offline) --------------------


@pytest.mark.skipif(
    importlib.util.find_spec("presidio_analyzer") is None
    or importlib.util.find_spec("en_core_web_sm") is None,
    reason="presidio-analyzer + the en_core_web_sm spaCy model must BOTH be installed for the "
    "real-weights path — gate on the model too so a deps-present-but-model-absent env skips "
    "rather than attempting a network download (offline-safe).",
)
def test_real_presidio_detects_email() -> None:  # pragma: no cover - runs only on a rig
    """With the real dep + a spaCy model present, Presidio detects the email at a real offset.

    The honest real-weights boundary (same discipline as the gliner trio + the ST embedder): it
    runs only where ``presidio-analyzer`` and a spaCy model are actually installed (a rig), and is
    skipped in the offline CPU-first CI env. It asserts the REAL analyzer masks the email and the
    span round-trips to the raw value on the chunk coordinate system.
    """
    provider = PresidioRedactor({"entities": ["EMAIL_ADDRESS"]})
    rec = next(iter(provider.redact([_chunk(_TEXT)])))
    assert rec.masked_content is not None and _EMAIL not in rec.masked_content
    assert any(s.pii_type == "email" for s in rec.pii_spans)
    span = next(s for s in rec.pii_spans if s.pii_type == "email")
    assert _TEXT[span.char_start : span.char_end] == _EMAIL
