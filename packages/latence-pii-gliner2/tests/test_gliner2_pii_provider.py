"""Gliner2PIIRedactor: learned PII Redaction over the fastino GLiNER2 framework (W3, ADR-0024).

W16 (ADR-0042): Redaction is a **chunk→chunk transform** — the Provider consumes ChunkRecords and
yields the SAME chunks with ``masked_content`` + ``pii_spans`` populated (the clean ``content`` stays
UNMASKED, offsets are chunk-local). The real ``gliner2`` stack (torch/transformers, mdeberta-v3) is
intentionally NOT installed in the dev/test env (ADR-0007/0016). These tests inject a tiny fake
``gliner2`` module whose model's ``extract_entities`` returns the **REAL** GLiNER2 output shape —
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
    ChunkRecord,
    Classification,
    InducedLabels,
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
    RedactionAction,
)
from latence_core.errors import ProviderError
from latence_pii_gliner2 import Gliner2PIIRedactor


def _chunk(
    text: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
    record_id: str = "doc1#chunk:0",
    document_record_id: str = "doc1",
    index: int = 0,
    induced_labels: InducedLabels | None = None,
) -> ChunkRecord:
    """A ChunkRecord — Redaction scans this chunk's ``content`` (W16, ADR-0042).

    Spans are chunk-local (offsets into ``content``); an identity ``offset_map`` + optional
    ``page_map`` let a span still resolve to its source page (the shared chunk page-resolution seam).
    ``induced_labels`` carries the corpus-induced pii_types the W12 effective-label union reads.
    """
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=0,
        char_end=len(text),
        page_start=1,
        page_end=1,
    )
    return ChunkRecord(
        record_id=record_id,
        document_record_id=document_record_id,
        index=index,
        provenance=prov,
        classification=Classification(language="en", sensitivity=sensitivity),
        content=text,
        token_count=max(len(text.split()), 0),
        page_map=page_map,
        induced_labels=induced_labels,
        offset_map=OffsetMap(segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text))
        if text
        else None,
    )


class BackendCompilerFailed(RuntimeError):
    """A stand-in for ``torch._dynamo.exc.BackendCompilerFailed`` — matched by class NAME.

    ``torch.compile`` raises this LAZILY on the first forward when its backend cannot compile the
    model. The provider's torch-free ``is_compile_backend_failure`` matches on the class name /
    module chain, so a same-named local class exercises the exact guard without importing torch.
    """


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
    #: the equivalence test needs distinct per-chunk results to catch an off-by-one / wrong-order map.
    per_text: dict[str, list[dict[str, Any]]] | None = None
    model_id: str = ""
    raise_on_extract: bool = False
    #: capture the from_pretrained kwargs (W9: assert quantize/compile threaded on cuda).
    last_kwargs: dict[str, Any] = {}
    #: capture the last ``max_len`` threaded into the extract call (W12 model-boundary guard).
    last_max_len: int | None = None
    #: force from_pretrained(quantize=…/compile=…) to raise, to exercise the guarded plain fallback.
    raise_on_optimized_load: bool = False
    #: when True, a model loaded with compile=True raises BackendCompilerFailed on its FIRST forward
    #: (torch.compile compiles lazily at inference) — the W9 inference-time fault the guard degrades.
    compiled_forward_raises: bool = False

    def __init__(self) -> None:
        self.batch_calls = 0
        self._compile_on = False

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeGLiNER2:
        # FAITHFUL to gliner2 1.3.2: from_pretrained ACCEPTS native quantize/compile kwargs.
        if cls.raise_on_optimized_load and ("quantize" in kwargs or "compile" in kwargs):
            msg = "simulated fp16/compile fault (torch/CUDA)"
            raise RuntimeError(msg)
        cls.last_kwargs = dict(kwargs)
        inst = cls()
        inst.model_id = model_id
        # torch.compile compiles LAZILY on the first forward: a compile=True load succeeds, then the
        # first batch_extract_entities raises. A reload with compile=False does not (the fallback).
        inst._compile_on = bool(kwargs.get("compile"))
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
        max_len: int = 768,
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        # The real batched API: one forward pass over all texts; result[i] is texts[i]'s per-label
        # dict — the exact single-text shape, one per input.
        self.batch_calls += 1
        type(self).last_max_len = max_len  # W12: record the threaded model-window bound
        if type(self).compiled_forward_raises and self._compile_on:
            # Mimic torch._dynamo.exc.BackendCompilerFailed on the FIRST (compiled) forward.
            raise BackendCompilerFailed("inductor backend cannot compile this model")
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
    _FakeGLiNER2.last_kwargs = {}
    _FakeGLiNER2.last_max_len = None
    _FakeGLiNER2.raise_on_optimized_load = False
    _FakeGLiNER2.compiled_forward_raises = False
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
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))
    assert isinstance(rec, ChunkRecord)
    # The clean content stays UNMASKED; masked_content carries the redaction.
    assert rec.content == _TEXT
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content
    assert "alice@acme.com" not in rec.masked_content
    assert "[PERSON]" in rec.masked_content
    assert "[EMAIL]" in rec.masked_content
    assert {s.pii_type for s in rec.pii_spans} == {"person", "email"}
    assert all(s.detector.startswith("gliner2:") for s in rec.pii_spans)


def test_span_offsets_are_chunk_local(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # The Provider runs the model over the chunk's ``content``, so the model's offsets ARE
    # chunk-local offsets: the stamped span offset matches, and the slice round-trips.
    doc_e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": doc_e_start, "end": doc_e_start + 14, "label": "email", "confidence": 0.9}
    ]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))
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
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))
    assert len(rec.pii_spans) == 1
    assert rec.pii_spans[0].char_start == e_start


def test_threshold_drops_low_scoring_spans(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.2}]
    cfg = {"pii_labels": ["person"], "threshold": 0.5}
    rec = next(iter(Gliner2PIIRedactor(cfg).redact([_chunk("Alice waved at the crowd.")])))
    assert len(rec.pii_spans) == 0


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
    public = next(iter(provider.redact([_chunk(_TEXT, sensitivity="public")])))
    internal = next(iter(provider.redact([_chunk(_TEXT, sensitivity="internal")])))
    assert public.masked_content is not None and "[PERSON]" in public.masked_content
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
    rec = next(iter(Gliner2PIIRedactor(cfg).redact([_chunk(_TEXT)])))
    # Only 'email' is on the allow-list; 'person' is dropped by the policy.
    assert {s.pii_type for s in rec.pii_spans} == {"email"}
    assert rec.masked_content is not None and "Alice" in rec.masked_content


def test_span_page_resolves_through_page_map(fake_gliner2: type[_FakeGLiNER2]) -> None:
    assembly = PageMap.from_page_texts(["Nothing here.", "Reach alice@acme.com."])
    text = assembly.text
    e_start = text.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9}
    ]
    rec = next(
        iter(Gliner2PIIRedactor(_config()).redact([_chunk(text, page_map=assembly.page_map)]))
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
        [s.char_start for s in next(iter(provider.redact([_chunk(_TEXT)]))).pii_spans]
        for _ in range(3)
    ]
    assert starts[0] == starts[1] == starts[2] == sorted(starts[0])


def test_raw_pii_never_stored(fake_gliner2: type[_FakeGLiNER2]) -> None:
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9}
    ]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))
    assert "alice@acme.com" not in rec.pii_spans[0].model_dump_json()


def test_empty_chunk_skipped_without_model_call(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk("")])))
    assert len(rec.pii_spans) == 0
    assert rec.masked_content == ""


def test_native_fault_becomes_typed_error_no_pii_leak(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # A native gliner2 exception (echoing the input text) → typed ProviderError carrying NO
    # record content / PII: only the provider + op + native type name (C2/C6).
    fake_gliner2.raise_on_extract = True
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    with pytest.raises(ProviderError) as excinfo:
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))
    message = str(excinfo.value)
    assert "redaction.gliner2" in message
    assert "extract_entities" in message
    assert "alice@acme.com" not in message
    assert "Alice" not in message


def test_missing_gliner2_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner2", None)
    provider = Gliner2PIIRedactor(_config())
    with pytest.raises(ImportError, match="pip install latence-pii-gliner2"):
        list(provider.redact([_chunk(_TEXT)]))


# --- W12 model-boundary guard: max_len threaded into the extract call -------------------------


def test_max_len_default_768_threaded_into_extract(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # The hard model-boundary guard: the provider threads a max_len (default 768) into the native
    # batch_extract_entities call so the mdeberta window is bounded, never a silent truncation.
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    list(Gliner2PIIRedactor(_config()).redact([_chunk("Alice waved at the crowd.")]))
    assert fake_gliner2.last_max_len == 768


def test_max_len_config_override_threaded(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    list(Gliner2PIIRedactor({**_config(), "max_len": 512}).redact([_chunk("Alice waved.")]))
    assert fake_gliner2.last_max_len == 512


def test_long_chunk_past_max_len_handled_no_crash(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # A chunk whose text is far longer than the (small, configured) window is handled without a
    # crash; the provider bounds the model window (documented truncation) rather than faulting.
    long_text = "Alice. " + ("filler word " * 500)
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    rec = next(iter(Gliner2PIIRedactor({**_config(), "max_len": 64}).redact([_chunk(long_text)])))
    assert isinstance(rec, ChunkRecord)
    assert fake_gliner2.last_max_len == 64  # the bound the model was asked to honor


# --- W12 induced pii_types honored via the effective_labels seam (ADR-0038) ------------------


_IBAN = "DE89370400440532013000"
_IBAN_TEXT = f"Alice holds account {_IBAN} today."


def _induced_iban_chunk() -> ChunkRecord:
    induced = InducedLabels(model_id="test-model", pii_types=["iban"])
    return _chunk(_IBAN_TEXT, induced_labels=induced)


def _iban_and_person_entities() -> list[dict[str, Any]]:
    a_start = _IBAN_TEXT.index("Alice")
    i_start = _IBAN_TEXT.index(_IBAN)
    return [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95},
        {"start": i_start, "end": i_start + len(_IBAN), "label": "iban", "confidence": 0.9},
    ]


def test_induced_pii_types_reach_the_effective_label_set(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    # THE W12 wiring: a corpus-induced 'iban' pii_type (on the ChunkRecord) is UNIONed onto the
    # config floor via effective_labels(config ∪ induced), so it is actually detected + redacted.
    fake_gliner2.entities = _iban_and_person_entities()
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_induced_iban_chunk()])))
    # config floor ['person','email'] ∪ induced ['iban'] → both the config 'person' AND the induced
    # 'iban' are in the effective label set, so both are masked.
    assert {s.pii_type for s in rec.pii_spans} == {"person", "iban"}
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content
    assert _IBAN not in rec.masked_content


def test_without_induction_iban_is_floored_not_model_detected(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    # The control that gives the wiring test teeth: with NO induced labels, 'iban' is outside the
    # config PII-label set, so the MODEL path filters it out — the model surfaces only the config
    # 'person'. This proves the induced union (not the model) is what puts 'iban' in the model's
    # effective set in `test_induced_pii_types_reach_the_effective_label_set`.
    #
    # W18 (ADR-0044): the UNIVERSAL FINANCIAL-PII FLOOR still masks the IBAN regardless — a floor type
    # is never left leaking whichever backend is configured (the pre-W18 leak this fixes). So the
    # 'iban' span is present but it comes from the deterministic floor (`detector == 'floor:iban'`),
    # NOT from the gliner2 model (`gliner2:person`), and the IBAN is gone from masked_content.
    fake_gliner2.entities = _iban_and_person_entities()
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_IBAN_TEXT)])))
    by_type = {s.pii_type: s.detector for s in rec.pii_spans}
    assert by_type == {"person": "gliner2:person", "iban": "floor:iban"}
    assert rec.masked_content is not None
    assert _IBAN not in rec.masked_content  # the floor guarantees the IBAN is masked


def test_financial_floor_masks_when_model_detects_nothing(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    """W18 (ADR-0044): the universal financial-PII floor is non-bypassable through the gliner2 path.

    The fake model detects NOTHING (``entities == []``) and no induced labels widen the set, so
    pre-W18 the IBAN leaked. The shared ``finalize_redacted_chunk`` floor masks it regardless
    (``detector == 'floor:iban'``).
    """
    fake_gliner2.entities = []  # the model finds nothing
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_IBAN_TEXT)])))
    assert rec.masked_content is not None
    assert _IBAN not in rec.masked_content
    assert [(s.pii_type, s.detector) for s in rec.pii_spans] == [("iban", "floor:iban")]


def test_config_floor_stays_mandatory_with_induction(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    # Induction EXTENDS but never replaces the config floor: the config 'person' is still detected
    # alongside the induced 'iban' (both present in the effective set).
    fake_gliner2.entities = _iban_and_person_entities()
    rec = next(
        iter(Gliner2PIIRedactor({"pii_labels": ["person"]}).redact([_induced_iban_chunk()]))
    )
    assert {s.pii_type for s in rec.pii_spans} == {"person", "iban"}


# --- W7 encoder batching (ADR-0036 §1 / ADR-0033) --------------------------------------------


def _distinct_chunks(n: int) -> list[ChunkRecord]:
    # N chunks each with a DISTINCT person name AND a DISTINCT span (chunk i is prefixed with i
    # filler chars so its name sits at offset i, span [i, i+9)). Distinct spans give the equivalence
    # test teeth: a mis-mapped batch hands chunk i another chunk's raw PII dict (start=j != i), so
    # chunk i masks the WRONG offsets and its masked_content diverges from the single-item path —
    # the assertions FAIL. (A shared [0,9) span would let a reversed map slip through, since every
    # record still masks its OWN content at [0,9); H-E1/E2 PII false-green hardening.)
    return [
        _chunk(
            ("." * i) + f"Person{i:03d} was here.",
            record_id=f"doc{i}#chunk:0",
            document_record_id=f"doc{i}",
            index=0,
        )
        for i in range(n)
    ]


def _per_text_person_map(chunks: list[ChunkRecord]) -> dict[str, list[dict[str, Any]]]:
    # Each chunk "<i dots>PersonNNN was here." -> a person entity spanning the name at its OWN chunk
    # offset i (span [i, i+9)). Keyed by index so chunk i carries start=i; a batch that maps results
    # back out of order gives a chunk a DIFFERENT start than i — caught by the record-for-record
    # test.
    return {
        c.content: [{"start": i, "end": i + 9, "label": "person", "confidence": 0.9}]
        for i, c in enumerate(chunks)
    }


def test_batched_equals_unbatched_record_for_record(fake_gliner2: type[_FakeGLiNER2]) -> None:
    chunks = _distinct_chunks(7)
    fake_gliner2.per_text = _per_text_person_map(chunks)
    cfg = {"pii_labels": ["person"]}

    unbatched = [
        r for c in chunks for r in Gliner2PIIRedactor({**cfg, "batch_size": 1}).redact([c])
    ]
    batched = list(Gliner2PIIRedactor({**cfg, "batch_size": 4}).redact(chunks))

    assert len(batched) == len(unbatched) == 7
    for b, u in zip(batched, unbatched, strict=True):
        assert b.record_id == u.record_id
        assert b.masked_content == u.masked_content
        assert [(s.char_start, s.char_end, s.pii_type) for s in b.pii_spans] == [
            (s.char_start, s.char_end, s.pii_type) for s in u.pii_spans
        ]
    for i, rec in enumerate(batched):
        assert rec.masked_content is not None
        assert f"Person{i:03d}" not in rec.masked_content
        assert "[PERSON]" in rec.masked_content


def test_model_called_ceil_n_over_batch_size_times(fake_gliner2: type[_FakeGLiNER2]) -> None:
    chunks = _distinct_chunks(7)
    fake_gliner2.per_text = _per_text_person_map(chunks)
    provider = Gliner2PIIRedactor({"pii_labels": ["person"], "batch_size": 3})
    list(provider.redact(chunks))
    # 7 chunks at batch_size=3 => ceil(7/3)=3 forward passes, not 7.
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
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _NoBatchGLiNER2:  # noqa: ARG003
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
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))


# --- W9 quantize/compile guarded optimized load (the redact 2.83× fix) ------------------------


def test_invalid_quantize_compile_knob_is_config_error() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="invalid quantize"):
        Gliner2PIIRedactor({**_config(), "quantize": "sometimes"})
    with pytest.raises(ConfigError, match="invalid compile"):
        Gliner2PIIRedactor({**_config(), "compile": "maybe"})


def test_cpu_default_loads_plain_no_optimized_kwargs(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # On the CPU-only test host (LATENCE_CUDA=0 in CI) quantize/compile 'auto' resolve to False, so
    # from_pretrained is called PLAIN (no quantize/compile kwargs) — the determinism-safe default.
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    list(Gliner2PIIRedactor(_config()).redact([_chunk("Alice waved.")]))
    assert "quantize" not in fake_gliner2.last_kwargs
    assert "compile" not in fake_gliner2.last_kwargs


def test_optimized_load_fault_degrades_to_plain_load(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # A quantize/compile fault degrades to a plain load (never a crash) at identical PII. Force the
    # knobs on (config override) AND make the optimized load raise — the guard must retry plain.
    fake_gliner2.raise_on_optimized_load = True
    a_start = _TEXT.index("Alice")
    fake_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95}
    ]
    cfg = {**_config(), "quantize": True, "compile": True}
    rec = next(iter(Gliner2PIIRedactor(cfg).redact([_chunk(_TEXT)])))
    # The plain fallback loaded and masked the identical span (no crash).
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content
    assert "[PERSON]" in rec.masked_content


def test_inference_time_compile_fault_degrades_to_uncompiled(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    """THE inference-time compile guard (pod-found): torch.compile compiles LAZILY on the first
    forward, so a backend that cannot compile the model raises ``BackendCompilerFailed`` INSIDE
    ``batch_extract_entities`` — NOT at load. The provider must catch it ONCE (unwrapping the
    guard's ProviderError to inspect the cause), re-load the UNCOMPILED model (quantize kept), and
    still mask the PII correctly — never a ProviderError crash.
    """
    fake_gliner2.compiled_forward_raises = True   # compile=True load's first forward raises
    a_start = _TEXT.index("Alice")
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95},
        {"start": e_start, "end": e_start + 14, "label": "email", "confidence": 0.9},
    ]
    cfg = {**_config(), "quantize": True, "compile": True}   # force compile on (cuda-style)
    provider = Gliner2PIIRedactor(cfg)
    rec = next(iter(provider.redact([_chunk(_TEXT)])))
    # Degraded to the uncompiled model + still masked the PII (no crash).
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content
    assert "alice@acme.com" not in rec.masked_content
    assert {s.pii_type for s in rec.pii_spans} == {"person", "email"}
    # The fallback ran (latched once) and re-loaded with compile=False, quantize kept.
    assert provider._compile_disabled is True
    assert fake_gliner2.last_kwargs.get("compile") is False
    assert fake_gliner2.last_kwargs.get("quantize") is True
