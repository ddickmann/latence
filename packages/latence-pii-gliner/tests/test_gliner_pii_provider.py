"""GlinerPIIRedactor: learned zero-shot PII Redaction (S6, ADR-0024).

W16 (ADR-0042): Redaction is a **chunk→chunk transform** — the Provider consumes ChunkRecords
and yields the SAME chunks with ``masked_content`` + ``pii_spans`` populated (the clean ``content``
stays UNMASKED, offsets are chunk-local). The real ``gliner`` stack (torch/transformers) is
intentionally NOT installed in the dev/test env (ADR-0007/0016). These tests inject a tiny fake
``gliner`` module whose model returns fixed chunk-offset PII spans, so the Provider's redaction
output, span stamping, offset→page math, sensitivity-keyed policy, and "raw PII never stored"
invariant are exercised deterministically and offline. One test covers the missing-dependency error
path and the zero-shot config validation (pii_labels required).
"""

from __future__ import annotations

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
from latence_pii_gliner import GlinerPIIRedactor


def _chunk(
    text: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
    record_id: str = "doc1#chunk:0",
    document_record_id: str = "doc1",
    index: int = 0,
) -> ChunkRecord:
    """A ChunkRecord — Redaction scans this chunk's ``content`` (W16, ADR-0042).

    Spans are chunk-local (offsets into ``content``); an identity ``offset_map`` + optional
    ``page_map`` let a span still resolve to its source page (the shared chunk page-resolution seam).
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
        offset_map=OffsetMap(segments=[OffsetSegment(local_start=0, doc_start=0)], length=len(text))
        if text
        else None,
    )


class _FakeGLiNER:
    """Stands in for gliner.GLiNER with predict_entities AND batch_predict_entities (W7).

    The real batched API is ``batch_predict_entities(texts, labels, threshold=…) ->
    list[list[dict]]`` (result ``i`` for ``texts[i]``). The W7 provider calls the batched form;
    the fake keeps a per-instance ``batch_calls`` counter so a test asserts ``ceil(N/batch_size)``
    calls, not N. An optional ``per_text`` map gives DISTINCT results per chunk text so a
    mis-mapped batch (off-by-one / wrong order) is caught by the equivalence test.
    """

    entities: list[dict[str, Any]] = []
    per_text: dict[str, list[dict[str, Any]]] | None = None
    model_id: str = ""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.cast_dtype: Any = None  # W8: dtype applied via .to(), None until cast
        #: FAITHFUL to the real GLiNER model: a ``config`` with a ``max_len`` the library truncates
        #: to. The provider's W12 model-boundary guard sets this so a test can assert the bound.
        self.config = types.SimpleNamespace(max_len=384)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeGLiNER:
        # FAITHFUL to real gliner 0.2.27: from_pretrained REJECTS torch_dtype/attn_implementation.
        for bad in ("torch_dtype", "attn_implementation"):
            if bad in kwargs:
                msg = f"UniEncoderSpanModel.__init__() got an unexpected keyword argument '{bad}'"
                raise TypeError(msg)
        inst = cls()
        inst.model_id = model_id
        return inst

    def to(self, dtype: Any) -> _FakeGLiNER:
        self.cast_dtype = dtype  # W8: the plain-gliner dtype path is .to(bf16) on cuda
        return self

    def _for_text(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        per_text = type(self).per_text
        source = per_text.get(text, []) if per_text is not None else type(self).entities
        return [e for e in source if e["label"] in labels and e["score"] >= threshold]

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        return self._for_text(text, labels, threshold)

    def batch_predict_entities(
        self, texts: list[str], labels: list[str], threshold: float = 0.5
    ) -> list[list[dict[str, Any]]]:
        self.batch_calls += 1
        return [self._for_text(text, labels, threshold) for text in texts]


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _FakeGLiNER.entities = []
    _FakeGLiNER.per_text = None
    return _FakeGLiNER


def _config() -> dict[str, Any]:
    return {"pii_labels": ["person", "email"]}


_TEXT = "Reach Alice at alice@acme.com."


def test_requires_pii_labels() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'pii_labels' list"):
        GlinerPIIRedactor({})


def test_default_model_is_permissive_apache() -> None:
    assert GlinerPIIRedactor(_config())._model_id == "urchade/gliner_multi_pii-v1"


def test_pii_bounds_model_window_default_768(fake_gliner: type[_FakeGLiNER]) -> None:
    # W12 model-boundary guard: the provider sets model.config.max_len (the gliner truncation lever)
    # so a long chunk is not SILENTLY truncated past the mdeberta window. Default 768.
    provider = GlinerPIIRedactor(_config())
    list(provider.redact([_chunk(_TEXT)]))  # triggers the lazy load + window bound
    assert provider._model.config.max_len == 768


def test_pii_model_window_config_override(fake_gliner: type[_FakeGLiNER]) -> None:
    provider = GlinerPIIRedactor({**_config(), "max_len": 640})
    list(provider.redact([_chunk(_TEXT)]))
    assert provider._model.config.max_len == 640


def test_masks_detected_pii(fake_gliner: type[_FakeGLiNER]) -> None:
    a_start = _TEXT.index("Alice")
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "score": 0.95},
        {"start": e_start, "end": e_start + 14, "label": "email", "score": 0.9},
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_chunk(_TEXT)]))[0]
    assert isinstance(rec, ChunkRecord)
    # The clean content stays UNMASKED (detection still sees the entities); masked_content carries it.
    assert rec.content == _TEXT
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content
    assert "alice@acme.com" not in rec.masked_content
    assert "[PERSON]" in rec.masked_content
    assert "[EMAIL]" in rec.masked_content
    assert {s.pii_type for s in rec.pii_spans} == {"person", "email"}
    assert all(s.detector == "gliner-pii" for s in rec.pii_spans)


def test_span_offsets_are_chunk_local(fake_gliner: type[_FakeGLiNER]) -> None:
    # The Provider runs the model over the chunk's ``content``, so the model's offsets ARE
    # chunk-local offsets: the stamped span offset matches, and the slice round-trips to the raw
    # value (the coordinate system the spans use).
    doc_e_start = _TEXT.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": doc_e_start, "end": doc_e_start + 14, "label": "email", "score": 0.9}
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_chunk(_TEXT)]))[0]
    span = rec.pii_spans[0]
    assert span.char_start == doc_e_start
    assert _TEXT[span.char_start : span.char_end] == "alice@acme.com"


def test_threshold_drops_low_scoring_spans(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.2}]
    cfg = {"pii_labels": ["person"], "threshold": 0.5}
    rec = list(GlinerPIIRedactor(cfg).redact([_chunk("Alice waved at the crowd.")]))[0]
    assert len(rec.pii_spans) == 0


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
    public = list(provider.redact([_chunk(_TEXT, sensitivity="public")]))[0]
    internal = list(provider.redact([_chunk(_TEXT, sensitivity="internal")]))[0]
    # Same text, different disposition driven purely by Classification.sensitivity.
    assert public.masked_content is not None and "[PERSON]" in public.masked_content
    assert internal.masked_content == _TEXT  # tagged, not masked
    assert internal.pii_spans[0].action is RedactionAction.TAG


def test_span_page_resolves_through_page_map(fake_gliner: type[_FakeGLiNER]) -> None:
    assembly = PageMap.from_page_texts(["Nothing here.", "Reach alice@acme.com."])
    text = assembly.text
    e_start = text.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "score": 0.9}
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_chunk(text, page_map=assembly.page_map)]))[0]
    assert rec.pii_spans[0].page_start == 2


def test_raw_pii_never_stored(fake_gliner: type[_FakeGLiNER]) -> None:
    e_start = _TEXT.index("alice@acme.com")
    fake_gliner.entities = [
        {"start": e_start, "end": e_start + 14, "label": "email", "score": 0.9}
    ]
    rec = list(GlinerPIIRedactor(_config()).redact([_chunk(_TEXT)]))[0]
    assert "alice@acme.com" not in rec.pii_spans[0].model_dump_json()


def test_empty_chunk_skipped_without_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    rec = list(GlinerPIIRedactor(_config()).redact([_chunk("")]))[0]
    assert len(rec.pii_spans) == 0
    assert rec.masked_content == ""


def test_missing_gliner_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner", None)
    provider = GlinerPIIRedactor(_config())
    with pytest.raises(ImportError, match="pip install latence-pii-gliner"):
        list(provider.redact([_chunk(_TEXT)]))


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
        c.content: [{"start": i, "end": i + 9, "label": "person", "score": 0.9}]
        for i, c in enumerate(chunks)
    }


def test_batched_equals_unbatched_record_for_record(fake_gliner: type[_FakeGLiNER]) -> None:
    chunks = _distinct_chunks(7)
    fake_gliner.per_text = _per_text_person_map(chunks)
    cfg = {"pii_labels": ["person"]}

    unbatched = [
        r for c in chunks for r in GlinerPIIRedactor({**cfg, "batch_size": 1}).redact([c])
    ]
    batched = list(GlinerPIIRedactor({**cfg, "batch_size": 4}).redact(chunks))

    assert len(batched) == len(unbatched) == 7
    for b, u in zip(batched, unbatched, strict=True):
        assert b.record_id == u.record_id
        assert b.masked_content == u.masked_content
        assert [(s.char_start, s.char_end, s.pii_type) for s in b.pii_spans] == [
            (s.char_start, s.char_end, s.pii_type) for s in u.pii_spans
        ]
    # Each chunk masked its OWN name (a wrong-order batch would mask a different chunk's name).
    for i, rec in enumerate(batched):
        assert rec.masked_content is not None
        assert f"Person{i:03d}" not in rec.masked_content
        assert "[PERSON]" in rec.masked_content


def test_model_called_ceil_n_over_batch_size_times(fake_gliner: type[_FakeGLiNER]) -> None:
    chunks = _distinct_chunks(7)
    fake_gliner.per_text = _per_text_person_map(chunks)
    provider = GlinerPIIRedactor({"pii_labels": ["person"], "batch_size": 3})
    list(provider.redact(chunks))
    # 7 chunks at batch_size=3 => ceil(7/3)=3 forward passes, not 7.
    assert provider._model.batch_calls == 3


def test_skipped_chunks_need_no_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    # A skip-policy chunk + an empty chunk need no model call; a real chunk still batches once.
    chunks = _distinct_chunks(1)
    fake_gliner.per_text = _per_text_person_map(chunks)
    cfg = {
        "pii_labels": ["person"],
        "batch_size": 8,
        "policies": {"secret": {"skip": True}},
    }
    empty = _chunk("", record_id="empty#chunk:0", document_record_id="empty")
    secret = chunks[0].model_copy(
        update={"classification": Classification(language="en", sensitivity="secret")}
    )
    provider = GlinerPIIRedactor(cfg)
    out = list(provider.redact([empty, secret, chunks[0]]))
    assert len(out) == 3
    # Only the one real chunk triggered detection — a single batched forward pass.
    assert provider._model.batch_calls == 1


def test_batch_size_must_be_positive_int() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="batch_size must be >= 1"):
        GlinerPIIRedactor({"pii_labels": ["person"], "batch_size": 0})
