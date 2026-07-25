"""Gliner2FusedExtractor: fused zero-shot Entity+Relation Extraction over GLiNER2 (W9, ADR-0013).

The real ``gliner2`` stack (torch/transformers, mdeberta-v3) is intentionally NOT installed in the
dev/test env (ADR-0007/0016). These tests inject a tiny fake ``gliner2`` module whose model
returns the **REAL** GLiNER2 output shapes — ``extract_entities`` →
``{"entities": {label: [{text, confidence, start, end}]}}`` and ``extract_relations`` →
``{"relation_extraction": {rel: [{"head": {text,confidence,start,end}, "tail": {...}}]}}`` — so the
Provider's fused output, contract stamping, offset→page math, relation→mention resolution, and
typed-error-no-PII path are exercised deterministically and offline.

Anti-false-green (the PaddleOCR/relex lesson): the fake is FAITHFUL to gliner2's nested per-label /
per-relation dict shapes (NOT a flat list, NOT the ``gliner`` library tuple). Two teeth close the
trap: ``test_flat_list_misparse_fails_anchor`` (a provider mis-parsing the entity dict as a flat
list recovers ZERO spans) and ``test_offsets_round_trip_under_variable_offsets`` (a variable-offset
fixture proves the char offsets round-trip to the chunk EXACTLY — the recurring coordinate trap).
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import pytest
from latence_core.capability import FusedExtraction
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    InducedLabels,
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
    RelationMention,
)
from latence_extract_gliner2 import Gliner2FusedExtractor


def _chunk(
    content: str,
    *,
    char_start: int = 100,
    char_end: int | None = None,
    page_map: PageMap | None = None,
    offset_map: OffsetMap | None = None,
) -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=char_start,
        char_end=char_start + len(content) if char_end is None else char_end,
        page_start=2,
        page_end=2,
    )
    return ChunkRecord(
        record_id="doc1#chunk-0",
        document_record_id="doc1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        token_count=len(content.split()),
        page_map=page_map,
        offset_map=offset_map,
    )


class BackendCompilerFailed(RuntimeError):
    """A stand-in for ``torch._dynamo.exc.BackendCompilerFailed`` — matched by class NAME.

    ``torch.compile`` raises this (from its inductor/dynamo backend) when it cannot compile the model
    — LAZILY, on the first forward. The provider's ``is_compile_backend_failure`` is torch-free and
    matches on the class name / module chain, so a same-named local class exercises the exact guard
    without importing torch.
    """


class _FakeGLiNER2:
    """Stands in for ``gliner2.GLiNER2`` with the REAL nested ``extract_entities`` +
    ``extract_relations`` dict shapes (pod-validated, W9).

    ``entities`` is a flat list of ``{start, end, label, confidence}`` rows the test seeds;
    ``extract_entities`` reshapes it into the real ``{"entities": {label: [{text, confidence,
    start, end}]}}`` dict. ``relations`` is a flat list of ``{head, tail, relation}`` rows whose
    head/tail sub-dicts carry chunk-local ``start``/``end`` AND per-entity ``confidence`` (the
    coordinates the provider resolves endpoints by); ``extract_relations`` reshapes it into the real
    ``{"relation_extraction": {rel_type: [{"head": {...}, "tail": {...}}]}}`` dict. Per the REAL
    gliner2 1.3.2 shape, confidence lives ONLY on head/tail, NOT at the relation level — so the fake
    never injects a top-level relation score (the provider derives it from the endpoints).

    A provider that mis-parses either as a flat list (the ``gliner`` shape) or a tuple (the relex
    shape) recovers NOTHING — the false-green trap this fake closes.
    """

    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    model_id: str = ""
    #: capture the from_pretrained kwargs so a test can assert quantize/compile were threaded.
    last_kwargs: dict[str, Any] = {}
    #: capture the last ``max_len`` threaded into extract_entities / extract_relations (W12 guard).
    last_max_len: int | None = None
    #: force from_pretrained(quantize=…/compile=…) to raise, to exercise the guarded plain fallback.
    raise_on_optimized_load: bool = False
    raise_on_extract: bool = False

    #: when True, a model loaded with compile=True raises BackendCompilerFailed on its FIRST
    #: forward (torch.compile compiles lazily at inference, not at load) — the W9 fault the
    #: inference-time guard must degrade past. A compile=False (re)load never raises it.
    compiled_forward_raises: bool = False

    def __init__(self) -> None:
        self.extract_entities_calls = 0
        self.extract_relations_calls = 0
        self._compile_on = False

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeGLiNER2:
        # FAITHFUL to gliner2 1.3.2: from_pretrained ACCEPTS native quantize/compile kwargs. A
        # provider that passes them here loads clean; the guarded fallback re-calls plain on error.
        if cls.raise_on_optimized_load and ("quantize" in kwargs or "compile" in kwargs):
            msg = "simulated fp16/compile fault (torch/CUDA)"
            raise RuntimeError(msg)
        cls.last_kwargs = dict(kwargs)
        inst = cls()
        inst.model_id = model_id
        # torch.compile compiles LAZILY on the first forward: a compile=True load succeeds, then the
        # first extract raises. A reload with compile=False does not (the guard's fallback).
        inst._compile_on = bool(kwargs.get("compile"))
        return inst

    def _maybe_compile_fault(self) -> None:
        if type(self).compiled_forward_raises and self._compile_on:
            # Mimic torch._dynamo.exc.BackendCompilerFailed by class NAME (the provider's torch-free
            # is_compile_backend_failure matches on the name / module chain, never imports torch).
            raise BackendCompilerFailed("inductor backend cannot compile this model")

    def _rows(self, source: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
        return [r for r in source if r.get("end", 0) <= len(text)]

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,  # noqa: ARG002 - API parity; the fake always spans
        include_confidence: bool = False,  # noqa: ARG002 - API parity
        max_len: int = 768,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        self.extract_entities_calls += 1
        type(self).last_max_len = max_len  # W12: record the threaded model-window bound
        self._maybe_compile_fault()
        if type(self).raise_on_extract:
            raise RuntimeError(f"cuda oom while extracting {text!r}")  # must NOT leak into the error
        by_label: dict[str, list[dict[str, Any]]] = {}
        for e in self._rows(type(self).entities, text):
            label = e["label"]
            if label not in entity_types or e["confidence"] < threshold:
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

    def extract_relations(
        self,
        text: str,
        relation_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,  # noqa: ARG002 - API parity
        include_confidence: bool = False,  # noqa: ARG002 - API parity
        max_len: int = 768,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        self.extract_relations_calls += 1
        type(self).last_max_len = max_len  # W12: record the threaded model-window bound
        self._maybe_compile_fault()
        by_rel: dict[str, list[dict[str, Any]]] = {}
        for r in type(self).relations:
            rel_type = r["relation"]
            if rel_type not in relation_types or r["head"]["end"] > len(text):
                continue
            # REAL shape: confidence ONLY on head/tail, no relation-level score. Copy them verbatim.
            by_rel.setdefault(rel_type, []).append(
                {"head": dict(r["head"]), "tail": dict(r["tail"])}
            )
        return {"relation_extraction": by_rel}


@pytest.fixture()
def fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER2]:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _FakeGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    _FakeGLiNER2.entities = []
    _FakeGLiNER2.relations = []
    _FakeGLiNER2.last_kwargs = {}
    _FakeGLiNER2.last_max_len = None
    _FakeGLiNER2.raise_on_optimized_load = False
    _FakeGLiNER2.raise_on_extract = False
    _FakeGLiNER2.compiled_forward_raises = False
    return _FakeGLiNER2


def _config() -> dict[str, Any]:
    return {"labels": ["person", "organization"], "relation_labels": ["works_for"]}


def test_max_len_threaded_into_extract_calls(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # W12 model-boundary guard: the provider threads max_len (default 768) into gliner2's native
    # extract_entities / extract_relations so the mdeberta window is bounded, no silent truncation.
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor(_config()).extract([_chunk("Alice waved.")])
    assert fake_gliner2.last_max_len == 768


def test_max_len_config_override_threaded(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor({**_config(), "max_len": 600}).extract([_chunk("Alice waved.")])
    assert fake_gliner2.last_max_len == 600


# --- config validation + profile ------------------------------------------------------------


def test_requires_entity_and_relation_labels() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        Gliner2FusedExtractor({"relation_labels": ["works_for"]})
    with pytest.raises(ValueError, match="requires a non-empty 'relation_labels' list"):
        Gliner2FusedExtractor({"labels": ["person"]})


def test_default_model_is_permissive_apache() -> None:
    assert Gliner2FusedExtractor(_config())._model_id == "fastino/gliner2-multi-v1"


def test_profile_is_verified_apache_either() -> None:
    prof = Gliner2FusedExtractor.profile()
    assert prof.compute == "either"
    assert prof.license == "Apache-2.0"
    assert prof.license_verified is True
    assert prof.license_weights == "Apache-2.0"
    assert prof.license_code == "Apache-2.0"
    assert prof.model_id == "fastino/gliner2-multi-v1"
    assert prof.batch is True


def test_invalid_quantize_knob_is_config_error() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="invalid quantize"):
        Gliner2FusedExtractor({**_config(), "quantize": "sometimes"})
    with pytest.raises(ConfigError, match="invalid compile"):
        Gliner2FusedExtractor({**_config(), "compile": "maybe"})


# --- fused extraction: the REAL shapes map to the contract ----------------------------------


def test_fused_emits_mentions_and_relations(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # "Alice works for Acme Corp." — Alice [0,5), Acme Corp [16,25) in chunk-local text.
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.95},
        {"start": 16, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            # REAL shape: confidence on head/tail, NO relation-level score. The provider derives the
            # relation confidence as the min (weakest link) of the two endpoints: min(0.95, 0.9)=0.9.
            "head": {"start": 0, "end": 5, "text": "Alice", "confidence": 0.95},
            "tail": {"start": 16, "end": 25, "text": "Acme Corp", "confidence": 0.9},
            "relation": "works_for",
        }
    ]
    chunk = _chunk("Alice works for Acme Corp.", char_start=100)
    result = Gliner2FusedExtractor(_config()).extract_fused([chunk])
    assert isinstance(result, FusedExtraction)
    assert [m.text for m in result.mentions] == ["Alice", "Acme Corp"]
    assert all(isinstance(m, EntityMention) for m in result.mentions)
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert isinstance(rel, RelationMention)
    assert rel.label == "works_for"
    ids = {m.text: m.record_id for m in result.mentions}
    assert rel.head_mention_id == ids["Alice"]
    assert rel.tail_mention_id == ids["Acme Corp"]
    assert rel.confidence == 0.9  # min(head 0.95, tail 0.9) — the weakest-link endpoint score
    # Chunk-local offsets shifted into the parent-document coordinate system.
    assert result.mentions[0].provenance.char_start == 100
    assert rel.provenance.char_start == 100
    assert rel.provenance.char_end == 125
    assert rel.provenance.page_start == 2


def test_degenerate_zero_length_span_skipped_not_crash(
    fake_gliner2: type[_FakeGLiNER2], caplog: pytest.LogCaptureFixture
) -> None:
    """dogfood-2b-process: a DEGENERATE zero-length span (``start == end``) must be SKIPPED, not
    crash the stage on the EntityMention >=1-char contract.

    GLiNER2 can return a span whose resolved surface text is empty; constructing an EntityMention
    with ``text=""`` raises pydantic ``string_too_short`` — which killed a full pipeline run at the
    extract stage. Seed a mix (two valid spans + one degenerate ``[10, 10)`` span among them) and
    assert: (a) no crash / no ValidationError, (b) the degenerate span produces NO mention, (c) the
    valid spans still emit their mentions unchanged, (d) the skip is counted + logged (ADR-0034).

    PRE-FIX EVIDENCE: on the unpatched provider this test FAILS with
    ``pydantic_core.ValidationError: EntityMention.text — String should have at least 1 character``
    raised from ``_build_records`` — exactly the pod crash. It passes only once the guard skips it.
    """
    # "Alice works for Acme Corp." — Alice [0,5), a DEGENERATE span at [10, 10) (empty slice), and
    # Acme Corp [16,25). All above threshold so only the empty-text guard can drop the middle one.
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.95},
        {"start": 10, "end": 10, "label": "person", "confidence": 0.9},  # degenerate: text == ""
        {"start": 16, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    chunk = _chunk("Alice works for Acme Corp.", char_start=100)
    with caplog.at_level("INFO", logger="latence.provider.gliner2_fused"):
        result = Gliner2FusedExtractor(_config()).extract_fused([chunk])
    # (a) no crash + (b/c) the degenerate span is absent; both valid spans survive UNCHANGED.
    assert [m.text for m in result.mentions] == ["Alice", "Acme Corp"]
    assert all(len(m.text) >= 1 for m in result.mentions)
    assert result.mentions[0].provenance.char_start == 100  # valid span offsets unaffected
    # (d) the skip is COUNTED and surfaced (not silent — ADR-0034).
    skip_logs = [r for r in caplog.records if "degenerate zero-length entity span" in r.message]
    assert len(skip_logs) == 1
    assert "1 degenerate" in skip_logs[0].getMessage()


def test_offsets_round_trip_under_variable_offsets(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """THE coordinate-trap tooth: entity + relation char offsets round-trip to the chunk EXACTLY.

    A VARIABLE-offset fixture (three names at three DISTINCT, non-trivial chunk-local starts, on a
    chunk that itself starts at a non-zero document offset) proves the char offsets are not merely
    a constant shift: every stamped mention slice equals its text, and the relation covering span
    slices back to the head..tail span — so an off-by-one / wrong-base coordinate bug FAILS here.
    """
    content = "   Alice   met   Globex   and   Initech."
    a = content.index("Alice")
    g = content.index("Globex")
    i = content.index("Initech")
    fake_gliner2.entities = [
        {"start": a, "end": a + 5, "label": "person", "confidence": 0.9},
        {"start": g, "end": g + 6, "label": "organization", "confidence": 0.9},
        {"start": i, "end": i + 7, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": a, "end": a + 5},
            "tail": {"start": i, "end": i + 7},
            "relation": "works_for",
        }
    ]
    chunk = _chunk(content, char_start=100)
    result = Gliner2FusedExtractor(_config()).extract_fused([chunk])
    # Every mention's DOCUMENT-offset slice (through the whole document coordinate system) equals
    # its text. The chunk starts at doc offset 100 and has no offset_map, so doc = 100 + local.
    doc_text = " " * 100 + content
    for m in result.mentions:
        assert doc_text[m.provenance.char_start : m.provenance.char_end] == m.text
    # The three distinct starts are preserved (not collapsed to one shift).
    starts = sorted(m.provenance.char_start or 0 for m in result.mentions)
    assert starts == [100 + a, 100 + g, 100 + i]
    # The relation covering span slices back to head..tail (Alice ... Initech).
    rel = result.relations[0]
    assert doc_text[rel.provenance.char_start : rel.provenance.char_end] == content[a : i + 7]


def test_offsets_resolve_through_offset_map_past_stripped_markup(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    # Original "**Note:** Alice joined Globex." strips to "Note: Alice joined Globex.": the stripped
    # ``**`` shifts every later mention right in the original, so char_start+local is a lower bound.
    # The carried offset_map corrects it (ADR-0031) — the exact relex round-trip, now for gliner2.
    original = "**Note:** Alice joined Globex."
    content = "Note: Alice joined Globex."
    offset_map = OffsetMap(
        segments=[
            OffsetSegment(local_start=0, doc_start=2),
            OffsetSegment(local_start=5, doc_start=9),
        ],
        length=len(content),
    )
    fake_gliner2.entities = [
        {"start": 6, "end": 11, "label": "person", "confidence": 0.9},
        {"start": 19, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": 6, "end": 11},
            "tail": {"start": 19, "end": 25},
            "relation": "works_for",
        }
    ]
    chunk = _chunk(content, char_start=2, char_end=len(original), offset_map=offset_map)
    result = Gliner2FusedExtractor(_config()).extract_fused([chunk])
    by_text = {m.text: m for m in result.mentions}
    assert set(by_text) == {"Alice", "Globex"}
    for m in result.mentions:
        assert original[m.provenance.char_start : m.provenance.char_end] == m.text
    assert by_text["Alice"].provenance.char_start == 10  # not 2 + 6 = 8
    assert by_text["Globex"].provenance.char_start == 23  # not 2 + 19 = 21
    rel = result.relations[0]
    assert (rel.provenance.char_start, rel.provenance.char_end) == (10, 29)
    assert original[rel.provenance.char_start : rel.provenance.char_end] == "Alice joined Globex"


def test_relation_endpoint_offsets_shift_to_document_pages(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    assembly = PageMap.from_page_texts(["Alice is here.", "She joined Globex."])
    text = assembly.text
    g_start = text.find("Globex")
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.9},
        {"start": g_start, "end": g_start + 6, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": g_start, "end": g_start + 6},
            "relation": "works_for",
        }
    ]
    chunk = _chunk(text, char_start=0, page_map=assembly.page_map)
    result = Gliner2FusedExtractor(_config()).extract_fused([chunk])
    rel = result.relations[0]
    assert rel.provenance.page_start == 1
    assert rel.provenance.page_end == 2


def test_empty_chunk_skipped_without_model_call(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    result = Gliner2FusedExtractor(_config()).extract_fused([_chunk("")])
    assert result.mentions == []
    assert result.relations == []


def test_relation_with_unresolvable_endpoint_dropped(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.9},
        {"start": 16, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 999, "end": 1004},  # no such entity
            "relation": "works_for",
        }
    ]
    result = Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])
    assert len(result.mentions) == 2
    assert result.relations == []


def test_extract_half_returns_only_mentions(fake_gliner2: type[_FakeGLiNER2]) -> None:
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    fake_gliner2.relations = []
    mentions = Gliner2FusedExtractor(_config()).extract([_chunk("Alice waved.")])
    assert [m.text for m in mentions] == ["Alice"]


# --- W4 induced-label seam (drop-in with relex) ---------------------------------------------


class _CapturingGLiNER2(_FakeGLiNER2):
    """Records the (entity_types, relation_types) each call is asked for (the W4 slot capture)."""

    captured_entities: list[list[str]] = []
    captured_relations: list[list[str]] = []

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,
        include_confidence: bool = False,
        max_len: int = 768,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        type(self).captured_entities.append(list(entity_types))
        return super().extract_entities(
            text, entity_types, threshold, include_spans, include_confidence, max_len
        )

    def extract_relations(
        self,
        text: str,
        relation_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,
        include_confidence: bool = False,
        max_len: int = 768,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        type(self).captured_relations.append(list(relation_types))
        return super().extract_relations(
            text, relation_types, threshold, include_spans, include_confidence, max_len
        )


@pytest.fixture()
def capturing_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_CapturingGLiNER2]:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _CapturingGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    _CapturingGLiNER2.entities = []
    _CapturingGLiNER2.relations = []
    _CapturingGLiNER2.captured_entities = []
    _CapturingGLiNER2.captured_relations = []
    _CapturingGLiNER2.raise_on_optimized_load = False
    _CapturingGLiNER2.raise_on_extract = False
    return _CapturingGLiNER2


def _induced_chunk(
    content: str = "Alice works for Acme Corp.",
    *,
    entity_types: list[str] | None = None,
    relation_types: list[str] | None = None,
) -> ChunkRecord:
    return _chunk(content).model_copy(
        update={
            "induced_labels": InducedLabels(
                model_id="test-model",
                entity_types=["location"] if entity_types is None else entity_types,
                relation_types=["located_in"] if relation_types is None else relation_types,
                pii_types=[],
            )
        }
    )


def test_induced_labels_union_onto_both_slots_under_universal_floor(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    # The OPT-IN historical policy: config floor ∪ induced, on every chunk (T2 `label_floor`).
    Gliner2FusedExtractor({**_config(), "label_floor": "universal"}).extract_fused(
        [_induced_chunk()]
    )
    assert capturing_gliner2.captured_entities == [["person", "organization", "location"]]
    assert capturing_gliner2.captured_relations == [["works_for", "located_in"]]


def test_no_induced_labels_asks_for_config_floor_only(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])
    assert capturing_gliner2.captured_entities == [["person", "organization"]]
    assert capturing_gliner2.captured_relations == [["works_for"]]


# --- T2: a chunk is extracted against ITS OWN induced labels, not a global floor -------------


def test_per_chunk_floor_asks_only_for_the_chunks_own_induced_labels(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    """T2 AC1: the label set handed to the model IS the chunk's own induced set.

    Pre-change this asked for ``config ∪ induced`` on every chunk — with corpus-granularity
    induction that is the whole ~40-type corpus schema stamped onto every chunk, which is the
    zero-shot precision tax (and the label dimension of the OOM'd span-scoring tensor) the redesign
    removes. Asserted on the labels ACTUALLY passed to the model, not on a config flag.
    """
    Gliner2FusedExtractor(_config()).extract_fused([_induced_chunk()])
    assert capturing_gliner2.captured_entities == [["location"]]
    assert capturing_gliner2.captured_relations == [["located_in"]]


def test_per_chunk_floor_gives_each_chunk_a_different_label_set(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    """Two chunks with different induced types are scored against different, small sets.

    The corpus-union failure mode is invisible on ONE chunk (the union of one chunk's labels is
    that chunk's labels); it only shows up when two chunks disagree. Here each chunk must see its
    own types and NEITHER may see the other's — the property "no corpus-level operation inflates a
    chunk's label set" reduces to.
    """
    chunks = [
        _induced_chunk(entity_types=["location"], relation_types=["located_in"]),
        _induced_chunk(entity_types=["product"], relation_types=["sells"]),
    ]
    Gliner2FusedExtractor(_config()).extract_fused(chunks)
    assert capturing_gliner2.captured_entities == [["location"], ["product"]]
    assert capturing_gliner2.captured_relations == [["located_in"], ["sells"]]


def test_per_chunk_floor_falls_back_to_config_when_a_slot_induced_nothing(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    """An empty induced slot falls back to the config floor — never an EMPTY label prompt.

    A chunk whose induction returned no relation types must not be scored against zero relation
    labels (which would silently extract nothing); the config list is the recall net for exactly
    that case. Per slot: the entity slot induced, so it keeps its own set.
    """
    chunk = _induced_chunk(entity_types=["location"], relation_types=[])
    Gliner2FusedExtractor(_config()).extract_fused([chunk])
    assert capturing_gliner2.captured_entities == [["location"]]
    assert capturing_gliner2.captured_relations == [["works_for"]]


def test_predefined_only_mode_runs_the_fixed_labels_per_chunk(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    """T2 AC4 (extract half): predefined-only ⇒ no induced labels ⇒ exactly the fixed set per chunk.

    In predefined-only mode the ``induce`` Stage is skipped entirely (T1), so every chunk reaches
    the extractor with ``induced_labels is None``. Each chunk must then be scored against exactly
    the operator's fixed lists — under EITHER floor policy, so flipping the knob cannot change a
    predefined-only run.
    """
    chunks = [_chunk("Alice works for Acme Corp."), _chunk("Bob works for Beta Ltd.")]
    for floor in ("per_chunk", "universal"):
        capturing_gliner2.captured_entities = []
        capturing_gliner2.captured_relations = []
        Gliner2FusedExtractor({**_config(), "label_floor": floor}).extract_fused(chunks)
        assert capturing_gliner2.captured_entities == [
            ["person", "organization"],
            ["person", "organization"],
        ]
        assert capturing_gliner2.captured_relations == [["works_for"], ["works_for"]]


def test_per_chunk_output_equals_running_that_chunks_labels_directly(
    capturing_gliner2: type[_CapturingGLiNER2],
) -> None:
    """T2 AC5: the emitted records equal a run configured with that chunk's own labels directly.

    The scope reduction is the ONLY intended change: extracting a chunk whose induced set is
    {location, located_in} must produce exactly what a Provider configured with
    ``labels=[location] / relation_labels=[located_in]`` produces on the same, uninduced chunk.
    """
    content = "Alice works for Acme Corp."
    capturing_gliner2.entities = [
        {"start": 0, "end": 5, "label": "location", "confidence": 0.95},
        {"start": 16, "end": 25, "label": "location", "confidence": 0.9},
    ]
    capturing_gliner2.relations = [
        {
            "head": {"start": 0, "end": 5, "text": "Alice", "confidence": 0.95},
            "tail": {"start": 16, "end": 25, "text": "Acme Corp", "confidence": 0.9},
            "relation": "located_in",
        }
    ]
    induced = Gliner2FusedExtractor(_config()).extract_fused(
        [_induced_chunk(content, entity_types=["location"], relation_types=["located_in"])]
    )
    direct = Gliner2FusedExtractor(
        {"labels": ["location"], "relation_labels": ["located_in"]}
    ).extract_fused([_chunk(content)])
    assert [m.model_dump() for m in induced.mentions] == [
        m.model_dump() for m in direct.mentions
    ]
    assert [r.model_dump() for r in induced.relations] == [
        r.model_dump() for r in direct.relations
    ]
    assert induced.mentions and induced.relations  # the comparison is not vacuously empty


def test_unknown_label_floor_value_is_a_config_error() -> None:
    """A typo'd policy is a loud ConfigError, never a silently ignored key (ADR-0034)."""
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="label_floor must be one of"):
        Gliner2FusedExtractor({**_config(), "label_floor": "per-chunk"})


# --- anti-false-green teeth (the paddle/relex lesson) ----------------------------------------


def test_flat_list_misparse_fails_anchor(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """ANCHOR: the REAL nested per-label / per-relation dict shape is required, not a flat list.

    Proves the fake is faithful to gliner2's ``{"entities": {label: [rows]}}`` /
    ``{"relation_extraction": {rel: [rows]}}`` shapes and that a provider treating either as a flat
    list (the ``gliner``-library shape) recovers NOTHING — iterating the inner dict yields the label
    STRINGS (keys), not rows, so no span/endpoint is recoverable. The FIXED provider parses the
    nesting and DOES recover both.
    """
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.95},
        {"start": 16, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 16, "end": 25},
            "relation": "works_for",
        }
    ]
    model = fake_gliner2.from_pretrained("x")
    ents = model.extract_entities("Alice works for Acme Corp.", ["person", "organization"])
    rels = model.extract_relations("Alice works for Acme Corp.", ["works_for"])
    # The correct nested shapes the Provider relies on:
    assert isinstance(ents["entities"], dict)
    assert ents["entities"]["person"][0]["start"] == 0
    assert isinstance(rels["relation_extraction"], dict)
    assert rels["relation_extraction"]["works_for"][0]["head"]["start"] == 0
    # A naive flat-list mis-parse yields label/relation-type keys (str), never rows -> zero records.
    assert list(ents["entities"]) == ["person", "organization"]
    assert list(rels["relation_extraction"]) == ["works_for"]
    assert all(isinstance(k, str) for k in list(ents["entities"]) + list(rels["relation_extraction"]))

    # End-to-end: the FAITHFUL provider DOES recover mentions + the relation from the nesting.
    result = Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])
    assert len(result.mentions) == 2
    assert len(result.relations) == 1


def test_emitted_relations_have_real_endpoints(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # Provider-level GUARD: every emitted RelationMention resolves to head/tail record_ids that
    # EXIST among the emitted mentions with a non-empty label — a silent mis-parse FAILS here.
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.95},
        {"start": 16, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 16, "end": 25},
            "relation": "works_for",
        }
    ]
    result = Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])
    mention_ids = {m.record_id for m in result.mentions}
    assert result.relations
    for rel in result.relations:
        assert rel.head_mention_id in mention_ids
        assert rel.tail_mention_id in mention_ids
        assert rel.head_mention_id != rel.tail_mention_id
        assert rel.label


def test_native_fault_becomes_typed_error_no_pii_leak(fake_gliner2: type[_FakeGLiNER2]) -> None:
    from latence_core.errors import ProviderError

    fake_gliner2.raise_on_extract = True
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    text = "Reach Alice at alice@acme.com."
    with pytest.raises(ProviderError) as excinfo:
        Gliner2FusedExtractor(_config()).extract_fused([_chunk(text)])
    message = str(excinfo.value)
    assert "fused_entity_relation.gliner2" in message
    assert "extract_entities" in message
    assert "alice@acme.com" not in message
    assert "Alice" not in message


def test_missing_gliner2_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner2", None)
    extractor = Gliner2FusedExtractor(_config())
    with pytest.raises(ImportError, match="pip install latence-extract-gliner2"):
        extractor.extract_fused([_chunk("Alice works for Acme Corp.")])


# --- W9 quantize/compile guarded load --------------------------------------------------------


def test_optimized_load_fault_degrades_to_plain_load(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """A quantize/compile fault degrades to a plain load (never a crash) at identical output.

    On a CPU host quantize/compile resolve to False (no kwargs), so force cuda via the config knobs
    AND make the optimized load raise — the guard must retry plain and still extract identically.
    """
    fake_gliner2.raise_on_optimized_load = True
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    cfg = {**_config(), "quantize": True, "compile": True}
    result = Gliner2FusedExtractor(cfg).extract_fused([_chunk("Alice waved.")])
    # The plain fallback loaded and produced the identical mention (no crash).
    assert [m.text for m in result.mentions] == ["Alice"]


def test_cpu_default_loads_plain_no_optimized_kwargs(fake_gliner2: type[_FakeGLiNER2]) -> None:
    # On the CPU-only test host (LATENCE_CUDA=0 in CI) quantize/compile 'auto' resolve to False, so
    # from_pretrained is called PLAIN (no quantize/compile kwargs) — the determinism-safe default.
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice waved.")])
    assert "quantize" not in fake_gliner2.last_kwargs
    assert "compile" not in fake_gliner2.last_kwargs


def test_inference_time_compile_fault_degrades_to_uncompiled(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    """THE inference-time compile guard (pod-found): torch.compile compiles LAZILY on the first
    forward, so a backend that cannot compile the model raises ``BackendCompilerFailed`` INSIDE
    ``extract_entities`` — NOT at load. The provider must catch it ONCE, re-load the UNCOMPILED
    model (quantize kept), and still return correct records — never a ProviderError crash.
    """
    fake_gliner2.compiled_forward_raises = True   # compile=True load's first forward raises
    fake_gliner2.entities = [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.95},
        {"start": 16, "end": 25, "label": "organization", "confidence": 0.9},
    ]
    fake_gliner2.relations = [
        {
            "head": {"start": 0, "end": 5, "confidence": 0.95},
            "tail": {"start": 16, "end": 25, "confidence": 0.9},
            "relation": "works_for",
        }
    ]
    # Force compile on (cuda-style) so the first forward hits the lazy compile fault.
    cfg = {**_config(), "quantize": True, "compile": True}
    provider = Gliner2FusedExtractor(cfg)
    result = provider.extract_fused([_chunk("Alice works for Acme Corp.")])
    # Degraded to the uncompiled model + still returned the correct records (no crash).
    assert [m.text for m in result.mentions] == ["Alice", "Acme Corp"]
    assert len(result.relations) == 1
    assert result.relations[0].label == "works_for"
    # The fallback ran (uncompiled model loaded) and is latched so it runs at most once.
    assert provider._compile_disabled is True
    # The reloaded model was loaded with compile=False (quantize kept) — the last from_pretrained.
    assert fake_gliner2.last_kwargs.get("compile") is False
    assert fake_gliner2.last_kwargs.get("quantize") is True


def test_inference_time_compile_fault_runs_fallback_once_across_chunks(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    # Over MANY chunks the fallback re-load happens at most once; every chunk still extracts.
    fake_gliner2.compiled_forward_raises = True
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    cfg = {**_config(), "quantize": True, "compile": True}
    provider = Gliner2FusedExtractor(cfg)
    chunks = [_chunk("Alice waved.") for _ in range(4)]
    result = provider.extract_fused(chunks)
    assert len(result.mentions) == 4  # every chunk produced its mention
    assert provider._compile_disabled is True


# --- W9-scaling: the Blackwell hang fixes (compile-off default, thread cap, batch progress, ------
# --- fp16 cuda placement). CPU-testable via LATENCE_CUDA + a fake torch / a fake that records. ---


def test_compile_defaults_off_even_on_cuda(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """W9-scaling fix #1: compile is defaulted OFF (opt-in) EVEN on cuda (recompile-thrash guard).

    Forces the cuda load path (LATENCE_CUDA=1): quantize 'auto' still resolves True, but compile is
    forced False by the provider default — so from_pretrained gets compile=False, NOT True. This is
    the whole fix for the compile=True recompile-thrash hang: the safe default, not the crash latch.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice waved.")])
    assert fake_gliner2.last_kwargs.get("quantize") is True
    assert fake_gliner2.last_kwargs.get("compile") is False


def test_compile_opt_in_still_enables_on_cuda(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    # An operator who explicitly asks for compile still gets it (the knob is opt-in, not removed).
    monkeypatch.setenv("LATENCE_CUDA", "1")
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor({**_config(), "compile": True}).extract_fused([_chunk("Alice waved.")])
    assert fake_gliner2.last_kwargs.get("compile") is True


def _fake_torch_recording(calls: list[int]) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.set_num_threads = calls.append  # type: ignore[attr-defined]
    # Real torch always exposes ``inference_mode`` (the no-grad forward context the provider now
    # runs the extract/relation forward inside, dogfood-2b-process); a faithful stand-in must too.
    module.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
    return module


def test_intra_op_threads_capped_at_load(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """W9-scaling fix #2: torch intra-op threads are capped at load (default 8), clamped to cores.

    Anti-false-green: a fake ``torch`` records the ``set_num_threads`` value, so a provider that
    silently skipped the cap (the oversubscription hang) FAILS here.
    """
    import os

    calls: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_recording(calls))
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice waved.")])
    assert calls == [min(8, os.cpu_count() or 8)]


def test_intra_op_threads_config_override(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_recording(calls))
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor({**_config(), "intra_op_threads": 4}).extract_fused([_chunk("Alice waved.")])
    assert calls == [4]


def test_intra_op_threads_disabled_leaves_torch_default(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    # intra_op_threads <= 0 is the explicit opt-out: no set_num_threads call (torch default kept).
    calls: list[int] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_recording(calls))
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor({**_config(), "intra_op_threads": 0}).extract_fused([_chunk("Alice waved.")])
    assert calls == []


def test_invalid_intra_op_threads_is_config_error() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="invalid intra_op_threads"):
        Gliner2FusedExtractor({**_config(), "intra_op_threads": "lots"})


def test_batch_size_clamped_to_max() -> None:
    # W9-scaling fix #4: a large configured batch_size is CLAMPED so progress is always logged
    # often enough that a big corpus never runs silently ("appears hung").
    assert Gliner2FusedExtractor({**_config(), "batch_size": 1000})._batch_size == 64
    assert Gliner2FusedExtractor({**_config(), "batch_size": 8})._batch_size == 8


def test_incremental_batch_progress_recorded(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """W9-scaling fix #4: extraction streams in bounded batches with per-batch progress.

    5 chunks at batch_size=2 → 3 batches; the cumulative progress trail is (2, 4, 5) chunks, so a
    provider that ran the whole corpus as one opaque pass (no incremental progress) FAILS here.
    """
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    chunks = [_chunk("Alice waved.") for _ in range(5)]
    provider = Gliner2FusedExtractor({**_config(), "batch_size": 2})
    result = provider.extract_fused(chunks)
    assert len(result.mentions) == 5
    assert [p[0] for p in provider._progress] == [2, 4, 5]  # cumulative chunks per batch
    assert [p[1] for p in provider._progress] == [2, 4, 5]  # cumulative mentions (1/chunk)


def test_below_min_chars_chunks_not_counted_in_batches(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    # Empty/short chunks are skipped without a model call and do NOT count toward a batch — the
    # progress cadence measures real extraction work, not skipped empties.
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    chunks = [_chunk("Alice waved."), _chunk("   "), _chunk("Bob ran now.")]
    provider = Gliner2FusedExtractor({**_config(), "batch_size": 16})
    provider.extract_fused(chunks)
    assert len(provider._progress) == 1
    assert provider._progress[0][0] == 2  # only the 2 non-empty chunks processed


class _PlacingGLiNER2(_FakeGLiNER2):
    """Records ``.to(target)`` calls + exposes a ``device`` so the fp16 cuda-placement fix is testable."""

    to_calls: list[str] = []
    device: str = "cpu"

    def to(self, target: str) -> _PlacingGLiNER2:
        type(self).to_calls.append(target)
        type(self).device = target
        return self


@pytest.fixture()
def placing_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_PlacingGLiNER2]:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _PlacingGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    _PlacingGLiNER2.entities = []
    _PlacingGLiNER2.relations = []
    _PlacingGLiNER2.last_kwargs = {}
    _PlacingGLiNER2.last_max_len = None
    _PlacingGLiNER2.raise_on_optimized_load = False
    _PlacingGLiNER2.raise_on_extract = False
    _PlacingGLiNER2.compiled_forward_raises = False
    _PlacingGLiNER2.to_calls = []
    _PlacingGLiNER2.device = "cpu"
    return _PlacingGLiNER2


def test_fp16_model_moved_to_cuda_when_gpu_present(
    placing_gliner2: type[_PlacingGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """W9-scaling fix #3: when a GPU is present the (fp16) model is explicitly moved to cuda.

    The defence against the fp16-eager GPU-idle hang (a fp16-converted model that silently stayed on
    CPU → a CPU-bound spin). A provider that trusted from_pretrained to place it — and left it on
    CPU — FAILS here.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    placing_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice waved.")])
    assert placing_gliner2.to_calls == ["cuda"]
    assert placing_gliner2.device == "cuda"


def test_no_cuda_no_move(
    placing_gliner2: type[_PlacingGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a CPU-only host (no GPU) the model is NOT moved — placement is a cuda-only concern.
    monkeypatch.setenv("LATENCE_CUDA", "0")
    placing_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    Gliner2FusedExtractor(_config()).extract_fused([_chunk("Alice waved.")])
    assert placing_gliner2.to_calls == []
