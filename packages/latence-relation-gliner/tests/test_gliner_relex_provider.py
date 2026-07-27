"""GlinerRelexExtractor: fused zero-shot Entity+Relation Extraction (S5, ADR-0013).

The real ``gliner`` stack (torch/transformers) is intentionally NOT installed in the
dev/test env (ADR-0007/0016). These tests inject a tiny fake ``gliner`` module whose
model returns fixed chunk-local entity + relation spans, so the Provider's fused
output, contract stamping, offset→page math, and relation→mention resolution are
exercised deterministically and offline. One test covers the missing-dependency error
path and the zero-shot config validation (both entity AND relation labels required).

W1-relex FIX: the fake is faithful to the REAL ``predict_relations`` TUPLE contract (there is
NO ``predict_with_relations`` — that method never existed on the real checkpoint; the old fake
mirrored a non-existent API, the exact PaddleOCR false-green class). Two teeth close the trap:
``test_dict_parse_path_fails_against_tuple_api`` (a provider that ``.get(...)``s the tuple gets
NO relations — the anti-false-green anchor) and ``test_emitted_relations_have_real_endpoints``
(the provider-level guard: every emitted RelationMention resolves to real emitted mentions with a
non-empty label, so a silent drop-all-relations regression FAILS).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    PageSlice,
    PageSpan,
    Provenance,
    RelationMention,
)
from latence_core.errors import ConfigError, ProviderError
from latence_relation_gliner import GlinerRelexExtractor

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        # v19: the chunk carries its OWN page spans, cut from the parent document's map.
        # Absent a map the fixture is a one-page document, so the slice is that page.
        page_slice=(
            PageSlice.for_span(
                page_map, char_start, char_start + len(content) if char_end is None else char_end
            )
            if page_map is not None
            else PageSlice(
                pages=[
                    PageSpan(
                        page_number=2,
                        char_start=char_start,
                        char_end=char_start + len(content) if char_end is None else char_end,
                    )
                ],
                total_chars=char_start + len(content) if char_end is None else char_end,
            )
        ),
        offset_map=offset_map,
    )


# The dtype names ``load_then_cast`` may hand ``.to()`` on a torch-free host (it passes the NAME
# string when torch is absent) — distinguished from a device string so the fake routes like torch.
_TORCH_DTYPE_NAMES = frozenset({"bfloat16", "float16", "float32", "float64"})
# The device types torch accepts at the start of a device string; anything else is a RuntimeError.
_TORCH_DEVICE_TYPES = frozenset({"cpu", "cuda", "mps", "xpu", "meta"})


@dataclass(frozen=True)
class _FakeParam:
    """A stand-in for a ``torch.nn.Parameter``: the only thing under test is its ``device``."""

    device: str


class _FakeGLiNER:
    """Stands in for gliner.GLiNER with the REAL ``predict_relations`` TUPLE API.

    Faithful to ``knowledgator/gliner-relex-multi-v1.0`` (W1-relex FIX, introspected + run
    on the pod): the model exposes ``predict_relations(text, labels, relations, threshold=...)``
    returning a ``(entities, relations)`` **TUPLE** — NOT a dict, and there is NO
    ``predict_with_relations`` (that method never existed; a provider calling it AttributeErrors).
    Entity dicts carry ``start``/``end``/``text``/``label``/``score``; relation dicts carry
    ``head``/``tail`` sub-dicts (``start``/``end``/``text``/``type``/``entity_idx``) + ``relation``
    + ``score``. A provider that ``.get(...)``s this tuple or calls ``predict_with_relations``
    FAILS — the false-green trap this fix closes.
    """

    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    #: optional per-text maps so batched inputs get DISTINCT entities/relations per text — the
    #: off-by-one / wrong-order equivalence trap needs distinct per-chunk results.
    per_text_entities: dict[str, list[dict[str, Any]]] | None = None
    per_text_relations: dict[str, list[dict[str, Any]]] | None = None
    model_id: str = ""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.cast_dtype: Any = None  # W8: dtype applied via .to(), None until cast
        #: FAITHFUL to the real GLiNER model: a ``config`` with a ``max_len`` the library truncates
        #: to. The provider's W12 model-boundary guard sets this so a test can assert the bound.
        self.config = types.SimpleNamespace(max_len=384)
        #: FAITHFUL to ``GLiNER.from_pretrained`` (a ``torch.nn.Module`` + PyTorchModelHubMixin):
        #: the weights land on **CPU** — the library does NOT auto-place on cuda (#191). Only an
        #: explicit ``.to(device)`` moves them, exactly like the real ``nn.Module``.
        self._param = _FakeParam("cpu")

    def parameters(self) -> Iterator[_FakeParam]:
        """The real model's parameter iterator — the ground truth for "where does it run"."""
        yield self._param

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

    def to(self, arg: Any) -> _FakeGLiNER:
        """``nn.Module.to`` — one call takes EITHER a device or a dtype (both are real usages).

        FAITHFUL to torch: a device string MOVES the parameters (so ``parameters()[0].device``
        changes); a dtype casts them (W8: the plain-gliner dtype path is ``.to(bf16)`` on cuda);
        an unknown device string is a ``RuntimeError`` exactly as ``torch.device("gpu")`` is, so a
        typo can never be mistaken for a successful placement. Returns self, like a real module.
        """
        if isinstance(arg, str) and arg in _TORCH_DTYPE_NAMES:
            self.cast_dtype = arg
            return self
        if isinstance(arg, str):
            base = arg.split(":", 1)[0]
            if base not in _TORCH_DEVICE_TYPES:
                msg = (
                    f"Expected one of {sorted(_TORCH_DEVICE_TYPES)} device type at start of "
                    f"device string: {arg}"
                )
                raise RuntimeError(msg)
            self._param = _FakeParam(arg)
            return self
        self.cast_dtype = arg
        return self

    def _ents_for(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        per_text = type(self).per_text_entities
        source = per_text.get(text, []) if per_text is not None else type(self).entities
        return [e for e in source if e["label"] in labels and e["score"] >= threshold]

    def _rels_for(self, text: str, relations: list[str], threshold: float) -> list[dict[str, Any]]:
        per_text = type(self).per_text_relations
        source = per_text.get(text, []) if per_text is not None else type(self).relations
        return [
            r
            for r in source
            if r.get("relation", r.get("label")) in relations
            and r.get("score", 1.0) >= threshold
        ]

    def predict_relations(
        self,
        text: str,
        labels: list[str],
        relations: list[str],
        threshold: float = 0.5,
        **kwargs: Any,  # noqa: ARG002 - flat_ner/adjacency_threshold/... accepted for parity
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # The real single-text convenience wrapper over ``inference`` — kept for API-shape tests.
        return (
            self._ents_for(text, labels, threshold),
            self._rels_for(text, relations, threshold),
        )

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        relations: list[str],
        threshold: float = 0.5,
        return_relations: bool = True,  # noqa: ARG002 - real arg; the fake always returns relations
        **kwargs: Any,  # noqa: ARG002 - relation_threshold/flat_ner/... accepted for parity
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        # The REAL batched fused entry point (verified on the pod): ONE forward pass over all
        # texts; returns two parallel lists (entities-per-text, relations-per-text) indexed by
        # input position — entities[i]/relations[i] belong to texts[i].
        self.batch_calls += 1
        entities_per_text = [self._ents_for(text, labels, threshold) for text in texts]
        relations_per_text = [self._rels_for(text, relations, threshold) for text in texts]
        return entities_per_text, relations_per_text


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _FakeGLiNER.entities = []
    _FakeGLiNER.relations = []
    _FakeGLiNER.per_text_entities = None
    _FakeGLiNER.per_text_relations = None
    return _FakeGLiNER


def _config() -> dict[str, Any]:
    return {"labels": ["person", "organization"], "relation_labels": ["works_for"]}


def test_requires_entity_and_relation_labels() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        GlinerRelexExtractor({"relation_labels": ["works_for"]})
    with pytest.raises(ValueError, match="requires a non-empty 'relation_labels' list"):
        GlinerRelexExtractor({"labels": ["person"]})


def test_default_model_is_permissive_apache() -> None:
    extractor = GlinerRelexExtractor(_config())
    assert extractor._model_id == "knowledgator/gliner-relex-multi-v1.0"


def test_relex_bounds_model_window_default_768(fake_gliner: type[_FakeGLiNER]) -> None:
    # W12 model-boundary guard: the provider sets model.config.max_len (the gliner truncation lever)
    # so a chunk longer than the mdeberta window is not SILENTLY truncated. Default 768.
    extractor = GlinerRelexExtractor(_config())
    extractor.extract_fused([_chunk("Alice works for Acme.")])  # triggers load + window bound
    assert extractor._model.config.max_len == 768


def test_relex_model_window_config_override(fake_gliner: type[_FakeGLiNER]) -> None:
    extractor = GlinerRelexExtractor({**_config(), "max_len": 512})
    extractor.extract_fused([_chunk("Alice works for Acme.")])
    assert extractor._model.config.max_len == 512


def test_fused_emits_mentions_and_relations(fake_gliner: type[_FakeGLiNER]) -> None:
    # "Alice works for Acme Corp." — Alice [0,5), Acme Corp [16,25) in chunk-local text.
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.95},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 16, "end": 25},
            "relation": "works_for",
            "score": 0.88,
        }
    ]
    chunk = _chunk("Alice works for Acme Corp.", char_start=100)
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    assert isinstance(result, FusedExtraction)
    assert [m.text for m in result.mentions] == ["Alice", "Acme Corp"]
    assert all(isinstance(m, EntityMention) for m in result.mentions)
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert isinstance(rel, RelationMention)
    assert rel.label == "works_for"
    # The relation's endpoints resolve to the mentions this pass produced.
    ids = {m.text: m.record_id for m in result.mentions}
    assert rel.head_mention_id == ids["Alice"]
    assert rel.tail_mention_id == ids["Acme Corp"]
    assert rel.confidence == 0.88
    # Chunk-local offsets shifted into the parent-document coordinate system.
    assert result.mentions[0].provenance.char_start == 100
    assert rel.provenance.char_start == 100
    assert rel.provenance.char_end == 125
    # Page inherited from the chunk (no multi-page map here).
    assert rel.provenance.page_start == 2


def test_relation_endpoint_offsets_shift_to_document(fake_gliner: type[_FakeGLiNER]) -> None:
    # Multi-page chunk: head on page 1, tail on page 2 (via the covering span).
    assembly = PageMap.from_page_texts(["Alice is here.", "She joined Globex."])
    # page 1 [0,16), page 2 [16,...); "Alice" at 0, "Globex" at 27 in assembled text.
    text = assembly.text
    g_start = text.find("Globex")
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": g_start, "end": g_start + 6, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": g_start, "end": g_start + 6},
            "relation": "works_for",
            "score": 0.8,
        }
    ]
    chunk = _chunk(text, char_start=0, page_map=assembly.page_map)
    cfg = {"labels": ["person", "organization"], "relation_labels": ["works_for"]}
    result = GlinerRelexExtractor(cfg).extract_fused([chunk])
    rel = result.relations[0]
    # Covering span straddles the page boundary: records both pages.
    assert rel.provenance.page_start == 1
    assert rel.provenance.page_end == 2


def _two_chunk_document() -> tuple[ChunkRecord, ChunkRecord]:
    """A v19 document stream: each chunk carries the page spans it overlaps.

    Page 1 is ``[0,16)``, page 2 is ``[16,37)``. Chunk 0 lies wholly on page 1; chunk 1
    STRADDLES the boundary (doc ``[13,37)``, inherited page range ``1..2``) so its own slice
    holds BOTH spans. In chunk 1, "Bob" is doc ``[16,19)`` and "Globex" is doc ``[30,36)`` —
    both wholly on page 2, as is the relation's covering span.
    """
    assembly = PageMap.from_page_texts(["Alice is here.", "Bob works for Globex."])

    def _mk(
        *, index: int, char_start: int, char_end: int, pages: tuple[int, int]
    ) -> ChunkRecord:
        return ChunkRecord(
            record_id=f"doc1#chunk-{index}",
            document_record_id="doc1",
            index=index,
            provenance=Provenance(
                source_uri="file:///c/a.txt",
                file_name="a.txt",
                file_type="txt",
                file_size=1000,
                document_id="sha256:doc1",
                char_start=char_start,
                char_end=char_end,
                page_start=pages[0],
                page_end=pages[1],
            ),
            classification=Classification(language="en", sensitivity="internal"),
            content=assembly.text[char_start:char_end],
            token_count=4,
            page_slice=PageSlice.for_span(assembly.page_map, char_start, char_end),
            offset_map=None,
        )

    first = _mk(index=0, char_start=0, char_end=14, pages=(1, 1))
    second = _mk(index=1, char_start=13, char_end=37, pages=(1, 2))
    return first, second


def test_page_resolves_on_a_page_straddling_later_chunk(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    """A later, page-straddling chunk must resolve its OWN pages (mentions AND relations).

    Inheriting the chunk's page range degrades everything on it to ``(1, 2)`` — an
    ADR-0021/0031 provenance regression, and a plausible-looking wrong page.
    """
    first, second = _two_chunk_document()
    fake_gliner.per_text_entities = {
        first.content: [],
        second.content: [
            {"start": 3, "end": 6, "label": "person", "score": 0.9},
            {"start": 17, "end": 23, "label": "organization", "score": 0.9},
        ],
    }
    fake_gliner.per_text_relations = {
        first.content: [],
        second.content: [
            {
                "head": {"start": 3, "end": 6},
                "tail": {"start": 17, "end": 23},
                "relation": "works_for",
                "score": 0.8,
            }
        ],
    }
    result = GlinerRelexExtractor(_config()).extract_fused([first, second])
    by_text = {m.text: m for m in result.mentions}
    assert set(by_text) == {"Bob", "Globex"}
    # The whole point: page 2 for both, NOT the chunk's inherited (1, 2).
    assert (by_text["Bob"].provenance.page_start, by_text["Bob"].provenance.page_end) == (2, 2)
    rel = result.relations[0]
    assert (rel.provenance.page_start, rel.provenance.page_end) == (2, 2)


def test_pages_still_resolve_when_the_first_chunk_is_below_min_chars(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    """A chunk the ``min_chars`` filter skips must not cost any OTHER chunk its page.

    Under v18 it did — the map rode the document's first chunk, which is often the one this
    filter drops. Since v19 the property is the record's; this pins the outcome either way.
    """
    first, second = _two_chunk_document()
    assert len(first.content.strip()) == 14  # below the min_chars floor set below
    fake_gliner.per_text_entities = {
        second.content: [{"start": 3, "end": 6, "label": "person", "score": 0.9}],
    }
    fake_gliner.per_text_relations = {second.content: []}
    cfg = {**_config(), "min_chars": 15}
    result = GlinerRelexExtractor(cfg).extract_fused([first, second])
    assert [m.text for m in result.mentions] == ["Bob"]
    prov = result.mentions[0].provenance
    assert (prov.page_start, prov.page_end) == (2, 2)


def test_offsets_resolve_through_offset_map_past_stripped_markup(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    # Original "**Note:** Alice joined Globex." strips to "Note: Alice joined Globex.":
    # the stripped ``**`` before the prose shifts every later mention right in the
    # original, so char_start+local is a lower bound. The carried offset_map corrects it.
    original = "**Note:** Alice joined Globex."
    content = "Note: Alice joined Globex."
    offset_map = OffsetMap(
        segments=[
            OffsetSegment(local_start=0, doc_start=2),
            OffsetSegment(local_start=5, doc_start=9),
        ],
        length=len(content),
    )
    fake_gliner.entities = [
        {"start": 6, "end": 11, "label": "person", "score": 0.9},
        {"start": 19, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 6, "end": 11},
            "tail": {"start": 19, "end": 25},
            "relation": "works_for",
            "score": 0.8,
        }
    ]
    chunk = _chunk(content, char_start=2, char_end=len(original), offset_map=offset_map)
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    by_text = {m.text: m for m in result.mentions}
    assert set(by_text) == {"Alice", "Globex"}
    for m in result.mentions:
        # TRUE original span (not the char_start+local lower bound).
        assert original[m.provenance.char_start : m.provenance.char_end] == m.text
    assert by_text["Alice"].provenance.char_start == 10  # not 2 + 6 = 8
    assert by_text["Globex"].provenance.char_start == 23  # not 2 + 19 = 21
    # The relation's covering span is the min/max of the corrected endpoints.
    rel = result.relations[0]
    assert (rel.provenance.char_start, rel.provenance.char_end) == (10, 29)
    assert original[rel.provenance.char_start : rel.provenance.char_end] == "Alice joined Globex"


def test_empty_chunk_skipped_without_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    result = GlinerRelexExtractor(_config()).extract_fused([_chunk("")])
    assert result.mentions == []
    assert result.relations == []


def test_relation_with_unresolvable_endpoint_dropped(fake_gliner: type[_FakeGLiNER]) -> None:
    # A relation referencing an entity span the model did not emit is dropped, so no
    # un-resolvable record leaks out.
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 999, "end": 1004},  # no such entity
            "relation": "works_for",
            "score": 0.8,
        }
    ]
    chunk = _chunk("Alice works for Acme Corp.")
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    assert len(result.mentions) == 2
    assert result.relations == []


def test_extract_half_returns_only_mentions(fake_gliner: type[_FakeGLiNER]) -> None:
    # The Provider also fits the plain EntityExtractor seam via .extract().
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    fake_gliner.relations = []
    chunk = _chunk("Alice waved.")
    mentions = GlinerRelexExtractor(_config()).extract([chunk])
    assert [m.text for m in mentions] == ["Alice"]


def test_missing_gliner_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner", None)
    extractor = GlinerRelexExtractor(_config())
    with pytest.raises(ImportError, match="pip install latence-relation-gliner"):
        extractor.extract_fused([_chunk("Alice works for Acme Corp.")])


# --- W1-relex FIX teeth: the anti-false-green anchor + the provider-level endpoint guard ---


def test_predict_relations_returns_a_tuple_not_a_dict(fake_gliner: type[_FakeGLiNER]) -> None:
    # Pin the REAL API shape: predict_relations returns a (entities, relations) TUPLE, and the
    # non-existent predict_with_relations is NOT exposed (a provider calling it AttributeErrors).
    model = fake_gliner.from_pretrained("knowledgator/gliner-relex-multi-v1.0")
    fake_gliner.entities = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    fake_gliner.relations = []
    out = model.predict_relations("Alice waved.", ["person"], ["works_for"], threshold=0.3)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert not hasattr(model, "predict_with_relations")


def test_dict_parse_path_fails_against_tuple_api(fake_gliner: type[_FakeGLiNER]) -> None:
    # THE anti-false-green anchor. The old provider parsed the model output as a DICT
    # (``result.get("entities")`` / ``result.get("relations")``). Against the faithful
    # tuple-returning fake, that path extracts NOTHING (a tuple has no ``.get``), so a
    # regression that reverts to dict-parsing silently drops every mention AND relation —
    # which THIS test catches. The real (fixed) provider unpacks the tuple and finds them.
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.95},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5, "text": "Alice", "type": "person", "entity_idx": 0},
            "tail": {"start": 16, "end": 25, "text": "Acme Corp", "type": "org", "entity_idx": 1},
            "relation": "works_for",
            "score": 0.88,
        }
    ]
    chunk = _chunk("Alice works for Acme Corp.")

    # The FIXED provider: tuple-unpack finds mentions + the relation.
    result = GlinerRelexExtractor(_config()).extract_fused([chunk])
    assert len(result.mentions) == 2
    assert len(result.relations) == 1

    # Simulate the OLD dict-parse path against the SAME faithful tuple fake: it gets nothing,
    # so an un-fixed provider fails to emit the relation this test demands. (A tuple has no
    # ``.get`` — dict-parsing raises; either way the false green is impossible now.)
    model = fake_gliner.from_pretrained("knowledgator/gliner-relex-multi-v1.0")
    raw = model.predict_relations(chunk.content, ["person", "organization"], ["works_for"])
    with pytest.raises(AttributeError):
        _ = raw.get("entities")  # type: ignore[attr-defined]  # tuple has no .get


class _CapturingGLiNER(_FakeGLiNER):
    """Records the (entity_labels, relation_labels) each batched ``inference`` call is asked for.

    W7: the fused provider now issues the batched ``inference`` entry point (not a per-text
    ``predict_relations`` loop), so the W4 label-slot capture reads the labels passed to
    ``inference`` — same assertion (config floor UNIONed with induced, per slot), one call per
    batch instead of per chunk.
    """

    captured: list[tuple[list[str], list[str]]] = []

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        relations: list[str],
        threshold: float = 0.5,
        return_relations: bool = True,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        type(self).captured.append((list(labels), list(relations)))
        return super().inference(
            texts, labels, relations, threshold, return_relations, **kwargs
        )


@pytest.fixture()
def capturing_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_CapturingGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _CapturingGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _CapturingGLiNER.entities = []
    _CapturingGLiNER.relations = []
    _CapturingGLiNER.per_text_entities = None
    _CapturingGLiNER.per_text_relations = None
    _CapturingGLiNER.captured = []
    return _CapturingGLiNER


def test_induced_labels_union_onto_both_entity_and_relation_slots(
    capturing_gliner: type[_CapturingGLiNER],
) -> None:
    # W4 seam (TESTS lens): the fused extractor must UNION the induced entity_types onto the config
    # entity floor AND the induced relation_types onto the config relation floor — each on its OWN
    # slot. A regression that mis-slotted them (e.g. passing ENTITY to the relation arg) FAILS here.
    chunk = _chunk("Alice works for Acme Corp.")
    chunk = chunk.model_copy(
        update={
            "induced_labels": InducedLabels(
                model_id="test-model",
                entity_types=["location"],  # NOT in the config entity floor
                relation_types=["located_in"],  # NOT in the config relation floor
                pii_types=[],
            )
        }
    )
    GlinerRelexExtractor(_config()).extract_fused([chunk])
    assert capturing_gliner.captured, "the model was never asked to predict"
    entity_labels, relation_labels = capturing_gliner.captured[0]
    # entity slot: config floor first, then the induced entity type (order-stable union).
    assert entity_labels == ["person", "organization", "location"]
    # relation slot: config floor first, then the induced relation type — NOT the entity type.
    assert relation_labels == ["works_for", "located_in"]


def test_no_induced_labels_asks_for_config_floor_only(
    capturing_gliner: type[_CapturingGLiNER],
) -> None:
    # Byte-identical backward-compat: a chunk with no induced labels ⇒ exactly the config floors.
    GlinerRelexExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])
    assert capturing_gliner.captured
    entity_labels, relation_labels = capturing_gliner.captured[0]
    assert entity_labels == ["person", "organization"]
    assert relation_labels == ["works_for"]


def test_emitted_relations_have_real_endpoints(fake_gliner: type[_FakeGLiNER]) -> None:
    # The provider-level GUARD (teeth, like GLM #126): every emitted RelationMention must resolve
    # to head/tail record_ids that EXIST among the emitted mentions, with a non-empty label — so a
    # regression that mis-parses the tuple and silently drops/mangles endpoints FAILS here.
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.95},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5, "text": "Alice", "type": "person", "entity_idx": 0},
            "tail": {"start": 16, "end": 25, "text": "Acme Corp", "type": "org", "entity_idx": 1},
            "relation": "works_for",
            "score": 0.88,
        }
    ]
    result = GlinerRelexExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])
    mention_ids = {m.record_id for m in result.mentions}
    assert result.relations, "expected ≥1 relation with resolved endpoints"
    for rel in result.relations:
        assert rel.head_mention_id in mention_ids
        assert rel.tail_mention_id in mention_ids
        assert rel.head_mention_id != rel.tail_mention_id
        assert rel.label


# --- W7 encoder batching (ADR-0036 §1 / ADR-0033) --------------------------------------------


def _distinct_chunks(n: int) -> list[ChunkRecord]:
    # N chunks each "NameNNN leads OrgNNN." with a distinct person + org + a works relation, so a
    # mis-mapped batch (off-by-one / wrong order) crosses names — the equivalence crux.
    chunks: list[ChunkRecord] = []
    for i in range(n):
        content = f"Name{i:03d} leads Org{i:03d}."
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
        chunks.append(
            ChunkRecord(
                record_id=f"doc{i}#chunk-0",
                document_record_id=f"doc{i}",
                index=0,
                provenance=prov,
                classification=Classification(language="en", sensitivity="internal"),
                content=content,
                token_count=3,
                # v19: a one-page fixture — the chunk's own page spans are that single page.
                page_slice=PageSlice(
                    pages=[PageSpan(page_number=1, char_start=0, char_end=len(content))],
                    total_chars=len(content),
                ),
            )
        )
    return chunks


def _per_text_maps(
    chunks: list[ChunkRecord],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    # "NameNNN leads OrgNNN." — person [0,7), org [14,20); a works_for(head=person, tail=org).
    ents: dict[str, list[dict[str, Any]]] = {}
    rels: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        ents[c.content] = [
            {"start": 0, "end": 7, "label": "person", "score": 0.95},
            {"start": 14, "end": 20, "label": "organization", "score": 0.9},
        ]
        rels[c.content] = [
            {
                "head": {"start": 0, "end": 7},
                "tail": {"start": 14, "end": 20},
                "relation": "works_for",
                "score": 0.88,
            }
        ]
    return ents, rels


def test_batched_equals_unbatched_record_for_record(fake_gliner: type[_FakeGLiNER]) -> None:
    chunks = _distinct_chunks(7)
    ents, rels = _per_text_maps(chunks)
    fake_gliner.per_text_entities = ents
    fake_gliner.per_text_relations = rels
    cfg = {"labels": ["person", "organization"], "relation_labels": ["works_for"]}

    unbatched_m: list[EntityMention] = []
    unbatched_r: list[RelationMention] = []
    for c in chunks:
        out = GlinerRelexExtractor({**cfg, "batch_size": 1}).extract_fused([c])
        unbatched_m.extend(out.mentions)
        unbatched_r.extend(out.relations)
    batched = GlinerRelexExtractor({**cfg, "batch_size": 4}).extract_fused(chunks)

    assert len(batched.mentions) == len(unbatched_m) == 14
    for b, u in zip(batched.mentions, unbatched_m, strict=True):
        assert b.text == u.text
        assert b.label == u.label
        assert (b.provenance.char_start, b.provenance.char_end) == (
            u.provenance.char_start,
            u.provenance.char_end,
        )
        assert b.record_id == u.record_id
    assert len(batched.relations) == len(unbatched_r) == 7
    for br, ur in zip(batched.relations, unbatched_r, strict=True):
        assert br.label == ur.label
        assert br.head_text == ur.head_text
        assert br.tail_text == ur.tail_text
        assert br.head_mention_id == ur.head_mention_id
        assert br.tail_mention_id == ur.tail_mention_id
    # Each relation's endpoints are within its OWN chunk (a wrong-order batch crosses documents).
    for rel in batched.relations:
        assert rel.head_text.startswith("Name")
        assert rel.tail_text.startswith("Org")


def test_model_called_ceil_n_over_batch_size_times(fake_gliner: type[_FakeGLiNER]) -> None:
    chunks = _distinct_chunks(7)
    ents, rels = _per_text_maps(chunks)
    fake_gliner.per_text_entities = ents
    fake_gliner.per_text_relations = rels
    cfg = {"labels": ["person", "organization"], "relation_labels": ["works_for"], "batch_size": 3}
    extractor = GlinerRelexExtractor(cfg)
    extractor.extract_fused(chunks)
    # 7 chunks at batch_size=3 => ceil(7/3)=3 fused forward passes, not 7.
    assert extractor._model.batch_calls == 3


def test_batch_size_must_be_positive_int() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="batch_size must be >= 1"):
        GlinerRelexExtractor({**_config(), "batch_size": 0})


class _NoInferenceGLiNER:
    """A checkpoint that exposes ONLY the W1-verified single-text ``predict_relations``, NOT the
    W7 batched ``inference`` — the false-green class the provider guard must catch legibly.

    Does NOT subclass ``_FakeGLiNER`` (which defines ``inference``): a bare class genuinely
    lacking the method is what makes ``hasattr(model, "inference")`` False at runtime, unlike an
    ``= None`` shadow (still an attribute — ``hasattr`` stays True).
    """

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _NoInferenceGLiNER:  # noqa: ARG003
        return cls()

    def to(self, _dtype: Any) -> _NoInferenceGLiNER:
        return self  # W8: the plain-gliner dtype path is .to(bf16) on cuda

    def predict_relations(
        self,
        text: str,  # noqa: ARG002
        labels: list[str],  # noqa: ARG002
        relations: list[str],  # noqa: ARG002
        threshold: float = 0.5,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return ([], [])


def test_missing_inference_api_raises_clear_error_naming_the_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The batched ``inference`` entry point is offline-unverifiable, so if the real checkpoint
    # lacks it the provider must fail with an actionable message (naming the batched API AND the
    # W1-verified single-text ``predict_relations`` fallback), not a bare AttributeError. This is
    # the offline stand-in for the pod script's Gate 0 introspection assertion.
    module = types.ModuleType("gliner")
    module.GLiNER = _NoInferenceGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    with pytest.raises(AttributeError, match="predict_relations"):
        GlinerRelexExtractor(_config()).extract_fused([_chunk("Alice works for Acme Corp.")])


# --- #191 device placement (ADR-0036 §2) -----------------------------------------------------
#
# #191 established that the plain ``gliner`` library NEVER auto-places its model
# (``from_pretrained`` map-locations to CPU) and encoded the fix in ``place_on_device`` — but only
# ``redaction.gliner_pii`` was wired to it. ``relation.gliner_relex`` passed ``device=None`` into
# ``load_then_cast``, so it cast to bf16 off a bare CUDA-availability probe while the weights stayed
# resident on CPU, and its ``device`` config was read by nobody. These tests bite that gap.


def _model_param_device(provider: GlinerRelexExtractor) -> str:
    """Where the LOADED model's parameters actually live — the only honest "what device ran this"."""
    param: _FakeParam = next(iter(provider._model.parameters()))
    return param.device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_relex_model_parameters_reside_on_the_resolved_device(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """#191 regression: the model is PLACED on the resolved device at load, not left on CPU.

    Device-parameterized so CPU-only CI still runs both lanes meaningfully: ``LATENCE_CUDA=1`` is
    the framework's deterministic CUDA probe override, so the ``cuda`` lane exercises the real
    resolution + placement path on a GPU-less box. Pre-fix the provider never called ``.to(device)``
    at all, so the cuda lane found the parameters on ``cpu`` — the silent ~10x throughput cliff.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1" if device == "cuda" else "0")
    extractor = GlinerRelexExtractor({**_config(), "device": device})
    extractor.extract_fused([_chunk("Alice works for Acme Corp.")])
    assert _model_param_device(extractor) == device


def test_relex_bf16_cast_never_lands_on_a_cpu_resident_model(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bf16 cast and the resident device agree — never bf16 weights running eager on CPU.

    This is the precise pre-fix corruption: on a CUDA host the seam resolved bf16 from a bare
    availability probe while the model stayed on CPU (``load_then_cast``'s own docstring says the
    cast applies "only when ... the model is on cuda").
    """
    fake_torch = types.ModuleType("torch")
    bf16 = object()
    fake_torch.bfloat16 = bf16  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LATENCE_CUDA", "1")

    extractor = GlinerRelexExtractor({**_config(), "device": "cuda"})
    extractor.extract_fused([_chunk("Alice works for Acme Corp.")])
    assert extractor._model.cast_dtype is bf16
    assert _model_param_device(extractor) == "cuda"


def test_relex_output_is_identical_across_devices(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output contract does not move with the device — only throughput does (#191 AC 3)."""
    fake_gliner.entities = [
        {"start": 0, "end": 5, "label": "person", "score": 0.95},
        {"start": 16, "end": 25, "label": "organization", "score": 0.9},
    ]
    fake_gliner.relations = [
        {
            "head": {"start": 0, "end": 5},
            "tail": {"start": 16, "end": 25},
            "relation": "works_for",
            "score": 0.88,
        }
    ]

    def _run(device: str) -> tuple[list[Any], list[Any]]:
        monkeypatch.setenv("LATENCE_CUDA", "1" if device == "cuda" else "0")
        extractor = GlinerRelexExtractor({**_config(), "device": device})
        res = extractor.extract_fused([_chunk("Alice works for Acme Corp.")])
        return (
            [(m.text, m.label, m.provenance.char_start, m.provenance.char_end) for m in res.mentions],
            [(r.label, r.confidence, r.provenance.char_start, r.provenance.char_end) for r in res.relations],
        )

    assert _run("cuda") == _run("cpu")
    assert _run("cpu") != ([], [])  # a non-empty comparison, not two empty results


def test_relex_cuda_requested_without_a_gpu_fails_loudly(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``device: cuda`` on a CPU-only host is an error, NEVER a silent CPU run (#191: silence IS
    the bug). Pre-fix the config key was ignored entirely and the run went to CPU without a word."""
    monkeypatch.setenv("LATENCE_CUDA", "0")
    extractor = GlinerRelexExtractor({**_config(), "device": "cuda"})
    with pytest.raises(ConfigError, match="no CUDA device is available"):
        extractor.extract_fused([_chunk("Alice works for Acme Corp.")])


def test_relex_placement_mismatch_is_refused(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that reports the WRONG device after placement is refused, not logged and run.

    Simulates the field failure directly: ``.to(device)`` succeeds but the weights stay on CPU.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    monkeypatch.setattr(_FakeGLiNER, "to", lambda self, arg: self)  # noqa: ARG005
    extractor = GlinerRelexExtractor({**_config(), "device": "cuda"})
    with pytest.raises(ProviderError, match="reports device 'cpu'"):
        extractor.extract_fused([_chunk("Alice works for Acme Corp.")])


def test_relex_invalid_device_string_is_a_config_error() -> None:
    """An unknown device is a ConfigError at CONSTRUCTION — before any weights are downloaded."""
    with pytest.raises(ConfigError, match="invalid device config"):
        GlinerRelexExtractor({**_config(), "device": "gpu"})
