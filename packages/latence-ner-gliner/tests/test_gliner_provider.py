"""GlinerEntityExtractor: learned zero-shot Entity Extraction over GLiNER (S4).

The real ``gliner`` stack (torch/transformers) is intentionally NOT installed in
the dev/test environment (ADR-0007/0016: CPU-first, offline, deterministic). These
tests inject a tiny fake ``gliner`` module so the Provider's contract-stamping and
offset math are exercised deterministically and offline — the model itself is
GLiNER's concern, not this seam's. One test also covers the missing-dependency
error path and the zero-shot config validation.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    EntityMention,
    OffsetMap,
    OffsetSegment,
    PageMap,
    Provenance,
)
from latence_core.errors import ConfigError, ProviderError
from latence_ner_gliner import GlinerEntityExtractor

if TYPE_CHECKING:
    from collections.abc import Iterator


def _chunk(
    content: str,
    *,
    char_start: int = 100,
    char_end: int | None = None,
    page_map: PageMap | None = None,
    offset_map: OffsetMap | None = None,
    record_id: str = "doc1#chunk-0",
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
        record_id=record_id,
        document_record_id="doc1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=content,
        token_count=len(content.split()),
        page_map=page_map,
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
    """Stands in for gliner.GLiNER: returns fixed chunk-local spans + scores.

    Exposes BOTH the single-text ``predict_entities`` and the batched
    ``batch_predict_entities`` (the real GLiNER API: ``batch_predict_entities(texts, labels,
    threshold=…) -> list[list[dict]]``, result ``i`` for ``texts[i]``). The W7 provider calls the
    batched form; the fake keeps a per-instance ``batch_calls`` counter so a test can assert the
    model is called ``ceil(N/batch_size)`` times, not N.
    """

    #: chunk-local entity dicts the fake "model" returns for any predict call.
    returns: list[dict[str, Any]] = []
    #: an optional per-text mapping (text -> its entity dicts) so batched inputs get DISTINCT
    #: results per item — the off-by-one / wrong-order equivalence trap needs distinct results.
    per_text: dict[str, list[dict[str, Any]]] | None = None
    model_id: str = ""

    def __init__(self) -> None:
        self.batch_calls = 0
        #: the dtype the provider applied via ``.to(dtype)`` (W8) — None until cast. FAITHFUL: the
        #: plain ``gliner`` library casts via ``.to()``, NOT via ``from_pretrained`` perf kwargs.
        self.cast_dtype: Any = None
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
        # FAITHFUL to the REAL ``gliner`` 0.2.27 API (pod-proven): ``GLiNER.from_pretrained``
        # REJECTS ``torch_dtype`` / ``attn_implementation`` with a TypeError. A provider that
        # wrongly spreads the perf kwargs here CRASHES offline exactly as it does on real weights —
        # closing the false-green that shipped in the first W8 cut.
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

    def _for_text(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        per_text = type(self).per_text
        source = per_text.get(text, []) if per_text is not None else type(self).returns
        return [e for e in source if e["label"] in labels and e["score"] >= threshold]

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        # Echo only entities whose label is in the requested set (zero-shot), above
        # the threshold — mirroring the real API's contract.
        return self._for_text(text, labels, threshold)

    def batch_predict_entities(
        self, texts: list[str], labels: list[str], threshold: float = 0.5
    ) -> list[list[dict[str, Any]]]:
        # The real batched API: one forward pass over all texts; result[i] is texts[i]'s entities.
        self.batch_calls += 1
        return [self._for_text(text, labels, threshold) for text in texts]


@pytest.fixture()
def fake_gliner(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER]:
    module = types.ModuleType("gliner")
    module.GLiNER = _FakeGLiNER  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", module)
    _FakeGLiNER.returns = []
    _FakeGLiNER.per_text = None
    return _FakeGLiNER


def test_requires_labels_config() -> None:
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        GlinerEntityExtractor({})
    with pytest.raises(ValueError, match="requires a non-empty 'labels' list"):
        GlinerEntityExtractor({"labels": []})


def test_default_model_is_permissive_apache() -> None:
    extractor = GlinerEntityExtractor({"labels": ["person"]})
    # The default checkpoint is the Apache-2.0 v2.1 (never a CC-BY-NC one).
    assert extractor._model_id == "urchade/gliner_multi-v2.1"


# --- W8: plain-gliner dtype path — bf16 via .to() on cuda, NO perf kwargs to from_pretrained ------


def test_gliner_applies_bf16_via_to_on_cuda_no_perf_kwargs_to_from_pretrained(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On cuda, the provider casts the model with ``.to(bf16)`` — and passes NO perf kwargs to
    ``from_pretrained`` (the plain ``gliner`` 0.2.27 API rejects them; the fake would raise if it did).
    """
    # gliner 0.2.27 rejects torch_dtype/attn_implementation in from_pretrained, so the ONLY correct
    # dtype path is model.to(dtype). Inject a fake torch (torch isn't installed offline) so the perf
    # seam resolves a REAL bf16 dtype object to cast to, and pin a cuda host.
    fake_torch = types.ModuleType("torch")
    bf16 = object()
    fake_torch.bfloat16 = bf16  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LATENCE_CUDA", "1")

    extractor = GlinerEntityExtractor({"labels": ["person"]})
    list(extractor.extract([_chunk("Alice spoke.")]))  # triggers the lazy load + cast
    # The provider loaded bare (no TypeError from the faithful from_pretrained) AND cast to bf16.
    assert extractor._model.cast_dtype is bf16


def test_gliner_bounds_model_window_default_768(fake_gliner: type[_FakeGLiNER]) -> None:
    # W12 model-boundary guard: the provider sets model.config.max_len (the gliner truncation lever)
    # so a chunk longer than the window is not SILENTLY truncated. Default 768 (mdeberta-v3 window).
    extractor = GlinerEntityExtractor({"labels": ["person"]})
    list(extractor.extract([_chunk("Alice spoke.")]))  # triggers the lazy load + window bound
    assert extractor._model.config.max_len == 768


def test_gliner_model_window_config_override(fake_gliner: type[_FakeGLiNER]) -> None:
    extractor = GlinerEntityExtractor({"labels": ["person"], "max_len": 640})
    list(extractor.extract([_chunk("Alice spoke.")]))
    assert extractor._model.config.max_len == 640


def test_gliner_no_cast_on_cpu(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a CPU host the dtype default is fp32 — no ``.to()`` cast is applied (bf16 CPU is not a win)."""
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LATENCE_CUDA", "0")

    extractor = GlinerEntityExtractor({"labels": ["person"]})
    list(extractor.extract([_chunk("Alice spoke.")]))
    assert extractor._model.cast_dtype is None  # fp32 on CPU → no cast


def test_gliner_load_never_crashes_without_torch(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With torch NOT importable (the real offline env), a cuda host still loads — cast gracefully
    skipped (no real dtype to cast to), never a crash. This is the seam's torch-free contract.
    """
    monkeypatch.setitem(sys.modules, "torch", None)  # torch import fails
    monkeypatch.setenv("LATENCE_CUDA", "1")
    extractor = GlinerEntityExtractor({"labels": ["person"]})
    mentions = list(extractor.extract([_chunk("Alice spoke.")]))
    assert extractor._model.cast_dtype is None  # no torch → no dtype object → cast skipped
    assert isinstance(mentions, list)  # loaded + ran, no crash


def test_extracts_and_offsets_to_document_span(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.returns = [
        {"start": 0, "end": 5, "label": "person", "score": 0.91},
        {"start": 10, "end": 13, "label": "person", "score": 0.77},
    ]
    chunk = _chunk("Alice and Bob spoke.", char_start=100)
    mentions = list(GlinerEntityExtractor({"labels": ["person"]}).extract([chunk]))
    assert [m.text for m in mentions] == ["Alice", "Bob"]
    assert all(isinstance(m, EntityMention) for m in mentions)
    # Chunk-local offsets shifted into the parent-document span.
    assert mentions[0].provenance.char_start == 100
    assert mentions[0].provenance.char_end == 105
    assert mentions[1].provenance.char_start == 110
    # Page range inherited from the parent chunk (S4 round-trip).
    assert mentions[0].provenance.page_start == 2
    # GLiNER's score becomes the mention confidence.
    assert mentions[0].confidence == 0.91
    # Links + classification stamped from the chunk.
    assert mentions[0].chunk_record_id == "doc1#chunk-0"
    assert mentions[0].classification.sensitivity == "internal"


def test_mention_page_resolves_to_its_own_page_in_multipage_chunk(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    # A chunk spanning two pages (carried page map): a mention on page 2 must cite
    # page 2, not inherit the chunk's first page (S4 "correct source page").
    assembly = PageMap.from_page_texts(["Alice is here.", "Bob is over here."])
    # page 1 [0,16), page 2 [16,33); "Alice" at 0, "Bob" at 16.
    fake_gliner.returns = [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": 16, "end": 19, "label": "person", "score": 0.9},
    ]
    chunk = _chunk(assembly.text, char_start=0, page_map=assembly.page_map)
    mentions = list(GlinerEntityExtractor({"labels": ["person"]}).extract([chunk]))
    by_text = {m.text: m for m in mentions}
    assert (by_text["Alice"].provenance.page_start, by_text["Alice"].provenance.page_end) == (1, 1)
    assert (by_text["Bob"].provenance.page_start, by_text["Bob"].provenance.page_end) == (2, 2)


def test_mention_offset_resolves_through_offset_map_past_stripped_markup(
    fake_gliner: type[_FakeGLiNER],
) -> None:
    # Original "**Note:** Alice joined Globex." strips to "Note: Alice joined Globex.":
    # the stripped ``**`` before the prose shifts every later mention right in the
    # original, so char_start+local is a lower bound. The carried offset_map corrects it
    # — the exact learned-NER blind spot issue #48 names (the gazetteer fix must reach
    # here too, not just the in-core Provider).
    original = "**Note:** Alice joined Globex."
    content = "Note: Alice joined Globex."
    offset_map = OffsetMap(
        segments=[
            OffsetSegment(local_start=0, doc_start=2),
            OffsetSegment(local_start=5, doc_start=9),
        ],
        length=len(content),
    )
    fake_gliner.returns = [
        {"start": 6, "end": 11, "label": "person", "score": 0.9},
        {"start": 19, "end": 25, "label": "organization", "score": 0.9},
    ]
    chunk = _chunk(
        content, char_start=2, char_end=len(original), offset_map=offset_map
    )
    mentions = list(
        GlinerEntityExtractor({"labels": ["person", "organization"]}).extract([chunk])
    )
    by_text = {m.text: m for m in mentions}
    assert set(by_text) == {"Alice", "Globex"}
    for m in mentions:
        # TRUE original span (not the char_start+local lower bound) — would fail on the
        # bare shift: original[8:13] == 'Alice' is false ("ce jo" or ":** A" etc.).
        assert original[m.provenance.char_start : m.provenance.char_end] == m.text
    assert by_text["Alice"].provenance.char_start == 10  # not 2 + 6 = 8
    assert by_text["Globex"].provenance.char_start == 23  # not 2 + 19 = 21


def test_threshold_filters_low_scores(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.returns = [
        {"start": 0, "end": 5, "label": "person", "score": 0.4},
        {"start": 10, "end": 13, "label": "person", "score": 0.9},
    ]
    chunk = _chunk("Alice and Bob spoke.")
    out = list(GlinerEntityExtractor({"labels": ["person"], "threshold": 0.5}).extract([chunk]))
    assert [m.text for m in out] == ["Bob"]


def test_empty_chunk_skipped_without_model_call(fake_gliner: type[_FakeGLiNER]) -> None:
    fake_gliner.returns = [{"start": 0, "end": 5, "label": "person", "score": 0.9}]
    out = list(GlinerEntityExtractor({"labels": ["person"]}).extract([_chunk("")]))
    assert out == []


def test_missing_gliner_dependency_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate the dep not being installed: importing gliner fails.
    monkeypatch.setitem(sys.modules, "gliner", None)
    extractor = GlinerEntityExtractor({"labels": ["person"]})
    with pytest.raises(ImportError, match="pip install latence-ner-gliner"):
        list(extractor.extract([_chunk("Alice spoke.")]))


def test_declares_profile_either_verified_apache() -> None:
    # P3-F1 (ADR-0036 §2): the GLiNER Provider declares a profile the registry surfaces.
    # compute="either" (CPU or GPU), verified Apache-2.0 (weights AND code, ADR-0012),
    # the default checkpoint as model_id, and deterministic=False (learned float inference).
    from latence_core.providers.profile import ProviderProfile, profile_of

    prof = profile_of(GlinerEntityExtractor)
    assert isinstance(prof, ProviderProfile)
    assert prof.compute == "either"
    assert prof.model_id == "urchade/gliner_multi-v2.1"
    assert prof.license == "Apache-2.0"
    assert prof.license_verified is True
    assert prof.deterministic is False
    assert prof.batch is True
    assert prof.cost_per_1k is None


# --- W7 encoder batching (ADR-0036 §1 / ADR-0033) --------------------------------------------


def _distinct_chunks(n: int) -> list[ChunkRecord]:
    # N chunks each with a DISTINCT single-entity content AND a DISTINCT chunk-local span (chunk i
    # is prefixed with i filler chars so its name sits at offset i, span [i, i+7)). Distinct spans
    # are what give the equivalence test teeth: a mis-mapped batch (off-by-one / wrong order) hands
    # chunk i the raw dict of some OTHER chunk j (start=j), so this chunk slices content[j:j+7] —
    # the WRONG substring / offset — and the record-for-record assertions FAIL. (With a shared
    # [0,7) span across all items a reversed map would be invisible, because every record still
    # slices its OWN content at [0,7); H-E1/E2 false-green hardening.)
    chunks: list[ChunkRecord] = []
    for i in range(n):
        content = ("." * i) + f"Name{i:03d} spoke."
        chunks.append(_chunk(content, char_start=100 + i * 1000, record_id=f"doc{i}#chunk-0"))
    return chunks


def _per_text_person_map(chunks: list[ChunkRecord]) -> dict[str, list[dict[str, Any]]]:
    # Each chunk's content is "<i dots>NameNNN spoke." -> a person entity spanning the name at its
    # OWN chunk-local offset i (span [i, i+7)), keyed by index so item i carries start=i. A batch
    # that maps results back out of order gives a chunk a DIFFERENT start than i — caught below.
    return {
        c.content: [{"start": i, "end": i + 7, "label": "person", "score": 0.9}]
        for i, c in enumerate(chunks)
    }


def test_batched_equals_unbatched_record_for_record(fake_gliner: type[_FakeGLiNER]) -> None:
    # THE equivalence crux: batched(N) must equal N single-item calls, record-for-record
    # (text/label/offsets/order). Distinct per-chunk results make a mis-map FAIL.
    chunks = _distinct_chunks(7)
    fake_gliner.per_text = _per_text_person_map(chunks)
    cfg = {"labels": ["person"]}

    unbatched = [
        m
        for c in chunks
        for m in GlinerEntityExtractor({**cfg, "batch_size": 1}).extract([c])
    ]
    batched = list(GlinerEntityExtractor({**cfg, "batch_size": 4}).extract(chunks))

    assert len(batched) == len(unbatched) == 7
    for b, u in zip(batched, unbatched, strict=True):
        assert b.text == u.text
        assert b.label == u.label
        assert (b.provenance.char_start, b.provenance.char_end) == (
            u.provenance.char_start,
            u.provenance.char_end,
        )
        assert b.record_id == u.record_id
    # Each mention's text matches its OWN chunk (a wrong-order batch would cross names).
    assert [m.text for m in batched] == [f"Name{i:03d}" for i in range(7)]


def test_model_called_ceil_n_over_batch_size_times(fake_gliner: type[_FakeGLiNER]) -> None:
    # Batching is REAL: 7 chunks at batch_size=3 => ceil(7/3)=3 forward passes, not 7.
    chunks = _distinct_chunks(7)
    fake_gliner.per_text = _per_text_person_map(chunks)
    extractor = GlinerEntityExtractor({"labels": ["person"], "batch_size": 3})
    list(extractor.extract(chunks))
    assert extractor._model.batch_calls == 3


def test_batch_size_must_be_positive_int() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="batch_size must be >= 1"):
        GlinerEntityExtractor({"labels": ["person"], "batch_size": 0})
    with pytest.raises(ConfigError, match="batch_size must be an int"):
        GlinerEntityExtractor({"labels": ["person"], "batch_size": "8"})


# --- #191 device placement (ADR-0036 §2) -----------------------------------------------------
#
# #191 established that the plain ``gliner`` library NEVER auto-places its model
# (``from_pretrained`` map-locations to CPU) and encoded the fix in ``place_on_device`` — but only
# ``redaction.gliner_pii`` was wired to it. ``entity.gliner`` passed ``device=None`` into
# ``load_then_cast``, so it cast to bf16 off a bare CUDA-availability probe while the weights stayed
# resident on CPU, and its ``device`` config was read by nobody. These tests bite that gap.


def _model_param_device(provider: GlinerEntityExtractor) -> str:
    """Where the LOADED model's parameters actually live — the only honest "what device ran this"."""
    param: _FakeParam = next(iter(provider._model.parameters()))
    return param.device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_entity_model_parameters_reside_on_the_resolved_device(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """#191 regression: the model is PLACED on the resolved device at load, not left on CPU.

    Device-parameterized so CPU-only CI still runs both lanes meaningfully: ``LATENCE_CUDA=1`` is
    the framework's deterministic CUDA probe override, so the ``cuda`` lane exercises the real
    resolution + placement path on a GPU-less box. Pre-fix the provider never called ``.to(device)``
    at all, so the cuda lane found the parameters on ``cpu`` — the silent ~10x throughput cliff.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1" if device == "cuda" else "0")
    extractor = GlinerEntityExtractor({"labels": ["person"], "device": device})
    list(extractor.extract([_chunk("Alice spoke.")]))
    assert _model_param_device(extractor) == device


def test_entity_bf16_cast_never_lands_on_a_cpu_resident_model(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bf16 cast and the resident device agree — never bf16 weights running eager on CPU.

    This is the precise pre-fix corruption: on a CUDA host the seam resolved bf16 from a bare
    availability probe while the model stayed on CPU (``load_then_cast``'s own docstring says the
    cast applies "only when ... the model is on cuda"). Asserting the pair together means neither
    half can drift back.
    """
    fake_torch = types.ModuleType("torch")
    bf16 = object()
    fake_torch.bfloat16 = bf16  # type: ignore[attr-defined]
    fake_torch.float32 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("LATENCE_CUDA", "1")

    extractor = GlinerEntityExtractor({"labels": ["person"], "device": "cuda"})
    list(extractor.extract([_chunk("Alice spoke.")]))
    assert extractor._model.cast_dtype is bf16
    assert _model_param_device(extractor) == "cuda"


def test_entity_mentions_are_identical_across_devices(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The output contract does not move with the device — only throughput does (#191 AC 3)."""
    fake_gliner.returns = [{"start": 0, "end": 5, "text": "Alice", "label": "person", "score": 0.9}]

    def _run(device: str) -> list[tuple[Any, ...]]:
        monkeypatch.setenv("LATENCE_CUDA", "1" if device == "cuda" else "0")
        extractor = GlinerEntityExtractor({"labels": ["person"], "device": device})
        return [
            (
                m.text,
                m.label,
                m.confidence,
                m.provenance.char_start,
                m.provenance.char_end,
                m.provenance.page_start,
            )
            for m in extractor.extract([_chunk("Alice spoke.")])
        ]

    assert _run("cuda") == _run("cpu")
    assert _run("cpu")  # and it is a non-empty comparison, not two empty lists


def test_entity_cuda_requested_without_a_gpu_fails_loudly(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``device: cuda`` on a CPU-only host is an error, NEVER a silent CPU run (#191: silence IS
    the bug). Pre-fix the config key was ignored entirely and the run went to CPU without a word."""
    monkeypatch.setenv("LATENCE_CUDA", "0")
    extractor = GlinerEntityExtractor({"labels": ["person"], "device": "cuda"})
    with pytest.raises(ConfigError, match="no CUDA device is available"):
        list(extractor.extract([_chunk("Alice spoke.")]))


def test_entity_placement_mismatch_is_refused(
    fake_gliner: type[_FakeGLiNER], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that reports the WRONG device after placement is refused, not logged and run.

    Simulates the field failure directly: ``.to(device)`` succeeds but the weights stay on CPU.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    monkeypatch.setattr(_FakeGLiNER, "to", lambda self, arg: self)  # noqa: ARG005
    extractor = GlinerEntityExtractor({"labels": ["person"], "device": "cuda"})
    with pytest.raises(ProviderError, match="reports device 'cpu'"):
        list(extractor.extract([_chunk("Alice spoke.")]))


def test_entity_invalid_device_string_is_a_config_error() -> None:
    """An unknown device is a ConfigError at CONSTRUCTION — before any weights are downloaded."""
    with pytest.raises(ConfigError, match="invalid device config"):
        GlinerEntityExtractor({"labels": ["person"], "device": "gpu"})
