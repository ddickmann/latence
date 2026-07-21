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

import json
import sys
import tomllib
import types
from pathlib import Path
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
from latence_core.providers.perf import (
    DEFAULT_MICROBATCH_MAX_TEXTS,
    DEFAULT_MICROBATCH_TOKEN_BUDGET,
    estimate_token_length,
)
from latence_pii_gliner2 import Gliner2PIIRedactor
from latence_pii_gliner2._compat import SUPPORTED_SPEC


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


class OutOfMemoryError(RuntimeError):
    """A stand-in for ``torch.cuda.OutOfMemoryError`` — matched by class NAME (torch-free).

    ``torch`` raises this (a ``RuntimeError`` subclass) mid-forward when a CUDA allocation exceeds
    free VRAM; the message reads ``"CUDA out of memory. Tried to allocate ..."``. The provider's
    torch-free ``is_cuda_oom`` matches on the class name / message, so a same-named local class
    exercises the exact micro-batch backoff without importing torch or a real GPU.
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
    #: a batched forward over MORE than this many texts raises a CUDA OutOfMemoryError (None = never);
    #: models the span-scoring activation that outgrows VRAM for a large group but fits once halved.
    oom_above_batch: int | None = None
    #: when True, EVERY batched forward OOMs regardless of size — the genuinely-unrecoverable case
    #: (a single chunk that will not fit) the provider must surface as a typed ProviderError.
    oom_always: bool = False

    def __init__(self) -> None:
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
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
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        # SIGNATURE-FAITHFUL to gliner2 1.3.2 (verified against the released wheel, 2026-07-18):
        # ``(texts, entity_types, batch_size=8, threshold=0.5, format_results=True,
        #    include_confidence=False, include_spans=False, max_len=None)``. The parameter ORDER
        # matters: the real 3rd positional is ``batch_size``, NOT ``threshold`` — a Provider that
        # passed threshold positionally would be silently mis-wired against the real library, so
        # the fake must not quietly accept that.
        # The real batched API: one forward pass over all texts; result[i] is texts[i]'s per-label
        # dict — the exact single-text shape, one per input.
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        type(self).last_max_len = max_len  # W12: record the threaded model-window bound
        if type(self).compiled_forward_raises and self._compile_on:
            # Mimic torch._dynamo.exc.BackendCompilerFailed on the FIRST (compiled) forward.
            raise BackendCompilerFailed("inductor backend cannot compile this model")
        oom_ceiling = type(self).oom_above_batch
        if type(self).oom_always or (oom_ceiling is not None and len(texts) > oom_ceiling):
            # Mimic torch.cuda.OutOfMemoryError: the span-scoring activation for THIS many texts
            # exceeds free VRAM. The message mirrors the real one the pod logged (dogfood-2b).
            raise OutOfMemoryError(
                "CUDA out of memory. Tried to allocate 6.28 GiB. "
                "GPU 0 has a total capacity of 31.37 GiB of which 247.81 MiB is free."
            )
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


#: The gliner2 release this repo pins + validated on the pod. The fake module carries it because
#: the REAL gliner2 carries ``__version__`` on every released 1.x (verified against the released
#: distributions, 2026-07-18) and the Provider's #192 compatibility gate reads exactly that.
_PINNED_GLINER2 = "1.3.2"


def _fake_gliner2_module(model_cls: type, version: str | None) -> types.ModuleType:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = model_cls  # type: ignore[attr-defined]
    if version is not None:
        module.__version__ = version  # type: ignore[attr-defined]
    return module


@pytest.fixture()
def fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_FakeGLiNER2]:
    monkeypatch.setitem(
        sys.modules, "gliner2", _fake_gliner2_module(_FakeGLiNER2, _PINNED_GLINER2)
    )
    _FakeGLiNER2.entities = []
    _FakeGLiNER2.per_text = None
    _FakeGLiNER2.raise_on_extract = False
    _FakeGLiNER2.last_kwargs = {}
    _FakeGLiNER2.last_max_len = None
    _FakeGLiNER2.raise_on_optimized_load = False
    _FakeGLiNER2.compiled_forward_raises = False
    _FakeGLiNER2.oom_above_batch = None
    _FakeGLiNER2.oom_always = False
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


# --- dogfood-2b-process: the forward runs under torch.inference_mode (kills the 27 GiB OOM) -------


class _GradTrackingTorch(types.ModuleType):
    """A faithful ``torch`` stand-in that models grad-enabled state around ``inference_mode()``.

    ``is_grad_enabled()`` reads a flag that ``inference_mode()`` flips off for its ``with`` block and
    restores on exit — exactly torch's own behaviour (real ``torch`` is NOT installed offline,
    ADR-0016). A forward that records ``is_grad_enabled()`` at call time therefore proves whether it
    ran INSIDE the inference context. This is the anti-false-green pin: if the provider does NOT enter
    ``inference_context`` around the model call, grad stays enabled and the assertion below fails.
    """

    def __init__(self) -> None:
        super().__init__("torch")
        self.grad_enabled = True

        class _InferenceMode:
            def __init__(self, owner: _GradTrackingTorch) -> None:
                self._owner = owner
                self._prev = True

            def __enter__(self) -> Any:
                self._prev = self._owner.grad_enabled
                self._owner.grad_enabled = False
                return self

            def __exit__(self, *_exc: object) -> None:
                self._owner.grad_enabled = self._prev

        self._inference_mode_cls = _InferenceMode

    def inference_mode(self) -> Any:
        return self._inference_mode_cls(self)

    def is_grad_enabled(self) -> bool:
        return self.grad_enabled


class _GradRecordingModel:
    """A gliner2-shaped model whose ``batch_extract_entities`` RECORDS grad state at forward time.

    The signature is faithful to gliner2 1.3.2 (``batch_size`` is the 3rd positional, NOT
    ``threshold``), so a mis-wired positional call fails here rather than false-greening. The forward
    reads ``torch.is_grad_enabled()`` from the injected fake ``torch`` — the same object
    ``inference_context`` toggled — so it observes exactly whether the provider entered the context.
    """

    def __init__(self, torch_mod: _GradTrackingTorch) -> None:
        self._torch = torch_mod
        self.grad_at_forward: bool | None = None

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: list[str],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        self.grad_at_forward = self._torch.is_grad_enabled()
        return [{"entities": {}} for _ in texts]


def test_extract_forward_runs_under_inference_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """REGRESSION (dogfood-2b-process): the ``batch_extract_entities`` forward runs with autograd OFF.

    A redaction-only encoder never backpropagates; a forward under the default grad-enabled mode
    still builds an autograd graph and pins its activations alive, ACCUMULATING VRAM run-over-run to
    the pod's 27.28 GiB CUDA OOM at the redact stage. The cure is running the forward inside
    ``inference_context`` (``torch.inference_mode()``). This test injects a faithful fake ``torch``
    whose grad flag ``inference_mode()`` toggles, and a model that records the flag at forward time —
    proving the provider entered the context. If the ``inference_context`` wrap is removed from
    ``_extract_once``, grad stays enabled and ``grad_at_forward`` is True, failing the assertion.
    """
    fake_torch = _GradTrackingTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert fake_torch.is_grad_enabled() is True  # default: grad ON before any forward

    provider = Gliner2PIIRedactor(_config())
    model = _GradRecordingModel(fake_torch)
    provider._extract_once(model, ["Reach Alice."], ["person"])

    assert model.grad_at_forward is False  # the forward ran with autograd DISABLED (inference_mode)
    assert fake_torch.is_grad_enabled() is True  # and the context restored grad on exit


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


# --- dogfood-2b-process: token-budget micro-batching decouples the forward from batch_size -----


class _RecordingGLiNER2(_FakeGLiNER2):
    """A faithful :class:`_FakeGLiNER2` that ALSO records the texts of every batched forward.

    Instance-level ``calls`` (one entry per ``batch_extract_entities`` call = the exact list of
    texts fed to it) lets the regression test read back BOTH the num_texts and the summed token
    length of every native forward, and prove the provider splits a large ``batch_size`` group into
    token-budget MICRO-BATCHES instead of one giant padded tensor (the measured 32 GB OOM). It
    delegates to the real fake for behaviour (per-label dict shape, threshold/label filtering, the
    OOM/compile hooks), so it rejects exactly what the real gliner2 API rejects.
    """

    def __init__(self) -> None:
        super().__init__()
        #: one entry per batched forward: the texts handed to it, in call order.
        self.calls: list[list[str]] = []

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: list[str],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        self.calls.append(list(texts))
        return super().batch_extract_entities(
            texts,
            entity_types,
            batch_size,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len,
        )


def _long_distinct_chunks(n: int, body_chars: int) -> list[ChunkRecord]:
    """N distinct chunks, each padded to ``body_chars`` so its estimated token length is large.

    Like :func:`_distinct_chunks` (a distinct ``PersonNNN`` at chunk-offset ``i`` → distinct spans,
    so a mis-ordered micro-batch map is caught), but each chunk carries a long body — enough that a
    whole ``batch_size`` group would pad into an OOM-sized activation without micro-batching.
    """
    filler = " lorem ipsum dolor sit amet consectetur"
    chunks: list[ChunkRecord] = []
    for i in range(n):
        head = ("." * i) + f"Person{i:03d} was here."
        pad = (filler * (body_chars // len(filler) + 1))[:body_chars]
        chunks.append(
            _chunk(
                head + pad,
                record_id=f"doc{i}#chunk:0",
                document_record_id=f"doc{i}",
                index=0,
            )
        )
    return chunks


@pytest.fixture()
def recording_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingGLiNER2]:
    monkeypatch.setitem(
        sys.modules, "gliner2", _fake_gliner2_module(_RecordingGLiNER2, _PINNED_GLINER2)
    )
    # Reset the inherited class-level knobs (a prior test may have set them on the base class).
    _RecordingGLiNER2.entities = []
    _RecordingGLiNER2.per_text = None
    _RecordingGLiNER2.raise_on_extract = False
    _RecordingGLiNER2.last_kwargs = {}
    _RecordingGLiNER2.last_max_len = None
    _RecordingGLiNER2.raise_on_optimized_load = False
    _RecordingGLiNER2.compiled_forward_raises = False
    _RecordingGLiNER2.oom_above_batch = None
    _RecordingGLiNER2.oom_always = False
    return _RecordingGLiNER2


def test_large_batch_size_split_into_token_budget_microbatches(
    recording_gliner2: type[_RecordingGLiNER2],
) -> None:
    """dogfood-2b-process: a large ``batch_size`` group is split into token-budget micro-batches.

    The confirmed OOM: the provider fed a whole ``batch_size`` group (default 32) of long chunks
    into ONE ``batch_extract_entities`` call; gliner2 padded them into a single tensor whose
    span-scoring activation (~num_texts × seq_len) demanded ~17 GiB and OOM'd on a 32 GB card. The
    cure decouples the forward from the progress ``batch_size``: each forward carries a MICRO-BATCH
    bounded by BOTH a text-count cap and a token budget. This test feeds a large ``batch_size`` (32)
    of long chunks and asserts (a) MULTIPLE forwards happened, (b) NO forward exceeded either bound,
    and (c) the output is record-for-record identical to the one-chunk-per-call reference.

    It FAILS on the pre-fix code (one giant call of all 10 texts — 10 > max_texts and
    10 × 768 tokens ≫ the budget), which is what makes it an anti-false-green regression.
    """
    chunks = _long_distinct_chunks(10, body_chars=4000)
    recording_gliner2.per_text = _per_text_person_map(chunks)
    cfg = {"pii_labels": ["person"], "batch_size": 32}

    provider = Gliner2PIIRedactor(cfg)
    batched = list(provider.redact(chunks))
    calls = provider._model.calls

    # (a) The whole group did NOT go through one giant forward — it was micro-batched.
    assert len(calls) >= 2, f"expected multiple micro-batch forwards, got {len(calls)}"
    max_len = 768  # the default gliner window the provider threads into the call
    for texts in calls:
        # (b1) the text-count cap — the PRIMARY bound on the padded worst case (max_texts × max_len)
        assert len(texts) <= DEFAULT_MICROBATCH_MAX_TEXTS
        # (b2) the token budget — summed per-text token length (each capped at max_len)
        total = sum(estimate_token_length(t, max_len) for t in texts)
        assert total <= DEFAULT_MICROBATCH_TOKEN_BUDGET
    # every chunk was still scanned exactly once (the micro-batches tile the group in order)
    assert sum(len(texts) for texts in calls) == len(chunks)

    # (c) batch-invariance: identical to the one-chunk-per-forward path, record for record.
    unbatched = [
        r
        for c in chunks
        for r in Gliner2PIIRedactor({**cfg, "batch_size": 1}).redact([c])
    ]
    assert len(batched) == len(unbatched) == 10
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


def test_token_budget_bites_before_count_when_max_len_raised(
    recording_gliner2: type[_RecordingGLiNER2],
) -> None:
    """The token budget (not only the count cap) bounds the forward when ``max_len`` is raised.

    With a larger window each long chunk contributes more tokens, so the token budget forces
    micro-batches SMALLER than ``max_texts`` — proving the budget is a real, independent bound (the
    ``max_texts`` cap alone would let a bigger padded tensor through at a raised window).
    """
    chunks = _long_distinct_chunks(4, body_chars=12000)
    recording_gliner2.per_text = _per_text_person_map(chunks)
    # max_len 2048 → each chunk estimates at ~2048 tokens, so 2 per forward already exceed the 3072
    # budget → micro-batches of ONE, well below the max_texts=4 count cap.
    provider = Gliner2PIIRedactor(
        {"pii_labels": ["person"], "batch_size": 32, "max_len": 2048}
    )
    list(provider.redact(chunks))
    calls = provider._model.calls

    assert len(calls) >= 2
    assert max(len(texts) for texts in calls) < DEFAULT_MICROBATCH_MAX_TEXTS
    for texts in calls:
        total = sum(estimate_token_length(t, 2048) for t in texts)
        assert total <= DEFAULT_MICROBATCH_TOKEN_BUDGET


def test_microbatch_bound_config_must_be_positive_int() -> None:
    from latence_core.errors import ConfigError

    with pytest.raises(ConfigError, match="microbatch_max_texts"):
        Gliner2PIIRedactor({"pii_labels": ["person"], "microbatch_max_texts": 0})
    with pytest.raises(ConfigError, match="microbatch_token_budget"):
        Gliner2PIIRedactor({"pii_labels": ["person"], "microbatch_token_budget": -1})


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
    """#192: a checkpoint without the batched API fails TYPED (ADR-0034), not as a native fault.

    This test previously asserted a bare ``AttributeError`` — and that is exactly what the operator
    got on the live pod run: unactionable. The Provider must name the batched API, the supported
    gliner2 range, and the install command, as a framework :class:`ProviderError`.
    """
    monkeypatch.setitem(
        sys.modules, "gliner2", _fake_gliner2_module(_NoBatchGLiNER2, _PINNED_GLINER2)
    )
    with pytest.raises(ProviderError, match="batch_extract_entities") as excinfo:
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))
    assert not isinstance(excinfo.value, AttributeError)
    assert SUPPORTED_SPEC in str(excinfo.value)


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


# --- dogfood-2b-process: CUDA-OOM micro-batch backoff at redaction (real-hardware fix) ----------


def test_cuda_oom_recovers_by_micro_batching(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """REGRESSION (dogfood-2b-process): a batched forward that OOMs is split + retried, not crashed.

    On the tutorial's own 32 GB Blackwell pod, ``latence process`` (enterprise-sota) died Exit 1 at
    the redact stage with a native CUDA ``OutOfMemoryError`` — GLiNER2's span-scoring activation for
    a whole group of long chunks demanded a single 6.28 GiB allocation the GPU could not satisfy, so
    NO run dir / exports were ever written. Because the emitted RedactionRecords are
    batch-size-invariant (W7), the provider must split the OOM'd group in half and retry smaller
    until it fits — masking every chunk, never dropping one (redaction floor, ADR-0044).

    Pre-fix ``_batch_extract`` issued ONE forward over the group and let the OOM propagate through the
    guard as a ``ProviderError`` → this test FAILS (the whole redact call raises, zero records). With
    the backoff it splits until each forward carries ``<= oom_above_batch`` texts and completes.
    """
    chunks = _distinct_chunks(8)
    fake_gliner2.per_text = _per_text_person_map(chunks)
    # A forward over > 2 texts OOMs; halving 8 -> 4 (OOM) -> 2 (fits) recovers every chunk.
    fake_gliner2.oom_above_batch = 2
    provider = Gliner2PIIRedactor({"pii_labels": ["person"], "batch_size": 8})

    recs = list(provider.redact(chunks))

    assert len(recs) == 8
    for i, rec in enumerate(recs):
        assert rec.masked_content is not None
        assert f"Person{i:03d}" not in rec.masked_content  # every chunk actually scanned + masked
        assert "[PERSON]" in rec.masked_content
    # The successful forwards ran at the recovered size (<= the OOM ceiling), proving the split bit —
    # every text that actually got scanned went through a forward small enough to fit.
    assert provider._model.batch_sizes  # forwards happened
    assert any(size <= 2 for size in provider._model.batch_sizes)


def test_cuda_oom_output_is_identical_to_the_no_oom_path(fake_gliner2: type[_FakeGLiNER2]) -> None:
    """The recovered output is byte-identical to the batch that never OOM'd (backoff is transparent)."""
    chunks = _distinct_chunks(6)
    fake_gliner2.per_text = _per_text_person_map(chunks)
    cfg = {"pii_labels": ["person"], "batch_size": 6}

    clean = list(Gliner2PIIRedactor(cfg).redact(chunks))
    fake_gliner2.oom_above_batch = 1  # force maximal splitting on the second run
    recovered = list(Gliner2PIIRedactor(cfg).redact(chunks))

    assert len(recovered) == len(clean) == 6
    for r, c in zip(recovered, clean, strict=True):
        assert r.record_id == c.record_id
        assert r.masked_content == c.masked_content
        assert [(s.char_start, s.char_end, s.pii_type) for s in r.pii_spans] == [
            (s.char_start, s.char_end, s.pii_type) for s in c.pii_spans
        ]


def test_unrecoverable_cuda_oom_surfaces_typed_error_no_dropped_chunk(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    """No fake green: an OOM that persists even at batch-of-one is a typed ProviderError, not a drop.

    The backoff halves down to a single chunk; if THAT still OOMs the hardware genuinely cannot run
    the stage, so the provider must surface a typed :class:`ProviderError` (ADR-0034) — never emit a
    chunk un-scanned or silently skip it (redaction floor, ADR-0044). The raised error carries no PII.
    """
    chunks = _distinct_chunks(4)
    fake_gliner2.per_text = _per_text_person_map(chunks)
    fake_gliner2.oom_always = True  # even a batch of one will not fit
    provider = Gliner2PIIRedactor({"pii_labels": ["person"], "batch_size": 4})

    with pytest.raises(ProviderError) as excinfo:
        list(provider.redact(chunks))
    assert "redaction.gliner2" in str(excinfo.value)
    # No PII leaked into the typed error message (guard discipline: type name only, no content).
    assert "Person" not in str(excinfo.value)


# --- W9-scaling (audit R4): compile is OFF by default on cuda too, like the twin extractor -------


def test_compile_defaults_off_even_on_cuda(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4 regression: compile is defaulted OFF (opt-in) EVEN on cuda — the recompile-thrash guard.

    The redactor drives ``batch_extract_entities`` over the same varied-length production chunk
    stream the twin ``fused_entity_relation.gliner2`` extractor does, and gliner2 compiles with
    ``dynamic=True`` — pod-measured to pin the worker at ~290% CPU with the GPU at 0% and no
    progress. The extractor forces ``compile=False`` at construction for exactly this reason; the
    redactor must too. Pre-fix this Provider had no default of its own, so the generic
    ``resolve_gliner2_knob`` default (``auto`` → True on cuda) turned compile ON. ``quantize`` is
    untouched: the fp16 win still resolves True on cuda.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    cfg = {**_config(), "device": "cuda"}
    list(Gliner2PIIRedactor(cfg).redact([_chunk("Alice waved.")]))
    assert fake_gliner2.last_kwargs.get("quantize") is True
    assert fake_gliner2.last_kwargs.get("compile") is False


def test_compile_opt_in_still_enables_on_cuda(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The knob is defaulted off, not removed: an operator who explicitly asks for compile gets it.
    monkeypatch.setenv("LATENCE_CUDA", "1")
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    cfg = {**_config(), "device": "cuda", "compile": True}
    list(Gliner2PIIRedactor(cfg).redact([_chunk("Alice waved.")]))
    assert fake_gliner2.last_kwargs.get("compile") is True


@pytest.mark.parametrize("profile", ["lite", "enterprise-sota"])
def test_wizard_blessed_redact_config_never_enables_compile(
    fake_gliner2: type[_FakeGLiNER2], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str
) -> None:
    """The REAL emitted config, not a hand-written dict: the blessed redact stage never compiles.

    ``latence setup --profile enterprise-sota --pii`` emits the redact stage with ``device: cuda``
    and no ``compile`` key, so on the documented GPU pod the knob resolved True and the redact Stage
    thrashed. The ``_reload_uncompiled`` latch cannot save it — the thrash is silent slowness, not a
    ``BackendCompilerFailed``. This drives the wizard's own output through the Provider.
    """
    from latence_core.pipeline import Pipeline
    from latence_core.setup.wizard import (
        BLESSED_REDACT_PROVIDER,
        SetupAnswers,
        build_stack_config,
    )

    monkeypatch.setenv("LATENCE_CUDA", "1")
    src = tmp_path / "corpus"
    src.mkdir()
    (src / "a.txt").write_text("Alice waved.", encoding="utf-8")
    text = build_stack_config(
        SetupAnswers(
            source=f"file://{src}",
            storage=f"file://{tmp_path}/out",
            parser="parser.render",
            device="cuda",
            pii=True,
            extract_on="unmasked",
            profile=profile,
            extensions=("txt",),
        )
    )
    stage = next(s for s in Pipeline.from_yaml_text(text).stages if s.name == "redact")
    assert stage.provider == BLESSED_REDACT_PROVIDER
    assert "compile" not in stage.config  # the wizard stays silent; the Provider owns the default
    fake_gliner2.entities = [{"start": 0, "end": 5, "label": "person", "confidence": 0.9}]
    list(Gliner2PIIRedactor(dict(stage.config)).redact([_chunk("Alice waved.")]))
    assert fake_gliner2.last_kwargs.get("compile") is False


# ---------------------------------------------------------------------------
# #192 — gliner2 version/API compatibility. The field failure: a fresh env resolved a
# gliner2 the Provider cannot drive and the load died with a NATIVE AttributeError, which is
# unactionable for an operator. Two gates now: the package pin (resolution) and a typed
# ProviderError (runtime), and the ADR-0044 financial-PII floor must survive both.
# ---------------------------------------------------------------------------


def _gliner2_specifier() -> str:
    """The ``gliner2`` requirement string declared in this package's own metadata."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = [d for d in data["project"]["dependencies"] if d.startswith("gliner2")]
    assert len(deps) == 1, deps
    return str(deps[0])


def test_package_pins_the_supported_gliner2_range() -> None:
    """The RESOLUTION gate (ADR-0016): the package pin excludes every unsupported gliner2.

    Pre-#192 the pin was ``gliner2[local]>=1.0,<2``, which resolves happily to 1.0.x — a line with
    no ``batch_extract_entities`` at all. The floor must be the supported one, and it must equal
    the range the runtime gate reports so metadata and error message cannot drift.
    """
    assert _gliner2_specifier() == f"gliner2[local]{SUPPORTED_SPEC}"


@pytest.mark.parametrize("version", ["1.0.0", "1.0.2", "1.1.0", "1.2.0", "1.2.6", "2.0.0"])
def test_runtime_gate_rejects_every_unsupported_gliner2_line(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """Each line the pin excludes is ALSO refused at runtime — the two gates agree.

    1.0.x has no ``batch_extract_entities``; 1.1.x rejects ``include_spans``/``max_len``;
    1.2.0-1.2.5 reject ``max_len``; 1.2.6 predates the pod-validated line; 2.x is past the major
    cap. (Verified against the released gliner2 distributions, 2026-07-18.)
    """
    monkeypatch.setitem(sys.modules, "gliner2", _fake_gliner2_module(_FakeGLiNER2, version))
    with pytest.raises(ProviderError, match=r"requires gliner2 >=1\.3,<2"):
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))


@pytest.mark.parametrize("version", ["1.3", "1.3.0", "1.3.2", "1.3.2.post1", "1.9.9"])
def test_runtime_gate_accepts_every_supported_gliner2_line(
    monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    """The gate is a gate, not a wall: every version the pin admits still loads and redacts."""
    monkeypatch.setitem(sys.modules, "gliner2", _fake_gliner2_module(_FakeGLiNER2, version))
    a_start = _TEXT.index("Alice")
    _FakeGLiNER2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95}
    ]
    _FakeGLiNER2.per_text = None
    _FakeGLiNER2.raise_on_extract = False
    _FakeGLiNER2.raise_on_optimized_load = False
    _FakeGLiNER2.compiled_forward_raises = False
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content


def test_unsupported_gliner2_version_is_a_typed_error_not_a_native_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An env with gliner2 1.0.x fails as a TYPED ProviderError naming the fix (ADR-0034).

    The check runs BEFORE ``from_pretrained``, so no weights are downloaded first.
    """
    # _NoBatchGLiNER2 mirrors the 1.0.x surface faithfully: ``extract_entities`` only, no batched
    # entry point (verified against the released gliner2 1.0.0 distribution, 2026-07-18).
    monkeypatch.setitem(sys.modules, "gliner2", _fake_gliner2_module(_NoBatchGLiNER2, "1.0.2"))
    with pytest.raises(ProviderError) as excinfo:
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))
    message = str(excinfo.value)
    assert "gliner2 >=1.3,<2" in message          # the supported range
    assert "1.0.2" in message                     # what is actually installed
    assert "pip install 'gliner2[local]>=1.3,<2'" in message   # the actionable fix
    # And it is NOT the native fault the operator got in the field.
    assert not isinstance(excinfo.value, AttributeError)


def test_unparseable_gliner2_version_is_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A nonsense version string is refused (typed), not optimistically accepted."""
    monkeypatch.setitem(sys.modules, "gliner2", _fake_gliner2_module(_FakeGLiNER2, "nightly"))
    with pytest.raises(ProviderError, match=r"gliner2 >=1\.3,<2"):
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))


def test_supported_gliner2_loads_and_redacts_the_financial_pii_floor(
    fake_gliner2: type[_FakeGLiNER2],
) -> None:
    """AC1 + the ADR-0044 floor: on the PINNED gliner2 the Provider loads and still redacts.

    The version gate must be a gate, not a wall: with gliner2 1.3.2 (the pinned line) the load
    proceeds and the financial-PII surface — IBAN, credit card, tax id — is masked exactly as
    before, with the raw values absent from both the masked text and the emitted spans.
    """
    # Reuses the module's audited _IBAN; the card / tax-id surfaces are deliberately non-numeric
    # placeholders (the secret-scan gate flags a real-looking PAN / high-entropy id). The fake model
    # detects exactly the spans the test seeds, so the surface FORM is irrelevant to the behaviour.
    card, tax = "4111-XXXX-XXXX-0000", "TAX-ID-PLACEHOLDER"
    text = f"IBAN {_IBAN}, card {card}, tax id {tax}."
    iban = _IBAN
    fake_gliner2.entities = [
        {"start": text.index(v), "end": text.index(v) + len(v), "label": lbl, "confidence": 0.97}
        for v, lbl in ((iban, "iban"), (card, "credit card number"), (tax, "tax id"))
    ]
    cfg = {"pii_labels": ["iban", "credit card number", "tax id"]}
    rec = next(iter(Gliner2PIIRedactor(cfg).redact([_chunk(text)])))
    assert rec.masked_content is not None
    for raw in (iban, card, tax):
        assert raw not in rec.masked_content
        assert all(raw not in str(span) for span in rec.pii_spans)
    assert {s.pii_type for s in rec.pii_spans} == {"iban", "credit card number", "tax id"}
    assert [text[s.char_start : s.char_end] for s in rec.pii_spans] == [iban, card, tax]


def test_version_gate_runs_before_any_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is cheap-first: an unsupported version never reaches ``from_pretrained``."""
    loaded: list[str] = []

    class _Tripwire(_FakeGLiNER2):
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeGLiNER2:
            loaded.append(model_id)
            return super().from_pretrained(model_id, **kwargs)

    monkeypatch.setitem(sys.modules, "gliner2", _fake_gliner2_module(_Tripwire, "1.2.6"))
    with pytest.raises(ProviderError):
        list(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)]))
    assert loaded == []


# ---------------------------------------------------------------------------
# R5 (audit) — cuda placement. The #191 fp16-eager GPU-idle shape: a quantize=True (fp16) model
# that silently stays on CPU is "pathologically slow and CPU-bound" (ensure_gliner2_on_cuda's own
# docstring). ``load_gliner2_optimized`` uses ``device`` ONLY to resolve the quantize/compile knobs
# — it performs NO placement — so a Provider that omits the ``ensure_gliner2_on_cuda`` seam runs
# fp16 eager on CPU with the GPU at 0%, correct output, and no warning. The twin
# ``fused_entity_relation.gliner2`` calls the seam at BOTH load sites; #197 made THIS Provider the
# wizard's blessed default redactor in both profiles.
# ---------------------------------------------------------------------------


class _PlacingGLiNER2(_FakeGLiNER2):
    """Records ``.to(target)`` calls + exposes a ``device`` so cuda placement is observable."""

    to_calls: list[str] = []
    device: str = "cpu"

    def to(self, target: str) -> _PlacingGLiNER2:
        type(self).to_calls.append(target)
        type(self).device = target
        return self


@pytest.fixture()
def placing_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_PlacingGLiNER2]:
    monkeypatch.setitem(
        sys.modules, "gliner2", _fake_gliner2_module(_PlacingGLiNER2, _PINNED_GLINER2)
    )
    _PlacingGLiNER2.entities = []
    _PlacingGLiNER2.per_text = None
    _PlacingGLiNER2.raise_on_extract = False
    _PlacingGLiNER2.last_kwargs = {}
    _PlacingGLiNER2.last_max_len = None
    _PlacingGLiNER2.raise_on_optimized_load = False
    _PlacingGLiNER2.compiled_forward_raises = False
    _PlacingGLiNER2.to_calls = []
    _PlacingGLiNER2.device = "cpu"
    return _PlacingGLiNER2


def test_fp16_model_moved_to_cuda_when_gpu_present(
    placing_gliner2: type[_PlacingGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: when a GPU is present the (fp16) redactor model is explicitly moved to cuda.

    A Provider that trusted ``from_pretrained``/``load_gliner2_optimized`` to place it — and left
    the fp16 model on CPU — FAILS here with ``to_calls == []``.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    a_start = _TEXT.index("Alice")
    placing_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95}
    ]
    provider = Gliner2PIIRedactor({**_config(), "device": "cuda"})
    rec = next(iter(provider.redact([_chunk(_TEXT)])))
    # The knobs that make the CPU-resident model pathological were in fact applied (fp16).
    assert placing_gliner2.last_kwargs.get("quantize") is True
    # …and the model is on cuda, so fp16 lands where fp16 belongs.
    assert placing_gliner2.to_calls == ["cuda"]
    assert placing_gliner2.device == "cuda"
    # Placement is throughput-only: the redaction output is unchanged.
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content


def test_no_cuda_no_move(
    placing_gliner2: type[_PlacingGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a CPU-only host the model is NOT moved — placement is a cuda-only concern (off-by-default
    # is byte-identical), and masking is unaffected.
    monkeypatch.setenv("LATENCE_CUDA", "0")
    a_start = _TEXT.index("Alice")
    placing_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95}
    ]
    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))
    assert placing_gliner2.to_calls == []
    assert placing_gliner2.device == "cpu"
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content


def test_uncompiled_reload_also_placed_on_cuda(
    placing_gliner2: type[_PlacingGLiNER2], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5: the compile-fault reload path is placed on cuda too.

    ``_reload_uncompiled`` re-enters ``load_gliner2_optimized`` (which never places), so a fix that
    only patched ``_load_model`` would leave the post-fallback model — the one that serves the REST
    of the run — fp16 on CPU. Two loads ⇒ two placements.
    """
    monkeypatch.setenv("LATENCE_CUDA", "1")
    placing_gliner2.compiled_forward_raises = True
    a_start = _TEXT.index("Alice")
    placing_gliner2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95}
    ]
    provider = Gliner2PIIRedactor({**_config(), "quantize": True, "compile": True})
    rec = next(iter(provider.redact([_chunk(_TEXT)])))
    assert provider._compile_disabled is True
    assert placing_gliner2.last_kwargs.get("compile") is False  # the reload happened
    assert placing_gliner2.to_calls == ["cuda", "cuda"]  # initial load + uncompiled reload
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content


# ---------------------------------------------------------------------------
# dogfood-2b-process: the default PII checkpoint's tokenizer_config.json stores
# ``extra_special_tokens`` as a LIST, which transformers >= 4.57 (pulled transitively via
# gliner2[local]) rejects at tokenizer build with
# ``AttributeError: 'list' object has no attribute 'keys'`` — killing ``latence process`` at the
# mandatory redact stage before any run dir / export is written. The provider now coerces that field
# to the dict shape 4.57 requires BEFORE the load (byte-identical tokenization). These tests inject a
# faithful ``huggingface_hub`` + a ``from_pretrained`` that reproduces the REAL transformers loader,
# so the field crash — and its fix — are exercised deterministically and offline (no torch download).
# ---------------------------------------------------------------------------

#: The 10 GLiNER2 structural special tokens the real ``fastino/GLiNER2-Guardrails-PII-Multi``
#: ``tokenizer_config.json`` stores as a LIST (verified against the live HF file, 2026-07-21). Every
#: one is ALSO a ``special`` added token in that checkpoint's ``tokenizer.json``, so coercing the
#: list to a ``{token: token}`` dict is byte-identical in tokenization — only the config *shape* changes.
_STRUCT_TOKENS = [
    "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]", "[L]",
    "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]",
]


def _transformers_set_model_specific_special_tokens(extra: Any) -> list[str]:
    """FAITHFUL stand-in for transformers >= 4.57 ``_set_model_specific_special_tokens``.

    The 4.57 loader treats ``extra_special_tokens`` as a **dict** ``{name: token}`` — it calls
    ``special_tokens.keys()`` then ``.items()`` on it. So a **list** raises exactly
    ``AttributeError: 'list' object has no attribute 'keys'`` (the field crash), and a **dict** whose
    values are strings is accepted — the precise accept/reject boundary the real dep enforces, so this
    fake is not a false-green (a list is rejected here iff transformers 4.57 rejects it).
    """
    added = list(extra.keys())  # dies on a list with 'list' object has no attribute 'keys'
    for key, value in extra.items():
        if not isinstance(value, str):
            msg = f"Special token {key} has to be either str or AddedToken but got: {type(value)}"
            raise TypeError(msg)
    return added


class _TokenizerLoadingGLiNER2(_FakeGLiNER2):
    """A ``_FakeGLiNER2`` whose ``from_pretrained`` ALSO builds the tokenizer the way transformers
    >= 4.57 does — it resolves the checkpoint's cached ``tokenizer_config.json`` (via the injected
    ``huggingface_hub.hf_hub_download``) and runs the faithful ``_set_model_specific_special_tokens``.

    This is what reproduces the dogfood field crash offline: a list-valued config raises
    ``AttributeError`` at load; a dict-valued one loads cleanly.
    """

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeGLiNER2:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "tokenizer_config.json")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        # The exact transformers 4.57 tokenizer-build step that crashed on the pod.
        _transformers_set_model_specific_special_tokens(config.get("extra_special_tokens", {}))
        return super().from_pretrained(model_id, **kwargs)


def _fake_huggingface_hub_module(config_path: Path) -> types.ModuleType:
    """A ``huggingface_hub`` whose ``hf_hub_download`` returns our fake-cache tokenizer_config path.

    The SAME path the shim patches and the fake ``from_pretrained`` reads back — so a patch by the
    shim is observed by the subsequent load, exactly as it is on a real shared HF cache.
    """
    module = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id: str, filename: str, **_kwargs: Any) -> str:
        if filename != "tokenizer_config.json":
            msg = f"unexpected file requested from the fake hub: {filename!r}"
            raise FileNotFoundError(msg)
        return str(config_path)

    module.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    return module


@pytest.fixture()
def list_valued_pii_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """A fake HF cache whose default-PII-checkpoint ``tokenizer_config.json`` is the CRASHING shape.

    ``extra_special_tokens`` is a *list* (the real, field-failing shape), and both the shim and the
    faithful ``from_pretrained`` resolve it through the injected ``huggingface_hub``. Returns the
    config path so a test can assert it was coerced list -> dict.
    """
    config_path = tmp_path / "tokenizer_config.json"
    config_path.write_text(
        json.dumps(
            {
                "tokenizer_class": "DebertaV2Tokenizer",
                "extra_special_tokens": list(_STRUCT_TOKENS),  # the shape transformers 4.57 rejects
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_huggingface_hub_module(config_path))
    monkeypatch.setitem(
        sys.modules, "gliner2", _fake_gliner2_module(_TokenizerLoadingGLiNER2, _PINNED_GLINER2)
    )
    _FakeGLiNER2.entities = []
    _FakeGLiNER2.per_text = None
    _FakeGLiNER2.raise_on_extract = False
    _FakeGLiNER2.last_kwargs = {}
    _FakeGLiNER2.last_max_len = None
    _FakeGLiNER2.raise_on_optimized_load = False
    _FakeGLiNER2.compiled_forward_raises = False
    return config_path


def test_list_extra_special_tokens_crashes_transformers_faithful_load(
    list_valued_pii_checkpoint: Path,  # noqa: ARG001 - fixture installs the crashing checkpoint
) -> None:
    """ANCHOR (anti-false-green): the offline fake must reproduce the REAL transformers >= 4.57 crash.

    A LIST-valued ``extra_special_tokens`` must raise ``AttributeError: 'list' object has no attribute
    'keys'`` at ``from_pretrained`` — the exact pod failure. If this fake ever silently tolerated a
    list, the regression below would pass for the wrong reason (a load that never crashed), so this
    anchor pins the fake to the real accept/reject boundary.
    """
    import gliner2

    with pytest.raises(AttributeError, match="'list' object has no attribute 'keys'"):
        gliner2.GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi")


def test_process_survives_list_valued_extra_special_tokens(
    list_valued_pii_checkpoint: Path,
) -> None:
    """REGRESSION (dogfood-2b-process): redact must NOT crash on the default PII checkpoint.

    With the checkpoint's ``extra_special_tokens`` stored as a list, the pre-fix load died with a
    ``ProviderError`` (native ``AttributeError``) at the mandatory redact stage, so ``latence process``
    exited non-zero and wrote no run dir. The shim coerces the cached config to the dict shape
    transformers >= 4.57 requires before the load, so redact loads the model, completes, and yields a
    record — and the cached config is now dict-shaped.
    """
    a_start = _TEXT.index("Alice")
    _TokenizerLoadingGLiNER2.entities = [
        {"start": a_start, "end": a_start + 5, "label": "person", "confidence": 0.95},
    ]

    rec = next(iter(Gliner2PIIRedactor(_config()).redact([_chunk(_TEXT)])))

    # The load no longer crashes: redact runs end-to-end and the PII is masked.
    assert isinstance(rec, ChunkRecord)
    assert rec.masked_content is not None
    assert "Alice" not in rec.masked_content

    # The cached tokenizer_config was coerced list -> dict — the exact shape transformers 4.57
    # iterates, keyed token->token (every struct token preserved as a model-specific special token).
    patched = json.loads(list_valued_pii_checkpoint.read_text(encoding="utf-8"))
    assert patched["extra_special_tokens"] == {token: token for token in _STRUCT_TOKENS}
