"""#190 — the fused GLiNER2 Provider survives the Blackwell/CUDA-13 TorchScript/nvrtc fault.

Field evidence (live 519-page German-corpus run, 2026-07-17): ``fused_entity_relation.gliner2``
crashed at inference with a native ``failed to open libnvrtc-builtins.so.13.0``, raised from the
mdeberta-v3 encoder's ``@torch.jit.script``-ed ``build_relative_position``. The W9 compile guard
covers ``torch.compile`` (dynamo/inductor) and NOT the TorchScript JIT — that gap was the defect.
The fix is :func:`latence_core.providers.perf.neutralize_torch_jit`, applied at the SHARED perf
seam (``load_gliner2_optimized`` / ``load_then_cast``), so every mdeberta-backed Provider inherits
it rather than each copying an env-var workaround.

Anti-false-green: the fake ``torch`` here mirrors the REAL JIT switch semantics — ``torch.jit.script``
returns the function unchanged only while ``torch.jit._state._enabled.enabled`` is False, and
otherwise returns a ScriptFunction that dies with the exact nvrtc message when CALLED (the real
Blackwell behaviour). The fake ``gliner2`` model scripts its relative-position helper at
``from_pretrained`` and calls it inside every ``extract_entities`` / ``extract_relations``, so with
the JIT live this suite reproduces the production crash and can only pass if the JIT is genuinely
neutralized by the time the model loads.

**What this suite does NOT cover (audit R1 / #190-4).** Load-time scripting is *later* than the
real dependency: transformers evaluates its ``@torch.jit.script`` decorators at MODULE IMPORT, i.e.
at the Provider's ``from gliner2 import GLiNER2`` — before the seam is entered. This suite is
therefore blind to that ordering and green either way; the ordering is owned by
``latence-core/tests/test_perf_jit_import_ordering.py`` (behavioural, an import-time-scripting
fake) and by ``test_perf_fp32_regression_guard.py::
test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone`` (structural, every
Provider source). ``test_output_is_byte_identical_to_the_jit_free_path``
closes the other half of the ADR-0036 discipline: the neutralization is throughput-only, so the
emitted mentions/relations must be record-for-record identical to a model that never scripted.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import pytest
from latence_core.contracts import (
    ChunkRecord,
    Classification,
    PageSlice,
    PageSpan,
    Provenance,
)
from latence_core.errors import ProviderError
from latence_extract_gliner2 import Gliner2FusedExtractor

TEXT = "Acme Corp employs Ada Lovelace in Berlin."

# Chunk-local char spans over TEXT, seeded into the fake model's output.
_ENTITIES: list[dict[str, Any]] = [
    {"start": 0, "end": 9, "label": "organization", "confidence": 0.91},
    {"start": 18, "end": 30, "label": "person", "confidence": 0.88},
]
_RELATIONS: list[dict[str, Any]] = [
    {
        "relation": "employs",
        "head": {"text": "Acme Corp", "start": 0, "end": 9, "confidence": 0.83},
        "tail": {"text": "Ada Lovelace", "start": 18, "end": 30, "confidence": 0.79},
    }
]


def _chunk() -> ChunkRecord:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=100,
        char_end=100 + len(TEXT),
        page_start=2,
        page_end=2,
    )
    return ChunkRecord(
        record_id="doc1#chunk-0",
        document_record_id="doc1",
        index=0,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
        content=TEXT,
        token_count=len(TEXT.split()),
        # v19: a one-page fixture — the chunk's own page spans are that single page.
        page_slice=PageSlice(
            pages=[PageSpan(page_number=2, char_start=100, char_end=100 + len(TEXT))],
            total_chars=100 + len(TEXT),
        ),
    )


def build_relative_position(query_size: int, key_size: int) -> int:
    """Stands in for transformers' ``@torch.jit.script``-ed mdeberta ``build_relative_position``."""
    return query_size - key_size


class _ScriptFunction:
    """What ``torch.jit.script`` returns with the JIT live — and what dies on Blackwell/CUDA-13.

    The real ScriptFunction asks nvrtc to compile its fused kernels at CALL time; on the CUDA-13 pod
    that fails in native code. Raising here (not at script time) mirrors the field evidence exactly:
    the fault surfaces on the FIRST forward, which is why no load-time try/except can catch it.
    """

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        msg = (
            "The following operation failed in the TorchScript interpreter: Compilation error: "
            "nvrtc: failed to open libnvrtc-builtins.so.13.0"
        )
        raise RuntimeError(msg)


class _Enabled:
    """Mirrors ``torch.jit._state._enabled`` — the proxy torch parses ``PYTORCH_JIT`` into."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled


@pytest.fixture()
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A ``torch`` whose JIT starts LIVE (``PYTORCH_JIT=1``), mirroring the real switch.

    ``torch.jit.script`` is a pass-through only while the JIT is disabled; ``torch.jit._state``
    exposes ``_enabled`` + ``disable()`` — the two levers the seam drives.
    """
    monkeypatch.setenv("PYTORCH_JIT", "1")
    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    state = types.ModuleType("torch.jit._state")
    enabled = _Enabled(enabled=True)

    def disable() -> None:
        enabled.enabled = False

    def script(fn: Any) -> Any:
        return _ScriptFunction(fn) if enabled.enabled else fn

    state._enabled = enabled  # type: ignore[attr-defined]
    state.disable = disable  # type: ignore[attr-defined]
    jit._state = state  # type: ignore[attr-defined]
    jit.script = script  # type: ignore[attr-defined]
    torch.jit = jit  # type: ignore[attr-defined]
    # Real torch always exposes ``inference_mode`` (the no-grad context the provider now runs the
    # forward inside, dogfood-2b-process); a faithful stand-in must too, or the forward would error.
    torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.jit", jit)
    return torch


class _MdebertaGLiNER2:
    """A ``gliner2.GLiNER2`` double whose encoder behaves like mdeberta-v3 (W9-jit).

    ``from_pretrained`` scripts the relative-position helper and every extraction call runs it, so
    the nvrtc failure mode is reproduced end-to-end through the real Provider.

    **Scope, stated honestly (audit R1 / #190-4).** This scripts at LOAD time, which is *later* than
    real transformers: transformers evaluates the ``@torch.jit.script`` decorators on the DeBERTa-v2
    modeling module at **module import**, i.e. when the Provider runs ``from gliner2 import
    GLiNER2`` — one statement BEFORE it enters the perf seam. So this suite proves the seam
    neutralizes the JIT for the *load-and-forward* path; it CANNOT distinguish "neutralized before
    the gliner2 import" from "after", and must never be read as covering that ordering. The
    import-time shape — the one that decides whether the Blackwell/CUDA-13 crash is really fixed —
    is owned by ``latence-core/tests/test_perf_jit_import_ordering.py`` (behavioural) and
    ``latence-core/tests/test_perf_fp32_regression_guard.py::
    test_mdeberta_providers_neutralize_the_jit_before_importing_the_backbone`` (structural).
    """

    #: when False the encoder never scripts (the reference "JIT could not have run" baseline).
    scripts_relative_position: bool = True

    def __init__(self, rel_pos: Any) -> None:
        self._rel_pos = rel_pos

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _MdebertaGLiNER2:  # noqa: ARG003
        import torch

        if cls.scripts_relative_position:
            return cls(torch.jit.script(build_relative_position))
        return cls(build_relative_position)

    def _forward(self, text: str) -> None:
        """The encoder forward — it consults the relative-position helper, like mdeberta."""
        self._rel_pos(len(text), 0)

    def _entities_for(
        self, text: str, entity_types: list[str], threshold: float
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        self._forward(text)
        by_label: dict[str, list[dict[str, Any]]] = {}
        for e in _ENTITIES:
            if e["label"] not in entity_types or e["confidence"] < threshold:
                continue
            by_label.setdefault(str(e["label"]), []).append(
                {
                    "text": text[e["start"] : e["end"]],
                    "confidence": e["confidence"],
                    "start": e["start"],
                    "end": e["end"],
                }
            )
        return {"entities": by_label}

    def _relations_for(
        self, text: str, relation_types: list[str]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        self._forward(text)
        by_rel: dict[str, list[dict[str, Any]]] = {}
        for r in _RELATIONS:
            if r["relation"] not in relation_types:
                continue
            by_rel.setdefault(str(r["relation"]), []).append(
                {"head": dict(r["head"]), "tail": dict(r["tail"])}
            )
        return {"relation_extraction": by_rel}

    # The BATCHED entry points the Provider drives since T7 — same real gliner2 signature (one
    # schema for the whole call, one per-text result dict in input order). The scripted
    # relative-position helper runs per text, so the nvrtc failure mode still reproduces end to end.
    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: list[str],
        batch_size: int = 8,  # noqa: ARG002 - API parity
        threshold: float = 0.5,
        format_results: bool = True,  # noqa: ARG002 - API parity
        include_confidence: bool = False,  # noqa: ARG002 - API parity
        include_spans: bool = False,  # noqa: ARG002 - API parity; the fake always spans
        max_len: int | None = None,  # noqa: ARG002 - API parity
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        return [self._entities_for(text, entity_types, threshold) for text in texts]

    def batch_extract_relations(
        self,
        texts: list[str],
        relation_types: list[str],
        batch_size: int = 8,  # noqa: ARG002 - API parity
        threshold: float = 0.5,  # noqa: ARG002 - API parity
        format_results: bool = True,  # noqa: ARG002 - API parity
        include_confidence: bool = False,  # noqa: ARG002 - API parity
        include_spans: bool = False,  # noqa: ARG002 - API parity
        max_len: int | None = None,  # noqa: ARG002 - API parity
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        return [self._relations_for(text, relation_types) for text in texts]

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return self.batch_extract_entities(
            [text],
            entity_types,
            1,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len,
        )[0]

    def extract_relations(
        self,
        text: str,
        relation_types: list[str],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return self.batch_extract_relations(
            [text],
            relation_types,
            1,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len,
        )[0]


@pytest.fixture()
def fake_gliner2(monkeypatch: pytest.MonkeyPatch) -> type[_MdebertaGLiNER2]:
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _MdebertaGLiNER2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner2", module)
    _MdebertaGLiNER2.scripts_relative_position = True
    return _MdebertaGLiNER2


def _extractor() -> Gliner2FusedExtractor:
    return Gliner2FusedExtractor(
        {"labels": ["organization", "person"], "relation_labels": ["employs"]}
    )


def test_scripted_encoder_reproduces_the_blackwell_nvrtc_crash(fake_torch: Any) -> None:
    """The fault under test: with the JIT live, the mdeberta forward dies in nvrtc.

    Guards the fake against false green — if this ever stops raising, the suite below proves
    nothing.
    """
    model = _MdebertaGLiNER2(fake_torch.jit.script(build_relative_position))
    with pytest.raises(RuntimeError, match="libnvrtc-builtins.so.13.0"):
        model.extract_entities(TEXT, ["organization"])


def test_extraction_survives_the_jit_on_a_blackwell_style_host(
    fake_torch: Any,  # noqa: ARG001 - installs the fake torch whose JIT starts live
    fake_gliner2: type[_MdebertaGLiNER2],  # noqa: ARG001 - installs the fake gliner2
) -> None:
    """REGRESSION (#190): the Provider extracts without the nvrtc crash and without env tweaking.

    Pre-fix (no ``neutralize_torch_jit`` at the shared seam) this raises
    ``ProviderError: ... provider failure in 'extract_entities' (native RuntimeError)`` wrapping
    ``failed to open libnvrtc-builtins.so.13.0``.
    """
    out = _extractor().extract_fused([_chunk()])
    assert [(m.label, m.text) for m in out.mentions] == [
        ("organization", "Acme Corp"),
        ("person", "Ada Lovelace"),
    ]
    assert [(r.label, r.head_text, r.tail_text) for r in out.relations] == [
        ("employs", "Acme Corp", "Ada Lovelace")
    ]


def test_the_jit_is_actually_neutralized_not_merely_configured(
    fake_torch: Any,
    fake_gliner2: type[_MdebertaGLiNER2],  # noqa: ARG001 - installs the fake gliner2
) -> None:
    """After a load, scripting is a genuine pass-through — the behaviour, not a recorded flag.

    Asserts the observable consequence: ``torch.jit.script`` hands back the eager function, which
    computes the same value, so nothing can reach nvrtc on the next forward either.
    """
    _extractor().extract_fused([_chunk()])
    scripted = fake_torch.jit.script(build_relative_position)
    assert scripted is build_relative_position
    assert scripted(7, 2) == build_relative_position(7, 2) == 5


def test_output_is_byte_identical_to_the_jit_free_path(
    fake_torch: Any,  # noqa: ARG001 - installs the fake torch whose JIT starts live
    fake_gliner2: type[_MdebertaGLiNER2],
) -> None:
    """ADR-0036 discipline: the neutralization degrades THROUGHPUT only, never the output.

    The same corpus is extracted twice — once through an encoder that would have scripted (the
    guarded path) and once through one that never scripts at all (the reference) — and every
    emitted mention/relation record must be identical field-for-field.
    """
    guarded = _extractor().extract_fused([_chunk()])
    fake_gliner2.scripts_relative_position = False
    reference = _extractor().extract_fused([_chunk()])
    assert [m.model_dump() for m in guarded.mentions] == [
        m.model_dump() for m in reference.mentions
    ]
    assert [r.model_dump() for r in guarded.relations] == [
        r.model_dump() for r in reference.relations
    ]


def test_a_genuine_native_fault_still_surfaces_typed(
    fake_torch: Any,
    fake_gliner2: type[_MdebertaGLiNER2],  # noqa: ARG001 - installs the fake gliner2
) -> None:
    """The guard is narrow: neutralizing the JIT does not swallow a real inference error.

    A model whose forward raises for its own reasons still maps to the typed ProviderError carrying
    no input text (C2/C6) — the JIT seam adds no new swallow-everything path.
    """
    extractor = _extractor()

    def boom(*_args: Any, **_kwargs: Any) -> None:
        msg = "cuda oom while extracting"
        raise RuntimeError(msg)

    monkey = _MdebertaGLiNER2(build_relative_position)
    monkey.batch_extract_entities = boom  # type: ignore[method-assign, assignment]
    extractor._model = monkey  # noqa: SLF001 - inject the loaded model directly
    with pytest.raises(ProviderError, match="batch_extract_entities"):
        extractor.extract_fused([_chunk()])
    assert fake_torch is not None
