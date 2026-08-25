"""A faithful, offline stand-in for ``gliner2`` **2.0.0**, plus the chunk fixtures.

Deliberately NOT named ``conftest.py``. Under pytest's default ``prepend`` import mode every
package's ``tests/conftest.py`` is imported as the same bare top-level module ``conftest``, so the
first one collected wins for the whole session — which is why exactly one package in this workspace
(``latence-disambig-glinker``) gets away with ``from conftest import ...`` and a second one cannot.
A unique module name is what lets these tests run in the same invocation as everybody else's. (It
is the same collision ``scripts/typecheck.sh`` exists to work around, ADR-0032.)

The real ``gliner2`` 2.x stack (torch/transformers, an mDeBERTa-v3 boundary encoder) is
deliberately NOT installed in the dev/test env (ADR-0007/0016), so these tests inject a fake
``gliner2`` module. *Faithful* here has a precise meaning, and it is the whole point of the file:
**the fake must reject what the real library rejects.** A stub that always agrees turns a passing
test into a decoration.

Every behaviour below was read out of the unpacked ``gliner2`` 2.0.0 wheel on 2026-08-25, with the
source file named next to it:

* ``gliner2/__init__.py`` sets ``__version__ = "2.0.0"`` eagerly and keeps ``AutoExtractor`` /
  ``GLiNER2`` / ``BoundaryExtractor`` / the architecture errors behind a lazy ``__getattr__``. The
  fake exposes them as plain attributes — the laziness is a torch-import optimisation, not a
  contract, and the Provider only ever does ``getattr(gliner2, "AutoExtractor")``.
* ``gliner2/auto.py::AutoExtractor.from_pretrained`` resolves the checkpoint's SAVED architecture,
  compares it with any requested one, and raises ``ArchitectureMismatchError`` unless
  ``allow_architecture_override``. ``_resolve_class`` raises ``UnknownArchitectureError`` for an
  unregistered name.
* ``gliner2/models/loading.py::split_load_kwargs`` raises ``TypeError`` for any load kwarg outside
  ``HUB_LOAD_OPTIONS | MODEL_LOAD_OPTIONS`` — ``quantize`` and ``compile`` ARE in the latter, which
  is what lets the shared perf seam thread them through unchanged.
* ``gliner2/inference/runtime.py::ExtractorRuntimeMixin`` holds the batched public API for BOTH
  architectures. ``batch_extract`` raises ``ValueError`` on ``batch_size <= 0`` and returns ``[]``
  for no texts. ``batch_size`` is the THIRD POSITIONAL parameter of ``batch_extract_entities`` /
  ``batch_extract_relations`` — a Provider that passes ``threshold`` positionally silently sets the
  model's internal sub-batch size instead, so the fake keeps that ordering exactly.
* ``_format_spans`` / ``_format_entity_dict`` return dict rows ONLY when ``include_spans`` and/or
  ``include_confidence`` are set — with neither, entities come back as bare **strings** and a
  relation as a ``(head, tail)`` **tuple**. That is the trap ``include_spans=True`` exists to
  avoid, and the fake reproduces it so a Provider that forgot the flag recovers nothing groundable.
* ``format_results`` fills ``relations[rel] = []`` for every REQUESTED relation type that produced
  nothing, so the ``relation_extraction`` envelope is present with empty lists rather than absent.
* ``BoundaryExtractor._decode_relations`` returns ``{}`` immediately when ``enable_relations`` is
  false — the silent structural emptiness the Provider warns about.
"""

from __future__ import annotations

import contextlib
import sys
import types
from typing import TYPE_CHECKING, Any

from latence_core.contracts import (
    ChunkRecord,
    Classification,
    InducedLabels,
    OffsetMap,
    OffsetSegment,
    PageMap,
    PageSlice,
    PageSpan,
    Provenance,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

#: The load options ``gliner2.models.loading.split_load_kwargs`` accepts, verbatim from the 2.0.0
#: wheel. Anything else is a ``TypeError`` — which is exactly how a Provider that invents a load
#: kwarg is caught offline instead of on a pod.
HUB_LOAD_OPTIONS = frozenset(
    {"cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder", "proxies"}
)
MODEL_LOAD_OPTIONS = frozenset(
    {"quantize", "compile", "map_location", "use_flashdeberta", "word_splitter"}
)
LOAD_OPTIONS = HUB_LOAD_OPTIONS | MODEL_LOAD_OPTIONS

#: Which architecture the fake Hub "saves" for a given checkpoint id. Anything not listed resolves
#: to ``"span"``, mirroring ``gliner2/configuration.py`` — "a missing ``architecture`` field means
#: an old (span) checkpoint".
SAVED_ARCHITECTURES: dict[str, str] = {
    "fastino/gliner2.5-multi-v1": "boundary",
    "fastino/gliner2.5-base-v1": "boundary",
    "fastino/gliner2.5-small-v1": "boundary",
    "fastino/gliner2-multi-v1": "span",
    "fastino/GLiNER2-Guardrails-PII-Multi": "span",
}


class ArchitectureMismatchError(ValueError):
    """``gliner2.auto.ArchitectureMismatchError`` — span and boundary heads are incompatible."""


class UnknownArchitectureError(ValueError):
    """``gliner2.auto.UnknownArchitectureError`` — no model class registered for that name."""


class BackendCompilerFailed(RuntimeError):
    """A stand-in for ``torch._dynamo.exc.BackendCompilerFailed`` — matched by class NAME.

    ``torch.compile`` raises this LAZILY on the first forward when its backend cannot compile the
    model. ``is_compile_backend_failure`` is torch-free and matches on the class name / module
    chain, so a same-named local class exercises the exact guard without importing torch.
    """


class OutOfMemoryError(RuntimeError):
    """A stand-in for ``torch.cuda.OutOfMemoryError`` — matched by class NAME (torch-free).

    The message shape matters as much as the name: ``is_cuda_oom`` also matches the
    ``"CUDA out of memory"`` fragment.
    """


class _FakeExtractor:
    """The shared runtime surface — ``gliner2.inference.runtime.ExtractorRuntimeMixin``.

    Seeded with FLAT rows (``entities`` / ``relations``) which the batched calls reshape into the
    real nested envelopes, filtering by label and threshold on the way exactly as the model does.
    ``per_text`` lets a test give distinct texts distinct results, which is what catches an
    off-by-one in the scatter-back.
    """

    architecture = "span"
    enable_relations = True

    #: flat ``{start, end, label, confidence}`` rows.
    entities: list[dict[str, Any]] = []
    #: flat ``{relation, head: {...}, tail: {...}}`` rows.
    relations: list[dict[str, Any]] = []
    #: optional ``text -> flat entity rows`` override, for the batch-equivalence tests.
    per_text: dict[str, list[dict[str, Any]]] | None = None

    #: what the last ``from_pretrained`` received (the perf-knob assertions read this).
    last_kwargs: dict[str, Any] = {}
    #: EVERY ``from_pretrained`` kwarg set, in order — a reload is only visible as a second entry.
    load_kwargs_history: list[dict[str, Any]] = []
    #: the checkpoint id the last load asked for.
    last_model_id: str = ""
    #: every ``.to(...)`` target, in order (the placement assertions read this).
    moves: list[str] = []
    #: the ``max_len`` threaded into the last batched call.
    last_max_len: int | None = None
    #: the text-count of every batched call, in order — the micro-batch assertions read this.
    call_sizes: list[int] = []
    #: force the OPTIMIZED load (quantize/compile) to raise, to exercise the guarded plain retry.
    raise_on_optimized_load: bool = False
    #: a model loaded with ``compile=True`` raises BackendCompilerFailed on its FIRST forward.
    compiled_forward_raises: bool = False
    #: a batched forward over MORE than this many texts raises a CUDA OOM (``None`` = never).
    oom_above: int | None = None
    #: raise a plain native fault from every forward (the typed-error mapping test).
    raise_on_extract: bool = False
    #: MEASURED 2026-08-25: the GLiNER 2.5 checkpoints store ``extra_special_tokens`` as a LIST in
    #: their ``tokenizer_config.json``, and transformers >= 4.57 dies on it inside
    #: ``_set_model_specific_special_tokens`` with ``AttributeError: 'list' object has no attribute
    #: 'keys'``. The fake reproduces that refusal so the shim's ABSENCE is a red test rather than a
    #: silent pass: ``from_pretrained`` raises unless :attr:`shim_applied` was set first.
    tokenizer_config_is_list: bool = True
    #: MEASURED 2026-08-25 (pod, RTX 2000 Ada): the model does NOT read a long input to the end.
    #: When set, entity rows whose ``end`` is past this char offset are DROPPED and the result is
    #: returned well-formed, with no error and no warning — the fail-open the Providers' window
    #: guard exists to prevent. ``None`` = read everything (the ordinary tests).
    truncates_beyond_chars: int | None = None
    #: Set by the test's spy around ``coerce_list_extra_special_tokens``. The real shim rewrites a
    #: file in the HF cache, which does not exist offline — so what the fake can honestly check is
    #: that the Provider CALLED it before the load, in the right order.
    shim_applied: bool = False

    def __init__(self, *, compiled: bool = False) -> None:
        self.device = "cpu"
        self._compiled = compiled
        self._forwards = 0

    # -- loading ---------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _FakeExtractor:
        unknown = sorted(set(kwargs) - LOAD_OPTIONS)
        if unknown:
            # gliner2/models/loading.py::split_load_kwargs
            msg = f"from_pretrained does not accept {unknown}; accepted options are {LOAD_OPTIONS}."
            raise TypeError(msg)
        if cls.tokenizer_config_is_list and not cls.shim_applied:
            # The verbatim native failure, from the real traceback:
            #   transformers/tokenization_utils_base.py:1210 _set_model_specific_special_tokens
            msg = "'list' object has no attribute 'keys'"
            raise AttributeError(msg)
        if cls.raise_on_optimized_load and (kwargs.get("quantize") or kwargs.get("compile")):
            msg = "simulated optimized-load failure"
            raise RuntimeError(msg)
        _FakeExtractor.last_kwargs = dict(kwargs)
        _FakeExtractor.load_kwargs_history.append(dict(kwargs))
        _FakeExtractor.last_model_id = model_id
        return cls(compiled=bool(kwargs.get("compile")))

    def to(self, target: str) -> _FakeExtractor:
        _FakeExtractor.moves.append(target)
        self.device = target
        return self

    # -- the batched public API -----------------------------------------

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: list[str],
        batch_size: int = 8,
        threshold: float = 0.5,
        format_results: bool = True,  # noqa: ARG002, FBT001, FBT002 - API parity with 2.0.0
        include_confidence: bool = False,  # noqa: FBT001, FBT002 - API parity with 2.0.0
        include_spans: bool = False,  # noqa: FBT001, FBT002 - API parity with 2.0.0
        max_len: int | None = None,
        overlap_policy: str | None = None,  # noqa: ARG002 - new in 2.0.0; the Provider passes none
    ) -> list[dict[str, Any]]:
        self._guard_batch(texts, batch_size, max_len)
        return [
            {
                "entities": _nest_entities(
                    self._rows_for(text), text, entity_types, threshold,
                    include_spans=include_spans, include_confidence=include_confidence,
                )
            }
            for text in texts
        ]

    def batch_extract_relations(
        self,
        texts: list[str],
        relation_types: list[str],
        batch_size: int = 8,
        threshold: float = 0.5,  # noqa: ARG002 - the seeded rows are already above threshold
        format_results: bool = True,  # noqa: ARG002, FBT001, FBT002 - API parity with 2.0.0
        include_confidence: bool = False,  # noqa: FBT001, FBT002 - API parity with 2.0.0
        include_spans: bool = False,  # noqa: FBT001, FBT002 - API parity with 2.0.0
        max_len: int | None = None,
        overlap_policy: str | None = None,  # noqa: ARG002 - new in 2.0.0; the Provider passes none
    ) -> list[dict[str, Any]]:
        self._guard_batch(texts, batch_size, max_len)
        return [
            {
                "relation_extraction": _nest_relations(
                    type(self).relations if self.enable_relations else [],
                    text,
                    relation_types,
                    include_spans=include_spans,
                    include_confidence=include_confidence,
                )
            }
            for text in texts
        ]

    # -- internals -------------------------------------------------------

    def _rows_for(self, text: str) -> list[dict[str, Any]]:
        per_text = type(self).per_text
        if per_text is not None:
            return per_text.get(text, [])
        return type(self).entities

    def _guard_batch(self, texts: list[str], batch_size: int, max_len: int | None) -> None:
        """The refusals ``ExtractorRuntimeMixin.batch_extract`` itself performs, plus the fakes'."""
        if batch_size <= 0:
            msg = "batch_size must be greater than 0"
            raise ValueError(msg)
        _FakeExtractor.last_max_len = max_len
        _FakeExtractor.call_sizes.append(len(texts))
        if type(self).raise_on_extract:
            msg = "simulated native gliner2 fault"
            raise RuntimeError(msg)
        self._forwards += 1
        if self._compiled and type(self).compiled_forward_raises and self._forwards == 1:
            msg = "backend compiler failed"
            raise BackendCompilerFailed(msg)
        oom_above = type(self).oom_above
        if oom_above is not None and len(texts) > oom_above:
            msg = "CUDA out of memory. Tried to allocate 17.00 GiB"
            raise OutOfMemoryError(msg)


class _FakeSpanExtractor(_FakeExtractor):
    """``gliner2.inference.engine.SpanExtractor`` — the legacy (gliner2 1.x checkpoint) head."""

    architecture = "span"


class _FakeBoundaryExtractor(_FakeExtractor):
    """``gliner2.models.boundary.engine.BoundaryExtractor`` — what a GLiNER 2.5 checkpoint is."""

    architecture = "boundary"


class _FakeAutoExtractor:
    """``gliner2.auto.AutoExtractor`` — the architecture-dispatching loader."""

    _registry: dict[str, type[_FakeExtractor]] = {
        "span": _FakeSpanExtractor,
        "boundary": _FakeBoundaryExtractor,
    }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        architecture: str | None = None,
        allow_architecture_override: bool = False,
        **kwargs: Any,
    ) -> _FakeExtractor:
        saved = SAVED_ARCHITECTURES.get(pretrained_model_name_or_path, "span")
        requested = architecture if architecture is not None else saved
        if requested not in cls._registry:
            msg = (
                f"No model class registered for architecture {requested!r}. "
                f"Registered: {sorted(cls._registry)}"
            )
            raise UnknownArchitectureError(msg)
        if requested != saved and not allow_architecture_override:
            msg = (
                f"Checkpoint architecture is {saved!r}, but {requested!r} was requested. "
                "Span and boundary heads are not checkpoint-compatible."
            )
            raise ArchitectureMismatchError(msg)
        return cls._registry[requested].from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )


def _nest_entities(
    rows: list[dict[str, Any]],
    text: str,
    entity_types: list[str],
    threshold: float,
    *,
    include_spans: bool,
    include_confidence: bool,
) -> dict[str, Any]:
    """``{label: [row]}`` in the exact shape ``_format_entity_dict`` produces for each flag pair."""
    by_label: dict[str, list[Any]] = {}
    limit = _FakeExtractor.truncates_beyond_chars
    for row in rows:
        label = str(row["label"])
        if label not in entity_types or row["confidence"] < threshold or row["end"] > len(text):
            continue
        if limit is not None and row["end"] > limit:
            # SILENT partial coverage, exactly as measured: no exception, no warning, and the
            # envelope below is indistinguishable from a complete result.
            continue
        surface = text[row["start"] : row["end"]]
        by_label.setdefault(label, []).append(
            _format_span(surface, row, include_spans=include_spans, confidence=include_confidence)
        )
    return by_label


def _format_span(
    surface: str, row: dict[str, Any], *, include_spans: bool, confidence: bool
) -> Any:
    """``ExtractorRuntimeMixin._format_spans`` — a bare STRING when neither flag is set."""
    if include_spans and confidence:
        return {
            "text": surface,
            "confidence": row["confidence"],
            "start": row["start"],
            "end": row["end"],
        }
    if include_spans:
        return {"text": surface, "start": row["start"], "end": row["end"]}
    if confidence:
        return {"text": surface, "confidence": row["confidence"]}
    return surface


def _nest_relations(
    rows: list[dict[str, Any]],
    text: str,
    relation_types: list[str],
    *,
    include_spans: bool,
    include_confidence: bool,
) -> dict[str, Any]:
    """``{rel_type: [row]}`` with an EMPTY list for every requested type that produced nothing.

    That last part is ``format_results``: ``for rel in requested_relations: relations.setdefault(
    rel, [])``. It is why an inactive relation head yields a well-formed, entirely empty result.
    """
    by_rel: dict[str, list[Any]] = {rel: [] for rel in relation_types}
    for row in rows:
        rel_type = str(row["relation"])
        if rel_type not in relation_types or row["head"]["end"] > len(text):
            continue
        by_rel[rel_type].append(
            _format_pair(
                row, include_spans=include_spans, include_confidence=include_confidence
            )
        )
    return by_rel


def _format_pair(row: dict[str, Any], *, include_spans: bool, include_confidence: bool) -> Any:
    """``_extract_relations`` / ``_decode_relations`` — a ``(head, tail)`` TUPLE with no flags.

    The full form's key SET is ``{text, start, end, confidence}`` — pinned in
    :mod:`latence_core.conformance.stub` from the cross-major pod A/B and asserted by
    ``test_this_packages_fake_matches_the_pod_measured_record_shape``. The ORDER used here follows
    the boundary decoder's source (it builds the span dict first and attaches the pair score after)
    and is deliberately NOT claimed as measured: the A/B sorted the keys, so it establishes the set
    and nothing about order — and no consumer in this framework reads these dicts positionally.
    """
    head, tail = row["head"], row["tail"]
    if include_spans and include_confidence:
        return {
            "head": {k: head[k] for k in ("text", "start", "end", "confidence")},
            "tail": {k: tail[k] for k in ("text", "start", "end", "confidence")},
        }
    if include_spans:
        return {
            "head": {k: head[k] for k in ("text", "start", "end")},
            "tail": {k: tail[k] for k in ("text", "start", "end")},
        }
    if include_confidence:
        return {
            "head": {k: head[k] for k in ("text", "confidence")},
            "tail": {k: tail[k] for k in ("text", "confidence")},
        }
    return (head["text"], tail["text"])


@contextlib.contextmanager
def install_gliner2(
    *,
    version: str = "2.0.0",
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
    per_text: dict[str, list[dict[str, Any]]] | None = None,
    with_auto: bool = True,
    enable_relations: bool = True,
    tokenizer_config_is_list: bool = True,
) -> Iterator[type[_FakeExtractor]]:
    """Install the fake ``gliner2`` module and reset every recorded class attribute.

    ``with_auto=False`` drops ``AutoExtractor`` from the module, which is what a pre-2.0 install
    looks like to the compat gate. ``version`` drives the version gate. Yields the shared
    ``_FakeExtractor`` class so a test can read ``last_kwargs`` / ``moves`` / ``call_sizes``.
    """
    _FakeExtractor.entities = entities or []
    _FakeExtractor.relations = relations or []
    _FakeExtractor.per_text = per_text
    _FakeExtractor.last_kwargs = {}
    _FakeExtractor.load_kwargs_history = []
    _FakeExtractor.last_model_id = ""
    _FakeExtractor.moves = []
    _FakeExtractor.last_max_len = None
    _FakeExtractor.call_sizes = []
    _FakeExtractor.raise_on_optimized_load = False
    _FakeExtractor.compiled_forward_raises = False
    _FakeExtractor.oom_above = None
    _FakeExtractor.raise_on_extract = False
    _FakeExtractor.truncates_beyond_chars = None
    _FakeExtractor.tokenizer_config_is_list = tokenizer_config_is_list
    # Every test EXCEPT the one that pins the shim runs as if the shim had done its job — otherwise
    # a single missing call would fail forty unrelated assertions and say nothing useful. The
    # shim test flips this back off and installs a spy.
    _FakeExtractor.shim_applied = True
    _FakeBoundaryExtractor.enable_relations = enable_relations

    module = types.ModuleType("gliner2")
    module.__version__ = version  # type: ignore[attr-defined]
    module.GLiNER2 = _FakeSpanExtractor  # type: ignore[attr-defined]
    module.SpanExtractor = _FakeSpanExtractor  # type: ignore[attr-defined]
    module.BoundaryExtractor = _FakeBoundaryExtractor  # type: ignore[attr-defined]
    module.ArchitectureMismatchError = ArchitectureMismatchError  # type: ignore[attr-defined]
    module.UnknownArchitectureError = UnknownArchitectureError  # type: ignore[attr-defined]
    if with_auto:
        module.AutoExtractor = _FakeAutoExtractor  # type: ignore[attr-defined]
    saved = sys.modules.get("gliner2")
    sys.modules["gliner2"] = module
    try:
        yield _FakeExtractor
    finally:
        _FakeBoundaryExtractor.enable_relations = True
        _FakeExtractor.shim_applied = False
        _FakeExtractor.tokenizer_config_is_list = True
        if saved is not None:
            sys.modules["gliner2"] = saved
        else:
            sys.modules.pop("gliner2", None)


# ---------------------------------------------------------------------------
# Record fixtures
# ---------------------------------------------------------------------------


def chunk(
    text: str,
    *,
    sensitivity: str = "public",
    page_map: PageMap | None = None,
    record_id: str = "doc1#chunk-0",
    document_record_id: str = "doc1",
    index: int = 0,
    induced_labels: InducedLabels | None = None,
    char_start: int = 0,
) -> ChunkRecord:
    """A ChunkRecord carrying its own ``page_slice`` (v19) and a chunk-anchored ``offset_map``."""
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id="sha256:doc1",
        char_start=char_start,
        char_end=char_start + len(text),
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
        token_count=len(text.split()),
        page_slice=(
            PageSlice.for_span(page_map, char_start, char_start + len(text))
            if page_map is not None
            else PageSlice(
                pages=[PageSpan(page_number=1, char_start=0, char_end=char_start + len(text))],
                total_chars=char_start + len(text),
            )
        ),
        induced_labels=induced_labels,
        offset_map=OffsetMap(
            segments=[OffsetSegment(local_start=0, doc_start=char_start)], length=len(text)
        )
        if text
        else None,
    )
