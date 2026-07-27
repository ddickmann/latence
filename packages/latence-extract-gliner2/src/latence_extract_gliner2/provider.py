"""Fused Entity + Relation Extraction Provider over the fastino **GLiNER2** framework (W9).

The unified extraction engine (ADR-0036, W9): ``fastino/gliner2-multi-v1`` does joint zero-shot
NER **and** Relation Extraction over ONE mdeberta-v3 backbone, so a single Provider fulfils BOTH
the Entity Extraction (S4) and Relation Extraction (S5) Capabilities (ADR-0013). It is a **drop-in
swap** for :class:`latence_relation_gliner.provider.GlinerRelexExtractor` — the SAME
:class:`~latence_core.capability.FusedEntityRelationExtractor` seam, the SAME config shape
(``labels`` + ``relation_labels``), the SAME mentions+relations+evidence+provenance+offset contract
— so a Pipeline swaps ``provider: relation.gliner_relex`` for ``provider:
fused_entity_relation.gliner2`` with no other change. Consolidating extraction onto GLiNER2 retires
the separate ``relation.gliner_relex`` architecture (kept-but-superseded in the roster): one
backbone for NER + RE + PII, native ``quantize``/``compile``, more throughput, less handling.

Pod-validated gliner2 1.3.2 API (REAL — mirrored exactly in the offline fake). The Provider drives
the **batched** entry points; the single-text ``extract_entities``/``extract_relations`` are, in
gliner2 itself, thin wrappers that call the batched ones with a one-element list:

.. code-block:: python

    from gliner2 import GLiNER2
    model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1", quantize=True, compile=True)
    ents = model.batch_extract_entities(texts, entity_types, threshold=0.5,
                                        include_spans=True, include_confidence=True, max_len=768)
    # -> [{"entities": {label: [{"text", "confidence", "start", "end"}, ...], ...}}, ...]
    rels = model.batch_extract_relations(texts, relation_types, threshold=0.5,
                                         include_spans=True, include_confidence=True, max_len=768)
    # -> [{"relation_extraction": {rel_type: [{"head": {"text","confidence","start","end"},
    #                                          "tail": {...}}, ...], ...}}, ...]

**T7 — efficient per-chunk extraction.** Per-chunk induction (T1/T2) gives every chunk its own
small label set, but a batched forward takes ONE schema per call. The Provider therefore BUCKETS
each streaming batch by label set (:func:`~latence_core.providers.perf.iter_label_buckets`) and
runs ONE native batched forward per bucket at that bucket's own small label count — never the
corpus union, never one call per chunk. gliner2's own batched span-rep path documents itself as
"bit-identical to calling compute_span_rep per sample", and the Provider re-orders nothing, so the
emitted records are bucket- and batch-INVARIANT (asserted against a per-chunk reference).

**No server, and we do not pretend otherwise.** gliner2 ships no self-hostable inference server —
its only server surface is ``GLiNER2.from_api``, a CLIENT for fastino's *cloud* API (data egress +
an external key), which is disqualifying for an on-prem data framework. All the throughput here is
gliner2's NATIVE BATCHED FORWARD + ``compile`` + ``quantize``, running IN-PROCESS on the operator's
own hardware.

The entity ``start``/``end`` are chunk-local **char** offsets (half-open); the relation ``head``/
``tail`` sub-dicts carry the SAME chunk-local ``start``/``end`` char offsets, so a relation is
resolved to the two :class:`~latence_core.contracts.EntityMention`s it links by matching those
``(start, end)`` spans (exactly the relex resolution). ``include_spans=True`` is REQUIRED — without
it gliner2 returns entity *strings* / relation head/tail *text* only, which the framework contract
cannot ground; the Provider always passes it.

Offsets (the recurring coordinate trap, ADR-0031): the chunk-local model offsets are resolved to
TRUE original-markdown offsets through the chunk's stripped→original ``offset_map`` via
:class:`~latence_core.offsetmap.OffsetIndex` (``char_start + local`` is only a lower bound once
markdown markup was stripped; a pre-v11 chunk with no map degrades to that shift). Each page span is
re-resolved through :class:`~latence_core.page_index.PageIndexResolver`, which reads the chunk's
OWN ``page_slice`` (v19) — so every chunk resolves exactly, in any order, whichever siblings a
Stage dropped. ONE resolver serves the whole ``extract_fused`` call so the drift roll-up is per
call. A relation's covering span spans two chunks, so it is answered from the endpoints' own
already-exact pages (:func:`~latence_core.page_index.pages_for_covering_span`), not by a lookup.

Licensing (ADR-0012 / research-diligence, verified 2026-07-10 — weights AND code, separately):
    * ``gliner2`` library (PyPI + github.com/fastino-ai/GLiNER2): **Apache-2.0** (code) — LICENSE
      file + README badge + PyPI ``OSI Approved :: Apache Software License`` classifier.
    * Default checkpoint ``fastino/gliner2-multi-v1``: **Apache-2.0** (weights, per the HF model
      card "License: apache-2.0"; NOT gated / freely downloadable).
    Both weights AND code are permissive; nothing downloads unless a Pipeline actually configures
    this Provider.

The ``gliner2`` import is deferred to first use (ADR-0016) so importing this module (for Provider
discovery) never requires torch; a missing optional dep raises a clear error only when the Provider
is run. W9 perf seam: the native ``quantize`` (fp16) load knob defaults True on cuda / False on cpu,
guarded so a fault degrades to a plain load (never a crash). W9-scaling hardening (Blackwell):
torch intra-op threads are capped at load
(default 8) to stop a many-core host oversubscribing to a CPU spin with the GPU idle; the fp16 model
is explicitly moved to cuda and its device logged (the fp16-on-CPU defence); and extraction streams
in bounded batches with per-batch progress logging so a large corpus never *appears* hung.
W9-jit hardening (#190): the shared load seam also neutralizes the **TorchScript JIT** (the guard
covered ``torch.compile`` and NOT ``torch.jit.script``) — the mdeberta encoder scripts its
relative-position helper, which dies in nvrtc (``libnvrtc-builtins.so.13.0``) on a Blackwell /
CUDA-13 box on the FIRST forward. Throughput-only; the emitted records are byte-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from latence_core.capability import FusedExtraction
from latence_core.contracts import EntityMention, RelationMention
from latence_core.errors import ConfigError, ProviderError
from latence_core.induced import LabelFloor, LabelSlot, effective_labels, resolve_label_floor
from latence_core.offsetmap import OffsetIndex
from latence_core.page_index import PageIndexResolver, pages_for_covering_span
from latence_core.providers.adapter import validate_batch_size
from latence_core.providers.device import DeviceDecision, select_device
from latence_core.providers.perf import (
    empty_cuda_cache,
    inference_context,
    is_compile_backend_failure,
    is_cuda_oom,
    iter_label_buckets,
    iter_token_budget_microbatches,
    load_gliner2_model,
    resolve_gliner2_knob,
    resolve_intra_op_threads,
    resolve_max_len,
    resolve_microbatch_bounds,
)
from latence_core.providers.profile import ProviderProfile

_log = logging.getLogger("latence.provider.gliner2_fused")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, Provenance

# Apache-2.0 weights (verified 2026-07-10, HF model card, not gated). Multilingual fused NER+RE
# over an mdeberta-v3 encoder via the fastino GLiNER2 framework.
_DEFAULT_MODEL = "fastino/gliner2-multi-v1"

# The default streaming batch size: the BUCKETING WINDOW — how many chunks are buffered before
# they are partitioned by label set and each bucket run through one native batched forward (T7).
# It is NOT the model-forward tensor size: every bucket is further split into token-budget
# micro-batches, so the padded activation is bounded by (microbatch_max_texts × max_len) for ANY
# configured batch_size. A wider window packs buckets better (more chunks share a label set) at a
# coarser progress cadence, so 16 is the balance point; _MAX_BATCH_SIZE bounds the cadence so a
# large corpus can never run silently for minutes (the "appears hung" failure). The emitted
# mentions/relations are record-for-record identical for any positive batch size (ADR-0033).
_DEFAULT_BATCH_SIZE = 16

# The hard cap on the streaming batch (progress cadence). A configured batch_size above this is
# clamped so extraction ALWAYS logs incremental progress at least this often on a large corpus —
# the "a large single batch must never wedge / silently stall the stage" guarantee (W9-scaling).
_MAX_BATCH_SIZE = 64

# T2 (per-chunk extraction): entity + relation labels are RELEVANCE labels, so this Provider's
# config labels are a PER-CHUNK floor, not a universal one — a chunk that induced its own types is
# scored against THOSE ALONE, and the config floor is the recall net for a chunk induction said
# nothing about. This is the SOTA choice for a zero-shot span-scoring model and it is not close:
#
#   * PRECISION. GLiNER2 scores every candidate span against every label; recall/precision degrade
#     as the label count grows because irrelevant types compete for the same spans. Unioning a
#     corpus-wide floor onto a chunk that already induced its own 5 types buys nothing on that
#     chunk and taxes every span decision on it.
#   * MEMORY. The span-scoring activation is (batch × positions × LABELS); the label dimension is
#     what turned a routine forward into a multi-GiB allocation. Small per-chunk label sets are the
#     ROOT fix for that, not the reactive micro-batch backoff.
#   * The trade, stated honestly: a UNIVERSAL floor is recall insurance — a chunk whose induction
#     under-called (missed ``person``) would still be scored for it. PER_CHUNK forfeits that
#     insurance in exchange for label-count purity, and the redesign accepts it BECAUSE the front
#     half now induces per chunk (the induced set is derived from the chunk's own text) and the
#     type vocabulary is reconciled downstream, not imposed upstream. An operator who wants the
#     insurance back sets ``label_floor: universal`` — one knob, no code change.
#
# A chunk with NO induced labels falls back to the config floor under either policy, so every stack
# without a ``schema_induction`` Stage (and predefined-only mode, where induction is skipped
# entirely) is byte-identical: exactly its config labels, per chunk.
_DEFAULT_LABEL_FLOOR = LabelFloor.PER_CHUNK


@dataclass(frozen=True)
class _PendingChunk:
    """A chunk with its two resolved label sets — the bucket keys for the batched forward (T7).

    Resolving the effective (entity, relation) label sets is model-independent, so it happens for
    the whole streaming window BEFORE any forward; the sets are then the keys the window is
    partitioned by. Tuples, not lists: they must be hashable to bucket, and label ORDER is part of
    the question asked of the model, so two chunks share a bucket only when they would have been
    asked the identical question.
    """

    chunk: ChunkRecord
    entity_labels: tuple[str, ...]
    relation_labels: tuple[str, ...]


@dataclass(frozen=True)
class _Slot:
    """One of the two extraction slots — the ONE place the entity/relation asymmetry lives (T7).

    The bucketing, micro-batching, compile-latch, OOM-backoff and scatter-back logic is identical
    for entities and relations; only three things differ: which native batched method to call,
    which top-level key wraps each per-text result, and which of a chunk's two label sets is the
    bucket key. Naming that difference once — instead of duplicating the whole batched path per
    slot, or branching on a string inside it — is what keeps the batched forward a single deep
    method rather than two near-copies that can drift.
    """

    #: the native gliner2 batched entry point (``batch_extract_entities`` / ``…_relations``).
    method: str
    #: the top-level key wrapping each per-text result (``entities`` / ``relation_extraction``).
    envelope: str
    #: whether this slot reads a chunk's ENTITY label set (else its RELATION set).
    entity: bool

    def labels_of(self, item: _PendingChunk) -> tuple[str, ...]:
        """This slot's label set for ``item`` — the bucket key."""
        return item.entity_labels if self.entity else item.relation_labels


_ENTITY_SLOT = _Slot(method="batch_extract_entities", envelope="entities", entity=True)
_RELATION_SLOT = _Slot(
    method="batch_extract_relations", envelope="relation_extraction", entity=False
)


def _unwrap(slot: _Slot, raw: Any) -> dict[str, list[dict[str, Any]]]:
    """One per-text result → its per-label / per-relation-type dict of rows (faithful shape check).

    The REAL gliner2 shapes are nested dicts — ``{"entities": {label: [rows]}}`` and
    ``{"relation_extraction": {rel_type: [rows]}}`` — NOT the ``gliner`` library's flat list. A
    provider that mis-parses either as a flat list recovers ZERO spans silently, which is exactly
    the false green the offline fake and its anchor test exist to close. Anything that is not the
    pinned shape is a typed :class:`ProviderError` (ADR-0034), never a quiet empty result.
    """
    payload: Any = raw.get(slot.envelope) if isinstance(raw, dict) else raw
    if not isinstance(payload, dict):
        got = type(payload).__name__
        msg = (
            f"fused_entity_relation.gliner2: unexpected {slot.method} output shape "
            f"(expected {{{slot.envelope!r}: {{label: [rows]}}}}, got {got})"
        )
        raise ProviderError(msg)
    return cast("dict[str, list[dict[str, Any]]]", payload)


def _require_batched_api(model: Any) -> None:
    """Typed :class:`ProviderError` if the loaded model lacks the batched entry points (T7).

    The backstop for anything the package pin cannot see (a hand-installed older ``gliner2``, a
    fork, a vendored wrapper, or a ``GLiNER2.from_api`` CLOUD client rather than a local model):
    the missing method surfaces as a typed, actionable framework error at LOAD rather than the raw
    ``AttributeError`` an operator got in the field on the twin redactor (#192).
    """
    missing = [
        slot.method
        for slot in (_ENTITY_SLOT, _RELATION_SLOT)
        if not callable(getattr(model, slot.method, None))
    ]
    if not missing:
        return
    msg = (
        f"fused_entity_relation.gliner2 batches via {' + '.join(missing)}(texts, types, "
        "threshold=…, include_spans=True, include_confidence=True, max_len=…), but the loaded "
        f"GLiNER2 object exposes no callable {missing!r}. Those reached their current signature in "
        "gliner2 1.2.6 and this Provider supports gliner2 >=1.3,<2. Install a supported version "
        "(pip install 'gliner2[local]>=1.3,<2') or point 'model' at a checkpoint the GLiNER2 "
        "framework can load locally."
    )
    raise ProviderError(msg)


class Gliner2FusedExtractor:
    """Fused zero-shot Entity+Relation Extractor backed by a GLiNER2 checkpoint (W9).

    A drop-in for :class:`latence_relation_gliner.provider.GlinerRelexExtractor`: it satisfies the
    SAME :class:`~latence_core.capability.FusedEntityRelationExtractor` seam and emits the SAME
    contract (mentions + relations + evidence + provenance + offsets), changing ONLY the native
    detection library — the ``gliner2`` framework's ``extract_entities`` + ``extract_relations``
    instead of the ``gliner`` library's ``inference``/``predict_relations`` tuple.

    Config:
        labels: list[str]  — REQUIRED. The zero-shot entity types to extract (e.g.
            ``["organization", "person"]``). GLiNER2 works best with lower/title case.
        relation_labels: list[str]  — REQUIRED. The zero-shot relation types to extract between
            the entities (e.g. ``["works for", "located in"]``).
        label_floor: str  — ``per_chunk`` (default) or ``universal``: what the config ``labels`` /
            ``relation_labels`` MEAN for a chunk that carries induced types (T2). ``per_chunk``
            treats them as a RELEVANCE floor — a chunk that induced its own types is scored against
            THOSE ALONE, and the config lists are the recall net for a chunk induction said nothing
            about. ``universal`` restores the historical union (config ∪ induced) on every chunk.
            Default ``per_chunk``: zero-shot precision AND the span-scoring activation both degrade
            with label count, so stamping a global floor onto a chunk that already induced its own
            types is a pure tax (see :data:`_DEFAULT_LABEL_FLOOR` for the full trade). A chunk with
            no induced labels uses exactly the config lists under either value.
        model: str  — the checkpoint id (default ``fastino/gliner2-multi-v1``, Apache-2.0).
        threshold: float  — score threshold; entities/relations below it are dropped (default 0.5).
        min_chars: int  — a chunk whose stripped length is below this is skipped without invoking
            the model (default 1).
        max_len: int  — the mdeberta-v3 model token window (default 768), threaded as the native
            ``max_len`` of ``batch_extract_entities`` / ``batch_extract_relations`` (W12). A chunk
            that tokenizes past it is SILENTLY truncated by the model (tail entities/relations lost,
            offsets unreliable); this bounds the window explicitly. CAVEAT: the chunker
            (``chunk.markdown`` ``max_tokens``) counts in a DIFFERENT tokenizer, ~1.2× fewer tokens
            than mdeberta — keep the chunk cap ≤ ~640 for this 768 window; the guard is the backstop
        batch_size: int  — the **bucketing window** and progress cadence: how many chunks are
            buffered before they are partitioned by label set and each bucket run through one native
            batched forward (default 16; clamped to 64; must be ``>= 1``). It is NOT the model
            forward tensor — each bucket is split into token-budget micro-batches below — so a wider
            window only packs buckets better and coarsens the progress trail. The emitted mentions
            AND relations (offsets/labels/endpoints/order) are record-for-record identical for any
            positive batch size and any bucketing (the T7 invariance test drives a per-chunk
            reference); the clamp guarantees a large corpus always logs progress often enough to
            never *appear* hung.
        microbatch_max_texts: int  — the hard cap on texts packed into ONE native batched forward
            (default 4, the pod-safe count). gliner2 pads a micro-batch to its longest sequence, so
            this bounds the padded forward to ``max_texts × max_len`` positions regardless of
            ``batch_size`` or bucket size. With per-chunk label sets the label dimension has already
            collapsed (T7: the ~17 GiB OOM tensor was ``batch × ~768 tokens × ~40 labels``), so this
            is now the rarely-biting safety net rather than the cure.
        microbatch_token_budget: int  — the token budget (summed per-text token lengths, each capped
            at ``max_len``) for ONE forward (default 3072 = 4 × 768). The SECONDARY bound, tighter
            than ``max_texts`` when ``max_len`` is raised.
        intra_op_threads: int  — cap on torch **intra-op** threads applied at load (W9-scaling,
            default 8; ``<= 0`` disables the cap). On a many-core host gliner2's internal batching
            oversubscribes torch's default ``os.cpu_count()`` intra-op pool to ~cpu_count threads —
            a measured CPU spin with the GPU idle and no progress (``OMP_NUM_THREADS`` does NOT fix
            it: torch owns its own pool once imported). Clamped to ``os.cpu_count()``; guarded so a
            torch-free/failed cap is a logged no-op.
        quantize: bool | 'auto'  — gliner2 native ``quantize`` (fp16) load knob (W9, default
            ``auto`` → True on cuda / False on cpu). Guarded → plain load on any fault. When on, the
            model is explicitly moved to cuda after load and its device is logged (W9-scaling), so a
            fp16-converted model that silently stayed on CPU is caught, not a silent CPU spin. Set
            ``quantize: false`` if a run is CPU-bound with the GPU idle despite placement.
        compile: bool | 'auto'  — gliner2 native ``compile`` (``torch.compile`` + fused GPU kernels)
            load knob. ``auto`` (the default) resolves **True on cuda / False on cpu** — T7
            UNLOCKED it. It was forced OFF while extraction ran one varied-length chunk against a
            corpus-wide ~40-label schema per forward: gliner2 compiles with
            ``torch.compile(..., dynamic=True)``, and a fresh (label-count, batch) shape family per
            chunk thrashed torch._dynamo past its recompile_limit (pod-measured: worker pinned
            ~290% CPU, GPU 0%, no progress). Bucketing removes BOTH varying dimensions: within a
            bucket the label dimension is constant by construction, and the micro-batch caps the
            batch dimension at ``microbatch_max_texts`` — so dynamo sees one shape family per bucket
            instead of one per chunk, which is the regime the pod-measured 1.74×/2.83× came from.
            Guarded twice: a compile fault at LOAD degrades to a plain load, and the lazy
            ``BackendCompilerFailed`` on the first forward trips the inference-time
            reload-uncompiled latch (:meth:`_reload_uncompiled`) — so a backend that cannot compile
            (Blackwell / CUDA-13 nvrtc, ADR-0036) costs throughput, never correctness or a crash.
            Set ``compile: false`` to pin it off.

    Fulfils BOTH the EntityExtractor and the FusedEntityRelationExtractor seam: it also exposes
    ``extract`` so it can stand in for a plain Entity-Extraction Stage when a Pipeline wants only
    its NER half. The model is loaded lazily and cached on first use, so constructing the Provider
    never downloads weights.

    Labels (W4 seam, ADR-0038; T2 floor policy): the per-chunk entity + relation label sets come
    from the shared :func:`~latence_core.induced.effective_labels` seam. Under the default
    ``label_floor: per_chunk`` a chunk that carries induced types is scored against ITS OWN types;
    under ``label_floor: universal`` the config floor is unioned onto every chunk (the historical
    behaviour, kept as a knob). Either way a chunk with no induced labels — every stack without a
    ``schema_induction`` Stage, and predefined-only mode where induction is skipped entirely — is
    scored against exactly the config labels.

    Batching (T7): each streaming ``batch_size`` window is BUCKETED by label set — separately for
    the entity slot and the relation slot, since a chunk can share one and not the other — and each
    bucket runs through ONE native ``batch_extract_entities`` / ``batch_extract_relations`` forward
    per token-budget micro-batch, at that bucket's own small label count. Per-text results are
    scattered back by index, so emission stays in input-chunk order and the output is
    bucket-invariant (record-for-record identical to a per-chunk reference). Absent induction every
    chunk shares the config label set → ONE bucket → the whole window is one forward.

    Why this is the ROOT fix for the CUDA-OOM (dogfood-2b-process): the span-scoring activation is
    ``batch × positions × LABELS``, and the measured ~17 GiB spike was ``batch × ~768 tokens × ~40``
    corpus-wide ``labels``. Per-chunk induction collapses the label dimension to a handful, so the
    tensor never forms; the reactive micro-batch/OOM-halving backoff stays as the last-resort net
    and should now fire zero times.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default gliner2-multi-v1 checkpoint.

        ``compute="either"``: the GLiNER2 mdeberta-v3 model runs in-process on a laptop CPU
        (ADR-0007) and accelerates on a GPU when present, so the device seam routes it rather than
        skipping it on a CPU-only host.

        License **verified** per ADR-0012 / research-diligence (recorded 2026-07-10, weights AND
        code separately): the ``gliner2`` library is Apache-2.0 (code: github.com/fastino-ai/GLiNER2
        LICENSE + PyPI classifier) and the default checkpoint ``fastino/gliner2-multi-v1`` is
        Apache-2.0 (weights, per the HF model card, not gated) — both permissive, so
        ``license_verified=True``.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical across
        hardware/library versions, so the profile declares it non-deterministic rather than
        overclaim. ``memory_mb`` is the approximate resident footprint (declared, not measured).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=2048,
            model_id=_DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=(
                "https://huggingface.co/fastino/gliner2-multi-v1 (weights, Apache-2.0, not "
                "gated); https://github.com/fastino-ai/GLiNER2 (library, Apache-2.0 — LICENSE + "
                "PyPI 'OSI Approved :: Apache Software License' classifier)"
            ),
            license_verified_on="2026-07-10",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note=(
                "gliner2 native quantize (fp16) on cuda (W9; pod-measured 1.74× combined NER+RE at "
                "byte-identical output; guarded → plain load on any fault; model moved to cuda + "
                "device logged post-load). compile now rides the cuda default (T7): chunks are "
                "bucketed by label set and each bucket runs one native batched forward, so the "
                "label + batch dimensions are constant within a bucket and gliner2's "
                "torch.compile(dynamic=True) no longer recompile-thrashes; guarded at load AND by "
                "the inference-time reload-uncompiled latch. No inference server is involved: "
                "gliner2 ships none (from_api is a cloud client), so this is the native batched "
                "forward in-process. "
                "Intra-op threads capped (default 8) against many-core oversubscription; "
                "extraction streams in bounded batches with per-batch progress logging. "
                "TorchScript JIT neutralized at the shared load seam (#190: mdeberta's scripted "
                "relative-position helper dies in nvrtc on Blackwell/CUDA-13) — throughput only, "
                "byte-identical output"
            ),
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "Gliner2FusedExtractor requires a non-empty 'labels' list in its config "
                "(zero-shot NER: the entity label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        relation_labels = config.get("relation_labels")
        if not relation_labels or not isinstance(relation_labels, list):
            msg = (
                "Gliner2FusedExtractor requires a non-empty 'relation_labels' list in its "
                "config (zero-shot RE: the relation label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._relation_labels = [str(label) for label in relation_labels]
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        # W12 model-boundary guard: bound the mdeberta token window (default 768). gliner2's
        # extract_entities/extract_relations take a native ``max_len``; past it the model silently
        # truncates (tail entities/relations lost, offsets unreliable). Explicit + configurable.
        self._max_len = resolve_max_len(config)
        # T2: is the config label list a PER-CHUNK relevance floor (default — see
        # _DEFAULT_LABEL_FLOOR) or a UNIVERSAL one stamped onto every chunk? Validated eagerly, so a
        # typo'd value is an operator ConfigError at construction rather than a silently ignored key
        # that leaves the run on the other policy.
        try:
            self._label_floor = resolve_label_floor(config, default=_DEFAULT_LABEL_FLOOR)
        except ValueError as exc:
            msg = f"fused_entity_relation.gliner2: {exc}"
            raise ConfigError(msg) from exc
        # Streaming progress cadence, clamped to _MAX_BATCH_SIZE so a large configured value can
        # never let the stage run for minutes with no visible progress (W9-scaling): a corpus always
        # checkpoints/logs at least every _MAX_BATCH_SIZE chunks.
        requested_batch = validate_batch_size(
            config.get("batch_size", _DEFAULT_BATCH_SIZE),
            provider_name="fused_entity_relation.gliner2",
        )
        self._batch_size = min(requested_batch, _MAX_BATCH_SIZE)
        # T7: ``compile`` no longer carries a forced-OFF provider default — it rides the shared
        # ``auto`` resolution (True on cuda, False on cpu). The forced-off default existed because
        # a per-chunk forward against a corpus-wide ~40-label schema handed gliner2's
        # torch.compile(dynamic=True) a fresh (label-count, batch) shape family per chunk and
        # thrashed dynamo. Bucketing by label set pins the label dimension within a bucket and the
        # micro-batch pins the batch dimension, so the thrash precondition is gone; the load-time
        # guard and the inference-time reload-uncompiled latch still make a non-compiling backend a
        # throughput-only degradation.
        load_config: dict[str, Any] = dict(config)
        # The perf-seam config knobs (quantize / compile) + the intra-op thread cap, read at load
        # (W9 / W9-scaling). Validate them eagerly (fail-fast at construction, like batch_size): an
        # invalid value is an operator ConfigError up front, not a native fault buried in the lazy
        # load. device=None ⇒ the torch-free CUDA probe (GLiNER2 auto-places), matching where the
        # model will actually land.
        # (a): resolve the device ONCE, through the ADR-0036 seam, like every other learned
        # Provider. This Provider used to probe ``cuda_available()`` directly and never look at
        # ``config["device"]`` at all — so ``device: cuda`` in the blessed pod stack was a no-op
        # and, worse, ``device: cpu`` on a GPU box was silently overridden: it still resolved
        # ``quantize=True`` and still moved the model to cuda. That is #191's failure class, in the
        # Provider carrying the heaviest Stage. Resolving at construction also turns an invalid
        # ``device`` string into an operator ConfigError before any weights are downloaded.
        self._device = self._resolve_device(config.get("device", "auto"))
        try:
            resolve_gliner2_knob("quantize", self._device.device, load_config)
            resolve_gliner2_knob("compile", self._device.device, load_config)
            # Validation only: the cap itself is APPLIED by the load choreography (which must own
            # it, because it has to land before the backbone import — capping threads is what first
            # imports torch). Resolving it here keeps a bad value an operator ConfigError at
            # construction rather than a native fault buried in the lazy load.
            resolve_intra_op_threads(load_config)
            # T7: the token-budget micro-batch bounds for the bucketed forward — the safety net that
            # keeps ONE padded activation bounded (max_texts × max_len) for any bucket size.
            self._microbatch_max_texts, self._microbatch_token_budget = resolve_microbatch_bounds(
                load_config
            )
        except ValueError as exc:
            msg = f"fused_entity_relation.gliner2: {exc}"
            raise ConfigError(msg) from exc
        self._config: dict[str, Any] = load_config
        self._model: Any = None
        # Set once the inference-time compile guard has re-loaded the uncompiled model (W9), so the
        # fallback runs at most once per provider instance.
        self._compile_disabled: bool = False
        # W9-scaling: the per-batch cumulative progress trail (chunks, mentions, relations), filled
        # as extract_fused streams batches — empty until the first extract, readable for forensics.
        self._progress: list[tuple[int, int, int]] = []

    # -- lazy model load ----------------------------------------------------

    def _gliner2_cls(self) -> Any:
        """The backbone-import factory the perf seam calls at the one correct instant (#190).

        It no longer neutralizes the JIT itself: it is HANDED to
        :func:`~latence_core.providers.perf.load_gliner2_model`, which kills the TorchScript JIT and
        caps the intra-op threads and only then calls this. That import transitively pulls in
        transformers' ``deberta_v2`` module, whose relative-position helpers are compiled by
        ``@torch.jit.script`` AT IMPORT and cannot be un-scripted afterwards (the Blackwell/CUDA-13
        ``libnvrtc-builtins.so.13.0`` crash). The ordering used to be a rule this Provider had to
        re-state in a comment and an AST linter had to police; now it is a straight line of code
        inside the seam that no caller can reorder. The native dependency and its install hint stay
        here, where the package that owns them is (ADR-0036 §2).
        """
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner2' package is required to run Gliner2FusedExtractor. "
                "Install it with: pip install latence-extract-gliner2"
            )
            raise ImportError(msg) from exc
        return GLiNER2

    @classmethod
    def _resolve_device(cls, requested: object) -> DeviceDecision:
        """Route the ``device`` config value (``auto|cpu|cuda``) through the ADR-0036 device seam.

        ``compute="either"`` (see :meth:`profile`), so ``cpu`` pins CPU, ``auto`` follows
        availability, and ``cuda`` resolves to CUDA when a GPU is present — else the seam returns a
        **skip** decision, which the load choreography turns into a loud ConfigError rather than a
        silent CPU fallback (#191). A malformed device string is a ConfigError, not a TypeError.
        """
        try:
            return select_device(str(requested), cls.profile())
        except ValueError as exc:
            msg = f"fused_entity_relation.gliner2: invalid device config: {exc}"
            raise ConfigError(msg) from exc

    def _load_model(self) -> Any:
        """Load the checkpoint on first use (cached) via the shared load choreography (C4).

        One call. :func:`~latence_core.providers.perf.load_gliner2_model` owns the ordered sequence
        — skip-decision → JIT kill → intra-op thread cap → backbone import → native quantize/compile
        load (guarded, degrades to plain) → cuda placement keyed off the RESOLVED device. What this
        Provider still owns is what is genuinely its own: the batched-API check below.
        """
        if self._model is not None:
            return self._model
        self._model = load_gliner2_model(
            self._gliner2_cls,
            self._model_id,
            self._config,
            self._device,
            provider_name="fused_entity_relation.gliner2",
        )
        # T7: the loaded object must expose the BATCHED entry points this Provider now drives.
        # Checked at LOAD (not at the first forward) so a gliner2 too old to have them — or an
        # api-client object rather than a local model — is a typed, actionable ProviderError up
        # front (ADR-0034), never the raw native AttributeError an operator cannot act on (the #192
        # field failure on the twin redactor).
        _require_batched_api(self._model)
        return self._model

    def _reload_uncompiled(self) -> Any:
        """Re-load the model with ``compile=False`` (quantize kept) after an inference-time compile
        fault — the second half of the W9 compile guard (:func:`is_compile_backend_failure`).

        ``torch.compile`` compiles lazily on the first forward, so a backend that cannot compile the
        model fails INSIDE ``extract_entities``/``extract_relations`` (a ``BackendCompilerFailed``),
        which the load-time guard cannot catch. On the first such fault the provider re-loads ONCE
        with ``compile`` forced off (``quantize`` unchanged — fp16 alone is fine) and uses the
        uncompiled model for the rest of the run, so a bad compile costs only throughput, never a
        crash. Idempotent: sets ``_compile_disabled`` so the fallback runs at most once.
        """
        self._compile_disabled = True
        cfg = {**self._config, "compile": False}
        _log.warning(
            "fused_entity_relation.gliner2: torch.compile failed at inference; re-loading the "
            "uncompiled model (quantize kept) — throughput only, never a crash"
        )
        # The SAME choreography as the first load — placement included, which is the point: skipping
        # it here would leave the model that serves the REST of the run unplaced.
        self._model = load_gliner2_model(
            self._gliner2_cls,
            self._model_id,
            cfg,
            self._device,
            provider_name="fused_entity_relation.gliner2",
        )
        return self._model

    # -- fused extraction ---------------------------------------------------

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Extract entities + relations per chunk; yield mentions + relations together (S4/S5).

        Chunks below ``min_chars`` are skipped without a model call. Every remaining chunk resolves
        its OWN effective (entity, relation) label sets through the W4 seam under the T2 floor
        policy — with the default ``per_chunk`` floor a chunk that induced its own types is scored
        against THOSE ALONE, and a chunk that induced nothing (every predefined-only /
        no-induction stack) falls back to exactly the config lists.

        **T7 — bucket, then batch.** Within each ``batch_size`` window the chunks are partitioned by
        label set (:func:`~latence_core.providers.perf.iter_label_buckets`), separately per slot
        because a chunk can share the entity set and not the relation set, and each bucket is run
        through ONE native ``batch_extract_entities`` / ``batch_extract_relations`` forward per
        token-budget micro-batch, with the bucket's own SMALL schema. Per-text results scatter back
        by index, so the per-chunk record building (offset/page resolution, relation→mention span
        resolution) and the emission order are UNCHANGED — the output is bucket-invariant,
        record-for-record identical to calling the model once per chunk.

        **W9-scaling — incremental, batch-granular progress:** after each window the provider logs
        cumulative progress (chunks / mentions / relations). On a large corpus (the diagnosed
        2,802-chunk run) the stage emits a steady progress trail instead of running silently for
        tens of minutes — so a slow run is visible, never "appears hung", and the log doubles as a
        coarse resume-forensics trail alongside the Runner's checkpoint writes.
        """
        loaded = False
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        processed = 0
        #: cumulative (chunks_processed, mentions, relations) recorded per batch — the progress
        #: trail an operator (or a test) reads to confirm the stage streams incrementally.
        self._progress = []
        # ONE resolver for the whole call: page maps are DOCUMENT-scoped, so a document's index
        # is built on the first chunk of it we see and reused for every later chunk. It is handed
        # to _iter_batches so it observes EVERY chunk, including the ones min_chars drops.
        resolver = PageIndexResolver()

        for batch_count, batch in enumerate(
            _iter_batches(chunks, self._batch_size, self._min_chars, resolver), start=1
        ):
            if not loaded:
                self._load_model()
                loaded = True
            pending = [self._prepare(chunk) for chunk in batch]
            raw_entities = self._bucketed_forward(pending, _ENTITY_SLOT)
            raw_relations = self._bucketed_forward(pending, _RELATION_SLOT)
            for item, ents, rels in zip(pending, raw_entities, raw_relations, strict=True):
                chunk_mentions, chunk_relations = self._build_records(
                    item.chunk, ents, rels, resolver
                )
                mentions.extend(chunk_mentions)
                relations.extend(chunk_relations)
                processed += 1
            self._progress.append((processed, len(mentions), len(relations)))
            _log.info(
                "fused_entity_relation.gliner2: extract progress — batch %d (%d chunks done, "
                "%d mentions, %d relations)",
                batch_count,
                processed,
                len(mentions),
                len(relations),
            )
        return FusedExtraction(mentions=mentions, relations=relations)

    def _prepare(self, chunk: ChunkRecord) -> _PendingChunk:
        """Resolve a chunk's two effective label sets BEFORE any model call (the bucket keys, T7).

        W4 label-source seam (ADR-0038) under the T2 per-chunk floor policy: with the default
        PER_CHUNK floor a chunk that induced its own entity/relation types is scored against THOSE
        ALONE (no global floor re-inflating the label count); a chunk that induced nothing — and
        every predefined-only / no-induction stack — falls back to exactly the config labels.
        ``label_floor: universal`` restores the historical union.

        The two sets are kept as TUPLES because they double as the bucket keys: label ORDER is
        significant (it is the order handed to the model, and gliner2's per-label output dict is
        keyed by it), so two chunks bucket together only when they would have been asked the
        identical question.
        """
        return _PendingChunk(
            chunk=chunk,
            entity_labels=tuple(
                effective_labels(
                    self._labels, chunk.induced_labels, LabelSlot.ENTITY, floor=self._label_floor
                )
            ),
            relation_labels=tuple(
                effective_labels(
                    self._relation_labels,
                    chunk.induced_labels,
                    LabelSlot.RELATION,
                    floor=self._label_floor,
                )
            ),
        )

    def _bucketed_forward(
        self, pending: list[_PendingChunk], slot: _Slot
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Run one slot (entities or relations) over ``pending`` as ONE forward per label bucket.

        The T7 core. ``pending`` is a whole ``batch_size`` window; its chunks are partitioned by
        this slot's label set, and each bucket is handed to the slot's native BATCHED entry point
        with that bucket's own small schema. A bucket is further split by the token-budget
        micro-batch bound, so ONE padded activation stays bounded at
        ``microbatch_max_texts × max_len`` positions no matter how large the bucket is.

        Returns the per-chunk raw output ALIGNED with ``pending`` (index i ↔ pending[i]) — results
        are scattered back by index, never by output order, so bucketing can never permute a
        chunk's result onto another chunk. That, plus gliner2's own batched span-rep path being
        documented as "bit-identical to calling compute_span_rep per sample", is why the emitted
        records are bucket-invariant.
        """
        keys = [slot.labels_of(item) for item in pending]
        results: list[dict[str, list[dict[str, Any]]] | None] = [None] * len(pending)
        for labels, indices in iter_label_buckets(keys):
            texts = [pending[i].chunk.content for i in indices]
            for start, end in iter_token_budget_microbatches(
                texts,
                max_len=self._max_len,
                max_texts=self._microbatch_max_texts,
                token_budget=self._microbatch_token_budget,
            ):
                raw = self._batch_forward(slot, texts[start:end], list(labels))
                if len(raw) != end - start:
                    msg = (
                        f"fused_entity_relation.gliner2: {slot.method} returned {len(raw)} "
                        f"results for a micro-batch of {end - start} texts."
                    )
                    raise ProviderError(msg)
                for offset, item in enumerate(raw):
                    results[indices[start + offset]] = _unwrap(slot, item)
        # Every index belongs to exactly one bucket and every bucket is tiled by micro-batches, so
        # no slot can stay unfilled; the assertion-free narrowing keeps mypy --strict honest.
        return [{} if item is None else item for item in results]

    def _batch_forward(
        self, slot: _Slot, texts: list[str], labels: list[str]
    ) -> list[Any]:
        """One native batched forward, with the compile latch AND the CUDA-OOM halving backoff.

        Two inference-time faults are recovered here so ``latence process`` never dies mid-run:

        * **compile fault** — ``torch.compile`` compiles LAZILY on the first forward, so a backend
          that cannot compile raises a ``BackendCompilerFailed`` (dynamo/inductor) HERE, not at load
          (:func:`~latence_core.providers.perf.is_compile_backend_failure`). The provider re-loads
          the uncompiled model ONCE (:meth:`_reload_uncompiled`; quantize kept) and retries — the
          graceful degradation that keeps the T7 compile default safe on a Blackwell/CUDA-13 box.

        * **CUDA OOM** — the last-resort net (T7 makes it near-dead code: with per-chunk label sets
          the span-scoring tensor's label dimension has collapsed, and that dimension is what made
          it ~17 GiB). Because the output is batch-invariant, the micro-batch is split in half and
          each half retried after a best-effort cache release; a batch of ONE that still OOMs is
          genuinely unrecoverable and surfaces as a typed error — no dropped chunk, no fake green.

        Any other native fault is already a typed :class:`~latence_core.errors.ProviderError`
        carrying only the op + native type name (no chunk text / PII — C2/C6).
        """
        try:
            return self._native_batch(self._model, slot, texts, labels)
        except Exception as exc:  # noqa: BLE001 - intercept recoverable faults before surfacing
            # ``_native_batch`` wraps the native fault as ProviderError; both predicates walk the
            # cause chain to the original BackendCompilerFailed / OutOfMemoryError underneath.
            if not self._compile_disabled and is_compile_backend_failure(exc):
                return self._native_batch(self._reload_uncompiled(), slot, texts, labels)
            if is_cuda_oom(exc) and len(texts) > 1:
                empty_cuda_cache(provider_name="fused_entity_relation.gliner2")
                mid = len(texts) // 2
                # Re-enter through _batch_forward so a compile-reload inside one half is picked up
                # by the next (self._model is the source of truth after _reload_uncompiled).
                left = self._batch_forward(slot, texts[:mid], labels)
                right = self._batch_forward(slot, texts[mid:], labels)
                return left + right
            raise

    def _native_batch(
        self, model: Any, slot: _Slot, texts: list[str], labels: list[str]
    ) -> list[Any]:
        """The single native call, under ``inference_mode``, typed on any fault.

        ``include_spans=True`` is REQUIRED — without it gliner2 returns entity *strings* / relation
        head/tail *text* only, which the framework contract cannot ground; the Provider always
        passes it. The forward runs INSIDE
        :func:`~latence_core.providers.perf.inference_context` (``torch.inference_mode()``): this is
        an extraction-only encoder that never backpropagates, so building an autograd graph and
        pinning its activations alive would only ACCUMULATE VRAM run-over-run to a CUDA OOM
        (dogfood-2b-process). Any native fault becomes a typed :class:`ProviderError` naming the op
        and the native exception type ONLY — never the chunk text (C2/C6).
        """
        try:
            with inference_context():
                return list(
                    getattr(model, slot.method)(
                        texts,
                        labels,
                        threshold=self._threshold,
                        include_spans=True,
                        include_confidence=True,
                        max_len=self._max_len,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - map native → typed, carrying NO input text/PII
            msg = (
                f"fused_entity_relation.gliner2: provider failure in {slot.method!r} "
                f"(native {type(exc).__name__})"
            )
            raise ProviderError(msg) from exc

    # -- plain Entity-Extraction half (so it also fits the EntityExtractor seam) --

    def extract(self, chunks: Iterable[ChunkRecord]) -> list[EntityMention]:
        """The NER half alone (for a Pipeline that wants only entities from this model)."""
        return self.extract_fused(chunks).mentions

    def _build_records(
        self,
        chunk: ChunkRecord,
        entities_by_label: dict[str, list[dict[str, Any]]],
        relations_by_type: dict[str, list[dict[str, Any]]],
        resolver: PageIndexResolver,
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        """Map the REAL gliner2 per-label/per-relation dicts → the framework mention/relation set.

        Entities: flatten the ``{label: [{"start","end","text","confidence"}]}`` dict → one mention
        per row, offsets resolved through the chunk's stripped→original ``offset_map`` (ADR-0031),
        keyed by chunk-local ``(start, end)`` so a relation endpoint referencing those offsets
        resolves to the mention. Relations: flatten the ``{rel_type: [{"head": {...}, "tail":
        {...}}]}`` dict → one relation per row, resolving head/tail by their chunk-local
        ``(start, end)`` to the mentions above (a relation whose endpoint is not among the emitted
        mentions is dropped — no un-resolvable record leaks). A relation's covering span is the
        min/max of its resolved endpoints, so it inherits the offset correction. Deterministic
        order: entities by ``(start, label)``, relations by ``(head start, head end, rel_type)``.
        """
        # Flatten entities per label into deterministic (start, end, label, confidence) rows.
        flat_entities: list[dict[str, Any]] = []
        for label, rows in entities_by_label.items():
            label_str = str(label)
            if not label_str:
                continue
            for row in rows:
                if "start" not in row or "end" not in row:
                    continue  # no span (include_spans not honoured) — cannot ground, skip
                flat_entities.append(
                    {
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "label": label_str,
                        "confidence": float(row.get("confidence", 1.0)),
                    }
                )
        flat_entities.sort(key=lambda e: (e["start"], str(e["label"])))

        offset_index = OffsetIndex(chunk.offset_map) if chunk.offset_map is not None else None
        chunk_char_start = chunk.provenance.char_start or 0
        # Each record's page span comes from the DOCUMENT-scoped resolver, so a chunk carrying
        # no page map (every non-first chunk since v18) still resolves its own pages.

        mentions: list[EntityMention] = []
        by_span: dict[tuple[int, int], EntityMention] = {}
        skipped_degenerate = 0
        for i, ent in enumerate(flat_entities):
            local_start = ent["start"]
            local_end = ent["end"]
            # dogfood-2b-process: GLiNER2 can return a DEGENERATE span whose resolved surface
            # text is empty (``local_start == local_end``, or a slice that yields ""). Such a span
            # carries no entity, and an EntityMention with ``text=""`` violates its >=1-char
            # contract (pydantic ``string_too_short``) — which crashed a full pipeline run at the
            # extract stage. Skip it (its ids stay unique; no by_span entry means any relation
            # endpoint referencing it is dropped, so the relation path is guarded the same way).
            # NOT silent (ADR-0034): count skips and log the count below.
            mention_text = chunk.content[local_start:local_end]
            if len(mention_text) == 0:
                skipped_degenerate += 1
                continue
            if offset_index is not None:
                doc_start, doc_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_start, doc_end = chunk_char_start + local_start, chunk_char_start + local_end
            provenance = _mention_provenance(
                chunk.provenance,
                doc_start,
                doc_end,
                resolver.pages_for_span(chunk, doc_start, doc_end),
            )
            mention = EntityMention(
                record_id=f"{chunk.record_id}#entity-{i}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=i,
                label=str(ent["label"]),
                text=mention_text,
                confidence=min(max(float(ent["confidence"]), 0.0), 1.0),
                provenance=provenance,
                classification=chunk.classification,
            )
            mentions.append(mention)
            # First mention wins a given span (deterministic), matching relex's by_span semantics.
            by_span.setdefault((local_start, local_end), mention)

        if skipped_degenerate:
            # ADR-0034 (no silent errors): a degenerate zero-length span carries no entity and is
            # dropped rather than crashing the stage — but the drop is COUNTED and surfaced.
            _log.info(
                "fused_entity_relation.gliner2: skipped %d degenerate zero-length entity span(s) "
                "in chunk %s (empty surface text — cannot ground a mention)",
                skipped_degenerate,
                chunk.record_id,
            )

        # Flatten relations per type into deterministic rows carrying their rel_type label.
        flat_relations: list[dict[str, Any]] = []
        for rel_type, rows in relations_by_type.items():
            rel_type_str = str(rel_type)
            if not rel_type_str:
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                flat_relations.append({**row, "relation": rel_type_str})

        relations: list[RelationMention] = []
        emitted = 0
        for rel in sorted(flat_relations, key=_relation_sort_key):
            head = _resolve_endpoint(rel, "head", by_span)
            tail = _resolve_endpoint(rel, "tail", by_span)
            label = str(rel.get("relation", ""))
            if head is None or tail is None or head.record_id == tail.record_id or not label:
                continue
            span_start = min(head.provenance.char_start or 0, tail.provenance.char_start or 0)
            span_end = max(head.provenance.char_end or 0, tail.provenance.char_end or 0)
            rel_prov = _relation_provenance(
                head,
                tail,
                span_start,
                span_end,
                # A covering span runs between two endpoints and is not a sub-chunk offset,
                # so it is answered from the endpoints' own already-exact pages rather than by
                # a lookup against any one chunk.
                pages_for_covering_span(head.provenance, tail.provenance),
            )
            relations.append(
                RelationMention(
                    record_id=f"{chunk.record_id}#relation-{emitted}",
                    document_record_id=chunk.document_record_id,
                    index=emitted,
                    label=label,
                    head_mention_id=head.record_id,
                    tail_mention_id=tail.record_id,
                    head_text=head.text,
                    tail_text=tail.text,
                    confidence=min(max(_relation_confidence(rel), 0.0), 1.0),
                    provenance=rel_prov,
                    classification=chunk.classification,
                )
            )
            emitted += 1
        return mentions, relations


def _iter_batches(
    chunks: Iterable[ChunkRecord],
    batch_size: int,
    min_chars: int,
    resolver: PageIndexResolver,
) -> Iterable[list[ChunkRecord]]:
    """Group the (lazy) chunk stream into bounded batches of processable chunks (W9-scaling).

    Chunks whose stripped content is below ``min_chars`` are dropped here (skipped without a model
    call, exactly as before), so a batch is ``batch_size`` chunks that will each be extracted — the
    progress cadence counts real work, not skipped empties. ``batch_size`` is already clamped to
    :data:`_MAX_BATCH_SIZE` by the provider, so a batch is bounded regardless of config. Streaming:
    it consumes the input iterable lazily and never materialises more than one batch at a time (the
    accumulated OUTPUT records are inherent to the FusedExtraction contract, but the input transient
    stays O(batch_size), ADR-0033).
    """
    batch: list[ChunkRecord] = []
    for chunk in chunks:
        if len(chunk.content.strip()) < min_chars:
            continue
        batch.append(chunk)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _relation_confidence(rel: dict[str, Any]) -> float:
    """A relation's confidence: the min of its head/tail endpoint confidences (weakest link).

    Per the REAL gliner2 1.3.2 shape, ``extract_relations(include_confidence=True)`` carries
    confidence ONLY on the head/tail sub-dicts, NOT at the relation level — so a relation's score is
    derived from its endpoints (the weakest-link min), falling back to 1.0 when neither endpoint
    carries one. (A top-level ``confidence`` is still honoured if a future version adds one, so the
    provider is forward-safe without depending on a field the current API does not emit.)
    """
    if "confidence" in rel:
        return float(rel["confidence"])
    scores: list[float] = []
    for role in ("head", "tail"):
        endpoint = rel.get(role)
        if isinstance(endpoint, dict) and "confidence" in endpoint:
            scores.append(float(endpoint["confidence"]))
    return min(scores) if scores else 1.0


def _relation_sort_key(rel: dict[str, Any]) -> tuple[int, int, str]:
    head = rel.get("head", {})
    start = int(head.get("start", 0)) if isinstance(head, dict) else 0
    end = int(head.get("end", 0)) if isinstance(head, dict) else 0
    label = str(rel.get("relation", ""))
    return (start, end, label)


def _resolve_endpoint(
    rel: dict[str, Any], role: str, by_span: dict[tuple[int, int], EntityMention]
) -> EntityMention | None:
    endpoint = rel.get(role)
    if not isinstance(endpoint, dict):
        return None
    try:
        key = (int(endpoint["start"]), int(endpoint["end"]))
    except (KeyError, TypeError, ValueError):
        return None
    return by_span.get(key)


def _mention_provenance(
    chunk_prov: Provenance,
    doc_char_start: int,
    doc_char_end: int,
    pages: tuple[int, int] | None,
) -> Provenance:
    """``pages`` is what the :class:`~latence_core.page_index.PageIndexResolver` resolved for the
    mention's own span; ``None`` (no page map available) degrades to the chunk's inherited range."""
    update: dict[str, int] = {"char_start": doc_char_start, "char_end": doc_char_end}
    if pages is not None:
        update["page_start"], update["page_end"] = pages
    return chunk_prov.model_copy(update=update)


def _relation_provenance(
    head: EntityMention,
    tail: EntityMention,
    span_start: int,
    span_end: int,
    pages: tuple[int, int] | None,
) -> Provenance:
    base = head.provenance
    update: dict[str, int] = {"char_start": span_start, "char_end": span_end}
    if pages is not None:
        update["page_start"], update["page_end"] = pages
    else:
        hp, tp = head.provenance, tail.provenance
        starts = [p for p in (hp.page_start, tp.page_start) if p is not None]
        ends = [p for p in (hp.page_end, tp.page_end) if p is not None]
        if starts:
            update["page_start"] = min(starts)
        if ends:
            update["page_end"] = max(ends)
    return base.model_copy(update=update)
