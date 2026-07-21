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

Pod-validated gliner2 1.3.2 API (REAL — mirrored exactly in the offline fake):

.. code-block:: python

    from gliner2 import GLiNER2
    model = GLiNER2.from_pretrained("fastino/gliner2-multi-v1", quantize=True, compile=True)
    ents = model.extract_entities(text, entity_types, threshold=0.5,
                                  include_spans=True, include_confidence=True)
    # -> {"entities": {label: [{"text", "confidence", "start", "end"}, ...], ...}}
    rels = model.extract_relations(text, relation_types, threshold=0.5,
                                   include_spans=True, include_confidence=True)
    # -> {"relation_extraction": {rel_type: [{"head": {"text","confidence","start","end"},
    #                                         "tail": {...}}, ...], ...}}

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
re-resolved through the parent's page map via :class:`~latence_core.pagemap.PageOffsetIndex`. A
relation's covering span is the min/max of its resolved endpoints, so it inherits the correction.

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
guarded so a fault degrades to a plain load (never a crash). W9-scaling hardening (Blackwell): the
native ``compile`` knob defaults **OFF** (opt-in) because gliner2's ``dynamic=True`` compile
recompile-thrashes on varied-length production chunks; torch intra-op threads are capped at load
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
from typing import TYPE_CHECKING, Any

from latence_core.capability import FusedExtraction
from latence_core.contracts import EntityMention, RelationMention
from latence_core.errors import ConfigError, ProviderError
from latence_core.induced import LabelSlot, effective_labels
from latence_core.offsetmap import OffsetIndex
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.adapter import validate_batch_size
from latence_core.providers.device import cuda_available
from latence_core.providers.perf import (
    cap_intra_op_threads,
    ensure_gliner2_on_cuda,
    inference_context,
    is_compile_backend_failure,
    load_gliner2_optimized,
    neutralize_torch_jit,
    resolve_gliner2_knob,
    resolve_intra_op_threads,
    resolve_max_len,
)
from latence_core.providers.profile import ProviderProfile

_log = logging.getLogger("latence.provider.gliner2_fused")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from latence_core.contracts import ChunkRecord, Provenance

# Apache-2.0 weights (verified 2026-07-10, HF model card, not gated). Multilingual fused NER+RE
# over an mdeberta-v3 encoder via the fastino GLiNER2 framework.
_DEFAULT_MODEL = "fastino/gliner2-multi-v1"

# The default streaming batch size: the number of chunks processed between progress checkpoints
# (each chunk is still ONE bounded ≤ max_len forward — this provider does not pack multiple chunks
# into a single native call, so batch_size shapes the PROGRESS cadence, not a padded batch tensor).
# Bounded by _MAX_BATCH_SIZE so a large configured value can never make the stage run silently for
# minutes with no visible progress (the "appears hung" failure). The emitted mentions/relations are
# record-for-record identical for any positive batch size (input-order, per-chunk — ADR-0033).
_DEFAULT_BATCH_SIZE = 16

# The hard cap on the streaming batch (progress cadence). A configured batch_size above this is
# clamped so extraction ALWAYS logs incremental progress at least this often on a large corpus —
# the "a large single batch must never wedge / silently stall the stage" guarantee (W9-scaling).
_MAX_BATCH_SIZE = 64

# W9-scaling: torch.compile defaults OFF for this provider (opt-in only). gliner2 compiles with
# ``dynamic=True``, which RECOMPILES per distinct chunk-shape; on a real varied-length corpus that
# thrashes torch._dynamo past its recompile_limit (measured: worker pinned ~290% CPU, GPU 0%, no
# progress after ~50 min). The pod-measured 2.83×/1.74× was on UNIFORM batch shapes, not production
# corpora. Compile stays available as a config knob (``compile: true``) for a homogeneous workload,
# with the docstring warning. NOTE: the W9 inference-time reload-uncompiled latch only catches
# compile *crashes* (BackendCompilerFailed), NOT this silent recompile *thrash* — so the safe
# default is the real fix, not the latch.
_COMPILE_DEFAULT = False


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
        model: str  — the checkpoint id (default ``fastino/gliner2-multi-v1``, Apache-2.0).
        threshold: float  — score threshold; entities/relations below it are dropped (default 0.5).
        min_chars: int  — a chunk whose stripped length is below this is skipped without invoking
            the model (default 1).
        max_len: int  — the mdeberta-v3 model token window (default 768), threaded as the native
            ``max_len`` of ``extract_entities`` / ``extract_relations`` (W12). A chunk that
            tokenizes past it is SILENTLY truncated by the model (tail entities/relations lost,
            offsets unreliable); this bounds the window explicitly. CAVEAT: the chunker
            (``chunk.markdown`` ``max_tokens``) counts in a DIFFERENT tokenizer, ~1.2× fewer tokens
            than mdeberta — keep the chunk cap ≤ ~640 for this 768 window; the guard is the backstop
        batch_size: int  — the streaming **progress cadence**: how many chunks are processed
            between incremental progress checkpoints (default 16; clamped to 64). Each chunk is
            still ONE bounded ≤ ``max_len`` forward — this provider does NOT pack multiple chunks
            into one native call — so batch_size shapes the progress/GC granularity, not a padded
            batch tensor; the emitted mentions AND relations (offsets/labels/endpoints/order) are
            record-for-record identical for any positive batch size. Must be ``>= 1``. The clamp
            guarantees a large corpus always logs progress often enough to never *appear* hung.
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
        compile: bool | 'auto'  — gliner2 native ``compile`` (torch.compile + fused GPU kernels)
            load knob. **Default OFF** (opt-in) for this provider (W9-scaling): gliner2 compiles
            with ``dynamic=True`` and RECOMPILES per distinct chunk-shape, which thrashes
            torch._dynamo past its recompile_limit on a real varied-length corpus (CPU-pinned, GPU
            idle, no progress). Enable ``compile: true`` ONLY for a homogeneous / uniform-length
            workload. Guarded like ``quantize``; the inference-time reload-uncompiled latch catches
            compile *crashes*, but NOT the recompile *thrash* — hence the off default.

    Fulfils BOTH the EntityExtractor and the FusedEntityRelationExtractor seam: it also exposes
    ``extract`` so it can stand in for a plain Entity-Extraction Stage when a Pipeline wants only
    its NER half. The model is loaded lazily and cached on first use, so constructing the Provider
    never downloads weights.

    Labels (W4 seam, ADR-0038): the entity + relation label sets are the config floors UNIONed with
    any types the OPTIONAL ``schema_induction`` Stage induced for a chunk's document (induced ∪
    config, deduped + order-stable, per slot) — exactly like relex. Absent (the default, every
    existing stack) these are exactly the config labels.
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
                "device logged post-load). compile defaults OFF (W9-scaling: dynamic=True "
                "recompile-thrashes on varied-length corpora; opt-in for uniform workloads). "
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
        # Streaming progress cadence, clamped to _MAX_BATCH_SIZE so a large configured value can
        # never let the stage run for minutes with no visible progress (W9-scaling): a corpus always
        # checkpoints/logs at least every _MAX_BATCH_SIZE chunks.
        requested_batch = validate_batch_size(
            config.get("batch_size", _DEFAULT_BATCH_SIZE),
            provider_name="fused_entity_relation.gliner2",
        )
        self._batch_size = min(requested_batch, _MAX_BATCH_SIZE)
        # W9-scaling: default torch.compile OFF for this provider (opt-in only) — gliner2's
        # dynamic=True compile recompile-thrashes on varied-length production chunks. An operator
        # who sets compile explicitly still gets it; absent, it is forced off. The perf-seam load
        # (load_gliner2_optimized) reads compile from self._config, so injecting the default here is
        # the single source of truth for both the eager validation below and the lazy load.
        load_config: dict[str, Any] = dict(config)
        if "compile" not in load_config:
            load_config["compile"] = _COMPILE_DEFAULT
        # The perf-seam config knobs (quantize / compile) + the intra-op thread cap, read at load
        # (W9 / W9-scaling). Validate them eagerly (fail-fast at construction, like batch_size): an
        # invalid value is an operator ConfigError up front, not a native fault buried in the lazy
        # load. device=None ⇒ the torch-free CUDA probe (GLiNER2 auto-places), matching where the
        # model will actually land.
        try:
            resolve_gliner2_knob("quantize", None, load_config)
            resolve_gliner2_knob("compile", None, load_config)
            self._intra_op_threads = resolve_intra_op_threads(load_config)
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
        # W9-jit (#190, audit R1): neutralize the TorchScript JIT BEFORE the gliner2 import — that
        # import transitively pulls in transformers' deberta_v2 modeling module, whose
        # relative-position helpers are compiled by @torch.jit.script AT IMPORT. Neutralizing later
        # (inside load_gliner2_optimized, which is entered only AFTER this class is resolved) cannot
        # un-script them: the env switch is read by torch only at ITS import, and
        # jit._state.disable() only affects FUTURE script() calls — so the scripted helpers would
        # still die in nvrtc (libnvrtc-builtins.so.13.0) on the first forward on a Blackwell/CUDA-13
        # box. Throughput-only, byte-identical output.
        neutralize_torch_jit(provider_name="fused_entity_relation.gliner2")
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner2' package is required to run Gliner2FusedExtractor. "
                "Install it with: pip install latence-extract-gliner2"
            )
            raise ImportError(msg) from exc
        return GLiNER2

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        # W9-jit (#190, audit R1): FIRST, before anything imports torch — ``cap_intra_op_threads``
        # below does — so the JIT env switch is already off in the environment torch reads at
        # ITS import, not only after. Idempotent with the call in ``_gliner2_cls`` (which
        # guarantees the ordering
        # against the gliner2/deberta_v2 import for both load sites).
        neutralize_torch_jit(provider_name="fused_entity_relation.gliner2")
        # W9-scaling: cap torch intra-op threads BEFORE the load/first forward so gliner2's
        # internal batching cannot oversubscribe a many-core host (the ~cpu_count-thread CPU spin
        # with the GPU idle). Guarded → a torch-free/failed cap is a logged no-op.
        cap_intra_op_threads(
            self._intra_op_threads, provider_name="fused_entity_relation.gliner2"
        )
        # W9 perf seam: the gliner2 framework exposes NATIVE quantize (fp16) + compile
        # (torch.compile + fused GPU kernels) kwargs on from_pretrained — pod-measured 1.74×
        # combined NER+RE at byte-identical output. quantize defaults True on cuda / False on cpu;
        # compile is defaulted OFF here (W9-scaling recompile-thrash guard) unless the operator opts
        # in. GUARDED so a quantize/compile fault degrades to a plain load (throughput only, never a
        # crash). device=None ⇒ the torch-free CUDA probe (GLiNER2 auto-places on cuda).
        self._model = load_gliner2_optimized(
            self._gliner2_cls().from_pretrained,
            self._model_id,
            None,
            self._config,
            provider_name="fused_entity_relation.gliner2",
        )
        # W9-scaling: when a GPU is present, ensure the (fp16) model is actually on cuda and LOG its
        # device — the defence against the fp16-eager GPU-idle hang (a fp16-converted model that
        # silently stayed on CPU → a CPU-bound spin). Best-effort + guarded (never crashes); a model
        # from_pretrained already placed correctly makes the move a harmless no-op.
        ensure_gliner2_on_cuda(
            self._model,
            cuda=cuda_available(),
            provider_name="fused_entity_relation.gliner2",
        )
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
        self._model = load_gliner2_optimized(
            self._gliner2_cls().from_pretrained,
            self._model_id,
            None,
            cfg,
            provider_name="fused_entity_relation.gliner2",
        )
        # Keep the uncompiled reload on cuda too (W9-scaling) — the placement/logging is idempotent.
        ensure_gliner2_on_cuda(
            self._model,
            cuda=cuda_available(),
            provider_name="fused_entity_relation.gliner2",
        )
        return self._model

    # -- fused extraction ---------------------------------------------------

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Extract entities + relations per chunk; yield mentions + relations together (S4/S5).

        Chunks below ``min_chars`` are skipped without a model call. For each remaining chunk the
        effective (entity, relation) label set is the config floor UNIONed with any induced types
        (the W4 seam), and the model is called ONCE for entities and once for relations — the two
        native gliner2 calls (``extract_entities`` + ``extract_relations``, both
        ``include_spans=True``). The per-chunk record building (offset/page resolution, relation→
        mention span resolution) is UNCHANGED from the relex path; emission is in input (chunk)
        order. The in-flight *model* state is bounded per chunk (each forward is one ≤ ``max_len``
        call, never the whole corpus, ADR-0033).

        **W9-scaling — incremental, batch-granular progress:** chunks are processed in bounded
        batches of ``batch_size`` (clamped to :data:`_MAX_BATCH_SIZE`); after each batch the
        provider logs cumulative progress (chunks / mentions / relations). On a large corpus (the
        diagnosed 2,802-chunk run) this means the stage emits a steady progress trail instead of
        running silently for tens of minutes — so a slow run is visible, never "appears hung", and
        the log doubles as a coarse resume-forensics trail alongside the Runner's checkpoint writes.
        """
        loaded = False
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        processed = 0
        #: cumulative (chunks_processed, mentions, relations) recorded per batch — the progress
        #: trail an operator (or a test) reads to confirm the stage streams incrementally.
        self._progress = []

        for batch_count, batch in enumerate(
            _iter_batches(chunks, self._batch_size, self._min_chars), start=1
        ):
            if not loaded:
                self._load_model()
                loaded = True
            for chunk in batch:
                # W4 label-source seam (ADR-0038): UNION the config labels with the chunk's induced
                # entity/relation types, deduped + order-stable, per slot. Absent ⇒ config labels.
                entity_labels = effective_labels(
                    self._labels, chunk.induced_labels, LabelSlot.ENTITY
                )
                relation_labels = effective_labels(
                    self._relation_labels, chunk.induced_labels, LabelSlot.RELATION
                )
                # The forward reads self._model (so an inference-time compile fallback that re-loads
                # the uncompiled model is picked up transparently on the next call, W9).
                raw_entities = self._extract_entities(chunk.content, entity_labels)
                raw_relations = self._extract_relations(chunk.content, relation_labels)
                chunk_mentions, chunk_relations = self._build_records(
                    chunk, raw_entities, raw_relations
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

    def _forward(self, op: str, invoke: Callable[[Any], Any]) -> Any:
        """Run one native forward (``invoke(model)``), with the W9 inference-time compile guard.

        ``torch.compile`` compiles lazily on the first forward, so a backend that cannot compile the
        model raises a ``BackendCompilerFailed`` (dynamo/inductor) INSIDE ``invoke`` — which the
        load-time guard cannot catch. This wrapper catches exactly that class of fault
        (:func:`~latence_core.providers.perf.is_compile_backend_failure`) ONCE, re-loads the model
        with ``compile`` forced off (:meth:`_reload_uncompiled`; ``quantize`` kept), and retries the
        forward against the uncompiled model — so a bad compile costs only throughput, never a
        crash. The fallback runs at most once (``_compile_disabled``). Any OTHER native fault (a
        genuine model/runtime error) is mapped to a typed :class:`ProviderError` carrying NO input
        text / PII (only the op + native type name — C2/C6), exactly as before.

        Every forward (both the first attempt and the uncompiled retry) runs INSIDE
        :func:`~latence_core.providers.perf.inference_context` (``torch.inference_mode()``): this is
        an extraction-only encoder that never backpropagates, so building an autograd graph and
        pinning its activations alive would only ACCUMULATE VRAM run-over-run to a CUDA OOM
        (dogfood-2b-process). Under inference mode no graph is built and each forward's activations
        free immediately — same entities/relations, bounded memory (torch-free-safe offline).
        """
        try:
            with inference_context():
                return invoke(self._model)
        except Exception as exc:  # noqa: BLE001 - map native → typed / compile → uncompiled retry
            if not self._compile_disabled and is_compile_backend_failure(exc):
                model = self._reload_uncompiled()
                # Retry once on the uncompiled model. A fault here is a genuine error (surfaced).
                try:
                    with inference_context():
                        return invoke(model)
                except Exception as exc2:  # noqa: BLE001 - typed, carrying NO input text/PII
                    msg = (
                        f"fused_entity_relation.gliner2: provider failure in {op!r} "
                        f"(native {type(exc2).__name__})"
                    )
                    raise ProviderError(msg) from exc2
            msg = (
                f"fused_entity_relation.gliner2: provider failure in {op!r} "
                f"(native {type(exc).__name__})"
            )
            raise ProviderError(msg) from exc

    def _extract_entities(
        self, text: str, entity_labels: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Call the native ``extract_entities`` and return its per-label dict of rows.

        The REAL gliner2 shape (pod-validated, mirrored in the offline fake):
        ``{"entities": {label: [{"text", "confidence", "start", "end"}, ...]}}`` — a per-label
        dict, NOT the ``gliner`` library's flat list. ``include_spans=True`` is REQUIRED for the
        ``start``/``end`` char offsets the framework grounds on. The forward runs through
        :meth:`_forward` (the W9 compile guard + typed-error mapping, no PII leak).
        """
        out: Any = self._forward(
            "extract_entities",
            lambda model: model.extract_entities(
                text,
                entity_labels,
                threshold=self._threshold,
                include_spans=True,
                include_confidence=True,
                max_len=self._max_len,
            ),
        )
        by_label: Any = out["entities"] if isinstance(out, dict) else out
        if not isinstance(by_label, dict):
            got = type(by_label).__name__
            msg = (
                "fused_entity_relation.gliner2: unexpected extract_entities output shape "
                f"(expected {{'entities': {{label: [rows]}}}}, got {got})"
            )
            raise ProviderError(msg)
        return by_label

    def _extract_relations(
        self, text: str, relation_labels: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Call the native ``extract_relations`` and return its per-relation dict of head/tail rows.

        The REAL gliner2 shape (pod-validated, mirrored in the offline fake):
        ``{"relation_extraction": {rel_type: [{"head": {"text","confidence","start","end"},
        "tail": {...}}, ...]}}`` — a per-relation-type dict of head/tail sub-dict rows, NOT a flat
        list. ``include_spans=True`` is REQUIRED so head/tail carry the ``start``/``end`` char
        offsets a relation is resolved to its mentions by. The forward runs through :meth:`_forward`
        (the W9 compile guard + typed-error mapping, no PII leak).
        """
        out: Any = self._forward(
            "extract_relations",
            lambda model: model.extract_relations(
                text,
                relation_labels,
                threshold=self._threshold,
                include_spans=True,
                include_confidence=True,
                max_len=self._max_len,
            ),
        )
        by_rel: Any = out["relation_extraction"] if isinstance(out, dict) else out
        if not isinstance(by_rel, dict):
            got = type(by_rel).__name__
            msg = (
                "fused_entity_relation.gliner2: unexpected extract_relations output shape "
                f"(expected {{'relation_extraction': {{rel: [rows]}}}}, got {got})"
            )
            raise ProviderError(msg)
        return by_rel

    # -- plain Entity-Extraction half (so it also fits the EntityExtractor seam) --

    def extract(self, chunks: Iterable[ChunkRecord]) -> list[EntityMention]:
        """The NER half alone (for a Pipeline that wants only entities from this model)."""
        return self.extract_fused(chunks).mentions

    def _build_records(
        self,
        chunk: ChunkRecord,
        entities_by_label: dict[str, list[dict[str, Any]]],
        relations_by_type: dict[str, list[dict[str, Any]]],
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
        page_index = PageOffsetIndex(chunk.page_map) if chunk.page_map is not None else None

        mentions: list[EntityMention] = []
        by_span: dict[tuple[int, int], EntityMention] = {}
        for i, ent in enumerate(flat_entities):
            local_start = ent["start"]
            local_end = ent["end"]
            if offset_index is not None:
                doc_start, doc_end = offset_index.resolve_span(local_start, local_end)
            else:
                doc_start, doc_end = chunk_char_start + local_start, chunk_char_start + local_end
            provenance = _mention_provenance(chunk.provenance, doc_start, doc_end, page_index)
            mention = EntityMention(
                record_id=f"{chunk.record_id}#entity-{i}",
                document_record_id=chunk.document_record_id,
                chunk_record_id=chunk.record_id,
                index=i,
                label=str(ent["label"]),
                text=chunk.content[local_start:local_end],
                confidence=min(max(float(ent["confidence"]), 0.0), 1.0),
                provenance=provenance,
                classification=chunk.classification,
            )
            mentions.append(mention)
            # First mention wins a given span (deterministic), matching relex's by_span semantics.
            by_span.setdefault((local_start, local_end), mention)

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
            rel_prov = _relation_provenance(head, tail, span_start, span_end, page_index)
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
    chunks: Iterable[ChunkRecord], batch_size: int, min_chars: int
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
    page_index: PageOffsetIndex | None,
) -> Provenance:
    update: dict[str, int] = {"char_start": doc_char_start, "char_end": doc_char_end}
    if page_index is not None:
        update["page_start"] = page_index.page_for_offset(doc_char_start)
        update["page_end"] = page_index.page_for_offset(max(doc_char_end - 1, doc_char_start))
    return chunk_prov.model_copy(update=update)


def _relation_provenance(
    head: EntityMention,
    tail: EntityMention,
    span_start: int,
    span_end: int,
    page_index: PageOffsetIndex | None,
) -> Provenance:
    base = head.provenance
    update: dict[str, int] = {"char_start": span_start, "char_end": span_end}
    if page_index is not None:
        update["page_start"] = page_index.page_for_offset(span_start)
        update["page_end"] = page_index.page_for_offset(max(span_end - 1, span_start))
    else:
        hp, tp = head.provenance, tail.provenance
        starts = [p for p in (hp.page_start, tp.page_start) if p is not None]
        ends = [p for p in (hp.page_end, tp.page_end) if p is not None]
        if starts:
            update["page_start"] = min(starts)
        if ends:
            update["page_end"] = max(ends)
    return base.model_copy(update=update)
