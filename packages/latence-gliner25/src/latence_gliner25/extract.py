"""Fused Entity + Relation Extraction over a **GLiNER 2.5** checkpoint (ADR-0064, ADR-0013/0023).

``fused_entity_relation.gliner25``. The same
:class:`~latence_core.capability.FusedEntityRelationExtractor` seam, the same config shape
(``labels`` + ``relation_labels``) and the same mentions+relations+evidence+provenance+offset
contract as ``fused_entity_relation.gliner2`` — so a Pipeline swaps the provider name and nothing
else moves. What changes is the native library major and the model head:

* ``gliner2`` **2.x** (released 2026-08-24), loaded through
  ``gliner2.auto.AutoExtractor.from_pretrained`` — the architecture-dispatching loader. The legacy
  ``GLiNER2`` class is span-only by the library's own docstring and cannot load a 2.5 checkpoint.
* ``fastino/gliner2.5-multi-v1`` (287M, mDeBERTa-v3-base, multilingual) is the **boundary**
  architecture: a sparse candidate decoder with an exact half-open token→character conversion,
  where the 1.x span models scored every span position against every label.
* the checkpoint card states ``max_len=4096``, so the window is no longer the 768 the 1.x
  Providers inherit (:data:`~latence_gliner25._backbone.DEFAULT_MAX_LEN`).

**The 2.0.0 API, read from the wheel (2026-08-25) — not from a changelog.** Both public model
classes mix in ``gliner2.inference.runtime.ExtractorRuntimeMixin``, so the batched surface the 1.x
Providers drive SURVIVED the major bump unchanged in shape:

.. code-block:: python

    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained("fastino/gliner2.5-multi-v1", quantize=True, compile=True)
    ents = model.batch_extract_entities(texts, entity_types, threshold=0.5,
                                        include_spans=True, include_confidence=True, max_len=1280)
    # -> [{"entities": {label: [{"text", "confidence", "start", "end"}, ...], ...}}, ...]
    rels = model.batch_extract_relations(texts, relation_types, threshold=0.5,
                                         include_spans=True, include_confidence=True, max_len=1280)
    # -> [{"relation_extraction": {rel_type: [{"head": {"text","confidence","start","end"},
    #                                          "tail": {...}}, ...], ...}}, ...]

Both shapes are **identical to the 1.3.2 ones — MEASURED, not inferred.** The two architectures
build them in DIFFERENT code (the span path through
``ExtractorRuntimeMixin._extract_entities`` / ``._extract_relations``, the boundary path through
``BoundaryExtractor._decode_entities`` / ``._decode_relations``, with the shared
``ExtractorRuntimeMixin.format_results`` folding either into one envelope), and reading two
implementations cannot establish that they agree — so both were RUN. A cross-major A/B on the pod
(RTX 2000 Ada 16 GB, driver 580.159.04, 2026-08-25) drove gliner2 1.3.2 + ``gliner2-multi-v1``
(span) and gliner2 2.0.0 + ``gliner2.5-multi-v1`` (boundary) in two venvs on the same text with the
same labels under identical transformers 4.57.6 / torch 2.4.1+cu124, and diffed the structural
signature: every field SAME — envelopes, entity row keys, relation row keys, endpoint keys, int
offsets — with only ``head_class`` and ``gliner2_version`` differing. The key sets are pinned in
:mod:`latence_core.conformance.stub` and every fake in the workspace is asserted against them
(ADR-0064). Key ORDER is NOT part of the contract: the diff sorted the keys, so it establishes the
SET only, and nothing here reads these dicts positionally.

**So this Provider's parsing did not have to adapt at all.** Three genuine 2.0.0 deltas, none of
which change the contract:

* ``batch_size`` is the THIRD POSITIONAL parameter of both batched methods (it was in 1.3.2 too);
  every argument here is passed by keyword, so the position is irrelevant and cannot silently
  become the threshold.
* new ``overlap_policy`` kwarg (``None`` = the model's own default) — not passed: a policy the
  Provider does not measure is a knob it should not pick.
* new ``*_long`` variants taking ``chunk_size``/``chunk_overlap`` — deliberately NOT used. This
  framework's Chunk Stage already owns segmentation, with the offset map and page slice that make
  a span resolvable; a second, model-internal chunker would produce offsets nothing downstream can
  ground (ADR-0021/0031). ``chunk.page`` is the long-window answer, not ``batch_extract_*_long``.

``include_spans=True`` is REQUIRED and always passed: without it gliner2 returns entity *strings*
and relation head/tail *text* only, which the contract cannot ground.

**Relations can be structurally absent, and that is said out loud.** The boundary decoder opens
``_decode_relations`` with ``if not enable_relations: return {}``, and ``format_results`` then fills
every requested relation type with an empty list — a checkpoint without an active relation head
answers a relation query with a well-formed, entirely empty result. The load seam logs that loudly
(:func:`~latence_gliner25._backbone.warn_if_relations_disabled`) and this Provider reports its
relation count per batch, so "zero relations" is visible rather than inferred.

Offsets (the recurring coordinate trap, ADR-0031) are handled exactly as the 1.x twin does:
chunk-local char offsets are resolved to TRUE original-markdown offsets through the chunk's
stripped→original ``offset_map`` via :class:`~latence_core.offsetmap.OffsetIndex`, and each page
span is re-resolved through :class:`~latence_core.page_index.PageIndexResolver` reading the chunk's
OWN ``page_slice`` (v19). A relation's covering span runs between two chunks, so it is answered
from the endpoints' own already-exact pages via
:func:`~latence_core.page_index.pages_for_covering_span`.

Licensing (ADR-0012 / research-diligence — weights AND code, separately) is recorded on the
:class:`ProviderProfile` with the date each was checked; see
:data:`~latence_gliner25._backbone.DEFAULT_MODEL` for which of the two was read from a local
artifact and which was carried in.
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
    resolve_gliner2_knob,
    resolve_intra_op_threads,
)
from latence_core.providers.profile import ProviderProfile

from latence_gliner25._backbone import (
    DEFAULT_MODEL,
    load_gliner25_model,
    require_chunk_fits_window,
    resolve_gliner25_max_len,
    resolve_window_safe_microbatch_bounds,
    warn_if_relations_disabled,
)
from latence_gliner25._compat import BATCHED_METHODS

_log = logging.getLogger("latence.provider.gliner25_fused")

if TYPE_CHECKING:
    from collections.abc import Iterable

    from latence_core.contracts import ChunkRecord, Provenance

PROVIDER_NAME = "fused_entity_relation.gliner25"

#: The streaming batch size: the BUCKETING WINDOW — how many chunks are buffered before they are
#: partitioned by label set and each bucket run through one native batched forward. It is NOT the
#: model-forward tensor size: every bucket is further split into token-budget micro-batches, so the
#: padded activation is bounded by ``microbatch_max_texts x max_len`` for ANY configured
#: ``batch_size``. Emission is record-for-record identical for any positive batch size.
_DEFAULT_BATCH_SIZE = 16

#: The hard cap on the streaming batch (progress cadence), so a large corpus always logs
#: incremental progress and can never *appear* hung.
_MAX_BATCH_SIZE = 64

#: Entity and relation labels are RELEVANCE labels, so the config lists are a PER-CHUNK floor: a
#: chunk that induced its own types is scored against THOSE ALONE, and the config lists are the
#: recall net for a chunk induction said nothing about. Same policy, same reasoning, as the 1.x
#: twin — zero-shot precision and the label dimension of the scoring activation both degrade with
#: label count. ``label_floor: universal`` restores the union.
_DEFAULT_LABEL_FLOOR = LabelFloor.PER_CHUNK


@dataclass(frozen=True)
class _PendingChunk:
    """A chunk with its two resolved label sets — the bucket keys for the batched forward.

    Tuples, not lists: they must be hashable to bucket, and label ORDER is part of the question
    asked of the model, so two chunks share a bucket only when they would have been asked the
    identical question.
    """

    chunk: ChunkRecord
    entity_labels: tuple[str, ...]
    relation_labels: tuple[str, ...]


@dataclass(frozen=True)
class _Slot:
    """One of the two extraction slots — the ONE place the entity/relation asymmetry lives.

    Bucketing, micro-batching, the compile latch, the OOM backoff and the scatter-back are
    identical for entities and relations; only the native method, the envelope key wrapping each
    per-text result, and which of a chunk's two label sets keys the bucket differ.
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

    The REAL gliner2 2.0.0 shapes are nested dicts — ``{"entities": {label: [rows]}}`` and
    ``{"relation_extraction": {rel_type: [rows]}}`` — NOT a flat list. A Provider that mis-parses
    either as a flat list recovers ZERO spans silently, which is exactly the false green the
    offline fake and its anchor test exist to close. Anything that is not the pinned shape is a
    typed :class:`ProviderError` (ADR-0034), never a quiet empty result.
    """
    payload: Any = raw.get(slot.envelope) if isinstance(raw, dict) else raw
    if payload is None and isinstance(raw, dict):
        # ``format_results`` omits the ``relation_extraction`` key entirely when NO relation type
        # was requested. An empty dict is the honest reading of that; a missing ENVELOPE on a slot
        # that WAS asked something is caught by the type check below.
        payload = {}
    if not isinstance(payload, dict):
        got = type(payload).__name__
        msg = (
            f"{PROVIDER_NAME}: unexpected {slot.method} output shape "
            f"(expected {{{slot.envelope!r}: {{label: [rows]}}}}, got {got})"
        )
        raise ProviderError(msg)
    return cast("dict[str, list[dict[str, Any]]]", payload)


class Gliner25FusedExtractor:
    """Fused zero-shot Entity+Relation Extractor backed by a GLiNER 2.5 checkpoint (ADR-0064).

    A drop-in for :class:`latence_extract_gliner2.provider.Gliner2FusedExtractor`: the SAME
    Capability seam, the SAME config keys, the SAME emitted contract. It cannot be installed
    beside it in one environment — the two packages pin disjoint ``gliner2`` majors on purpose,
    and ``deploy/env/`` enforces that offline (ADR-0064).

    Config:
        labels: list[str]  — REQUIRED. The zero-shot entity types (e.g.
            ``["organization", "person"]``). GLiNER-family models work best with lower/title case.
        relation_labels: list[str]  — REQUIRED. The zero-shot relation types between the entities.
        label_floor: str  — ``per_chunk`` (default) or ``universal``: what the config lists MEAN
            for a chunk that carries induced types. ``per_chunk`` treats them as a RELEVANCE floor;
            ``universal`` restores the config ∪ induced union on every chunk. A chunk with no
            induced labels uses exactly the config lists under either value.
        model: str  — the checkpoint id (default ``fastino/gliner2.5-multi-v1``, Apache-2.0). It
            must be a **boundary**-architecture checkpoint; a 1.x ``span`` checkpoint is refused at
            load with a typed error naming the Provider that does load it.
        threshold: float  — score threshold; entities/relations below it are dropped (default 0.5).
        min_chars: int  — a chunk whose stripped length is below this is skipped without invoking
            the model (default 1).
        max_len: int  — the model token window (default 1280, capped at the checkpoint's 4096).
            Threaded as the native ``max_len``; past it the model truncates SILENTLY, so this
            bounds the window explicitly. The chunker counts in a DIFFERENT tokenizer (~1.2x fewer
            tokens than mdeberta), which is why 1280 pairs with a 1024-token chunk budget.
        batch_size: int  — the **bucketing window** and progress cadence (default 16; clamped to
            64; must be ``>= 1``). NOT the model forward tensor. Emission is record-for-record
            identical for any positive batch size and any bucketing.
        microbatch_max_texts: int  — the hard cap on texts packed into ONE native forward.
            Defaulted so that ``max_texts x max_len`` stays inside the same padded-position
            envelope the 768-window default implies — raising ``max_len`` therefore does NOT widen
            the forward (see
            :func:`~latence_gliner25._backbone.resolve_window_safe_microbatch_bounds`).
        microbatch_token_budget: int  — the summed-token budget for ONE forward; defaulted to
            ``max_texts x max_len`` for the same reason.
        intra_op_threads: int  — cap on torch intra-op threads applied at load (default 8;
            ``<= 0`` disables). Guarded so a torch-free/failed cap is a logged no-op.
        quantize: bool | 'auto'  — the gliner2 native ``quantize`` (fp16) load knob (default
            ``auto`` → True on cuda / False on cpu). Guarded → plain load on any fault.
        compile: bool | 'auto'  — the gliner2 native ``compile`` load knob, ``auto`` → True on cuda
            / False on cpu. Guarded twice: a compile fault at LOAD degrades to a plain load, and
            the lazy ``BackendCompilerFailed`` on the first forward trips the reload-uncompiled
            latch — so a backend that cannot compile costs throughput, never correctness.
        device: str  — ``auto`` (default) / ``cpu`` / ``cuda``, routed by the ADR-0036 device seam.

    Fulfils BOTH the EntityExtractor and the FusedEntityRelationExtractor seam: it also exposes
    ``extract`` so it can stand in for a plain Entity-Extraction Stage. The model is loaded lazily
    and cached on first use, so constructing the Provider never downloads weights.
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default gliner2.5-multi-v1 checkpoint.

        ``compute="either"``: the mDeBERTa-v3 backbone runs in-process on a laptop CPU (ADR-0007)
        and accelerates on a GPU when present, so the device seam routes it rather than skipping it
        on a CPU-only host.

        License **verified** per ADR-0012 / research-diligence, weights AND code SEPARATELY, each
        with the date it was checked — see :data:`~latence_gliner25._backbone.DEFAULT_MODEL` for
        exactly which artifact each fact was read from.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical across
        hardware/library versions, so the profile declares it rather than overclaim. ``memory_mb``
        is the approximate resident footprint (declared, not measured).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=2048,
            model_id=DEFAULT_MODEL,
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=(
                "https://huggingface.co/fastino/gliner2.5-multi-v1 (weights, Apache-2.0, not "
                "gated — model card, checked 2026-08-25); "
                "https://github.com/fastino-ai/GLiNER2 (library, Apache-2.0 — the gliner2 2.0.0 "
                "wheel METADATA carries 'License-Expression: Apache-2.0' and bundles LICENSE, "
                "read 2026-08-25)"
            ),
            license_verified_on="2026-08-25",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note=(
                "gliner2 2.x boundary architecture over a 4096-token checkpoint window; this "
                "Provider defaults max_len to 1280 (the window covering a 1024-token chunk "
                "budget) and derives microbatch_max_texts so max_texts x max_len does NOT exceed "
                "the padded-position envelope the 768-window default implies. Native "
                "quantize/compile ride the shared cuda defaults, guarded at load and by the "
                "inference-time reload-uncompiled latch; TorchScript JIT neutralized and intra-op "
                "threads capped by the shared load seam (#190). NO throughput number is claimed "
                "for the 2.5 family: nothing has been measured on hardware yet (docs/BENCHMARKS.md)"
            ),
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        labels = config.get("labels")
        if not labels or not isinstance(labels, list):
            msg = (
                "Gliner25FusedExtractor requires a non-empty 'labels' list in its config "
                "(zero-shot NER: the entity label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        relation_labels = config.get("relation_labels")
        if not relation_labels or not isinstance(relation_labels, list):
            msg = (
                "Gliner25FusedExtractor requires a non-empty 'relation_labels' list in its "
                "config (zero-shot RE: the relation label set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in labels]
        self._relation_labels = [str(label) for label in relation_labels]
        self._model_id = str(config.get("model", DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        try:
            self._label_floor = resolve_label_floor(config, default=_DEFAULT_LABEL_FLOOR)
        except ValueError as exc:
            msg = f"{PROVIDER_NAME}: {exc}"
            raise ConfigError(msg) from exc
        requested_batch = validate_batch_size(
            config.get("batch_size", _DEFAULT_BATCH_SIZE), provider_name=PROVIDER_NAME
        )
        self._batch_size = min(requested_batch, _MAX_BATCH_SIZE)
        load_config: dict[str, Any] = dict(config)
        # Resolve the device ONCE through the ADR-0036 seam (#191): `device: cpu` on a GPU box must
        # be honoured, and an invalid device string must be an operator ConfigError before any
        # weights are downloaded.
        self._device = self._resolve_device(config.get("device", "auto"))
        # Every perf knob is validated EAGERLY (fail-fast at construction, like batch_size) so an
        # invalid value is an operator ConfigError up front, not a native fault buried in the lazy
        # load. `resolve_gliner25_max_len` additionally raises its own typed ConfigError above the
        # checkpoint's 4096 ceiling, so it is called outside this ValueError translation.
        self._max_len = resolve_gliner25_max_len(
            load_config,
            fail_open=False,
            provider_name=PROVIDER_NAME,
            model_id=self._model_id,
        )
        try:
            resolve_gliner2_knob("quantize", self._device.device, load_config)
            resolve_gliner2_knob("compile", self._device.device, load_config)
            resolve_intra_op_threads(load_config)
            self._microbatch_max_texts, self._microbatch_token_budget = (
                resolve_window_safe_microbatch_bounds(load_config, max_len=self._max_len)
            )
        except ValueError as exc:
            msg = f"{PROVIDER_NAME}: {exc}"
            raise ConfigError(msg) from exc
        self._config: dict[str, Any] = load_config
        self._model: Any = None
        self._compile_disabled: bool = False
        #: the per-batch cumulative progress trail (chunks, mentions, relations), filled as
        #: ``extract_fused`` streams batches — empty until the first extract, readable for
        #: forensics and for seeing a structurally-dead relation head as it happens.
        self._progress: list[tuple[int, int, int]] = []

    # -- lazy model load ----------------------------------------------------

    @classmethod
    def _resolve_device(cls, requested: object) -> DeviceDecision:
        """Route the ``device`` config value (``auto|cpu|cuda``) through the ADR-0036 device seam.

        ``compute="either"``, so ``cpu`` pins CPU, ``auto`` follows availability, and ``cuda``
        resolves to CUDA when a GPU is present — else the seam returns a **skip** decision, which
        the load choreography turns into a loud ConfigError rather than a silent CPU fallback
        (#191). A malformed device string is a ConfigError, not a TypeError.
        """
        try:
            return select_device(str(requested), cls.profile())
        except ValueError as exc:
            msg = f"{PROVIDER_NAME}: invalid device config: {exc}"
            raise ConfigError(msg) from exc

    def _load_model(self) -> Any:
        """Load the checkpoint on first use (cached) via the shared load choreography."""
        if self._model is not None:
            return self._model
        self._model = load_gliner25_model(
            self._model_id,
            self._config,
            self._device,
            provider_name=PROVIDER_NAME,
            methods=BATCHED_METHODS,
        )
        warn_if_relations_disabled(self._model, provider_name=PROVIDER_NAME)
        return self._model

    def _reload_uncompiled(self) -> Any:
        """Re-load with ``compile=False`` (quantize kept) after an inference-time compile fault.

        ``torch.compile`` compiles lazily on the first forward, so a backend that cannot compile
        fails INSIDE the batched call, which the load-time guard cannot catch. On the first such
        fault the Provider re-loads ONCE with ``compile`` forced off and uses the uncompiled model
        for the rest of the run — a bad compile costs only throughput. Idempotent.
        """
        self._compile_disabled = True
        _log.warning(
            "%s: torch.compile failed at inference; re-loading the uncompiled model "
            "(quantize kept) — throughput only, never a crash",
            PROVIDER_NAME,
        )
        # The SAME choreography as the first load — placement included, which is the point:
        # skipping it would leave the model that serves the REST of the run unplaced.
        self._model = load_gliner25_model(
            self._model_id,
            {**self._config, "compile": False},
            self._device,
            provider_name=PROVIDER_NAME,
            methods=BATCHED_METHODS,
        )
        return self._model

    # -- fused extraction ---------------------------------------------------

    def extract_fused(self, chunks: Iterable[ChunkRecord]) -> FusedExtraction:
        """Extract entities + relations per chunk; yield mentions + relations together (S4/S5).

        Chunks below ``min_chars`` are skipped without a model call. Every remaining chunk resolves
        its OWN effective (entity, relation) label sets through the shared label seam under the
        floor policy. Within each ``batch_size`` window the chunks are partitioned by label set —
        separately per slot, because a chunk can share the entity set and not the relation set —
        and each bucket runs through ONE native batched forward per token-budget micro-batch.
        Per-text results scatter back BY INDEX, so bucketing can never permute a chunk's result
        onto another chunk and emission stays in input-chunk order.

        After each window the Provider logs cumulative progress (chunks / mentions / relations), so
        a large corpus emits a steady trail instead of running silently — and so a checkpoint whose
        relation head is inactive shows as a relation count pinned at zero rather than as a
        conclusion nobody drew.
        """
        loaded = False
        mentions: list[EntityMention] = []
        relations: list[RelationMention] = []
        processed = 0
        self._progress = []
        # ONE resolver for the whole call, so the drift roll-up is per call rather than thrown
        # away per chunk. It does NOT need to observe the chunks ``min_chars`` drops: since v19
        # each chunk carries its own ``page_slice``, so the resolver answers from the record in
        # hand, in any order, whichever siblings a Stage removed.
        resolver = PageIndexResolver()

        for batch_count, batch in enumerate(
            _iter_batches(chunks, self._batch_size, self._min_chars), start=1
        ):
            if not loaded:
                self._load_model()
                loaded = True
            for chunk in batch:
                # Before anything is asked of the model: a chunk longer than the window is
                # truncated SILENTLY, and the tail's entities are then missed by nothing that
                # reports it. Refused rather than trimmed — segmentation is the Chunk Stage's
                # (ADR-0021), and a Capability that cannot be served is a loud error.
                require_chunk_fits_window(
                    token_count=chunk.token_count,
                    record_id=chunk.record_id,
                    max_len=self._max_len,
                    provider_name=PROVIDER_NAME,
                )
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
                "%s: extract progress — batch %d (%d chunks done, %d mentions, %d relations)",
                PROVIDER_NAME,
                batch_count,
                processed,
                len(mentions),
                len(relations),
            )
        return FusedExtraction(mentions=mentions, relations=relations)

    def extract(self, chunks: Iterable[ChunkRecord]) -> list[EntityMention]:
        """The NER half alone (for a Pipeline that wants only entities from this model)."""
        return self.extract_fused(chunks).mentions

    def _prepare(self, chunk: ChunkRecord) -> _PendingChunk:
        """Resolve a chunk's two effective label sets BEFORE any model call (the bucket keys)."""
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
        """Run one slot over ``pending`` as ONE forward per label bucket, per micro-batch.

        Returns the per-chunk raw output ALIGNED with ``pending`` (index i ↔ pending[i]) — results
        are scattered back by index, never by output order.
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
                        f"{PROVIDER_NAME}: {slot.method} returned {len(raw)} results for a "
                        f"micro-batch of {end - start} texts."
                    )
                    raise ProviderError(msg)
                for offset, item in enumerate(raw):
                    results[indices[start + offset]] = _unwrap(slot, item)
        # Every index belongs to exactly one bucket and every bucket is tiled by micro-batches, so
        # no slot can stay unfilled; the assertion-free narrowing keeps mypy --strict honest.
        return [{} if item is None else item for item in results]

    def _batch_forward(self, slot: _Slot, texts: list[str], labels: list[str]) -> list[Any]:
        """One native batched forward, with the compile latch AND the CUDA-OOM halving backoff.

        A ``BackendCompilerFailed`` from the lazy ``torch.compile`` re-loads the uncompiled model
        ONCE and retries. A CUDA OOM splits the micro-batch in half and retries each half after a
        best-effort cache release — the output is batch-invariant, so that is byte-identical; a
        batch of ONE that still OOMs is genuinely unrecoverable and surfaces as a typed error (no
        dropped chunk, no fake green). Any other native fault is already a typed ProviderError
        carrying only the op + native type name (no chunk text / PII — C2/C6).
        """
        try:
            return self._native_batch(self._model, slot, texts, labels)
        except Exception as exc:  # noqa: BLE001 - intercept recoverable faults before surfacing
            if not self._compile_disabled and is_compile_backend_failure(exc):
                return self._native_batch(self._reload_uncompiled(), slot, texts, labels)
            if is_cuda_oom(exc) and len(texts) > 1:
                empty_cuda_cache(provider_name=PROVIDER_NAME)
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

        Every argument after ``texts``/``types`` is passed BY KEYWORD. That is not style: in
        gliner2 2.0.0 the third positional parameter of both batched methods is ``batch_size``, and
        a positional ``threshold`` would silently become the model's internal sub-batch size. The
        forward runs inside ``torch.inference_mode()`` — an extraction-only encoder never
        backpropagates, so building an autograd graph would only accumulate VRAM run over run.
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
                f"{PROVIDER_NAME}: provider failure in {slot.method!r} "
                f"(native {type(exc).__name__})"
            )
            raise ProviderError(msg) from exc

    def _build_records(
        self,
        chunk: ChunkRecord,
        entities_by_label: dict[str, list[dict[str, Any]]],
        relations_by_type: dict[str, list[dict[str, Any]]],
        resolver: PageIndexResolver,
    ) -> tuple[list[EntityMention], list[RelationMention]]:
        """Map the gliner2 per-label / per-relation dicts → the framework mention/relation set.

        Entities: flatten ``{label: [{"start","end","text","confidence"}]}`` → one mention per row,
        offsets resolved through the chunk's stripped→original ``offset_map`` (ADR-0031), keyed by
        chunk-local ``(start, end)`` so a relation endpoint referencing those offsets resolves to
        the mention. Relations: flatten ``{rel_type: [{"head": {...}, "tail": {...}}]}`` → one
        relation per row, resolving head/tail by their chunk-local ``(start, end)``; a relation
        whose endpoint is not among the emitted mentions is dropped, so no un-groundable record
        leaks. Deterministic order: entities by ``(start, label)``, relations by
        ``(head start, head end, rel_type)``.
        """
        flat_entities: list[dict[str, Any]] = []
        for label, rows in entities_by_label.items():
            label_str = str(label)
            if not label_str or not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict) or "start" not in row or "end" not in row:
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

        mentions: list[EntityMention] = []
        by_span: dict[tuple[int, int], EntityMention] = {}
        skipped_degenerate = 0
        for i, ent in enumerate(flat_entities):
            local_start = ent["start"]
            local_end = ent["end"]
            # A degenerate span whose resolved surface text is empty carries no entity, and an
            # EntityMention with text="" violates its >=1-char contract. Skip it — but NOT
            # silently (ADR-0034): the count is logged below.
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
            by_span.setdefault((local_start, local_end), mention)

        if skipped_degenerate:
            _log.info(
                "%s: skipped %d degenerate zero-length entity span(s) in chunk %s (empty surface "
                "text — cannot ground a mention)",
                PROVIDER_NAME,
                skipped_degenerate,
                chunk.record_id,
            )

        flat_relations: list[dict[str, Any]] = []
        for rel_type, rows in relations_by_type.items():
            rel_type_str = str(rel_type)
            if not rel_type_str or not isinstance(rows, list):
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
                # A covering span runs between two endpoints and is not a sub-chunk offset, so it
                # is answered from the endpoints' own already-exact pages, not by a lookup.
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
    chunks: Iterable[ChunkRecord], batch_size: int, min_chars: int
) -> Iterable[list[ChunkRecord]]:
    """Group the (lazy) chunk stream into bounded batches of processable chunks.

    Chunks whose stripped content is below ``min_chars`` are dropped here (skipped without a model
    call), so a batch is ``batch_size`` chunks that will each be extracted. Streaming: it consumes
    the input iterable lazily and never materialises more than one batch at a time (ADR-0033).
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

    Per the REAL gliner2 2.0.0 shape, ``include_confidence=True`` carries the score on the
    head/tail sub-dicts, not at the relation level — in the boundary decoder both endpoints carry
    the SAME pair score, and in the span decoder they carry their own — so the weakest-link min is
    correct for both and falls back to 1.0 when neither endpoint carries one. A top-level
    ``confidence`` is still honoured if a future version adds one.
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
    """``pages`` is what the :class:`PageIndexResolver` resolved for the mention's own span;
    ``None`` (no page map available) degrades to the chunk's inherited range."""
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
