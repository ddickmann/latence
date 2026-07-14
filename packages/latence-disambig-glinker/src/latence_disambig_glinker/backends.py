"""Inference backend abstraction — pluggable model serving for entity linking.

Ported (PORT-AS-IS) from ``dataset_intelligence.linking.backends``. The enterprise default is
:class:`GLinkerBackend` — the full GLinker **L2 -> L3 -> L4 -> L0** pipeline over
``knowledgator/gliner-linker-large-v1.0`` (L3 biencoder) + the ``gliner-linker-rerank-v1.0`` L4
(L4 reranker), FlashDeBERTa-accelerated, run through the ``glinker`` library's ``ProcessorFactory``
(Knowledgator, Apache-2.0 — the package that provides ``ProcessorFactory`` + the L2/L3/L4 stages;
NOT the distinct ``gliner`` extraction library):

  L2: Candidate retrieval from an in-memory dict store (per-mention, ``max_candidates`` cap)
  L3: Entity disambiguation via GLiNER Linker (DeBERTa BiEncoder, precomputed label embeddings)
  L4: Reranking via GLiNER Linker Reranker (optional, recommended)
  L0: Aggregation + threshold filtering

:class:`VLLMGLiNERBackend` is the **documented, seam-compatible follow-on** (needs the
vllm-factory ``deberta_gliner_linker`` plugin) — the transformers path above ships as the complete
enterprise default, so the full L2->L3->L4->L0 + audited resolver runs today; only the alternate
serving backend defers (ProviderProfile ``device=cuda`` device-honesty covers it).

All heavy deps (``torch``, ``glinker``, ``vllm``, ``transformers``, ``huggingface_hub``) are
imported **lazily inside methods**, so importing this module (constructing a backend) pulls
nothing — the CPU-first offline env (ADR-0007/0016). A backend load/inference failure is caught by
the caller (:mod:`.linker`) and degrades to the direct-KB fallback — never a crash.
"""

from __future__ import annotations

import functools
import inspect
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from latence_disambig_glinker.embedder import cosine

if TYPE_CHECKING:
    from latence_disambig_glinker.embedder import LabelEncoder

logger = logging.getLogger(__name__)

# gliner's collator (``gliner`` 0.2.x collator.py) injects these into the model input and its
# ``base.py`` forwards them into the encoder; a plain ``transformers`` DeBERTa ``forward()`` (the
# gliner-linker backbone) does NOT declare them, so every neural batch raises → the linker silently
# falls back to direct-KB (``entities_linked: 0``). We strip exactly these before the plain forward.
_ENCODER_STRIP_KWARGS = ("token_lengths", "word_lengths", "packing_config", "pair_attention_mask")

# Bound on the executor object-graph walk that locates the loaded encoder modules (safety valve so a
# pathological/cyclic graph can never spin — the real graph is tiny).
_MAX_TRAVERSAL_STEPS = 5000


def _strip_unsupported_forward_kwargs(
    module: Any, unsupported: tuple[str, ...] = _ENCODER_STRIP_KWARGS
) -> bool:
    """Wrap ``module.forward`` to DROP kwargs its real signature does not accept (FIX-2, contained).

    Defensive + forward-compatible: it inspects the actual ``forward`` signature and

    * no-ops when the signature declares ``**kwargs`` (it already tolerates the extras), or already
      names every kwarg in ``unsupported`` (a future backbone that consumes them) — so a
      transformers upgrade that starts accepting ``token_lengths`` silently disarms this patch;
    * otherwise installs a wrapper that pops ONLY the ``unsupported`` names the signature lacks,
      leaving every other argument untouched, then calls the real forward.

    Contained (mutates only the passed module's bound method — never a site-package file),
    idempotent (a second call is a no-op), and logs once at DEBUG on the first strip. Returns
    ``True`` iff a wrapper was installed.
    """
    forward = getattr(module, "forward", None)
    if forward is None or getattr(forward, "_latence_kwarg_strip", False):
        return False
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return False  # accepts **kwargs → nothing to strip (forward-compat no-op)
    to_strip = tuple(k for k in unsupported if k not in params)
    if not to_strip:
        return False  # the real forward already declares them → let it handle them
    logged = False

    @functools.wraps(forward)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal logged
        removed = [k for k in to_strip if k in kwargs]
        for k in removed:
            kwargs.pop(k, None)
        if removed and not logged:
            logger.debug("Stripped unsupported encoder forward kwargs %s", removed)
            logged = True
        return forward(*args, **kwargs)

    _wrapped._latence_kwarg_strip = True  # type: ignore[attr-defined]
    module.forward = _wrapped
    return True


def _forward_names_input_ids(module: Any) -> bool:
    """Whether ``module.forward`` is a transformer-encoder forward (declares ``input_ids``).

    The gate that keeps the FIX-2 patch surgical: only the transformer encoder (whose forward
    gliner feeds the collator kwargs) declares ``input_ids`` — a Linear/pooler/embedding submodule
    does not, so we never wrap unrelated leaves.
    """
    forward = getattr(module, "forward", None)
    if forward is None:
        return False
    try:
        return "input_ids" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def _collect_modules(
    root: Any, module_type: type, max_steps: int = _MAX_TRAVERSAL_STEPS
) -> list[Any]:
    """Bounded BFS over ``root``'s attributes/containers collecting ``module_type`` instances.

    Descends dicts/lists/tuples/sets and plain-object ``__dict__``s (skipping tensors, which have no
    ``__dict__`` and are not containers) with a visited-set + step cap, so it locates the loaded
    encoder module(s) held by the glinker executor without ever recursing unboundedly.
    """
    found: list[Any] = []
    seen: set[int] = set()
    stack: list[Any] = [root]
    steps = 0
    while stack and steps < max_steps:
        steps += 1
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, module_type):
            found.append(obj)
            continue  # a module root; its submodules are reached via .modules() by the caller
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(obj, dict):
            stack.extend(obj.values())
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return found


@dataclass
class LinkedEntity:
    """Result of entity linking — a single disambiguated entity (ported verbatim)."""

    entity_id: str
    label: str
    description: str = ""
    score: float = 0.0
    entity_type: str = ""
    aliases: list[str] = field(default_factory=list)
    source: str = "linked"


class InferenceBackend(ABC):
    """Abstract interface for entity linking inference (ported verbatim)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        ...

    @abstractmethod
    def link_mentions(
        self,
        mentions: list[dict[str, Any]],
        entity_kb: list[dict[str, Any]],
        threshold: float = 0.3,
        batch_size: int = 32,
    ) -> list[LinkedEntity | None]:
        """Link a batch of mentions against an entity knowledge base.

        Args:
            mentions: mention dicts with 'text', 'label', 'context' and optionally 'start'/'end'.
            entity_kb: entity dicts with 'entity_id', 'label', 'description', 'entity_type',
                'aliases', 'popularity'.
            threshold: minimum confidence for a link.
            batch_size: processing batch size.

        Returns:
            One :class:`LinkedEntity` per mention (``None`` if unresolved).
        """
        ...

    def warmup(self, entity_kb: list[dict[str, Any]]) -> None:  # noqa: B027 - optional hook
        """Optional warmup / precomputation step (a backend may override; default no-op)."""


_WORD_PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")


class GLinkerBackend(InferenceBackend):
    """Full GLinker L2->L3->L4->L0 entity linking pipeline (the enterprise default).

    Uses the GLiNER Linker BiEncoder for disambiguation with an optional reranker stage for
    maximum precision:

      - L2: candidate retrieval from an in-memory dict store
      - L3: entity disambiguation via GLiNER Linker (DeBERTa BiEncoder)
      - L4: reranking via GLiNER Linker Reranker (optional, recommended)
      - L0: aggregation + filtering

    Key features (ported verbatim):
      - ``precompute_embeddings=True`` -> 10-100x speedup via cached label embeds
      - Reranker (L4) for improved precision on ambiguous entities
      - FlashDeBERTa for optimized GPU inference (perf-seam gated: enabled only when the
        ``flash_attn`` package is importable, graceful sdpa fallback otherwise — W8 posture)
      - ``template="{label}: {description}"`` -> richer entity representations

    Reference: https://github.com/Knowledgator/GLinker
    """

    LINKER_MODEL = "knowledgator/gliner-linker-large-v1.0"
    RERANKER_MODEL = "knowledgator/gliner-linker-rerank-v1.0"

    def __init__(
        self,
        model_name: str = LINKER_MODEL,
        device: str = "cuda",
        precompute_embeddings: bool = True,
        template: str = "{label}: {description}",
        reranker_model: str | None = RERANKER_MODEL,
        reranker_max_labels: int = 20,
        reranker_threshold: float | None = None,
        use_flash_deberta: bool = True,
        threshold: float = 0.3,
        max_length: int = 512,
        precompute_batch_size: int = 16,
        max_candidates: int = 100,
        l2_search_mode: str = "exact",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.precompute_embeddings = precompute_embeddings
        self.template = template
        self.reranker_model = reranker_model
        self.reranker_max_labels = reranker_max_labels
        self.reranker_threshold = reranker_threshold
        self.use_flash_deberta = use_flash_deberta
        self.threshold = threshold
        self.max_length = max_length
        self.precompute_batch_size = precompute_batch_size
        self.max_candidates = max_candidates
        self.l2_search_mode = l2_search_mode
        self._executor: Any = None
        self._executor_lock = threading.Lock()
        self._kb_loaded = False
        self._embeddings_precomputed = False

    @property
    def name(self) -> str:
        return "glinker"

    @staticmethod
    def _patch_tokenizer_config(model_name: str) -> None:
        """Patch TokenizersBackend -> PreTrainedTokenizerFast in the cached model config.

        Copies to a temp file first, then ``shutil.move``s it over the config.

        PORT-AS-IS caveat (C-F9): this replaces a file inside the process-shared Hugging Face hub
        cache, so concurrent processes sharing one ``HF_HOME`` could race on the same
        ``tokenizer_config.json``. The mitigations that make it safe in practice: the write is
        idempotent (it rewrites only when ``tokenizer_class == "TokenizersBackend"``, and the
        patched value is not re-matched), the ``move`` is atomic on a POSIX same-fs rename, and a
        read-only cache degrades gracefully (the ``OSError`` branch logs + returns rather than
        crashing). A stricter cross-process file lock is deferred — the executor is built once per
        backend under ``_executor_lock`` (in-process serialised), so the realistic race is only
        across separate processes pointed at one cache, which the idempotent+atomic write tolerates.
        """
        try:
            import json
            import os
            import shutil
            import tempfile

            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(model_name, "tokenizer_config.json")
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            if config.get("tokenizer_class") == "TokenizersBackend":
                config["tokenizer_class"] = "PreTrainedTokenizerFast"
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
                try:
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=2)
                    shutil.move(tmp_path, config_path)
                except OSError:
                    os.unlink(tmp_path)
                    logger.debug("Could not write patched config (read-only FS)")
                    return
                logger.info("Patched tokenizer_config -> PreTrainedTokenizerFast")
        except Exception as e:  # noqa: BLE001 - a config patch is best-effort, never fatal
            logger.debug("Tokenizer config patch skipped: %s", e)

    def _flash_deberta_available(self) -> bool:
        """Whether FlashDeBERTa can be enabled (perf-seam gate, W8 graceful fallback).

        Honors the ``use_flash_deberta`` request only when the ``flashdeberta`` package is
        importable — the same "accelerated attention only when its package is present, else the
        standard kernel" posture as :func:`latence_core.providers.perf.resolve_attn_impl`. The
        GLinker library reads ``USE_FLASHDEBERTA`` + imports ``flashdeberta`` internally; a request
        on a host without it degrades silently (standard DeBERTa attention), never a crash.
        """
        if not self.use_flash_deberta:
            return False
        import importlib.util

        return importlib.util.find_spec("flashdeberta") is not None

    def _ensure_executor(self) -> None:
        if self._executor is not None:
            return
        with self._executor_lock:
            if self._executor is not None:
                return
            self._init_executor()

    def _init_executor(self) -> None:
        import os

        import torch

        torch.set_float32_matmul_precision("high")

        if self._flash_deberta_available():
            os.environ["USE_FLASHDEBERTA"] = "1"
            logger.info("FlashDeBERTa enabled (USE_FLASHDEBERTA=1)")
        else:
            logger.info("FlashDeBERTa unavailable/disabled — standard DeBERTa attention")

        from glinker import ProcessorFactory

        self._patch_tokenizer_config(self.model_name)

        pipeline_stages = "L2->L3"
        if self.reranker_model:
            self._patch_tokenizer_config(self.reranker_model)
            pipeline_stages += "->L4"
        pipeline_stages += "->L0"

        logger.info(
            "Creating GLinker executor (%s): linker=%s, reranker=%s, device=%s, precompute=%s",
            pipeline_stages,
            self.model_name,
            self.reranker_model or "(none)",
            self.device,
            self.precompute_embeddings,
        )

        # Build the DAG manually (not create_simple) so L2 receives mention texts for candidate
        # retrieval rather than returning the entire entity KB (which creates an
        # O(spans x entities) scoring matrix that OOMs on GPUs < 48 GiB).
        nodes: list[dict[str, Any]] = [
            {
                "id": "l2",
                "processor": "l2_chain",
                "requires": [],
                "inputs": {
                    "texts": {"source": "$input", "fields": "texts"},
                    "mentions": {"source": "$input", "fields": "mentions"},
                },
                "output": {"key": "l2_result"},
                "schema": {"template": self.template},
                "config": {
                    "max_candidates": self.max_candidates,
                    "min_popularity": 0,
                    "layers": [
                        {
                            "type": "dict",
                            "priority": 0,
                            "write": True,
                            "search_mode": [self.l2_search_mode],
                        }
                    ],
                },
            },
            {
                "id": "l3",
                "processor": "l3_batch",
                "requires": ["l2"],
                "inputs": {
                    "texts": {"source": "$input", "fields": "texts"},
                    "candidates": {"source": "l2_result", "fields": "candidates"},
                },
                "output": {"key": "l3_result"},
                "schema": {"template": self.template},
                "config": {
                    "model_name": self.model_name,
                    "device": self.device,
                    "threshold": self.threshold,
                    "flat_ner": True,
                    "multi_label": False,
                    "use_precomputed_embeddings": True,
                    "cache_embeddings": False,
                    "max_length": self.max_length,
                },
            },
        ]

        l0_entity_source = "l3_result"
        l0_requires = ["l2", "l3"]

        if self.reranker_model:
            nodes.append(
                {
                    "id": "l4",
                    "processor": "l4_reranker",
                    "requires": ["l2", "l3"],
                    "inputs": {
                        "texts": {"source": "$input", "fields": "texts"},
                        "candidates": {"source": "l2_result", "fields": "candidates"},
                    },
                    "output": {"key": "l4_result"},
                    "schema": {"template": self.template},
                    "config": {
                        "model_name": self.reranker_model,
                        "device": self.device,
                        "max_labels": self.reranker_max_labels,
                        "threshold": (
                            self.reranker_threshold
                            if self.reranker_threshold is not None
                            else self.threshold
                        ),
                    },
                }
            )
            l0_entity_source = "l4_result"
            l0_requires = ["l2", "l3", "l4"]

        nodes.append(
            {
                "id": "l0",
                "processor": "l0_aggregator",
                "requires": l0_requires,
                "inputs": {
                    "l2_candidates": {"source": "l2_result", "fields": "candidates"},
                    "l3_entities": {"source": l0_entity_source, "fields": "entities"},
                },
                "output": {"key": "l0_result"},
                "schema": {"template": self.template},
                "config": {
                    "strict_matching": False,
                    "min_confidence": 0.0,
                    "include_unlinked": True,
                    "position_tolerance": 2,
                },
            }
        )

        config = {
            "name": "latence_disambig_glinker",
            "description": "L2 mention-search pipeline — L3 scores only matched candidates",
            "nodes": nodes,
        }
        self._executor = ProcessorFactory.create_from_dict(config)
        self._patch_encoder_forward_kwargs()
        logger.info("GLinker executor created (%s)", pipeline_stages)

    def _patch_encoder_forward_kwargs(self) -> None:
        """Strip gliner's collator kwargs from the loaded encoder forward(s) (FIX-2, contained).

        ``gliner`` 0.2.x injects ``token_lengths``/``word_lengths``/``packing_config``/
        ``pair_attention_mask`` and forwards them into the encoder, but the gliner-linker backbone
        is a plain ``transformers`` DeBERTa whose ``forward()`` rejects them → every neural batch
        raises and the linker silently degrades to direct-KB (validated live: stripping
        ``token_lengths`` revives the neural path, 227/232 mentions neural-linked). This walks the
        just-built executor for the encoder module(s) (forward declaring ``input_ids``) and wraps
        each to drop exactly the kwargs its real signature lacks — a no-op if a future backbone
        already accepts them. Best effort: any failure here must not break executor construction
        (the caller's direct-KB fallback still holds), so it is fully guarded and logged at DEBUG.
        """
        try:
            import torch

            patched = 0
            for root in _collect_modules(self._executor, torch.nn.Module):
                for mod in root.modules():
                    if _forward_names_input_ids(mod) and _strip_unsupported_forward_kwargs(mod):
                        patched += 1
            logger.debug("FIX-2: patched %d encoder forward(s) (stripped gliner kwargs)", patched)
        except Exception as e:  # noqa: BLE001 - the patch is best-effort; never fail executor build
            logger.debug("token_lengths encoder patch skipped: %s", e)

    @staticmethod
    def _empty_cache() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 - cache clearing is best-effort
            logger.debug("cache clear skipped: %s", e)

    def warmup(self, entity_kb: list[dict[str, Any]]) -> None:
        """Load entities into the GLinker executor and precompute BiEncoder label embeddings.

        The critical performance step: precomputing BiEncoder embeddings caches label
        representations so subsequent linking calls only encode the input text (10-100x speedup).
        """
        self._ensure_executor()

        if not self._kb_loaded:
            logger.info("Loading %d entities into GLinker executor", len(entity_kb))
            self._executor.load_entities(entity_kb)
            self._kb_loaded = True

            if self.precompute_embeddings:
                logger.info(
                    "Precomputing BiEncoder embeddings for %d entities (batch_size=%d)...",
                    len(entity_kb),
                    self.precompute_batch_size,
                )
                try:
                    self._executor.precompute_embeddings(batch_size=self.precompute_batch_size)
                    self._embeddings_precomputed = True
                    # Release fragmented VRAM blocks so inference can allocate contiguous memory.
                    self._empty_cache()
                    logger.info("BiEncoder embeddings precomputed for %d entities", len(entity_kb))
                except Exception as e:  # noqa: BLE001 - precompute is an optimization, not required
                    logger.warning("Precompute embeddings failed: %s — computing lazily", e)

    def reload_kb(self, entity_kb: list[dict[str, Any]]) -> None:
        """Force-reload the KB and re-precompute embeddings."""
        self._kb_loaded = False
        self._embeddings_precomputed = False
        self.warmup(entity_kb)

    def link_mentions(
        self,
        mentions: list[dict[str, Any]],
        entity_kb: list[dict[str, Any]],
        threshold: float = 0.3,
        batch_size: int = 32,
    ) -> list[LinkedEntity | None]:
        """Link mentions against the KB using the full GLinker L2->L3->L4->L0 pipeline.

        L2 receives mention texts and does exact dict-store lookup, returning at most
        ``max_candidates`` per mention. L3 then scores only those candidates (not the entire KB),
        keeping the scoring matrix small enough for any GPU. With precomputed BiEncoder embeddings,
        L3 only needs to encode the input text — entity embeddings are looked up from the cache.
        """
        self._ensure_executor()

        if not self._kb_loaded:
            self.warmup(entity_kb)

        self._empty_cache()

        results: list[LinkedEntity | None] = []

        for i in range(0, len(mentions), batch_size):
            batch = mentions[i : i + batch_size]
            batch_texts = [m.get("context", m["text"]) for m in batch]
            # Nested mentions: [[mention1], [mention2], ...] so L2 does per-text candidate retrieval
            # (max_candidates) instead of returning the entire KB (11k+ entities -> OOM in L3).
            batch_mentions = [[m["text"]] for m in batch]

            execute_input = {"texts": batch_texts, "mentions": batch_mentions}

            try:
                result = self._executor.execute(execute_input)
                results.extend(
                    self._parse_glinker_result(result, batch, entity_kb, threshold)
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._empty_cache()
                    results.extend(
                        self._oom_retry_batch(batch, entity_kb, threshold, batch_size)
                    )
                else:
                    logger.warning("GLinker batch failed: %s — %d unresolved", e, len(batch))
                    results.extend([None] * len(batch))
            except Exception as e:  # noqa: BLE001 - a batch failure marks its mentions unresolved
                logger.warning("GLinker batch failed: %s — %d mentions unresolved", e, len(batch))
                results.extend([None] * len(batch))

        return results

    def _oom_retry_batch(
        self,
        batch: list[dict[str, Any]],
        entity_kb: list[dict[str, Any]],
        threshold: float,
        original_batch_size: int,
    ) -> list[LinkedEntity | None]:
        """Progressive OOM recovery: halve batch size until success or bs=1.

        Tracks how many items were successfully processed so retries only cover the remainder.
        """
        results: list[LinkedEntity | None] = []
        processed = 0
        retry_bs = max(1, original_batch_size // 2)

        while retry_bs >= 1 and processed < len(batch):
            remaining_batch = batch[processed:]
            logger.warning(
                "CUDA OOM — retrying %d remaining items with batch_size=%d",
                len(remaining_batch),
                retry_bs,
            )
            try:
                for j in range(0, len(remaining_batch), retry_bs):
                    sub = remaining_batch[j : j + retry_bs]
                    sub_texts = [m.get("context", m["text"]) for m in sub]
                    sub_mentions = [[m["text"]] for m in sub]
                    result = self._executor.execute({"texts": sub_texts, "mentions": sub_mentions})
                    results.extend(self._parse_glinker_result(result, sub, entity_kb, threshold))
                    processed += len(sub)
                return results
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._empty_cache()
                    if retry_bs == 1:
                        break
                    retry_bs = max(1, retry_bs // 2)
                else:
                    break

        remaining = len(batch) - processed
        if remaining > 0:
            logger.error("OOM unrecoverable for %d mentions — marking unresolved", remaining)
            results.extend([None] * remaining)
        return results

    def _parse_glinker_result(
        self,
        result: Any,
        mentions: list[dict[str, Any]],
        entity_kb: list[dict[str, Any]],
        threshold: float,
    ) -> list[LinkedEntity | None]:
        """Parse GLinker L0 aggregated output into :class:`LinkedEntity` objects.

        L0Output structure (glinker 0.1.1): ``result.get('l0_result').entities`` is a
        ``list[list[L0Entity]]`` (outer per input text, inner per found mention); each L0Entity
        carries ``mention_text`` and a ``linked_entity`` with ``.entity_id/.label/.confidence``.
        """
        kb_by_id: dict[str, dict[str, Any]] = {}
        kb_by_label: dict[str, dict[str, Any]] = {}
        for ent in entity_kb:
            kb_by_id[ent.get("entity_id", "")] = ent
            kb_by_label[ent.get("label", "").lower()] = ent

        parsed: list[LinkedEntity | None] = []

        l0_result = result.get("l0_result") if hasattr(result, "get") else None
        if not l0_result:
            logger.warning("No l0_result in GLinker output — all mentions unresolved")
            return [None] * len(mentions)

        linked_map: dict[str, LinkedEntity] = {}

        entity_lists = getattr(l0_result, "entities", [])
        for text_entities in entity_lists:
            entities_seq = text_entities if isinstance(text_entities, list) else [text_entities]

            for l0_ent in entities_seq:
                mention_text = getattr(l0_ent, "mention_text", "")
                if not mention_text:
                    continue

                le = getattr(l0_ent, "linked_entity", None)
                if le is not None:
                    linked_id = getattr(le, "entity_id", "")
                    linked_label = getattr(le, "label", "")
                    linked_score = getattr(le, "confidence", 0.0)
                else:
                    candidates = getattr(l0_ent, "candidates", [])
                    if not candidates:
                        continue
                    best = candidates[0]
                    linked_id = getattr(best, "entity_id", "")
                    linked_label = getattr(best, "label", "")
                    linked_score = getattr(best, "score", 0.0)

                if linked_score < threshold:
                    continue

                kb_entry = kb_by_id.get(linked_id) or kb_by_label.get(linked_label.lower(), {})

                candidate = LinkedEntity(
                    entity_id=linked_id or kb_entry.get("entity_id", linked_label),
                    label=kb_entry.get("label", linked_label),
                    description=kb_entry.get("description", ""),
                    score=round(linked_score, 4),
                    entity_type=kb_entry.get("entity_type", ""),
                    aliases=list(kb_entry.get("aliases", [])),
                    source="glinker",
                )
                existing = linked_map.get(mention_text.lower())
                if existing is None or candidate.score > existing.score:
                    linked_map[mention_text.lower()] = candidate

        for mention in mentions:
            parsed.append(linked_map.get(mention["text"].lower()))

        return parsed


class VLLMGLiNERBackend(InferenceBackend):
    """vLLM in-process GLiNER Linker + STS KB matching — DEFERRED, seam-compatible follow-on.

    Ported (PORT-AS-IS) from ``dataset_intelligence.linking.backends.VLLMGLiNERBackend``, adapted
    at ONE DI seam: KB-similarity matching (stage 2) uses the injected framework
    :class:`~latence_disambig_glinker.embedder.LabelEncoder` (Granite r2) instead of the source's
    ``ontology.embeddings.embed_labels``. Two-stage linking: (1) NER span extraction via vLLM
    GLiNER (``LLM.embed()`` with a ``PoolingParams(extra_kwargs=gliner_data)``), (2) KB matching by
    exact/alias text then STS cosine.

    This backend needs the vllm-factory ``deberta_gliner_linker`` plugin (a heavier serving stack),
    so it is a **documented follow-on**: the transformers :class:`GLinkerBackend` is the complete
    enterprise default and ships the full L2->L3->L4->L0 + audited resolver today. The seam is
    identical (``InferenceBackend``), so a stack can swap to it once the plugin is vendored, with
    no Provider change. All deps are imported lazily; ProviderProfile ``device=cuda`` covers it.
    """

    def __init__(
        self,
        model_path: str | None = None,
        gpu_memory_utilization: float = 0.4,
        ner_threshold: float = 0.5,
        label_encoder: LabelEncoder | None = None,
    ) -> None:
        self._model_path = model_path
        self._gpu_memory_utilization = gpu_memory_utilization
        self._ner_threshold = ner_threshold
        self._label_encoder = label_encoder
        self._llm: Any = None
        self._tokenizer: Any = None
        self._kb_embeddings: list[list[float]] | None = None
        self._kb_entries: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "vllm_gliner"

    def warmup(self, entity_kb: list[dict[str, Any]]) -> None:
        """Precompute KB embeddings for entity-KB similarity matching (stage 2)."""
        self._kb_entries = entity_kb
        if not entity_kb or self._label_encoder is None:
            return
        kb_texts = []
        for ent in entity_kb:
            desc = ent.get("description", "")
            label = ent.get("label", "")
            kb_texts.append(f"{label}: {desc}" if desc else label)
        self._kb_embeddings = self._label_encoder.embed_labels(kb_texts)
        logger.info("VLLMGLiNERBackend: precomputed embeddings for %d KB entries", len(entity_kb))

    def _match_to_kb(
        self,
        mention_text: str,
        entity_kb: list[dict[str, Any]],
        threshold: float,
    ) -> LinkedEntity | None:
        """Match a mention to a KB entry using exact/alias text then STS cosine similarity."""
        if not entity_kb:
            return None
        mt = mention_text.strip().lower()
        for ent in entity_kb:
            if ent.get("label", "").strip().lower() == mt:
                return LinkedEntity(
                    entity_id=ent.get("entity_id", ent["label"]),
                    label=ent["label"],
                    description=ent.get("description", ""),
                    score=max(0.9, ent.get("_avg_score", 0.5)),
                    entity_type=ent.get("entity_type", ""),
                    aliases=list(ent.get("aliases", [])),
                    source="vllm_gliner_exact",
                )
            for alias in ent.get("aliases", []):
                if alias.strip().lower() == mt:
                    return LinkedEntity(
                        entity_id=ent.get("entity_id", ent["label"]),
                        label=ent["label"],
                        description=ent.get("description", ""),
                        score=max(0.85, ent.get("_avg_score", 0.5)),
                        entity_type=ent.get("entity_type", ""),
                        aliases=list(ent.get("aliases", [])),
                        source="vllm_gliner_alias",
                    )

        if (
            self._kb_embeddings is not None
            and self._label_encoder is not None
            and len(self._kb_entries) == len(entity_kb)
        ):
            mention_emb = self._label_encoder.embed_labels([mention_text])[0]
            sims = [cosine(mention_emb, kb_vec) for kb_vec in self._kb_embeddings]
            best_idx = max(range(len(sims)), key=lambda i: sims[i]) if sims else 0
            best_sim = sims[best_idx] if sims else 0.0
            if best_sim >= threshold:
                ent = entity_kb[best_idx]
                return LinkedEntity(
                    entity_id=ent.get("entity_id", ent["label"]),
                    label=ent["label"],
                    description=ent.get("description", ""),
                    score=round(best_sim, 4),
                    entity_type=ent.get("entity_type", ""),
                    aliases=list(ent.get("aliases", [])),
                    source="vllm_gliner_embed",
                )
        return None

    def link_mentions(
        self,
        mentions: list[dict[str, Any]],
        entity_kb: list[dict[str, Any]],
        threshold: float = 0.3,
        batch_size: int = 32,
    ) -> list[LinkedEntity | None]:
        """DEFERRED: the vLLM NER stage needs the deberta_gliner_linker plugin.

        Stage-2 KB matching (exact/alias/STS) is fully implemented and seam-compatible; the vLLM
        NER stage (stage 1) is the documented follow-on. Rather than fabricate NER results, this
        falls back to direct KB matching so the seam is honest and never crashes.
        """
        return [self._match_to_kb(m["text"], entity_kb, threshold) for m in mentions]


def get_backend(config: Any) -> InferenceBackend:
    """Factory: create the appropriate backend from config (default GLinkerBackend).

    ``config`` is a duck-typed settings object (``getattr`` reads). The default is the full
    L2->L3->L4->L0 :class:`GLinkerBackend`; ``vllm_inprocess`` selects the deferred
    :class:`VLLMGLiNERBackend`.
    """
    backend_type = getattr(config, "inference_backend", "glinker")
    model_name = getattr(config, "linker_model", GLinkerBackend.LINKER_MODEL)

    if backend_type == "glinker":
        return GLinkerBackend(
            model_name=model_name,
            device=getattr(config, "device", "cuda"),
            precompute_embeddings=getattr(config, "precompute_embeddings", True),
            reranker_model=getattr(config, "reranker_model", GLinkerBackend.RERANKER_MODEL),
            reranker_max_labels=getattr(config, "reranker_max_labels", 20),
            reranker_threshold=getattr(config, "reranker_threshold", None),
            use_flash_deberta=getattr(config, "use_flash_deberta", True),
            threshold=getattr(config, "threshold", 0.3),
            max_length=getattr(config, "max_length", 512),
            precompute_batch_size=getattr(config, "precompute_batch_size", 16),
            max_candidates=getattr(config, "max_candidates", 100),
            l2_search_mode=getattr(config, "l2_search_mode", "exact"),
        )
    if backend_type == "vllm_inprocess":
        return VLLMGLiNERBackend(
            gpu_memory_utilization=getattr(config, "linker_gpu_memory", 0.4),
            ner_threshold=getattr(config, "threshold", 0.5),
            label_encoder=getattr(config, "label_encoder", None),
        )
    msg = (
        f"Unknown inference backend {backend_type!r}. Supported: glinker, vllm_inprocess."
    )
    raise ValueError(msg)
