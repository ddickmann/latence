"""Learned PII Redaction Provider over the fastino **GLiNER2** framework (ADR-0024/0036).

A **twin of** :class:`latence_pii_gliner.provider.GlinerPIIRedactor`: it reuses ALL the shared
redaction machinery (the :class:`~latence_core.redaction_policy.RedactionPolicy`, masking /
placeholder / action logic, :class:`~latence_core.contracts.PIISpan` /
:class:`~latence_core.contracts.RedactionRecord` contracts, offset handling, the
``PARSE_ERROR`` pass-through, and the ``Classification.sensitivity``-keyed policy) and swaps
ONLY the detection call — from the ``gliner`` library to the **``gliner2``** framework, a
*different* Python package with a *different* API.

The ``gliner2`` framework wraps an mdeberta-v3 encoder and exposes a single
``extract_entities`` call that returns a **per-label dict of dict-rows** (NOT the flat list of
dicts the ``gliner`` library returns). The REAL API (introspected + run on the pod, W3):

.. code-block:: python

    from gliner2 import GLiNER2
    # W9: gliner2 accepts NATIVE quantize (fp16) + compile (torch.compile + fused kernels) kwargs.
    model = GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi",
                                    quantize=True, compile=True)  # 2.83× at byte-identical PII
    out = model.extract_entities(text, entity_types, threshold=0.5,
                                 include_spans=True, include_confidence=True)
    # -> {"entities": {"<label>": [{"text": str, "confidence": float,
    #                               "start": int, "end": int}, ...], ...}}

Offsets are document-char ``start``/``end`` (half-open), exact — they map directly to
``PIISpan(char_start=start, char_end=end, pii_type=label, confidence=confidence)``.

It fulfils the same :class:`~latence_core.capability.PIIDetector` seam as the in-core
``redaction.hybrid_rule`` and the twin ``redaction.gliner_pii``, so it is a drop-in behind
Redaction — the Pipeline swaps ``provider: redaction.hybrid_rule`` (or ``redaction.gliner_pii``)
for ``provider: redaction.gliner2`` with no code change, same ``RedactionRecord`` contract. The
**raw PII value is never persisted** on a :class:`~latence_core.contracts.PIISpan` (S6 AC): the
surface string is used only transiently to derive a deterministic surrogate.

Licensing (verified 2026-07-09):
    * ``gliner2`` library (PyPI + github.com/fastino-ai/GLiNER2): **Apache-2.0** (code) —
      LICENSE file + README badge + PyPI ``OSI Approved :: Apache Software License`` classifier.
    * Default checkpoint ``fastino/GLiNER2-Guardrails-PII-Multi``: **Apache-2.0** (weights, per
      the HF model card; NOT gated / freely downloadable).
    Both weights AND code are permissive; nothing downloads unless a Pipeline actually
    configures this Provider.

The ``gliner2`` import is deferred to first use (ADR-0016) so importing this module (for Provider
discovery) never requires torch, and a missing optional dep raises a clear error only when the
Provider is run. It builds on :class:`~latence_core.providers.adapter.AdapterBase` for the
device seam (``compute="either"``: mdeberta runs on CPU or GPU, skip-with-flag routed by the
device seam) and the typed-error guard — a native ``gliner2`` fault becomes a typed
:class:`~latence_core.errors.ProviderError` carrying NO record content / PII (C2/C6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError
from latence_core.induced import LabelSlot, effective_labels
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.perf import (
    is_compile_backend_failure,
    load_gliner2_optimized,
    resolve_gliner2_knob,
    resolve_max_len,
)
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    ChunkRedactionPlan,
    RedactionPolicy,
    finalize_redacted_chunk,
    parse_action,
    plan_chunk_redaction,
)

_log = logging.getLogger("latence.provider.gliner2_pii")

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord


@dataclass(frozen=True)
class _PendingChunk:
    """A chunk's model-independent redaction context, resolved before the batched call (W7/W16).

    The :class:`~latence_core.redaction_policy.ChunkRedactionPlan` (policy action/allow-list, the
    skip/near-empty decision, the chunk offset/page maps) and the effective PII label set (``config
    ∪ induced``, W12) are per-chunk and model-independent, so they are computed once up front; a
    flush then groups the detect-needing chunks by their ``labels`` and issues one
    ``batch_extract_entities`` call per label group, mapping each per-text result back to its chunk
    by identity, keeping emission in input-chunk order and the output record-for-record identical to
    the per-chunk path.
    """

    chunk: ChunkRecord
    text: str
    plan: ChunkRedactionPlan
    labels: tuple[str, ...]

# Apache-2.0 weights (verified 2026-07-09, HF model card, not gated). Multilingual PII NER over
# an mdeberta-v3 encoder via the fastino GLiNER2 framework.
_DEFAULT_MODEL = "fastino/GLiNER2-Guardrails-PII-Multi"


class Gliner2PIIRedactor(AdapterBase):
    """Learned PIIDetector Provider backed by the fastino GLiNER2 framework (mdeberta-v3).

    A near-clone of :class:`latence_pii_gliner.provider.GlinerPIIRedactor` that reuses the shared
    redaction machinery and changes ONLY detection (the ``gliner2`` framework instead of the
    ``gliner`` library).

    Config:
        pii_labels: list[str]  — REQUIRED. The PII entity types to detect (e.g.
            ``["person", "email", "phone number"]``). GLiNER2 works best with lower-case,
            natural-language type names.
        model: str  — the checkpoint id (default ``fastino/GLiNER2-Guardrails-PII-Multi``,
            Apache-2.0).
        threshold: float  — score threshold; spans below it are dropped (default 0.5).
        default_action: str  — action when a policy names none: ``mask`` (default),
            ``replace``, ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map (identical shape to the
            twin Provider): per ``Classification.sensitivity`` value an ``action`` / ``types``
            allow-list / ``skip`` flag. Keys off the document's sensitivity (S6 AC).
        salt: str  — salt mixed into ``replace``/``hash`` surrogates (default 'latence').
        min_chars: int  — a document whose assembled text is shorter is emitted clean
            (default 1).
        max_len: int  — the mdeberta-v3 model token window (default 768), threaded as the native
            ``max_len`` of ``batch_extract_entities`` (W12). A document that tokenizes past it is
            SILENTLY truncated by the model (tail PII spans lost, offsets past the window
            unreliable); this bounds the window explicitly. CAVEAT: the chunker counts in a
            DIFFERENT tokenizer, ~1.2× fewer tokens than mdeberta — keep the chunk cap ≤ ~640 for
            this 768 window; this guard is the hard backstop.
        batch_size: int  — how many documents share ONE ``batch_extract_entities`` forward pass
            (default from AdapterBase; the P8 autotuner overrides per-device). Purely
            memory-shaping: the emitted RedactionRecords (PIISpans/masking/order) are
            record-for-record identical for any positive batch size (W7). Must be ``>= 1``.
        quantize: bool | 'auto'  — load fp16 via the gliner2 native ``quantize`` kwarg (W9,
            default ``auto`` → True on cuda / False on cpu). Pod-measured 2.83× at byte-identical
            PII. Guarded: a quantize fault degrades to a plain load, never a crash.
        compile: bool | 'auto'  — ``torch.compile`` + fused GPU kernels via the gliner2 native
            ``compile`` kwarg (W9, default ``auto`` → True on cuda / False on cpu). Guarded like
            ``quantize``: a compile/warmup fault degrades to a plain load.
        device: str  — ``auto`` (default) / ``cpu`` / ``cuda``; routed by the device seam.

    The model is loaded lazily and cached on first use, so constructing the Provider never
    downloads weights.

    Induced labels (W12, ADR-0038): the detection label set is ``effective_labels(config ∪
    induced)`` per document — the SAME seam the twin ``redaction.gliner_pii`` and the extractors
    use — so a ``pii_type`` an upstream ``schema_induction`` Stage induced (e.g. a corpus-wide
    ``IBAN`` / ``tax id``) is actually redacted. The config ``pii_labels`` floor stays mandatory
    (validated non-empty at construction); induction only EXTENDS it. Absent induction the effective
    set is exactly the config labels, byte-identical to before.

    Batching (W7, ADR-0036 §1 / ADR-0033): GLiNER2 (mdeberta-v3) is an ENCODER, so a per-document
    ``extract_entities`` loop under-feeds the GPU. This Provider buffers up to ``batch_size``
    documents and, on flush, GROUPS the detect-needing docs by their effective label set (one label
    set per ``batch_extract_entities`` call) and runs ONE ``model.batch_extract_entities(texts,
    labels, threshold=…, include_spans=True, include_confidence=True)`` forward pass per group (the
    batched entry point — ``hasattr``-guarded here and pod-Gate-0 asserted; offline-unverifiable so
    not yet pod-confirmed) — ``ceil(N/batch_size)`` calls when all docs share a label set, not N —
    mapping each per-text result back to its document **by identity**. Documents that a policy SKIPS
    or that are near-empty need no model call and are emitted clean, in input order. The
    per-document policy resolution, span building, masking, and Provenance carry are UNCHANGED; the
    in-flight
    buffer is bounded to ``batch_size`` (never the whole corpus, ADR-0033).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default GLiNER2-PII checkpoint.

        ``compute="either"``: the GLiNER2 mdeberta-v3 model runs in-process on a laptop CPU
        (ADR-0007) and accelerates on a GPU when present, so the device seam routes it rather
        than skipping it on a CPU-only host.

        License **verified** (recorded 2026-07-09): the ``gliner2`` library is Apache-2.0 (code:
        github.com/fastino-ai/GLiNER2 LICENSE + PyPI classifier) and the default checkpoint
        ``fastino/GLiNER2-Guardrails-PII-Multi`` is Apache-2.0 (weights, per the HF model card,
        not gated) — both permissive, so ``license_verified=True``.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical across
        hardware/library versions, so the profile does not overclaim reproducibility (unlike the
        pure-code hybrid-rule Redactor). ``memory_mb`` is the approximate resident footprint of
        the checkpoint + transformers stack (declared, not measured).
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
                "https://huggingface.co/fastino/GLiNER2-Guardrails-PII-Multi (weights, "
                "Apache-2.0, not gated); "
                "https://github.com/fastino-ai/GLiNER2 (library, Apache-2.0 — LICENSE + PyPI "
                "'OSI Approved :: Apache Software License' classifier)"
            ),
            license_verified_on="2026-07-09",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note=(
                "gliner2 native quantize (fp16) + compile (torch.compile + fused GPU kernels) on "
                "cuda, off on cpu (W9; pod-measured 2.83× PII at byte-identical output; guarded → "
                "plain load on any fault)"
            ),
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        pii_labels = config.get("pii_labels")
        if not pii_labels or not isinstance(pii_labels, list):
            msg = (
                "Gliner2PIIRedactor requires a non-empty 'pii_labels' list in its config "
                "(the PII entity type set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        # AdapterBase resolves the device seam (compute="either") against the declared profile.
        super().__init__(config)
        self._labels = [str(label) for label in pii_labels]
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        # W12 model-boundary guard: bound the mdeberta token window (default 768). gliner2's
        # batch_extract_entities takes a native ``max_len``; past it the model silently truncates
        # (tail PII spans lost, offsets past the window unreliable). Explicit + configurable.
        self._max_len = resolve_max_len(config)
        # Validate the batch_size knob eagerly (fail-fast at construction like the twin providers)
        # even though the batched call reads it lazily via AdapterBase.batch_size() (W7).
        self.batch_size()
        # Validate the W9 quantize/compile knobs eagerly too (fail-fast): an invalid value is an
        # operator ConfigError at construction, not a native fault buried in the lazy load.
        self._validate_perf_knobs()
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        # The framework owns policy + masking (shared with the in-core rule Provider and the
        # twin gliner Provider); this Provider owns only detection — the PIIDetector seam.
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        self._model: Any = None
        # Set once the inference-time compile guard has re-loaded the uncompiled model (W9), so the
        # fallback runs at most once per provider instance.
        self._compile_disabled: bool = False

    @property
    def provider_name(self) -> str:
        """The registered entry-point name, used only to tag typed errors (never carries PII)."""
        return "redaction.gliner2"

    def _validate_perf_knobs(self) -> None:
        """Resolve the W9 ``quantize``/``compile`` knobs eagerly → ConfigError on a bad value.

        Resolving them at construction (against the resolved device) turns an invalid
        ``quantize``/``compile`` config into an operator :class:`ConfigError` up front, rather than
        a native fault surfaced only at the lazy load — the same fail-fast discipline as
        ``batch_size``. The resolved values are recomputed at load time (device-stable), so this is
        purely validation.
        """
        try:
            resolve_gliner2_knob("quantize", self.device.device, self._config)
            resolve_gliner2_knob("compile", self.device.device, self._config)
        except ValueError as exc:
            msg = f"{self.provider_name}: {exc}"
            raise ConfigError(msg) from exc

    # -- lazy model load ----------------------------------------------------

    def _gliner2_cls(self) -> Any:
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner2' package is required to run Gliner2PIIRedactor. "
                "Install it with: pip install latence-pii-gliner2"
            )
            raise ImportError(msg) from exc
        return GLiNER2

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        # W9 perf seam: the ``gliner2`` framework (unlike plain ``gliner`` 0.2.27, and superseding
        # the W7 batching that REGRESSED redact via padding waste) exposes NATIVE ``quantize``
        # (fp16) + ``compile`` (torch.compile + fused GPU kernels, "no extra dependencies required")
        # kwargs on ``from_pretrained`` — pod-measured **2.83× at byte-identical PII**. Both default
        # True on cuda / False on cpu (or the ``quantize``/``compile`` config override), resolved
        # once here.
        # The optimized load is GUARDED: any quantize/compile fault degrades to a plain load
        # (throughput only, never a crash) — so a host whose torch/CUDA rejects fp16 or a compile
        # warmup fails still loads the identical model. An invalid ``quantize``/``compile`` value is
        # a ConfigError surfaced before the load (never a silent wrong knob).
        # The import is resolved OUTSIDE the guard so a missing-dep ImportError surfaces as the
        # clear "pip install latence-pii-gliner2" error, not a wrapped ProviderError.
        gliner2_cls = self._gliner2_cls()
        with self.guard("load_model"):
            self._model = load_gliner2_optimized(
                gliner2_cls.from_pretrained,
                self._model_id,
                self.device.device,
                self._config,
                provider_name=self.provider_name,
            )
        return self._model

    def _reload_uncompiled(self) -> Any:
        """Re-load the model with ``compile=False`` (quantize kept) after an inference-time compile
        fault — the second half of the W9 compile guard (:func:`is_compile_backend_failure`).

        ``torch.compile`` compiles lazily on the first forward, so a backend that cannot compile the
        model fails INSIDE ``batch_extract_entities`` (a ``BackendCompilerFailed``), which the
        load-time guard cannot catch. On the first such fault the provider re-loads ONCE with
        ``compile`` forced off (``quantize`` unchanged — fp16 alone is fine) and uses the uncompiled
        model for the rest of the run, so a bad compile costs only throughput, never a crash.
        Idempotent: sets ``_compile_disabled`` so the fallback runs at most once.
        """
        self._compile_disabled = True
        cfg = {**self._config, "compile": False}
        _log.warning(
            "redaction.gliner2: torch.compile failed at inference; re-loading the uncompiled "
            "model (quantize kept) — throughput only, never a crash"
        )
        gliner2_cls = self._gliner2_cls()
        with self.guard("load_model"):
            self._model = load_gliner2_optimized(
                gliner2_cls.from_pretrained,
                self._model_id,
                self.device.device,
                cfg,
                provider_name=self.provider_name,
            )
        return self._model

    def _batch_extract(self, model: Any, texts: list[str], labels: list[str]) -> list[Any]:
        """One batched forward (``batch_extract_entities``) with the W9 inference compile guard.

        ``labels`` is this group's effective label set (``config ∪ induced``, W12). The forward runs
        under :meth:`AdapterBase.guard` (native fault → typed
        :class:`~latence_core.errors.ProviderError`, no PII leak). BUT ``torch.compile`` compiles
        lazily on the first forward, so a backend that cannot compile the model raises a
        ``BackendCompilerFailed`` (dynamo/inductor) HERE, not at load. This wrapper catches exactly
        that class of fault (:func:`~latence_core.providers.perf.is_compile_backend_failure`) BEFORE
        the guard maps it, re-loads the uncompiled model ONCE (:meth:`_reload_uncompiled`; quantize
        kept), and retries the batch — so a bad compile costs only throughput, never a crash. Any
        OTHER native fault is surfaced through the guard as before (typed, no PII).
        """
        try:
            with self.guard("batch_extract_entities"):
                # ONE forward pass over the group; results[i] is the per-label dict for texts[i].
                return list(
                    model.batch_extract_entities(
                        texts,
                        labels,
                        threshold=self._threshold,
                        include_spans=True,
                        include_confidence=True,
                        max_len=self._max_len,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - intercept a compile fault before surfacing
            # ``guard`` wraps a compile fault as ProviderError; the predicate walks the cause chain
            # to find the original BackendCompilerFailed underneath.
            if not self._compile_disabled and is_compile_backend_failure(exc):
                model = self._reload_uncompiled()
                with self.guard("batch_extract_entities"):
                    return list(
                        model.batch_extract_entities(
                            texts,
                            labels,
                            threshold=self._threshold,
                            include_spans=True,
                            include_confidence=True,
                            max_len=self._max_len,
                        )
                    )
            raise

    # -- redaction ----------------------------------------------------------

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Detect PII per chunk with the model; yield the SAME chunks with PII masked (W16).

        Each chunk's ``content`` (≤ the model window) is scanned in full — no truncation — and the
        chunk is re-emitted with ``masked_content`` + ``pii_spans`` set (``content`` UNCHANGED).
        Chunks are threaded 1:1 in first-seen order; a chunk-local span resolves to its ORIGINAL
        source page through the chunk's offset_map + page_map (ADR-0031/0042).
        """
        buffer: list[_PendingChunk] = []

        def flush() -> Iterator[ChunkRecord]:
            yield from self._flush_buffer(buffer)
            buffer.clear()

        for chunk in chunks:
            buffer.append(self._prepare(chunk))
            if len(buffer) >= self.batch_size():
                yield from flush()
        yield from flush()

    def _prepare(self, chunk: ChunkRecord) -> _PendingChunk:
        """Resolve everything a chunk needs BEFORE the (possibly batched) model call (W7/W16).

        The :class:`~latence_core.redaction_policy.ChunkRedactionPlan` (policy resolution, the
        near-empty/skip decision, the chunk offset/page maps) and the effective PII label set are
        per-chunk and model-independent, so they are computed here up front; the flush then issues
        one batched model call per label group over the detect-needing chunks.
        """
        plan = plan_chunk_redaction(
            chunk,
            self._policy,
            detector_types=set(self._labels),
            sensitive_set=set(),
            min_chars=self._min_chars,
        )
        # W12 label-source seam (ADR-0038): UNION the config PII floor with any pii_types the
        # OPTIONAL schema_induction Stage induced for this chunk (``induced_labels``), deduped +
        # order-stable — the SAME seam the twin ``redaction.gliner_pii`` and the extractors use, so
        # a corpus-induced IBAN/tax-id label is actually redacted. Absent (the default, every
        # existing stack) these are exactly the config labels — byte-identical. The config floor
        # stays mandatory (``pii_labels`` is validated non-empty in __init__); induction only
        # EXTENDS it.
        pii_labels = effective_labels(self._labels, chunk.induced_labels, LabelSlot.PII)
        return _PendingChunk(chunk=chunk, text=chunk.content, plan=plan, labels=tuple(pii_labels))

    def _flush_buffer(self, buffer: list[_PendingChunk]) -> Iterator[ChunkRecord]:
        """Batch the detect-needing chunks (grouped by label set), then emit in input order (W16).

        ``batch_extract_entities`` takes ONE label set per call, so the detect-needing chunks are
        grouped by their effective ``labels`` (``config ∪ induced``, W12) and each group goes
        through one forward pass — ``ceil`` per group, not one-per-chunk; each per-text result maps
        back to its chunk by identity. Absent induction every chunk shares the config label set → a
        single group → byte-identical to the pre-W12 single-call path. Chunks that skip / are
        near-empty need no model call. Emission is in the buffer's (input) order regardless of
        grouping, so the output is record-for-record identical to the per-chunk path (the
        equivalence invariant).
        """
        if not buffer:
            return
        raw_by_chunk: dict[int, Any] = {}
        detect = [p for p in buffer if p.plan.needs_detection]
        if detect:
            # The batched entry point is offline-unverifiable (the fake mirrors it); guard against
            # a renamed/missing method with a clear, actionable error rather than a bare
            # AttributeError (the guard would otherwise surface only the op label). The pod script's
            # Gate 0 asserts this method EXISTS on the real checkpoint before any throughput number.
            if not hasattr(self._load_model(), "batch_extract_entities"):
                msg = (
                    "redaction.gliner2 batches via model.batch_extract_entities(texts, "
                    "entity_types, threshold=…), but the loaded GLiNER2 checkpoint exposes no "
                    "'batch_extract_entities' method. Re-introspect the checkpoint and fix the "
                    "batched call. See scripts/validate-encoder-batching-on-gpu.md Gate 0."
                )
                raise AttributeError(msg)
            # Group by effective label set (one label set per batched call). Order-stable: a dict
            # preserves first-seen group order, and emission below re-walks the buffer in input
            # order.
            groups: dict[tuple[str, ...], list[_PendingChunk]] = {}
            for pending in detect:
                groups.setdefault(pending.labels, []).append(pending)
            for labels, pendings in groups.items():
                # Re-fetch per group so a prior group's W9 uncompiled reload (which updates
                # self._model) is picked up here, not the stale compiled handle.
                model = self._load_model()
                texts = [p.text for p in pendings]
                # ONE forward pass over the group (results[i] is the per-label dict for texts[i]),
                # with the W9 inference-time compile guard: a lazy torch.compile fault degrades to
                # the uncompiled model and retries, never a crash.
                results_list = self._batch_extract(model, texts, list(labels))
                if len(results_list) != len(pendings):
                    msg = (
                        "redaction.gliner2: batch_extract_entities returned "
                        f"{len(results_list)} results for a batch of {len(pendings)} texts."
                    )
                    raise ValueError(msg)
                for pending, raw in zip(pendings, results_list, strict=True):
                    raw_by_chunk[id(pending)] = raw
        for pending in buffer:
            accepted: list[tuple[int, int, str, str, str, float]] = []
            if pending.plan.needs_detection:
                accepted = self._accepted_from_raw(
                    pending.text,
                    raw_by_chunk[id(pending)],
                    pending.plan.allowed,
                )
            yield finalize_redacted_chunk(pending.chunk, pending.plan, accepted, self._salt)

    @staticmethod
    def _accepted_from_raw(
        text: str,
        raw: Any,
        allowed: set[str] | None,
    ) -> list[tuple[int, int, str, str, str, float]]:
        """Build an ordered, non-overlapping candidate list from THIS chunk's raw GLiNER2 output.

        ``gliner2``'s ``extract_entities`` / ``batch_extract_entities`` return a **per-label dict**
        of dict-rows: ``{"entities": {"<label>": [{"text", "confidence", "start", "end"}, ...],
        ...}}`` — a DIFFERENT shape from the ``gliner`` library's flat list. ``raw`` is this chunk's
        result (mapped back by index from the batched call, W7). The ``start``/``end`` are
        chunk-offset (relative to the chunk ``content`` passed in). This flattens the per-label dict
        → one candidate per row, then a type not in a policy's allow-list is dropped and overlaps
        resolved (earlier start, longer span, type) so a stable, deterministic, non-overlapping set
        results. Returns ``(local_start, local_end, pii_type, value, detector, confidence)`` tuples
        (chunk-content offsets); :func:`~latence_core.redaction_policy.finalize_redacted_chunk`
        derives spans + pages + placeholders + masking.
        """
        # Faithful parse of the REAL gliner2 shape: a top-level {"entities": {label: [rows]}}
        # dict. A flat-list mis-parse (the gliner-library shape) would find no per-label rows
        # here and silently produce zero spans — the anchor test asserts the correct shape bites.
        entities_by_label: Any = raw["entities"] if isinstance(raw, dict) else raw
        if not isinstance(entities_by_label, dict):
            got = type(entities_by_label).__name__
            msg = (
                "redaction.gliner2: unexpected extract_entities output shape "
                f"(expected {{'entities': {{label: [rows]}}}}, got {got})"
            )
            raise ValueError(msg)

        # (start, end, type, value, score) tuples, filtered + sorted deterministically.
        candidates: list[tuple[int, int, str, str, float]] = []
        for label, rows in entities_by_label.items():
            pii_type = str(label)
            if not pii_type:
                continue
            if allowed is not None and pii_type not in allowed:
                continue
            for row in rows:
                start = int(row["start"])
                end = int(row["end"])
                if end <= start or start < 0 or end > len(text):
                    continue  # bounds-safe: drop a malformed/out-of-range row
                score = min(max(float(row.get("confidence", 1.0)), 0.0), 1.0)
                candidates.append((start, end, pii_type, text[start:end], score))
        # Deterministic order: by start, then longest span, then label (matches the twin).
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        accepted: list[tuple[int, int, str, str, str, float]] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            accepted.append((start, end, pii_type, value, f"gliner2:{pii_type}", score))
            last_end = end
        return accepted
