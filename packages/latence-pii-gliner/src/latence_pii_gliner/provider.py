"""Learned zero-shot PII Redaction Provider over GLiNER-PII (CPU path, ADR-0024).

The S6 *learned* reference Provider — the "model half" of the hybrid rule+model
redaction the issue calls for. ``urchade/gliner_multi_pii-v1`` is a zero-shot PII NER
model (fine-tuned from ``urchade/gliner_multi-v2.1`` on a synthetic PII dataset), so a
single checkpoint recognises ``person``/``email``/``phone number``/``credit card``…
with the type set supplied per Pipeline — no retraining, nothing hardcoded in core. It
wraps the ``gliner`` library (plain ``transformers``/ONNX, no vLLM), so it runs
in-process on a laptop CPU (ADR-0007): ``pip install latence-pii-gliner`` → learned
PII redaction, no CUDA.

It fulfils the same :class:`~latence_core.capability.PIIDetector` seam as the in-core
``redaction.hybrid_rule``, so it is a drop-in behind Redaction — the Pipeline swaps
``provider: redaction.hybrid_rule`` for ``provider: redaction.gliner_pii`` with no
code change. The two share the framework's masking/placeholder/policy machinery
(imported from :mod:`latence_core.stages.redaction`), so a detected span is masked,
replaced, hashed, or tagged identically whether it came from a regex or the model —
the only difference is detection. The **raw PII value is never persisted** on a
:class:`~latence_core.contracts.PIISpan` (S6 AC): the surface string is used only
transiently to derive a deterministic surrogate.

Licensing (ADR-0012, verified 2026-07-06):
    * ``gliner`` library (PyPI + github.com/urchade/GLiNER): **Apache-2.0** (code).
    * Default checkpoint ``urchade/gliner_multi_pii-v1``: **Apache-2.0** (weights,
      per the model card), base ``urchade/gliner_multi-v2.1`` also Apache-2.0.
    Both weights AND code are permissive; nothing downloads unless a Pipeline
    actually configures this Provider.

The ``gliner`` import is deferred to first use so importing this module (for Provider
discovery) never requires torch, and a missing optional dep raises a clear error only
when the Provider is run. Redaction is metadata-aware: policy keys off the document's
``Classification.sensitivity`` exactly as the in-core Provider does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.errors import ConfigError
from latence_core.induced import LabelSlot, effective_labels
from latence_core.providers.adapter import validate_batch_size
from latence_core.providers.device import DeviceDecision, select_device
from latence_core.providers.perf import load_gliner_model, resolve_max_len
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    ChunkRedactionPlan,
    RedactionPolicy,
    finalize_redacted_chunk,
    parse_action,
    plan_chunk_redaction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord


@dataclass(frozen=True)
class _PendingChunk:
    """A chunk's model-independent redaction context, resolved before the batched call (W7/W16).

    Everything the detection + chunk building needs that does NOT depend on the model output is
    computed once up front (the :class:`~latence_core.redaction_policy.ChunkRedactionPlan` — policy
    action/allow-list, the skip/near-empty decision, the chunk offset/page maps — and the effective
    PII label set), so a flush issues at most one ``batch_predict_entities`` call per same-label
    group
    and maps each raw entity list back to its chunk by identity — keeping emission in input-chunk
    order and the output record-for-record identical to the per-chunk path.
    """

    chunk: ChunkRecord
    text: str
    plan: ChunkRedactionPlan
    labels: tuple[str, ...]

# Apache-2.0 weights (verified 2026-07-06). Multilingual zero-shot PII NER.
_DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"

# The default number of documents sharing ONE ``batch_predict_entities`` forward pass (W7). A
# sensible GPU value (the P8 autotuner overrides per-device); purely memory-shaping — the emitted
# PIISpans/masking are record-for-record identical to a per-document loop for any positive batch
# size (ADR-0033: the in-flight tensor is bounded to ``batch_size``, never the whole corpus).
_DEFAULT_BATCH_SIZE = 16


def _import_gliner() -> Any:
    """Return the ``GLiNER`` class — the backbone-import factory the perf seam calls (#190).

    Handed to :func:`~latence_core.providers.perf.load_gliner_model` rather than called here, so the
    TorchScript JIT is provably dead **before** this import evaluates: it transitively pulls in
    transformers' ``deberta_v2`` module, whose relative-position helpers are compiled by
    ``@torch.jit.script`` AT IMPORT and can never be un-scripted afterwards (the Blackwell/CUDA-13
    ``libnvrtc-builtins.so.13.0`` crash, #190). Passing the import as a factory is what makes that
    ordering un-gettable-wrong: this package still owns the native dependency and its install hint
    (ADR-0036 §2) — only the INSTANT the import runs moved into the seam.
    """
    try:
        from gliner import GLiNER
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        msg = (
            "The 'gliner' package is required to run GlinerPIIRedactor. "
            "Install it with: pip install latence-pii-gliner"
        )
        raise ImportError(msg) from exc
    return GLiNER


class GlinerPIIRedactor:
    """Learned zero-shot PIIDetector Provider backed by a GLiNER-PII checkpoint.

    Config:
        pii_labels: list[str]  — REQUIRED. The zero-shot PII types to detect (e.g.
            ``["person", "email", "phone number", "credit card number"]``). GLiNER
            works best with lower-case, natural-language type names.
        model: str  — the checkpoint id (default ``urchade/gliner_multi_pii-v1``,
            Apache-2.0).
        threshold: float  — score threshold; spans below it are dropped (default 0.5).
        default_action: str  — action when a policy names none: ``mask`` (default),
            ``replace``, ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map (identical shape
            to the in-core Provider): per ``Classification.sensitivity`` value an
            ``action`` / ``types`` allow-list / ``skip`` flag. Keys off the document's
            sensitivity (S6 AC).
        device: str  — ``auto`` (default) / ``cpu`` / ``cuda``, routed by the ADR-0036 device
            seam and APPLIED at load: the model is moved onto the resolved device and its
            residency verified (#191 — the plain ``gliner`` library never auto-places, so before
            this a configured ``cuda`` silently ran on CPU at ~10× the cost). ``cuda`` on a host
            without a GPU is a loud ConfigError, never a silent CPU fallback.
        salt: str  — salt mixed into ``replace``/``hash`` surrogates (default 'latence').
        min_chars: int  — a document whose assembled text is shorter is emitted clean
            (default 1).
        max_len: int  — the mdeberta-v3 model token window (default 768), set as
            ``model.config.max_len`` at load (W12). A document that tokenizes past it is SILENTLY
            truncated by gliner (tail PII spans lost / offsets past the window unreliable); this
            bounds the window explicitly. CAVEAT: the chunker counts in a DIFFERENT tokenizer, ~1.2×
            fewer tokens than mdeberta — keep the chunk cap ≤ ~640 for this 768 window; this guard
            is the hard backstop.
        batch_size: int  — how many documents share ONE ``batch_predict_entities`` forward pass
            (default 16; the P8 autotuner overrides per-device). Purely memory-shaping: the
            emitted RedactionRecords (PIISpans/masking/order) are record-for-record identical for
            any positive batch size (W7). Must be ``>= 1``.

    The model is loaded lazily and cached on first use, so constructing the Provider
    never downloads weights.

    Batching (W7, ADR-0036 §1 / ADR-0033): GLiNER-PII is an ENCODER, so a per-document
    ``predict_entities`` loop under-feeds the GPU. This Provider buffers up to ``batch_size``
    documents and, on flush, runs ONE ``model.batch_predict_entities(texts, labels, threshold=…)``
    forward pass per group of same-label detect-needing documents (``ceil(N/batch_size)`` calls,
    not N), mapping each raw entity list back to its document **by index**. Documents that a
    policy SKIPS or that are near-empty need no model call and are emitted clean, in input order.
    The per-document policy resolution, span building, masking, and Provenance carry are UNCHANGED
    (batching only changes how many documents share a forward pass); the in-flight buffer is
    bounded to ``batch_size`` (never the whole corpus, ADR-0033).
    """

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default GLiNER-PII checkpoint.

        ``compute="either"``: the GLiNER-PII model runs in-process on a laptop CPU (ADR-0007 —
        first contact is ``pip install`` → learned PII NER, no CUDA) and accelerates on a GPU
        when present, so the device seam routes it rather than skipping it on a CPU-only host.

        License **verified** per ADR-0012 (recorded 2026-07-06): the ``gliner`` library is
        Apache-2.0 (code) and the default checkpoint ``urchade/gliner_multi_pii-v1`` is
        Apache-2.0 (weights, base ``urchade/gliner_multi-v2.1`` also Apache-2.0) — both
        permissive, so ``license_verified=True``.

        ``deterministic=False``: learned float inference is not guaranteed byte-identical
        across hardware/library versions, so the profile does not overclaim reproducibility
        (unlike the pure-code hybrid-rule Redactor). ``memory_mb`` is the approximate resident
        footprint of the checkpoint + transformers stack (declared, not measured).
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
                "https://huggingface.co/urchade/gliner_multi_pii-v1 (weights, Apache-2.0; "
                "base urchade/gliner_multi-v2.1 also Apache-2.0); "
                "https://github.com/urchade/GLiNER/blob/main/LICENSE (library, Apache-2.0)"
            ),
            license_verified_on="2026-07-06",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
            perf_note="bf16 via .to() on GPU / fp32 on CPU (gliner 0.2.27: no FA2/FlashDeBERTa)",
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        pii_labels = config.get("pii_labels")
        if not pii_labels or not isinstance(pii_labels, list):
            msg = (
                "GlinerPIIRedactor requires a non-empty 'pii_labels' list in its config "
                "(zero-shot PII NER: the PII type set is supplied per Pipeline)."
            )
            raise ValueError(msg)
        self._labels = [str(label) for label in pii_labels]
        self._model_id = str(config.get("model", _DEFAULT_MODEL))
        self._threshold = float(config.get("threshold", 0.5))
        self._min_chars = int(config.get("min_chars", 1))
        # W12 model-boundary guard: bound the mdeberta token window (default 768) so a chunk that
        # tokenizes past it is not SILENTLY truncated by gliner (tail PII spans lost / offsets bad).
        self._max_len = resolve_max_len(config)
        self._batch_size = validate_batch_size(
            config.get("batch_size", _DEFAULT_BATCH_SIZE),
            provider_name="redaction.gliner_pii",
        )
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        # The framework owns policy + masking (shared with the in-core rule Provider);
        # this Provider owns only detection — the PIIDetector seam (ADR-0024).
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        # The perf-seam config knobs (dtype / attn_implementation / compile), read at load (W8).
        self._config: dict[str, Any] = dict(config)
        # #191: resolve the device ONCE here (cheap, torch-free routing over the declared profile);
        # ``_load_model`` then PLACES the model on it. Resolving at construction turns an invalid
        # ``device`` string into an operator ConfigError before any weights are downloaded.
        self._device = self._resolve_device(config.get("device", "auto"))
        self._model: Any = None

    @classmethod
    def _resolve_device(cls, requested: object) -> DeviceDecision:
        """Route the ``device`` config value (``auto|cpu|cuda``) through the ADR-0036 device seam.

        ``compute="either"`` (see :meth:`profile`), so ``cpu`` pins CPU, ``auto`` follows
        availability, and ``cuda`` resolves to CUDA when a GPU is present — else the seam returns a
        **skip** decision, which :meth:`_load_model` turns into a loud ConfigError rather than the
        pre-#191 silent CPU fallback. A malformed device string is a ConfigError, not a TypeError.
        """
        try:
            return select_device(str(requested), cls.profile())
        except ValueError as exc:
            msg = f"redaction.gliner_pii: invalid device config: {exc}"
            raise ConfigError(msg) from exc

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        """Load the checkpoint on first use (cached) via the shared load choreography (C4).

        One call. :func:`~latence_core.providers.perf.load_gliner_model` owns the whole ordered
        sequence — skip-decision → JIT kill → intra-op thread cap → backbone import → plain load +
        bf16 cast → placement on the resolved device WITH read-back verification → mdeberta window
        bound → guarded compile — so none of those five "before" constraints is this Provider's to
        remember. They used to be: this body was copy-paste identical across the three plain-gliner
        Providers modulo one string, and #190 (JIT after the import) and #191 (placement wired to
        one Provider of three) are what a hand-copied sequence costs.
        """
        if self._model is None:
            self._model = load_gliner_model(
                _import_gliner,
                self._model_id,
                self._config,
                self._device,
                provider_name="redaction.gliner_pii",
            )
        return self._model

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
            if len(buffer) >= self._batch_size:
                yield from flush()
        yield from flush()

    def _prepare(self, chunk: ChunkRecord) -> _PendingChunk:
        """Resolve everything a chunk needs BEFORE the (possibly batched) model call.

        The :class:`~latence_core.redaction_policy.ChunkRedactionPlan` (policy resolution, the
        near-empty/skip decision, the chunk offset/page maps) and the effective PII label set
        (config ∪ induced, read off the chunk's ``induced_labels``) are per-chunk and
        model-independent, so they are computed here up front; the flush then issues at most one
        model call per same-label group.
        """
        plan = plan_chunk_redaction(
            chunk,
            self._policy,
            detector_types=set(self._labels),
            sensitive_set=set(),
            min_chars=self._min_chars,
        )
        # W4 label-source seam (ADR-0038): UNION the config PII types with any pii_types the
        # OPTIONAL schema_induction Stage induced for this chunk (``induced_labels``), deduped +
        # order-stable. Absent (the default, every existing stack) these are exactly the config
        # labels — byte-identical. The detection LOGIC is unchanged; only where the label list
        # comes from and how many chunks share the forward pass (W7).
        pii_labels = effective_labels(self._labels, chunk.induced_labels, LabelSlot.PII)
        return _PendingChunk(chunk=chunk, text=chunk.content, plan=plan, labels=tuple(pii_labels))

    def _flush_buffer(self, buffer: list[_PendingChunk]) -> Iterator[ChunkRecord]:
        """Batch the detect-needing chunks (grouped by label set), then emit in input order.

        The model is called ONCE per group of same-label detect-needing chunks via
        ``batch_predict_entities`` (``ceil/`` group, not one-per-chunk), and each raw entity list is
        mapped back to its chunk by identity. Chunks that skip / are near-empty need no model call.
        Emission is in the buffer's (input chunk) order regardless of grouping, so the output is
        record-for-record identical to the per-chunk path (the equivalence invariant).
        """
        if not buffer:
            return
        # Group detect-needing chunks by their effective label set (GLiNER's batch call takes ONE
        # label set). Map each pending chunk to its raw entity list by identity so emission stays in
        # input order. Chunks that need no model call get an empty raw list.
        raw_by_chunk: dict[int, list[dict[str, Any]]] = {}
        groups: dict[tuple[str, ...], list[_PendingChunk]] = {}
        for pending in buffer:
            if pending.plan.needs_detection:
                groups.setdefault(pending.labels, []).append(pending)
        if groups:
            model = self._load_model()
            for labels, pendings in groups.items():
                texts = [p.text for p in pendings]
                # ONE forward pass over the group; results[i] is the entity list for texts[i].
                results: list[list[dict[str, Any]]] = model.batch_predict_entities(
                    texts, list(labels), threshold=self._threshold
                )
                for pending, raw in zip(pendings, results, strict=True):
                    raw_by_chunk[id(pending)] = list(raw)
        for pending in buffer:
            raw = raw_by_chunk.get(id(pending), [])
            accepted = self._accepted_from_raw(pending.text, raw, pending.plan.allowed)
            yield finalize_redacted_chunk(pending.chunk, pending.plan, accepted, self._salt)

    @staticmethod
    def _accepted_from_raw(
        text: str,
        raw: list[dict[str, Any]],
        allowed: set[str] | None,
    ) -> list[tuple[int, int, str, str, str, float]]:
        """Build an ordered, non-overlapping candidate list from THIS chunk's raw GLiNER entities.

        GLiNER's ``predict_entities`` / ``batch_predict_entities`` return entity dicts with
        chunk-offset ``start``/``end`` (relative to the text passed in — the chunk ``content``),
        ``label``, and ``score``. A type not in a policy's allow-list is dropped; overlaps are
        resolved (earlier start, longer span, type) so a stable non-overlapping set results — the
        invariant :func:`~latence_core.redaction_policy.finalize_redacted_chunk` masks over. Returns
        ``(local_start, local_end, pii_type, value, detector, confidence)`` tuples (chunk-content
        offsets); the shared helper derives the spans + pages + placeholders.
        """
        candidates: list[tuple[int, int, str, str, float]] = []
        for ent in raw:
            pii_type = str(ent.get("label", ""))
            if not pii_type:
                continue
            if allowed is not None and pii_type not in allowed:
                continue
            start = int(ent["start"])
            end = int(ent["end"])
            if end <= start or start < 0 or end > len(text):
                continue
            score = min(max(float(ent.get("score", 1.0)), 0.0), 1.0)
            candidates.append((start, end, pii_type, text[start:end], score))
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        accepted: list[tuple[int, int, str, str, str, float]] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            accepted.append((start, end, pii_type, value, "gliner-pii", score))
            last_end = end
        return accepted
