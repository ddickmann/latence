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

from latence_core.contracts import Disposition, PIISpan, RedactionAction, RedactionRecord
from latence_core.induced import LabelSlot, effective_labels
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.adapter import validate_batch_size
from latence_core.providers.perf import (
    apply_gliner_max_len,
    load_then_cast,
    maybe_compile,
    resolve_max_len,
)
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    RedactionPolicy,
    apply_masking,
    build_span,
    parse_action,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord


@dataclass(frozen=True)
class _PendingDoc:
    """A document's per-document, model-independent context, resolved before the batched call (W7).

    Everything the detection + record building needs that does NOT depend on the model output is
    computed once up front (policy action/allow-list, the skip/near-empty decision, the page index,
    the effective PII label set), so a flush issues at most one ``batch_predict_entities`` call per
    same-label group and maps each raw entity list back to its document by identity — keeping
    emission in input-document order and the output record-for-record identical to the per-doc path.
    """

    document: DocumentRecord
    text: str
    action: RedactionAction
    allowed: set[str] | None
    page_index: PageOffsetIndex | None
    policy_name: str
    needs_model: bool
    labels: tuple[str, ...]

# Apache-2.0 weights (verified 2026-07-06). Multilingual zero-shot PII NER.
_DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"

# The default number of documents sharing ONE ``batch_predict_entities`` forward pass (W7). A
# sensible GPU value (the P8 autotuner overrides per-device); purely memory-shaping — the emitted
# PIISpans/masking are record-for-record identical to a per-document loop for any positive batch
# size (ADR-0033: the in-flight tensor is bounded to ``batch_size``, never the whole corpus).
_DEFAULT_BATCH_SIZE = 16


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
        self._model: Any = None

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner' package is required to run GlinerPIIRedactor. "
                "Install it with: pip install latence-pii-gliner"
            )
            raise ImportError(msg) from exc
        # W8 perf seam (pod-corrected): plain ``gliner`` 0.2.27 REJECTS torch_dtype/attn_impl in
        # in from_pretrained and its encoder has no FA2/FlashDeBERTa — so load PLAIN, then .to(bf16)
        # on cuda (measured 1.36× faster, byte-identical). device=None reads the torch-free probe
        # (GLiNER auto-places on cuda); the cast is a no-op on CPU / fp32. Then optionally compile.
        model = load_then_cast(GLiNER.from_pretrained, self._model_id, self._config, None)
        # W12: bound the mdeberta token window BEFORE any compile wrap (the hard model-boundary
        # guard) so gliner never silently truncates a long document past config.max_len.
        apply_gliner_max_len(model, self._max_len)
        self._model = maybe_compile(model, self._config, provider_name="redaction.gliner_pii")
        return self._model

    # -- redaction ----------------------------------------------------------

    def redact(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """Detect PII per document with the model; emit one RedactionRecord per document.

        Each :class:`DocumentRecord` carries the whole document's assembled markdown at
        its original-document offsets plus the page map — the coordinate system the
        masked variant and every :class:`PIISpan` use. A ``PARSE_ERROR`` document (no
        usable content) is skipped. Documents are emitted in first-seen order.
        """
        buffer: list[_PendingDoc] = []

        def flush() -> Iterator[RedactionRecord]:
            yield from self._flush_buffer(buffer)
            buffer.clear()

        for document in documents:
            if document.disposition is Disposition.PARSE_ERROR:
                continue
            buffer.append(self._prepare(document))
            if len(buffer) >= self._batch_size:
                yield from flush()
        yield from flush()

    def _prepare(self, document: DocumentRecord) -> _PendingDoc:
        """Resolve everything a document needs BEFORE the (possibly batched) model call.

        Policy resolution, the near-empty/skip decision, the page index, and the effective PII
        label set (config ∪ induced) are all per-document and model-independent, so they are
        computed here up front; the flush then issues at most one model call per same-label group.
        """
        classification = document.classification
        text = document.content
        action, allowed, skip, policy_name = self._policy.resolve(classification.sensitivity)
        needs_model = not skip and len(text.strip()) >= self._min_chars
        page_index = (
            PageOffsetIndex(document.page_map) if document.page_map is not None else None
        )
        # W4 label-source seam (ADR-0038): UNION the config PII types with any pii_types the
        # OPTIONAL schema_induction Stage induced for this document (``induced_labels``), deduped +
        # order-stable. Absent (the default, every existing stack) these are exactly the config
        # labels — byte-identical. The detection LOGIC is unchanged; only where the label list
        # comes from and how many documents share the forward pass (W7).
        pii_labels = effective_labels(self._labels, document.induced_labels, LabelSlot.PII)
        return _PendingDoc(
            document=document,
            text=text,
            action=action,
            allowed=allowed,
            page_index=page_index,
            policy_name=policy_name,
            needs_model=needs_model,
            labels=tuple(pii_labels),
        )

    def _flush_buffer(self, buffer: list[_PendingDoc]) -> Iterator[RedactionRecord]:
        """Batch the detect-needing docs (grouped by label set), then emit records in input order.

        The model is called ONCE per group of same-label detect-needing documents via
        ``batch_predict_entities`` (``ceil/`` group, not one-per-doc), and each raw entity list is
        mapped back to its document by index. Documents that skip / are near-empty need no model
        call. Emission is in the buffer's (input document) order regardless of grouping, so the
        output is record-for-record identical to the per-document path (the equivalence invariant).
        """
        if not buffer:
            return
        # Group detect-needing docs by their effective label set (GLiNER's batch call takes ONE
        # label set). Map each pending doc to its raw entity list by identity so emission stays in
        # input order. Docs that need no model call get an empty raw list.
        raw_by_doc: dict[int, list[dict[str, Any]]] = {}
        groups: dict[tuple[str, ...], list[_PendingDoc]] = {}
        for pending in buffer:
            if pending.needs_model:
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
                    raw_by_doc[id(pending)] = list(raw)
        for pending in buffer:
            raw = raw_by_doc.get(id(pending), [])
            spans = self._spans_from_raw(
                pending.text, raw, pending.action, pending.allowed, pending.page_index
            )
            masked = apply_masking(pending.text, spans) if spans else pending.text
            doc_id = pending.document.record_id
            provenance = pending.document.provenance.model_copy(
                update={
                    "char_start": 0,
                    "char_end": len(pending.text),
                    "page_start": None,
                    "page_end": None,
                }
            )
            yield RedactionRecord(
                record_id=f"{doc_id}#redaction",
                document_record_id=doc_id,
                masked_content=masked,
                media_type="text/markdown",
                policy=pending.policy_name,
                pii_spans=spans,
                provenance=provenance,
                classification=pending.document.classification,
            )

    def _spans_from_raw(
        self,
        text: str,
        raw: list[dict[str, Any]],
        action: RedactionAction,
        allowed: set[str] | None,
        page_index: PageOffsetIndex | None,
    ) -> list[PIISpan]:
        """Build non-overlapping PIISpans from THIS document's raw GLiNER entity list.

        GLiNER's ``predict_entities`` / ``batch_predict_entities`` return entity dicts with
        document-offset ``start``/``end``, ``label``, and ``score``; ``raw`` is this document's list
        (mapped back by index from the batched call, W7). A type not in a policy's allow-list is
        dropped; overlaps are resolved (earlier start, longer span, type) so a stable,
        non-overlapping span set results — the invariant the RedactionRecord contract enforces and
        the masking relies on. This logic is UNCHANGED from the per-document path.
        """
        # (start, end, type, value, score) tuples, filtered + sorted deterministically.
        candidates: list[tuple[int, int, str, str, float]] = []
        for ent in raw:
            pii_type = str(ent.get("label", ""))
            if not pii_type:
                continue
            if allowed is not None and pii_type not in allowed:
                continue
            start = int(ent["start"])
            end = int(ent["end"])
            if end <= start:
                continue
            score = min(max(float(ent.get("score", 1.0)), 0.0), 1.0)
            candidates.append((start, end, pii_type, text[start:end], score))
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        spans: list[PIISpan] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            spans.append(
                build_span(
                    pii_type=pii_type,
                    detector="gliner-pii",
                    start=start,
                    end=end,
                    value=value,
                    action=action,
                    salt=self._salt,
                    page_index=page_index,
                    confidence=score,
                )
            )
            last_end = end
        return spans
