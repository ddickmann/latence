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
    model = GLiNER2.from_pretrained("fastino/GLiNER2-Guardrails-PII-Multi")
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, PIISpan, RedactionAction, RedactionRecord
from latence_core.pagemap import PageOffsetIndex
from latence_core.providers.adapter import AdapterBase
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

    Policy action/allow-list, the skip/near-empty decision, and the page index are per-document and
    model-independent, so they are computed once up front; a flush then issues at most one
    ``batch_extract_entities`` call over the detect-needing docs and maps each per-text result back
    to its document by identity, keeping emission in input-document order and the output
    record-for-record identical to the per-document path.
    """

    document: DocumentRecord
    text: str
    action: RedactionAction
    allowed: set[str] | None
    page_index: PageOffsetIndex | None
    policy_name: str
    needs_model: bool

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
        batch_size: int  — how many documents share ONE ``batch_extract_entities`` forward pass
            (default from AdapterBase; the P8 autotuner overrides per-device). Purely
            memory-shaping: the emitted RedactionRecords (PIISpans/masking/order) are
            record-for-record identical for any positive batch size (W7). Must be ``>= 1``.
        device: str  — ``auto`` (default) / ``cpu`` / ``cuda``; routed by the device seam.

    The model is loaded lazily and cached on first use, so constructing the Provider never
    downloads weights.

    Batching (W7, ADR-0036 §1 / ADR-0033): GLiNER2 (mdeberta-v3) is an ENCODER, so a per-document
    ``extract_entities`` loop under-feeds the GPU. This Provider buffers up to ``batch_size``
    documents and, on flush, runs ONE ``model.batch_extract_entities(texts, labels, threshold=…,
    include_spans=True, include_confidence=True)`` forward pass over the detect-needing docs (the
    batched entry point — ``hasattr``-guarded here and pod-Gate-0 asserted; offline-unverifiable so
    not yet pod-confirmed) — ``ceil(N/batch_size)`` calls, not N — mapping each per-text
    result back to its document **by index**. Documents that a policy SKIPS or that are near-empty
    need no model call and are emitted clean, in input order. The per-document policy resolution,
    span building, masking, and Provenance carry are UNCHANGED; the in-flight buffer is bounded to
    ``batch_size`` (never the whole corpus, ADR-0033).
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
        # Validate the batch_size knob eagerly (fail-fast at construction like the twin providers)
        # even though the batched call reads it lazily via AdapterBase.batch_size() (W7).
        self.batch_size()
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        # The framework owns policy + masking (shared with the in-core rule Provider and the
        # twin gliner Provider); this Provider owns only detection — the PIIDetector seam.
        self._policy = RedactionPolicy(config.get("policies"), default_action)
        self._model: Any = None

    @property
    def provider_name(self) -> str:
        """The registered entry-point name, used only to tag typed errors (never carries PII)."""
        return "redaction.gliner2"

    # -- lazy model load ----------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from gliner2 import GLiNER2
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'gliner2' package is required to run Gliner2PIIRedactor. "
                "Install it with: pip install latence-pii-gliner2"
            )
            raise ImportError(msg) from exc
        with self.guard("load_model"):
            self._model = GLiNER2.from_pretrained(self._model_id)
        return self._model

    # -- redaction ----------------------------------------------------------

    def redact(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """Detect PII per document with the model; emit one RedactionRecord per document.

        Each :class:`DocumentRecord` carries the whole document's assembled markdown at its
        original-document offsets plus the page map — the coordinate system the masked variant
        and every :class:`PIISpan` use. A ``PARSE_ERROR`` document (no usable content) is
        skipped. Documents are emitted in first-seen order.
        """
        buffer: list[_PendingDoc] = []

        def flush() -> Iterator[RedactionRecord]:
            yield from self._flush_buffer(buffer)
            buffer.clear()

        for document in documents:
            if document.disposition is Disposition.PARSE_ERROR:
                continue
            buffer.append(self._prepare(document))
            if len(buffer) >= self.batch_size():
                yield from flush()
        yield from flush()

    def _prepare(self, document: DocumentRecord) -> _PendingDoc:
        """Resolve everything a document needs BEFORE the (possibly batched) model call (W7).

        Policy resolution, the near-empty/skip decision, and the page index are per-document and
        model-independent, so they are computed here up front; the flush then issues at most one
        batched model call over the detect-needing docs.
        """
        classification = document.classification
        text = document.content
        action, allowed, skip, policy_name = self._policy.resolve(classification.sensitivity)
        needs_model = not skip and len(text.strip()) >= self._min_chars
        page_index = (
            PageOffsetIndex(document.page_map) if document.page_map is not None else None
        )
        return _PendingDoc(
            document=document,
            text=text,
            action=action,
            allowed=allowed,
            page_index=page_index,
            policy_name=policy_name,
            needs_model=needs_model,
        )

    def _flush_buffer(self, buffer: list[_PendingDoc]) -> Iterator[RedactionRecord]:
        """Batch the detect-needing docs through ONE call, then emit records in input order (W7).

        GLiNER2 shares one label set (this Provider has no per-document induced-label seam), so all
        detect-needing docs go through a single ``batch_extract_entities`` forward pass; each
        per-text result maps back to its document by index. Docs that skip / are near-empty need no
        model call. Emission is in the buffer's (input) order regardless, so the output is
        record-for-record identical to the per-document path (the equivalence invariant).
        """
        if not buffer:
            return
        raw_by_doc: dict[int, Any] = {}
        detect = [p for p in buffer if p.needs_model]
        if detect:
            model = self._load_model()
            texts = [p.text for p in detect]
            # The batched entry point is offline-unverifiable (the fake mirrors it); guard against
            # a renamed/missing method with a clear, actionable error rather than a bare
            # AttributeError (the guard would otherwise surface only the op label). The pod script's
            # Gate 0 asserts this method EXISTS on the real checkpoint before any throughput number.
            if not hasattr(model, "batch_extract_entities"):
                msg = (
                    "redaction.gliner2 batches via model.batch_extract_entities(texts, "
                    "entity_types, threshold=…), but the loaded GLiNER2 checkpoint exposes no "
                    "'batch_extract_entities' method. Re-introspect the checkpoint and fix the "
                    "batched call. See scripts/validate-encoder-batching-on-gpu.md Gate 0."
                )
                raise AttributeError(msg)
            with self.guard("batch_extract_entities"):
                # ONE forward pass over the group; results[i] is the per-label dict for texts[i].
                results: Any = model.batch_extract_entities(
                    texts,
                    self._labels,
                    threshold=self._threshold,
                    include_spans=True,
                    include_confidence=True,
                )
            results_list = list(results)
            if len(results_list) != len(detect):
                msg = (
                    "redaction.gliner2: batch_extract_entities returned "
                    f"{len(results_list)} results for a batch of {len(detect)} texts."
                )
                raise ValueError(msg)
            for pending, raw in zip(detect, results_list, strict=True):
                raw_by_doc[id(pending)] = raw
        for pending in buffer:
            spans: list[PIISpan] = []
            if pending.needs_model:
                spans = self._spans_from_raw(
                    pending.text,
                    raw_by_doc[id(pending)],
                    pending.action,
                    pending.allowed,
                    pending.page_index,
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
        raw: Any,
        action: RedactionAction,
        allowed: set[str] | None,
        page_index: PageOffsetIndex | None,
    ) -> list[PIISpan]:
        """Build non-overlapping PIISpans from THIS document's raw GLiNER2 output (mapped by index).

        ``gliner2``'s ``extract_entities`` / ``batch_extract_entities`` return a **per-label dict**
        of dict-rows: ``{"entities": {"<label>": [{"text", "confidence", "start", "end"}, ...],
        ...}}`` — a DIFFERENT shape from the ``gliner`` library's flat list. ``raw`` is this
        document's result (mapped back by index from the batched call, W7). This flattens that
        per-label dict → one candidate per row (``pii_type=<label>``, ``char_start=start``,
        ``char_end=end``, ``confidence=confidence``), then reuses the twin's span-building shape: a
        type not in a policy's allow-list is dropped; overlaps are resolved (earlier start, longer
        span, type) so a stable, deterministic, non-overlapping span set results — the invariant the
        RedactionRecord contract enforces and the masking relies on. This logic is UNCHANGED.
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

        spans: list[PIISpan] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            spans.append(
                build_span(
                    pii_type=pii_type,
                    detector=f"gliner2:{pii_type}",
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
