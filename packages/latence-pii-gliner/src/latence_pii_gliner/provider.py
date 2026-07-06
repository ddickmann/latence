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

from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, PIISpan, RedactionAction, RedactionRecord
from latence_core.pagemap import PageOffsetIndex
from latence_core.redaction_policy import (
    RedactionPolicy,
    apply_masking,
    build_span,
    parse_action,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord

# Apache-2.0 weights (verified 2026-07-06). Multilingual zero-shot PII NER.
_DEFAULT_MODEL = "urchade/gliner_multi_pii-v1"


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

    The model is loaded lazily and cached on first use, so constructing the Provider
    never downloads weights.
    """

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
        self._salt = str(config.get("salt", "latence"))
        default_action = parse_action(config.get("default_action", "mask"))
        # The framework owns policy + masking (shared with the in-core rule Provider);
        # this Provider owns only detection — the PIIDetector seam (ADR-0024).
        self._policy = RedactionPolicy(config.get("policies"), default_action)
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
        self._model = GLiNER.from_pretrained(self._model_id)
        return self._model

    # -- redaction ----------------------------------------------------------

    def redact(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """Detect PII per document with the model; emit one RedactionRecord per document.

        Each :class:`DocumentRecord` carries the whole document's assembled markdown at
        its original-document offsets plus the page map — the coordinate system the
        masked variant and every :class:`PIISpan` use. A ``PARSE_ERROR`` document (no
        usable content) is skipped. Documents are emitted in first-seen order.
        """
        model = None
        for document in documents:
            if document.disposition is Disposition.PARSE_ERROR:
                continue
            doc_id = document.record_id
            text = document.content
            page_map = document.page_map
            classification = document.classification
            base_prov = document.provenance
            action, allowed, skip, policy_name = self._policy.resolve(classification.sensitivity)

            spans: list[PIISpan] = []
            masked = text
            if not skip and len(text.strip()) >= self._min_chars:
                if model is None:
                    model = self._load_model()
                page_index = PageOffsetIndex(page_map) if page_map is not None else None
                spans = self._detect(model, text, action, allowed, page_index)
                masked = apply_masking(text, spans)

            provenance = base_prov.model_copy(
                update={
                    "char_start": 0,
                    "char_end": len(text),
                    "page_start": None,
                    "page_end": None,
                }
            )
            yield RedactionRecord(
                record_id=f"{doc_id}#redaction",
                document_record_id=doc_id,
                masked_content=masked,
                media_type="text/markdown",
                policy=policy_name,
                pii_spans=spans,
                provenance=provenance,
                classification=classification,
            )

    def _detect(
        self,
        model: Any,
        text: str,
        action: RedactionAction,
        allowed: set[str] | None,
        page_index: PageOffsetIndex | None,
    ) -> list[PIISpan]:
        """Run the model over the whole document text and build non-overlapping PIISpans.

        GLiNER's ``predict_entities`` returns entity dicts with document-offset
        ``start``/``end``, ``label``, and ``score``. A type not in a policy's allow-list
        is dropped; overlaps are resolved (earlier start, longer span, type) so a
        stable, non-overlapping span set results — the invariant the RedactionRecord
        contract enforces and the masking relies on.
        """
        raw: list[dict[str, Any]] = model.predict_entities(
            text, self._labels, threshold=self._threshold
        )
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
