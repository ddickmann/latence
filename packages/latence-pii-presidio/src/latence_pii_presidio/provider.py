"""Real Microsoft Presidio PII Redaction Provider over AdapterBase (W1-pii, #95).

The first real *rule+learned* PII Redaction adapter built through
:class:`~latence_core.providers.adapter.AdapterBase` (P3-F2/ADR-0036 §1). Microsoft
Presidio (**MIT**) pairs a **spaCy NER model** with **pattern + context recognizers** — a
genuinely different detection engine from the in-core deterministic regex hybrid
(``redaction.hybrid_rule``) and the zero-shot GLiNER model (``redaction.gliner_pii``), so it
is a real alternative for the bake-off. It runs in-process on a laptop CPU (spaCy inference is
CPU-first, ADR-0007): ``pip install latence-pii-presidio`` + ``python -m spacy download
en_core_web_sm`` → real Presidio redaction, no CUDA.

It fulfils the SAME :class:`~latence_core.capability.PIIDetector` seam as the in-core
``redaction.hybrid_rule`` and the learned ``redaction.gliner_pii``, so it is a **true drop-in**
behind Redaction — a Pipeline swaps ``provider: redaction.hybrid_rule`` for
``provider: redaction.presidio`` with no other config change. All three share the framework's
masking/placeholder/policy machinery (:mod:`latence_core.redaction_policy`) and the H-C1
sensitive-doc no-op floor, so a detected span is masked, replaced, hashed, or tagged
identically whether it came from a regex, a zero-shot model, or Presidio — the only difference
is *detection*. The **raw PII value is never persisted** on a
:class:`~latence_core.contracts.PIISpan` (S6 AC): the surface string is used only transiently
to derive a deterministic surrogate, and it is never interpolated into an error message (C2).

**Offsets are the point (ADR-0031).** Presidio's ``AnalyzerEngine`` returns char ``start``/
``end`` on the text it analysed. Redaction scans the **parsed document** markdown directly
(``record.content`` at its original-document offsets — capability.py:228), so the analyzed text
IS the document text and Presidio's spans ARE document offsets. Each span's source page is
resolved through the AdapterBase :meth:`~latence_core.providers.adapter.AdapterBase.resolve_pages`
seam over the parent :class:`~latence_core.contracts.PageMap` — the ported ``PageOffsetIndex``
asset (never a weaker hand-rolled mapper), so a span straddling a page boundary cites its own
page. Provenance is carried onto the RedactionRecord via
:meth:`~latence_core.providers.adapter.AdapterBase.carry_provenance` (the source lineage intact,
the offset span narrowed to ``[0, len)``). Spans are made **non-overlapping + stable** (the
RedactionRecord invariant): earlier start wins, then longer span, then type name.

**AdapterBase for real.** :meth:`~latence_core.providers.adapter.AdapterBase.guard` maps a
native Presidio/spaCy failure → a typed :class:`~latence_core.errors.ProviderError` carrying
**no detected PII / no source text** (only the provider, the op, and the native exception's
*type name*) — extra care here because this provider handles PII. Batching is over documents
(``batch=True``). ``compute="either"`` (spaCy runs CPU, GPU-optional).

Licensing (ADR-0012, verified 2026-07-08):
    * ``presidio-analyzer`` (PyPI + github.com/microsoft/presidio): **MIT** (code) —
      "Copyright (c) Presidio Contributors".
    * ``spacy`` library (PyPI + github.com/explosion/spaCy): **MIT** (code).
    * Default spaCy model ``en_core_web_sm``: **MIT** (weights; ``en_core_web_lg`` also MIT).
    Both weights AND code are permissive; nothing is downloaded unless a Pipeline actually
    configures this Provider (the analyzer import + the spaCy model load are deferred to first
    use, so importing this module for Provider discovery never needs presidio or a spaCy model).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from latence_core.contracts import Disposition, PIISpan, RedactionAction, RedactionRecord
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    RedactionPolicy,
    parse_action,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import DocumentRecord

# MIT weights (verified 2026-07-08). The light default spaCy model; en_core_web_lg (~500MB, also
# MIT) is a documented config-selectable alternative for higher NER recall.
_DEFAULT_MODEL = "en_core_web_sm"
# The BCP-47 language Presidio analyses in (the default spaCy model is English).
_DEFAULT_LANGUAGE = "en"

# Presidio's built-in entity types → the policy-facing ``pii_type`` label a Pipeline's
# sensitivity policy (``types`` allow-list) and the per-type metric key off. Presidio names its
# entities in UPPER_SNAKE (``EMAIL_ADDRESS``, ``PHONE_NUMBER``); the framework's policy/metric
# vocabulary is the same lower-case type names the in-core hybrid_rule and gliner_pii use
# (``email``, ``phone``, ``person``…), so a ``types: ["email"]`` policy covers a Presidio
# ``EMAIL_ADDRESS`` unchanged and the drop-in needs no policy rewrite. A Presidio type not in
# this map falls back to its own lower-cased name (so a custom recognizer still round-trips).
_PRESIDIO_TYPE_TO_PII_TYPE: dict[str, str] = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "US_SSN": "ssn",
    "CREDIT_CARD": "credit_card",
    "IP_ADDRESS": "ip",
    "IBAN_CODE": "iban",
    "PERSON": "person",
    "LOCATION": "location",
    "ORGANIZATION": "organization",
    "URL": "url",
    "DATE_TIME": "date_time",
    "US_BANK_NUMBER": "bank_account",
    "US_DRIVER_LICENSE": "driver_license",
    "US_PASSPORT": "passport",
    "CRYPTO": "crypto",
    "MEDICAL_LICENSE": "medical_license",
    "NRP": "nationality",
}
# The default Presidio entity set the analyzer is asked for when a Pipeline names none. The
# well-known enterprise PII types the in-core hybrid_rule also covers, so the drop-in default is
# comparable coverage. A Pipeline narrows/extends this via config ``entities``.
_DEFAULT_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "IBAN_CODE",
    "PERSON",
    "LOCATION",
    "URL",
)


def _pii_type_for(entity_type: str) -> str:
    """Map a Presidio entity type to the framework's lower-case ``pii_type`` (fallback: lower)."""
    return _PRESIDIO_TYPE_TO_PII_TYPE.get(entity_type, entity_type.lower())


class PresidioRedactor(AdapterBase):
    """Real Presidio PIIDetector Provider (spaCy NER + pattern/context recognizers), CPU.

    An :class:`~latence_core.providers.adapter.AdapterBase` subclass satisfying the
    :class:`~latence_core.capability.PIIDetector` Protocol (``redact(documents) ->
    Iterator[RedactionRecord]``). It composes the AdapterBase seams: :meth:`device` for
    cpu/cuda placement (spaCy is CPU-first, GPU-optional), :meth:`batched` for input-order
    document batching, :meth:`guard` for native-exception → typed
    :class:`~latence_core.errors.ProviderError` mapping (NO raw text / PII in the message), and
    :meth:`carry_provenance` / :meth:`resolve_pages` for the ADR-0031 offset/page carry.

    Config:
        model: str  — the spaCy model id (default ``en_core_web_sm``, MIT; ``en_core_web_lg``,
            also MIT, ~500MB, is the higher-recall alternative). The model is loaded lazily on
            first ``redact``, so constructing the Provider never loads spaCy.
        language: str  — the BCP-47 language Presidio analyses in (default ``en``).
        entities: list[str]  — the Presidio entity types to detect (default the well-known
            enterprise set: ``EMAIL_ADDRESS``/``PHONE_NUMBER``/``US_SSN``/``CREDIT_CARD``/
            ``IP_ADDRESS``/``IBAN_CODE``/``PERSON``/``LOCATION``/``URL``). Presidio's UPPER_SNAKE
            type names; each is mapped to the framework's lower-case ``pii_type`` for the policy.
        threshold: float  — Presidio's ``score_threshold``; spans scoring below it are dropped
            (default 0.5).
        default_action: str  — action when a policy names none: ``mask`` (default), ``replace``,
            ``hash``, or ``tag``.
        policies: dict[str, dict]  — the sensitivity-keyed policy map (identical shape to the
            in-core Provider): per ``Classification.sensitivity`` value an ``action`` / ``types``
            allow-list (lower-case ``pii_type``) / ``skip`` flag. Keys off the document's
            sensitivity (S6 AC).
        sensitive_set: list[str]  — the ``Classification.sensitivity`` values treated as
            SENSITIVE by the H-C1 §4 no-op floor (default ``["confidential", "restricted"]``).
        salt: str  — salt mixed into ``replace``/``hash`` surrogates (default 'latence').
        min_chars: int  — a document whose assembled text (stripped) is shorter is emitted clean
            (default 1).
        batch_size: int  — documents per batch through AdapterBase (default from AdapterBase);
            purely memory-shaping — output is input-order identical for any positive size.
        device: str  — ``auto|cpu|cuda`` through the shared device seam.
    """

    #: A native ``ValueError``/``KeyError``/``OSError`` from Presidio/spaCy means an operator
    #: handed a bad config (a bad ``model`` id, an unknown entity, a missing spaCy model at load)
    #: → :class:`~latence_core.errors.ConfigError` via :meth:`guard`; everything else caught is a
    #: runtime :class:`~latence_core.errors.ProviderError`.
    config_exceptions = (ValueError, KeyError, OSError)

    @classmethod
    def profile(cls) -> ProviderProfile:
        """The declared descriptor (ADR-0036 §2) for the default ``en_core_web_sm`` model.

        ``compute="either"``: Presidio's spaCy pipeline runs in-process on a laptop CPU (ADR-0007
        — first contact is ``pip install`` + a spaCy model download → real PII NER, no CUDA) and
        can accelerate on a GPU when present, so the device seam routes it rather than skipping it
        on a CPU-only host.

        License is **verified** per ADR-0012, recorded evidence-bearingly in the structured
        fields (H-B1 — the claim is inseparable from its proof): weights and code are recorded
        **separately** (``license_code="MIT"`` for the ``presidio-analyzer`` library and the
        ``spacy`` library; ``license_weights="MIT"`` for the default ``en_core_web_sm`` model),
        with a ``license_source`` citing the GitHub LICENSE + the HF model-card where a human
        checked, and ``license_verified_on="2026-07-08"``. The headline ``license`` is ``"MIT"``
        (Presidio's own license, and the effective license across the stack) and
        ``license_verified=True`` because BOTH sides are permissive.

        ``memory_mb=200``: the approximate resident footprint of Presidio + the small spaCy model
        (declared, not measured — P3-F1 is descriptor-only; ``en_core_web_lg`` would be ~600MB and
        an operator selecting it should raise this expectation). ``deterministic=False``: Presidio
        analysis IS deterministic on a fixed input + config + model version (verified: the spaCy
        pipeline + pattern recognizers are pure functions of the input), but the profile does not
        promise **cross-environment** byte-identity across spaCy/model versions, so it declares
        ``False`` rather than overclaim reproducibility (C4 then records a documented skip — the
        same honest posture the learned embedder + gliner adapters hold). ``batch=True`` (batches
        over documents). ``cost_per_1k=None`` (in-process, no per-call vendor cost).
        """
        return ProviderProfile(
            compute="either",
            memory_mb=200,
            model_id=_DEFAULT_MODEL,
            license="MIT",
            license_verified=True,
            license_code="MIT",
            license_weights="MIT",
            license_source=(
                "https://github.com/microsoft/presidio/blob/main/LICENSE "
                "(presidio-analyzer library, MIT, 'Copyright (c) Presidio Contributors'); "
                "https://github.com/explosion/spaCy/blob/master/LICENSE (spaCy library, MIT); "
                "https://huggingface.co/spacy/en_core_web_sm (model weights, MIT; "
                "en_core_web_lg also MIT)"
            ),
            license_verified_on="2026-07-08",
            deterministic=False,
            batch=True,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = self.config
        self._model_id = str(cfg.get("model", _DEFAULT_MODEL))
        self._language = str(cfg.get("language", _DEFAULT_LANGUAGE))
        self._threshold = float(cfg.get("threshold", 0.5))
        self._min_chars = int(cfg.get("min_chars", 1))
        self._salt = str(cfg.get("salt", "latence"))

        entities = cfg.get("entities")
        if entities is not None and not isinstance(entities, list):
            msg = (
                f"{self.provider_name}: 'entities' config must be a list of Presidio entity types."
            )
            raise ValueError(msg)
        # The Presidio entity types the analyzer is asked for (UPPER_SNAKE). Defaults to the
        # well-known enterprise set; a Pipeline narrows/extends it.
        self._entities: list[str] = (
            [str(e) for e in entities] if entities is not None else list(_DEFAULT_ENTITIES)
        )

        # The framework owns policy + masking (shared with the in-core rule Provider + gliner);
        # this Provider owns only detection — the PIIDetector seam (ADR-0024). Policy narrows on
        # the lower-case ``pii_type`` the Presidio types map to.
        default_action = parse_action(cfg.get("default_action", "mask"))
        self._policy = RedactionPolicy(cfg.get("policies"), default_action)

        # The set of ``pii_type`` values some enabled entity can produce — the coverage the applied
        # policy narrows. Used by the H-C1 §4 no-op floor to tell "control ran no detectors"
        # (empty intersection with the policy allow-list) apart from "genuinely no PII found".
        self._detector_types = {_pii_type_for(e) for e in self._entities}

        # The Classification.sensitivity values considered SENSITIVE for the §4 no-op floor.
        # Configurable; default confidential/restricted (identical to the in-core Provider so the
        # drop-in floor behaves the same). A disabled control on one of these surfaces as a loud,
        # first-class signal rather than a silent no-op.
        sensitive = cfg.get("sensitive_set", ("confidential", "restricted"))
        if not isinstance(sensitive, (list, tuple, set)):
            msg = (
                f"{self.provider_name}: 'sensitive_set' config must be a list of "
                "Classification.sensitivity values."
            )
            raise ValueError(msg)
        self._sensitive_set = {str(s) for s in sensitive}

        self._analyzer: Any = None

    # -- lazy analyzer load -------------------------------------------------

    def _load_analyzer(self) -> Any:
        """Import Presidio and build the ``AnalyzerEngine`` on first use (cached).

        Deferred so Provider discovery (importing this module) never needs presidio or a spaCy
        model, and a missing optional dependency surfaces as a clear install hint only when the
        Provider is actually run. A missing/uninstalled spaCy model surfaces at engine construction
        as an ``OSError`` from spaCy — mapped to a :class:`~latence_core.errors.ConfigError` (an
        operator config gap: "download the model") by :meth:`guard`, never a silent wrong result.
        """
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            msg = (
                "The 'presidio-analyzer' package is required to run PresidioRedactor. "
                "Install it with: pip install latence-pii-presidio  (and a spaCy model, e.g. "
                "python -m spacy download en_core_web_sm)."
            )
            raise ImportError(msg) from exc
        with self.guard("load_model"):
            nlp_engine = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": self._language, "model_name": self._model_id}],
                }
            ).create_engine()
            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=[self._language]
            )
        return self._analyzer

    # -- redaction ----------------------------------------------------------

    def redact(self, documents: Iterable[DocumentRecord]) -> Iterator[RedactionRecord]:
        """Detect PII per document with Presidio; emit one RedactionRecord per document.

        Each :class:`DocumentRecord` carries the whole document's assembled markdown at its
        original-document offsets plus the page map — the coordinate system the masked variant and
        every :class:`PIISpan` use. A ``PARSE_ERROR`` document (no usable content) is skipped,
        mirroring Chunk. Documents flow through the AdapterBase :meth:`batched` helper (profile
        ``batch=True``) so the corpus is consumed one bounded batch at a time (memory O(batch_size),
        never the whole corpus) and emitted in first-seen order. The analyzer is loaded once on the
        first document that needs it (a batch of only clean/skipped docs never loads spaCy).
        """
        analyzer_state: dict[str, Any] = {"engine": None}

        def run_batch(batch: list[DocumentRecord]) -> list[RedactionRecord]:
            out: list[RedactionRecord] = []
            for document in batch:
                if document.disposition is Disposition.PARSE_ERROR:
                    continue
                out.append(self._redact_one(document, analyzer_state))
            return out

        yield from self.batched(documents, run_batch)

    def _redact_one(
        self, document: DocumentRecord, analyzer_state: dict[str, Any]
    ) -> RedactionRecord:
        doc_id = document.record_id
        text = document.content
        classification = document.classification
        action, allowed, skip, policy_name = self._policy.resolve(classification.sensitivity)

        # §4 no-op floor (H-C1): would the applied policy run ANY detector on this document? No, if
        # the policy skips, or restricts to a type set that intersects none of the enabled entity
        # types (including an empty ``types: []``), or the Provider has no entities at all. When the
        # document is also classified sensitive, an empty pii_spans there is "control OFF", not
        # "clean" — a loud, first-class signal. Reuses the core policy/flag machinery; does not
        # reinvent it.
        would_run = (not skip) and bool(
            self._detector_types if allowed is None else self._detector_types & allowed
        )
        disabled_for_sensitive = (
            not would_run and classification.sensitivity in self._sensitive_set
        )

        spans: list[PIISpan] = []
        masked = text
        if not skip and len(text.strip()) >= self._min_chars:
            if analyzer_state["engine"] is None:
                analyzer_state["engine"] = self._load_analyzer()
            spans = self._detect(analyzer_state["engine"], document, action, allowed)
            masked = _apply_masking(text, spans)

        # Carry the source Provenance/Classification via the AdapterBase seam (H-A1): the source
        # lineage (source_uri/document_id/file_*) intact, the offset span narrowed to the document
        # extent [0, len). A RedactionRecord covers the WHOLE document, so its own page range is
        # undefined (per-span pages are resolved on each PIISpan) — hybrid_rule sets it None, so a
        # true drop-in must too. NOTE: carry_provenance only applies non-None updates (it cannot
        # *clear* a field), so passing page_start=None there is a silent no-op that would leave the
        # source doc's stale page range on the record; clear it explicitly for hybrid_rule parity.
        # (AdapterBase gap in the seam-report: carry_provenance can set but not null a field.)
        provenance, carried_classification = self.carry_provenance(
            document, char_start=0, char_end=len(text)
        )
        provenance = provenance.model_copy(update={"page_start": None, "page_end": None})
        return RedactionRecord(
            record_id=f"{doc_id}#redaction",
            document_record_id=doc_id,
            masked_content=masked,
            media_type="text/markdown",
            policy=policy_name,
            pii_spans=spans,
            redaction_disabled_for_sensitive=disabled_for_sensitive,
            provenance=provenance,
            classification=carried_classification,
        )

    def _detect(
        self,
        analyzer: Any,
        document: DocumentRecord,
        action: RedactionAction,
        allowed: set[str] | None,
    ) -> list[PIISpan]:
        """Run Presidio over the whole document text and build non-overlapping PIISpans.

        ``AnalyzerEngine.analyze`` returns ``RecognizerResult`` objects with document-offset
        ``start``/``end``, an ``entity_type`` (UPPER_SNAKE), and a ``score``. Because Redaction
        scans the parsed document markdown directly, the analyzed text IS the document text and
        Presidio's offsets ARE document offsets. Each Presidio type is mapped to the framework's
        lower-case ``pii_type``; a type not in a policy's allow-list is dropped; overlaps are
        resolved (earlier start, longer span, type) so a stable, non-overlapping span set results —
        the invariant the RedactionRecord contract enforces and the masking relies on. Each span's
        page is resolved through the AdapterBase ADR-0031 page seam (never a weaker mapper).

        The whole Presidio call is inside :meth:`guard`, so a native spaCy/Presidio failure becomes
        a typed :class:`~latence_core.errors.ProviderError` carrying **no source text / PII**.
        """
        text = document.content
        with self.guard("analyze"):
            results: list[Any] = analyzer.analyze(
                text=text,
                language=self._language,
                entities=self._entities,
                score_threshold=self._threshold,
            )

        # (start, end, pii_type, value, score) tuples, filtered + sorted deterministically.
        candidates: list[tuple[int, int, str, str, float]] = []
        for res in results:
            pii_type = _pii_type_for(str(res.entity_type))
            if allowed is not None and pii_type not in allowed:
                continue
            start = int(res.start)
            end = int(res.end)
            if start < 0 or end <= start or end > len(text):
                continue
            score = min(max(float(res.score), 0.0), 1.0)
            candidates.append((start, end, pii_type, text[start:end], score))
        # Deterministic order: earlier start, then longer span, then type name — so a stable,
        # non-overlapping set results regardless of Presidio's internal recognizer order.
        candidates.sort(key=lambda c: (c[0], -(c[1] - c[0]), c[2]))

        page_map = document.page_map
        base_prov = document.provenance
        spans: list[PIISpan] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            # Resolve the source page(s) through the ADR-0031 seam (PageOffsetIndex) — never a
            # weaker hand-rolled mapper. When the document carries no page map, fall back to the
            # parent's page range (the same fallback the in-core path uses).
            page_start, page_end = self.resolve_pages(
                start,
                end,
                page_map=page_map,
                fallback_page_start=base_prov.page_start,
                fallback_page_end=base_prov.page_end,
            )
            spans.append(
                _build_span_with_pages(
                    pii_type=pii_type,
                    detector="presidio",
                    start=start,
                    end=end,
                    value=value,
                    action=action,
                    salt=self._salt,
                    confidence=score,
                    page_start=page_start,
                    page_end=page_end,
                )
            )
            last_end = end
        return spans


def _apply_masking(text: str, spans: list[PIISpan]) -> str:
    """Rewrite ``text`` replacing each non-TAG span with its placeholder (right-to-left).

    Thin wrapper over :func:`latence_core.redaction_policy.apply_masking` — imported here so the
    module reads as one masking call site; the shared implementation is the single source of truth
    (no divergence from the in-core Provider's masking).
    """
    from latence_core.redaction_policy import apply_masking

    return apply_masking(text, spans)


def _build_span_with_pages(
    *,
    pii_type: str,
    detector: str,
    start: int,
    end: int,
    value: str,
    action: RedactionAction,
    salt: str,
    confidence: float,
    page_start: int | None,
    page_end: int | None,
) -> PIISpan:
    """Build one PIISpan reusing the shared placeholder derivation + the ADR-0031-resolved pages.

    :func:`latence_core.redaction_policy.build_span` derives the deterministic placeholder from the
    value (which it never persists) but resolves pages from a ``PageOffsetIndex`` it is handed. This
    Provider already resolved the span's page(s) through the AdapterBase
    :meth:`~latence_core.providers.adapter.AdapterBase.resolve_pages` seam (so the mapping matches
    every other adapter), so it derives the placeholder via the shared helper and stamps the
    already-resolved pages, rather than re-resolving. The raw ``value`` is used only to derive the
    surrogate and is never stored on the span (S6 AC).
    """
    from latence_core.redaction_policy import placeholder_for

    placeholder = placeholder_for(pii_type, value, action, salt)
    return PIISpan(
        pii_type=pii_type,
        detector=detector,
        char_start=start,
        char_end=end,
        action=action,
        placeholder=placeholder,
        page_start=page_start,
        page_end=page_end,
        confidence=confidence,
    )
