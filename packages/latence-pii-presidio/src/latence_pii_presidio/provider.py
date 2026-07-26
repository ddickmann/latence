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

from latence_core.page_index import PageIndexResolver
from latence_core.providers.adapter import AdapterBase
from latence_core.providers.profile import ProviderProfile
from latence_core.redaction_policy import (
    RedactionPolicy,
    finalize_redacted_chunk,
    parse_action,
    plan_chunk_redaction,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from latence_core.contracts import ChunkRecord

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

    def redact(self, chunks: Iterable[ChunkRecord]) -> Iterator[ChunkRecord]:
        """Detect PII per chunk with Presidio; yield the SAME chunks with PII masked (W16/ADR-0042).

        Each chunk's ``content`` (≤ the model window) is analysed in full — no truncation — and
        the chunk is re-emitted with ``masked_content`` + ``pii_spans`` populated (``content``
        UNCHANGED). Chunks flow through the AdapterBase :meth:`batched` helper (profile
        ``batch=True``) so the corpus is consumed one bounded batch at a time (memory O(batch_size),
        never the whole corpus) and emitted in first-seen order. The analyzer is loaded once on the
        first chunk that needs it (a batch of only clean/skipped chunks never loads spaCy). A
        chunk-local span resolves to its ORIGINAL source page through the chunk's offset_map + the
        document-scoped :class:`~latence_core.page_index.PageIndexResolver` (ADR-0031/0042).

        ONE resolver covers the whole call — across batches: since schema v18 a document's
        ``page_map`` rides its FIRST chunk only, so a per-chunk resolver would resolve pages for
        that chunk and emit ``page_start=None`` on every later one.
        """
        analyzer_state: dict[str, Any] = {"engine": None}
        resolver = PageIndexResolver()

        def run_batch(batch: list[ChunkRecord]) -> list[ChunkRecord]:
            return [self._redact_chunk(chunk, analyzer_state, resolver) for chunk in batch]

        yield from self.batched(chunks, run_batch)

    def _redact_chunk(
        self,
        chunk: ChunkRecord,
        analyzer_state: dict[str, Any],
        resolver: PageIndexResolver,
    ) -> ChunkRecord:
        # The shared plan owns policy resolution + the §4 no-op floor (the Provider's own
        # _detector_types + _sensitive_set feed it) + the chunk offset map. ``resolver`` is the
        # call-scoped page resolver: planning runs for EVERY chunk, so it absorbs the document's
        # map from its first chunk and every later chunk resolves its own source page.
        plan = plan_chunk_redaction(
            chunk,
            self._policy,
            detector_types=self._detector_types,
            sensitive_set=self._sensitive_set,
            min_chars=self._min_chars,
            resolver=resolver,
        )
        accepted: list[tuple[int, int, str, str, str, float]] = []
        if plan.needs_detection:
            if analyzer_state["engine"] is None:
                analyzer_state["engine"] = self._load_analyzer()
            accepted = self._accepted_from_analyzer(
                analyzer_state["engine"], chunk.content, plan.allowed
            )
        return finalize_redacted_chunk(chunk, plan, accepted, self._salt)

    def _accepted_from_analyzer(
        self,
        analyzer: Any,
        text: str,
        allowed: set[str] | None,
    ) -> list[tuple[int, int, str, str, str, float]]:
        """Run Presidio over the chunk text and build an ordered, non-overlapping candidate list.

        ``AnalyzerEngine.analyze`` returns ``RecognizerResult`` objects with chunk-offset
        ``start``/``end`` (relative to the ``text`` analysed — the chunk ``content``), an
        ``entity_type`` (UPPER_SNAKE), and a ``score``. Each Presidio type is mapped to the
        framework's lower-case ``pii_type``; a type not in a policy's allow-list is dropped;
        overlaps are resolved (earlier start, longer span, type) so a stable non-overlapping set
        results.
        Returns ``(local_start, local_end, pii_type, value, detector, confidence)`` tuples
        (chunk-content offsets); :func:`~latence_core.redaction_policy.finalize_redacted_chunk`
        derives the spans + ORIGINAL-resolved pages + placeholders + masking (the ADR-0031/0042 page
        seam, via the chunk's offset_map+page_map, never a weaker mapper).

        The whole Presidio call is inside :meth:`guard`, so a native spaCy/Presidio failure becomes
        a typed :class:`~latence_core.errors.ProviderError` carrying **no source text / PII**.
        """
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

        accepted: list[tuple[int, int, str, str, str, float]] = []
        last_end = -1
        for start, end, pii_type, value, score in candidates:
            if start < last_end:
                continue  # overlaps an accepted span; skip for a clean, ordered set
            accepted.append((start, end, pii_type, value, "presidio", score))
            last_end = end
        return accepted
