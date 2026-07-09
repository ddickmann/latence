"""Deterministic stubs for the endpoint / GPU-model Providers (P3-F3 §3, honestly bounded).

Endpoint Providers (``*.endpoint``, ``relation.llm``, ``embedding.endpoint``) and GPU model
Providers (``*.gliner``) cannot run their real backend in the CPU-first, offline test env
(ADR-0007/0016) — the ``openai`` client and ``torch``/``gliner`` are not installed. Their real
quality/throughput is explicitly **out of scope** (needs a rig, #64); conformance verifies the
**contract / graceful-failure / license / determinism-declaration** subset against a
deterministic stub — a fake backend returning a canned typed response — reusing the existing
endpoint/gliner test-double pattern (a fake ``openai`` / ``gliner`` module injected into
``sys.modules``).

Each ``install_*`` returns a context manager that installs the fake module for the duration of
one Provider invocation and removes it after, so the stub never leaks across cases. The canned
responses are keyed to the canonical fixture text (:mod:`~latence_core.conformance.fixtures`) so
the produced records carry real, resolvable offsets — a genuine contract check, not a no-op.

The harness flags every stub-scoped Provider ``conformance-verified against stub; live numbers
need a rig`` in its result, so a reader never mistakes a stub-pass for a live quality number.
"""

from __future__ import annotations

import contextlib
import json
import sys
import types
from collections.abc import Iterator
from typing import Any

# ---------------------------------------------------------------------------
# openai chat/embeddings stub (the endpoint Providers)
# ---------------------------------------------------------------------------


class _StubCompletions:
    """A fake ``chat.completions`` returning a canned reply set by the installer.

    ``finish_reason`` is set on every returned choice so a Provider that DETECTS truncation (the W4
    schema-inducer checks ``finish_reason == "length"``) sees a faithful field — a Provider that
    ignores it (the pre-W4 endpoint Providers) is unaffected, since the real OpenAI response carries
    it too. Default ``"stop"`` (a complete reply), matching the real API.
    """

    reply: str = "{}"
    finish_reason: str = "stop"

    def create(self, **_kwargs: Any) -> Any:
        message = types.SimpleNamespace(content=type(self).reply)
        choice = types.SimpleNamespace(message=message, finish_reason=type(self).finish_reason)
        return types.SimpleNamespace(choices=[choice])


class _StubChat:
    def __init__(self) -> None:
        self.completions = _StubCompletions()


class _StubEmbeddings:
    """A fake ``embeddings`` returning fixed-width deterministic vectors in input order."""

    dimension: int = 8

    def create(self, **kwargs: Any) -> Any:
        inputs = list(kwargs.get("input", []))
        data = []
        for i, _text in enumerate(inputs):
            # A deterministic, fixed-width vector (not semantic — the stub only proves the
            # contract: a uniform-width vector per input, in order).
            vector = [float((i + j) % 3) for j in range(type(self).dimension)]
            data.append(types.SimpleNamespace(index=i, embedding=vector))
        return types.SimpleNamespace(data=data)


class _StubOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        self.chat = _StubChat()
        self.embeddings = _StubEmbeddings()


@contextlib.contextmanager
def install_openai(
    *, reply: str = "{}", dimension: int = 8, finish_reason: str = "stop"
) -> Iterator[None]:
    """Install a fake ``openai`` module returning the given canned chat reply / embedding width.

    ``finish_reason`` (default ``"stop"``) is stamped on every returned chat choice so a Provider
    that checks it for truncation (W4) is exercised faithfully.
    """
    _StubCompletions.reply = reply
    _StubCompletions.finish_reason = finish_reason
    _StubEmbeddings.dimension = dimension
    module = types.ModuleType("openai")
    module.OpenAI = _StubOpenAI  # type: ignore[attr-defined]
    saved = sys.modules.get("openai")
    sys.modules["openai"] = module
    try:
        yield
    finally:
        _StubCompletions.finish_reason = "stop"
        if saved is not None:
            sys.modules["openai"] = saved
        else:
            sys.modules.pop("openai", None)


# ---------------------------------------------------------------------------
# gliner stub (the GPU model Providers)
# ---------------------------------------------------------------------------


class _StubGLiNER:
    """Stands in for ``gliner.GLiNER``: returns fixed chunk-local spans + a canned relation."""

    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_id: str) -> _StubGLiNER:  # noqa: ARG003 - model id unused offline
        return cls()

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        return [
            e
            for e in type(self).entities
            if e["label"] in labels and e["score"] >= threshold and e["end"] <= len(text)
        ]

    def predict_relations(
        self,
        text: str,
        labels: list[str],
        relation_labels: list[str],  # noqa: ARG002 - accepted for API parity
        threshold: float = 0.5,
        **kwargs: Any,  # noqa: ARG002 - flat_ner/adjacency_threshold/... accepted for parity
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        # Faithful to the REAL knowledgator/gliner-relex predict_relations TUPLE contract
        # (W1-relex FIX): returns ``(entities, relations)``, NOT a dict. A provider that
        # still calls ``predict_with_relations`` (which never existed) or does
        # ``result.get(...)`` on this tuple FAILS — the false-green trap this closes.
        ents = self.predict_entities(text, labels, threshold)
        rels = [r for r in type(self).relations if r.get("score", 1.0) >= threshold]
        return ents, rels


@contextlib.contextmanager
def install_gliner(
    *,
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Install a fake ``gliner`` module returning the canned chunk-local entities/relations."""
    _StubGLiNER.entities = entities or []
    _StubGLiNER.relations = relations or []
    module = types.ModuleType("gliner")
    module.GLiNER = _StubGLiNER  # type: ignore[attr-defined]
    saved = sys.modules.get("gliner")
    sys.modules["gliner"] = module
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["gliner"] = saved
        else:
            sys.modules.pop("gliner", None)


# ---------------------------------------------------------------------------
# gliner2 stub (the fastino GLiNER2 PII Redaction Provider, redaction.gliner2)
# ---------------------------------------------------------------------------


class _StubGLiNER2:
    """Stands in for ``gliner2.GLiNER2``: the REAL per-label-dict ``extract_entities`` shape.

    Faithful to the fastino GLiNER2 framework's ``extract_entities(text, entity_types,
    threshold, include_spans=True, include_confidence=True)`` →
    ``{"entities": {label: [{text, confidence, start, end}], ...}}`` (a per-label dict of
    dict-rows), the DIFFERENT shape from the ``gliner`` library's flat list. A provider that
    mis-parses it as a flat list recovers ZERO spans — the false-green trap the anchor test
    closes. ``entities`` is seeded as flat rows keyed to the canonical fixture; ``extract_entities``
    filters by label + threshold and nests them per-label, as the real model does.
    """

    entities: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_id: str) -> _StubGLiNER2:  # noqa: ARG003 - id unused offline
        return cls()

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.5,
        include_spans: bool = False,  # noqa: ARG002 - accepted for API parity
        include_confidence: bool = False,  # noqa: ARG002 - accepted for API parity
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        by_label: dict[str, list[dict[str, Any]]] = {}
        for e in type(self).entities:
            label = e["label"]
            if label not in entity_types or e["confidence"] < threshold or e["end"] > len(text):
                continue
            by_label.setdefault(label, []).append(
                {
                    "text": text[e["start"] : e["end"]],
                    "confidence": e["confidence"],
                    "start": e["start"],
                    "end": e["end"],
                }
            )
        return {"entities": by_label}


@contextlib.contextmanager
def install_gliner2(*, entities: list[dict[str, Any]] | None = None) -> Iterator[None]:
    """Install a fake ``gliner2`` module returning the canned per-label PII entity dict."""
    _StubGLiNER2.entities = entities or []
    module = types.ModuleType("gliner2")
    module.GLiNER2 = _StubGLiNER2  # type: ignore[attr-defined]
    saved = sys.modules.get("gliner2")
    sys.modules["gliner2"] = module
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["gliner2"] = saved
        else:
            sys.modules.pop("gliner2", None)


def gliner2_pii_entities() -> list[dict[str, Any]]:
    """Canned document-offset PII rows the GLiNER2 stub returns for the canonical fixture.

    The canonical document is ``"Alice works for Acme Corp in Berlin. Contact
    alice@example.com about Acme Corp."`` — the email ``alice@example.com`` sits at offsets
    [45, 62). Returning it as an ``email`` row (GLiNER2's lower-case zero-shot type) exercises the
    Provider's per-label-dict flatten, offset round-trip, and masking end to end.
    """
    return [{"start": 45, "end": 62, "label": "email", "confidence": 0.95}]


# ---------------------------------------------------------------------------
# sentence-transformers stub (the learned in-process Embedder Provider)
# ---------------------------------------------------------------------------


class _StubSentenceTransformer:
    """Stands in for ``sentence_transformers.SentenceTransformer``: fixed-width vectors in order.

    The learned Embedder Provider (``embedding.sentence_transformers``) can't run its real backend
    (``sentence-transformers`` → torch) in the CPU-first, offline test env (ADR-0007/0016). This
    stub returns a deterministic, fixed-width vector per input text (not semantic — it only proves
    the contract: one uniform-width vector per input, in order, at the model's declared width), so
    the conformance C1/C2/C3/C5/C6 subset is checked without a live model.
    """

    dimension: int = 8

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return type(self).dimension

    def encode(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        # A deterministic, fixed-width vector per input (keyed by index+width), input order.
        return [
            [float((i + j) % 3) for j in range(type(self).dimension)]
            for i, _text in enumerate(texts)
        ]


@contextlib.contextmanager
def install_sentence_transformers(*, dimension: int = 8) -> Iterator[None]:
    """Install a fake ``sentence_transformers`` returning fixed-width deterministic vectors."""
    _StubSentenceTransformer.dimension = dimension
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer  # type: ignore[attr-defined]
    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = module
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["sentence_transformers"] = saved
        else:
            sys.modules.pop("sentence_transformers", None)


# ---------------------------------------------------------------------------
# presidio-analyzer stub (the real Microsoft Presidio Redaction Provider)
# ---------------------------------------------------------------------------


class _StubRecognizerResult:
    """Stands in for ``presidio_analyzer.RecognizerResult``: entity_type + start/end + score."""

    def __init__(self, entity_type: str, start: int, end: int, score: float) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class _StubAnalyzerEngine:
    """Stands in for ``presidio_analyzer.AnalyzerEngine``: returns canned document-offset spans.

    The real Presidio stack (presidio-analyzer -> spaCy + a spaCy model) is not installed in the
    CPU-first, offline test env (ADR-0007/0016). This stub returns fixed spans (set by the
    installer) keyed to the canonical fixture text, filtered to the requested entities + threshold
    and bounded to the analysed text length — so the produced RedactionRecord carries a real,
    resolvable, in-bounds offset (a genuine contract check, not a no-op), without a live model.
    """

    spans: list[dict[str, Any]] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def analyze(
        self,
        text: str,
        language: str,  # noqa: ARG002 - accepted for API parity
        entities: list[str] | None = None,
        score_threshold: float = 0.0,
        **_kwargs: Any,
    ) -> list[_StubRecognizerResult]:
        wanted = set(entities) if entities is not None else None
        out: list[_StubRecognizerResult] = []
        for s in type(self).spans:
            if wanted is not None and s["entity_type"] not in wanted:
                continue
            if float(s["score"]) < score_threshold:
                continue
            if int(s["end"]) > len(text):
                continue
            out.append(
                _StubRecognizerResult(
                    entity_type=str(s["entity_type"]),
                    start=int(s["start"]),
                    end=int(s["end"]),
                    score=float(s["score"]),
                )
            )
        return out


class _StubNlpEngine:
    """A do-nothing NLP engine handle (the stub AnalyzerEngine ignores it)."""


class _StubNlpEngineProvider:
    """Stands in for ``presidio_analyzer.nlp_engine.NlpEngineProvider``."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def create_engine(self) -> _StubNlpEngine:
        return _StubNlpEngine()


@contextlib.contextmanager
def install_presidio(*, spans: list[dict[str, Any]] | None = None) -> Iterator[None]:
    """Install a fake ``presidio_analyzer`` (+ its ``nlp_engine`` submodule) returning canned spans.

    Fakes both the top-level ``presidio_analyzer`` module (``AnalyzerEngine`` +
    ``RecognizerResult``) and its ``presidio_analyzer.nlp_engine`` submodule
    (``NlpEngineProvider``), since the Provider imports from both. The canned ``spans`` are
    document-offset dicts
    (``{"entity_type", "start", "end", "score"}``) keyed to the fixture text.
    """
    _StubAnalyzerEngine.spans = spans or []
    module = types.ModuleType("presidio_analyzer")
    module.AnalyzerEngine = _StubAnalyzerEngine  # type: ignore[attr-defined]
    module.RecognizerResult = _StubRecognizerResult  # type: ignore[attr-defined]
    nlp_module = types.ModuleType("presidio_analyzer.nlp_engine")
    nlp_module.NlpEngineProvider = _StubNlpEngineProvider  # type: ignore[attr-defined]
    module.nlp_engine = nlp_module  # type: ignore[attr-defined]
    saved = sys.modules.get("presidio_analyzer")
    saved_nlp = sys.modules.get("presidio_analyzer.nlp_engine")
    sys.modules["presidio_analyzer"] = module
    sys.modules["presidio_analyzer.nlp_engine"] = nlp_module
    try:
        yield
    finally:
        for name, prior in (
            ("presidio_analyzer", saved),
            ("presidio_analyzer.nlp_engine", saved_nlp),
        ):
            if prior is not None:
                sys.modules[name] = prior
            else:
                sys.modules.pop(name, None)


def presidio_pii_spans() -> list[dict[str, Any]]:
    """Canned document-offset PII spans the Presidio stub returns for the canonical fixture.

    The canonical document is ``"Alice works for Acme Corp in Berlin. Contact
    alice@example.com about Acme Corp."`` — the email ``alice@example.com`` sits at offsets
    [45, 62). Returning it as an ``EMAIL_ADDRESS`` (Presidio's UPPER_SNAKE type) exercises the
    Provider's type mapping, offset round-trip, and masking end to end.
    """
    return [{"entity_type": "EMAIL_ADDRESS", "start": 45, "end": 62, "score": 0.95}]


# ---------------------------------------------------------------------------
# Canned typed replies keyed to the canonical fixture
# ---------------------------------------------------------------------------


def entity_chat_reply() -> str:
    """A canned NER endpoint reply: two entities at real offsets in the canonical chunk text."""
    return json.dumps(
        {
            "entities": [
                {"start": 0, "end": 5, "label": "person"},
                {"start": 16, "end": 25, "label": "org"},
            ]
        }
    )


def relation_chat_reply() -> str:
    """A canned LLM-RE reply: one relation between mention 0 and mention 1."""
    return json.dumps(
        {"relations": [{"head": 0, "tail": 1, "label": "works_for", "confidence": 0.8}]}
    )


def parse_chat_reply() -> str:
    """A canned Parse-endpoint reply: a clean markdown transcription (no page delimiter)."""
    return "Alice works for Acme Corp in Berlin."


def schema_induction_reply() -> str:
    """A canned schema-induction reply: the exact ``{entity,relation,pii}`` JSON shape (W4).

    Keyed to the canonical fixture ("Alice works for Acme Corp in Berlin. Contact
    alice@example.com about Acme Corp."): person/organization/location entities, a works_for/
    located_in relation schema, and an email PII type — a realistic induced schema the conformance
    case then confirms annotates the chunks.
    """
    return json.dumps(
        {
            "entity_types": ["person", "organization", "location"],
            "relation_types": ["works_for", "located_in"],
            "pii_types": ["email"],
        }
    )


def gliner_entities() -> list[dict[str, Any]]:
    """Canned chunk-local entity spans the GLiNER stub returns for the canonical chunk."""
    return [
        {"start": 0, "end": 5, "label": "person", "score": 0.9},
        {"start": 16, "end": 25, "label": "org", "score": 0.9},
    ]


def gliner_pii_entities() -> list[dict[str, Any]]:
    """Canned chunk-local PII spans the GLiNER-PII stub returns (an email-shaped span)."""
    return [{"start": 0, "end": 5, "label": "person", "score": 0.9}]


def gliner_relations() -> list[dict[str, Any]]:
    """Canned relation the gliner-relex stub returns, referencing the two canned entities.

    Faithful to the REAL ``predict_relations`` relation-dict shape (W1-relex FIX): ``head``/
    ``tail`` are sub-dicts carrying ``start``/``end``/``text``/``type``/``entity_idx`` (keyed by
    the chunk-local span, the same coordinates the provider resolves endpoints by) — NOT bare
    entity indices. person [0,5) works_for org [16,25), matching ``gliner_entities()``.
    """
    return [
        {
            "head": {"start": 0, "end": 5, "text": "Alice", "type": "person", "entity_idx": 0},
            "tail": {"start": 16, "end": 25, "text": "Acme Corp", "type": "org", "entity_idx": 1},
            "relation": "works_for",
            "score": 0.8,
        }
    ]
