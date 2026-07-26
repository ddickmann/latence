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

**Known ADR-0016 violation, tracked.** Every fake in this module stands in for a dependency
``latence-core`` does not declare (``openai``, ``gliner``, ``gliner2``, ``sentence_transformers``,
``transformers``, ``presidio``, ``pypdfium2``/``PIL``), so shipping a Provider package still
requires editing the core. The right shape is already proven in-tree —
``latence_source_sharepoint.testing`` ships its Graph fake in its own ``src``, and
``cases._LAZY_PROVIDER_CASES`` builds a case only when the owning package is installed — and the
generalisation is a second entry-point group, ``latence.conformance_cases``, with core keeping only
``harness`` (the C1–C6 checks) and ``fixtures``. See
``issues/backlog-post-v1/conformance-cases-per-package.md``. The roster half of the same problem is
already fixed: the expected Provider set is derived from the workspace manifests
(``packages/latence-core/tests/_workspace_providers.py``), not restated in a test.
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


# -- async twins (the W6 LLM Stages use the openai AsyncOpenAI client) --------


class _StubAsyncCompletions:
    """Async ``chat.completions`` for the W6 concurrent LLM Providers — same canned reply, awaited.

    ``label_inducer.llm`` / ``relation.llm`` now fan their calls out with ``AsyncOpenAI``, so their
    ``create`` is a coroutine. It returns the SAME ``choices[0].message.content`` + finish_reason
    shape as the sync stub, off the same class-level ``_StubCompletions.reply`` — one installer
    drives both the sync endpoint Providers and the async LLM Stages the same.
    """

    async def create(self, **_kwargs: Any) -> Any:
        message = types.SimpleNamespace(content=_StubCompletions.reply)
        choice = types.SimpleNamespace(
            message=message, finish_reason=_StubCompletions.finish_reason
        )
        return types.SimpleNamespace(choices=[choice])


class _StubAsyncChat:
    def __init__(self) -> None:
        self.completions = _StubAsyncCompletions()


class _StubAsyncOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        self.chat = _StubAsyncChat()
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
    # The W6 LLM Stages (label_inducer.llm / relation.llm) import AsyncOpenAI — expose the async
    # twin off the same canned reply so one installer drives sync endpoint + async LLM Providers.
    module.AsyncOpenAI = _StubAsyncOpenAI  # type: ignore[attr-defined]
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
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> _StubGLiNER:  # noqa: ARG003 - id unused
        # FAITHFUL to real gliner 0.2.27: from_pretrained REJECTS torch_dtype/attn_implementation
        # (the plain-gliner providers cast via ``.to(dtype)`` instead — W8 pod fix).
        for bad in ("torch_dtype", "attn_implementation"):
            if bad in kwargs:
                msg = f"UniEncoderSpanModel.__init__() got an unexpected keyword argument '{bad}'"
                raise TypeError(msg)
        return cls()

    def to(self, _dtype: Any) -> _StubGLiNER:
        return self  # W8: the plain-gliner dtype path is .to(bf16) on cuda

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        return [
            e
            for e in type(self).entities
            if e["label"] in labels and e["score"] >= threshold and e["end"] <= len(text)
        ]

    def batch_predict_entities(
        self, texts: list[str], labels: list[str], threshold: float = 0.5
    ) -> list[list[dict[str, Any]]]:
        # The REAL batched GLiNER API (W7): one forward pass over all texts; result[i] is the
        # entity list for texts[i] — the exact per-text ``predict_entities`` shape, one per input.
        return [self.predict_entities(text, labels, threshold) for text in texts]

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

    def inference(
        self,
        texts: list[str],
        labels: list[str],
        relations: list[str],  # noqa: ARG002 - accepted for API parity
        threshold: float = 0.5,
        return_relations: bool = True,  # noqa: ARG002 - always returns relations offline
        **kwargs: Any,  # noqa: ARG002 - relation_threshold/flat_ner/... accepted for parity
    ) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
        # The REAL batched fused entry point (W7): ONE forward pass over all texts; two parallel
        # lists (entities-per-text, relations-per-text) indexed by input position. Each text sees
        # the same canned relations (score-filtered), matching the per-text predict_relations path.
        entities_per_text = [self.predict_entities(text, labels, threshold) for text in texts]
        rels = [r for r in type(self).relations if r.get("score", 1.0) >= threshold]
        relations_per_text = [list(rels) for _ in texts]
        return entities_per_text, relations_per_text


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
    #: relation rows (flat, seeded by the fused case) reshaped into the real
    #: ``{"relation_extraction": {rel: [{"head": {...}, "tail": {...}}]}}`` dict by
    #: :meth:`extract_relations` — the W9 fused-extraction gliner2 path.
    relations: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _StubGLiNER2:  # noqa: ARG003 - id/quantize/compile unused offline
        return cls()

    def _entities_for(
        self, text: str, entity_types: list[str], threshold: float
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

    def _relations_for(
        self, text: str, relation_types: list[str]
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        # The REAL fused-extraction relation shape (W9): a per-relation-type dict of head/tail
        # sub-dict rows, NOT a flat list or a tuple — ``{"relation_extraction": {rel: [{"head":
        # {text,confidence,start,end}, "tail": {...}}]}}``. A provider mis-parsing it recovers none.
        by_rel: dict[str, list[dict[str, Any]]] = {}
        for r in type(self).relations:
            rel_type = r["relation"]
            if rel_type not in relation_types or r["head"]["end"] > len(text):
                continue
            by_rel.setdefault(rel_type, []).append(
                {"head": dict(r["head"]), "tail": dict(r["tail"])}
            )
        return {"relation_extraction": by_rel}

    # The REAL batched GLiNER2 entry points (signature mirrored from gliner2 1.3.2's
    # ``Extractor.batch_extract_entities`` / ``…_relations``: ONE schema for the whole call,
    # ``batch_size`` third positional, ``max_len`` keyword-last, one per-text result dict IN ORDER).
    # These are what both gliner2 Providers drive — the redactor since W7, the fused extractor since
    # T7 (one native forward per label bucket).
    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: list[str],
        batch_size: int = 8,  # noqa: ARG002 - gliner2's internal sub-batch knob; API parity
        threshold: float = 0.5,
        format_results: bool = True,  # noqa: ARG002 - API parity
        include_confidence: bool = False,  # noqa: ARG002 - API parity
        include_spans: bool = False,  # noqa: ARG002 - API parity
        max_len: int | None = None,  # noqa: ARG002 - W12 model-window bound; API parity
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        return [self._entities_for(text, entity_types, threshold) for text in texts]

    def batch_extract_relations(
        self,
        texts: list[str],
        relation_types: list[str],
        batch_size: int = 8,  # noqa: ARG002 - API parity
        threshold: float = 0.5,  # noqa: ARG002 - the seeded rows are already above threshold
        format_results: bool = True,  # noqa: ARG002 - API parity
        include_confidence: bool = False,  # noqa: ARG002 - API parity
        include_spans: bool = False,  # noqa: ARG002 - API parity
        max_len: int | None = None,  # noqa: ARG002 - API parity
    ) -> list[dict[str, dict[str, list[dict[str, Any]]]]]:
        return [self._relations_for(text, relation_types) for text in texts]

    # In gliner2 itself the single-text calls are thin wrappers over the batched ones
    # (``extract`` → ``batch_extract([text], …)[0]``); mirrored so the stub cannot make the batched
    # path look like the only path when it is not.
    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return self.batch_extract_entities(
            [text],
            entity_types,
            1,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len,
        )[0]

    def extract_relations(
        self,
        text: str,
        relation_types: list[str],
        threshold: float = 0.5,
        format_results: bool = True,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return self.batch_extract_relations(
            [text],
            relation_types,
            1,
            threshold,
            format_results,
            include_confidence,
            include_spans,
            max_len,
        )[0]


@contextlib.contextmanager
def install_gliner2(
    *,
    entities: list[dict[str, Any]] | None = None,
    relations: list[dict[str, Any]] | None = None,
) -> Iterator[None]:
    """Install a fake ``gliner2`` module returning the canned per-label / per-relation dicts.

    ``entities`` seeds the per-label ``extract_entities`` dict (PII + NER); ``relations`` seeds the
    per-relation-type ``extract_relations`` dict (the W9 fused-extraction path). Both are the REAL
    nested gliner2 shapes so a mis-parsing provider recovers nothing (the anti-false-green trap).
    """
    _StubGLiNER2.entities = entities or []
    _StubGLiNER2.relations = relations or []
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


def gliner2_entities() -> list[dict[str, Any]]:
    """Canned chunk-local NER entity rows the GLiNER2 fused stub returns for the canonical chunk.

    person "Alice" [0,5), org "Acme Corp" [16,25) — the SAME coordinates :func:`gliner_entities`
    uses, so the fused-gliner2 conformance case matches the relex case record-for-record. Reshaped
    into the real per-label ``{"entities": {label: [rows]}}`` dict by the stub's extract_entities.
    """
    return [
        {"start": 0, "end": 5, "label": "person", "confidence": 0.9},
        {"start": 16, "end": 25, "label": "org", "confidence": 0.9},
    ]


def gliner2_relations() -> list[dict[str, Any]]:
    """Canned relation the GLiNER2 fused stub returns, referencing the two canned entities.

    The REAL fused-extraction relation shape (W9): ``head``/``tail`` sub-dicts carrying chunk-local
    ``start``/``end`` char offsets (the coordinates the provider resolves endpoints by), NOT bare
    indices and NOT a tuple. person [0,5) works_for org [16,25), matching :func:`gliner2_entities`.
    """
    return [
        {
            "head": {"start": 0, "end": 5, "text": "Alice", "confidence": 0.9},
            "tail": {"start": 16, "end": 25, "text": "Acme Corp", "confidence": 0.9},
            "relation": "works_for",
        }
    ]


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

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.model_kwargs = dict(model_kwargs or {})  # W8 perf seam pass-through

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
# transformers stub (the real multilingual SPLADE SparseEmbedder Provider)
# ---------------------------------------------------------------------------


_SPLADE_STUB_VOCAB = 256
#: Public alias of the stub SPLADE vocab bound — the conformance sparse case's C1 index bound
#: (every emitted index must lie in ``[0, SPLADE_STUB_VOCAB)``).
SPLADE_STUB_VOCAB = _SPLADE_STUB_VOCAB


def _splade_token_id(token: str) -> int:
    """A stable vocab id for a token — byte-stable ``sha256`` (not the salted builtin ``hash``)."""
    import hashlib

    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SPLADE_STUB_VOCAB


def _splade_token_ids(text: str) -> list[int]:
    import re

    return [_splade_token_id(t) for t in re.findall(r"[a-z0-9]+", text.lower())]


class _StubArray:
    """A minimal stand-in for a torch tensor: the only surface the adapter uses is ``.tolist()``."""

    def __init__(self, data: Any) -> None:
        self._data = data

    def tolist(self) -> Any:
        return self._data


class _StubSpladeTokenizer:
    """Stands in for a ``transformers`` ``AutoTokenizer`` for a SPLADE checkpoint.

    Faithful to the surface the real adapter touches: a batch call returns padded ``input_ids`` +
    an ``attention_mask`` (masking the padding), a single-string call (used for injected-term
    resolution) returns that term's ids, and a bad checkpoint id raises ``OSError`` exactly as the
    real ``from_pretrained`` does — so the adapter's config-error mapping is genuinely exercised.
    """

    @classmethod
    def from_pretrained(cls, model_id: str, **_kwargs: Any) -> _StubSpladeTokenizer:
        if model_id == "bad/model":
            raise OSError("stub: model 'bad/model' not found")
        return cls()

    def __call__(self, texts: Any, **_kwargs: Any) -> dict[str, _StubArray]:
        if isinstance(texts, str):
            ids = _splade_token_ids(texts)
            return {"input_ids": _StubArray(ids), "attention_mask": _StubArray([1] * len(ids))}
        seqs = [_splade_token_ids(t) for t in texts]
        width = max((len(s) for s in seqs), default=0)
        padded = [s + [0] * (width - len(s)) for s in seqs]
        mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
        return {"input_ids": _StubArray(padded), "attention_mask": _StubArray(mask)}


class _StubSpladeConfig:
    vocab_size = _SPLADE_STUB_VOCAB


class _StubSpladeOutput:
    def __init__(self, logits: _StubArray) -> None:
        self.logits = logits


class _StubSpladeMaskedLM:
    """Stands in for a ``transformers`` ``AutoModelForMaskedLM``: deterministic MLM logits.

    Emits a positive logit at each token's own vocab id (zero elsewhere), so the adapter's real
    SPLADE pooling (``max_t log(1+relu)``) yields a genuine non-empty term-weight map keyed by the
    tokens present — a faithful shape, not a passthrough that lies.

    Faithful about placement too, since ``sparse.splade`` joined the shared load choreography: a
    real ``nn.Module`` is resident on **cpu** after ``from_pretrained`` and is moved by ``.to()``,
    reporting the result via ``.device``. A stub without those is *less capable* than the dependency
    it stands in for, which makes the seam's "prove it landed or raise" step fail against the stub
    rather than against a real defect.
    """

    def __init__(self) -> None:
        self.config = _StubSpladeConfig()
        self.device = "cpu"

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs: Any) -> _StubSpladeMaskedLM:
        return cls()

    def to(self, target: Any) -> _StubSpladeMaskedLM:
        if isinstance(target, str):
            self.device = target
        return self

    def eval(self) -> _StubSpladeMaskedLM:
        return self

    def __call__(
        self, input_ids: _StubArray, attention_mask: _StubArray | None = None, **_kwargs: Any
    ) -> _StubSpladeOutput:
        logits: list[list[list[float]]] = []
        for seq in input_ids.tolist():
            seq_logits: list[list[float]] = []
            for token_id in seq:
                row = [0.0] * _SPLADE_STUB_VOCAB
                row[token_id] = 3.0
                seq_logits.append(row)
            logits.append(seq_logits)
        return _StubSpladeOutput(_StubArray(logits))


@contextlib.contextmanager
def install_splade() -> Iterator[None]:
    """Install a fake ``transformers`` module so the real SPLADE adapter runs offline (torch-free).

    The heavy ``transformers``/``torch`` stack is not installed in the CPU-first, offline test env
    (ADR-0007/0016). This installs a faithful fake ``transformers`` exposing ``AutoTokenizer`` +
    ``AutoModelForMaskedLM``, so ``sparse.splade``'s real ``TransformersSpladeModel`` — its
    tokenise → MLM → SPLADE-pool path — is exercised for the C1/C2/C3/C5/C6 conformance subset
    without weights (the live numbers need a rig, and a *cleared* checkpoint, per ADR-0053/0012).
    """
    module = types.ModuleType("transformers")
    module.AutoTokenizer = _StubSpladeTokenizer  # type: ignore[attr-defined]
    module.AutoModelForMaskedLM = _StubSpladeMaskedLM  # type: ignore[attr-defined]
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = module
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["transformers"] = saved
        else:
            sys.modules.pop("transformers", None)


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


# ---------------------------------------------------------------------------
# Universal render Parser stub (parser.render, W5) — a faithful offline renderer + OCR
# ---------------------------------------------------------------------------
#
# ``parser.render`` (compute="either") genuinely RUNS through conformance on a CPU host — it is not
# device-skipped. Its native rasterisers (pypdfium2 / Pillow) are not installed in the CPU-first env
# (ADR-0016), so this installs FAITHFUL fake ``pypdfium2`` + ``PIL`` modules so the REAL render code
# path runs offline: a fake PDF page renders to a fake image whose ``.save(..., "PNG")`` emits
# deterministic bytes carrying the page text, and a stub OCR (:class:`StubRenderOcr`) decodes those
# bytes back to markdown — the exact renderer→PNG-bytes→OCR round-trip the Provider performs. The
# canonical fixture text is rendered as one page so the produced offsets match the canonical page
# map (a genuine C1 offset check, not a no-op). The single fabricated thing is the *pixels*; the
# dispatch, security guards, OCR-delegation seam, and PageMap assembly all run for real — the
# anti-false-green discipline the GPU OCR parsers carry.

_RENDER_FAKE_PNG_MAGIC = b"RENDEROCR\x00"


class _FakeRenderBitmap:
    """A pypdfium2 bitmap stand-in: ``.to_pil()`` yields a fake page image carrying its text."""

    def __init__(self, text: str) -> None:
        self._text = text

    def to_pil(self) -> Any:
        return _FakeRenderImage(self._text)


class _FakeRenderPage:
    """A pypdfium2 page stand-in: reports a normal page size and renders to a fake bitmap."""

    def __init__(self, text: str) -> None:
        self._text = text

    def get_size(self) -> tuple[float, float]:
        return (595.0, 842.0)  # A4 points — well under the pixel-bomb budget

    def render(self, scale: float = 1.0) -> _FakeRenderBitmap:
        return _FakeRenderBitmap(self._text)


class _FakeRenderPdfDocument:
    """A pypdfium2 ``PdfDocument`` stand-in over the fake PDF framing (``%PDF-`` + ``|||``)."""

    def __init__(self, stream: Any) -> None:
        data = stream.read() if hasattr(stream, "read") else stream
        body = bytes(data)[len(b"%PDF-") :].decode("utf-8")
        self._pages = [_FakeRenderPage(seg) for seg in body.split("|||")]

    def __iter__(self) -> Iterator[_FakeRenderPage]:
        return iter(self._pages)

    def close(self) -> None:
        return None


class _FakeRenderImageModule:
    """A Pillow ``Image`` module stand-in: ``open``/``MAX_IMAGE_PIXELS`` for the image path."""

    MAX_IMAGE_PIXELS: int | None = None

    @staticmethod
    def open(stream: Any) -> Any:
        data = stream.read() if hasattr(stream, "read") else stream
        raw = bytes(data)
        # Strip a leading image magic (the fake PNG magic) to recover the page text.
        for magic in (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM"):
            if raw.startswith(magic):
                return _FakeRenderImage(raw[len(magic) :].decode("utf-8"))
        return _FakeRenderImage(raw.decode("utf-8", errors="replace"))


class _FakeRenderImage:
    """A Pillow image stand-in: ``.load``/``.convert`` no-op; ``.save`` emits round-trip bytes."""

    def __init__(self, text: str) -> None:
        self._text = text

    def load(self) -> None:
        return None

    def convert(self, mode: str) -> _FakeRenderImage:
        return self

    def save(self, buffer: Any, format: str = "PNG") -> None:  # noqa: A002 - Pillow's kw name
        payload = self._text.encode("utf-8")
        buffer.write(_RENDER_FAKE_PNG_MAGIC + payload)


class StubRenderOcr:
    """A faithful offline OCR Provider for the render conformance case: decode a page's bytes.

    A REAL Parser Provider (structural Capability) resolved through ``parser.render``'s registry
    seam exactly as ``parser.lighton``/``parser.glm`` would be. It decodes the deterministic page
    bytes the fake renderer emits back into that page's markdown — the faithful stand-in for a
    pod-validated OCR model. Undecodable bytes → a graceful PARSE_ERROR (a real OCR degrade).
    """

    @classmethod
    def profile(cls) -> Any:
        from latence_core.providers.profile import REPO_LICENSE_SOURCE, ProviderProfile

        return ProviderProfile(
            compute="either",
            memory_mb=16,
            model_id="stub-render-ocr",
            license="Apache-2.0",
            license_verified=True,
            license_code="Apache-2.0",
            license_weights="Apache-2.0",
            license_source=REPO_LICENSE_SOURCE,
            license_verified_on="2026-07-09",
            deterministic=True,
            batch=False,
            cost_per_1k=None,
        )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    def parse(self, inputs: Any) -> Iterator[Any]:
        from latence_core.contracts import Disposition, DocumentRecord, PageMap

        for raw in inputs:
            data = raw.content if isinstance(raw.content, bytes) else raw.content.encode("utf-8")
            if data.startswith(_RENDER_FAKE_PNG_MAGIC):
                text = data[len(_RENDER_FAKE_PNG_MAGIC) :].decode("utf-8")
                assembly = PageMap.from_page_texts([text])
                yield DocumentRecord(
                    record_id=raw.provenance.document_id,
                    provenance=raw.provenance.model_copy(
                        update={"char_start": 0, "char_end": assembly.page_map.total_chars}
                    ),
                    classification=raw.classification,
                    content=assembly.text,
                    media_type="text/markdown",
                    page_map=assembly.page_map,
                    disposition=Disposition.PARSED,
                )
            else:
                yield DocumentRecord(
                    record_id=raw.provenance.document_id,
                    provenance=raw.provenance,
                    classification=raw.classification,
                    content="",
                    media_type="text/markdown",
                    page_map=None,
                    disposition=Disposition.PARSE_ERROR,
                    error="stub OCR could not decode the page image",
                )


def _stub_render_registry_cls() -> type:
    """Build a ProviderRegistry subclass resolving the OCR name to :class:`StubRenderOcr`.

    Built lazily (a factory, not a module-level class) so the stub module has no import-time
    dependency on the capability module's ProviderRegistry — the render Provider's ``ocr_registry``
    seam accepts a ``ProviderRegistry`` (or subclass), so this must subclass it to be honored.
    """
    from latence_core.capability import ProviderRegistry

    class _StubRenderRegistry(ProviderRegistry):
        def load(self, name: str) -> type:
            return StubRenderOcr

    return _StubRenderRegistry


def render_fake_pdf(pages: list[str]) -> bytes:
    """Build a spoof-guard-passing fake PDF the fake renderer splits into ``pages``."""
    return b"%PDF-" + "|||".join(pages).encode("utf-8")


def render_stub_registry() -> Any:
    """The stub registry a render conformance case passes as ``config['ocr_registry']``."""
    return _stub_render_registry_cls()()


@contextlib.contextmanager
def install_render_offline() -> Iterator[None]:
    """Install faithful fake ``pypdfium2`` + ``PIL.Image`` so ``parser.render`` runs offline.

    Injects a fake ``pypdfium2`` module (its ``PdfDocument`` renders the fake PDF framing to fake
    page images) and a fake ``PIL`` package (its ``Image`` decodes the fake image framing), for the
    duration of one Provider invocation, then removes them — so the REAL render code path runs
    without the heavy native deps and never leaks the fakes across cases. The stub OCR is wired via
    the Provider's ``ocr_registry`` config seam (see :func:`render_stub_registry`), not here.
    """
    fake_pdfium = types.ModuleType("pypdfium2")
    fake_pdfium.PdfDocument = _FakeRenderPdfDocument  # type: ignore[attr-defined]
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.MAX_IMAGE_PIXELS = None  # type: ignore[attr-defined]
    fake_pil_image.open = _FakeRenderImageModule.open  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image  # type: ignore[attr-defined]
    saved = {
        "pypdfium2": sys.modules.get("pypdfium2"),
        "PIL": sys.modules.get("PIL"),
        "PIL.Image": sys.modules.get("PIL.Image"),
    }
    sys.modules["pypdfium2"] = fake_pdfium
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_pil_image
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Served-vLLM OCR Parser stub (parser.lighton_vllm, W10) — a faithful offline rasteriser
# ---------------------------------------------------------------------------
#
# ``parser.lighton_vllm`` (compute="either") is infra-gated the same way ``parser.endpoint`` is: it
# needs a running vLLM server, so its conformance case is ``stub_scoped`` — the async
# ``AsyncOpenAI`` stub (:func:`install_openai`) stands in for the served endpoint. But it ALSO
# rasterises its input (unlike the raw-bytes ``parser.endpoint``), so its REAL rasteriser path runs
# offline against a faithful fake ``pypdfium2`` + ``PIL`` (below): a fake PDF page renders to a
# fake image with a real ``.size``/``.resize``/``.save`` (so the shared 1540px downscale + the
# base64 data-URL encode run for real), and Pillow's ``MAX_IMAGE_PIXELS`` bomb cap is honoured. The
# single fabricated thing is
# the served OCR reply; the rasterise → data-URL → POST → PageMap path all runs for real. Distinct
# from ``install_render_offline``: that fake image is for ``parser.render``'s renderer→save→OCR
# round-trip (no ``.size``); this one exposes the ``.size``/``.mode`` the shared
# ``PageRasteriser._downscale_longest`` + the vLLM provider's ``image_to_data_url`` read.


class _FakeVllmImage:
    """A Pillow image stand-in for the vLLM parser: real ``.size``/``.mode``/``.resize``/save."""

    def __init__(self, text: str, size: tuple[int, int] = (800, 1000)) -> None:
        self._text = text
        self.size = size
        self.mode = "RGB"

    def resize(self, size: tuple[int, int], _resample: Any = None) -> _FakeVllmImage:
        return _FakeVllmImage(self._text, size=size)

    def convert(self, _mode: str) -> _FakeVllmImage:
        return self

    def save(self, buffer: Any, format: str = "PNG") -> None:  # noqa: A002 - Pillow's kw name
        buffer.write(self._text.encode("utf-8"))


class _FakeVllmBitmap:
    def __init__(self, text: str) -> None:
        self._text = text

    def to_pil(self) -> _FakeVllmImage:
        return _FakeVllmImage(self._text)


class _FakeVllmPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_size(self) -> tuple[float, float]:
        return (595.0, 842.0)  # A4 points — well under the pixel-bomb budget

    def render(self, scale: float = 1.0, **_kwargs: Any) -> _FakeVllmBitmap:
        return _FakeVllmBitmap(self._text)


class _FakeVllmPdfDocument:
    """A pypdfium2 ``PdfDocument`` stand-in over the fake PDF framing (``%PDF-`` + ``|||``)."""

    def __init__(self, stream: Any) -> None:
        data = stream.read() if hasattr(stream, "read") else stream
        body = bytes(data)[len(b"%PDF-") :].decode("utf-8")
        self._pages = [_FakeVllmPage(seg) for seg in body.split("|||")]

    def __iter__(self) -> Iterator[_FakeVllmPage]:
        return iter(self._pages)

    def close(self) -> None:
        return None


class _FakeVllmResampling:
    LANCZOS = "LANCZOS"


class _FakeVllmImageModule:
    MAX_IMAGE_PIXELS: int | None = None
    Resampling = _FakeVllmResampling

    class DecompressionBombError(Exception):
        """Stand-in for ``PIL.Image.DecompressionBombError``."""

    @staticmethod
    def open(stream: Any) -> _FakeVllmImage:
        data = stream.read() if hasattr(stream, "read") else stream
        raw = bytes(data)
        for magic in (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM"):
            if raw.startswith(magic):
                return _FakeVllmImage(raw[len(magic) :].decode("utf-8"))
        return _FakeVllmImage(raw.decode("utf-8", errors="replace"))


def lighton_vllm_fake_pdf(pages: list[str]) -> bytes:
    """Build a spoof-guard-passing fake PDF the vLLM parser's rasteriser splits into pages."""
    return b"%PDF-" + "|||".join(pages).encode("utf-8")


@contextlib.contextmanager
def install_lighton_vllm_offline() -> Iterator[None]:
    """Install faithful fake ``pypdfium2`` + ``PIL`` so ``parser.lighton_vllm`` rasterises offline.

    Injects a fake ``pypdfium2`` (its ``PdfDocument`` renders the fake PDF framing to fake page
    images with a real ``.size``) and a fake ``PIL`` (its ``Image`` exposes ``.size``/``.mode``/
    ``.resize``/``.save`` + ``MAX_IMAGE_PIXELS``/``Resampling``), for one Provider invocation, then
    removes them — so the shared rasteriser's REAL code path (render-scale clamp, 1540px downscale,
    base64 PNG data-URL encode) runs without the heavy native deps. The served OCR reply comes from
    the async ``AsyncOpenAI`` stub (:func:`install_openai`), installed alongside by the case.
    """
    fake_pdfium = types.ModuleType("pypdfium2")
    fake_pdfium.PdfDocument = _FakeVllmPdfDocument  # type: ignore[attr-defined]
    fake_pil = types.ModuleType("PIL")
    fake_pil_image = types.ModuleType("PIL.Image")
    fake_pil_image.MAX_IMAGE_PIXELS = _FakeVllmImageModule.MAX_IMAGE_PIXELS  # type: ignore[attr-defined]
    fake_pil_image.open = _FakeVllmImageModule.open  # type: ignore[attr-defined]
    fake_pil_image.Resampling = _FakeVllmResampling  # type: ignore[attr-defined]
    fake_pil_image.DecompressionBombError = _FakeVllmImageModule.DecompressionBombError  # type: ignore[attr-defined]
    fake_pil.Image = fake_pil_image  # type: ignore[attr-defined]
    saved = {
        "pypdfium2": sys.modules.get("pypdfium2"),
        "PIL": sys.modules.get("PIL"),
        "PIL.Image": sys.modules.get("PIL.Image"),
    }
    sys.modules["pypdfium2"] = fake_pdfium
    sys.modules["PIL"] = fake_pil
    sys.modules["PIL.Image"] = fake_pil_image
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


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
