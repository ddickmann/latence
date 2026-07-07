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
    """A fake ``chat.completions`` returning a canned reply set by the installer."""

    reply: str = "{}"

    def create(self, **_kwargs: Any) -> Any:
        message = types.SimpleNamespace(content=type(self).reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


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
def install_openai(*, reply: str = "{}", dimension: int = 8) -> Iterator[None]:
    """Install a fake ``openai`` module returning the given canned chat reply / embedding width."""
    _StubCompletions.reply = reply
    _StubEmbeddings.dimension = dimension
    module = types.ModuleType("openai")
    module.OpenAI = _StubOpenAI  # type: ignore[attr-defined]
    saved = sys.modules.get("openai")
    sys.modules["openai"] = module
    try:
        yield
    finally:
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

    def predict_with_relations(
        self,
        text: str,
        labels: list[str],
        relation_labels: list[str],  # noqa: ARG002 - accepted for API parity
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        ents = self.predict_entities(text, labels, threshold)
        rels = [r for r in type(self).relations if r.get("score", 1.0) >= threshold]
        return {"entities": ents, "relations": rels}


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
    """Canned relation the gliner-relex stub returns, referencing the two canned entities."""
    return [{"head": 0, "tail": 1, "relation": "works_for", "label": "works_for", "score": 0.8}]
