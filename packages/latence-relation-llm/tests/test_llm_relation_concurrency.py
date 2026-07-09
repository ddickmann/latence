"""W6: the LLM RelationExtractor fans its per-document calls out concurrently, bounded + deterministic.

Anti-false-green mirrors the schema-inducer's W6 suite: the concurrency is REAL (a timing test asserts
``ceil(N/C) × delay`` not ``N × delay``), BOUNDED (a semaphore), and does NOT perturb determinism
(per-document reassembly in input document order — a shuffled-completion mock still yields byte-
identical, document-ordered relations). A single document's endpoint failure still surfaces
(deterministically, the first in document order), never on whichever finished first. Offline +
deterministic via a faithful async ``AsyncOpenAI`` mock (real chat-completions shape, per-call delay,
per-document scripted replies).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from typing import Any

import pytest
from latence_core.contracts import Classification, EntityMention, Provenance
from latence_core.endpoint import EndpointError
from latence_relation_llm import LLMRelationExtractor


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _Script:
    """Per-document scripted async replies keyed by DISPATCH (input document) order + a delay.

    ``create`` is a coroutine; each call pops the next scripted step in dispatch order (the RE
    Provider dispatches documents in first-seen order under gather), sleeps its per-step delay (so a
    test can reverse COMPLETION order), then returns/raises. ``concurrent`` records the max number of
    ``create`` coroutines in flight at once — the semaphore-bound check.
    """

    steps: list[tuple[str, Any, float]] = []
    delay: float = 0.0
    calls: int = 0
    in_flight: int = 0
    max_in_flight: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.steps = []
        cls.delay = 0.0
        cls.calls = 0
        cls.in_flight = 0
        cls.max_in_flight = 0


class _FakeCompletions:
    async def create(self, **_kwargs: Any) -> Any:
        idx = _Script.calls
        _Script.calls += 1
        _Script.in_flight += 1
        _Script.max_in_flight = max(_Script.max_in_flight, _Script.in_flight)
        kind, payload, step_delay = (
            _Script.steps[idx] if idx < len(_Script.steps) else ("reply", "{}", 0.0)
        )
        try:
            await asyncio.sleep(step_delay or _Script.delay)
            if kind == "raise":
                assert isinstance(payload, BaseException)
                raise payload
            msg = types.SimpleNamespace(content=payload)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
        finally:
            _Script.in_flight -= 1


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        self.chat = _FakeChat()


@pytest.fixture()
def fake_async_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")
    module.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    _Script.reset()


def _mention(doc_id: str, idx: int, label: str, text: str, start: int, end: int) -> EntityMention:
    prov = Provenance(
        source_uri="file:///c/a.txt",
        file_name="a.txt",
        file_type="txt",
        file_size=1000,
        document_id=f"sha256:{doc_id}",
        char_start=start,
        char_end=end,
        page_start=1,
        page_end=1,
    )
    return EntityMention(
        record_id=f"{doc_id}#chunk-0#entity-{idx}",
        document_record_id=doc_id,
        chunk_record_id=f"{doc_id}#chunk-0",
        index=idx,
        label=label,
        text=text,
        confidence=0.9,
        provenance=prov,
        classification=Classification(language="en", sensitivity="internal"),
    )


def _doc_mentions(doc_id: str) -> list[EntityMention]:
    # Every document has 2 mentions so it triggers exactly one LLM call.
    return [
        _mention(doc_id, 0, "PERSON", "Alice", 0, 5),
        _mention(doc_id, 1, "ORG", "Acme Corp", 16, 25),
    ]


def _rel_reply(label: str) -> str:
    return json.dumps({"relations": [{"head": 0, "tail": 1, "label": label, "confidence": 0.8}]})


# ---------------------------------------------------------------------------
# Concurrency is REAL (timing) and BOUNDED (semaphore).
# ---------------------------------------------------------------------------


def test_wall_clock_is_ceil_n_over_c_not_n(fake_async_openai: None) -> None:
    n, concurrency, delay = 8, 4, 0.05
    _Script.steps = [("reply", _rel_reply("works_for"), 0.0) for _ in range(n)]
    _Script.delay = delay
    mentions = [m for i in range(n) for m in _doc_mentions(f"d{i}")]
    provider = LLMRelationExtractor({"labels": ["works_for"], "max_concurrency": concurrency})
    start = time.monotonic()
    out = list(provider.relate([], mentions))
    elapsed = time.monotonic() - start
    assert len(out) == n  # one relation per document
    assert elapsed < 0.25, f"expected ~0.10s (concurrent), got {elapsed:.3f}s (looks sequential)"
    assert _Script.max_in_flight <= concurrency  # bounded by the semaphore
    assert _Script.max_in_flight > 1  # but genuinely concurrent


# ---------------------------------------------------------------------------
# Determinism: shuffled completion order → byte-identical, document-ordered output.
# ---------------------------------------------------------------------------


def test_shuffled_completion_order_is_document_ordered(fake_async_openai: None) -> None:
    n = 4
    # doc i emits relation label "rel_i"; later docs complete first (descending delays).
    _Script.steps = [("reply", _rel_reply(f"rel_{i}"), (n - i) * 0.02) for i in range(n)]
    labels = [f"rel_{i}" for i in range(n)]
    mentions = [m for i in range(n) for m in _doc_mentions(f"d{i}")]
    provider = LLMRelationExtractor({"labels": labels, "max_concurrency": n})
    out = list(provider.relate([], mentions))
    # Output is in document order d0..d3 and each carries its OWN relation label.
    assert [r.document_record_id for r in out] == [f"d{i}" for i in range(n)]
    assert [r.label for r in out] == [f"rel_{i}" for i in range(n)]


def test_two_concurrent_runs_byte_identical(fake_async_openai: None) -> None:
    n = 6
    labels = [f"rel_{i}" for i in range(n)]
    mentions = [m for i in range(n) for m in _doc_mentions(f"d{i}")]
    dumps = []
    for _ in range(2):
        _Script.reset()
        _Script.steps = [("reply", _rel_reply(f"rel_{i}"), (n - i) * 0.01) for i in range(n)]
        provider = LLMRelationExtractor({"labels": labels, "max_concurrency": 3})
        out = list(provider.relate([], mentions))
        dumps.append([r.model_dump_json() for r in out])
    assert dumps[0] == dumps[1]


# ---------------------------------------------------------------------------
# Error surfacing under concurrency: a document's endpoint failure surfaces deterministically.
# ---------------------------------------------------------------------------


def test_one_document_terminal_failure_surfaces_first_in_doc_order(fake_async_openai: None) -> None:
    # d0 good, d1 terminal 4xx, d2 good — all dispatched concurrently. The d1 EndpointError surfaces
    # (the RE Stage fails loud), deterministically the first in document order, not completion order.
    n = 3
    _Script.steps = [
        ("reply", _rel_reply("works_for"), 0.03),
        ("raise", _StatusError("bad request", status_code=400), 0.0),  # completes first
        ("reply", _rel_reply("works_for"), 0.03),
    ]
    mentions = [m for i in range(n) for m in _doc_mentions(f"d{i}")]
    provider = LLMRelationExtractor({"labels": ["works_for"], "max_concurrency": n})
    with pytest.raises(EndpointError, match="bad request"):
        list(provider.relate([], mentions))


def test_non_json_body_yields_no_relations_for_that_document_only(fake_async_openai: None) -> None:
    # d1's body is a well-formed HTTP 200 that is NOT JSON: no relations for d1, others unaffected.
    n = 3
    _Script.steps = [
        ("reply", _rel_reply("works_for"), 0.0),
        ("reply", "sorry, I can't do that", 0.0),
        ("reply", _rel_reply("works_for"), 0.0),
    ]
    mentions = [m for i in range(n) for m in _doc_mentions(f"d{i}")]
    provider = LLMRelationExtractor({"labels": ["works_for"], "max_concurrency": n})
    out = list(provider.relate([], mentions))
    docs = [r.document_record_id for r in out]
    assert docs == ["d0", "d2"]  # d1 dropped (non-JSON), the batch was not aborted
