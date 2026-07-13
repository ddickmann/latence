"""W6: the LabelInducer fans its per-document LLM calls out concurrently, bounded + deterministic.

Anti-false-green: the concurrency must be REAL (a timing test asserts the wall-clock for N items at
concurrency C is ``ceil(N/C) × delay``, NOT ``N × delay``), BOUNDED (a semaphore — more than C items
never run at once), and it must NOT perturb determinism (index-keyed reassembly → byte-identical
output regardless of completion order; proven with a shuffled-completion mock). Per-item fail-open is
preserved: one document's failure falls back for THAT document only, never aborting the batch. All
offline + deterministic via the faithful async ``_fake_openai`` mock (real chat-completions shape,
per-call delay).
"""

from __future__ import annotations

import json
import time

import _fake_openai as fake
from latence_core.contracts import ChunkRecord, Classification, Provenance
from latence_schema_inducer.provider import LLMLabelInducer


def _chunk(doc_id: str, content: str = "Alice works for Acme Corp.") -> ChunkRecord:
    prov = Provenance(
        source_uri="mem://t/doc.txt",
        file_name="doc.txt",
        file_type="txt",
        file_size=100,
        document_id=f"sha256:{doc_id}",
        char_start=0,
        char_end=len(content),
    )
    cls = Classification(language="en", category="memo", sensitivity="internal")
    return ChunkRecord(
        record_id=f"{doc_id}#chunk-0",
        document_record_id=doc_id,
        index=0,
        content=content,
        token_count=max(1, len(content) // 4),
        provenance=prov,
        classification=cls,
    )


def _good_reply(entity: list[str]) -> str:
    return json.dumps({"entity_types": entity, "relation_types": [], "pii_types": []})


def _provider(**config: object) -> LLMLabelInducer:
    return LLMLabelInducer({"base_url": "http://stub/v1", **config})


# ---------------------------------------------------------------------------
# Concurrency is REAL (timing) and BOUNDED (semaphore).
# ---------------------------------------------------------------------------


def test_wall_clock_is_ceil_n_over_c_not_n() -> None:
    # 8 documents, concurrency 4, each LLM call sleeps 0.05s. Concurrent ⇒ ~ceil(8/4)*0.05 = 0.10s.
    # Sequential would be 8*0.05 = 0.40s. Assert we are well under the sequential wall-clock.
    n, concurrency, delay = 8, 4, 0.05
    chunks = [_chunk(f"d{i}") for i in range(n)]
    provider = _provider(max_concurrency=concurrency)
    with fake.install(
        [fake.reply(_good_reply(["person"])) for _ in range(n)], delay=delay
    ) as script:
        start = time.monotonic()
        out = list(provider.induce(chunks))
        elapsed = time.monotonic() - start
    assert len(out) == n
    # ceil(8/4)*0.05 = 0.10s; allow generous scheduling headroom but stay far under sequential 0.40s.
    assert elapsed < 0.25, f"expected ~0.10s (concurrent), got {elapsed:.3f}s (looks sequential)"
    # BOUNDED: the semaphore holds — never more than ``concurrency`` calls in flight at once — yet the
    # fan-out is GENUINE — more than one at a time. Without this an unbounded run (no semaphore, or a
    # semaphore that ignores max_concurrency for values > 1) would pass the timing check above but is
    # caught here: max_in_flight would climb to n (8), not stay at concurrency (4).
    assert script.max_in_flight <= concurrency, (
        f"semaphore breached: {script.max_in_flight} calls in flight > bound {concurrency}"
    )
    assert script.max_in_flight > 1, (
        f"not genuinely concurrent: max_in_flight={script.max_in_flight} (looks sequential)"
    )


def test_concurrency_one_is_sequential_wall_clock() -> None:
    # max_concurrency=1 is today's sequential behaviour: 4 docs * 0.03s ≈ 0.12s (calls serialize).
    n, delay = 4, 0.03
    chunks = [_chunk(f"d{i}") for i in range(n)]
    provider = _provider(max_concurrency=1)
    with fake.install([fake.reply(_good_reply(["person"])) for _ in range(n)], delay=delay):
        start = time.monotonic()
        list(provider.induce(chunks))
        elapsed = time.monotonic() - start
    # Serialized: must be at least (n-1) delays of wall time — proves concurrency=1 truly serializes.
    assert elapsed >= delay * (n - 1)


# ---------------------------------------------------------------------------
# Determinism: shuffled completion order → byte-identical, input-ordered output.
# ---------------------------------------------------------------------------


def test_shuffled_completion_order_is_byte_identical() -> None:
    # Bind a distinct reply to each document by dispatch (input) order, but make the LATER documents
    # complete FIRST (descending delays). If reassembly keyed on completion order, the labels would
    # be mis-assigned; index-keyed reassembly keeps each doc's own labels.
    n = 4
    chunks = [_chunk(f"d{i}") for i in range(n)]
    # doc i gets label "type_i" (canonicalized to "type i"); doc i completes after (n-i)*0.02s so
    # d3 finishes before d0.
    steps = [fake.reply_after(_good_reply([f"type_{i}"]), delay=(n - i) * 0.02) for i in range(n)]
    provider = _provider(max_concurrency=n)
    with fake.install(steps):
        out = list(provider.induce(chunks))
    # Each document kept its OWN labels despite the reversed completion order.
    for i, c in enumerate(out):
        assert c.induced_labels is not None
        assert c.induced_labels.entity_types == [f"type {i}"], f"doc {i} got wrong labels"


def test_two_concurrent_runs_byte_identical() -> None:
    n = 6
    chunks = [_chunk(f"d{i}") for i in range(n)]
    dumps = []
    for _ in range(2):
        provider = _provider(max_concurrency=3)
        steps = [fake.reply_after(_good_reply([f"t{i}"]), delay=(n - i) * 0.01) for i in range(n)]
        with fake.install(steps):
            out = list(provider.induce(chunks))
        dumps.append([c.model_dump_json() for c in out])
    assert dumps[0] == dumps[1]


# ---------------------------------------------------------------------------
# Per-item fail-open under concurrency: one bad item never aborts the batch.
# ---------------------------------------------------------------------------


def test_one_item_failure_fails_open_without_aborting_batch() -> None:
    # 3 documents dispatched concurrently in input order: d0 good, d1 auth-fails (terminal, fail-open
    # for d1 only), d2 good. The batch completes; only d1 falls back.
    chunks = [_chunk("d0"), _chunk("d1"), _chunk("d2")]
    steps = [
        fake.reply(_good_reply(["person"])),
        fake.raises(fake.FakeAPIStatusError("invalid api key", status_code=401)),
        fake.reply(_good_reply(["place"])),
    ]
    provider = _provider(max_concurrency=3)
    with fake.install(steps):
        out = list(provider.induce(chunks))  # MUST NOT raise
    assert out[0].induced_labels is not None
    assert out[0].induced_labels.entity_types == ["person"]
    assert out[1].induced_labels is None  # d1 failed open — config labels downstream
    assert out[2].induced_labels is not None
    assert out[2].induced_labels.entity_types == ["place"]
    assert provider.induction_fallbacks == 1  # exactly the one bad item
    assert provider.induction_documents == 3  # all three attempted


def test_transient_5xx_retried_per_item_under_concurrency() -> None:
    # d0 sees a 5xx then a good reply (one retry, succeeds); d1 is a clean good reply. The retry is
    # tallied on endpoint_retries; both documents get their labels. Dispatch order = input order, so
    # the two d0 steps + one d1 step are consumed in order.
    chunks = [_chunk("d0"), _chunk("d1")]
    steps = [
        fake.raises(fake.FakeAPIStatusError("overloaded", status_code=503)),
        fake.reply(_good_reply(["person"])),  # d0 retry
        fake.reply(_good_reply(["place"])),  # d1
    ]
    # concurrency=1 so d0's two steps are consumed before d1's — a deterministic step assignment.
    provider = _provider(max_concurrency=1, retry_attempts=3)
    with fake.install(steps):
        out = list(provider.induce(chunks))
    assert out[0].induced_labels is not None
    assert out[0].induced_labels.entity_types == ["person"]
    assert out[1].induced_labels is not None
    assert out[1].induced_labels.entity_types == ["place"]
    assert provider.endpoint_retries == 1
    assert provider.induction_fallbacks == 0
