# W6-llm-concurrency — parallelize the OpenAI-compatible LLM stages (throughput, retry, like superpod)

**The LLM steps are sequential and latency-bound — the fix is concurrency + robust retry, like the
superpod's async enricher.** In the real lbbw E2E, `schema_induction` (`label_inducer.llm`) took 43.9s
for 24 chunks (~1.8s/chunk) — pure per-request latency, one call at a time. Fan the calls out
concurrently and it collapses to ~latency × ceil(N/concurrency). Apply the SAME treatment to
`relation.llm`. Throughput-first: this is a batch pipeline, not an online API.

## What to build (both `label_inducer.llm` and `relation.llm`)
Keep the provider's public `induce(chunks)` / `relate(...)` SYNC (the Runner is sync), but internally
**fan out the per-item LLM calls concurrently** and gather:
- Use the **`openai` async client** (`AsyncOpenAI`) + an `asyncio.Semaphore(max_concurrency)` + a
  single `asyncio.run(_gather_all(items))` inside the sync method (mirrors the superpod's async
  enricher — collect items, bound concurrency, gather). One event loop per stage-run.
- **Config `max_concurrency`** (default **8**; 1 = today's sequential behaviour). Also honor a
  `request_timeout` (per call) and the existing retry knobs.
- **Robust per-request retry/backoff** (like superpod / `latence_core.endpoint.call_with_retry`):
  retry on 429 + 5xx + timeouts with **exponential backoff + jitter**, capped attempts
  (`max_retries`, default 4); a 4xx-other is terminal. Retries fold into the Quality Report
  (`endpoint_retries`) as they do for the other endpoint providers.
- **ALL safety nets preserved, applied PER concurrent result** (the W4 discipline, unchanged):
  structured `json_object`, parse-and-repair, truncation→doubled-budget retry, schema validation,
  label sanitization, **fail-open-to-config-never-crash**. A single item's failure fails open for THAT
  item only — never aborts the batch (`return_exceptions=True` on the gather; a failed item → config
  labels + a typed no-content log).
- **Determinism preserved (Baseline bar):** each item is independent; results are re-assembled in
  input order (index-keyed), temperature 0 → a seeded run is byte-identical regardless of completion
  order. A test asserts two runs (and a shuffled-completion mock) produce identical output.

## Anti-false-green + tests
- The faithful mock async `openai` client returns the real chat-completions response shape with a
  per-call delay; a test asserts wall-clock for N items at concurrency C ≈ ceil(N/C)×delay (NOT N×delay)
  — the concurrency actually happens. A retryable-then-success mock proves backoff; a
  malformed/timeout mock proves per-item fail-open without aborting the batch. Order-independence test
  (shuffled completion) proves determinism.
- `bash scripts/verify-local.sh` green; no roster change (same providers, faster). ADR-0036 note (the
  async fan-out + concurrency/retry contract). Real numbers from the pod (lbbw induce re-timed).

## Definition of Done
- `label_inducer.llm` + `relation.llm` fan out concurrent LLM calls (bounded by `max_concurrency`,
  default 8) with exponential-backoff+jitter retry; public method stays sync; all W4 safety nets applied
  per result; per-item fail-open never aborts the batch; determinism byte-identical (order-independent).
- Concurrency proven by a timing test (N/C not N); retry + fail-open + determinism tested. verify-local
  green; ADR note. Pod: the lbbw `induce` stage re-timed (expect ~44s → ~6-8s at concurrency 8).

## Reviewer should check
- Concurrency is REAL and bounded (semaphore; timing test shows ceil(N/C) not N) and does not break
  determinism (index-ordered reassembly, order-independence test).
- Every W4 safety net still fires per concurrent result; one bad item fails open without killing the
  batch (gather return_exceptions).
- Retry is exponential+jitter, capped, on the right status classes; retries surface in the Quality
  Report; the async client is the `openai` SDK (OpenAI/OpenRouter/vLLM base_url unchanged).
