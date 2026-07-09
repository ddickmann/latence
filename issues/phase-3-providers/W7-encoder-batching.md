# W7-encoder-batching — batch the learned ENCODER stages for GPU throughput

**The encoder stages leave the GPU idle by running one item per model call — batch them.** In the
lbbw E2E: `redact` (GLiNER2) 60.5s/3 docs, `extract` (relex) 26s/24 chunks, and the embedder — all
processing a single chunk/document per forward pass, so the GPU is under-fed. GLiNER exposes
`batch_predict_entities`; relex/gliner batch over inputs; sentence-transformers has native `batch_size`.
Feeding B items per call collapses these stages ~5-10× on a GPU. (These are ENCODERS — vLLM does NOT
apply; the win is a real batch dimension + SDPA/bf16, not a server.)

## What to build (each learned encoder provider gets a `batch_size` + batched model calls)
Providers: `entity.gliner`, `relation.gliner_relex` (fused), `redaction.gliner_pii`,
`redaction.gliner2`, `embedding.sentence_transformers` (+ `disambiguation.embedding`'s embed path).
- **`batch_size` config** (default a sensible GPU value, e.g. 16; the P8 autotuner will override
  per-device). Accumulate up to `batch_size` chunks/documents, call the model ONCE per batch:
  - GLiNER family → `model.batch_predict_entities(list_of_texts, labels, threshold=…)` (verify the real
    API on the pod; it exists) instead of a per-text `predict_entities` loop; relex → the batched
    `predict_relations` path over a list. Map each result back to its record by index.
  - sentence-transformers → pass `batch_size` to `model.encode(...)` (already batched; just thread it).
- **Offsets/contracts UNCHANGED** — batching changes only how many texts share a forward pass; each
  record's char offsets, PIISpans, mentions, evidence, and provenance are identical to today (a test
  asserts batched output == unbatched output, record-for-record). Determinism byte-identical.
- **Memory-aware**: `batch_size` bounds the per-call tensor; the provider must not accumulate the whole
  corpus (respect the Runner's bounded-batch streaming, ADR-0033) — cap the in-flight batch to
  `batch_size`. `ProviderProfile.memory_mb` stays honest for the batched footprint.
- Reuse `AdapterBase.batched` where it fits; native fault → typed error (no leak) unchanged.

## Anti-false-green + tests
- **Equivalence test (the crux):** for each provider, batched(N items) yields EXACTLY the same records
  (same offsets/labels/spans/order) as N single-item calls — the offline fake makes this checkable
  without weights. A provider that mis-maps batch results to records (off-by-one, wrong order) FAILS.
- A test asserts the model is called `ceil(N/batch_size)` times, not N (batching actually happens).
- `bash scripts/verify-local.sh` green; no roster change. ADR-0036 note. Real pod numbers (lbbw
  `extract`/`redact` re-timed at batch 16 — expect the multi-× drop).

## Definition of Done
- Each learned encoder provider accepts `batch_size` and issues batched model calls
  (`ceil(N/batch_size)` calls, tested); output is record-for-record identical to unbatched
  (equivalence test); offsets/contracts/determinism unchanged; memory bounded to the batch.
- verify-local green; ADR note. Pod: lbbw `extract` + `redact` re-timed at batch 16 (expect ~5-10×
  faster); numbers folded in, never fabricated.

## Reviewer should check
- Batched output == unbatched output, record-for-record (offsets/spans/mentions/order) — the
  correctness crux; a mis-mapped batch FAILS the equivalence test.
- The model is genuinely called `ceil(N/batch_size)` times (batching real), and memory stays bounded
  to `batch_size` (no whole-corpus accumulation — ADR-0033 streaming honored).
- Determinism byte-identical; native fault → typed error, no leak. `batch_size` is honest in the profile.
