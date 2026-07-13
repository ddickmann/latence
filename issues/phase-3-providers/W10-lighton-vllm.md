# W10-lighton-vllm — served-vLLM OCR provider (production OCR path, ~2.7× at concurrency)

**Production OCR = a warm vLLM server, not in-process transformers.** Pod-measured on the same GPU:
vLLM at concurrency ≈ **2.7× faster than optimized transformers** (LightOn's recommended path; their
5.71 pg/s H100 figure is vLLM). New `parser.lighton_vllm` provider hits a running vLLM
OpenAI-compatible endpoint over HTTP (same shape as `embedding.endpoint` / the LLM providers) — the
in-process `parser.lighton` stays for single-box/simple. Concurrency + retry like the superpod/W6.

## What to build — `parser.lighton_vllm` (package `latence-parser-lighton-vllm` or a provider in the lighton pkg)
Satisfies the SAME `Parser` Capability as `parser.lighton` (drop-in). Per document:
1. **Rasterize** PDF→page-images REUSING parser.lighton's exact logic: `dpi` (default 200) + megapixel
   bomb-cap + **`max_longest_dim` (default 1540, LANCZOS)** — identical preprocessing to the in-process
   path (factor the rasterizer into a shared helper if clean, else mirror it). Bare images = one page.
2. **POST each page-image** as a base64 data-URL to `{base_url}/v1/chat/completions` (OpenAI multimodal:
   `messages=[{role:user, content:[{type:image_url,image_url:{url:data}}, {type:text,text:prompt}]}]`)
   with `max_tokens` (default **4096** — content-safe, NOT 1024 which truncates dense pages),
   `temperature` **0.2**, `top_p` **0.9** (LightOn's official sampling). Decode → markdown per page →
   `PageMap.from_page_texts` (same exact-offset seam parser.lighton uses). Provenance via AdapterBase.
3. **Async concurrency + retry** (the W6/superpod pattern, REUSE `latence_core.endpoint.call_with_retry_async`
   + `AsyncOpenAI`): fan out page requests bounded by `max_concurrency` (default 8) with
   exp-backoff+jitter on 429/5xx/timeout; per-page fail-open → PARSE_ERROR for that page only, never
   aborts the doc. Keep the pixel-bomb/spoof/timeout guards. Public `parse` stays sync (asyncio.run inside).
4. **Config:** `base_url` (e.g. http://localhost:8000/v1), `api_key` (default "EMPTY"), `model`
   (lightonai/LightOnOCR-2-1B), `prompt`, `max_tokens` 4096, `temperature` 0.2, `top_p` 0.9,
   `max_concurrency` 8, `dpi` 200, `max_longest_dim` 1540, `request_timeout`.
5. **ProviderProfile:** compute="endpoint" (infra-gated like embedding.endpoint), evidence-bearing
   license (LightOnOCR-2 Apache-2.0, weights+code, verified — reuse parser.lighton's citation),
   deterministic=False, cost_per_1k=None (self-hosted).

## The production serve recipe (document in the provider docstring + a docs/ + pod script) — CUDA GRAPHS ON
LightOn's official + optimal-throughput config (NO --enforce-eager in production — cudagraphs give the
extra speed the head-to-head's enforce-eager floor did NOT capture):
```
vllm serve lightonai/LightOnOCR-2-1B --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --max-model-len 10000 --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching --max-num-seqs 32
```
Document the KNOWN caveat: on bleeding-edge Blackwell (SM 12.x) vLLM 0.24's bundled FlashInfer can't
JIT cudagraphs (`FlashInfer requires sm75+`) → add `--enforce-eager` there (a ~perf floor) OR upgrade
flashinfer; mainstream GPUs (Hopper/Ada/Ampere, CUDA≥12.6) run cudagraphs fine.

## Anti-false-green + tests
- Faithful async mock of the vLLM OpenAI chat-completions response (choices[].message.content =
  markdown) — mirrors the REAL shape. Tests: batched pages == per-page (record-for-record, offsets),
  `ceil(N/max_concurrency)`-bounded concurrency (timing test), per-page fail-open, determinism
  (index-ordered), guards intact. NO real server in CI (infra-gated, skip-with-flag).
- Conformance C1–C6 (endpoint-skip on no server); roster+1; `bash scripts/verify-local.sh` GREEN; ADR note.

## Definition of Done
- `parser.lighton_vllm` drop-in Parser: rasterize (200dpi/1540px, identical to in-process) → concurrent
  POST to vLLM OpenAI endpoint (4096 tok / temp 0.2 / top_p 0.9) → PageMap with exact offsets; retry +
  per-page fail-open + determinism; guards intact. Serve recipe documented WITH cudagraphs (+ Blackwell
  caveat). verify-local green; conformance; ADR note.
- Pod validation (maintainer): stand up vLLM (cudagraphs where the GPU allows), OCR the lbbw pages
  through the provider, confirm markdown + offsets correct + the concurrency throughput; fold real
  numbers into PERF-RESULTS.md. NO fabricated numbers.

## Reviewer should check
- Preprocessing (dpi/1540px/bomb-cap) is IDENTICAL to parser.lighton (no silent divergence).
- Offsets from the served markdown round-trip through PageMap exactly; concurrency real+bounded;
  per-page fail-open never aborts a doc; determinism index-ordered.
- max_tokens default 4096 (not 1024); temp 0.2/top_p 0.9 (LightOn); the mock mirrors the REAL vLLM
  response shape (no fictional API); serve recipe uses cudagraphs in production.
