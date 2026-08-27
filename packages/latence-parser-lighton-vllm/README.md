# latence-parser-lighton-vllm

The **production OCR path**: a `image→markdown` OCR Parser Provider (`parser.lighton_vllm`) for the
[Latence framework](https://github.com/ddickmann/latence) that hits a **running vLLM
OpenAI-compatible endpoint** serving
[**LightOn-OCR-2**](https://huggingface.co/lightonai/LightOnOCR-2-1B) (`lightonai/LightOnOCR-2-1B`) —
a compact vision-language OCR model, **Apache-2.0 weights AND code**.

**Production OCR = a warm vLLM server, not in-process transformers.** Pod-measured on the same GPU,
vLLM at concurrency is ~**2.7× faster than optimised transformers** (LightOn's recommended path;
their 5.71 pg/s H100 headline figure is the vLLM path). This Provider is a **drop-in** for
`parser.lighton` — the *same* `Parser` Capability, the *same* output contract — so a Pipeline swaps
the in-process OCR for the served one with **one config change** (point `base_url` at the endpoint).
The in-process `parser.lighton` stays for single-box / simple deployments.

```yaml
# In a Pipeline YAML — the production OCR path:
parse:
  capability: parse
  provider: parser.lighton_vllm
  config:
    base_url: http://localhost:8000/v1   # your running vLLM server
    max_concurrency: 8                    # fan pages out concurrently
```

## What it does per document

1. **Rasterise** PDF→page-images through the **same** rasteriser `parser.lighton` uses (the shared
   `latence_parser_lighton.rasterize.PageRasteriser`): identical `dpi` (200) + megapixel bomb-cap +
   **1540px LANCZOS longest-side** + offset-0 type-spoof guards — so the two OCR back-ends **cannot
   silently diverge** at the pixel boundary. A bare image is one page; a PDF is one image per page.
2. **POST each page-image** as a base64 PNG `data:` URL to `{base_url}/v1/chat/completions` (OpenAI
   multimodal: an `image_url` part + a text prompt), with `max_tokens` **4096** (content-safe — NOT
   1024, which truncates dense pages), `temperature` **0.2**, `top_p` **0.9** (LightOn's **official**
   sampling). The reply's `choices[0].message.content` is that page's markdown → the exact-offset
   `PageMap.from_page_texts` seam (offsets round-trip by construction).
3. **Async concurrency + retry** (the W6/superpod pattern, reusing
   `latence_core.endpoint.call_with_retry_async` + `openai.AsyncOpenAI`): a document's page requests
   fan out under an `asyncio.Semaphore(max_concurrency)` (default 8), each with
   exponential-backoff-with-jitter retry on 429/5xx/timeout; a **per-page** failure **fails open** to
   an empty page (that page only), never aborting the document. The public `parse` stays **sync**
   (`asyncio.run` inside), so the sync Runner seam is unchanged.

## The production vLLM serve recipe (CUDA GRAPHS ON)

LightOn's official + optimal-throughput config — **no `--enforce-eager` in production** (cudagraphs
give the extra speed the transformers head-to-head's enforce-eager floor did not capture):

```bash
vllm serve lightonai/LightOnOCR-2-1B --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --max-model-len 10000 --limit-mm-per-prompt '{"image": 1}' --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching --max-num-seqs 32
```

**Known caveat:** on bleeding-edge **Blackwell (SM 12.x)** vLLM 0.24's bundled FlashInfer cannot JIT
the cudagraphs (`FlashInfer requires sm75+`) → add `--enforce-eager` there (a ~perf floor) OR upgrade
flashinfer; mainstream GPUs (Hopper / Ada / Ampere, CUDA ≥ 12.6) run cudagraphs fine. Full recipe +
the turnkey pod script are in `docs/serving-lighton-vllm.md` and `scripts/validate-lighton-vllm-on-gpu.md`.

## Install

```bash
pip install latence-parser-lighton-vllm   # light: openai client + the shared rasteriser (pypdfium2/pillow)
```

This Provider ships the **light** deps — it is a thin async HTTP client, so **no torch/transformers**
here (those live behind the vLLM server). It reuses `parser.lighton`'s rasteriser via a dependency on
`latence-parser-lighton` (only its pure-Python `rasterize` module is imported; the transformers/torch
extras are never touched by this path). Its tests use a **faithful async mock of the vLLM
chat-completions response** — no real server in CI; the real OCR is validated against a live vLLM
server on a CUDA pod (`scripts/validate-lighton-vllm-on-gpu.md`).

## Infra gating (compute="either", like `embedding.endpoint`)

The profile declares `compute="either"`: the adapter is a thin HTTP client that runs on the local CPU
(rasterisation is CPU; the OCR model lives behind the vLLM endpoint), so the device seam does **not**
skip it on a CPU-only host. It is **infra-gated** — it needs a running vLLM server — the same way
`embedding.endpoint` / `parser.endpoint` are: conformance runs the
contract/graceful-failure/license/determinism subset against a faithful stub (stub-scoped), and the
real throughput needs a live rig (#64) — flagged, never fabricated.

## Security discipline (OCR output is untrusted, PII-bearing)

* A native rasteriser fault (pypdfium2/Pillow) maps to a typed `ProviderError` carrying only the
  provider + op + native *type name* — **never** image bytes or OCR text (C2/C6); it degrades
  per-document to a graceful `PARSE_ERROR` record (ADR-0020).
* An **offset-0 type-spoof guard** (shared with `parser.lighton`) rejects a `.pdf`/image whose bytes
  do not match the declared type **before** any endpoint call — defense-in-depth.
* A **pixel-bomb DoS cap** (the shared `_MAX_RENDER_MEGAPIXELS` clamp + `Image.MAX_IMAGE_PIXELS`
  discipline) bounds an oversized page / decompression-bomb image to a graceful `PARSE_ERROR`.

## License (ADR-0012, verified 2026-07-09)

Recorded evidence-bearingly (H-B1), weights and code **separately**:

* **code** — this framework code: **Apache-2.0** (the repository-wide licence); the `openai` client: **Apache-2.0**.
* **weights** — `lightonai/LightOnOCR-2-1B`: **Apache-2.0**
  ([model card](https://huggingface.co/lightonai/LightOnOCR-2-1B)).

Both permissive; the headline `license` is the weights SPDX (`Apache-2.0`). Passes conformance C3.
