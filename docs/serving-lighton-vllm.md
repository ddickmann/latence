# Serving LightOn-OCR with vLLM — the production OCR path (`parser.lighton_vllm`, W10)

**Production OCR is a warm vLLM server, not in-process transformers.** Pod-measured on the same GPU,
vLLM at concurrency is ~**2.7× faster than optimised in-process transformers** (LightOn's recommended
path; their 5.71 pg/s H100 headline figure is the vLLM path). The `parser.lighton_vllm` Provider is a
drop-in for `parser.lighton` that POSTs each rasterised page to a running vLLM OpenAI-compatible
endpoint. This note is the turnkey serve recipe + the client wiring.

## 1. Serve LightOn-OCR-2 with vLLM (CUDA GRAPHS ON — production)

LightOn's official + optimal-throughput config. **No `--enforce-eager` in production** — cudagraphs
give the extra speed the in-process transformers head-to-head's enforce-eager floor did NOT capture:

```bash
vllm serve lightonai/LightOnOCR-2-1B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 10000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --max-num-seqs 32
```

What each flag buys:

| flag | why |
|------|-----|
| `--dtype bfloat16` | the mandatory perf/accuracy default (fp16 → garbage for LightOn; fp32 ~2× slower) |
| `--gpu-memory-utilization 0.90` | headroom for the KV cache at `--max-num-seqs 32` |
| `--max-model-len 10000` | one page image (~1540px) + a 4096-token markdown continuation fits comfortably |
| `--limit-mm-per-prompt '{"image": 1}'` | one image per request — exactly what the Provider POSTs (one page/request) |
| `--mm-processor-cache-gb 0` | every page image is distinct — the mm-processor cache is pure overhead |
| `--no-enable-prefix-caching` | OCR prompts share no long prefix — prefix caching is pure overhead |
| `--max-num-seqs 32` | the batch of concurrent sequences vLLM runs — pair with the client `max_concurrency` |

### The Blackwell / FlashInfer caveat (KNOWN)

On bleeding-edge **Blackwell (SM 12.x)** GPUs, vLLM 0.24's bundled FlashInfer cannot JIT the
cudagraphs and fails with `FlashInfer requires sm75+`. Two options there:

1. add `--enforce-eager` (a ~perf floor — you lose the cudagraph speedup, correctness is unaffected), or
2. upgrade `flashinfer` to a build that supports SM 12.x, then keep cudagraphs on.

Mainstream GPUs (Hopper / Ada / Ampere, CUDA ≥ 12.6) run cudagraphs fine — **do not** add
`--enforce-eager` there; it only floors your throughput.

> Infra note (from W8): a pod whose driver is too old for the torch (cu126+) a LightOnOCR-2-supporting
> vLLM needs cannot stand this server up at all — needs driver ≥ 12.6. That constraint is why the W8
> transformers slice could not measure the vLLM head-to-head; W10 is the provider, its numbers come
> from a pod that meets the driver bar (never fabricated).

## 2. Point the Provider at the server

```yaml
# In a Pipeline / stack YAML:
parse:
  capability: parse
  provider: parser.lighton_vllm
  config:
    base_url: http://localhost:8000/v1   # the vLLM server's OpenAI endpoint
    api_key: EMPTY                        # a local vLLM accepts any placeholder
    model: lightonai/LightOnOCR-2-1B      # the served model id
    max_concurrency: 8                    # fan pages out concurrently (pair with --max-num-seqs)
    max_tokens: 4096                      # content-safe (NOT 1024 — truncates dense pages)
    temperature: 0.2                      # LightOn's official
    top_p: 0.9                            # LightOn's official
    dpi: 200                              # rasterisation DPI (identical to parser.lighton)
    max_longest_dim: 1540                 # longest-side downscale (identical to parser.lighton)
```

The Provider rasterises each document through the **same** rasteriser `parser.lighton` uses (200 dpi
/ megapixel bomb-cap / 1540px LANCZOS longest-side / offset-0 spoof guards — the shared
`PageRasteriser`), so the pixel preprocessing is byte-for-byte identical between the in-process and
served back-ends. It then POSTs each page as a base64 PNG `data:` URL to `{base_url}/v1/chat/completions`
(OpenAI multimodal), fans the page requests out under an `asyncio.Semaphore(max_concurrency)`, retries
429/5xx/timeout with exponential backoff + jitter, and **fails open per page** (one bad page never
aborts the document). Per-page markdown assembles onto the exact-offset `PageMap.from_page_texts` seam.

## 3. Tuning the concurrency

`max_concurrency` (client) and `--max-num-seqs` (server) are the two throughput knobs:

- Set `max_concurrency` ≈ `--max-num-seqs` so the client keeps the server's sequence slots full
  without over-queuing.
- The wall-clock for an N-page document collapses from `N × latency` to `ceil(N / max_concurrency) × latency`.
- `max_concurrency: 1` is exactly sequential (byte-identical output) — a debugging / low-memory floor.

## 4. Validate on the pod

`scripts/validate-lighton-vllm-on-gpu.md` is the turnkey pod run: stand up vLLM with the recipe above,
OCR the LBBW pages through `parser.lighton_vllm`, and confirm the markdown + offsets are correct and
the concurrency throughput is real. Fold the **real** numbers into `PERF-RESULTS.md` — never fabricate.
