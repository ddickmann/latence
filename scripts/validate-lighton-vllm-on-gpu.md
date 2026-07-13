# Validate `parser.lighton_vllm` (served vLLM LightOn-OCR) on a real GPU — turnkey pod steps (W10)

The offline slice is validated against a **faithful async vLLM mock** (`openai.AsyncOpenAI` is
monkeypatched; the fake `chat.completions.create` is a coroutine returning the real vLLM response
shape `choices[0].message.content`, decoding the base64 page image the Provider actually POSTed back
to its markdown — so the rasterise → data-URL → POST → PageMap round-trip is genuinely exercised) and
the offline rasteriser fakes (fake `pypdfium2`/`PIL` so the REAL shared rasteriser runs). The **real**
numbers — extracted-text fidelity, offset/PageMap round-trip on real scanned pages, and the
**concurrency throughput** against a live vLLM server — come from a **CUDA pod run**. Until this runs,
the vLLM throughput is explicitly **"pending pod validation"**, never fabricated.

> **The whole point of W10: the vLLM throughput number.** W8 measured the in-process transformers
> ladder (bf16 → page-batch → FA2) but could NOT measure the vLLM head-to-head (that pod's driver was
> too old for the cu126+ torch a LightOnOCR-2 vLLM needs). This script closes that gap on a pod that
> meets the driver bar. Record `pages/second at concurrency C` — that is the ~2.7× claim's evidence.

These steps are turnkey on a single ≥16 GB CUDA GPU (an H100 slice, an A10, an RTX 4090/2000 Ada, a
RunPod pod, …) whose **driver is ≥ 12.6** (LightOnOCR-2 vLLM needs cu126+ torch).

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device AND a driver >= 12.6
python --version               # 3.11+
```

## 1. Stand up the vLLM server (CUDA GRAPHS ON — the production recipe)

```bash
pip install "vllm>=0.24"       # a build with LightOnOCR-2 multimodal support (cu126+ torch)

vllm serve lightonai/LightOnOCR-2-1B \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 10000 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --max-num-seqs 32
# Serves the OpenAI-compatible endpoint at http://localhost:8000/v1 by default.
```

**Blackwell (SM 12.x) caveat:** if vLLM aborts with `FlashInfer requires sm75+`, either add
`--enforce-eager` (a ~perf floor — record that you did) OR upgrade `flashinfer` and keep cudagraphs
on. On Hopper/Ada/Ampere (CUDA ≥ 12.6) do NOT add `--enforce-eager` — it only floors throughput.

Confirm the server is up:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool   # lists lightonai/LightOnOCR-2-1B
```

## 2. Install the framework + this provider (light client deps only)

```bash
cd latence-framework
pip install -e packages/latence-core
pip install -e packages/latence-parser-lighton          # for the shared rasteriser (pypdfium2/pillow)
pip install -e packages/latence-parser-lighton-vllm     # the served-vLLM provider + openai client
python -c "from latence_parser_lighton_vllm.provider import LightOnVllmParser; print('ok')"
```

## 3. Real OCR over the LBBW pages through the provider — markdown + offsets

Point `LBBW_PDF` at a real (scanned) multi-page German banking PDF (the LBBW corpus the W8 ladder
used), then OCR it through the served endpoint:

```bash
python - <<'PY'
import os, time
from pathlib import Path
from latence_core.contracts import Classification, ParserInput, Provenance
from latence_core.pagemap import PageOffsetIndex
from latence_parser_lighton_vllm.provider import LightOnVllmParser

data = Path(os.environ["LBBW_PDF"]).read_bytes()
inp = ParserInput(
    provenance=Provenance(source_uri="file://lbbw.pdf", file_name="lbbw.pdf", file_type="pdf",
                          file_size=len(data), document_id="sha256:lbbw"),
    classification=Classification(language="und"),
    content=data,
)
provider = LightOnVllmParser({
    "base_url": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "max_concurrency": 8,     # fan pages out concurrently
})
t0 = time.monotonic()
(rec,) = list(provider.parse([inp]))
elapsed = time.monotonic() - t0

pages = len(rec.page_map.pages) if rec.page_map else 0
print("disposition:", rec.disposition)
print("pages:", pages, "| chars:", len(rec.content), "| endpoint_retries:", provider.endpoint_retries)
print(f"wall-clock: {elapsed:.2f}s  =>  {pages/elapsed:.2f} pages/second at concurrency 8")

# Offset round-trip: every page span resolves to its own page (ADR-0031), zero drift.
idx = PageOffsetIndex(rec.page_map)
for span in rec.page_map.pages:
    assert idx.page_for_offset(span.char_start) == span.page_number
assert idx.diagnostics.recovered == 0, "page-map drift!"
print("offset round-trip: OK (recovered =", idx.diagnostics.recovered, ")")

print("---- first page markdown (eyeball it: legible German, NO chat scaffolding) ----")
print(rec.content[: rec.page_map.pages[0].char_end])
PY
```

**Text-fidelity check (the false-green catch):** the markdown must be legible German banking text,
NOT prompt/chat scaffolding, and NOT truncated mid-page (that is why `max_tokens` defaults to 4096,
not 1024 — 1024 truncated ~25% of dense LBBW pages in W8). Record a sample.

## 4. The concurrency throughput sweep (the ~2.7× evidence)

Re-time the same document at several concurrencies to show the collapse from `N × latency` to
`ceil(N / C) × latency` — and compare pages/second against the W8 in-process transformers ladder in
`PERF-RESULTS.md` (the ~2.7× claim):

```bash
python - <<'PY'
import os, time
from pathlib import Path
from latence_core.contracts import Classification, ParserInput, Provenance
from latence_parser_lighton_vllm.provider import LightOnVllmParser

data = Path(os.environ["LBBW_PDF"]).read_bytes()
def run(c):
    inp = ParserInput(
        provenance=Provenance(source_uri="file://lbbw.pdf", file_name="lbbw.pdf", file_type="pdf",
                              file_size=len(data), document_id="sha256:lbbw"),
        classification=Classification(language="und"), content=data)
    p = LightOnVllmParser({"base_url": "http://localhost:8000/v1", "max_concurrency": c})
    t0 = time.monotonic(); (rec,) = list(p.parse([inp])); dt = time.monotonic() - t0
    n = len(rec.page_map.pages)
    print(f"concurrency {c:2d}: {dt:6.2f}s  {n/dt:5.2f} pg/s")
for c in (1, 2, 4, 8, 16):
    run(c)
PY
```

## 5. Live conformance verdict against the running server

On the pod, wire the provider's `base_url` at the live server and confirm C1–C6 (the endpoint-skip
subset runs against the real endpoint, not the stub):

```bash
python - <<'PY'
from latence_core.capability import ProviderRegistry
from latence_core.conformance import run_conformance
v = run_conformance(ProviderRegistry(), "parser.lighton_vllm")
print("passed:", v.passed, "stub_scoped:", v.stub_scoped, "device_skipped:", v.device_skipped)
for c in v.checks:
    print(f"  {c.check}: {c.status.value} — {c.detail}")
PY
```

(The conformance case is stub-scoped by default — the offline gate runs it against the async
`AsyncOpenAI` stub. Step 3/4 against the live server are the authoritative real-OCR + throughput
evidence.)

## What to fold back into the PR after the pod run

- The extracted-markdown sample + the offset round-trip result from step 3.
- The **concurrency throughput sweep** from step 4 (pages/second at 1/2/4/8/16) → `PERF-RESULTS.md`,
  and the vLLM-vs-transformers ratio (the ~2.7× claim's real number).
- Whether `--enforce-eager` was needed (Blackwell) or cudagraphs ran (mainstream).
- The live conformance verdict from step 5.

Until this pod run is folded in, the vLLM throughput for `parser.lighton_vllm` is **pending pod
validation** — the offline slice proves the adapter code, the shared rasteriser, the faithful async
mock, the concurrency + fail-open + determinism discipline, the license, and the graceful-failure
path, but NOT the real served throughput number.
