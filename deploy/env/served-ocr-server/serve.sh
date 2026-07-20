#!/usr/bin/env bash
# deploy/env/served-ocr-server/serve.sh — stand the LightOn-OCR vLLM endpoint up (#196).
#
# The image's ENTRYPOINT, and equally the command to run on a bare pod after install.sh. Extra
# arguments are appended verbatim, so an operator can override any flag without editing this file.
#
#   bash deploy/env/served-ocr-server/serve.sh
#   bash deploy/env/served-ocr-server/serve.sh --port 8001
#
# MODEL (ADR-0012 — weights and code licences verified SEPARATELY):
#   lightonai/LightOnOCR-2-1B — WEIGHTS Apache-2.0, CODE Apache-2.0 (the vLLM serving path and
#   LightOn's model code alike). Both sides permissive; recorded in THIRD-PARTY-LICENSES.md and
#   docs/deployment.md. No weights are vendored here — the server pulls them at first boot.
#
# EVERY SETTING BELOW IS FROM THE LIVE BLACKWELL / CUDA-13 RUN, NOT FROM A DOC.
set -euo pipefail

MODEL="${LATENCE_OCR_MODEL:-lightonai/LightOnOCR-2-1B}"
PORT="${LATENCE_OCR_PORT:-8000}"

# THE Blackwell workaround. FlashInfer's sampler refused a genuinely capable sm_120 device with
# "requires sm75 or higher": its version check misparses compute capability 12.0 as the integer 12,
# which compares below 75. Disabling only the FlashInfer *sampler* keeps every other FlashInfer
# path (and cudagraphs) on — it is not `--enforce-eager`, which would floor throughput.
# Remove this when flashinfer's capability parse is fixed; the pinned flashinfer-python>=0.6,<0.7
# boundary is what makes that a deliberate, reviewed change.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  `# 0.72, NOT the 0.90 the generic serving doc suggests. At 0.90 the multimodal path OOM'd on` \
  `# the live 34-concurrent-image run: vLLM sizes the KV cache from this fraction, but the` \
  `# per-request image tensors + the mm processor allocate OUTSIDE it, so a high fraction leaves` \
  `# no room for the pixels. 0.72 was stable across the full 519-page corpus.` \
  --gpu-memory-utilization "${LATENCE_OCR_GPU_MEM_UTIL:-0.72}" \
  `# 16, NOT 32/64. Same OOM: each in-flight sequence holds a decoded page image. Pair the` \
  `# client's max_concurrency (12 on the live run) BELOW this so the server is never oversubscribed` \
  `# — see deploy/env/served-ocr-client/stack.yaml.` \
  --max-num-seqs "${LATENCE_OCR_MAX_NUM_SEQS:-16}" \
  `# one ~1540px page image + a 4096-token markdown continuation fits comfortably` \
  --max-model-len 10000 \
  `# one image per request — exactly what parser.lighton_vllm POSTs (one page per request)` \
  --limit-mm-per-prompt '{"image": 1}' \
  `# every page image is distinct, so the mm-processor cache is pure overhead` \
  --mm-processor-cache-gb 0 \
  `# OCR prompts share no long prefix, so prefix caching is pure overhead` \
  --no-enable-prefix-caching \
  "$@"
