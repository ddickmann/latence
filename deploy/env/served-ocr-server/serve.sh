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
# EVERY SETTING BELOW IS FROM THE LIVE BLACKWELL / CUDA-13 RUN, NOT FROM A DOC — except where a
# comment says otherwise and names what it is derived from.
set -euo pipefail

MODEL="${LATENCE_OCR_MODEL:-lightonai/LightOnOCR-2-1B}"
PORT="${LATENCE_OCR_PORT:-8000}"

# --- FM-FLASHINFER-SM120: apply the workaround only where the failure actually exists ----------
#
# THE Blackwell workaround. FlashInfer's sampler refused a genuinely capable sm_120 device with
# "requires sm75 or higher": its version check misparses compute capability 12.0 as the integer 12,
# which compares below 75. Disabling only the FlashInfer *sampler* keeps every other FlashInfer
# path (and cudagraphs) on — it is not `--enforce-eager`, which would floor throughput.
#
# It used to be exported for EVERY host. That is a tax on every pre-Blackwell card: `8.6` parses to
# 86, `9.0` to 90 — all comfortably above 75, so the misparse cannot occur there and the fast
# sampler works. The gate below is what makes the workaround follow its own failure mode.
#
# The gate FAILS SAFE, and deliberately so. The two wrong answers are not symmetric: leaving the
# sampler enabled on an sm_120 host is a hard startup abort, while disabling it on an Ampere host
# is merely slower. So an unreadable, absent or unparseable capability probe keeps the workaround,
# and only a POSITIVELY IDENTIFIED pre-12.0 device opts out of it. (Same asymmetry argument as the
# TorchScript-JIT gate — see issues/e2e-extended/DECISIONS.md § D2.)
#
# Two escape hatches, both honoured ahead of the probe:
#   VLLM_USE_FLASHINFER_SAMPLER=<v>  — set it yourself and the gate never second-guesses you.
#   LATENCE_OCR_COMPUTE_CAP=8.6      — declare the capability where `nvidia-smi` is not exposed
#                                      (some containers), instead of paying the fail-safe tax.
# Remove all of this when flashinfer's capability parse is fixed; the pinned
# flashinfer-python>=0.6,<0.7 boundary is what makes that a deliberate, reviewed change.

detect_compute_caps() {
  # Echo one compute capability per visible GPU ("8.6"), or return non-zero if we cannot know.
  if [[ -n "${LATENCE_OCR_COMPUTE_CAP:-}" ]]; then
    printf '%s\n' "${LATENCE_OCR_COMPUTE_CAP}"
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local reported
  reported="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)" || return 1
  [[ -n "${reported}" ]] || return 1
  printf '%s\n' "${reported}"
}

flashinfer_sampler_is_broken_here() {
  # True (0) when the FlashInfer capability misparse can bite: any GPU at >= 12.0, or a capability
  # we could not read at all. False (1) only for a fully-identified pre-12.0 machine.
  local reported cap major
  reported="$(detect_compute_caps)" || return 0   # unknown -> fail safe
  while IFS= read -r cap; do
    cap="${cap// /}"
    [[ -z "${cap}" ]] && continue
    major="${cap%%.*}"
    if [[ ! "${major}" =~ ^[0-9]+$ ]]; then
      return 0                                    # unparseable ("N/A") -> fail safe
    fi
    if [[ "${major}" -ge 12 ]]; then
      return 0                                    # sm_120 and up -> the misparse is real here
    fi
  done <<< "${reported}"
  return 1
}

if [[ -n "${VLLM_USE_FLASHINFER_SAMPLER+set}" ]]; then
  export VLLM_USE_FLASHINFER_SAMPLER
  printf '[serve.sh] VLLM_USE_FLASHINFER_SAMPLER=%s (operator override, capability gate skipped)\n' \
    "${VLLM_USE_FLASHINFER_SAMPLER}" >&2
elif flashinfer_sampler_is_broken_here; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  printf '[serve.sh] VLLM_USE_FLASHINFER_SAMPLER=0 — compute capability >= 12.0 or unreadable, so\n' >&2
  printf '[serve.sh] FM-FLASHINFER-SM120 applies (fail-safe). cudagraphs stay ON.\n' >&2
else
  printf '[serve.sh] VLLM_USE_FLASHINFER_SAMPLER left unset — pre-12.0 device, FlashInfer parses\n' >&2
  printf '[serve.sh] its capability correctly here, so the sampler fast path stays enabled.\n' >&2
fi

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  `# 0.72, NOT the 0.90 the generic serving doc suggests. At 0.90 the multimodal path OOM'd on` \
  `# the live 34-concurrent-image run: vLLM sizes the KV cache from this fraction, but the` \
  `# per-request image tensors + the mm processor allocate OUTSIDE it, so a high fraction leaves` \
  `# no room for the pixels. 0.72 was stable across the full 519-page corpus.` \
  `# 24 GB re-derivation (GPU-ONLY, not yet measured): sweep LATENCE_OCR_GPU_MEM_UTIL=0.80 on the` \
  `# 3090 and fall back to 0.72 on any OOM, then write the measured number back HERE. The default` \
  `# stays at the value that was actually measured rather than a copied constant.` \
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
  `# LightOn's own published serving recipe: overlap the scheduler with the forward pass. Pure` \
  `# throughput, no output change — the OCR workload is one image + a long markdown continuation` \
  `# per request, so the scheduler step is a real fraction of each decode step.` \
  --async-scheduling \
  `# STATE the cudagraph mode instead of inheriting it. FULL_AND_PIECEWISE is already the V1` \
  `# default; naming it makes the guarantee legible in the log and survives a vLLM default change.` \
  `# NOT "FULL": FlashAttention-2 declares UNIFORM_BATCH, so vLLM auto-downgrades a FULL request to` \
  `# exactly this — asking for FULL would only make the startup log misstate the intent.` \
  `# NOT cudagraph_mm_encoder: LightOnOCRForConditionalGeneration is absent from vLLM's` \
  `# supported-models table for encoder CUDA graphs, so it is a no-op at best.` \
  --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  `# The attention backend is deliberately NOT pinned: FlashAttention-2 is already the sm_86` \
  `# default, and VLLM_ATTENTION_BACKEND is deprecated for removal. Verify it from the startup log` \
  `# ("Using FlashAttention backend") rather than asserting it into existence here.` \
  "$@"
