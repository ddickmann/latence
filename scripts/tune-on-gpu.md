# Run the real `latence tune` sweep on a GPU — turnkey pod steps (W11-autotune)

The offline slice proves the **tuner logic** against **fake Providers** with a known
knob→(speed, correctness, crash) profile (`test_tune.py`): it asserts the gate PICKS the fastest
correct setting, REJECTS an inference-time `BackendCompilerFailed`, REJECTS an output change, writes a
fingerprinted config, and that the runner overlay is NOT applied on a device-fingerprint mismatch. It
fabricates **no** GPU numbers. The **real** per-device winners — and, crucially, the real
**rejected-compile** case that motivated the whole slice — come from this pod run. Until it runs, the
per-device numbers are explicitly **pending pod validation**, never fabricated.

> **The false-green this catches.** A `compile: true` (or a `quantize`) that the offline fake never
> exercises can load fine and only blow up on the FIRST forward pass on a real checkpoint + a specific
> torch/driver combo (the pod hit `BackendCompilerFailed` with `gliner2-multi-v1` + torch 2.4.1). The
> gate's whole point is to catch exactly that and simply not pick it on that box.

These steps are turnkey: run on a single ≥16 GB CUDA GPU (an H100/Blackwell slice, an A10, an RTX
4090, a RunPod pod, …).

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device is visible + note its name/VRAM
python --version               # 3.11+
```

## 1. Install the framework + the heavy Providers you want to tune WITH their deps

```bash
# from a clone of the repo on the pod:
cd latence-framework
pip install -e packages/latence-core
# The heavy Providers in stacks/gpu-sota.yaml (install the ones your stack uses, WITH deps):
pip install -e packages/latence-relation-gliner    # fused NER+RE (relation.gliner_relex)
pip install -e packages/latence-pii-gliner         # learned PII (redaction.gliner_pii)
pip install -e packages/latence-pii-gliner2        # gliner2 PII (redaction.gliner2 — the compile-crash family)
pip install -e packages/latence-embedder-st        # embedding.sentence_transformers
pip install -e packages/latence-parser-lighton     # OCR VLM (parser.lighton) — for the OCR stack
```

## 2. Sanity: the device fingerprint reads the real GPU

```bash
python - <<'PY'
from latence_core.tune.fingerprint import probe_device_fingerprint
fp = probe_device_fingerprint()
print("slug:", fp.slug(), "| key:", fp.key)
print("cuda:", fp.cuda, "gpu:", fp.gpu_name, "vram_mb:", fp.gpu_memory_mb,
      "cpu:", fp.cpu_count, "torch:", fp.torch_version, "cuda_ver:", fp.cuda_version)
assert fp.cuda and fp.gpu_name, "expected a real CUDA device with a name"
PY
```

## 3. Run the sweep on the real stack

```bash
# The GPU/high-accuracy stack. --sample 8 is the small real sample; --repeats 3 = timed median.
latence tune stacks/gpu-sota.yaml --sample 8 --repeats 3 --seed 0 --out tuned/$(python -c \
  "from latence_core.tune.fingerprint import probe_device_fingerprint as p; print(p().slug())").yaml
```

**Read the printed report against these expectations (fold the real numbers into the report — do NOT
fabricate):**

- Each heavy Provider has a **winner** row with a **baseline r/s → winner r/s → speedup** measured on
  THIS device. bf16 should generally win over fp32 (the W8 measured ~1.36× for gliner encoders, ~1.49×
  for OCR); batch/ocr_batch should add throughput where the encoder is under-fed.
- The **Rejected knobs** section is the crux. If this box's torch/driver cannot compile a given
  Provider, its `compile: true` candidate(s) MUST appear as `rejected (compile_failure)` with a
  `BackendCompilerFailed`/dynamo detail — and the winner MUST NOT be a compile candidate. If a
  `quantize`/`dtype` candidate changes output it MUST appear as `rejected (output_changed)`.
- **No fabricated numbers.** Every r/s is the median of the seeded timed runs; a Provider that is not
  installed / not runnable is recorded honestly (baseline-crash / un-tunable), not given a fake number.

## 4. Confirm the overlay applies on THIS box and is guarded off a mismatch

```bash
# Applies (fingerprint matches this host):
latence run stacks/gpu-sota.yaml --tuned tuned/<device>.yaml -v 2>&1 | grep -i "applied tuned config"

# Guarded: hand-edit a COPY's device_fingerprint.gpu_name to a different GPU and confirm it is refused:
cp tuned/<device>.yaml /tmp/mismatch.yaml
#   (edit gpu_name to e.g. "Some Other GPU")
latence run stacks/gpu-sota.yaml --tuned /tmp/mismatch.yaml -v 2>&1 | grep -i "NOT applying"
```

## 5. Fold the real report into the slice

Record the printed table (winners + speedups + rejected-compile case) into `PERF-RESULTS.md` alongside
the W8 numbers, noting the exact GPU + torch/driver versions (the fingerprint's `slug`/`key`). Commit
the `tuned/<device>.yaml` if you want it reused (it is device-scoped by fingerprint, so it is safe to
check in — it will only ever apply on a matching box). **The numbers are the maintainer's to fold in;
never fabricated here.**
