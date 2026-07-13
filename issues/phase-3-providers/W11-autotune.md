# W11-autotune — `latence tune`: per-device optimal-settings profiler (correctness-gated)

**"Deploys optimally on any box."** A client's GPU decides which perf knobs win — and which even
WORK (compile support is env-dependent: gliner2-multi-v1 + torch 2.4.1 raised `BackendCompilerFailed`
at inference on one pod). `latence tune` profiles the actual device, sweeps each provider's knobs on a
small real sample, KEEPS ONLY settings that don't crash AND don't change output, picks the fastest, and
writes a per-device tuned config the stacks can load. Throughput-first; correctness is a hard gate.

## What to build
1. **`latence tune <stack.yaml> [--sample N] [--out tuned/<device>.yaml]` CLI** (new subcommand). For the
   GPU/heavy providers in the stack (parser.lighton, fused_entity_relation.gliner2, redaction.gliner2,
   embedding.sentence_transformers, parser.lighton_vllm), it:
   - **Profiles the device**: torch-free where possible — GPU name/VRAM (nvidia-smi/torch), CPU cores,
     torch/CUDA availability. Records it as the tuned-config's `device_fingerprint` (so a tuned config
     is only reused on a matching device; a mismatch → re-tune or fall back to safe defaults).
   - **Sweeps the knobs** per provider over a small real sample (N records, default 8): `dtype`
     (fp32/bf16), `quantize` (gliner2), `compile` (on/off), `ocr_batch_size`/`batch_size`,
     `max_concurrency` (endpoint). Cartesian-but-pruned (skip combos that can't apply).
   - **CORRECTNESS GATE (hard):** each candidate setting runs the sample; a setting is REJECTED if it
     (a) raises ANY error — including inference-time compile/quantize failures (`BackendCompilerFailed`,
     dynamo errors, OOM) — or (b) changes output beyond tolerance vs the fp32/plain BASELINE (byte-identical
     labels/spans/offsets for encoders; text/offsets for OCR; cosine≈1 for embeddings). Only correct,
     non-crashing settings are eligible. **This is the whole point** — a box that can't compile simply never
     gets `compile: true`; slow-but-correct always beats fast-but-wrong/crashing.
   - **Picks the fastest eligible** setting per provider (median of K timed runs after warmup), writes
     `tuned/<device>.yaml`: `{device_fingerprint, per_provider: {provider: {winning knobs, measured
     throughput, baseline throughput, speedup}}}`. Deterministic given the same device+sample (seeded).
2. **Stack loading of a tuned config**: a stack can reference `tune: tuned/<device>.yaml` (or
   `latence run --tuned tuned/<device>.yaml`); the runner overlays the tuned per-provider knobs onto the
   stage configs IF the device_fingerprint matches, else warns + uses stack defaults (never silently
   applies a mismatched tune). Config-overlay only — no contract change.
3. **Honest report**: `latence tune` prints + writes a signed table (per provider: baseline→tuned
   throughput, speedup, winning knobs, REJECTED knobs + why — e.g. "compile: rejected (BackendCompilerFailed)").

## Anti-false-green + tests (offline-buildable; real sweep is a pod run)
- Fake providers with a known knob→(speed, correctness, crashes) profile: assert the tuner PICKS the
  fastest CORRECT one, REJECTS a knob whose fake forward raises BackendCompilerFailed (the real lesson),
  REJECTS a knob that changes output, and writes a fingerprinted config. A device-fingerprint-mismatch
  test asserts the overlay is NOT applied (falls back to defaults + warns).
- Determinism: same device+sample+seed → identical tuned config. `bash scripts/verify-local.sh` GREEN;
  ADR-0036 note (the tune seam + correctness gate + fingerprint). No fabricated device numbers.

## Definition of Done
- `latence tune` profiles the device, sweeps knobs, correctness-gates (rejects crashing/output-changing
  settings incl. inference-time compile failures), picks the fastest correct per provider, writes a
  fingerprinted `tuned/<device>.yaml`; the runner overlays it only on a fingerprint match (else safe
  defaults + warn). Offline tests prove pick-correct/reject-crash/reject-wrong/fingerprint-guard;
  verify-local green; ADR note. Pod: real sweep on the Blackwell folds actual per-provider winners +
  the rejected-compile case into a signed tune report. NO fabricated numbers.

## Reviewer should check
- The correctness gate genuinely rejects a knob that CRASHES (incl. lazy/inference-time compile failure)
  and one that CHANGES OUTPUT — not just load-time failures (the W9 compile-crash lesson).
- device_fingerprint prevents silently applying a tune from a different GPU (mismatch → defaults+warn).
- Tuner is deterministic (seeded); overlay is config-only (no contract/offset change); report is honest
  (winners AND rejections with reasons), never fabricated.
