# Tune a stack per device (`latence tune`, W11-autotune)

**"Deploys optimally on any box."** A client's GPU decides which perf knobs win — and which even
*work* (compile support is env-dependent: `gliner2-multi-v1` + torch 2.4.1 raised
`BackendCompilerFailed` **at inference** on one pod). `latence tune` profiles the actual device,
sweeps each heavy Provider's perf knobs on timed runs of the stack, **keeps only settings that don't
crash AND don't change output**, picks the fastest, and writes a per-device tuned config the runner
can overlay. Throughput-first; **correctness is a hard gate**. Architecture: ADR-0036 (the tune seam),
building on the W8 perf seam (`latence_core.providers.perf`).

## The command

```bash
latence tune stacks/gpu-sota.yaml \
  --sample 8 \        # the numerator of each reported figure (default 8); it does not size the run
  --repeats 3 \       # timed runs per candidate after a warmup; their MEDIAN is the divisor
  --seed 0 \          # sweep seed (determinism)
  --out tuned/<device>.yaml   # default: tuned/<device-slug>.yaml at the stack's repo root
```

**What a figure in the report is.** A timed run is one run of the whole stack over the corpus
bundled for stack validation, with only one Provider's knobs changed. The probe does not cut that
corpus down to `--sample` records. Each figure is `--sample` divided by the median wall time of the
timed runs, so it compares settings on one device and is not a count of records per second.

It:

1. **Profiles the device** into a `device_fingerprint` — torch-free where possible: CUDA present
   (`LATENCE_CUDA`-overridable), GPU name + total VRAM (`nvidia-smi`), logical CPU count, and the
   `torch`/CUDA version **only if torch is already importable** (an absent torch is an honest `None`,
   never faked). The fingerprint is the reuse key: a tuned config is only applied on a device whose
   fingerprint **matches** (see below).
2. **Sweeps each heavy Provider's knobs** — `dtype` (fp32/bf16), `quantize` (gliner2),
   `compile` (on/off), `batch_size` / `ocr_batch_size`, `max_concurrency` (endpoint). The grid is
   cartesian-but-pruned and **deterministic**, baseline (fp32/plain, compile off, batch 1) first.
3. **Applies the HARD correctness gate** (the whole point). Each candidate runs the stack and is
   **REJECTED** if it:
   - raises **any** error — including an **inference-time** compile/quantize failure
     (`BackendCompilerFailed`, a TorchDynamo/Inductor error) or an OOM. The compile-fault detection is
     **torch-free** (`latence_core.providers.perf.is_compile_backend_failure`): it walks the
     exception's `__cause__`/`__context__` chain and matches by class name / defining-module prefix /
     telltale message, so a box that can't compile simply never gets `compile: true` — even when the
     fault only surfaces on the **first forward pass**, not at load (the W9 lesson);
   - **changes output** beyond tolerance vs the fp32/plain baseline (the target-stage checkpoint bytes
     — labels/spans/offsets/text — hashed and compared; a different digest = rejected `output_changed`).
4. **Picks the fastest ELIGIBLE** setting per Provider (highest figure; ties break toward fewer
   pinned knobs, then label — a total, deterministic order) and writes `tuned/<device>.yaml`:
   `{device_fingerprint, per-provider: {winner knobs, baseline/winner figure, speedup, rejected
   knobs + reasons}}`. The two figures are stored under the keys `baseline_throughput` and
   `winner_throughput`.
5. **Prints a report** — per Provider: baseline and winner figure, speedup, winning knobs, AND
   the **rejected** knobs with reasons (e.g. `compile: rejected (compile_failure)`). Winners AND
   rejections; **no number is fabricated** — each is computed from the median of the timed runs on
   THIS box.

> **Slow-but-correct always beats fast-but-crashing/wrong.** The baseline is eligible by construction
> (it defines the correctness reference), so a Provider on which every acceleration is rejected still
> ships its safe baseline — never a crash, never a silently-wrong output.

## Applying a tuned config — `latence run --tuned`

```bash
latence run pipeline.yaml --tuned tuned/<device>.yaml
```

The runner overlays the tuned per-Provider knobs onto the stage configs **only when the tuned
config's `device_fingerprint` matches this host**. On a **mismatch** it logs a warning and uses the
**stack defaults** — it NEVER silently applies a tune measured on a different GPU (where a winning
`compile: true` could crash). The overlay is **config-only**: it shallow-merges the winning knobs onto
each stage's config (`{**stage.config, **winning_knobs}`), exactly as if you had hand-written them into
the YAML — **no contract change, no offset change**. A nested Export/Disambiguation `config.embedder`
whose provider is a tuned one is merged inside its own config block (the embedder is an opt-in
sub-config, not a DAG stage).

## The real sweep is a pod run

The offline suite proves the **logic** — pick-correct, reject-crash (inference-time
`BackendCompilerFailed`), reject-wrong, and the fingerprint guard — with **fake Providers** whose
knob→(speed, correctness, crash) profile is known; it fabricates **no** GPU numbers. The real
per-device winners (and the real rejected-compile case) come from the maintainer's pod run:
`scripts/tune-on-gpu.md`.
