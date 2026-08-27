# latence-linkpred-ultra

Learned **graph-completion** Provider for the [Latence Framework](https://github.com/ddickmann/latence):
registers `graph_completion.ultra` on the Slice-1 `GraphCompleter` seam (ADR-0037). It predicts
missing knowledge-graph edges with an **inductive ULTRA scorer** (ICLR'24, MIT) zero-shot from a
pretrained checkpoint, with a **PyKEEN** (MIT) graceful fallback, behind a `Scorer` abstraction.

It is a drop-in for the in-core `graph_completion.reference` (the deterministic 2-hop closure): same
Capability, same **inferred-edge contract** (`properties.inferred=True`, honest empty-mention
Evidence, deterministic `edge_id`, asserted-collision drop, separate `inferred_edge_count`) — a
learned scorer where the reference is a heuristic.

## What it does

1. **Candidate generation** — type-aware, tiered **2-hop** candidate triples per relation, **bounded**
   (a dense-relation fan-out cap; every cap that drops candidates is logged) with a **rare-relation
   recall guarantee** (a rarely-observed relation keeps ALL its type-valid candidates).
2. **Scoring** — the inductive **ULTRA** model, zero-shot from a pretrained checkpoint (default
   `ultra_3g`); **PyKEEN** (a small transductive KGE model) when ULTRA's deps/checkpoint are absent;
   a typed `ProviderError` when neither is available (never a silent no-op).
3. **Calibration** — **temperature scaling** + a **precision-targeted threshold**: only edges whose
   *calibrated probability* clears a configurable target precision are emitted; the emitted
   `confidence` is the calibrated probability, never a raw score (`properties.calibrated=True`).
4. **Evaluation** — **MRR / Hits@k** on a held-out edge split with genuine **leak detection** (the
   held-out edges *and their inverses* are removed from the observed graph before scoring; a leak
   raises rather than fabricating a metric).

## Install

Provider discovery stays torch-free; the heavy deps are optional extras (ADR-0016):

```bash
pip install latence-linkpred-ultra           # discovery only (no scorer runs)
# ULTRA scorer (needs matched torch-geometric / torch-scatter wheels):
pip install "latence-linkpred-ultra[ultra]" -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
# or the PyKEEN fallback:
pip install "latence-linkpred-ultra[pykeen]"
```

The ULTRA **checkpoint is a config path** (default `ultra_3g`) — **weights are NOT vendored**; point
`checkpoint` at your local ULTRA ckpt.

**Self-contained — no C++/CUDA build, no upstream ULTRA repo.** The vendored ULTRA subset ships a
pure-PyTorch fallback for ULTRA's fused `rspmm` message-passing kernel, so it runs on any CPU/GPU host
with just `torch` + `torch-geometric` installed. The fallback is **numerically identical** to the
compiled kernel (proven on GPU: same held-out MRR/Hits@k and predicted edges). If ULTRA's compiled
`rspmm` extension happens to be importable on the host, it is used automatically as a faster path;
otherwise the pure fallback runs. Nothing to build.

> **Security — trust your checkpoint.** The checkpoint is loaded with `torch.load(..., weights_only=True)`
> so a tampered `.pth` cannot execute code at load time. Still point `checkpoint` **only at a trusted
> local file** — never a co-tenant-writable path or an unverified download. If a specific ckpt
> legitimately carries a non-tensor global, register it with `torch.serialization.add_safe_globals`
> for that trusted file; the loader never silently falls back to the unsafe (arbitrary-code) path.

## Use (stack YAML)

```yaml
  - name: complete
    capability: graph_completion
    provider: graph_completion.ultra
    depends_on: [graph]
    config:
      corpus_id: my-corpus
      scorer: auto            # ultra | pykeen | auto (ultra, pykeen fallback)
      checkpoint: /path/to/ultra_3g.pth
      device: cuda            # auto | cpu | cuda (the device seam)
      target_precision: 0.9   # precision-targeted emit threshold
      skip_if_unavailable: true   # optional trailing stage: skip-with-flag when no scorer on host
```

## Licenses (verified 2026-07-09, ADR-0012 — weights AND code, separately)

- **ULTRA** code: MIT (MilaGraph). A minimal inference subset is vendored under
  `_vendor/ultra/` (attributed in `NOTICE` / `THIRD-PARTY-LICENSES`).
- **ULTRA** weights: MIT (HF `mgalkin/ultra_3g` / `ultra_4g` / `ultra_50g`). Not vendored.
- **PyKEEN**: MIT (PyKEEN Project Team). Optional dependency, not vendored.
