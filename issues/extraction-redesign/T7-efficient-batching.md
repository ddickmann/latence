# T7 — Efficient per-chunk extraction via gliner2's native batching + compile (and the real OOM fix)

**Stages:** `extract` (`fused_entity_relation.gliner2`), `redact` (`redaction.gliner2`) + the perf seam ·
**Depends on:** T1, T2 · **Severity:** major (makes per-chunk affordable; is the ROOT fix for the OOM)

## The tension per-chunk induction creates
Per-chunk induction (T1) gives each chunk its **own** label set. But gliner2's batched entry point
(`model.batch_extract_entities(texts, schema, …)`) takes **one schema for the whole batch**. So the
naive readings are both bad:
- one call per chunk → correct labels but terrible throughput (no batching);
- union the batch's labels → re-introduces the ~40-label degradation and the OOM this redesign exists
  to remove.

## The design: group by label-set, batch each group through gliner2's native forward
1. **Bucket chunks by their induced label-set** (entity-label-set for extract; PII/guard-category set
   for redact). Chunks that share a label set — which, on real corpora, is most chunks within a
   document or topic — batch together into **one** `batch_extract_entities` call with that shared,
   small schema. Chunks with distinct sets form their own buckets. This is the "intelligent batching":
   maximal batch width at each chunk's *own* small label count, never the corpus union.
2. **Run each bucket through gliner2's native batched forward** (not a Python loop over chunks) — this
   is gliner2's real throughput internal. Bound each forward by the existing **token-budget
   micro-batch** (from the earlier OOM fix) so a bucket of long chunks still can't blow the tensor;
   but with small label dims that budget now rarely bites.
3. **Unlock gliner2's `compile` (and keep `quantize`).** gliner2's native `compile` was defaulted OFF
   because varied-length / varied-label forwards made `torch._dynamo` recompile-thrash. Two things in
   this redesign remove that: label sets are now **small**, and bucketing by label-set gives **stable
   shapes within a bucket** (pad to the bucket's max length). Compile once per shape bucket → no
   thrash. Enable `compile=True` + `quantize` (fp16 on cuda) for the grouped path, guarded by the
   existing compile/OOM reload latch. This is the "compiling / gliner2 internals" the reviewer asked
   for — done where it is now actually safe.

**Honest note on "integrated server":** gliner2 does NOT ship a self-hostable inference server — its
only server surface is `from_api`, a client to fastino's *cloud* API (data egress + external key),
which is wrong for an on-prem data framework. The efficiency here comes from gliner2's **native
batched forward + compile + quantize** run in-process, not from a server. Framed accurately so the
pitch and the ADR don't perpetuate a "server" that isn't there.

## This is the real fix for the CUDA-OOM
The OOM was `batch × ~768 tokens × ~40 labels` → one ~17 GiB span-scoring tensor. Per-chunk induction
collapses the **label** dimension to a handful, so the tensor is small and the OOM does not form. The
reactive micro-batch backoff (shipped earlier) stays as a rare safety net; **this redesign is the root
cause fix.** Expect the redesign run's peak VRAM to be a fraction of the corpus-mode ~21.6 GiB.

## Acceptance criteria
- [ ] Chunks are bucketed by label-set; a bucket runs as ONE gliner2 batched forward, not N calls —
      asserted by counting `batch_extract_entities` calls vs. buckets (not vs. chunks).
- [ ] gliner2 `compile=True` + `quantize` are enabled for the grouped path and do NOT recompile-thrash
      (stable per-bucket shapes); a measured eager-vs-compiled throughput number is recorded.
- [ ] **Real GPU validation on the pod:** a per-chunk redesign run over the RFC corpus (or similar)
      shows peak VRAM a fraction of the 21.6 GiB corpus-mode baseline, and the redact stage does NOT
      OOM **without relying on the micro-batch backoff** (the backoff fires zero or near-zero times).
- [ ] Output byte-identical to a per-chunk unbatched reference (batch/bucket-invariant — same
      mentions/spans/order).
- [ ] `ruff` + `mypy --strict` clean; `verify-local` green; the compile path degrades gracefully
      (uncompiled reload) where a backend can't compile (Blackwell/CUDA-13 nvrtc — see ADR-0036 / the
      PYTORCH_JIT seam).

## Affected files
`packages/latence-extract-gliner2/.../provider.py` (bucketing + batched forward + compile knob),
`packages/latence-pii-gliner2/.../provider.py` (same for PII/guard buckets),
`packages/latence-core/src/latence_core/providers/perf.py` (compile/quantize/micro-batch seam),
`setup/wizard.py` (enable compile for the SOTA grouped path), the extract + pii tests + a GPU smoke.
