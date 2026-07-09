# Validate W7 encoder batching on the GPU pod — re-time lbbw `extract` + `redact` at batch 16

The W7 slice batches the learned ENCODER Providers (`entity.gliner`, `relation.gliner_relex`,
`redaction.gliner_pii`, `redaction.gliner2`, `embedding.sentence_transformers`) so B items share one
forward pass. The offline suite proves **correctness** (record-for-record equivalence + the
`ceil(N/batch_size)` call count) without weights; the **throughput** number needs a real GPU. This
is the turnkey pod run to re-time the lbbw E2E stages at `batch_size: 16` and confirm the expected
multi-× drop. **Fold the measured numbers into the Baseline bar; never fabricate them.**

## Baseline (batch_size = 1, the pre-W7 per-item loop)

From the lbbw E2E that motivated the slice:

| Stage    | Provider              | Baseline (batch 1)   |
|----------|-----------------------|----------------------|
| `redact` | `redaction.gliner2`   | 60.5 s / 3 docs      |
| `extract`| `relation.gliner_relex` | 26 s / 24 chunks   |

## Prereqs (on the CUDA pod)

```bash
cd /workspace/latence-framework            # the worktree/checkout on the pod
uv sync
for p in latence-ner-gliner latence-relation-gliner latence-pii-gliner latence-pii-gliner2 \
         latence-embedder-st latence-disambig-embedding; do
  uv pip install -e "packages/$p"          # WITH heavy deps (gliner, gliner2, sentence-transformers, torch)
done
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print(torch.cuda.get_device_name(0))"
```

## Gate 0 — the batched APIs EXIST on the real checkpoints (run this FIRST; do not skip)

W7 issues the batched forward passes through methods that are **offline-unverifiable** — the
offline fakes were hand-authored to match them, exactly the false-green pattern that already bit
`relation.gliner_relex` once (see `scripts/validate-relex-on-gpu.md`: the provider called a
plausible-but-nonexistent `predict_with_relations`, the offline fake mirrored it, offline was green,
and it only failed on real weights). So the offline suite **cannot** catch a wrong/renamed method.
This gate introspects the REAL checkpoints and asserts each batched entry point exists with the
pinned shape **before any timing**. If any assertion fails, STOP — the pinned API is wrong; do NOT
fabricate throughput numbers, re-introspect the model and fix the provider call. (Mirrors
`validate-relex-on-gpu.md` step 2, extended to every W7 batched provider.)

```bash
python - <<'PY'
import inspect

# 1. relation.gliner_relex — the fused batched entry point W7 calls: model.inference(texts, ...).
from gliner import GLiNER
relex = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0")
assert hasattr(relex, "inference"), (
    "relation.gliner_relex batches via model.inference(texts, labels=..., relations=..., "
    "threshold=..., return_relations=True) — the real checkpoint has NO 'inference' method. "
    "Re-introspect: single-text predict_relations(text, ...) -> (entities, relations) is the "
    "W1-verified fallback; either the batched method is named differently or you must loop "
    "predict_relations. Do NOT time until this passes."
)
sig = inspect.signature(relex.inference)
print("relex.inference signature:", sig)
# Smoke the exact call W7 issues (two-text batch) and assert the (ents_per_text, rels_per_text) shape.
ents_pt, rels_pt = relex.inference(
    ["Alice works for Acme.", "Bob leads Globex."],
    labels=["person", "organization"], relations=["works for"],
    threshold=0.3, return_relations=True,
)
assert len(ents_pt) == len(rels_pt) == 2, f"expected two parallel per-text lists, got {len(ents_pt)}/{len(rels_pt)}"
print(f"relex.inference OK: 2 texts -> {len(ents_pt)} entity-lists + {len(rels_pt)} relation-lists")

# 2. redaction.gliner2 — the batched entry point: model.batch_extract_entities(texts, ...).
from gliner2 import GLiNER2
g2 = GLiNER2.from_pretrained("fastino/gliner2-base")   # (use the checkpoint your redact stage pins)
assert hasattr(g2, "batch_extract_entities"), (
    "redaction.gliner2 batches via model.batch_extract_entities(texts, entity_types, threshold=...) "
    "— the real checkpoint has NO 'batch_extract_entities' method. Re-introspect and fix the "
    "provider call before timing."
)
print("gliner2.batch_extract_entities signature:", inspect.signature(g2.batch_extract_entities))

# 3. entity.gliner (redaction.gliner_pii shares it) — the documented real GLiNER batch method.
ner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
assert hasattr(ner, "batch_predict_entities"), "GLiNER.batch_predict_entities missing — re-introspect."
print("gliner.batch_predict_entities signature:", inspect.signature(ner.batch_predict_entities))

print("\nGATE 0 PASS: every W7 batched entry point exists on the real weights.")
PY
```

## Run — the A/B re-time (batch 1 vs batch 16)

Time each stage at `batch_size` 1 and 16 over the SAME lbbw corpus. Because batching changes only
how many texts share a forward pass, the emitted records are byte-identical between the two runs —
so this is a pure throughput A/B (verify the record equality too, as a live equivalence check).

```bash
# 1. relation.gliner_relex (fused NER+RE) — the `extract` stage.
for B in 1 16; do
  LATENCE_CUDA=1 uv run latence stack validate stacks/gpu-sota-pod.yaml \
      --set entities.config.batch_size=$B \
      --corpus /workspace/corpora/lbbw --time-stages   # (use the real lbbw path + your stage/flags)
done

# 2. redaction.gliner2 — the `redact` stage.
for B in 1 16; do
  LATENCE_CUDA=1 uv run latence stack validate stacks/gpu-sota-pod.yaml \
      --set redact.config.batch_size=$B \
      --corpus /workspace/corpora/lbbw --time-stages
done
```

If a `--time-stages` flag / `--set` override is not wired for your CLI build, run the two stages
directly with the batched providers (mirrors `scripts/_pod_probe_gliner.py`), timing the model calls:

```python
import time
from latence_relation_gliner.provider import GlinerRelexExtractor
from latence_pii_gliner2.provider import Gliner2PIIRedactor
# ... build the real lbbw ChunkRecords / DocumentRecords (24 chunks / 3 docs) ...

for B in (1, 16):
    relex = GlinerRelexExtractor({"labels": [...], "relation_labels": [...],
                                  "threshold": 0.3, "device": "cuda", "batch_size": B})
    t = time.perf_counter(); fused = relex.extract_fused(chunks); dt = time.perf_counter() - t
    print(f"extract  batch={B:2d}: {dt:6.2f}s  mentions={len(fused.mentions)} relations={len(fused.relations)}")

for B in (1, 16):
    pii = Gliner2PIIRedactor({"pii_labels": [...], "threshold": 0.4, "device": "cuda", "batch_size": B})
    t = time.perf_counter(); recs = list(pii.redact(docs)); dt = time.perf_counter() - t
    spans = sum(len(r.pii_spans) for r in recs)
    print(f"redact   batch={B:2d}: {dt:6.2f}s  docs={len(recs)} spans={spans}")
```

## What to record (fold into the Baseline bar)

| Stage    | Provider              | Baseline (batch 1) | Batched (batch 16) | Speed-up |
|----------|-----------------------|--------------------|--------------------|----------|
| `redact` | `redaction.gliner2`   | 60.5 s / 3 docs    | _(measure)_        | _(×)_    |
| `extract`| `relation.gliner_relex` | 26 s / 24 chunks | _(measure)_        | _(×)_    |

- **Equivalence (live):** assert the batch-16 run emits the SAME mentions/relations/PIISpans
  (offsets/labels/spans/order) as the batch-1 run — the offline equivalence test, re-confirmed on
  real weights. Any divergence is a defect, not a speed knob.
- **Peak VRAM at batch 16** (`nvidia-smi` during the run) — confirm it stays bounded to the batch,
  and reconcile with each Provider's declared `ProviderProfile.memory_mb` (bump the declared value
  if the measured batched footprint is materially higher — honesty over optimism).
- Expect a **multi-× drop** (~5–10× is the design target on a GPU); if the measured speed-up is
  small, note the real number and the likely cause (small corpus, CPU-bound pre/post, tokenizer),
  never round it up.
