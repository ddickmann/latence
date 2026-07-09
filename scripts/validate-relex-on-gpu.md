# Validate `relation.gliner_relex` on a real GPU — turnkey pod steps (W1-relex FIX)

`relation.gliner_relex` was a **FALSE GREEN**: the Provider called `model.predict_with_relations(...)`
and parsed a **dict**, but the real `knowledgator/gliner-relex-multi-v1.0` checkpoint has **no such
method** (`AttributeError: 'UniEncoderTokenRelexGLiNER' object has no attribute
'predict_with_relations'`) — the offline fake mirrored a non-existent API. The fix pins the **real**
`predict_relations(text, labels, relations, threshold=...) -> (entities, relations)` **TUPLE** API
and makes both offline stubs faithful to it. This script is the turnkey pod re-validation that proves
the real weights now flow through the fixed call/parse path end to end.

> **The false-green catch (what this run must prove).** The offline slice is validated against a
> **faithful tuple-returning fake** (`predict_relations` returns `(entities, relations)`; a provider
> calling `predict_with_relations` or `.get(...)`-ing the tuple FAILS — `test_dict_parse_path_fails_
> against_tuple_api`). But only a **real GPU run** proves the actual checkpoint exposes
> `predict_relations` with the pinned tuple signature and that the offsets round-trip on real output.
> If step 3 raises `AttributeError` on `predict_relations`, the pinned API is wrong — stop and
> re-introspect the model. Do NOT fabricate relation numbers; they come from this run only.

These steps are turnkey: run them on a single ≥16 GB CUDA GPU (an H100 slice, an A10, an RTX 4090, a
RunPod pod, …). gliner-relex-multi is a compact encoder model, ~2 GB VRAM.

## 0. Prereqs

```bash
nvidia-smi                     # confirm a CUDA device is visible
python --version               # 3.11+
```

## 1. Install the framework + this provider WITH its heavy deps

```bash
# from a clone of the repo on the pod:
cd latence-framework
pip install -e packages/latence-core
pip install -e packages/latence-relation-gliner   # pulls gliner (transformers/torch)
python -c "from gliner import GLiNER; print('gliner ok')"
```

## 2. Sanity: the real checkpoint exposes `predict_relations` (NOT `predict_with_relations`)

```bash
python - <<'PY'
from gliner import GLiNER
m = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0")
assert hasattr(m, "predict_relations"), "real API is predict_relations"
assert not hasattr(m, "predict_with_relations"), "predict_with_relations never existed (the false green)"
print("real API confirmed: predict_relations present, predict_with_relations absent")
PY
```

## 3. Real fused NER+RE — the two expected relations + offset round-trip

Runs the reference Provider (`GlinerRelexExtractor`) against the real weights over the canonical
sentence and checks: (a) `predict_relations` returns a `(entities, relations)` tuple, (b) mentions
round-trip (`mention.text == original[char_start:char_end]`), (c) the two expected relations appear
with head/tail resolved to real emitted mentions.

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"          # route the device seam to cuda
from latence_core.contracts import ChunkRecord, Classification, PageMap, Provenance
from latence_relation_gliner.provider import GlinerRelexExtractor

TEXT = ("IBM works with Alice Johnson in Berlin. Contact her at alice.johnson@example.com "
        "or call +1-202-555-0143 about the Acme partnership.")
CLS = Classification(language="en", category="news", sensitivity="internal")
prov = Provenance(source_uri="file://d1.txt", file_name="d1.txt", file_type="txt",
                  file_size=len(TEXT), document_id="d1", char_start=0, char_end=len(TEXT),
                  page_start=1, page_end=1)
chunk = ChunkRecord(record_id="c1", document_record_id="d1:doc", index=0, content=TEXT,
                    token_count=30, provenance=prov, classification=CLS,
                    page_map=PageMap.from_page_texts([TEXT]).page_map)

# Direct API check: the pinned tuple contract.
from gliner import GLiNER
raw = GLiNER.from_pretrained("knowledgator/gliner-relex-multi-v1.0").predict_relations(
    TEXT, ["organization", "person"], ["works for", "partner of"], threshold=0.3)
assert isinstance(raw, tuple) and len(raw) == 2, f"expected (entities, relations) tuple, got {type(raw)}"
print(f"predict_relations returned a tuple: {len(raw[0])} entities, {len(raw[1])} relations")

# Through the reference Provider.
relex = GlinerRelexExtractor({"labels": ["organization", "person"],
                              "relation_labels": ["works for", "partner of"],
                              "threshold": 0.3, "device": "cuda"})
fused = relex.extract_fused([chunk])
mention_ids = {m.record_id for m in fused.mentions}
ok = len(fused.mentions) >= 1
for m in fused.mentions:
    surf = TEXT[m.provenance.char_start:m.provenance.char_end]
    rt = surf == m.text
    ok &= rt
    print(f"  MENTION {m.label:14s} {m.text!r:22} off[{m.provenance.char_start}:{m.provenance.char_end}] round_trip={'Y' if rt else 'N (%r)' % surf}")
for r in fused.relations:
    endpoints_ok = r.head_mention_id in mention_ids and r.tail_mention_id in mention_ids
    ok &= endpoints_ok and bool(r.label)
    print(f"  RELATION {r.label!r}  {r.head_text!r} -> {r.tail_text!r}  conf={r.confidence:.2f}  endpoints_ok={endpoints_ok}")

# Expected (pod-verified): Alice Johnson works for IBM (~0.96); IBM partner of Acme (~0.79).
labels = {(r.head_text, r.label, r.tail_text) for r in fused.relations}
print("\nrelations:", labels)
print("RESULT:", "PASS" if ok and fused.relations else "FAIL — inspect above")
raise SystemExit(0 if (ok and fused.relations) else 1)
PY
```

## 4. Conformance on a CUDA host

```bash
python -m pytest packages/latence-core/tests/test_conformance.py \
                 packages/latence-core/tests/test_e2e_conformance.py -q \
                 -k gliner_relex
```

## 5. Full gpu-sota stack over a real corpus (optional, end-to-end)

```bash
# W2-gpu-sota-fix: extraction is now ONE fused `extract` stage (capability:
# fused_entity_relation, provider: relation.gliner_relex, labels + relation_labels +
# threshold 0.3) — NOT separate entity+relation stages. On a CUDA host with the heavy
# packages installed the learned stages run for real. Prefer the self-contained pod
# variant (in-process ST embedder, no served endpoint to stand up):
LATENCE_CUDA=1 latence stack validate stacks/gpu-sota-pod.yaml
# or the served-embedder blessed stack (needs an embedding endpoint host up):
LATENCE_CUDA=1 latence stack validate stacks/gpu-sota.yaml
```

Record the real relation set + confidences from step 3 back into ADR-0036 (folded in after the pod
run, flagged "pending pod validation" until then — never fabricated).
