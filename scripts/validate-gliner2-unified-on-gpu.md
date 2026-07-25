# Validate the gliner2 unified extraction + optimized redaction on a real GPU — turnkey pod steps (W9-gliner2-unified)

W9 consolidates extraction onto the fastino **GLiNER2** framework (`gliner2` 1.3.2): one backbone
does entities AND relations AND PII, with **native** `quantize`+`compile` on `from_pretrained`.
This script is the turnkey re-validation that proves the REAL weights flow through the
call/parse/offset path end to end, and that the native quantize+compile give the speedup at
byte-identical output. **The GPU numbers are the maintainer's to fold in — never fabricated here.**

Two claims this run must prove:

1. **`fused_entity_relation.gliner2`** (`latence-extract-gliner2`, `fastino/gliner2-multi-v1`):
   `extract_entities(..., include_spans=True)` + `extract_relations(..., include_spans=True,
   include_confidence=True)` return the pinned nested dict shapes, and the entity **and** relation
   char offsets **round-trip to the real chunk EXACTLY** (the recurring ADR-0031 coordinate trap).
2. **`redaction.gliner2`** under `quantize=True, compile=True`: PII spans are **unchanged** vs the
   plain load (byte-identical), and the native quantize+compile deliver the speedup (pod-measure the
   2.83× — do not fabricate it).

> **The false-green catch (what this run must prove).** Offline both providers are validated against
> a **faithful** fake `gliner2` mirroring the REAL nested `extract_entities` / `extract_relations`
> dict shapes + native `quantize`/`compile` kwargs. But only a real `gliner2` run proves the actual
> checkpoints expose these methods with the pinned kwargs and return the pinned shapes, and that the
> offsets round-trip on real output. If step 3/4 raises (wrong signature) or returns a flat
> list/tuple, the pinned API is wrong — **stop and re-introspect**, do not guess.

`gliner2`/mdeberta-v3 runs on CPU or a modest GPU (`compute="either"`, ~1–2 GB). A CUDA device is
required to exercise the quantize+compile path and the speedup (both default OFF on cpu).

## 0. Prereqs

```bash
python --version               # 3.11+
nvidia-smi                     # a CUDA device is needed for the quantize+compile path + timing
```

## 1. Install the framework + both gliner2 providers WITH their heavy deps

```bash
# from a clone of the repo on the pod:
cd latence-framework
pip install -e packages/latence-core
# Both providers declare gliner2[local] (torch/transformers live only in that extra — the base
# gliner2 dist ships neither) PLUS peft (gliner2's relation inference engine imports it and the
# [local] extra does NOT declare it — pod-found ModuleNotFoundError). An editable install pulls
# the full local-inference stack including peft.
pip install -e packages/latence-extract-gliner2   # fused NER+RE (fastino/gliner2-multi-v1) + peft
pip install -e packages/latence-pii-gliner2        # PII (fastino/GLiNER2-Guardrails-PII-Multi) + peft
python -c "from gliner2 import GLiNER2; import peft; print('gliner2 + peft ok')"
```

## 2. Verify the licenses on the pod (weights + library) — record cited, never guessed

```bash
python - <<'PY'
import importlib.metadata as md
meta = md.metadata("gliner2")
print("gliner2 version:", meta["Version"])
print("gliner2 License:", meta.get("License"))
print("gliner2 classifiers:", [c for c in meta.get_all("Classifier", []) if "License" in c])
# Weights: fastino/gliner2-multi-v1 + fastino/GLiNER2-Guardrails-PII-Multi are Apache-2.0, NOT gated
# (HF model cards "License: apache-2.0"). Re-confirm from each downloaded snapshot's README.
PY
```

## 3. Real fused NER+RE offsets round-trip on a real chunk (the core extraction claim)

Runs `Gliner2FusedExtractor` against the real `fastino/gliner2-multi-v1` weights over a real German
chunk and checks: (a) `extract_entities`/`extract_relations` return the pinned nested dicts, (b) each
mention slice round-trips (`chunk.content[local]` == mention text; provenance slice == text through
the document offsets), (c) every emitted relation resolves to real head/tail mentions.

```bash
python - <<'PY'
import os
os.environ["LATENCE_CUDA"] = "1"          # route to cuda; quantize/compile default ON there
from latence_core.contracts import ChunkRecord, Classification, PageMap, Provenance

# A real German banking-style chunk (offsets must round-trip on the REAL model output).
TEXT = ("Alice Johnson arbeitet bei der IBM Deutschland GmbH in Stuttgart. "
        "Die IBM Deutschland GmbH ist Partner der Acme AG.")
prov = Provenance(source_uri="file://c1.txt", file_name="c1.txt", file_type="txt",
                  file_size=len(TEXT), document_id="c1", char_start=0, char_end=len(TEXT),
                  page_start=1, page_end=1)
chunk = ChunkRecord(record_id="c1#chunk-0", document_record_id="c1", index=0, provenance=prov,
                    classification=Classification(language="de", sensitivity="internal"),
                    content=TEXT, token_count=len(TEXT.split()),
                    page_map=PageMap.from_page_texts([TEXT]).page_map)

# Direct API check: the pinned nested contracts (include_spans REQUIRED for char offsets).
from gliner2 import GLiNER2
m = GLiNER2.from_pretrained("fastino/gliner2-multi-v1", quantize=True, compile=True)
ents = m.extract_entities(TEXT, ["organization", "person"], threshold=0.3,
                          include_spans=True, include_confidence=True)
rels = m.extract_relations(TEXT, ["works for", "partner of"], threshold=0.3,
                           include_spans=True, include_confidence=True)
assert isinstance(ents, dict) and isinstance(ents.get("entities"), dict), \
    f"expected {{'entities': {{label: [rows]}}}}, got {type(ents)}"
assert isinstance(rels, dict) and isinstance(rels.get("relation_extraction"), dict), \
    f"expected {{'relation_extraction': {{rel: [rows]}}}}, got {type(rels)}"
print("entities labels:", list(ents["entities"]))
print("relations types:", list(rels["relation_extraction"]))
# Pin one entity row + one relation head/tail carry start/end (the offsets we ground on):
for label, rows in ents["entities"].items():
    for r in rows:
        assert "start" in r and "end" in r, f"entity row has no span: {r}"
        assert TEXT[r["start"]:r["end"]] == r["text"], f"offset mismatch: {r}"
for rel, rows in rels["relation_extraction"].items():
    for r in rows:
        for role in ("head", "tail"):
            ep = r[role]
            assert "start" in ep and "end" in ep, f"{role} has no span: {ep}"

# Through the reference Provider: mentions + relations with EXACT offsets, endpoints resolved.
from latence_extract_gliner2.provider import Gliner2FusedExtractor
prov_x = Gliner2FusedExtractor({"labels": ["organization", "person"],
                                "relation_labels": ["works for", "partner of"],
                                "threshold": 0.3, "device": "cuda"})
result = prov_x.extract_fused([chunk])
for mn in result.mentions:
    slice_ = TEXT[mn.provenance.char_start:mn.provenance.char_end]
    assert slice_ == mn.text, f"OFFSET DRIFT: {slice_!r} != {mn.text!r}"
    print(f"  MENTION {mn.label:14s} off[{mn.provenance.char_start}:{mn.provenance.char_end}] {mn.text!r}")
mention_ids = {mn.record_id for mn in result.mentions}
for rel in result.relations:
    assert rel.head_mention_id in mention_ids and rel.tail_mention_id in mention_ids, \
        "relation endpoint does not resolve to an emitted mention"
    print(f"  RELATION {rel.label:12s} {rel.head_text!r} -> {rel.tail_text!r} conf={rel.confidence:.3f}")
ok = bool(result.mentions) and all(
    TEXT[mn.provenance.char_start:mn.provenance.char_end] == mn.text for mn in result.mentions)
print("RESULT:", "PASS" if ok else "FAIL — inspect above")
raise SystemExit(0 if ok else 1)
PY
```

## 4. `redaction.gliner2` PII unchanged under quantize+compile + the speedup (the redact fix)

Runs `Gliner2PIIRedactor` twice — plain (`quantize=False, compile=False`) vs optimized
(`quantize=True, compile=True`) — and asserts **byte-identical** masked output + PII spans, then
times both to measure the speedup (pod-measure it; do NOT fabricate).

```bash
python - <<'PY'
import os, time
os.environ["LATENCE_CUDA"] = "1"
from latence_core.contracts import Classification, DocumentRecord, PageMap, Provenance
from latence_pii_gliner2.provider import Gliner2PIIRedactor

TEXT = ("IBM works with Alice Johnson in Berlin. Contact alice@example.com "
        "or +1-202-555-0143.")
prov = Provenance(source_uri="file://d1.txt", file_name="d1.txt", file_type="txt",
                  file_size=len(TEXT), document_id="d1", char_start=0, char_end=len(TEXT),
                  page_start=1, page_end=1)
doc = DocumentRecord(record_id="d1", provenance=prov,
                     classification=Classification(language="en", sensitivity="public"),
                     content=TEXT, page_map=PageMap.from_page_texts([TEXT]).page_map)
labels = ["person", "email", "phone number"]

def run(cfg):
    red = Gliner2PIIRedactor({"pii_labels": labels, "threshold": 0.4, "device": "cuda", **cfg})
    t0 = time.perf_counter()
    rec = next(iter(red.redact([doc])))
    return rec, time.perf_counter() - t0

plain, t_plain = run({"quantize": False, "compile": False})
opt,   t_opt   = run({"quantize": True,  "compile": True})   # warmup includes compile; re-time below
_, t_opt2      = run({"quantize": True,  "compile": True})

# BYTE-IDENTICAL: the optimized load must not change a single span or the masked text.
assert opt.masked_content == plain.masked_content, "quantize+compile CHANGED the masked output!"
assert [(s.pii_type, s.char_start, s.char_end) for s in opt.pii_spans] == \
       [(s.pii_type, s.char_start, s.char_end) for s in plain.pii_spans], "PII spans differ!"
print("byte-identical PII under quantize+compile: PASS")
print(f"plain: {t_plain*1000:.1f} ms  optimized(first): {t_opt*1000:.1f} ms  "
      f"optimized(warm): {t_opt2*1000:.1f} ms")
print("Fold the REAL speedup (plain / optimized-warm) into ADR-0036 — do NOT fabricate it.")
PY
```

## 4b. Inference-time compile degradation (the pod-found gap — must NOT crash)

`torch.compile` compiles **lazily on the first forward**, so on an env whose inductor/dynamo backend
cannot compile the model, the failure is a `torch._dynamo.exc.BackendCompilerFailed` raised INSIDE
`extract_entities` / `extract_relations` / `batch_extract_entities` — NOT at load. The W9 fix catches
that class of fault ONCE, re-loads the **uncompiled** model (quantize kept), and continues — the
provider must **never** surface a `ProviderError` for a compile fault. On an env where compile DOES
work this run simply succeeds compiled; on an env where it does not (e.g. torch 2.4.1 + this model),
it must **degrade gracefully and still return records**, not crash.

```bash
python - <<'PY'
import os, logging
os.environ["LATENCE_CUDA"] = "1"
logging.basicConfig(level=logging.WARNING)   # the fallback logs a "re-loading uncompiled" warning
from latence_core.contracts import ChunkRecord, Classification, PageMap, Provenance
from latence_extract_gliner2.provider import Gliner2FusedExtractor

TEXT = "Alice Johnson arbeitet bei der IBM Deutschland GmbH in Stuttgart."
prov = Provenance(source_uri="file://c.txt", file_name="c.txt", file_type="txt",
                  file_size=len(TEXT), document_id="c", char_start=0, char_end=len(TEXT),
                  page_start=1, page_end=1)
chunk = ChunkRecord(record_id="c#chunk-0", document_record_id="c", index=0, provenance=prov,
                    classification=Classification(language="de", sensitivity="internal"),
                    content=TEXT, token_count=len(TEXT.split()),
                    page_map=PageMap.from_page_texts([TEXT]).page_map)

# compile forced ON: if the backend cannot compile, the FIRST forward raises BackendCompilerFailed;
# the provider must degrade to the uncompiled model and STILL return records (never a ProviderError).
prov_x = Gliner2FusedExtractor({"labels": ["organization", "person"],
                                "relation_labels": ["works for"], "threshold": 0.3,
                                "device": "cuda", "quantize": True, "compile": True})
result = prov_x.extract_fused([chunk])   # must NOT raise ProviderError on a compile-incapable backend
print("mentions:", [m.text for m in result.mentions])
print("compile fell back to uncompiled:", prov_x._compile_disabled,
      "(True iff the backend could not compile — a graceful degrade, not a crash)")
assert result.mentions, "extraction returned nothing"
print("RESULT: PASS — no crash; records returned (compiled or gracefully uncompiled)")
PY
```

## 5. Conformance on the pod

```bash
python -m pytest packages/latence-core/tests/test_conformance.py \
                 packages/latence-core/tests/test_e2e_conformance.py -q \
                 -k "gliner2 or fused_entity_relation"
```

## 6. Bake-off + blessed-stack E2E over a real corpus (optional)

```bash
# The blessed SOTA path now defaults extraction to fused_entity_relation.gliner2 (device: cuda).
LATENCE_CUDA=1 latence stack validate stacks/gpu-sota-pod.yaml
```

Record the real offsets/relations from step 3, the byte-identical-under-optimization result + the
measured speedup from step 4, back into ADR-0036 (folded in after the pod run, flagged "pending pod
validation" until then — never fabricated).

---

## 7. T7 — the numbers this redesign owes the pod (eager vs compiled, peak VRAM, backoff count)

Three claims of `issues/extraction-redesign/T7-efficient-batching.md` can only be settled on a real
GPU. They are **not** measured offline and are **not** guessed here — run this and fold the real
numbers into `PERF-RESULTS.md` / ADR-0036.

**Claim A — one native batched forward per label BUCKET, not per chunk.** Offline this is asserted
against a faithful fake by counting `batch_extract_entities` calls; on the pod, count them against
the real model over a real corpus.

**Claim B — eager vs compiled throughput.** T7 unlocked gliner2's native `compile` because label-set
bucketing + micro-batching pin the two dimensions whose variation made `torch.compile(dynamic=True)`
recompile-thrash. That claim is a *throughput* claim and needs a measured pair.

**Claim C — peak VRAM is a fraction of the corpus-mode ~21.6 GiB, and the reactive micro-batch/OOM
backoff fires ZERO (or near-zero) times.** Per-chunk induction collapses the label dimension of the
`batch × positions × labels` span-scoring tensor — the ~17 GiB spike that OOM'd redact. If the
backoff still fires often, the root fix did NOT land and the micro-batch net is carrying the run.

```bash
# Count every native batched forward + every reactive backoff, then A/B compile on the SAME corpus.
python - <<'PY'
import os, time, logging
os.environ["LATENCE_CUDA"] = "1"
import torch
from latence_extract_gliner2.provider import Gliner2FusedExtractor

# A real, VARIED-length chunk stream with per-chunk induced label sets (that is the regime under
# test — a uniform-length uniform-label corpus proves nothing about the thrash this replaced).
chunks = ...   # load the RFC corpus chunks the redesign run produced (parquet under the run dir)

def run(compile_on: bool) -> tuple[float, int, int]:
    torch.cuda.reset_peak_memory_stats()
    prov = Gliner2FusedExtractor({
        "labels": ["person", "organization"], "relation_labels": ["works for"],
        "threshold": 0.3, "device": "cuda", "quantize": True, "compile": compile_on,
    })
    model_calls = {"n": 0}
    t0 = time.perf_counter()
    result = prov.extract_fused(chunks)
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"compile={compile_on}: {elapsed:.1f}s, peak {peak:.2f} GiB, "
          f"mentions={len(result.mentions)}, relations={len(result.relations)}, "
          f"compile_fell_back={prov._compile_disabled}")
    return elapsed, peak, len(result.mentions)

# Claim C: watch for the reactive backoff. It logs nothing by itself, so count the OOM retries by
# tapping the perf seam's cache-release (called ONLY on the OOM path).
import latence_core.providers.perf as perf
backoffs = {"n": 0}
_real = perf.empty_cuda_cache
def counted(**kw):
    backoffs["n"] += 1
    return _real(**kw)
perf.empty_cuda_cache = counted

eager_s, eager_peak, eager_mentions = run(False)
comp_s,  comp_peak,  comp_mentions  = run(True)

print(f"T7 compile speedup: {eager_s / comp_s:.2f}x   (eager {eager_s:.1f}s vs compiled {comp_s:.1f}s)")
print(f"T7 peak VRAM: eager {eager_peak:.2f} GiB / compiled {comp_peak:.2f} GiB "
      f"(corpus-mode baseline was 21.6 GiB)")
print(f"T7 OOM backoff fired {backoffs['n']} time(s) — MUST be 0 or near-0; a high count means the "
      "root fix did not land and the reactive net is carrying the run")
assert eager_mentions == comp_mentions, "compile changed the record count — NOT throughput-only"
PY
```

**Report honestly.** If compiled is *slower*, say so and pin `compile: false` in the blessed stack —
the point of T7 is that bucketing makes compile *safe to try*, not that it is guaranteed to win.
Likewise, if the backoff fires more than a handful of times, the label dimension did not collapse
as designed; investigate the induced label-set sizes before claiming the OOM is root-fixed.
