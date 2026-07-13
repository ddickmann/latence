# W9-gliner2-unified — gliner2 as the unified, optimized extraction+PII engine (retire relex)

**One fast model/library for NER + RE + PII, with native quantize+compile.** Pod-measured: gliner2
1.3.2 does entities AND relations AND PII in one backbone; `quantize=True, compile=True` gives
**2.83× (PII) / 1.74× (combined NER+RE), byte-identical output**. Consolidating extraction onto
gliner2 retires the separate `relation.gliner_relex` model (`knowledgator/gliner-relex`) — one fewer
architecture, more throughput, less handling. Dennis's call, pod-validated.

## Pod-validated gliner2 1.3.2 API (REAL — mirror it exactly in the offline fake)
- `GLiNER2.from_pretrained(model_id, quantize=True, compile=True)` — fp16 + torch.compile; loads clean;
  identical output. Also `.cuda()`, `.bfloat16()`, `set_attn_implementation(...)`, `use_kernels`.
- `extract_entities(text, entity_types, threshold=0.5, include_confidence=False, include_spans=False, format_results=True) -> Dict`
  → default `{"entities": {label: [text_str, ...]}}`. **With `include_spans=True` you get start/end char offsets** (VERIFY the exact key shape on the pod — the framework REQUIRES char offsets for provenance/evidence).
- `extract_relations(text, relation_types, threshold=0.5, include_confidence=False, include_spans=False) -> Dict`
  → `{"relation_extraction": {rel_type: [{"head": {"text","confidence"[,"start","end"]}, "tail": {...}}]}}`.
- unified `extract(text, schema, ...)` + `create_schema().entities({...}).classification(...).structure(...)`.
- Models: **`fastino/gliner2-multi-v1`** (NER+RE, Apache-2.0 — VERIFY license weights+code per research-diligence/ADR-0012), `fastino/GLiNER2-Guardrails-PII-Multi` (PII, Apache-2.0, already in roster).

## What to build
1. **`redaction.gliner2` optimized deployment (the redact fix):** add config `quantize` (default True on
   cuda) + `compile` (default True on cuda) → `GLiNER2.from_pretrained(id, quantize=…, compile=…)`. This
   SUPERSEDES the W7 batching that REGRESSED redact (60→95s, padding waste) — measured 2.83× at identical
   PII. Keep output/offset contract byte-identical (pod-verify PII spans unchanged).
2. **New fused extraction provider `fused_entity_relation.gliner2`** (package `latence-extract-gliner2` or
   extend the pii-gliner2 package) satisfying the SAME `fused_entity_relation` Capability that
   `relation.gliner_relex` satisfies — so it's a drop-in swap. Uses `gliner2-multi-v1`:
   `extract_entities(..., include_spans=True)` → Entity mentions with EXACT char offsets;
   `extract_relations(..., include_spans=True, include_confidence=True)` → relations as head→tail with
   confidence, mapped to the framework's relation/mention contract (same shape relex emits: mentions +
   relations + evidence + provenance + offsets). `quantize`/`compile` config knobs (default on, cuda).
   Reads induced ∪ config labels (the W4 label-source seam) exactly like relex.
   **CRITICAL (offsets): the char offsets from include_spans MUST map back to the chunk exactly** — this
   is where the ADR-0031 coordinate bugs bit before; the offline fake must mirror the REAL span shape and
   a test must assert round-trip offset correctness. Pod-validate the real spans.
3. **Default swap + retire relex:** make `fused_entity_relation.gliner2` the DEFAULT extraction provider
   in the blessed stacks (default.yaml, gpu-sota, lbbw); keep `relation.gliner_relex` in the roster (don't
   break existing stacks) but mark it superseded in ADR-0036 (gliner2 is the unified default).

## Anti-false-green + tests
- Offline fake `GLiNER2` mirrors the REAL return shapes ABOVE (entities dict, relation_extraction dict,
  spans when include_spans=True) — NOT a fictional shape (the paddle/relex false-green lesson). A test
  asserts the fused provider maps gliner2's head/tail/spans to the framework contract record-for-record
  with correct offsets; a variable-offset fixture catches a coordinate bug.
- `quantize`/`compile` config resolve correctly (on cuda default true; cpu false); guarded so a
  compile/quantize failure degrades to plain load (never crash).
- Conformance C1–C6 for the new provider; roster+1; `bash scripts/verify-local.sh` GREEN; ADR-0036 note.

## Definition of Done
- redaction.gliner2 loads with quantize+compile (config, default-on cuda), PII byte-identical (pod).
- New `fused_entity_relation.gliner2` (gliner2-multi-v1) emits entities+relations with EXACT offsets,
  drop-in for relex, quantize+compile default-on; blessed stacks default to it; relex kept-but-superseded.
- Offline fake mirrors REAL gliner2 shapes; offset round-trip tested; conformance+roster+verify-local green.
- Pod validation (maintainer): real gliner2-multi-v1 NER+RE offsets round-trip on real German chunks;
  redact PII spans unchanged under quantize+compile; numbers folded in. NO fabricated numbers.

## Reviewer should check
- The offline fake mirrors gliner2's REAL output (entities/relation_extraction/spans) — not invented.
- Entity/relation char offsets round-trip to the chunk EXACTLY (the recurring coordinate trap).
- quantize/compile guarded (failure → plain load); PII/entity output identical vs unoptimized (determinism).
- relex not deleted (existing stacks unbroken); gliner2 is the documented unified default.
