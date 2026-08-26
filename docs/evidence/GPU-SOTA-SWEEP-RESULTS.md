# Consolidated gpu-sota Pod Sweep — Results (2026-07-09)

**Purpose.** Run the full **learned/GPU** stack end-to-end on a real GPU with every learned provider on
real weights, prove it works together (real KG + Quality Report + determinism), and surface anything
that only a real GPU run exposes. Signed with real pod numbers — nothing fabricated.

**Rig.** RunPod RTX 2000 Ada (16 GB), Py3.11, torch 2.4.1+cu124, transformers 5.13, `gliner` 0.2.27,
sentence-transformers 5.6, PyG 2.8 + torch_scatter 2.1.2. Corpus: the bundled deterministic messy
corpus (3 documents + dangerous fixtures), staged by `latence stack validate`.

## Phase 1 — per-provider de-risk (real weights, offset round-trip)

The anti-false-green rule: a learned provider that passes offline tests can still be non-functional on
real weights (as PaddleOCR was). Each learned provider was run on the pod against its real checkpoint
and checked for correct output shape + char-offset round-trip **before** trusting the end-to-end run.

| Provider | Model (license) | Result |
|---|---|---|
| `entity.gliner` | urchade/gliner_multi-v2.1 (Apache-2.0) | ✅ PASS — IBM/Alice/Berlin/Acme, offsets round-trip |
| `redaction.gliner_pii` | urchade/gliner_multi_pii-v1 (Apache-2.0) | ✅ PASS — person/email/phone masked, offsets round-trip |
| `disambiguation.embedding` + `embedding.sentence_transformers` | multilingual-e5-small (MIT) | ✅ PASS — pod-validated in #128 (context ER, no over-merge) |
| `relation.gliner_relex` | knowledgator/gliner-relex-multi-v1.0 (Apache-2.0) | ❌ **FALSE-GREEN → FIXED (#129)** |

**The false-green (`relation.gliner_relex`).** The provider called `model.predict_with_relations(...)`
and parsed a dict — a method that does **not exist** on the real model. The real API is
`predict_relations(text, labels, relations, threshold) -> (entities, relations)` **tuple**. The offline
stub had invented the fictional method, so every offline test passed against an API that isn't real —
the exact PaddleOCR class. Fixed (#129) + pod-validated: *Alice Johnson —works for→ IBM* (0.96),
*IBM —partner of→ Acme* (0.79), all relation endpoints resolve.

## Phase 2 — full gpu-sota end-to-end (all learned providers, one pipeline, real GPU)

Running the full stack on the GPU surfaced **three more real bugs** the CPU gate could never see (CPU
`stack validate` skips every `device: cuda` stage with-flag, so the learned-stage wiring was never
exercised):

1. `redact` stage missing the required `pii_labels` list → `ValueError`.
2. **Fused-provider miswiring** — gpu-sota declared BOTH `entity.gliner` AND `relation.gliner_relex`
   as separate stages, but `gliner_relex` is a *fused* NER+RE provider that doesn't satisfy the
   standalone `RelationExtractor` seam → `ContractError`. Correct wiring is ONE
   `capability: fused_entity_relation` stage.
3. (Earlier) `relations` stage config keys (fixed #129).

Fixed in `stacks/gpu-sota-pod.yaml` (fused extraction stage + `pii_labels` + in-process ST embedder).
`LATENCE_CUDA=1 latence stack validate stacks/gpu-sota-pod.yaml` — **PASS**:

| check | result | detail |
|---|---|---|
| contracts | ✅ PASS | completeness 28/28 provenance, 28/28 classification, offsets_aligned, drift clean |
| **kg** | ✅ **PASS** | **nodes=3, edges=18, evidence_coverage=1.00, exported** |
| rag | ✅ PASS | 3 corpus rows, 1 JSONL export, **no PII leak** |
| report | ✅ PASS | schema v18, 12 stage metrics, profile columns, no PII leak |
| graceful_failure | ✅ PASS | 3 quarantined + 1 PARSE_ERROR (dangerous fixtures handled, never crashed) |
| **determinism** | ✅ **PASS** | 6 export artifacts **byte-identical across two fresh runs** |
| resume | ✅ PASS | 6 export artifacts byte-identical after same-run_id resume |
| device_honesty | ✅ PASS | no GPU stage silently skipped |

**Bottom line:** the learned/SOTA pipeline — GLiNER fused NER+RE → GLiNER-PII redaction → e5
context-embedding entity resolution → graph assembly + KG/RAG export — runs end-to-end on real GPU
weights, produces a fully-evidenced deterministic KG, and leaks no PII.

## Honest caveats / not-yet-covered
- **Scale.** This is the 3-document bundled corpus, not 100k/1M. Throughput/memory at scale is a
  separate perf-rig item (backlog), unchanged by this sweep.
- **Served embeddings.** `embedding.endpoint` (the served-embeddings provider in the blessed
  `gpu-sota.yaml`) needs a running host; the pod E2E used in-process `embedding.sentence_transformers`
  (`gpu-sota-pod.yaml`). Endpoint-mode validation is infra-gated (needs a served model).
- **OCR path.** `parser.lighton` / `parser.glm` (pod-validated in #125/#126) are not in this stack
  (it uses `parser.pdfplumber`); the OCR parsers have their own turnkey pod scripts.
- **Link prediction.** The graph-completion capability (ULTRA/PyKEEN) is in flight (seam Slice 1 +
  learned Slice 2); real ULTRA zero-shot inference is already proven on this pod (ranks a held-out
  edge #1 on a fresh KG).

## Repo fixes this sweep produced
- #129 — `relation.gliner_relex` real `predict_relations` tuple API (merged, pod-validated).
- W2-gpu-sota-fix — fused-stage rewire + `pii_labels` + `gpu-sota-pod.yaml` + a CPU
  capability-satisfaction guard test (so the fused/standalone mismatch fails in CI, not just on a GPU).
