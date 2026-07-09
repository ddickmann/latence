# W2-gpu-sota-fix — the blessed gpu-sota.yaml never ran end-to-end; fix the learned-stage wiring

**The consolidated pod sweep ran the full gpu-sota stack on a real GPU for the first time and it FAILED
three times before passing** — the blessed `stacks/gpu-sota.yaml` had never been executed end-to-end
(CPU `stack validate` skips every `device: cuda` stage with-flag, so the wiring was never exercised).
Real bugs found on the pod, in order:
1. **`relations` stage config** — passed `relations:` but the provider needs `labels` + `relation_labels`
   (fixed in #129, but see #3 — the deeper issue).
2. **`redact` stage** — `redaction.gliner_pii` requires a non-empty `pii_labels` list; the stage had
   only `{device: cuda}` → `ValueError` at run.
3. **Fused-provider miswiring (the architectural bug)** — the stack declared BOTH an `entities`
   (`entity.gliner`) stage AND a `relations` (`relation.gliner_relex`) stage. But `gliner_relex` is a
   **fused** provider (`FusedEntityRelationExtractor` — it does NER+RE in ONE pass via `extract_fused`);
   it does NOT satisfy the standalone `RelationExtractor.extract(chunks, mentions)` seam, so the run
   raised `ContractError: 'relation.gliner_relex' does not satisfy the RelationExtractor capability`.
   Pairing it with a separate `entity.gliner` is also double-NER. The correct wiring is ONE
   `capability: fused_entity_relation` stage (runner.py routes it to both downstream consumers).

## What to fix
1. **Rewire `stacks/gpu-sota.yaml` extraction to the fused stage** — replace the separate `entities`
   + `relations` stages with a single:
   ```yaml
   - name: extract
     capability: fused_entity_relation
     provider: relation.gliner_relex
     depends_on: [content_screen]
     config: {device: cuda, labels: [organization, person], relation_labels: [works for, partner of], threshold: 0.3}
   ```
   Update downstream `depends_on`: `profiling: [parse, extract]`, `disambiguate: [extract]`. (A stack
   that wants standalone learned NER can still use `entity.gliner` + a non-fused RE provider; but the
   blessed SOTA path is the fused pass.)
2. **`redact` stage** — add `pii_labels: [person, email, phone number, address, credit card number,
   social security number]` to the `redaction.gliner_pii` config.
3. **Keep `disambiguate` KB `entity_type: organization`** (lower-case, matching the fused stage's
   lower-case labels — the case-sensitive KB linker, #129).
4. **Add `stacks/gpu-sota-pod.yaml`** — a self-contained, GPU-pod-runnable variant identical to the
   fixed gpu-sota but with the RAG-export embedder as in-process `embedding.sentence_transformers`
   (the blessed gpu-sota keeps `embedding.endpoint` as the *served* story, which needs a running host).
   This is the stack the maintainer runs on a GPU pod out of the box.
5. **A CPU guard test that would have caught this** — assert every provider named in `gpu-sota.yaml`
   (and the pod variant) actually satisfies the capability its stage declares (load each provider via
   the registry and `isinstance`-check against the stage's capability Protocol), so a fused/standalone
   mismatch fails in CI even though the stage is device-skipped at run. This closes the
   "CPU-validate can't see cuda-stage wiring" gap that hid all three bugs.

## Pod evidence (this fix is already GPU-validated — fold the numbers, never fabricate)
`LATENCE_CUDA=1 latence stack validate stacks/gpu-sota-pod.yaml` on the RTX 2000 Ada pod (all learned
providers real): **PASS** — documents=3; contracts 28/28 prov+class, offsets aligned, drift clean;
**kg nodes=3 edges=18 evidence_coverage=1.00**; rag 3 rows no-PII-leak; report schema v18, 12 stage
metrics; graceful_failure 3 quarantined + 1 PARSE_ERROR handled; **determinism 6 artifacts
byte-identical across two runs**; resume byte-identical.

## Definition of Done
- `gpu-sota.yaml` uses the fused `extract` stage + `pii_labels` + lower-case KB; downstream depends_on
  corrected. `gpu-sota-pod.yaml` added (in-process ST embedder). Both parse `Pipeline.from_yaml_text`.
- The new capability-satisfaction guard test passes and FAILS if a stage names a provider that doesn't
  satisfy its declared capability (proven by a temporary mis-wire in the test).
- `bash scripts/verify-local.sh` green (CPU stack-validate of both stacks skips cuda stages with-flag,
  but the guard test now checks the wiring). ADR-0036 note: the fused-provider stage rule + the
  CPU-can't-see-cuda-wiring gap + the pod E2E result.

## Reviewer should check
- The fused stage is the correct single-stage wiring (no separate entity+relation stages; no double-NER).
- The capability-satisfaction guard genuinely bites (a fused provider under a `relation_extraction`
  stage fails it) — this is what would have caught the bug in CI.
- `gpu-sota-pod.yaml` is truly self-contained (no served endpoint needed); pod E2E numbers are real.
