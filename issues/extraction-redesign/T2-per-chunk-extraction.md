# T2 — Extraction and redaction run each chunk against its OWN label set

**Stages:** `extract` (`fused_entity_relation.gliner2`), `redact` (`redaction.gliner2`) ·
**Depends on:** T1 · **Severity:** major (extraction quality + the whole zero-shot argument)

## Current behavior (verified)
- Both providers compute their effective labels via
  `effective_labels(config_labels, chunk.induced_labels, slot)`
  (`packages/latence-core/src/latence_core/induced.py`) — the config floor **unioned** with the
  chunk's induced types. Extract call site: `latence-extract-gliner2/.../provider.py:435-438`.
- With corpus-granularity induction (today), `chunk.induced_labels` is the same ~40-type corpus
  schema for every chunk, so gliner2 runs each chunk against config-floor ∪ ~40 types.
- Redact does the same for PII: config PII floor ∪ induced `pii_types`
  (`latence-pii-gliner2/.../provider.py`), with the **ADR-0044 financial floor** enforced at the
  shared `finalize_redacted_chunk` seam.

## The flaw
Zero-shot GLiNER2 precision/recall degrades with label count; ~40 mostly-irrelevant types per chunk
is a needless tax on the extraction the entire KG is built from.

## Proposed change
Once T1 makes induction per-chunk, the `effective_labels` mechanism already yields a small per-chunk
set — but tighten two things:
1. **Keep the config floor minimal (or empty) on the per-chunk path.** The SOTA floor
   (`labels: [person, organization]`, `relation_labels: [works for, partner of]`) exists to guarantee
   *something* is extracted when induction is thin. With per-chunk induction that safety net is less
   needed; make the floor's union onto each chunk **opt-in**, so a chunk's label set is essentially
   *its own induced set*, not induced ∪ a global floor. Document the trade (floor = recall safety net
   vs. label-count purity).
2. **Redact: keep the ADR-0044 financial floor UNIVERSAL, but make induced PII types per-chunk.**
   Critical distinction: the financial-PII floor (card / IBAN / SSN / …) is a **security invariant** —
   it must mask on *every* chunk regardless of what was induced there (a leaked card in an
   "irrelevant" chunk must still be caught). So the floor is not a "relevance" label and must NOT be
   subsetted. Only the *induced* PII types become per-chunk. See T5-note: the floor is ~9 fixed
   labels, so PII label counts stay small regardless.
3. **Guardrailing follows the SAME per-chunk rule where it is learned.** Two distinct things carry the
   "guardrail" name, and they are handled differently:
   - **The learned guardrail categories** — the GLiNER2 model the redactor uses is
     `fastino/GLiNER2-Guardrails-PII-Multi`; its *guardrail* categories (like its PII categories) must
     be **per-chunk induced** and applied per chunk, exactly like PII. And, exactly like the PII floor,
     any **always-on safety categories** (harmful content, injection) form a **universal guard floor**
     that is NOT subsetted per chunk — it applies to every chunk because it is a security invariant.
     So: learned guard categories = per-chunk induced ∪ a small universal guard floor.
   - **The deterministic keyword pre-filter** (`content_screen` / `screening.content_keyword`) is not a
     learned model and has no label-count / OOM concern — it stays a per-chunk deterministic screen,
     **exempt** by nature. It runs before extraction as today.
   This closes the reviewer's "same goes for guardrailing" requirement precisely: wherever a GLiNER2
   model applies guard/PII labels, the per-chunk rule + universal safety floor apply; the keyword
   screen is exempt.

## Acceptance criteria
- [ ] With per-chunk induction, gliner2's per-chunk label count equals the chunk's induced set (+ the
      opt-in floor), NOT the corpus union — asserted by a test that counts labels passed to the model.
- [ ] Redact: the financial floor still masks a planted card/IBAN/SSN in a chunk whose induced PII
      types do NOT include finance (proves the floor stays universal); induced PII types are per-chunk.
- [ ] Guardrail: the learned guard categories are per-chunk; a planted always-on-safety hit (e.g.
      injection) is caught in a chunk whose induced guard categories omit it (proves the guard floor is
      universal); the keyword `content_screen` is unchanged.
- [ ] Predefined-only mode (from T1): extraction/redaction run against the fixed labels per chunk with
      no induction — verified end-to-end.
- [ ] Output for a given chunk is unchanged vs. running that chunk's own labels directly (no regression
      from the floor change beyond the intended scope reduction).
- [ ] `ruff` + `mypy --strict` clean; redaction floor (ADR-0044) invariant preserved; `verify-local` green.

## Affected files
`packages/latence-extract-gliner2/.../provider.py`, `packages/latence-pii-gliner2/.../provider.py`,
`packages/latence-core/src/latence_core/induced.py` (if the floor-union becomes opt-in there),
`setup/wizard.py` (floor config), the extract + pii tests.
