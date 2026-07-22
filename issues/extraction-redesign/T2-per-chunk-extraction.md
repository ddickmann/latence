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
3. `guardrail` (`content_screen` / `screening.content_keyword`) is deterministic keyword screening,
   not a GLiNER2 model — **no change**; it is unaffected by label count. (Noted so the redesign scope
   is explicit.)

## Acceptance criteria
- [ ] With per-chunk induction, gliner2's per-chunk label count equals the chunk's induced set (+ the
      opt-in floor), NOT the corpus union — asserted by a test that counts labels passed to the model.
- [ ] Redact: the financial floor still masks a planted card/IBAN/SSN in a chunk whose induced PII
      types do NOT include finance (proves the floor stays universal); induced PII types are per-chunk.
- [ ] Output for a given chunk is unchanged vs. running that chunk's own labels directly (no regression
      from the floor change beyond the intended scope reduction).
- [ ] `ruff` + `mypy --strict` clean; redaction floor (ADR-0044) invariant preserved; `verify-local` green.

## Affected files
`packages/latence-extract-gliner2/.../provider.py`, `packages/latence-pii-gliner2/.../provider.py`,
`packages/latence-core/src/latence_core/induced.py` (if the floor-union becomes opt-in there),
`setup/wizard.py` (floor config), the extract + pii tests.
