# T6 — Record the decision and correct the docs

**Scope:** ADR + `docs/DEEP-DIVE.md` · **Depends on:** T1–T3 landing (or the design note for T3) ·
**Severity:** minor (but required — the current docs actively sell the thing being replaced)

## The problem
The architecture docs present corpus-level induction as a *feature*:
- `docs/DEEP-DIVE.md` Act I: *"it derives one shared schema for the whole corpus … this is the
  difference between 'we ran NER' and 'we built a graph whose types mean something.'"*
- The inducer provider docstring claims corpus-mode *"keeps entity types consistent across the KG
  with no per-chunk/per-doc drift."*

After this redesign, corpus consistency comes from **downstream type canonicalization (T3)**, and the
extraction path is **per-chunk**. The docs must reflect that, or the pitch overclaims exactly the
mechanism a sharp evaluator will probe (as happened in the review that filed these tickets).

## Proposed change
1. **Amend ADR-0038** (schema induction) — or add a new ADR — recording the decision: per-chunk
   induction for extraction quality; type/entity consolidation in the disambiguation phase; the
   `MAX_LABELS` corpus-cap retired from the extraction path; the cost path (T5). State the trade-offs
   honestly (per-chunk cost vs. corpus-cap recall loss).
2. **Rewrite `docs/DEEP-DIVE.md` Act I induction + extraction paragraphs** to the new model:
   induction is per-chunk (each chunk gets its own relevant labels), extraction runs each chunk
   against its small set, and **consistency is earned in the resolution phase** where entities *and*
   types are canonicalized. Add a short honest note on the cost path (dedup + swappable inducer).
3. Update the inducer provider docstring's "consistent across the KG" claim to point at the
   disambiguation phase instead of the induction cap.
4. Add a one-paragraph **"extraction at scale"** note to the deep-dive so the doc pre-empts the exact
   question (why not 40 labels on every chunk) rather than inviting it.

## Acceptance criteria
- [ ] ADR-0038 (or a new ADR) records the per-chunk + downstream-consolidation decision and its
      trade-offs.
- [ ] `docs/DEEP-DIVE.md` no longer credits corpus-cap induction with KG consistency; it describes
      per-chunk induction + downstream type canonicalization and the cost path.
- [ ] The inducer docstring's consistency claim is corrected.
- [ ] `verify-local` green (docs-only, but the secret-scan + link checks pass).

## Affected files
`docs/adr/0038-*.md` (+ possibly a new ADR), `docs/DEEP-DIVE.md`,
`packages/latence-schema-inducer/.../provider.py` (docstring).
