# T1 — Per-chunk label induction as the extraction path

**Stage:** `induce` (`label_inducer.llm`) · **Depends on:** T3 (type canonicalization must land with/before
this flips the default, or the KG loses type consistency) · **Severity:** major (extraction quality)

## Current behavior (verified)
- `packages/latence-schema-inducer/src/latence_schema_inducer/provider.py` supports three
  granularities; the enterprise-SOTA wizard emits **`granularity: corpus`** (`setup/wizard.py`,
  induce block).
- Corpus mode induces per-document, merges all documents' label sets, ranks by document frequency,
  and **caps each slot at `MAX_LABELS = 40`** (`sanitize.py:41`). Overflow labels are dropped + logged.
- The single merged schema is broadcast to **every** chunk's `induced_labels`.
- Per-chunk induction ALREADY exists: `provider.py:497 _induce_per_chunk`, reached by
  `granularity == "chunk"` (`provider.py:361`). It makes one LLM call per chunk and stamps that
  chunk's own labels — with **no** corpus merge and **no** global cap.

## The flaw
The corpus cap drops rare-but-locally-critical labels (silent recall loss for the chunks that needed
them), and the broadcast schema means every chunk is extracted against ~40 mostly-irrelevant types
(zero-shot degradation — see T2). Both are artifacts of forcing one unified schema at induction time.

## Proposed change
1. Make **per-chunk induction the enterprise-SOTA extraction path.** Change the wizard's induce block
   (`setup/wizard.py`) to emit `granularity: chunk` for the SOTA profile (keep `corpus`/`doc`
   selectable).
2. On the per-chunk path, the `MAX_LABELS` cap is **not** the right bound — a chunk's own induced set
   is naturally small (only what is in it). Keep sanitization (control-char strip, length ≤64, dedup,
   drop-empties) but make the **count cap configurable and default to *off* (or a high safety bound
   like 64) for chunk granularity**, so a chunk that legitimately needs 15 types keeps all 15. Guard
   against a pathological reply with a generous safety ceiling, not a salience cap.
3. Type consistency is no longer the induction stage's job — it moves to T3. Update the provider
   docstring accordingly (today it claims corpus-mode "keeps entity types consistent across the KG";
   after T3 that guarantee comes from disambiguation).
4. **Add a first-class "predefined-only" mode.** When the operator supplies a fixed label set and
   opts into predefined-only, the `induce` stage is **skipped entirely** (no LLM call) and extraction
   runs each chunk against the fixed config labels. This is the escape hatch for corpora with a known
   schema (compliance taxonomies, fixed entity sets) — induction is per-chunk *by default*, off by
   choice. The wizard exposes it.
5. **Enforce the phase-boundary invariant:** with corpus-merge removed, `induce` is now chunk-level,
   so no corpus-level operation runs before `redact`. Add a pipeline-order assertion (a
   `latence stack validate` check or a runner invariant) that no corpus-aggregating stage is scheduled
   before `redact` in the SOTA profile — the front half is chunk/doc-level by construction.

## Acceptance criteria
- [ ] `latence setup --profile enterprise-sota` emits `induce.config.granularity: chunk`.
- [ ] A chunk's `induced_labels` contains only the types induced for *that chunk*, not a corpus union.
- [ ] Regression test: a corpus where a type appears in exactly one chunk — under `chunk` granularity
      that chunk retains the type; under the old `corpus`+cap path it is dropped (observe the
      difference; this is the recall-loss the redesign fixes).
- [ ] The per-chunk count safety-ceiling is configurable; default does not silently drop a chunk's
      own legitimate labels.
- [ ] **Predefined-only mode:** with a fixed label set + the flag, no `induce` LLM call is made and
      extraction uses the fixed labels per chunk (test asserts zero induction calls).
- [ ] **Phase boundary:** `latence stack validate` (or a runner invariant) rejects any corpus-level
      stage scheduled before `redact` in the SOTA profile; the front half is chunk/doc-level.
- [ ] `ruff` + `mypy --strict` clean; `verify-local` green.

## Affected files
`packages/latence-schema-inducer/src/latence_schema_inducer/provider.py`,
`.../sanitize.py`, `packages/latence-core/src/latence_core/setup/wizard.py`, the inducer tests.

## Note on cost
Per-chunk induction is one LLM call per chunk — see **T5** for the scale/cost mitigations (chunk-text
dedup, a pluggable cheaper/local inducer). T1 assumes T5's cost path exists for large corpora.
