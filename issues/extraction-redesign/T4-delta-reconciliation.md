# T4 — Reconcile incremental / frozen-schema runs with per-chunk induction

**Stages:** `induce` + the Delta Runner · **Depends on:** T1, T3 · **Severity:** major (incremental
maintenance is a first-class product promise, ADR-0018/0029/0043)

## Current behavior (verified)
- The frozen-schema path (ADR-0043) freezes the **one corpus schema** and, on a delta run, stamps it
  on new chunks with **no LLM call** (`provider.py:353` `frozen_induced_labels`; `--reinduce` opts
  out). This is what keeps incremental extraction consistent with the base run today.
- It assumes a single corpus schema — which per-chunk induction (T1) no longer produces.

## The flaw
With per-chunk induction there is no single schema to freeze. And the new canonical type vocabulary
(T3) is corpus-wide state that a delta run must **extend**, not recompute — otherwise a delta either
re-induces the whole corpus (cost) or drifts from the base run's canonical types (inconsistency).

## Proposed change
1. **Delta induction is per-chunk on the new/changed chunks only** — induce the affected chunks
   (cheap relative to the corpus), never re-induce unchanged chunks. Reuse the Delta Runner's
   affected-set logic (ADR-0029).
2. **Persist the canonical type vocabulary + alias map** (T3) as run state (alongside the corpus id).
   On a delta run, the type-canonicalization step **merges the new chunks' induced types into the
   persisted vocabulary**: a new type that matches an existing canonical cluster is aliased to it;
   a genuinely new type creates a new canonical entry. Existing canonical types stay stable (their
   ids/labels don't churn), so the KG's type space grows monotonically without rewriting history.
3. Replace the old "freeze one schema" semantics with "persist + extend the canonical vocabulary."
   `--reinduce` becomes "re-canonicalize from scratch" (a full rebuild), the rare path.

## Acceptance criteria
- [ ] A delta run induces only the affected chunks (per-chunk), not the whole corpus.
- [ ] New induced types canonicalize **against the persisted vocabulary**; matching types alias to the
      existing canonical entry; existing canonical type ids/labels are unchanged by the delta.
- [ ] A genuinely new type on a delta creates a new canonical entry and appears on the Quality-Report
      delta churn.
- [ ] Determinism + idempotency: the same delta applied twice yields the same canonical vocabulary.
- [ ] `verify-local` green; the ADR-0043 frozen-schema tests are updated (not weakened) to the new
      persist-and-extend model.

## Affected files
`packages/latence-schema-inducer/.../provider.py` (delta path), the Delta Runner
(`packages/latence-core/src/latence_core/delta_runner.py`), the T3 canonicalization stage (persistence),
`docs/adr/0043-*.md`, the delta tests.
