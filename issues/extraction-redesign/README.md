# Extraction redesign — per-chunk induction + downstream type consolidation

**Filed by the maintainer's architectural review of the induce → extract → disambiguate path.**

## The flaw, precisely

The enterprise-SOTA pipeline induces labels at **corpus granularity** (`induce` /
`label_inducer.llm`, `granularity: corpus`): it induces per-document, merges every document's label
sets, ranks them by document frequency, and **caps each type-slot at `MAX_LABELS = 40`**
(`packages/latence-schema-inducer/src/latence_schema_inducer/sanitize.py:41`). That single frozen
schema is then stamped on **every** chunk, and both `extract` (`fused_entity_relation.gliner2`) and
`redact` (`redaction.gliner2`) apply the config floor **unioned with that full ~40-type schema to
every chunk** (`effective_labels`, `packages/latence-core/src/latence_core/induced.py`).

This has two failure modes, both real:

1. **Capping silently drops labels a chunk needs.** A type that is rare corpus-wide but *critical*
   for the handful of chunks that contain it falls below the salience cap and is dropped — those
   chunks are extracted without it. Recall loss, invisible.
2. **Irrelevant labels degrade every chunk.** Zero-shot NER/RE precision degrades as the label count
   grows; a chunk about topic X is scored against ~40 types, ~35 of them irrelevant to it. Label
   competition lowers precision on exactly the extraction the KG is built from.

The reason corpus-mode exists at all is **type consistency**: if you induced per-chunk you'd get
`org` in one chunk and `organization` in the next, and the KG's types would drift. But **that
consistency belongs in the resolution phase, not the induction phase.** Today
`disambiguation.glinker` resolves entity *mentions* to canonical entities — it does **not**
canonicalize the type *vocabulary* (verified: no stage does). So corpus-mode imposes consistency
upstream at the cost of recall, precisely because the downstream phase that *should* earn it does
only half its job.

## The target architecture

Invert it:

```
  induce  (per CHUNK)  → each chunk gets only its own, uncapped, relevant labels
  extract (per CHUNK)  → gliner2 runs each chunk against its small, relevant label set  → best recall + precision
  redact  (per CHUNK)  → induced PII types per chunk;  ADR-0044 financial FLOOR stays universal (safety)
  disambiguate         → resolve ENTITIES  AND  canonicalize the TYPE vocabulary corpus-wide  ← consistency earned HERE
  graph                → consistent canonical types, no drift
```

Per-chunk induction+extraction gives complete, relevant extraction (no capping loss, no irrelevant
labels). The type drift it introduces is resolved *downstream* by a new **type-canonicalization**
step in the cleaning phase — the same place entity resolution already happens. Consistency is earned
after extraction, not paid for before it.

## Invariants this redesign MUST guarantee (the acceptance contract)

1. **The entire front half runs at chunk/document level — no corpus-level operation until redaction
   is finished.** After T1 removes the corpus-merge from `induce`, every stage through `redact` is
   chunk-level (induce, extract, redact) or per-document-deterministic (header_refs, inline_refs,
   the keyword content_screen). **`redact` is the last chunk-level stage.** The FIRST corpus-level
   operation is the corpus-building / resolution phase (profiling → disambiguate → graph). No
   corpus-level entity or type resolution happens before redaction.
2. **For every chunk, NER + relation labels are induced per-chunk — unless "predefined-only" mode.**
   In predefined-only mode the operator supplies a fixed label set, the `induce` stage is skipped,
   and extraction runs each chunk against the fixed labels. (T1/T2.)
3. **The same holds for PII redaction and guardrailing.** Induced PII/guard categories are per-chunk;
   the **universal safety floors** (the ADR-0044 financial-PII floor; the always-on guardrail
   categories) are NOT subsetted — they apply to every chunk because they are security invariants,
   not relevance labels. The deterministic keyword `content_screen` is a separate pre-filter, exempt
   by nature (no learned label count). (T2.)
4. **Only downstream — in the corpus-building step — is everything resolved corpus-wide, deduped,
   and cleaned**: entities canonicalized (glinker) AND the type vocabulary canonicalized (T3), so the
   KG has no corpus-level ambiguity. (T3.)

## Why this also solves the CUDA-OOM at its root (not by a band-aid)

The gliner2 OOM we hit earlier was `batch_size × ~768 tokens × ~40 induced labels` collapsing into a
single multi-GiB span-scoring tensor (measured: batch 32 → ~17 GiB → OOM). That giant label
dimension only exists **because corpus-mode applies ~40 labels to every chunk.** With per-chunk
induction each chunk carries *few* labels, so the tensor is small and the OOM never forms — the
reactive micro-batch backoff becomes a rare safety net, not the primary mechanism. **This redesign is
the real fix for the OOM; the earlier micro-batching was a symptom patch.** (T7 makes this explicit
and adds the efficient batching that keeps per-chunk fast.)

## Tickets

| # | Point in pipeline | What |
|---|---|---|
| [T1](T1-per-chunk-induction.md) | `induce` | Make per-chunk induction the extraction path; drop the corpus salience-cap on this path; wire the SOTA wizard. |
| [T2](T2-per-chunk-extraction.md) | `extract` + `redact` | Extraction runs each chunk against its own induced set (minimal floor); PII financial floor stays universal. |
| [T3](T3-type-canonicalization.md) | `disambiguate` | **The key one.** Add corpus-wide TYPE/label canonicalization to the cleaning phase — cluster drifted types to a canonical vocabulary + alias map, remap mentions. |
| [T4](T4-delta-reconciliation.md) | `delta` / frozen schema | Reconcile ADR-0043 incremental runs with per-chunk induction + a *persisted, mergeable* canonical type vocabulary. |
| [T5](T5-induction-cost.md) | `induce` / perf | Bound per-chunk induction cost at scale: chunk-text dedup, a pluggable cheap/local inducer, honest cost reporting. |
| [T6](T6-docs-and-decision.md) | docs / ADR | Record the decision (amend ADR-0038); correct the induction framing in `docs/DEEP-DIVE.md` (it currently sells corpus-mode as the feature). |
| [T7](T7-efficient-batching.md) | `extract` / `redact` perf | Efficient per-chunk extraction: group chunks by label-set and batch through gliner2's native batched forward, unlock gliner2 `compile`/`quantize` (now viable with stable grouped shapes). Makes per-chunk fast AND is the root fix for the OOM. |

**Suggested order:** T3 first as a short design note (it's the substantive new capability that makes
the whole inversion viable), then T1 → T2 → T5 in parallel, T4 after T1+T3, T6 last. T3 is the
lynchpin: without downstream type consolidation, per-chunk induction regresses KG consistency, so it
must land with (or before) T1 flips the default.
