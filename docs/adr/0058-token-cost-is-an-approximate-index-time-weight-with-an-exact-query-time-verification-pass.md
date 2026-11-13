# `token_cost` is an approximate index-time weight carrying its tokenizer's identity; the budget guarantee comes from an exact query-time verification pass

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

Hyperedge `token_cost` is precomputed at index time with an **operator-named, optional** tokenizer
(dependency-free heuristic by default) and stored **alongside the identity of the tokenizer that
produced it**. It is treated as an *approximate ranking weight*, not as truth. The guarantee that the
emitted context actually fits the generation model's window comes from an **exact re-count of the
selected set only** at query time. No tokenizer is pinned anywhere, and switching generation models
never requires a re-index.

## Context

Spec 1.7 requires `token_cost` precomputed "with the same tokenizer as the target generation model",
and forbids computing it online. Both halves are in tension with this repo:

- **`latence process` cannot know the generation model.** Index time runs long before anyone picks
  an LLM, and the adopter may switch models later — silently invalidating every stored weight.
- **Nothing may be pinned.** ADR-0012 and the `pack.py` licence discipline deliberately pin no
  tokenizer encoding: `HeuristicTokenCounter` is the day-one default and `TiktokenCounter` requires
  the operator to *name* an encoding before any tokenizer runs.

The "never compute online" half is nonetheless real for *selection*: encoding 2000 candidate
Hyperedges (~160k tokens) costs on the order of 160 ms against the ~2.3 ms online path ADR-0056
established. But it is **not** real for the *selected* set — 15–40 items is ~2–3 ms.

The existing `Packer` already models the right shape: `token_key` reads a precomputed cost from
metadata and **falls back** to a live `TokenCounter` when absent. This ADR generalises that
precedent rather than inventing a mechanism.

## Decision

- **Index time: an optional, operator-named tokenizer.** Same spec shape as `Packer`'s
  `tokenizer: {"encoding": …}`. Unnamed → the dependency-free heuristic. Nothing downloads unless
  the operator asks for it, so ADR-0012 holds unchanged.
- **The tokenizer's identity is stored next to the value.** A `token_cost` without knowing what
  produced it is uninterpretable. Recording it is what makes a mismatch *detectable* rather than
  silent, and it costs one column.
- **`token_cost` is an approximate ranking weight.** Selection orders and packs on it; a 10–30%
  per-item error changes packing *quality*, not *validity*.
- **The budget guarantee is an exact query-time verification pass over the selected set.** After
  selection, the 15–40 chosen Hyperedges are re-counted with the **generation** tokenizer and the
  tail is trimmed if the true total overflows. The emitted context provably fits.
- **A mismatch is detected and repaired, never trusted and never silently corrected.** If the
  recorded index tokenizer differs from the configured query tokenizer, selection still runs (the
  weights remain valid ranking signals) but the verification pass becomes **mandatory** rather than
  optional. No cross-family scale factor is ever applied — such a factor is a fiction.

## Considered alternatives

- **One `token_cost` column per tokenizer.** Rejected: combinatorial storage, mostly unused, and
  still wrong for the first model the operator did not enumerate.
- **Pin a tokenizer at index time and treat its counts as exact.** Rejected: violates ADR-0012, and
  it bakes a generation-model assumption into an artifact produced before that model is chosen.
- **Count exactly at query time for all candidates.** Rejected on cost: ~160 ms against a ~2.3 ms
  online path (ADR-0056).
- **Scale stored counts by a per-tokenizer-family ratio on mismatch.** Rejected: tokenizer families
  differ per-text, not by a constant; a scale factor would convert a detectable error into a
  plausible-looking wrong number.

## Consequences

- Switching generation models requires **no re-index** — only the verification tokenizer changes.
- The budget guarantee is stronger than the spec's: the spec trusts index-time counts, this trusts
  nothing and verifies what is actually emitted.
- The verification pass can only ever *shrink* a selection, so it cannot break the selector's
  budget invariant; it can leave the context slightly under-packed when the index tokenizer
  under-counted. Accepted — under-packing is safe, overflow is not.
- **Unmeasured:** how much packing quality degrades when selection runs on heuristic rather than
  exact weights. Folded into the M6 eval as an A/B rather than guessed here.
