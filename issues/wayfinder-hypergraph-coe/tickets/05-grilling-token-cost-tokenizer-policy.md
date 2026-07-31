# 05 — Grilling: token_cost tokenizer policy

Type: grilling
Status: resolved
Blocked by: 01

## Question

Spec 1.7 requires token_cost precomputed "with the same tokenizer as the target generation model"; the repo deliberately pins no tokenizer (pack.py licence discipline, ADR-0012). Decide: operator-named tokenizer at `latence process` time? Multiple token_cost columns per tokenizer? Heuristic default + exact opt-in (mirroring HeuristicTokenCounter/TiktokenCounter)? What happens on index/query tokenizer mismatch — recompute, scale, or refuse? Outcome: recorded in the ADR from 03 or its own.

## Answer

**Recorded as [ADR-0058](../../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md).**

The spec's "precompute with the generation model's tokenizer, never compute online" cannot hold as
written: `latence process` runs before any LLM is chosen, the adopter may switch models later, and
ADR-0012 pins no encoding. The resolution splits the concern by what each number is *for*:

- **Index time — optional, operator-named tokenizer** (same shape as `Packer`'s
  `tokenizer: {"encoding": …}`), dependency-free heuristic by default. Nothing pinned, nothing
  downloaded unless asked.
- **The tokenizer's identity is stored beside the value** — a `token_cost` is uninterpretable
  without it, and this is what makes mismatch detectable instead of silent. One column.
- **`token_cost` is an approximate ranking weight.** A 10–30% per-item error changes packing
  quality, not validity.
- **The budget guarantee is an exact re-count of the selected set only** (15–40 items, ~2–3 ms)
  with the *generation* tokenizer, trimming the tail on overflow. Counting all ~2000 candidates
  would cost ~160 ms against ADR-0056's ~2.3 ms online path — so the spec's "never online" holds for
  candidates and is wrong for the selected set.
- **Mismatch → verification becomes mandatory, never a scale factor.** Cross-family ratios are a
  fiction; they turn a detectable error into a plausible wrong number.

Consequence worth stating: **switching generation models requires no re-index.** This is a stronger
guarantee than the spec's, which trusts index-time counts outright.

Rejected: one column per tokenizer (combinatorial, still wrong for the un-enumerated model); pinning
a tokenizer at index time (violates ADR-0012, bakes in a model chosen later).

Residual folded into [ticket 17](17-task-m6-eval-gate.md): how much packing quality degrades on
heuristic vs exact selection weights is **unmeasured** — it ships as an eval A/B, not a guess.
