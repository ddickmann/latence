# The SPLADE sparse signal is a structured composition — model term-weights ⊕ confidence-gated NER entities ⊕ metadata — redacted by default

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

The sparse-retrieval signal is not plain `SPLADE(text)`. It is an **intelligent composition** that fuses
the learned multilingual SPLADE term-weights with the **structured signals the pipeline already
produces** — NER-extracted entities and selected metadata — into one enriched sparse vector. This
exploits an edge a vanilla SPLADE shop does not have (a latence corpus ships entities + metadata), and it
runs training-free and stateless.

## Context

A latence corpus carries, per chunk, the (optionally redacted) text, extracted `EntityMention`s with
confidence, KG canonical entities, and metadata (`classification`, `provenance`). Plain SPLADE ignores
all of it. Composing them into the sparse representation grounds the signal in the pipeline's structured
output — but couples it to NER quality, which must be managed.

## Decision

- **Input text — redacted by default.** SPLADE runs on the chunk's *corpus* text, following the corpus's
  own redaction policy (if the exported chunk is masked per the redaction floor, ADR-0044, the sparse
  signal is generated over the masked text — so it never re-leaks PII the corpus removed). Unredacted
  input is an explicit opt-in for fully-internal deployments.
- **Composition = term-injection (default), weighted (optional).** Default:
  `sparse = SPLADE(text) ⊕ boost·entity_terms ⊕ boost·metadata_terms` — union the model's weights with
  the pipeline's **entity surface forms** and selected **metadata tokens**, each with a configurable
  boost. Optional alternative: keep SPLADE and an entity/metadata sparse vector separate and combine with
  tunable weights at query time. Both are training-free; boosts live in `latence.retrieval` config.
- **Injected entity terms are gated by extraction confidence** — the `EntityMention` confidence the
  pipeline already carries — so weak NER does not pollute the sparse signal.
- **Model — license-verified, config-swappable.** The multilingual sparse model's **weights and code
  licenses are verified separately** (research-diligence directive / ADR-0012) and a permissive one is
  chosen before it is wired; it is swappable via the `latence.retrieval` seam. No model is hardcoded
  until the license is cleared.
- **Emission** follows ADR-0050: parallel `sparse_indices` / `sparse_values` columns (index-time), and
  the same composition is applied to the query at query-time.

## Consequences

- The sparse leg is richer and more grounded than plain SPLADE, and it is interpretable (you can read the
  injected terms) — a genuine differentiator.
- It couples the sparse signal to NER quality; the confidence gate is the mitigation, and the boosts are
  tunable so the structured contribution can be dialed down per corpus.
