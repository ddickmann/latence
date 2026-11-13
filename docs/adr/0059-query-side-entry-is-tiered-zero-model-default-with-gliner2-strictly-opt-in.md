# Query-side entry is tiered: a zero-model default over the alias KV and the caller's own query embedding, with GLiNER2 strictly opt-in

!!! warning "Superseded by [ADR-0068](0068-query-time-retrieval-tooling-is-removed-embedders-stay-only-as-a-resolution-signal.md) (2026-11-08)"
    **This decision is no longer in force.** Query-time retrieval tooling, hyperedges and
    Context Enrichment leave the product (ADR-0068, following the re-scope in ADR-0067). This
    page stays as the dated record of what was decided and why.

Entity entry for chain-of-evidence retrieval (spec 2.1) ships as **three independently useful
tiers**. The default tier loads **no model beyond what the adopter already has** — ANN over
`entity_vecs` with the query embedding the caller computed for their own first-stage search, plus
alias point-lookups against the sidecar KV. GLiNER2 query NER (and the `w_relq` relation-type
signal) is a **strictly opt-in tier** behind an extra. The multi-hop claim survives the default
tier; what degrades is anchor precision, not capability.

## Context

Spec 2.1 assumes a model-bearing query path: GLiNER2 NER on the query string, glinker linking, and
(spec 3.2) GLiNER2 zero-shot relation-type extraction for the `w_relq` bonus. Today `rerank` is the
only query-time component that loads a model at all.

Two repo constraints bound the design:

- **`latence-retrieval` depends on `pydantic` alone** — deliberately not on `latence-core` or any
  pipeline package, so "an adopter who wants only `rerank`/`fuse`/`pack` installs this without
  pulling the whole pipeline" (ADR-0016/0049 discipline; heavy deps live behind extras).
  A query-side model must therefore arrive as an **extra wrapping the upstream library directly**,
  never as a dependency on `latence-extract-gliner2`.
- **ADR-0012:** no default checkpoint. `Reranker` sets the precedent — `model: str` is required with
  no default, resolved lazily, and "constructing the `Reranker` never downloads weights."

Making query NER mandatory would force a model stack on every adopter and gate the entire feature
behind inference.

## Decision

Three tiers, each shippable and independently useful:

- **Tier 0 — no new models.** `V0` from ANN over `entity_vecs` using **the query embedding the
  caller already computed** for their own first-stage search. No extra, no inference beyond what the
  adopter's own retrieval already pays for. Propagation, chain assembly, and budgeted selection all
  run. This is the default.
- **Tier 1 — alias lookup, still zero models.** Query n-grams resolved against the alias table in
  the per-call sidecar KV (ADR-0056), yielding a real `V_llm` anchor set from **point lookups, no
  inference**. Expected to carry most of the linking value on enterprise corpora, where entities are
  named exactly (product codes, legal entity names, part numbers).
- **Tier 2 — GLiNER2, opt-in behind an extra.** Query NER **and** zero-shot relation-type detection
  for `w_relq` from **one model load** (the same fused checkpoint family the pipeline uses —
  ADR-0013/ticket 02: Apache-2.0 weights and code, no GLiREL anywhere). Named checkpoint, no
  default, lazily loaded, registered on the `latence.retrieval` seam — not `latence.providers`, and
  not on the Stage DAG (ADR-0049).

**`entity_kb` reaches the query path as the alias table inside the per-call KV.** No resident
knowledge base, no service, no state held between calls (ADR-0056).

**Degradation is a gradient, not a cliff.** Multi-hop reach comes from propagation over the
hypergraph, not from how entry entities were found, so tier 0 still surfaces evidence pure ANN never
would. What degrades: the restart vector's mass spreads over ANN-recalled entities instead of
concentrating on entities the query provably names (spec 2.1 anchors `r` on the linked branch), and
`w_relq` is absent with its weight redistributed across the remaining `v0` terms.

## Considered alternatives

- **Mandatory query-side NER + linking (the spec's literal path).** Rejected: forces a model stack
  on every adopter, breaks `latence-retrieval`'s pydantic-only promise, and gates the feature behind
  inference that tier 1 largely obviates.
- **Depending on `latence-extract-gliner2` for the query path.** Rejected: drags `latence-core` and
  the whole pipeline into a downstream library that exists precisely to avoid that.
- **A default checkpoint for convenience.** Rejected by ADR-0012 — a restricted or unvetted model
  must never become a hidden default.

## Consequences

- **Latency, stated plainly:** ADR-0056's ~2.3 ms is the *structural* path. Tier 1 adds ~nothing;
  tier 2 adds GLiNER2 inference — **tens of ms on CPU, an order of magnitude more than everything
  else combined**. The guiding budget is to stay under the adopter's own first-stage ANN latency so
  the tooling is never the bottleneck. Tier 2's cost is the reason it is opt-in.
- The tiers are a genuine quality gradient, so the eval (ticket 17) should report the curve **per
  tier** — otherwise the headline number silently assumes tier 2 and misrepresents the default
  install.
- `w_relq`'s weight must redistribute cleanly when tier 2 is off; the utility function cannot assume
  the signal exists.
- Tier 1's value is an untested hypothesis: alias-only linking may or may not approach tier 2's
  anchor precision on real corpora. The eval decides; nothing downstream depends on the answer.
