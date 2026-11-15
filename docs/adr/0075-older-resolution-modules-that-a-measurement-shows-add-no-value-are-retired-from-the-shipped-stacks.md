# Older resolution modules that a measurement shows add no value are retired from the shipped stacks

**Status:** Accepted (2026-11-15)

The owner set the rule on 2026-11-14 (map ticket 13, decision L13): an older resolution module of
the text path is kept only where a measurement shows that it adds value, and retired otherwise.
Map ticket 50 measured each one; tickets 60 and 61 retire the two that failed.

## Context

The measurement is `benchmark/resolution-modules/` (`measure.py`, its gold sets and
`results.json`). The rule was fixed in the harness before the run: a module is kept when it
raises pairwise F1 by at least 0.02 over the same run without it and its precision does not
fall. On 684 mentions of 179 organisations in synthetic English documents, against the same run
on exact normalised surfaces (P 0.946, R 0.265, F1 0.414):

| Module | P / R / F1 | F1 gain | Verdict |
| --- | --- | --- | --- |
| `disambiguation.cascade` (the string cascade's fuzzy rungs) | 0.754 / 0.304 / 0.433 | +0.019 | retire |
| `disambiguation.embedding` on the in-core hashing embedder | 0.616 / 0.265 / 0.371 | −0.043 | retire |
| Type Consolidation (lexical rungs), on 65 type labels | 1.000 / 0.879 / 0.936 | +0.546 | keep |

The embedding Disambiguator with a learned embedder and the GLinker linker need a GPU and were
not measured.

## Decision

1. **No shipped stack and not the demo names a retired use.** `stacks/default.yaml`,
   `stacks/schema-induction.yaml`, `stacks/linkpred-base.yaml` and the demo's Pipeline name
   `disambiguation.exact_surface` (exact normalised surfaces, configured aliases and the same KB
   linking) where they named `disambiguation.cascade`. No shipped stack ran the embedding
   Disambiguator on its hashing default; the GPU stacks configure a learned embedder.
2. **`latence stack check` says so.** A stack that names `disambiguation.cascade`, or
   `disambiguation.embedding` with no `embedder`, gets a `retired` WARNING that names the
   measurement and what to use instead (`latence_core.stacks.retired`). The Pipeline still runs.
3. **The Providers stay registered.** A Pipeline that names one keeps working, the bake-off
   matrix keeps the cascade as a candidate (a measurement needs it), and the examples that show
   the cascade itself keep it.
4. **The learned embedder decides the GPU stacks.** The embedding Disambiguator on a learned
   embedder, and the GLinker linker, are measured on the same gold set on a GPU host before the
   GPU stacks are judged by the same rule (a live request, map ticket 61).

## Consequences

- The demo graph holds 7 nodes and 34 edges where it held 6 and 33: "IBM" and "International
  Business Machines" are two nodes, both linked to the same KB entity, because no fuzzy rung
  joins them. An `aliases` entry joins them, and that is the documented way to do it.
- `Severity.WARNING` now covers two cases: a Provider not installed on the host, and a retired
  use.

## Alternatives considered

- **Remove the cascade Provider.** Rejected for now: the measurement is on template prose, which
  is strong enough to stop shipping a default but not to break every Pipeline that names it;
  dozens of tests and examples exercise the cascade as the resolver's reference behaviour, and
  the bake-off needs it as a candidate. Removal can follow on a measurement over real text.
- **Refuse a retired Provider at load.** Rejected for the same reason: a refusal breaks running
  Pipelines over a measurement the operator never saw; a warning at `stack check` reaches them
  first, with the reason.
- **Make the embedding Disambiguator refuse to run without an embedder.** Rejected: its offline
  tests and the conformance suite run it on the hashing embedder, which is what lets the CPU
  lane exercise it; the retirement is of the hashing default in a stack, which the check catches.
- **Keep the cascade as the default and add aliases to the stacks.** Rejected: the rule is the
  owner's, and the measured cascade lowers precision by 0.19; aliases make the exact-surface
  default merge what an operator names, without the fuzzy rungs.
