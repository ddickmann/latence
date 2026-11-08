# Prototype: resolution feasibility spike (go/no-go)

Type: prototype
Mode: HITL
Status: open
Blocked by:

## Question

Does the candidate design in [design/resolution-at-scale.md](../design/resolution-at-scale.md) resolve entities well
enough, at a reviewer effort that is acceptable, to be worth building? Nothing else is built
until this reports go or no-go.

Build a throwaway prototype on scratch code, outside the shipped packages. It has three parts:
anchor-first linking with NIL clustering, calibrated three-band scoring, and constrained clustering.
Run it on English data only (owner, 2026-11-08). The benchmarks used must have permissive
licences, which rules out REBEL and WebNLG. Run it on:

- **Structured:**
  - one public ER benchmark with a permissive licence (Leipzig/Magellan-style pairs or WDC
    Products; verify the licence before use);
  - the Metis synthetic estate's master data with injected duplicates (typos, abbreviations,
    abbreviations and acronyms, merged and split companies);
  - Splink on DuckDB. Check the Spark path on the sandbox once *get a Databricks sandbox* lands.
- **Text:**
  - one permissively licensed entity-linking benchmark;
  - a small hand-labelled English slice from an enterprise-like corpus (a few hundred mention pairs);
  - mentions are linked to anchors with a cross-encoder, and the leftovers are NIL-clustered.

**Report:**

- quality: B-cubed and pairwise P/R/F1;
- auto-accept precision under Learn-then-Test at the declared target;
- **review-band rate and estimated expert minutes per 1,000 records**, which measures the split of
  work between machine and expert;
- throughput and cost at 10⁵ and 10⁶ records;
- a side-by-side against the current cascade as a baseline.

**Go/no-go bars** (set by the owner, 2026-11-08):

- **auto-accept precision ≥ 0.95**, set by Learn-then-Test with its confidence stated;
- **no expert-capacity bar during development.** The owner judged it too vague to set now, and
  an obstacle the dev phase doesn't need. The review-band rate and the estimated expert minutes
  are still *reported*, but they gate nothing yet;
- 10⁶ records within an agreed time and cost budget.

The spike also produces the realistic review-band items that *the expert resolution workbench*
prototypes against.
