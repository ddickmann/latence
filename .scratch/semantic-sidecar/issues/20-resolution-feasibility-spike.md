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

## Model set (owner, 2026-11-08)

Agreed before any model arm runs. Full table, reasons and run order are in
[`../spike/NEXT.md`](../spike/NEXT.md).

- Extraction: `knowledgator/gliner-relex-large-v1.0`.
- Decisions: `fastino/GLiNER2.5-Decide` vs `fastino/GLiDE-*`, head to head. Fastino is a
  partner, so its licence does not exclude GLiDE.
- Retrieval: `google/embeddinggemma-2` together with char-ngram TF-IDF.
- Reranking: `Qwen/Qwen3-Reranker-0.6B`, zero-shot vs LoRA fine-tuned.
- Structured matching: Splink stays the backbone. Pairs it leaves uncertain are re-scored by the
  reranker and by Decide/GLiDE, whose scores become features before calibration.
- Generative model: hosted, behind a provider-neutral adapter. A deep tier writes definitions and
  induction proposals; a fast tier writes review-item explanations. The spike sends only
  synthetic or public data.
- Runtime: a Runpod pod, 1× L40S 48 GB with at least 64 GB system RAM (owner, 2026-11-08;
  replaces the M3).

**Related bar (owner, 2026-11-10, *Evaluation and the acceptance bars*).** For v1 the
precision bar gets a guard: on the synthetic estate the machine alone must find at least 60% of
the true matches while holding 0.95 precision. The spike's go/no-go bars above are unchanged;
its report states the machine-only recall next to that guard.

## Progress (2026-11-10): the structured side at 10^6 records

Status stays `open`: the go/no-go needs the model arms, which need the GPU machine.

- **Throughput: met on a small machine.** 989,658 records run end to end in 72 s at 5.4 GB peak
  memory on 4 CPUs with no GPU. The earlier attempt died at 12 GB. Pairs and clusters are now
  held as integer arrays; the method is unchanged, and three datasets give the same pairs,
  bands and clusterings before and after the change.
- **Precision at the label budget used so far: not certified.** With 5,000 uniformly drawn
  labelled pairs, Learn-then-Test abstains at 10^6 and the machine alone merges nothing. Only
  2.8% of candidate pairs are true matches at this size (16% at 10^5), so the labelled sample
  holds too few matches to certify any accept threshold.
- **Precision with ten times the labels: met.** With 50,000 labelled pairs the realised
  auto-accept precision is 0.980 and the machine alone finds 75.8% of the true pairs at
  pairwise precision 0.986, above the 60% guard.
- **Two design consequences for *The upgrade of the resolution engine*:** labels have to be
  drawn where the scores are informative, not uniformly over candidate pairs, with the test
  adjusted for that sampling; and the reject tolerance has to be stated against true pairs
  (1% of 28.8M rejected pairs is 17% of all true pairs).
- **Not measured:** any model arm, the text side beyond its lexical baseline, and a real
  (non-synthetic) estate at this size.

Details and result files: [`../spike/NEXT.md`](../spike/NEXT.md).
