# Prototype: resolution feasibility spike (go/no-go)

Type: prototype
Mode: HITL
Status: resolved
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

## Progress (2026-11-11): the model arms have run; report and recommendation

The report is
[`../spike/REPORT.md`](../spike/REPORT.md); it recommends **go**.

- **Machine:** a rented pod with one 24 GB GPU (owner, 2026-11-11; replaces the L40S plan).
  GLiDE starts with its smallest variant and goes up only if needed (owner, same day); only
  0.4B has run.
- **Precision bar:** met in every arm. Structured, 10⁵ records: 0.986 to 0.990. Structured,
  10⁶: 0.9955 with 7,500 labels, the second 2,500 drawn from the undecided pairs. Text: 0.959
  to 0.965.
- **Recall guard (≥ 60% machine only):** 0.757 with the matcher alone, 0.894 with GLiDE 0.4B
  (10⁵); 0.710 at 10⁶.
- **Review work at 10⁵:** 23,607 pairs with the matcher alone, 10,390 with GLiDE 0.4B, 8,962
  with the reranker as well (36.7, 17.3 and 14.4 expert minutes per 1,000 records).
- **Public person dataset (50k):** precision 0.971 with the matcher alone; neither model moves
  a pair out of its review band (21.7%). The models' gain is shown on synthetic data only.
- **Decide against GLiDE:** GLiDE 0.4B wins every comparison; zero-shot Decide carries almost
  no weight.
- **Acronyms** (owner's priority, 2026-11-11): a resolver over thread, definition, alias-column
  and letter evidence links 82% of acronym mentions correctly on a synthetic set written for
  it (0.1% before), 97% when the full name is in the same thread, 42% with context only.
- **Two defects found and fixed** in the spike code, each with a test that failed first.
- **Still open for the decision:** a time and cost budget for 10⁶ records; no real estate or
  real text; no entity-linking benchmark (licence checks pending); no Spark run.

## Answer

**Go** (owner, 2026-11-11). The design in
[design/resolution-at-scale.md](../design/resolution-at-scale.md) is built.

| Bar | Measured | Verdict |
|---|---|---|
| Auto-accept precision ≥ 0.95 under Learn-then-Test | 0.986 to 0.990 at 10⁵ records, 0.9955 at 10⁶, 0.959 to 0.965 on text; every Wilson lower bound above 0.95 | met |
| 10⁶ records within the budget | **Budget: 4 hours on one 24 GB GPU** (owner, 2026-11-11). Measured: 92 s for the matcher alone, 56 min with one neural pass over 609,117 pairs | met |
| Guard: the machine alone finds ≥ 60% of true matches | 0.757 to 0.900 at 10⁵, 0.710 at 10⁶ | met |
| Expert capacity | reported, not gated: 36.7 expert minutes per 1,000 records with the matcher alone, 14.4 with both neural scorers | reported |

**What the go rests on, and what it does not.**
- Every estate and document is synthetic, except one public person dataset. On that dataset
  the precision bar is met with the matcher alone, and neither neural model moves a pair out
  of review. The size of the models' gain is shown on synthetic data only.
- No Spark run exists; the Databricks sandbox is still open.
- The ticket's "one permissively licensed entity-linking benchmark" and "hand-labelled slice"
  are answered by the real-data work that follows this decision (public registers, public
  filings, CC0 targets), recorded in [`../spike/REPORT.md`](../spike/REPORT.md) when it has
  run. The licence review is in
  [`../research/23-entity-linking-benchmark-licences.md`](../research/23-entity-linking-benchmark-licences.md).

**Model set after measurement.** Splink stays the backbone. GLiDE 0.4B is the decision model
(the smallest variant; larger ones only if the review band has to shrink further). The
reranker is second on records and useful on text. Zero-shot GLiNER2.5-Decide is dropped. The
embedder has not earned its place. GLiNER-Relex extracts mentions.

**What follows.** *The upgrade of the resolution engine* loses its last blocker. The pass that
aims the neural scorer below the reject line, a second stage that uses labels without a
model, the reviewer's alias loop and the label strategy at 10⁶ are its first inputs.

## After the decision (2026-11-11): what real registers showed

Run the same day, after the go; details in [`../spike/REPORT.md`](../spike/REPORT.md). The
decision stands as the owner made it; this is what it did not yet know.

- **Two real company registers** (120,192 legal-entity records against the UK company
  register, the company number withheld): the matcher alone certifies nothing. With GLiDE
  0.4B auto-accept precision is 0.995 and the machine alone finds 36.5% of the true links;
  about 594,000 pairs stay in review. The 60% guard is defined on the synthetic estate and is
  not met here.
- **The first-stage matcher is the weak part on real names and addresses,** not the model.
  The fixes belong to *The upgrade of the resolution engine*.
- **The pass below the reject line** was built and certified nothing with the zero-shot
  model, at 10⁵ and at 10⁶ (68 minutes, inside the 4-hour budget).
- **A labels-only stage** reproduces the 10⁶ result in 83 s without a model.

## The register result, put right (2026-11-11)

The owner did not accept 36.5%. The cause was the first stage: it learns without labels what a
field is worth, and on registers where thousands of companies share an address it learned
that the address makes the match. A scorer fitted on the reviewer's labels over the evidence
of each pair replaces the first stage's calibration, with the same 5,000 labels and no GPU:

| Estate | Machine alone, before (best arm) | Machine alone, now | Left for review, now |
|---|---|---|---|
| Real registers, 302,924 records | P 0.991, R 0.365 | **P 0.998, R 0.972** | 6 pairs |
| Real registers, 1,222,924 records | nothing certified | **P 0.997, R 0.971** | none |
| Synthetic 10⁵ | P 0.991, R 0.900 (two models) | **P 0.993, R 0.913** | 2,179 pairs |
| Synthetic 10⁶ | P 0.997, R 0.710 (56 min, GPU) | **P 0.984, R 0.874** (6 min) | 8,196 pairs |
| Public person records | P 0.917, R 0.840 | **P 0.949, R 0.841** | 35,239 pairs |

The 60% guard is now met on the real registers too. Details, what failed on the way and the
limits: [`../spike/REPORT.md`](../spike/REPORT.md#a-scorer-fitted-on-the-reviewers-labels).

## From the map (moved 12 Nov)

[Prototype: resolution feasibility spike](issues/20-resolution-feasibility-spike.md) — **go** (owner, 2026-11-11). Auto-accept precision 0.986 to 0.990 at 10⁵ records, 0.9955 at 10⁶ and 0.96 on text; the machine alone finds 71% to 90% of true matches; 10⁶ records fit the 4-hour budget on one 24 GB GPU. Splink is the backbone, GLiDE 0.4B the decision model, the reranker second; zero-shot Decide is dropped. Labels drawn from the undecided pairs replace uniform labels. Shown on synthetic data and one public person dataset, where the models add nothing; real data follows. Report: [spike/REPORT.md](spike/REPORT.md). **Put right the same day:** on two real company registers the first run found 36.5% of the true links; a scorer fitted on the same number of reviewer labels finds 97% at 99.7% precision and beats every earlier arm on the other estates, on CPUs ([research note](research/24-company-record-linkage-methods.md)).
