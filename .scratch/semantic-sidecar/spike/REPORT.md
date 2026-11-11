# Resolution feasibility spike: report

Status on 2026-11-11. This is the report the go/no-go ticket
([`../issues/20-resolution-feasibility-spike.md`](../issues/20-resolution-feasibility-spike.md))
asks for. All data is synthetic or public. The decision is the owner's; the recommendation is
at the end of the first section.

## The bars and where the measurements stand

| Bar | Measured | Met? |
|---|---|---|
| Auto-accept precision ≥ 0.95 under Learn-then-Test (δ = 0.05) | Structured: 0.986 to 0.990 at 10⁵ records in every arm, 0.9955 at 10⁶. Text: 0.959 to 0.965 in every model arm. Wilson lower bounds are all above 0.95. | Yes, on synthetic data and one public dataset |
| Guard: the machine alone finds ≥ 60% of true matches at that precision (synthetic estate) | 0.757 with the statistical matcher alone, 0.894 with the smallest decision model added (10⁵); 0.710 at 10⁶ | Yes |
| 10⁶ records within an agreed time and cost budget | 92 s for the statistical matcher (48 CPU threads, 5.2 GB); 56 min when one small model re-scores 609,117 pairs on one 24 GB GPU. No budget has been agreed. | Open: needs the owner's budget |
| No expert-capacity bar; review effort is reported | 36.7 expert minutes per 1,000 records with the matcher alone, 14.4 with both neural scorers (10⁵, at 20 s per review group) | Reported |

**Recommendation: go.** The design clears the precision bar and the recall guard with room,
and on the synthetic estate the neural stage cuts review work by more than half with the
smallest decision model. **The one caution:** on the only non-synthetic dataset (public person
records) the bar is also met, but the models move nothing out of review. So the precision
claim travels; the size of the models' gain is shown on synthetic data only. What the spike
does not show is listed under [Not done](#not-done-and-limits): no real estate, no real text,
no entity-linking benchmark, no Spark run.

## Setup

- **Machine:** one NVIDIA RTX PRO 4000 Blackwell (24 GB), 48 CPU threads, 125 GB RAM, Python
  3.12, PyTorch 2.8.0 + CUDA 12.8. The decision model GLiDE runs in its own environment with
  the versions its model card pins (PyTorch 2.14.0, transformers 5.17.0).
- **Models,** each at the revision that was downloaded on 2026-11-11:

  | Role | Model | Revision | Licence |
  |---|---|---|---|
  | Extraction | `knowledgator/gliner-relex-large-v1.0`, zero-shot | 4aedc9226a5ac9e2f6b5ea3e91c1ee577c88a290 | Apache-2.0 |
  | Pair reranking | `Qwen/Qwen3-Reranker-0.6B`, zero-shot | e61197ed45024b0ed8a2d74b80b4d909f1255473 | Apache-2.0 |
  | Decision, candidate 1 | `fastino/GLiNER2.5-Decide`, zero-shot | 9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90 | Apache-2.0 |
  | Decision, candidate 2 | `fastino/GLiDE-0.4B`, zero-shot | 6bda7ab91aeb36dcad6d27828a8871e548c7fc71 | Fastino Community (partner agreement pending) |
  | Candidate retrieval | `google/embeddinggemma-2`, 256-d | f9972ac22e696a105c482202646979e33e4b5a1e | Apache-2.0 |
  | Structured matching | Splink 5.0.0 on DuckDB | – | MIT |

- **GLiDE size (owner, 2026-11-11):** start with the smallest and go up only if needed. Only
  the 0.4B variant has run; the 0.8B, 2B and 9B variants have not.
- **Labels:** the generator's truth stands in for the reviewer. The first stage uses 2,000
  labelled pairs to calibrate and 3,000 for Learn-then-Test; the second stage 1,000 and 1,500,
  drawn only from the pairs the first stage left undecided. Each stage runs at δ/2.
- **Every number here is one run with seed 7.** No variance across seeds has been measured.

## Structured records

### 10⁵ records (synthetic estate, 98,971 records, 41,250 organisations)

The statistical matcher scores every candidate pair. The pairs it leaves undecided (23,607)
are written as text and scored again by the named models; those scores and the matcher's are
calibrated together, tested, and clustered under the same constraints.

| Arm | Auto-accept precision (Wilson lower bound) | Machine only: pairwise P / R / F1 | Machine only: B-cubed F1 | Pairs left for review | Expert min per 1,000 records |
|---|---|---|---|---|---|
| Matcher alone | 0.990 (0.989) | 0.988 / 0.757 / 0.858 | 0.920 | 23,607 | 36.7 |
| + reranker | 0.987 (0.986) | 0.989 / 0.872 / 0.927 | 0.957 | 13,355 | 23.3 |
| + Decide | 0.990 (0.989) | 0.988 / 0.758 / 0.858 | 0.920 | 23,607 | 36.7 |
| + GLiDE 0.4B | 0.987 (0.986) | 0.992 / 0.894 / 0.940 | 0.964 | 10,390 | 17.3 |
| + reranker + GLiDE 0.4B | 0.986 (0.985) | 0.991 / 0.900 / 0.944 | 0.967 | 8,962 | 14.4 |

- **With the simulated reviewer** every arm ends at pairwise F1 0.953 to 0.956. The models do
  not raise the end quality; they move work from the reviewer to the machine.
- **Plain transitive closure** over the matcher's accepted pairs gives pairwise F1 0.799. The
  constrained clusterer is worth about six points before any model.
- **Recall is capped by the candidates:** 8.5% of true pairs are never proposed (pair
  completeness 0.915), and 0.64% of the rejected pairs are true matches.
- **The existing cascade** (measured earlier on the 12k estate, the largest it finishes in
  reasonable time): pairwise F1 0.540 in 400 s, against 0.913 for the matcher alone on the same records.

### 10⁶ records (synthetic estate, 989,658 records)

| Run | Labels | Auto-accept precision | Machine only: pairwise P / R / F1 | Review band | Time |
|---|---|---|---|---|---|
| Matcher alone, uniform labels | 5,000 | none certified | nothing merged | 609,117 pairs | 92 s |
| Matcher alone, uniform labels (earlier, 4 CPUs) | 50,000 | 0.980 | 0.986 / 0.758 / 0.857 | none | 83 s |
| Matcher + second stage with GLiDE 0.4B | 5,000 + 2,500 | 0.9955 (0.9953) | 0.997 / 0.710 / 0.829 | none | 56 min |

- **The second stage's labels did the work, not the model.** The first stage certified only a
  reject line. The 609,117 pairs above it are 99.5% true matches, so 1,500 test labels drawn
  from them certify accepting all of them. GLiDE's scores changed no decision.
- **This confirms the earlier finding:** uniform labels over candidate pairs do not scale
  (2.8% of candidates are matches at this size); labels drawn where the matcher is undecided
  do. The base rate alone certifies the accept, so a second stage without any model should
  give the same result in about two minutes; that arm does not exist in the code yet.
- **Where the recall goes:** 10% of true pairs are never candidates, and the reject line sits
  at 0.72, so true matches below it are lost. A model pass over the pairs just below the line
  is the obvious use of the GPU at this size. It is not built.
- **Neural throughput:** 190 pairs per second for GLiDE 0.4B at batch size 16, not tuned.

### Public dataset (Splink `historical_50k`, 50,578 person records)

| Arm | Auto-accept precision | Machine only: pairwise P / R / F1 | Review band |
|---|---|---|---|
| Matcher alone | 0.971 (0.970) | 0.918 / 0.822 / 0.868 | 58,399 pairs (21.7%) |
| + reranker | 0.971 (0.970) | 0.917 / 0.840 / 0.877 | 58,399 pairs, none moved |
| + GLiDE 0.4B | 0.971 (0.970) | 0.910 / 0.841 / 0.874 | 58,399 pairs, none moved |
| + reranker + GLiDE 0.4B | 0.971 (0.970) | 0.917 / 0.840 / 0.877 | 58,399 pairs, none moved |

- **On this dataset the models add nothing.** The second stage certifies no accept line with
  either model, so the review band keeps all 58,399 pairs. This is the only non-synthetic
  data in the spike, and it does not show the gain the synthetic estate shows. Why has not
  been investigated; the records are people with deliberately corrupted fields, which is a
  different problem from organisation names.

- No reject band is certified on this dataset, and candidate pair completeness is 0.694: the
  blocking passes are the weak point here, before any scoring.

## Text

### Mentions of organisations in short documents (3,000 documents, 4,106 mentions, 609 not in the knowledge base)

| Arm | Top-1 accuracy | Auto-linked | Auto-link precision (lower bound) | Machine only: B-cubed F1 | Left for review |
|---|---|---|---|---|---|
| Character n-grams only | 0.784 | none certified | – | 0.667 | 4,106 |
| + reranker | 0.854 | 55.9% | 0.964 (0.955) | 0.810 | 1,812 |
| + reranker + embedder | 0.851 | 59.7% | 0.961 (0.952) | 0.823 | 1,654 |
| + reranker + embedder + Decide | 0.851 | 61.0% | 0.959 (0.951) | 0.827 | 1,602 |
| + reranker + GLiDE 0.4B | 0.853 | 66.6% | 0.965 (0.958) | 0.854 | 1,371 |
| End to end: mentions extracted by Relex, + reranker + embedder | 0.852 | 60.6% | 0.960 (0.952) | 0.827 | 1,610 |

- **Extraction:** Relex finds 99.5% of the gold mentions at precision 0.845 (4,832 spans for
  4,106 mentions).
- **With the simulated reviewer:** pairwise F1 0.915 to 0.925; 87% to 92% of the mentions with
  no anchor are recognised as new.
- **The embedder adds little here:** candidate recall 0.920 → 0.928, no gain in top-1.

### Acronyms

In the set above an acronym ("GL" for a company) is linked correctly in at most 3% of cases:
it shares no characters with the name, and a two-letter form fits eight anchors on average.
The owner named this a priority (2026-11-11). [`aliases.py`](aliases.py) gathers the evidence
a reader would use, and a small ranker fitted on the calibration labels weighs it:

- the full name appears in the same thread and the letters fit it;
- a document defines it, "Lumenstone Foods Corp (LF)" or "IBM (International Business Machines)";
- the master data lists it as a short name;
- letters alone: every anchor that can be abbreviated that way, with skipped words and
  hyphen or CamelCase parts ("Mercedes-Benz" → "MB");
- plus how often the anchor is written about, and whether its city or industry is named nearby.

Measured on a second synthetic set written for this (`generate_text.py --threads`): 4,193
documents in 1,200 threads, 5,799 mentions, 1,183 acronym mentions with an anchor.

| Arm | Acronyms: top-1 | Acronyms: auto-linked | All mentions: top-1 | All mentions: auto-linked | Auto-link precision (lower bound) |
|---|---|---|---|---|---|
| Reranker, no alias evidence | 0.001 | 0% | 0.709 | 53.7% | 0.965 (0.958) |
| Alias evidence, no neural model | 0.762 | none certified | 0.846 | none certified | – |
| Reranker + alias evidence | 0.818 | 54.5% | 0.901 | 64.7% | 0.961 (0.955) |
| Reranker + alias evidence + GLiDE 0.4B | 0.824 | 58.6% | 0.905 | 70.7% | 0.959 (0.952) |
| End to end (Relex mentions), reranker + alias evidence | 0.826 | 55.4% | 0.905 | 66.9% | 0.959 (0.952) |

By the evidence an acronym mention has (reranker + alias evidence; a mention can have several):

| Evidence | Mentions | Top-1 | Auto-linked | Precision of those |
|---|---|---|---|---|
| Full name in the same thread | 704 | 0.974 | 84.4% | 0.997 |
| Defined somewhere in the corpus | 698 | 0.901 | 62.5% | 0.995 |
| Listed in an alias column | 239 | 0.841 | 60.3% | 0.972 |
| None (context only) | 201 | 0.423 | 4.5% | 1.0 |

- **How often real text carries each kind of evidence is a parameter of the generator, not a
  finding.** The table says what the method does given the evidence; the overall 82% follows
  from the assumed mix (12% of organisations are only ever written as acronyms in a thread,
  30% of first mentions define their acronym, 15% of anchors have an alias column).
- **A corpus-wide definition is not a rule.** Applied as one ("LF is always that company") it
  was right in 23% of cases, because two-letter forms collide. As one input to the ranker it
  helps.
- **Bare acronyms stay in review:** on the first text set, which has no threads, definitions
  or alias columns, the same code reaches 0.366 top-1 and auto-links almost nothing. The
  reviewer's choice has to become a stored alias; that loop is not built.
- **Not covered:** short forms that letters cannot derive ("VW" from "Volkswagen") unless a
  definition or alias column gives them; acronyms that are not organisations ("VP"), which is
  the extractor's typing decision and was not measured; real documents.

## What the models are worth

- **All of this is on synthetic organisations.** On the public person records neither the
  reranker nor GLiDE changed a decision.
- **GLiDE 0.4B beats GLiNER2.5-Decide in every comparison.** Zero-shot, Decide gives nearly
  the same answer for matches and non-matches (its weight in the calibrated score is near
  zero on records; on text it adds about one point of auto-linking). GLiDE 0.4B alone removes
  56% of the review pairs on records and adds 11 points of auto-linking on text.
- **The reranker is useful on text and second-best on records.** Combining it with GLiDE gains
  a further 14% fewer review pairs on records.
- **The embedder has not earned its place yet:** under one point of candidate recall on these
  sets. Acronyms and name stems are the retrieval gaps, and it closes neither.
- **Fine-tuning has not been tried.** The reranker's LoRA arm was to ship only if zero-shot
  missed the bar; zero-shot clears it.

## Throughput

| Step | Size | Time | Rate | Peak memory |
|---|---|---|---|---|
| Matcher end to end | 98,971 records | 13 s | 7,500 records/s | 1.0 GB |
| Matcher end to end | 989,658 records, 29.5M candidate pairs | 92 s | 10,800 records/s | 5.2 GB |
| Reranker 0.6B on record pairs | 23,607 pairs | 77 s | 307 pairs/s | – |
| GLiDE 0.4B on record pairs | 23,607 to 609,117 pairs | 2 to 53 min | 190 to 203 pairs/s | – |
| Decide on record pairs | 23,607 pairs | 157 s | 150 pairs/s | – |
| Relex extraction | 3,000 documents | 29 s | 104 documents/s | – |

Batch sizes are the defaults (16 or 32) and nothing is tuned; several jobs shared the GPU
during some runs, so the neural rates are lower bounds. Cost depends on the pod's hourly
price, which is not recorded here.

## Defects the runs found (both fixed, each with a test that failed first)

1. **The embedder searched with the whole document.** In documents naming two organisations
   it retrieved the other one, and the reranker preferred it: top-1 fell from 0.854 to 0.685
   and 681 of 1,100 wrong links were the document's other mention. It now searches with the
   mention alone. The numbers above are after the fix; the files before it are kept as
   `results/pod/b_*_docquery.json`.
2. **The public person dataset was scored with the organisation question.** The first three
   neural runs on it are discarded and were rerun with a person question. The outcome did not
   change: nothing certified in the second stage either way.

## Not done and limits

- **No real data.** Every estate and document is synthetic, except the public person
  dataset. The generator's hard cases follow a few patterns; a real estate will be harder.
- **No permissively licensed entity-linking benchmark** has been run. Candidates need a
  licence check first: Mewsli-X (code Apache-2.0, text from Wikinews), KILT's linking sets
  (MIT wrapper, mixed sources), ZESHEL (CC BY-SA, share-alike). None is cleared.
- **No hand-labelled slice** from an enterprise-like corpus.
- **No Spark run:** the Databricks sandbox does not exist yet.
- **No larger GLiDE variant, no fine-tuned arm, no model pass below the reject line at 10⁶.**
- **The review-item explanations** from the hosted model have not been generated (they gate
  nothing, and no key was set on this run).
- **One seed, one run per arm.** Differences of a point or less between arms are not
  established.
- **The label counts are simulated.** 7,500 labels for the 10⁶ estate is about 42 hours of
  review at 20 s each; who gives them and how fast is the workbench's question.

## Result files

- The model revisions are in the table under [Setup](#setup). `pin_models.py` writes them to
  `results/models.json` on the machine that runs the arms; that file is not tracked, because
  the repository's secret scan reads revision hashes inside JSON strings as secrets.
- `results/pod/a_100k_*.json`, `a_1m_*.json`, `a_hist50k_*.json`: structured arms.
- `results/pod/b_*.json`: text arms on `data/text_2k` (`*_docquery.json`: before defect 1).
- `results/pod/t_*.json`: text arms on `data/text_threads`.
- Earlier runs on the 4-CPU machine: `results/*.json`, described in [`NEXT.md`](NEXT.md).
