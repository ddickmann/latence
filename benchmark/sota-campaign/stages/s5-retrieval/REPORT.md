# S5 — Retrieval feature matrix (dense → +bm25 → +KG → +hyperedges → +CoE → +packer)

Query cap for this invocation: **none (full set)** — each dataset section states the cap its own cells were run with, since cells are written once and resumed. Gold resolution is always measured over the FULL query set. Legs are CUMULATIVE — leg *n* runs signals 1..*n* and fuses them with RRF, so every `Δ R@10` is the increment that one signal bought.

Dense retrieval is exact brute-force cosine over the exported 768-d Granite vectors (no ANN index exists — the S5 ticket's recorded choice); BM25 is the exported corpus statistics scored by the shipped `Bm25Rescorer`, which is exactly a full-corpus first stage because a document with no query term scores 0.

### How to read the columns

* `R@k` is **set** recall over gold CHUNKS: `|gold ∩ top-k| / |gold|`. Where a dataset's gold unit is a whole document (MultiHop-RAG articles, 2Wiki paragraphs, MuSiQue/HotpotQA supporting paragraphs), that document may have split into several chunks, so `R@k` is strictly harder than the published document-level number.
* `chain cov` **is** the document-level recall@10 for those datasets — each chain slot is one gold document, so the fraction of slots covered is the fraction of gold documents retrieved. Compare published document-level baselines against this column, not against `R@10`.
* `full chain` is the fraction of queries whose EVERY gold document made the top 10 — the number that separates a multi-hop retriever from a lucky single-hop one.
* The `packer` leg does not rank, it **selects**: its output is the token-budgeted subset that fits the budget, typically ~15 chunks. Its recall is therefore a different quantity from the legs above it (what survives into the prompt, not what the retriever could see) and a drop there is expected, not a regression.

## Gold resolution

`granularity` is the mix the gold **actually** resolved at, not the mix the dataset advertises: a page reference whose document has no reconstructable S2 page map keeps coarser *document*-level gold rather than being dropped, so the two counts are stated separately instead of implying uniform page precision. `excluded` counts queries whose evidence is absent from this corpus entirely — they are removed from the denominator with their reason, because scoring them as misses would report a corpus-coverage fact as a retrieval failure.

| dataset | refs resolved | rate | granularity (page / doc) | queries usable | excluded | verdict |
|---|---|---|---|---|---|---|
| hotpotqa | 7406/7406 | 100.0% | 0 / 7406 | 3703/3703 | 0 | ok |
| multihop_rag | 5908/5908 | 100.0% | 0 / 5908 | 2255/2556 | 0 | ok |
| musique | 3204/3204 | 100.0% | 0 / 3204 | 1209/1209 | 0 | ok |
| ohr_bench | 4541/4541 | 100.0% | 4523 / 18 | 4541/5039 | 498 (corpus-missing-domain 498) | ok |
| uda | 8583/8583 | 100.0% | 7310 / 1273 | 8583/8583 | 0 | ok |
| vidoseek | 1142/1142 | 100.0% | 1142 / 0 | 1142/1142 | 0 | ok |
| wiki2multihop | 30654/30654 | 100.0% | 0 / 30654 | 12576/12576 | 0 | ok |

## hotpotqa

*No leg has been run for this dataset yet — gold resolution only.*

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|

## multihop_rag

Legs run over **50** queries.

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.0607 | 0.2130 | 0.3030 | 0.4604 | 0.6322 | 0.2609 | 0.4800 | 0.7500 | — |
| bm25 | 0.0659 | 0.2230 | 0.3620 | 0.5080 | 0.7500 | 0.4783 | 269.6200 | 361.0300 | +0.0590 |
| kg | 0.0419 | 0.1955 | 0.2966 | 0.4144 | 0.6304 | 0.3043 | 272.9500 | 361.8200 | -0.0654 |
| hyperedges | 0.0332 | 0.1624 | 0.2622 | 0.3661 | 0.6793 | 0.3696 | 320.9400 | 429.4300 | -0.0344 |
| coe | 0.0273 | 0.1406 | 0.2383 | 0.3332 | 0.6793 | 0.3913 | 321.8000 | 445.8600 | -0.0239 |
| packer | 0.0273 | 0.1346 | 0.1777 | 0.2741 | 0.6377 | 0.3478 | 872.6400 | 1009.6800 | -0.0606 |

## musique

*No leg has been run for this dataset yet — gold resolution only.*

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|

## ohr_bench (single-hop CONTROL)

*No leg has been run for this dataset yet — gold resolution only.*

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|

## uda (single-hop CONTROL)

*No leg has been run for this dataset yet — gold resolution only.*

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|

## vidoseek (single-hop CONTROL)

Legs run over **50** queries.

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.5567 | 0.9067 | 0.9600 | 0.8059 | — | — | 0.4300 | 0.5700 | — |
| bm25 | 0.6067 | 0.9400 | 0.9900 | 0.8644 | — | — | 168.2200 | 254.0100 | +0.0300 |
| kg | 0.5867 | 0.9300 | 0.9900 | 0.8381 | — | — | 162.9800 | 244.3900 | +0.0000 |
| hyperedges | 0.4700 | 0.8200 | 0.9700 | 0.7659 | — | — | 172.5700 | 247.9800 | -0.0200 |
| coe | 0.3567 | 0.7800 | 0.9033 | 0.6803 | — | — | 171.2700 | 252.1200 | -0.0667 |
| packer | 0.3567 | 0.7500 | 0.8067 | 0.6479 | — | — | 736.7900 | 932.7900 | -0.0966 |

## wiki2multihop

*No leg has been run for this dataset yet — gold resolution only.*

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|

## Control check — where the hypergraph/CoE legs should buy nothing

ViDoSeek, OHR-Bench and UDA are single-hop: their gold is one page or one sentence, with no chain to assemble. A `+hyperedges` or `+coe` gain there is not a win — it indicates the leg is reaching gold through a path the question did not require (leakage), so it is reported as a warning, not a headline.

| control dataset | Δ R@10 from +hyperedges | Δ R@10 from +coe | reading |
|---|---|---|---|
| ohr_bench | — | — | — |
| uda | — | — | — |
| vidoseek | -0.0200 | -0.0667 | as expected (no gain on single-hop) |

## Scope

SPLADE, MaxSim/MUVERA and rerank/fusion ablations are OUT of this run (S5 ticket). Hop depth is 2–4 native only; 5–7-hop chains are untested and no claim is made about them.
