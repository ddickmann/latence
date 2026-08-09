# S5 — Retrieval feature matrix (dense → +bm25 → +KG → +hyperedges → +CoE → +packer)

Query cap per dataset: **50**. Legs are CUMULATIVE — leg *n* runs signals 1..*n* and fuses them with RRF, so every `Δ R@10` is the increment that one signal bought.

Dense retrieval is exact brute-force cosine over the exported 768-d Granite vectors (no ANN index exists — the S5 ticket's recorded choice); BM25 is the exported corpus statistics scored by the shipped `Bm25Rescorer`, which is exactly a full-corpus first stage because a document with no query term scores 0.

### How to read the columns

* `R@k` is **set** recall over gold CHUNKS: `|gold ∩ top-k| / |gold|`. Where a dataset's gold unit is a whole document (MultiHop-RAG articles, 2Wiki paragraphs, MuSiQue/HotpotQA supporting paragraphs), that document may have split into several chunks, so `R@k` is strictly harder than the published document-level number.
* `chain cov` **is** the document-level recall@10 for those datasets — each chain slot is one gold document, so the fraction of slots covered is the fraction of gold documents retrieved. Compare published document-level baselines against this column, not against `R@10`.
* `full chain` is the fraction of queries whose EVERY gold document made the top 10 — the number that separates a multi-hop retriever from a lucky single-hop one.
* The `packer` leg does not rank, it **selects**: its output is the token-budgeted subset that fits the budget, typically ~15 chunks. Its recall is therefore a different quantity from the legs above it (what survives into the prompt, not what the retriever could see) and a drop there is expected, not a regression.

## Gold resolution

| dataset | refs resolved | rate | queries usable | verdict |
|---|---|---|---|---|
| multihop_rag | 117/117 | 100.0% | 46/50 | ok |
| vidoseek | 50/50 | 100.0% | 50/50 | ok |

## multihop_rag

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.0607 | 0.2130 | 0.3030 | 0.4604 | 0.6322 | 0.2609 | 0.4800 | 0.7500 | — |
| bm25 | 0.0659 | 0.2230 | 0.3620 | 0.5080 | 0.7500 | 0.4783 | 269.6200 | 361.0300 | +0.0590 |
| kg | 0.0419 | 0.1955 | 0.2966 | 0.4144 | 0.6304 | 0.3043 | 272.9500 | 361.8200 | -0.0654 |
| hyperedges | 0.0332 | 0.1624 | 0.2622 | 0.3661 | 0.6793 | 0.3696 | 320.9400 | 429.4300 | -0.0344 |
| coe | 0.0273 | 0.1406 | 0.2383 | 0.3332 | 0.6793 | 0.3913 | 321.8000 | 445.8600 | -0.0239 |
| packer | 0.0273 | 0.1346 | 0.1777 | 0.2741 | 0.6377 | 0.3478 | 872.6400 | 1009.6800 | -0.0606 |

## vidoseek (single-hop CONTROL)

| leg | R@1 | R@5 | R@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|
| dense | 0.5567 | 0.9067 | 0.9600 | 0.8059 | — | — | 0.4300 | 0.5700 | — |
| bm25 | 0.6067 | 0.9400 | 0.9900 | 0.8644 | — | — | 168.2200 | 254.0100 | +0.0300 |
| kg | 0.5867 | 0.9300 | 0.9900 | 0.8381 | — | — | 162.9800 | 244.3900 | +0.0000 |
| hyperedges | 0.4700 | 0.8200 | 0.9700 | 0.7659 | — | — | 172.5700 | 247.9800 | -0.0200 |
| coe | 0.3567 | 0.7800 | 0.9033 | 0.6803 | — | — | 171.2700 | 252.1200 | -0.0667 |
| packer | 0.3567 | 0.7500 | 0.8067 | 0.6479 | — | — | 736.7900 | 932.7900 | -0.0966 |

## Control check — where the hypergraph/CoE legs should buy nothing

ViDoSeek, OHR-Bench and UDA are single-hop: their gold is one page or one sentence, with no chain to assemble. A `+hyperedges` or `+coe` gain there is not a win — it indicates the leg is reaching gold through a path the question did not require (leakage), so it is reported as a warning, not a headline.

| control dataset | Δ R@10 from +hyperedges | Δ R@10 from +coe | reading |
|---|---|---|---|
| vidoseek | -0.0200 | -0.0667 | as expected (no gain on single-hop) |

## Scope

SPLADE, MaxSim/MUVERA and rerank/fusion ablations are OUT of this run (S5 ticket). Hop depth is 2–4 native only; 5–7-hop chains are untested and no claim is made about them.
