# S5 — Retrieval feature matrix (dense → +bm25 → +KG → +hyperedges → +CoE → +packer)

Query cap for this invocation: **none (full set)** — each dataset section states the cap its own cells were run with, since cells are written once and resumed. Gold resolution is always measured over the FULL query set. Legs are CUMULATIVE — leg *n* runs signals 1..*n* and fuses them with RRF, so every `Δ R@10` is the increment that one signal bought.

**Every number in the ladder tables below is EQUAL-weight RRF** — each leg's signals enter the fusion with the same pull. That is the assumption-free baseline, and it is also why a graph leg can lower `R@10`: RRF sums `w / (k + rank)`, so appending a noisier list at equal weight to a strong `dense+bm25` pool necessarily drags the fused ranking toward the noise. The dev-tuned variant (its own section, if it was run) is the controlled answer to "does the signal help when fused properly?"; it is NOT a replacement for these numbers.

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

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.2201 | 0.4975 | 0.5576 | 0.6947 | 0.7456 | 0.5984 | 0.7305 | 0.5115 | 21.6300 | 22.9300 | — |
| bm25 | 0.2210 | 0.5035 | 0.5786 | 0.7449 | 0.8119 | 0.6075 | 0.7960 | 0.6130 | 25.3900 | 27.2100 | +0.0210 |
| kg | 0.2210 | 0.5035 | 0.5786 | 0.7449 | 0.8119 | 0.6075 | 0.7960 | 0.6130 | 25.6200 | 30.5500 | +0.0000 |
| hyperedges | 0.1853 | 0.4797 | 0.5710 | 0.7208 | 0.8056 | 0.5669 | 0.7879 | 0.5976 | 50.6200 | 64.7500 | -0.0076 |
| coe | 0.1506 | 0.3934 | 0.4965 | 0.6215 | 0.7278 | 0.4736 | 0.7143 | 0.4761 | 53.2700 | 69.7700 | -0.0745 |
| packer | 0.1490 | 0.3569 | 0.4103 | 0.5695 | 0.6204 | 0.4234 | 0.6181 | 0.3508 | 323.8800 | 567.2400 | -0.0862 |

## multihop_rag

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.0521 | 0.2123 | 0.3328 | 0.6117 | 0.7700 | 0.4590 | 0.6263 | 0.3415 | 0.5900 | 0.6500 | — |
| bm25 | 0.0603 | 0.2427 | 0.3772 | 0.7096 | 0.8523 | 0.5235 | 0.7254 | 0.4559 | 1.3300 | 1.5100 | +0.0444 |
| kg | 0.0441 | 0.1938 | 0.3267 | 0.6368 | 0.8303 | 0.4365 | 0.6521 | 0.3729 | 2.3100 | 3.0700 | -0.0505 |
| hyperedges | 0.0431 | 0.1769 | 0.2879 | 0.6164 | 0.7946 | 0.3932 | 0.6712 | 0.4035 | 16.7500 | 31.6000 | -0.0388 |
| coe | 0.0406 | 0.1624 | 0.2523 | 0.5722 | 0.7353 | 0.3536 | 0.6468 | 0.3685 | 19.0000 | 33.6300 | -0.0356 |
| packer | 0.0405 | 0.1546 | 0.2028 | 0.5452 | 0.6269 | 0.3071 | 0.6206 | 0.3282 | 724.2200 | 883.5600 | -0.0495 |

## musique

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.1283 | 0.2765 | 0.3353 | 0.4887 | 0.5726 | 0.3986 | 0.5478 | 0.1853 | 5.8600 | 6.1800 | — |
| bm25 | 0.1343 | 0.2861 | 0.3404 | 0.5239 | 0.6068 | 0.4161 | 0.5905 | 0.2266 | 7.2900 | 7.7100 | +0.0051 |
| kg | 0.1343 | 0.2861 | 0.3404 | 0.5239 | 0.6068 | 0.4161 | 0.5905 | 0.2266 | 7.6900 | 8.9000 | +0.0000 |
| hyperedges | 0.1240 | 0.2805 | 0.3392 | 0.5167 | 0.6011 | 0.4019 | 0.5834 | 0.2117 | 22.3300 | 34.6600 | -0.0012 |
| coe | 0.0985 | 0.2197 | 0.2847 | 0.4217 | 0.5159 | 0.3243 | 0.5026 | 0.1431 | 23.9400 | 36.9900 | -0.0545 |
| packer | 0.0979 | 0.2104 | 0.2500 | 0.4030 | 0.4638 | 0.3028 | 0.4575 | 0.1092 | 308.5400 | 536.9400 | -0.0347 |

## ohr_bench (single-hop CONTROL)

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.2279 | 0.4949 | 0.5890 | 0.9194 | 0.9518 | 0.5294 | — | — | 1.8900 | 2.7600 | — |
| bm25 | 0.2796 | 0.5697 | 0.6609 | 0.9599 | 0.9760 | 0.6137 | — | — | 2.8100 | 3.1400 | +0.0719 |
| kg | 0.2039 | 0.4875 | 0.6129 | 0.9513 | 0.9709 | 0.5164 | — | — | 3.2900 | 4.5000 | -0.0480 |
| hyperedges | 0.1536 | 0.3893 | 0.5265 | 0.9168 | 0.9584 | 0.4194 | — | — | 14.1400 | 27.8500 | -0.0864 |
| coe | 0.1218 | 0.2797 | 0.3657 | 0.8243 | 0.8833 | 0.3073 | — | — | 16.2400 | 30.2500 | -0.1608 |
| packer | 0.1207 | 0.2459 | 0.2793 | 0.7978 | 0.8381 | 0.2636 | — | — | 702.8700 | 979.8100 | -0.0864 |

## uda (single-hop CONTROL)

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.0145 | 0.0416 | 0.0594 | 0.2186 | 0.2927 | 0.0511 | — | — | 25.4700 | 29.1400 | — |
| bm25 | 0.0297 | 0.0860 | 0.1261 | 0.3501 | 0.4455 | 0.1055 | — | — | 33.0100 | 37.1400 | +0.0667 |
| kg | 0.0271 | 0.0833 | 0.1218 | 0.3431 | 0.4364 | 0.1014 | — | — | 37.1500 | 60.8800 | -0.0043 |
| hyperedges | 0.0194 | 0.0652 | 0.0993 | 0.2882 | 0.3880 | 0.0789 | — | — | 79.5600 | 107.6200 | -0.0225 |
| coe | 0.0090 | 0.0341 | 0.0570 | 0.1728 | 0.2610 | 0.0430 | — | — | 85.9100 | 113.8800 | -0.0423 |
| packer | 0.0084 | 0.0221 | 0.0266 | 0.1325 | 0.1864 | 0.0265 | — | — | 770.0300 | 987.4900 | -0.0304 |

## vidoseek (single-hop CONTROL)

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.5070 | 0.8702 | 0.9466 | 0.9764 | 0.9860 | 0.7993 | — | — | 0.5300 | 0.5700 | — |
| bm25 | 0.5743 | 0.9271 | 0.9705 | 0.9921 | 0.9956 | 0.8550 | — | — | 1.1200 | 1.2000 | +0.0239 |
| kg | 0.5144 | 0.8951 | 0.9638 | 0.9947 | 0.9965 | 0.8125 | — | — | 1.2900 | 1.6800 | -0.0067 |
| hyperedges | 0.4246 | 0.8230 | 0.9301 | 0.9851 | 0.9947 | 0.7400 | — | — | 10.7000 | 24.5500 | -0.0337 |
| coe | 0.3483 | 0.7585 | 0.8664 | 0.9510 | 0.9807 | 0.6612 | — | — | 10.7900 | 24.1200 | -0.0637 |
| packer | 0.3496 | 0.7230 | 0.7538 | 0.9457 | 0.9685 | 0.6165 | — | — | 721.0300 | 934.3700 | -0.1126 |

## wiki2multihop

Legs run over **the full query set** queries.

| leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | nDCG@10 | chain cov | full chain | lat med ms | lat p95 ms | Δ R@10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0.3644 | 0.6338 | 0.6753 | 0.6456 | 0.6862 | 0.6837 | 0.6863 | 0.3819 | 6.6300 | 7.6700 | — |
| bm25 | 0.3813 | 0.6618 | 0.7018 | 0.6735 | 0.7122 | 0.7172 | 0.7125 | 0.4076 | 8.3500 | 9.1700 | +0.0265 |
| kg | 0.3650 | 0.6668 | 0.7037 | 0.6785 | 0.7142 | 0.7104 | 0.7145 | 0.4139 | 9.9200 | 13.0600 | +0.0019 |
| hyperedges | 0.3232 | 0.6345 | 0.6865 | 0.6463 | 0.6967 | 0.6589 | 0.6969 | 0.3849 | 32.9500 | 45.8600 | -0.0172 |
| coe | 0.2805 | 0.5082 | 0.5949 | 0.5181 | 0.6055 | 0.5575 | 0.6059 | 0.2784 | 35.6600 | 49.0100 | -0.0916 |
| packer | 0.2805 | 0.4900 | 0.5446 | 0.4993 | 0.5540 | 0.5331 | 0.5547 | 0.2417 | 510.0500 | 965.5300 | -0.0503 |

## Ablation — what the W13 context header bought the dense leg

The `dense` rung ranks the vectors the S3 Export wrote, which embed each chunk's content **plus** its rendered ContextHeader (the chunk's KG entities and neighbour triples). `dense (content-only)` re-runs that same rung against vectors built from the content alone. Only the CORPUS side differs — queries carry no header, so both legs rank identical query vectors. A positive Δ is what the enrichment is worth; it also says how much of every leg above `dense` is header, not retrieval. This matters unevenly: the S3 corpora carry populated headers while the reused M6 exports carry `context_header: null` on every row, so without this table a single `dense` label would silently mean two different things across datasets.

| dataset | leg | baseline R@10 | ablated R@10 | Δ (header − content-only) |
|---|---|---|---|---|
| hotpotqa | dense_content_only | — | *missing inputs: corpus-vectors-content-only.parquet* | — |
| multihop_rag | dense_content_only | 0.3328 | 0.3347 | -0.0019 |
| musique | dense_content_only | — | *missing inputs: corpus-vectors-content-only.parquet* | — |
| ohr_bench | dense_content_only | 0.5890 | 0.5936 | -0.0046 |
| uda | dense_content_only | 0.0594 | 0.0621 | -0.0027 |
| vidoseek | dense_content_only | 0.9466 | 0.9450 | +0.0016 |
| wiki2multihop | dense_content_only | 0.6753 | 0.6769 | -0.0016 |

## Graph as RESCUE, not as a fused list (S5b)

The ladder applies the graph lane as a co-equal RRF list, which is the application the matrix above measures regressing. These legs apply the *same shipped components* the way the framework's seams (and `colsearch`'s shipped graph sidecar) describe: `dense+bm25` is the base ranking; `expand.Expander` rescores it from the KG columns already denormalised onto every chunk; the graph lanes only **append** candidates the base missed, deduplicated by id and never reordering the base; and a selector spends a token budget over the union. Entity resolution is seeded from the query **and** the top first-stage documents. Zero LLM calls.

| dataset | leg | R@1 | R@5 | R@10 | docR@5 | docR@10 | chain cov | full chain | gate hit rate | lat med ms |
|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | kg | 0.2210 | 0.5035 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 0.6130 | — (ladder leg: no gate) | 25.6200 |
| hotpotqa | hyperedges | 0.1853 | 0.4797 | 0.5710 | 0.7208 | 0.8056 | 0.7879 | 0.5976 | — (ladder leg: no gate) | 50.6200 |
| hotpotqa | coe | 0.1506 | 0.3934 | 0.4965 | 0.6215 | 0.7278 | 0.7143 | 0.4761 | — (ladder leg: no gate) | 53.2700 |
| hotpotqa | packer | 0.1490 | 0.3569 | 0.4103 | 0.5695 | 0.6204 | 0.6181 | 0.3508 | — (ladder leg: no gate) | 323.8800 |
| hotpotqa | graph_rescue | 0.2210 | 0.5035 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 0.6130 | 100.0% | 144.4700 |
| hotpotqa | graph_rescue_packed | 0.2210 | 0.5035 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 0.6130 | 100.0% | 176.0800 |
| hotpotqa | graph_rescue_coe | 0.2210 | 0.5035 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 0.6130 | 100.0% | 146.2800 |
| multihop_rag | kg | 0.0441 | 0.1938 | 0.3267 | 0.6368 | 0.8303 | 0.6521 | 0.3729 | — (ladder leg: no gate) | 2.3100 |
| multihop_rag | hyperedges | 0.0431 | 0.1769 | 0.2879 | 0.6164 | 0.7946 | 0.6712 | 0.4035 | — (ladder leg: no gate) | 16.7500 |
| multihop_rag | coe | 0.0406 | 0.1624 | 0.2523 | 0.5722 | 0.7353 | 0.6468 | 0.3685 | — (ladder leg: no gate) | 19.0000 |
| multihop_rag | packer | 0.0405 | 0.1546 | 0.2028 | 0.5452 | 0.6269 | 0.6206 | 0.3282 | — (ladder leg: no gate) | 724.2200 |
| multihop_rag | graph_rescue | 0.0597 | 0.2348 | 0.3759 | 0.6991 | 0.8551 | 0.7112 | 0.4288 | 100.0% | 509.5700 |
| multihop_rag | graph_rescue_packed | 0.0597 | 0.2348 | 0.3759 | 0.6981 | 0.8527 | 0.7112 | 0.4288 | 100.0% | 530.2000 |
| multihop_rag | graph_rescue_coe | 0.0597 | 0.2348 | 0.3759 | 0.6989 | 0.8534 | 0.7112 | 0.4288 | 100.0% | 523.5900 |
| musique | kg | 0.1343 | 0.2861 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 0.2266 | — (ladder leg: no gate) | 7.6900 |
| musique | hyperedges | 0.1240 | 0.2805 | 0.3392 | 0.5167 | 0.6011 | 0.5834 | 0.2117 | — (ladder leg: no gate) | 22.3300 |
| musique | coe | 0.0985 | 0.2197 | 0.2847 | 0.4217 | 0.5159 | 0.5026 | 0.1431 | — (ladder leg: no gate) | 23.9400 |
| musique | packer | 0.0979 | 0.2104 | 0.2500 | 0.4030 | 0.4638 | 0.4575 | 0.1092 | — (ladder leg: no gate) | 308.5400 |
| musique | graph_rescue | 0.1343 | 0.2861 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 0.2266 | 99.9% | 114.5900 |
| musique | graph_rescue_packed | 0.1343 | 0.2861 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 0.2266 | 99.9% | 145.2900 |
| musique | graph_rescue_coe | 0.1343 | 0.2861 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 0.2266 | 99.9% | 118.0500 |
| ohr_bench | kg | 0.2039 | 0.4875 | 0.6129 | 0.9513 | 0.9709 | — | — | — (ladder leg: no gate) | 3.2900 |
| ohr_bench | hyperedges | 0.1536 | 0.3893 | 0.5265 | 0.9168 | 0.9584 | — | — | — (ladder leg: no gate) | 14.1400 |
| ohr_bench | coe | 0.1218 | 0.2797 | 0.3657 | 0.8243 | 0.8833 | — | — | — (ladder leg: no gate) | 16.2400 |
| ohr_bench | packer | 0.1207 | 0.2459 | 0.2793 | 0.7978 | 0.8381 | — | — | — (ladder leg: no gate) | 702.8700 |
| ohr_bench | graph_rescue | 0.2487 | 0.5474 | 0.6589 | 0.9588 | 0.9738 | — | — | 100.0% | 440.9800 |
| ohr_bench | graph_rescue_packed | 0.2457 | 0.5320 | 0.6352 | 0.9560 | 0.9718 | — | — | 100.0% | 466.5800 |
| ohr_bench | graph_rescue_coe | 0.2487 | 0.5474 | 0.6589 | 0.9588 | 0.9738 | — | — | 100.0% | 460.4600 |
| uda | kg | 0.0271 | 0.0833 | 0.1218 | 0.3431 | 0.4364 | — | — | — (ladder leg: no gate) | 37.1500 |
| uda | hyperedges | 0.0194 | 0.0652 | 0.0993 | 0.2882 | 0.3880 | — | — | — (ladder leg: no gate) | 79.5600 |
| uda | coe | 0.0090 | 0.0341 | 0.0570 | 0.1728 | 0.2610 | — | — | — (ladder leg: no gate) | 85.9100 |
| uda | packer | 0.0084 | 0.0221 | 0.0266 | 0.1325 | 0.1864 | — | — | — (ladder leg: no gate) | 770.0300 |
| uda | graph_rescue | 0.0293 | 0.0865 | 0.1276 | 0.3510 | 0.4466 | — | — | 100.0% | 646.0400 |
| uda | graph_rescue_packed | 0.0293 | 0.0865 | 0.1271 | 0.3509 | 0.4444 | — | — | 100.0% | 673.4200 |
| uda | graph_rescue_coe | 0.0293 | 0.0865 | 0.1276 | 0.3509 | 0.4463 | — | — | 100.0% | 664.6100 |
| vidoseek | kg | 0.5144 | 0.8951 | 0.9638 | 0.9947 | 0.9965 | — | — | — (ladder leg: no gate) | 1.2900 |
| vidoseek | hyperedges | 0.4246 | 0.8230 | 0.9301 | 0.9851 | 0.9947 | — | — | — (ladder leg: no gate) | 10.7000 |
| vidoseek | coe | 0.3483 | 0.7585 | 0.8664 | 0.9510 | 0.9807 | — | — | — (ladder leg: no gate) | 10.7900 |
| vidoseek | packer | 0.3496 | 0.7230 | 0.7538 | 0.9457 | 0.9685 | — | — | — (ladder leg: no gate) | 721.0300 |
| vidoseek | graph_rescue | 0.5075 | 0.8957 | 0.9726 | 0.9921 | 0.9956 | — | — | 99.1% | 467.0100 |
| vidoseek | graph_rescue_packed | 0.4816 | 0.7985 | 0.8424 | 0.9921 | 0.9947 | — | — | 99.1% | 502.4000 |
| vidoseek | graph_rescue_coe | 0.5075 | 0.8957 | 0.9726 | 0.9921 | 0.9956 | — | — | 99.1% | 485.0100 |
| wiki2multihop | kg | 0.3650 | 0.6668 | 0.7037 | 0.6785 | 0.7142 | 0.7145 | 0.4139 | — (ladder leg: no gate) | 9.9200 |
| wiki2multihop | hyperedges | 0.3232 | 0.6345 | 0.6865 | 0.6463 | 0.6967 | 0.6969 | 0.3849 | — (ladder leg: no gate) | 32.9500 |
| wiki2multihop | coe | 0.2805 | 0.5082 | 0.5949 | 0.5181 | 0.6055 | 0.6059 | 0.2784 | — (ladder leg: no gate) | 35.6600 |
| wiki2multihop | packer | 0.2805 | 0.4900 | 0.5446 | 0.4993 | 0.5540 | 0.5547 | 0.2417 | — (ladder leg: no gate) | 510.0500 |
| wiki2multihop | graph_rescue | 0.3530 | 0.6583 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 0.4222 | 100.0% | 240.8600 |
| wiki2multihop | graph_rescue_packed | 0.3530 | 0.6583 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 0.4222 | 100.0% | 257.2300 |
| wiki2multihop | graph_rescue_coe | 0.3530 | 0.6583 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 0.4222 | 100.0% | 246.6700 |

### Recall over the token budget (the selection legs' native protocol)

A selector optimises *best context within a budget*, so a bare `recall@10` is the wrong shape of number for it. The curve below settles every budget against the **identical** candidate union — one retrieval per query, five selections — so what varies is the budget and nothing else. The headline row above is the `select_budget` point of this curve.

| dataset | leg | budget | R@10 | docR@5 | docR@10 | chain cov | tokens packed (mean) |
|---|---|---|---|---|---|---|---|
| hotpotqa | graph_rescue_packed | 2000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 1995.7000 |
| hotpotqa | graph_rescue_packed | 4000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 3798.3000 |
| hotpotqa | graph_rescue_packed | 8000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_packed | 10000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_packed | 16000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_packed | 24000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_packed | 32000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_coe | 2000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 1998.4000 |
| hotpotqa | graph_rescue_coe | 4000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 3799.5000 |
| hotpotqa | graph_rescue_coe | 8000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_coe | 10000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_coe | 16000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_coe | 24000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| hotpotqa | graph_rescue_coe | 32000 | 0.5786 | 0.7449 | 0.8119 | 0.7960 | 4118.9000 |
| multihop_rag | graph_rescue_packed | 2000 | 0.1665 | 0.5860 | 0.6150 | 0.6150 | 1922.1000 |
| multihop_rag | graph_rescue_packed | 4000 | 0.2784 | 0.6701 | 0.7707 | 0.7506 | 3931.9000 |
| multihop_rag | graph_rescue_packed | 8000 | 0.3702 | 0.6977 | 0.8449 | 0.7136 | 7935.9000 |
| multihop_rag | graph_rescue_packed | 10000 | 0.3759 | 0.6981 | 0.8527 | 0.7112 | 9931.3000 |
| multihop_rag | graph_rescue_packed | 16000 | 0.3759 | 0.6991 | 0.8538 | 0.7112 | 15925.3000 |
| multihop_rag | graph_rescue_packed | 24000 | 0.3759 | 0.6991 | 0.8551 | 0.7112 | 23919.2000 |
| multihop_rag | graph_rescue_packed | 32000 | 0.3759 | 0.6991 | 0.8551 | 0.7112 | 31906.6000 |
| multihop_rag | graph_rescue_coe | 2000 | 0.2016 | 0.5303 | 0.5303 | 0.5303 | 1950.9000 |
| multihop_rag | graph_rescue_coe | 4000 | 0.3401 | 0.6622 | 0.6840 | 0.6839 | 3950.3000 |
| multihop_rag | graph_rescue_coe | 8000 | 0.3759 | 0.6972 | 0.8157 | 0.7112 | 7946.8000 |
| multihop_rag | graph_rescue_coe | 10000 | 0.3759 | 0.6985 | 0.8403 | 0.7112 | 9947.9000 |
| multihop_rag | graph_rescue_coe | 16000 | 0.3759 | 0.6989 | 0.8534 | 0.7112 | 15944.4000 |
| multihop_rag | graph_rescue_coe | 24000 | 0.3759 | 0.6991 | 0.8551 | 0.7112 | 23935.8000 |
| multihop_rag | graph_rescue_coe | 32000 | 0.3759 | 0.6991 | 0.8551 | 0.7112 | 31916.7000 |
| musique | graph_rescue_packed | 2000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 1995.8000 |
| musique | graph_rescue_packed | 4000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 3890.2000 |
| musique | graph_rescue_packed | 8000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_packed | 10000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_packed | 16000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_packed | 24000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_packed | 32000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_coe | 2000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 1998.9000 |
| musique | graph_rescue_coe | 4000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 3891.6000 |
| musique | graph_rescue_coe | 8000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_coe | 10000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_coe | 16000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_coe | 24000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| musique | graph_rescue_coe | 32000 | 0.3404 | 0.5239 | 0.6068 | 0.5905 | 4281.4000 |
| ohr_bench | graph_rescue_packed | 2000 | 0.2526 | 0.8593 | 0.8663 | — | 1925.8000 |
| ohr_bench | graph_rescue_packed | 4000 | 0.4004 | 0.9267 | 0.9423 | — | 3930.0000 |
| ohr_bench | graph_rescue_packed | 8000 | 0.5820 | 0.9544 | 0.9689 | — | 7930.9000 |
| ohr_bench | graph_rescue_packed | 10000 | 0.6352 | 0.9560 | 0.9718 | — | 9930.6000 |
| ohr_bench | graph_rescue_packed | 16000 | 0.6576 | 0.9586 | 0.9736 | — | 15923.8000 |
| ohr_bench | graph_rescue_packed | 24000 | 0.6584 | 0.9588 | 0.9738 | — | 23922.1000 |
| ohr_bench | graph_rescue_packed | 32000 | 0.6589 | 0.9588 | 0.9738 | — | 31903.9000 |
| ohr_bench | graph_rescue_coe | 2000 | 0.5203 | 0.9353 | 0.9353 | — | 1960.0000 |
| ohr_bench | graph_rescue_coe | 4000 | 0.6407 | 0.9546 | 0.9599 | — | 3960.1000 |
| ohr_bench | graph_rescue_coe | 8000 | 0.6589 | 0.9584 | 0.9725 | — | 7957.5000 |
| ohr_bench | graph_rescue_coe | 10000 | 0.6589 | 0.9586 | 0.9734 | — | 9955.7000 |
| ohr_bench | graph_rescue_coe | 16000 | 0.6589 | 0.9588 | 0.9738 | — | 15952.0000 |
| ohr_bench | graph_rescue_coe | 24000 | 0.6589 | 0.9588 | 0.9738 | — | 23941.5000 |
| ohr_bench | graph_rescue_coe | 32000 | 0.6589 | 0.9588 | 0.9738 | — | 31921.6000 |
| uda | graph_rescue_packed | 2000 | 0.0440 | 0.2775 | 0.2882 | — | 1925.3000 |
| uda | graph_rescue_packed | 4000 | 0.0701 | 0.3159 | 0.3731 | — | 3930.7000 |
| uda | graph_rescue_packed | 8000 | 0.1246 | 0.3506 | 0.4391 | — | 7928.2000 |
| uda | graph_rescue_packed | 10000 | 0.1271 | 0.3509 | 0.4444 | — | 9927.3000 |
| uda | graph_rescue_packed | 16000 | 0.1276 | 0.3509 | 0.4463 | — | 15923.7000 |
| uda | graph_rescue_packed | 24000 | 0.1276 | 0.3509 | 0.4465 | — | 23914.3000 |
| uda | graph_rescue_packed | 32000 | 0.1276 | 0.3510 | 0.4466 | — | 31897.9000 |
| uda | graph_rescue_coe | 2000 | 0.0781 | 0.3136 | 0.3154 | — | 1946.6000 |
| uda | graph_rescue_coe | 4000 | 0.1200 | 0.3494 | 0.4098 | — | 3949.6000 |
| uda | graph_rescue_coe | 8000 | 0.1276 | 0.3506 | 0.4441 | — | 7946.7000 |
| uda | graph_rescue_coe | 10000 | 0.1276 | 0.3507 | 0.4454 | — | 9945.3000 |
| uda | graph_rescue_coe | 16000 | 0.1276 | 0.3509 | 0.4463 | — | 15940.4000 |
| uda | graph_rescue_coe | 24000 | 0.1276 | 0.3509 | 0.4465 | — | 23928.3000 |
| uda | graph_rescue_coe | 32000 | 0.1276 | 0.3510 | 0.4466 | — | 31906.5000 |
| vidoseek | graph_rescue_packed | 2000 | 0.2747 | 0.9378 | 0.9387 | — | 1920.5000 |
| vidoseek | graph_rescue_packed | 4000 | 0.5599 | 0.9895 | 0.9921 | — | 3930.6000 |
| vidoseek | graph_rescue_packed | 8000 | 0.7545 | 0.9930 | 0.9947 | — | 7936.1000 |
| vidoseek | graph_rescue_packed | 10000 | 0.8424 | 0.9921 | 0.9947 | — | 9931.8000 |
| vidoseek | graph_rescue_packed | 16000 | 0.9631 | 0.9921 | 0.9956 | — | 15931.4000 |
| vidoseek | graph_rescue_packed | 24000 | 0.9696 | 0.9921 | 0.9956 | — | 23922.4000 |
| vidoseek | graph_rescue_packed | 32000 | 0.9726 | 0.9921 | 0.9956 | — | 31912.8000 |
| vidoseek | graph_rescue_coe | 2000 | 0.8624 | 0.9877 | 0.9877 | — | 1966.5000 |
| vidoseek | graph_rescue_coe | 4000 | 0.9648 | 0.9904 | 0.9930 | — | 3966.8000 |
| vidoseek | graph_rescue_coe | 8000 | 0.9726 | 0.9921 | 0.9947 | — | 7964.8000 |
| vidoseek | graph_rescue_coe | 10000 | 0.9726 | 0.9921 | 0.9956 | — | 9965.0000 |
| vidoseek | graph_rescue_coe | 16000 | 0.9726 | 0.9921 | 0.9956 | — | 15960.7000 |
| vidoseek | graph_rescue_coe | 24000 | 0.9726 | 0.9921 | 0.9956 | — | 23951.3000 |
| vidoseek | graph_rescue_coe | 32000 | 0.9726 | 0.9921 | 0.9956 | — | 31930.3000 |
| wiki2multihop | graph_rescue_packed | 2000 | 0.6705 | 0.6451 | 0.6821 | 0.6838 | 1992.5000 |
| wiki2multihop | graph_rescue_packed | 4000 | 0.7092 | 0.6700 | 0.7195 | 0.7199 | 3992.0000 |
| wiki2multihop | graph_rescue_packed | 8000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 7896.3000 |
| wiki2multihop | graph_rescue_packed | 10000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 9488.2000 |
| wiki2multihop | graph_rescue_packed | 16000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 11206.5000 |
| wiki2multihop | graph_rescue_packed | 24000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 11273.9000 |
| wiki2multihop | graph_rescue_packed | 32000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 11273.9000 |
| wiki2multihop | graph_rescue_coe | 2000 | 0.7090 | 0.6701 | 0.7190 | 0.7196 | 1994.1000 |
| wiki2multihop | graph_rescue_coe | 4000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 3993.2000 |
| wiki2multihop | graph_rescue_coe | 8000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 7897.1000 |
| wiki2multihop | graph_rescue_coe | 10000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 9488.6000 |
| wiki2multihop | graph_rescue_coe | 16000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 11206.5000 |
| wiki2multihop | graph_rescue_coe | 24000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 11273.9000 |
| wiki2multihop | graph_rescue_coe | 32000 | 0.7096 | 0.6702 | 0.7198 | 0.7202 | 11273.9000 |

## Chain coverage vs recall — where the two disagree

`R@10` counts gold CHUNKS retrieved; `chain cov` counts gold DOCUMENTS covered. They can move in opposite directions, and on the graph legs they do. Where they diverge, the equal-weight recall drop is **not** evidence that the signal is worthless: the signal is surfacing documents the dense+bm25 pool missed, while the equal-weight fusion pays for them with top-10 slots. Whether that trade is worth making is exactly the question the dev-tuned variant below is there to answer.

| dataset | step | Δ R@10 | Δ chain cov@10 | reading |
|---|---|---|---|---|
| hotpotqa | dense → bm25 | +0.0210 | +0.0655 | same direction |
| hotpotqa | bm25 → kg | +0.0000 | +0.0000 | same direction |
| hotpotqa | kg → hyperedges | -0.0076 | -0.0081 | same direction |
| hotpotqa | hyperedges → coe | -0.0745 | -0.0736 | same direction |
| hotpotqa | coe → packer | -0.0862 | -0.0962 | same direction |
| multihop_rag | dense → bm25 | +0.0444 | +0.0991 | same direction |
| multihop_rag | bm25 → kg | -0.0505 | -0.0733 | same direction |
| multihop_rag | kg → hyperedges | -0.0388 | +0.0191 | **diverges — the signal reaches more gold documents while the fusion loses chunks** |
| multihop_rag | hyperedges → coe | -0.0356 | -0.0244 | same direction |
| multihop_rag | coe → packer | -0.0495 | -0.0262 | same direction |
| musique | dense → bm25 | +0.0051 | +0.0427 | same direction |
| musique | bm25 → kg | +0.0000 | +0.0000 | same direction |
| musique | kg → hyperedges | -0.0012 | -0.0071 | same direction |
| musique | hyperedges → coe | -0.0545 | -0.0808 | same direction |
| musique | coe → packer | -0.0347 | -0.0451 | same direction |
| wiki2multihop | dense → bm25 | +0.0265 | +0.0262 | same direction |
| wiki2multihop | bm25 → kg | +0.0019 | +0.0020 | same direction |
| wiki2multihop | kg → hyperedges | -0.0172 | -0.0176 | same direction |
| wiki2multihop | hyperedges → coe | -0.0916 | -0.0910 | same direction |
| wiki2multihop | coe → packer | -0.0503 | -0.0512 | same direction |

## Dev-tuned fusion weights (second variant — NOT the baseline)

The ladder above fuses every leg's signals at **equal** RRF weight. This section fits a per-signal weight vector per (dataset, leg) and reports it on held-out queries:

* **Split.** Each dataset's usable queries are partitioned by a stable keyed hash of the query id (`blake2b(salt:query_id) % 1000 < dev_permille`) — no RNG, no ordering dependence, and independent of `--limit`, so a query is on the same side in every run. Sizes are stated per row.
* **Fit.** Coordinate ascent over a fixed weight grid, from all-ones, maximising mean **nDCG@10** (the same definition this report prints) on the DEV half only. The first signal is pinned at 1.0 because RRF weights are scale-invariant. A value is adopted only on a strict improvement, so the fit can never land below the equal-weight start.
* **Report.** Metrics are computed on the TEST half only, and the equal-weight column is recomputed on those same test queries from the same retrieved lists — so `Δ` is a difference in **fusion** and nothing else.

**These tuned numbers are not comparable to published baselines.** Those baselines report an untuned system; these were fitted on this corpus's own dev queries. The comparable number remains the equal-weight ladder above. What this table answers is narrower and internal: *given the same candidate lists, does weighting the graph signals properly recover what equal-weight fusion loses?*

| dataset | leg | dev n | test n | R@10 equal | R@10 tuned | Δ R@10 | nDCG@10 equal | nDCG@10 tuned | Δ nDCG@10 | chain cov equal | chain cov tuned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | dense | 300 | 694 | 0.5535 | 0.5535 | +0.0000 | 0.5934 | 0.5934 | +0.0000 | 0.7334 | 0.7334 |
| hotpotqa | bm25 | 300 | 694 | 0.5795 | 0.5803 | +0.0008 | 0.6098 | 0.6132 | +0.0034 | 0.8084 | 0.7882 |
| hotpotqa | kg | 300 | 694 | 0.5795 | 0.5803 | +0.0008 | 0.6098 | 0.6132 | +0.0034 | 0.8084 | 0.7882 |
| hotpotqa | hyperedges | 300 | 694 | 0.5672 | 0.5803 | +0.0131 | 0.5651 | 0.6132 | +0.0481 | 0.7882 | 0.7882 |
| hotpotqa | coe | 300 | 694 | 0.4914 | 0.5803 | +0.0889 | 0.4697 | 0.6132 | +0.1435 | 0.7075 | 0.7882 |
| hotpotqa | packer | 300 | 694 | 0.4015 | 0.5522 | +0.1507 | 0.4179 | 0.5932 | +0.1753 | 0.6009 | 0.7514 |
| hotpotqa | dense_content_only | — | — | *missing-inputs* | | | | | | | | |
| multihop_rag | dense | 300 | 1570 | 0.3307 | 0.3307 | +0.0000 | 0.4521 | 0.4521 | +0.0000 | 0.6264 | 0.6264 |
| multihop_rag | bm25 | 300 | 1570 | 0.3769 | 0.3810 | +0.0041 | 0.5175 | 0.5234 | +0.0059 | 0.7213 | 0.7475 |
| multihop_rag | kg | 300 | 1570 | 0.3257 | 0.3810 | +0.0553 | 0.4299 | 0.5234 | +0.0935 | 0.6454 | 0.7475 |
| multihop_rag | hyperedges | 300 | 1570 | 0.2864 | 0.3801 | +0.0937 | 0.3861 | 0.5116 | +0.1255 | 0.6647 | 0.7608 |
| multihop_rag | coe | 300 | 1570 | 0.2503 | 0.3809 | +0.1306 | 0.3465 | 0.5198 | +0.1733 | 0.6416 | 0.7532 |
| multihop_rag | packer | 300 | 1570 | 0.2036 | 0.3134 | +0.1098 | 0.3020 | 0.4571 | +0.1551 | 0.6220 | 0.7799 |
| multihop_rag | dense_content_only | 300 | 1570 | 0.3325 | 0.3325 | +0.0000 | 0.4569 | 0.4569 | +0.0000 | 0.6351 | 0.6351 |
| musique | dense | 300 | 866 | 0.3288 | 0.3288 | +0.0000 | 0.3929 | 0.3929 | +0.0000 | 0.5376 | 0.5376 |
| musique | bm25 | 300 | 866 | 0.3332 | 0.3476 | +0.0144 | 0.4082 | 0.4175 | +0.0093 | 0.5843 | 0.5885 |
| musique | kg | 300 | 866 | 0.3332 | 0.3476 | +0.0144 | 0.4082 | 0.4175 | +0.0093 | 0.5843 | 0.5885 |
| musique | hyperedges | 300 | 866 | 0.3352 | 0.3517 | +0.0165 | 0.3962 | 0.4209 | +0.0247 | 0.5817 | 0.5933 |
| musique | coe | 300 | 866 | 0.2793 | 0.3507 | +0.0714 | 0.3179 | 0.4187 | +0.1008 | 0.4984 | 0.5887 |
| musique | packer | 300 | 866 | 0.2455 | 0.3366 | +0.0911 | 0.2970 | 0.4086 | +0.1116 | 0.4542 | 0.5764 |
| musique | dense_content_only | — | — | *missing-inputs* | | | | | | | | |
| ohr_bench | dense | 300 | 3171 | 0.5864 | 0.5864 | +0.0000 | 0.5269 | 0.5269 | +0.0000 | — | — |
| ohr_bench | bm25 | 300 | 3171 | 0.6598 | 0.6676 | +0.0078 | 0.6118 | 0.6247 | +0.0129 | — | — |
| ohr_bench | kg | 300 | 3171 | 0.6111 | 0.6676 | +0.0565 | 0.5134 | 0.6247 | +0.1113 | — | — |
| ohr_bench | hyperedges | 300 | 3171 | 0.5216 | 0.6676 | +0.1460 | 0.4149 | 0.6247 | +0.2098 | — | — |
| ohr_bench | coe | 300 | 3171 | 0.3602 | 0.6676 | +0.3074 | 0.3030 | 0.6247 | +0.3217 | — | — |
| ohr_bench | packer | 300 | 3171 | 0.2769 | 0.5320 | +0.2551 | 0.2607 | 0.5366 | +0.2759 | — | — |
| ohr_bench | dense_content_only | 300 | 3171 | 0.5914 | 0.5914 | +0.0000 | 0.5282 | 0.5282 | +0.0000 | — | — |
| uda | dense | 300 | 691 | 0.0648 | 0.0648 | +0.0000 | 0.0575 | 0.0575 | +0.0000 | — | — |
| uda | bm25 | 300 | 691 | 0.1343 | 0.1582 | +0.0239 | 0.1117 | 0.1321 | +0.0204 | — | — |
| uda | kg | 300 | 691 | 0.1318 | 0.1597 | +0.0279 | 0.1104 | 0.1345 | +0.0241 | — | — |
| uda | hyperedges | 300 | 691 | 0.1103 | 0.1567 | +0.0464 | 0.0904 | 0.1294 | +0.0390 | — | — |
| uda | coe | 300 | 691 | 0.0656 | 0.1597 | +0.0941 | 0.0527 | 0.1345 | +0.0818 | — | — |
| uda | packer | 300 | 691 | 0.0353 | 0.1068 | +0.0715 | 0.0349 | 0.1067 | +0.0718 | — | — |
| uda | dense_content_only | 300 | 691 | 0.0672 | 0.0672 | +0.0000 | 0.0558 | 0.0558 | +0.0000 | — | — |
| vidoseek | dense | 300 | 748 | 0.9395 | 0.9395 | +0.0000 | 0.7915 | 0.7915 | +0.0000 | — | — |
| vidoseek | bm25 | 300 | 748 | 0.9670 | 0.9643 | -0.0027 | 0.8488 | 0.8567 | +0.0079 | — | — |
| vidoseek | kg | 300 | 748 | 0.9588 | 0.9643 | +0.0055 | 0.8059 | 0.8567 | +0.0508 | — | — |
| vidoseek | hyperedges | 300 | 748 | 0.9322 | 0.9643 | +0.0321 | 0.7384 | 0.8567 | +0.1183 | — | — |
| vidoseek | coe | 300 | 748 | 0.8694 | 0.9643 | +0.0949 | 0.6546 | 0.8567 | +0.2021 | — | — |
| vidoseek | packer | 300 | 748 | 0.7503 | 0.8425 | +0.0922 | 0.6096 | 0.7804 | +0.1708 | — | — |
| vidoseek | dense_content_only | 300 | 748 | 0.9389 | 0.9389 | +0.0000 | 0.7953 | 0.7953 | +0.0000 | — | — |
| wiki2multihop | dense | 300 | 697 | 0.6804 | 0.6804 | +0.0000 | 0.6914 | 0.6914 | +0.0000 | 0.6903 | 0.6903 |
| wiki2multihop | bm25 | 300 | 697 | 0.7042 | 0.7049 | +0.0007 | 0.7201 | 0.7205 | +0.0004 | 0.7138 | 0.7138 |
| wiki2multihop | kg | 300 | 697 | 0.6996 | 0.7064 | +0.0068 | 0.7110 | 0.7201 | +0.0091 | 0.7075 | 0.7141 |
| wiki2multihop | hyperedges | 300 | 697 | 0.6901 | 0.7064 | +0.0163 | 0.6697 | 0.7201 | +0.0504 | 0.6977 | 0.7141 |
| wiki2multihop | coe | 300 | 697 | 0.5959 | 0.7064 | +0.1105 | 0.5645 | 0.7201 | +0.1556 | 0.6061 | 0.7141 |
| wiki2multihop | packer | 300 | 697 | 0.5496 | 0.6693 | +0.1197 | 0.5427 | 0.6980 | +0.1553 | 0.5575 | 0.6772 |
| wiki2multihop | dense_content_only | 300 | 697 | 0.6807 | 0.6807 | +0.0000 | 0.7056 | 0.7056 | +0.0000 | 0.6914 | 0.6914 |

### Fitted weights (auditable — every one of them)

A weight of `0.00` is the fit saying *drop this list*, which is a finding, not a failure — it is an answer the equal-weight run cannot express. `dev nDCG@10` shows what the fit bought **on the fitting set**; the honest number is the Δ in the table above, on the test set.

| dataset | leg | fitted weights | grid evals | search | dev nDCG@10 equal → tuned |
|---|---|---|---|---|---|
| hotpotqa | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.5778 → 0.5778 |
| hotpotqa | bm25 | dense 1.00, bm25 0.50 | 14 | converged | 0.5880 → 0.5964 |
| hotpotqa | kg | dense 1.00, bm25 0.50, kg 1.00 | 26 | converged | 0.5880 → 0.5964 |
| hotpotqa | hyperedges | dense 1.00, bm25 0.50, kg 1.00, hyperedges 0.00 | 57 | converged | 0.5484 → 0.5964 |
| hotpotqa | coe | dense 1.00, bm25 0.50, kg 1.00, hyperedges 0.00, coe 0.00 | 76 | converged | 0.4549 → 0.5964 |
| hotpotqa | packer | dense 1.00, bm25 0.50, kg 1.00, hyperedges 0.00, coe 0.00 | 76 | converged | 0.4549 → 0.5964 |
| multihop_rag | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.4782 → 0.4782 |
| multihop_rag | bm25 | dense 1.00, bm25 2.00 | 13 | converged | 0.5456 → 0.5533 |
| multihop_rag | kg | dense 1.00, bm25 2.00, kg 0.00 | 26 | converged | 0.4586 → 0.5533 |
| multihop_rag | hyperedges | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.50 | 39 | converged | 0.4310 → 0.5575 |
| multihop_rag | coe | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00, coe 0.25 | 52 | converged | 0.3994 → 0.5553 |
| multihop_rag | packer | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00, coe 0.25 | 52 | converged | 0.3994 → 0.5553 |
| multihop_rag | dense_content_only | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.4840 → 0.4840 |
| musique | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.4321 → 0.4321 |
| musique | bm25 | dense 1.00, bm25 0.50 | 14 | converged | 0.4525 → 0.4599 |
| musique | kg | dense 1.00, bm25 0.50, kg 1.00 | 26 | converged | 0.4525 → 0.4599 |
| musique | hyperedges | dense 1.00, bm25 0.50, kg 1.00, hyperedges 0.25 | 57 | converged | 0.4310 → 0.4611 |
| musique | coe | dense 1.00, bm25 0.50, kg 1.00, hyperedges 0.00, coe 0.25 | 76 | converged | 0.3540 → 0.4600 |
| musique | packer | dense 1.00, bm25 0.50, kg 1.00, hyperedges 0.00, coe 0.25 | 76 | converged | 0.3540 → 0.4600 |
| ohr_bench | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.3755 → 0.3755 |
| ohr_bench | bm25 | dense 1.00, bm25 2.00 | 13 | converged | 0.4862 → 0.5117 |
| ohr_bench | kg | dense 1.00, bm25 2.00, kg 0.00 | 26 | converged | 0.3980 → 0.5117 |
| ohr_bench | hyperedges | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00 | 39 | converged | 0.3247 → 0.5117 |
| ohr_bench | coe | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00, coe 0.00 | 52 | converged | 0.2411 → 0.5117 |
| ohr_bench | packer | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00, coe 0.00 | 52 | converged | 0.2411 → 0.5117 |
| ohr_bench | dense_content_only | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.3793 → 0.3793 |
| uda | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.0563 → 0.0563 |
| uda | bm25 | dense 1.00, bm25 2.00 | 13 | converged | 0.1160 → 0.1373 |
| uda | kg | dense 1.00, bm25 2.00, kg 0.25 | 26 | converged | 0.1097 → 0.1389 |
| uda | hyperedges | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.75 | 39 | converged | 0.0805 → 0.1381 |
| uda | coe | dense 1.00, bm25 2.00, kg 0.25, hyperedges 0.00, coe 0.00 | 76 | converged | 0.0431 → 0.1389 |
| uda | packer | dense 1.00, bm25 2.00, kg 0.25, hyperedges 0.00, coe 0.00 | 76 | converged | 0.0431 → 0.1389 |
| uda | dense_content_only | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.0593 → 0.0593 |
| vidoseek | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.8153 → 0.8153 |
| vidoseek | bm25 | dense 1.00, bm25 2.00 | 13 | converged | 0.8731 → 0.8827 |
| vidoseek | kg | dense 1.00, bm25 2.00, kg 0.00 | 26 | converged | 0.8268 → 0.8827 |
| vidoseek | hyperedges | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00 | 39 | converged | 0.7398 → 0.8827 |
| vidoseek | coe | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00, coe 0.00 | 52 | converged | 0.6771 → 0.8827 |
| vidoseek | packer | dense 1.00, bm25 2.00, kg 0.00, hyperedges 0.00, coe 0.00 | 52 | converged | 0.6771 → 0.8827 |
| vidoseek | dense_content_only | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.8122 → 0.8122 |
| wiki2multihop | dense | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.6796 → 0.6796 |
| wiki2multihop | bm25 | dense 1.00, bm25 0.50 | 14 | converged | 0.7062 → 0.7127 |
| wiki2multihop | kg | dense 1.00, bm25 0.75, kg 0.50 | 27 | converged | 0.7075 → 0.7132 |
| wiki2multihop | hyperedges | dense 1.00, bm25 0.75, kg 0.50, hyperedges 0.00 | 58 | converged | 0.6523 → 0.7132 |
| wiki2multihop | coe | dense 1.00, bm25 0.75, kg 0.50, hyperedges 0.00, coe 0.00 | 103 | converged | 0.5579 → 0.7132 |
| wiki2multihop | packer | dense 1.00, bm25 0.75, kg 0.50, hyperedges 0.00, coe 0.00 | 103 | converged | 0.5579 → 0.7132 |
| wiki2multihop | dense_content_only | — (single signal: the RRF weight is a pure scale, so there is nothing to fit) | 1 | converged | 0.6909 → 0.6909 |

## Control check — where the hypergraph/CoE legs should buy nothing

ViDoSeek, OHR-Bench and UDA are single-hop: their gold is one page or one sentence, with no chain to assemble. A `+hyperedges` or `+coe` gain there is not a win — it indicates the leg is reaching gold through a path the question did not require (leakage), so it is reported as a warning, not a headline.

| control dataset | Δ R@10 from +hyperedges | Δ R@10 from +coe | reading |
|---|---|---|---|
| ohr_bench | -0.0864 | -0.1608 | as expected (no gain on single-hop) |
| uda | -0.0225 | -0.0423 | as expected (no gain on single-hop) |
| vidoseek | -0.0337 | -0.0637 | as expected (no gain on single-hop) |

## Scope

SPLADE, MaxSim/MUVERA and rerank/fusion ablations are OUT of this run (S5 ticket). Hop depth is 2–4 native only; 5–7-hop chains are untested and no claim is made about them.
