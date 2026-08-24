# S5b — the graph lanes do not earn their latency

**A negative result, and a positive one nobody was looking for.**

Design and stop rule: [`EXPERIMENT.md`](EXPERIMENT.md). Full arm tables:
[`results/ARMS.md`](results/ARMS.md). Selector cells:
[`results/select_pool__multihop_rag__n500.json`](results/select_pool__multihop_rag__n500.json).
Reproduce: `probe.py` then `analyse.py` (traces are regenerable and not committed — ~50 KB/query).

---

## 1. The verdict

`RERUN-REPORT.md` §2 established that the traversal and hypergraph lanes are *architecturally
live and measurably inert at k=10*, and named three configurations that might un-inert them:
a larger `select_pool_k`, a larger `k`, a smaller `pool_k`. All three have now been run, plus
the interleave family the brief raised as a fourth.

**Every one of them is refuted, and none of them was ever the binding constraint.**

| | question | answer |
|---|---|---|
| **Q1** | Do the graph lanes reach gold the base does not already hold? | **Barely.** Oracle ceiling **+0.0056** recall@10 (multihop_rag), **+0.0077** (wiki2multihop — the best case), **+0.0000** (vidoseek). |
| **Q2** | Does any promotion policy realise it? | No policy beats `base` with a CI excluding zero. The best is indistinguishable from base; the rest are worse. |
| **Q3** | What else could that retrieval budget buy? | A perfect reranker over the base alone: **+0.3724** recall@10 on multihop_rag, **+0.0905** on wiki2multihop — **66×** and **12×** the graph's entire ceiling, same queries, same units. |

The graph lanes cost **221–538 ms/query** median against the `bm25` rung's **1.1–8.3 ms** on the
same three datasets (committed cells; the traces' own timings were taken under concurrent load and
are not a latency measurement). For that, at the ceiling, they buy **+0.0000 to +0.0077** recall@10.
**They do not earn their latency.**

### Validity check, before any of the numbers are quoted

The `base` arm reproduces the committed `graph_rescue` cell to four decimals, on the full query
population, on both datasets:

| dataset | committed cell r@10 / nDCG@10 | `base` arm r@10 / nDCG@10 | queries |
|---|---|---|---|
| multihop_rag | 0.3759 / 0.5185 | 0.3759 / 0.5185 | 2,255 / 2,255 |
| vidoseek | 0.9726 / 0.8138 | 0.9726 / 0.8138 | 1,142 / 1,142 |

So the traces and the campaign matrix describe the same run, and every arm below is directly
comparable to the S5 matrix rather than to a re-measurement of it.

---

## 2. Q1 — the ceiling, which is what actually ends this

The oracle promotes every rescued gold chunk to the front of the list. It is not shippable — it
reads the labels — and that is the point: **no promotion policy can beat it.**

| dataset | queries | gold/query | in base | **rescue-only** | queries with any rescued gold | ceiling Δr@10 |
|---|---|---|---|---|---|---|
| multihop_rag | 2,255 | 13.86 | 12.02 | **0.142** | 221 (9.8%) | **+0.0056** [+0.0047, +0.0065] |
| wiki2multihop | 2,500 | 2.52 | 1.93 | **0.021** | 50 (2.0%) | **+0.0077** [+0.0055, +0.0100] |
| vidoseek | 1,142 | 1.54 | 1.52 | **0.000** | **0 (0.0%)** | **+0.0000** [+0.0000, +0.0000] |

On vidoseek the traversal and hypergraph lanes reach **not one gold chunk** the base missed,
across 1,142 queries. On multihop_rag they reach 0.14 per query — and 160 of those 191 come from
the hypergraph lane, only 31 from traversal.

Lifting `rescue_cap` from 50 to unbounded raises the multihop_rag ceiling to +0.0201 and
vidoseek's to +0.0002. So the cap is not hiding anything either.

**This is the finding.** Everything below is the confirmation that no knob converts a ceiling
this small into a win — which was worth measuring, because "we never turned the knob" was a live
explanation until now.

---

## 3. Q2 — the four options, each refuted by its own measurement

### 3.1 Raise `select_pool_k` above |base| — *the knob works, and costs the packer recall*

446 multihop_rag queries, both budget selectors, one retrieval per query through the stateless
`select()` seam:

| | rescued candidates selected | recall@10 | nDCG@10 | **recall@selected** |
|---|---|---|---|---|
| `packed` @ 100 (shipped) | **0** | 0.3769 | 0.5307 | 0.5555 |
| `packed` @ 250 | 59 (54 queries) | 0.3769 | 0.5307 | **0.5523** |
| `packed` @ 1000 | 59 (54 queries) | 0.3769 | 0.5307 | **0.5523** |
| `coe` @ 100 (shipped) | **0** | 0.3769 | 0.5307 | 0.6592 |
| `coe` @ 250 / 1000 | 23 / 23 | 0.3769 | 0.5307 | 0.6592 |

The `0` at the shipped pool is the mechanism, measured rather than inferred. Raising the pool
does exactly what it was supposed to — the selectors start seeing and picking rescued candidates
— and 250 ≡ 1000, so the knob is *saturated*, not under-turned.

`recall@selected` is scored here because it is the cut that matches what a selector is *for*: its
output is the context handed to a model (~23 packed, ~35 coe candidates), and judging a budgeted
subset at rank 10 asks it to win a ranking contest it is not running. At that fair cut the packer
**loses 0.0032** — it spends token budget on rescued candidates and gets *less* gold into the
context. CoE gains nothing.

### 3.2 Measure at larger k — *already covered, and it changes nothing*

Every arm is reported at `recall@{1,5,10,20,50,100}`. `append` is identical to `base` at **every
cut including k=100**, on both datasets. The inertia is not a k=10 artifact: with `|base|` at
157–178 and the appendix starting at rank 110–142, a top-100 cut cannot see it either.

### 3.3 Lower `pool_k` — *makes graph candidates visible, and makes the system worse*

There is a structural bound worth stating, because it predicts the whole table: under
append-not-fuse a rescued candidate reaches a top-`k` cut **only if `|base| < k`**, and
`|base| ≥ pool_k`. So a top-10 appearance requires `pool_k < 10` — a base with the pool destroyed.

800-query paired sample, multihop_rag:

| `pool_k` | mean \|base\| | first rescued rank | `base` r@10 | `append` r@10 | graph ceiling | rerank ceiling |
|---|---|---|---|---|---|---|
| **100** (shipped) | 157.8 | 113 | 0.3747 | 0.3747 (+0.0000) | +0.0056 | **+0.3658** |
| 20 | 31.0 | 22 | 0.3703 | 0.3703 (+0.0000) | +0.0415 | +0.2020 |
| 8 | 12.8 | 9 | 0.3620 | **0.3623 (+0.0003)** | +0.0843 | +0.0519 |

`pool_k=8` is the first configuration where a rescued candidate reaches the top ten at all, and it
buys **+0.0003 recall@10** against a base that shrinking has already cost 0.0127 — **net −0.0124**
against the shipped system.

The graph's ceiling does rise as the pool shrinks (+0.0056 → +0.0843). That is not the graph
getting better; it is the graph being paid to partly repair damage the smaller pool inflicted. The
reranker ceiling moves the opposite way (+0.3658 → +0.0519) because there is nothing left to
rerank. **The two ceilings are substitutes, and at the shipped pool the reranker's is 66× larger.**

### 3.4 Interleave rather than append — *the rationale is confirmed, not discarded*

The brief asked that append-not-fuse not be discarded without arguing against its rationale
("graph signals must not displace strong lexical/dense hits"). It is not discarded — it is
**tested**, as a continuum, and it wins.

multihop_rag, 2,255 queries, Δrecall@10 vs `base` with 95% bootstrap CI:

| arm | r@10 | nDCG@10 | Δr@10 (95% CI) |
|---|---|---|---|
| `base` | 0.3759 | 0.5185 | — |
| `append` (shipped) | 0.3759 | 0.5185 | +0.0000 [+0.0000, +0.0000] |
| `fuse w=0.05` | 0.3764 | 0.5168 | +0.0005 [−0.0010, **+0.0020**] |
| `fuse w=0.10` | 0.3744 | 0.5117 | −0.0015 [−0.0036, +0.0007] |
| `fuse w=0.25` | 0.3572 | 0.4819 | −0.0187 [−0.0223, −0.0148] |
| `fuse w=0.50` | 0.3198 | 0.4274 | −0.0561 [−0.0617, −0.0508] |
| `fuse w=1.00` | 0.2638 | 0.3506 | −0.1121 [−0.1188, −0.1055] |
| `protect@5` | 0.3296 | 0.4745 | −0.0463 [−0.0488, −0.0438] |
| `protect@10` | 0.3759 | 0.5185 | +0.0000 (and −0.067 at r@20) |

Monotonically worse in `w` on both datasets, and `w=1.00` reproduces independently what the
ladder's `kg` rung already showed. The single arm whose point estimate is positive (`w=0.05`,
+0.0005) has a CI containing zero and costs nDCG@10. `protect@10` is neutral at k=10 *by
construction* and pays for it below the cut.

**The append-not-fuse design choice is right.** Fusing the graph lane into the base actively
destroys the ranking, and there is no weight small enough to help without being small enough to
do nothing. That is a finding worth keeping even though the lane it protects is worthless.

---

## 4. Q3 — what the retrieval budget should buy instead

The same oracle machinery, pointed at the base:

| dataset | `base` r@10 | `base` r@100 | perfect rerank of the base, r@10 | Δ |
|---|---|---|---|---|
| multihop_rag | 0.3759 | 0.8737 | **0.7483** | **+0.3724** [+0.3630, +0.3812] |
| wiki2multihop | 0.7091 | 0.7894 | **0.7996** | **+0.0905** [+0.0833, +0.0976] |
| vidoseek | 0.9726 | 0.9960 | 0.9969 | +0.0244 [+0.0173, +0.0319] |

On the multi-hop dataset the base already **retrieves** 87% of gold in its top 100 and **ranks**
38% of it into the top 10. The gap is a *ranking* problem, not a *reach* problem, and the graph
lanes are an expensive answer to reach.

vidoseek's small headroom is the honest counterweight: on a single-hop page corpus the base is
near-saturated (r@10 0.9726) and there is little for anything to buy — neither graph nor
reranker. It is the control, and it behaves like one.

---

## 5. What this does to the retrieval story

The brief proposed rewriting around "dense+BM25 RRF plus the Expander rescore". The first half
survives. **The second half does not, and should not be adopted without re-arguing it.**

`graph_rescue` − `bm25` is exactly the Expander (max-normalisation and the rank rescore are both
monotone; `docs/anatomy/05-retrieval.md` §5.4 derives this, and musique/hotpotqa — where no KG
columns exist — pin it at 0.0000 as a control). The committed matrix therefore already measures
the Expander in isolation, on full query populations:

| dataset | Δ recall@10 | Δ nDCG@10 |
|---|---|---|
| wiki2multihop | +0.0078 | **−0.0166** |
| vidoseek | +0.0021 | **−0.0412** |
| uda | +0.0015 | +0.0006 |
| multihop_rag | −0.0013 | −0.0050 |
| ohr_bench | −0.0020 | **−0.0288** |

Recall-neutral, and **nDCG-negative on four of the five datasets where it acts**. The Expander
trades head precision for a third-decimal recall gain. It is not the thing to build the story on.

**What the evidence supports:**

> Dense+BM25 RRF is the retrieval story. It reaches 87% of gold in the top 100 on multi-hop
> corpora and ranks 38% of it into the top 10 — so the next unit of quality is a **rescore over
> the base's own top-100**, worth up to +0.37 recall@10, not more reach. Applied as rescue, the
> graph never lowers recall (which fusing it demonstrably does, by up to −0.11), and that
> containment is a real property of the append design. But on these corpora the traversal and
> hypergraph lanes reach almost nothing the base missed — zero gold chunks in 1,142 vidoseek
> queries, 0.14/query on multihop_rag — and no configuration of `select_pool_k`, `k`, `pool_k`
> or fusion weight converts that into a measurable win. At 484–538 ms/query against BM25's 1.3 ms,
> they do not earn their latency.

---

## 6. What was not run, and why

* **`uda`, `ohr_bench`, `hotpotqa`, `musique`** — not run. The stop rule (`EXPERIMENT.md` §6) was
  set before the numbers: ceiling ≤ +0.01 recall@10 on both datasets → negative result, no
  escalation. It fired. Separately, **`musique` and `hotpotqa` cannot be re-run in this checkout
  at all** — their exports carry no `bm25-stats.json` / `bm25-postings.parquet`, so the committed
  cells for them describe artifacts no longer on disk.
* **`wiki2multihop`** — run as a robustness check rather than under the stop rule, because it is
  the dataset where `graph_rescue` looked *best* in the committed matrix (+0.0078 vs `bm25`, the
  only clearly positive cell) and therefore the strongest remaining case for the graph. §7 records
  it, and it agrees.
* **A tuning sweep over the expander/gate/seed knobs** — deliberately not run. A sweep can only
  find which knob converts a gain into a visible gain; it cannot manufacture a gain. With the
  ceiling at +0.0056 it would be arithmetic on zero.

---

## 7. The strongest remaining case, checked

`wiki2multihop` is the one cell in the committed matrix where `graph_rescue` clearly beat `bm25`
(+0.0078 recall@10, and `full_chain_hit@10` 0.4222 against 0.4076). If the graph lanes were going
to earn anything anywhere, it would be here. A 2,500-query fixed-seed sample (the committed cell
is the full 12,576, which is why `base` reads 0.7091 against the cell's 0.7096 — sampling, not
disagreement):

| arm | r@1 | r@10 | r@100 | nDCG@10 | Δr@10 (95% CI) |
|---|---|---|---|---|---|
| `base` | 0.3585 | 0.7091 | 0.7894 | 0.7000 | — |
| `append` (shipped) | 0.3585 | 0.7091 | 0.7894 | 0.7000 | +0.0000 [+0.0000, +0.0000] |
| `fuse w=0.05` | 0.3350 | 0.7100 | 0.7895 | 0.6863 | +0.0009 [−0.0005, +0.0024] |
| `fuse w=0.10` | 0.3068 | 0.7104 | 0.7896 | 0.6642 | +0.0013 [−0.0007, +0.0034] |
| `fuse w=1.00` | 0.2216 | 0.6568 | 0.7815 | 0.5535 | −0.0523 [−0.0594, −0.0451] |
| **ORACLE graph (cap 50)** | 0.3602 | 0.7168 | 0.7971 | 0.7068 | **+0.0077** [+0.0055, +0.0100] |
| ORACLE graph (uncapped) | 0.3618 | 0.7266 | 0.8070 | 0.7153 | +0.0175 [+0.0142, +0.0208] |
| **ORACLE rerank base only** | 0.4365 | **0.7996** | 0.7996 | 0.8498 | **+0.0905** [+0.0833, +0.0976] |

The same three facts, on the friendliest dataset available:

* **The ceiling is +0.0077** — still inside the stop rule's ≤ +0.01, with the CI's upper bound
  landing exactly on it. The graph lanes reach 0.021 rescue-only gold chunks per query (36 of the
  50 from the hypergraph lane, 5 from traversal).
* **`append` is identical to `base` at every cut**, k=100 included, for the third dataset running.
* **The reranker ceiling is 12× the graph's** (+0.0905 vs +0.0077) even here, where the base has
  much less headroom than on multihop_rag (r@10 0.7091 against r@100 0.7894).

The two `fuse` arms with positive point estimates (`w=0.05`, `w=0.10`) are the closest anything
comes to a win in this study — and both have CIs containing zero, both cost nDCG@10 (0.7000 →
0.6863 → 0.6642), and both collapse `recall@1` (0.3585 → 0.3350 → 0.3068). Buying a third-decimal
recall@10 that may be zero, by measurably wrecking the head of the ranking, is not a trade to ship.

**Three datasets, one answer.** The negative result stands on the dataset chosen to break it.
