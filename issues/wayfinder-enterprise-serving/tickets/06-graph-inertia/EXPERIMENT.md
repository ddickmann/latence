# S5b — does graph traversal help, or is it only inert?

**Status:** designed, running. **Owner:** ticket 06. **Predecessor:** `RERUN-REPORT.md` §2
("Read against the right baseline, though"), `docs/anatomy/05-retrieval.md` §5.4.

## 1. What is actually being asked

The S5 matrix reports `graph_rescue` as bitwise identical to `graph_rescue_coe` on 7/7 datasets
and to the `bm25` rung on 2/7, and within ±0.008 recall@10 (negative nDCG@10 on 4/7) everywhere
else. The mechanism is structural, not a tuning accident:

| step | value | consequence |
|---|---|---|
| `POOL_K` (`benchmark/s5/legs.py:286`), pipeline `pool_k` | 100 | dense and BM25 each return 100 |
| `_fuse_base` | RRF of both | the base is 100–200 candidates |
| `_append_and_dedupe` (`rescue.py:1071`) | base first, always | first `rescued` item sits at rank ≥ \|base\| |
| `select()` (`rescue.py:878`) | `union[:select_pool_k]`, `select_pool_k=100` | the selector's pool is all-base |

So no rescued candidate can reach a top-10 cut *or* the selector's pool. The lanes are
architecturally live and measurably inert. **What has never been measured is whether that
inertia is hiding a win or hiding nothing.**

## 2. The question, decomposed

The trap is going straight to a tuning sweep. A sweep can only ever tell you which knob converts
a gain into a visible gain — it cannot tell you whether a gain exists. So the experiment asks two
questions in order, and the first one is allowed to end it:

**Q1 — the ceiling.** Do the graph lanes reach gold chunks the base does not already hold?
This is an *oracle* quantity: promote every rescued gold chunk to the front and measure. No
promotion policy, however clever, can beat it. If the ceiling is ~0, every option in §3 is
arithmetic on zero and the answer is a negative result.

**Q2 — the conversion.** If the ceiling is non-trivial, which policy realises it, and does the
gain survive the latency it costs (114–646 ms/query median against BM25's 1–33 ms)?

**Q3 — the counterfactual.** What else could that retrieval budget buy? The base's own
`recall@100` vs `recall@10` bounds a rescore stage the same way, on the same queries, in the same
units — so "the graph is worth it" is a comparison rather than an assertion.

## 3. Arms

Every arm is a **re-ranking of candidates that were already retrieved**, so one expensive pass
(`probe.py`) serves all of them and a difference between two arms is the promotion policy and
nothing else. Scored at `recall@{1,5,10,20,50,100}`, `nDCG@{10,50}`, paired per query, with a
2,000-round percentile bootstrap on the `recall@10` delta.

| arm | what it tests | option from the brief |
|---|---|---|
| `base` | control: dense+BM25 RRF, max-normalised, Expander-rescored | — |
| `append (shipped)` | the status quo | — |
| `protect@5`, `protect@10` | interleave *below* a protected head | "interleave rather than append" |
| `fuse w ∈ {0.05 … 1.00}` | RRF the lane INTO the base at weight `w` | the append-vs-fuse continuum |
| `ORACLE graph (cap 50)` | the ceiling at the shipped `rescue_cap` | bounds every policy |
| `ORACLE graph (uncapped)` | the ceiling with the cap lifted | bounds `rescue_cap` too |
| `ORACLE rerank base only` | a perfect reranker over the base, no graph at all | the counterfactual (Q3) |
| `pool_k ∈ {100, 20}` (second pass) | a base that does not saturate the cut | "lower `POOL_K`" |

Reported at `k` up to 100 throughout, which covers "measure at larger k" — and note that
`select_pool_k > \|base\|` is only meaningful for the *selector* legs (`packed`, `coe`); for
ranking metrics it is subsumed by reporting `recall@50/@100` over the full union.

### On "interleave rather than append", and its rationale

The brief is right that append-not-fuse is deliberate: graph signals must not displace strong
lexical/dense hits. This experiment does not discard that rationale — it **tests** it. The
`fuse w` sweep *is* the rationale as a measurable curve: `w=0` is pure base, `w=1` is the
co-equal fusion the ladder's `kg` rung already showed regressing, and if the rationale is right
the curve is monotonically bad in `w`. `protect@k` is the weaker form the rationale leaves open
— a hard floor under the top `k` makes displacement of the strong hits impossible by
construction. If both lose, the rationale is confirmed by measurement rather than by assertion,
and no interleaving variant is worth shipping.

## 4. Datasets and cost

`musique` and `hotpotqa` **cannot be re-run in this checkout** — their exports carry no
`bm25-stats.json` / `bm25-postings.parquet` (verified via `legs.missing_inputs`), so their
committed cells describe artifacts no longer on disk. That rules out the brief's "musique is
cheap".

The runnable cheap pair, and why: **`multihop_rag`** (2,255 queries, 3,264 chunks, ~515 ms/query
— multi-hop, where traversal should pay) and **`vidoseek`** (1,142 queries — single-hop control,
where it should not). ~30 min for both at `pool_k=100`. `wiki2multihop` (12,576 queries) is the
escalation if the cheap pair is equivocal; `uda` (8,583 queries, 596 MB corpus) is the expensive
one and is not run unless a positive result needs confirming.

## 5. Determinism

`PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, asserted through
the matrix's own `require_deterministic()`. The probe writes `resolution/` under **this ticket**,
never under the campaign stage, so committed evidence cannot be overwritten. Bootstrap draws are
seeded. The 40-query smoke reproduced the committed cell's median latency (515.4 ms vs 509.57)
and its usable-query count (2,255/2,556), which is the check that the trace and the cell describe
the same run.

## 6. Stop rule, decided before the numbers

* **Ceiling ≤ +0.01 recall@10 on both datasets** → negative result. No tuning sweep, no `uda`
  run. Write it up, and rewrite the retrieval story around what does work.
* **Ceiling > +0.01 but no arm realises >½ of it** → the append policy is the binding constraint;
  report which arm comes closest and what it costs.
* **Some arm beats `base` with a bootstrap CI excluding 0, on the multi-hop dataset, without
  regressing the single-hop control** → the graph lanes earn a configuration change; confirm on
  `wiki2multihop` before claiming it.
