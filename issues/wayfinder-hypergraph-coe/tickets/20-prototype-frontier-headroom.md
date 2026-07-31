# 20 — Prototype: frontier headroom (what capability fits in a 20–50 ms budget)

Type: prototype
Status: resolved
Blocked by: —

## Question

Dennis, reviewing ADRs 0056–0059: *"we must approximate the frontier as much as possible while
keeping production-ready latency at scale... I only accept sacrifices that are backed by data,
otherwise it's counter-innovative behaviour."*

Audit of the four ADRs: exactly **one** sacrifice was evidence-backed (global `A_E`, infeasible at
2.6 TiB / 1M EUs, ticket 18). Everything else was parked out of caution against a latency budget
that was never established. Measured structural path = 2.3 ms; the adopter's own ANN costs 10–50 ms
and this repo already ships a 50–200 ms cross-encoder reranker. **~20× headroom is unspent.**

Measure the capability/latency curve at 1M EUs so the operating point is chosen, not defaulted to:

1. **`|E_c|` sweep** 2k → 20k — the cap was inherited from the spec, never tested.
2. **RWR steps `K`** 5 → 20 — same.
3. **2-hop candidate expansion** off/on (via postings union, since global `A_E` is gone) — the spec's
   optional expansion, which the ADR-0056 layout made awkward and I then quietly dropped.
4. **Weighted incidence** (`link_conf × role_weight`, spec 6.1) vs binary — expected ~free online;
   confirm, and measure whether it improves ranking.
5. **Heat-kernel propagator** (spec 6.2) vs RWR — second adapter on the M3 `Propagator` seam.
6. **Truncated PPR sketches** (spec 1.7 item 5) — index-time cost and size at top-200/entity.
   NOTE: unlike `A_E` these are **bounded by construction** (K per entity, not `Σd_v²`), so the
   measurement that killed `A_E` does not apply to them. This was my conflation.
7. **Real GLiNER2 query latency** on a ~10-token string — ADR-0059 made tier 2 opt-in on an
   *asserted* "tens of ms on CPU" that was never measured. A query is not a document.

**Quality axis (synthetic but real):** plant multi-hop chains with known ground-truth Hyperedges
plus lexical distractors, and measure **supporting-fact recall@budget** — the same metric shape as
the M6 gate. This prefigures the real eval and lets the knobs be tuned against quality, not guessed.

Outcome: amend ADRs 0056/0058/0059 with the measured operating point; recover every capability the
data does not justify dropping.

## Answer

Prototype: [`prototypes/residency/frontier.py`](../prototypes/residency/frontier.py); raw numbers in
`prototypes/residency/frontier-results.json`. Synthetic corpus (200k EUs, Zipf entities) with **40
planted 2-hop chains** as ground truth plus lexical distractors; metric = supporting-fact
recall@budget, decomposed into `cand_recall` (did the bridge reach `E_c`) and `recall` (was it
selected).

### The headline is a NEGATIVE result, and it is the most important thing here

**No configuration beat the 1-hop baseline.** Best-in-budget (cap=20000, K=15, hops=3, weighted,
prune=2000, w_prop=0.55, rho=max) scored **recall 0.487 against a 1-hop baseline of 0.500.** Every
knob was swept — hops, `|E_c|`, K, kernel, weighted incidence, rho pooling, rho-pruning, w_prop —
and none recovered multi-hop recall.

The mechanism's failure mode is precise and reproducible (clearest at 50k EUs, where the bridge does
reach `E_c`):

| hops | `cand_recall` | **selection** `recall` |
|---|---|---|
| 1 | 0.500 | 0.500 |
| 2–4 | **0.950** | **0.150** |

Expansion lifts the bridge into the candidate set (0.50 → 0.95) and selection then **drowns it**
(0.50 → 0.15). Pruning by `rho` "recovers" recall to 0.500 only by discarding the bridge
(`cand_recall` falls to 0.500) — i.e. by reverting to 1-hop behaviour. **`rho` does not rank the
bridging Hyperedge above corpus noise**, and mean/max/sum pooling are within noise of each other.

### Why this probably does NOT refute the design — the harness's own fatal limit

**A Zipf-random hypergraph has no community structure.** Real corpora cluster topically, and topical
clustering is exactly the signal diffusion exploits; here the planted chain is a single thread in
pure noise, so propagation has nothing to propagate *along*. The setup is also maximally adversarial
in three further ways: planted chains carry **zero** lexical signal (sim=0.1) while distractors carry
the maximum (sim=1.0) **and** higher γ (0.95 vs 0.80); 9 of 10 entry anchors are noise hubs.

**Conclusion: this harness can neither validate nor refute the multi-hop claim.** It is the wrong
instrument, and no amount of further tuning on it would produce a trustworthy number — that would be
fitting knobs to noise.

### What IS established (independent of corpus structure)

1. **1-hop candidate generation is a hard structural ceiling.** At hops=1, `cand_recall` = 0.500 —
   the bridge never enters `E_c`, so propagation cannot rank what is not there. **The spec's
   "optional" 2-hop expansion is load-bearing, not optional.**
2. **γ-ordered global truncation silently converts a multi-hop retriever into a 1-hop one** (chain
   γ=0.80 loses to distractor γ=0.95). Truncation must be **hop-priority**: drop the outermost hop
   first, never truncate globally by score.
3. **The dense `|E_c|²` overlap matrix is what caps the candidate universe**, not propagation — 1.6 GB
   at `|E_c|`=20000. Replacing it with an **incremental sparse** running-max (one sparse product per
   acceptance, ~40 total) removed the cap and cut latency ~4×. This is a pure engineering win and
   belongs in M4 regardless of everything else.
4. **Truncated PPR sketches are bounded by construction** — 7.2M entries / **82 MiB** at 200k
   entities, 0.4 s to build. The `Σd_v²` blowup that killed `A_E` (ticket 18) **does not apply**;
   conflating them was my error.
5. **Weighted incidence costs ~12% latency** (647→725 ms), not zero as I claimed. Cheap, but not free.
6. **Hub entities must be capped during expansion** — one hub (366k edges at 1M) otherwise drags the
   whole corpus into `E_c`.

### What remains UNMEASURED

Real GLiNER2 query latency (sweep 7 of the ticket) — not run; ADR-0059's "tens of ms on CPU" is
**still an assertion**, not data.

### Recommendation — reorder the plan

The multi-hop claim is the entire product thesis and it is **currently unvalidated**. Validating it
needs real structure, i.e. the MuSiQue/HotpotQA harness ([tickets 08](08-grilling-eval-design.md) /
[17](17-task-m6-eval-gate.md)) — which is currently scheduled **last**, after M1–M5 are built on the
assumption it works.

**Move the eval earlier: build a thin vertical slice (M1→M4, minimal) against a real multi-hop
benchmark before hardening any of it.** If the claim does not survive a real corpus, that must be
discovered at slice stage, not after five milestones of work. Latency absolutes above are
Python-bound and not product SLAs; only the comparisons carry.
