# 17 — Task: M6 — eval harness + SOTA gate run

Type: task
Status: claimed
Blocked by: 08, 16, 21

## Question

Implement the eval per 08's design and run it for real: MuSiQue/HotpotQA supporting-fact recall@budget curves + the enterprise-shaped corpus eval, vs the current chunk-topk baseline. Gate: strictly dominate the baseline curves. Publish tokens-in-context vs supporting-fact-coverage curves into the repo's benchmark docs. No fake green.

Also carries two A/Bs deferred here rather than guessed:
1. **Heuristic vs exact selection weights** — how much packing quality degrades when selection runs
   on approximate `token_cost` ([ADR-0058](../../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md)).
2. **Entry tier 0 / 1 / 2** ([ADR-0059](../../../docs/adr/0059-query-side-entry-is-tiered-zero-model-default-with-gliner2-strictly-opt-in.md)) — report the curve **per tier**; a single headline number silently assumes tier 2 and misrepresents the default install. Also settles whether tier 1's alias-only linking approaches tier 2's anchor precision.
3. **Weak Hyperedges on vs off** — they are expected to outnumber relation-bearing units several to
   one ([ADR-0057](../../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)); the eval decides the default.

## Enterprise leg — first curves (2026-08-01, tier 1, gate NOT yet passed — reported as measured)

Harness: `benchmark/enterprise-gold/run_eval.py` (budget parity via `estimate_tokens` both
sides; baseline = the repo's own BM25→Packer chunk-topk; CoE = tier-1 alias entry → signals →
select → chains). Results (`eval-results.json`):

| budget | baseline | CoE-pure | CoE+fallback |
|---|---|---|---|
| 1k | 0.707 | 0.484 | 0.484 |
| 2k | 0.707 | 0.484 | 0.498 |
| 4k | 0.707 | 0.484 | **0.804** |
| 8k | 0.707 | 0.484 | **0.813** |
| 16k | 0.707 | 0.484 | **0.813** |

- **The multi-hop lift is real: +9.7 to +10.6 points at every budget ≥ 4k.** The baseline
  plateaus at 0.707 regardless of budget — its packer never admits zero-lexical-overlap
  chunks (the exact mechanism CoE exists to beat).
- **Gate FAILS as-is** (AUC 0.683 vs 0.707; endpoints lose) because the naive composition
  starves the fallback at 1k/2k: CoE fills the whole small budget first.
- **Diagnosis (one pass, no knob-fiddling):** entry is NOT the issue (0/100 zero-anchor
  questions; median 1 anchor). The CoE-pure ceiling is **structural: 0.680** — only 153/225
  gold spans exist verbatim inside ANY τ window (the chunk-size/segmentation coupling
  already on the map as fog). Selection captures 71% of the reachable.
- **Deliberately NOT done:** tuning the CoE/baseline budget-split on this eval set (test-set
  fitting). The composition policy gets tuned on the PUBLIC DEV subset (the ticket-08
  "subset first" decision exists precisely for this), then re-scored here unchanged.
