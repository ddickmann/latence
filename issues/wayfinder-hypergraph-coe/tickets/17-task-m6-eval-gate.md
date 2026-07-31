# 17 — Task: M6 — eval harness + SOTA gate run

Type: task
Status: open
Blocked by: 08, 16

## Question

Implement the eval per 08's design and run it for real: MuSiQue/HotpotQA supporting-fact recall@budget curves + the enterprise-shaped corpus eval, vs the current chunk-topk baseline. Gate: strictly dominate the baseline curves. Publish tokens-in-context vs supporting-fact-coverage curves into the repo's benchmark docs. No fake green.

Also carries two A/Bs deferred here rather than guessed:
1. **Heuristic vs exact selection weights** — how much packing quality degrades when selection runs
   on approximate `token_cost` ([ADR-0058](../../../docs/adr/0058-token-cost-is-an-approximate-index-time-weight-with-an-exact-query-time-verification-pass.md)).
2. **Weak Hyperedges on vs off** — they are expected to outnumber relation-bearing units several to
   one ([ADR-0057](../../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)); the eval decides the default.
