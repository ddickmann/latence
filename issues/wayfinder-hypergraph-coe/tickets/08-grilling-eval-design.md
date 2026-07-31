# 08 — Grilling: eval design + enterprise gold set

Type: grilling
Status: open
Blocked by: 02

## Question

Design the gate that defines "SOTA reached": (a) MuSiQue/HotpotQA supporting-fact recall@budget — budget grid, baseline definition (current chunk-topk through latence-retrieval), strict curve-dominance criterion, seeds/determinism; (b) the enterprise-shaped corpus eval — which corpus (benchmark/e2e assets?), how supporting-fact gold labels get made, size; (c) where it lives and what hardware it needs. No fake green: real models, real datasets, real runs. Outcome: eval spec + thresholds that ticket 17 implements.
