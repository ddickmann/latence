# 08 — Grilling: eval design + enterprise gold set

Type: grilling
Status: resolved
Blocked by: 02

## Question

Design the gate that defines "SOTA reached": (a) MuSiQue/HotpotQA supporting-fact recall@budget — budget grid, baseline definition (current chunk-topk through latence-retrieval), strict curve-dominance criterion, seeds/determinism; (b) the enterprise-shaped corpus eval — which corpus (benchmark/e2e assets?), how supporting-fact gold labels get made, size; (c) where it lives and what hardware it needs. No fake green: real models, real datasets, real runs. Outcome: eval spec + thresholds that ticket 17 implements.

## Answer

Decided with Dennis (2026-07-31), building on ticket 02's licence/metric facts:

**Datasets + metrics.** MuSiQue-Ans (CC BY 4.0; supporting-fact F1 per its evaluate_v1.0)
and HotpotQA distractor (CC BY-SA 4.0, share-alike on derived artifacts; sentence-level Sup
EM/F1). Novel reporting: **supporting-fact recall@token-budget curves** over the grid
**1k/2k/4k/8k/16k**; additionally report **recall@5** for HippoRAG-2 comparability and the
canonical per-dataset supporting-fact metrics so the numbers are cross-paper legible.
HyCE-RAG's own numbers are NOT comparable (LLM-judge, segment cap — ticket 02) and are cited
as lineage only.

**Baseline.** Chunk-topk through this repo's own stack: dense first stage + `Packer` at the
same token budget — the strongest thing we currently ship, not a strawman.

**Gate rule (Dennis): AUC + endpoints.** CoE must dominate the area-under-curve on BOTH
datasets AND at the smallest and largest budget points; no single budget point may degrade
more than 2% relative. Robust to benchmark noise, still unambiguous.

**Enterprise-shaped eval (Dennis): MANUAL gold set** — the strongest evidence path, chosen
over planted chains: ~100 multi-hop questions with hand-marked supporting facts over the
real benchmark/e2e messy corpus. Dennis labels; the harness must make that efficient (a
labeling UI/flow is part of ticket 21's scope, not an afterthought). Spun out as
**[ticket 21](21-task-enterprise-gold-set.md)** (HITL task) blocking ticket 17.

**Compute (Dennis): subset-first.** Develop + tune against a dev subset (~1–2k questions'
supporting corpora; one short pod run for GLiNER2 fused indexing), then ONE full-corpus pod
run for the official gate numbers — two live runs total, per the diagnose-tight rule.

**Carried A/Bs** (from ADRs 0057/0058/0059, all reported per the standing data-over-caution
rule): entry tier 0/1/2 (curves per tier — a single headline would silently assume tier 2);
heuristic vs exact selection weights; weak Hyperedges on/off.

**Home:** `latence-benchmark`'s pinned-dataset seam; runs on the benchmark/e2e supervisor
pattern (on-host supervisor, done-probes — never me polling a runbook).
