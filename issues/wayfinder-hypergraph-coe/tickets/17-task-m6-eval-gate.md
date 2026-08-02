# 17 — Task: M6 — eval harness + SOTA gate run

Type: task
Status: resolved
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

## SUBSET GATE: PASSED (2026-08-02) — both datasets, dev-tuned piecewise policy, disclosed

Harness: `benchmark/musique-subset/run_eval_final.py` (+ the piecewise scorer); results in
`final-results.json` / `final-results-piecewise.json`. Discipline held: the composition
policy was tuned on the 500 MuSiQue DEV questions only; test (500q) and enterprise (100q)
scored with it unchanged. **Two scorings, both reported**: the flat dev-best share (0.25)
FAILED the pre-committed rule on MuSiQue's 1k endpoint (0.365 vs 0.474, −23%); the
dev-derived **budget-conditional policy** (baseline@1k, CoE-0.25@≥2k — dev showed every
share losing at 1k and 0.25 winning everywhere above) PASSES:

- **MuSiQue test**: AUC 0.5378 vs 0.478 (0.508 was a transcription error vs the stored
  `final-results-piecewise.json` gate — corrected by the 2026-08-02 close-out audit);
  endpoints hold; worst point 0.0%; +7.4 to +11.2
  points at every budget ≥2k against a baseline that plateaus at 0.479.
- **Enterprise**: AUC 0.820 vs 0.707; +9.8 to +15.6 points at ≥2k (peak 0.862); worst 0.0%.
- Mechanism honesty: the baseline plateaus because the packer never admits
  zero-lexical-overlap evidence — exactly the gap CoE fills; at 1k on wiki questions
  tier-1-only entry buys τ spans worth less than the BM25 chunks they displace, hence the
  budget-conditional policy.

**Scope of this result, precisely**: the ticket-08 SUBSET gate. Still open for the official
full-corpus numbers (ticket 08's "one full pod run at gate time"): the full MuSiQue dev
corpus, per-tier curves (this run is tier 1 only — tier 0 needs the caller-embedding path,
tier 2 the query-NER extra), the heuristic-vs-exact and weak-on/off A/Bs, the calibration
A/B (γ is continuous here: 8,572 distinct values — unblocked), and recall@5 for HippoRAG-2
comparability. Ops notes for that run: the 11h harness wall (pack-pool + Jaccard cache fixes,
result-identical, are in the harness now) and the linear exact-surface disambiguator that
replaced both shipped resolvers past ~10⁴ unique surfaces (a productization ticket candidate).

## OFFICIAL GATE: PASSED — both FULL corpora (2026-08-02)

The ticket-08 design executed in full: MuSiQue complete validation (2,417 q; 60,054
hyperedges) + HotpotQA dev distractor (7,405 q; **214,769 hyperedges**, 349k nodes), GLiNER2
indexing on the pod (one quota crash, checkpoint-resumed — the transactional Storage held),
dev-tuned per dataset, ONE test scoring each, gates:

| | MuSiQue-full test | HotpotQA-full test |
|---|---|---|
| AUC | **0.4428 vs 0.3846 (+15.1%)** | **0.6926 vs 0.6496 (+6.6%)** |
| @16k | 0.4894 vs 0.385 (**+10.4 pts**) | 0.7465 vs 0.6496 (**+9.7 pts**) |
| endpoints / worst | hold / 0.0% | hold / 0.0% |

**Per-tier (dev)**: tier2 (GLiNER2 query NER) wins BOTH datasets (MQ 0.4464, HP 0.6427 AUC)
— the ADR-0059 opt-in tier earns its place; tier1 n-grams close on MQ, weak at tight
budgets on HP; tier0 lexical-floor trails on MQ (0.373) but is a strong 1k performer on HP.
**A/Bs**: exact-vs-heuristic token costs ≈ equal (ADR-0058's bet confirmed at 375k-hyperedge
scale); weak Hyperedges = tail recall at big budgets, a 1k liability (the piecewise policy
absorbs this); **calibration = recall-neutral** (per-regime isotonic, dev A/B 0.4452 vs
0.4464 — γ carries 15% of utility; ships as adopter tooling, no recall claim). The judged
sample's real finding is upstream: GLiNER2@0.5 fabricates relations on wiki text (58% of
relation-bearing samples ≥1 unasserted triple; γ's high tail LEAST reliable; γ≤0.5 ⟺ weak
by construction → calibration must be per-regime) — the opening evidence for the post-M6
data-gen map. recall@5 comparability + Dennis's spot-check of LLM-generated judgments remain
open notes, not gate conditions. Harness honesty: profile-driven speedups only (pack-once/
prefix-truncate disclosed and uniform across arms; indexed BM25 with identical scoring math).

## Close-out audit addendum (2026-08-02)

recall@5 — left above as an "open note" — is now DELIVERED (`benchmark/official/run_recall5.py`
→ `recall5-results.json`, one scoring on the pre-registered test files): baseline BM25
0.4053/0.6960 (MuSiQue/Hotpot) vs pure-ρ CoE tier2 0.2094/0.3506. Standalone structural
ranking is far below BM25 — ρ feeds budgeted composition, and the hybrid is what the gates
measured; the number quantifies the missing dense signal (see CLOSEOUT-AUDIT.md F-VEC).
Also on the record there: the metric-granularity asymmetry (F-METRIC), the BM25-not-dense
baseline deviation from this ticket's pre-registration (F-BASE), and the corrected subset
AUC transcription.
