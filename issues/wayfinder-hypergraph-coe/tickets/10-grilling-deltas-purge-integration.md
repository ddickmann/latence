# 10 — Grilling: sidecar artifacts under Corpus Versions, Retraction, Purge

Type: grilling
Status: resolved
Blocked by: 03, 04

## Question

How do the hypergraph sidecar artifacts participate in the delta machinery (ADR-0029, S11): incremental append (spec 1.7 — postings/B/A_E grow, no global recompute) inside a Corpus Version transaction; Retraction tombstones filtered at query time; GDPR Purge physically expunging EU tau_text + provenance across versions; compaction cadence; drift/reconciliation triggers. Outcome: the sidecar's delta contract + ADR note.

## Answer

**The integration is structurally correct BY INHERITANCE — and now proven by test, not
asserted.** No new machinery was needed; the decision content is *why* that is true and
what stays deliberately unbuilt:

1. **Delta runs**: corpus-level Stages (Disambiguation, Graph Assembly, …) fully RE-RUN
   over reused + fresh records on every delta (`delta_runner.py` — the existing S11
   semantic). The hyperedge stream is Graph-Assembly output, so **hyperedges, the parquet,
   and the KV are regenerated per Corpus Version automatically.** Spec 1.7's incremental
   append machinery ("appends only touch postings/B/A_E") is moot twice over: A_E/B do not
   exist (ADR-0056), and the corpus-level re-run IS the delta semantic this repo already
   accepted.
2. **Retraction**: retracted documents' records are excluded from the re-run's input, so
   the new Version's sidecar simply never contains them. **No query-time tombstone
   filtering needed** — the committed Version's KV is already clean (per-call reads see
   only committed files, ADR-0056).
3. **Purge**: hyperedge GraphRecords are derived records in the Version's record log, so
   the existing expunge covers them; the rewritten Version's parquet/KV physically lack
   the purged `tau_text` — "parquet purge = rewrite" exactly as ADR-0056 anticipated.
   **Proven**: `test_purge_expunges_hyperedges_and_their_tau_text` (purged doc's tau_text
   gone from the head Version; co-resident doc's Hyperedges NOT collaterally erased).
4. **Scope safety**: every scope-partitioning site (delta churn, context enrichment, graph
   completion) uses positive NODE/EDGE filters — HYPEREDGE records flow through untouched.
5. **Residuals, deliberate**: delta churn statistics do not yet count hyperedges
   (added/retracted hyperedge counts in the delta Quality Report — small additive
   follow-up, not gating); incremental hyperedge maintenance (avoiding the corpus-level
   rebuild at very large scale) is a **measured-need optimisation** — per the standing
   rule it is not built without data showing the rebuild cost hurts, and reconciliation
   already bounds drift.
