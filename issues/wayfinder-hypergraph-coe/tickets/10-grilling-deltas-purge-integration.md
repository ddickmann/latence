# 10 — Grilling: sidecar artifacts under Corpus Versions, Retraction, Purge

Type: grilling
Status: open
Blocked by: 03, 04

## Question

How do the hypergraph sidecar artifacts participate in the delta machinery (ADR-0029, S11): incremental append (spec 1.7 — postings/B/A_E grow, no global recompute) inside a Corpus Version transaction; Retraction tombstones filtered at query time; GDPR Purge physically expunging EU tau_text + provenance across versions; compaction cadence; drift/reconciliation triggers. Outcome: the sidecar's delta contract + ADR note.
