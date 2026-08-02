# 10 — Grilling: chunk-size ↔ hyperedge-yield coupling

Type: grilling
Status: open
Blocked by: 01

## Question

Parent ticket-21 finding 3: the same enterprise corpus yielded **40 Hyperedges at chunk
`max_tokens: 220` vs 226 at 90** — sentence windows rarely cross chunk boundaries, so chunk
size gates segmentation. Is this coupling harmful on real corpora at real budgets, and if
so, where is the fix?

Run the sweep the parent map deferred: chunk size as a reported dimension on the official
corpora (subset first per the compute discipline), scored with the frozen retrieval harness —
does hyperedge yield move recall@budget, and does the piecewise policy shift? Only if the
coupling is confirmed harmful does the builder get cross-chunk windowing (the parent-map
ruling: a measured-need change, not built on suspicion).

## Outcome

The sweep numbers + ruling; a builder-windowing task ticket only on measured need.
