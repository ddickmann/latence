# 11 — Task: alias + description enrichment at export

Type: task
Status: open
Blocked by: 01

## Problem (carried fog, now the measured entry bottleneck)

The KV alias keyspace holds **canonical names only** — glinker's richer aliases die before
export (parent M2 fog). The official eval showed entry is where the tiers differ: tier-2
query NER beat tier-1 n-gram alias lookup on both datasets, partly because the alias table
is impoverished. A richer alias table is the zero-model way to close some of that gap —
and it is a data-gen deliverable (export-side), not a retrieval change.

## Task

Persist through to the alias keyspace: (a) glinker's alias/mention surface variants that
currently drop, (b) the surface forms ticket 07's merge rules unify (each merged surface
becomes an alias of the winner), (c) normalized date/title variants. Writer-side only; the
KV format already supports it (`a|` keyspace); the independent reader is format-frozen —
no format change, just more rows.

Measure: re-run the official harness tier-1 arm with the enriched table (subset first) —
does tier-1 close on tier-2? That number decides whether the no-model default entry gets
better for free.

## Outcome

Export enrichment + tests + the tier-1-vs-tier-2 delta with the enriched table.
