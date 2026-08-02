# 11 — Grilling: scope the data-gen SOTA sweep

Type: grilling
Status: resolved
Blocked by: —

## Question

Dennis widened this effort beyond the spec to a data-generation SOTA sweep. Enumerate candidates (parser quality frontier, extraction quality/calibration beyond 09, schema induction, redaction, profiling signals, synthetic augmentation, …), decide what is actually in, and whether the sweep gets its own sub-destination + tickets on this map or spins off as a separate map. Outcome: sweep scope decision; graduates the corresponding fog.

## Answer

**Resolved by Dennis's clarifying question, which dissolved the premise** (2026-07-31):

> "We create structured markdown with tables, charts, forms, headings … GLiNER2 NER,
> relations … centrality, enrichment … chunk first and do all the KG disambiguation. So we
> build a KG with all the data. Why should we apply something different to the hypergraph's
> edges?"

Correct — nothing different is applied, **by construction**: Hyperedges are a re-projection
of the SAME extracted records (M1's `assemble_with_documents` consumes the run's existing
chunks/mentions/relations/entities). Every upstream data-gen improvement — parsing, table
structure, extraction, disambiguation, enrichment — flows into the hypergraph automatically.
A "hyperedge-specific data-gen sweep" is a category error; my option framing (tables into
Hyperedges, etc.) implied missing capabilities that are not missing.

Consequences:
1. **The sweep leaves this map's destination.** Generic pipeline-quality work is its own
   effort: a **spin-off map charted AFTER the M6 gate** (Dennis), so its priorities come
   from the eval's measured weak spots, not guesses. Recorded under Out of scope.
2. One genuine refinement noted for the record, not a sweep pillar: table rows are not
   sentences — the RuleSentenceSplitter's newline-boundary rule makes each markdown table
   row its own segment, which is reasonable but unmeasured on real tables; if the eval
   shows table evidence fragmenting, that is a splitter refinement inside THIS map's
   follow-ups.
