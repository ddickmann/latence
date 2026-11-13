# Grilling: the text-path upgrade

Type: grilling
Mode: HITL
Status: open
Blocked by: 02, 07, 08

## Question

From the research shortlist and the recorded baseline, decide:

- which extraction stack (or stacks) enter the bake-off, and the bar each must clear on
  quality and throughput;
- the label strategy (per-chunk versus corpus induction, and the label-count ceiling);
- the window and chunking;
- how relation fabrication is controlled;
- the serving path (in-process versus endpoint);
- ULTRA's role and acceptance bar, given the numbers *Record the baseline* finds;
- which current Provider packages are retired.

## Input from the feasibility spike (2026-11-11)

Measured on synthetic documents ([`../spike/REPORT.md`](../spike/REPORT.md)); the baseline of
*Record the baseline* is still missing, so this ticket stays blocked.

- **Extraction:** GLiNER-Relex finds 99.5% of the gold organisation mentions at precision
  0.845.
- **Linking:** character n-grams for candidates, the reranker to choose, GLiDE 0.4B as a
  decision feature: 66.6% of mentions auto-linked at precision 0.965.
- **The embedder** adds under one point of candidate recall, and must search with the mention,
  not the whole document.
- **Acronyms** need evidence outside the string (thread, definition, alias column): 0.1% to
  81.8% correct with it on the set written for it.
