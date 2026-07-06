# NEXT — work queue (WIP = 1)

Phase 1 is a strict build-through. Pull the top unblocked issue, meet the **Baseline bar**, mark done, move the pointer.

## Now
- [ ] **S1 — walking-skeleton** ← start here

## Phase-1 queue (in order)
S2 parse · S3 chunk+screening · S4 entity-extraction · S5 relation-extraction · S6 redaction · S7 profiling · S8 disambiguation · S9 graph-assembly+export · S10 quality-report · S11 incremental-deltas · S12 performance-path+runner · S13 demo-package

## Then — HARD STOP
- [ ] **G1 — profound-testing-gate** (do not start Phase 2 until every G1 box is green)

## Phase-2 queue (after G1)
H-core · H2 · H3 · H4 · H5 · H6 · H7 · H8 · H9 · H10 · H11 · H12 · H13

## Rules
- One issue in progress at a time on the critical path.
- An issue is done only when its DoD bar is met AND CI is green.
- Detours go to `../docs/` or a new backlog issue — do not expand the active issue's scope.
