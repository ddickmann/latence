# 12 — Task: ANN-blocked disambiguation as correlation clustering (embed → k-NN graph → tabu-refined local search)

Type: task
Status: open
Blocked by: 01

## Problem (measured + architectural)

Two compounding limits in the shipped disambiguation, sighted live in the e2e-sota-run
campaign (S3, 2026-08-04):

1. **Candidate generation walls.** The ported cascade generates candidate pairs by
   normalized-token-overlap blocking; on financial/scientific corpora (shared tokens: "inc",
   "revenue", month names, section numbers) pair counts grow ~quadratically inside blocks.
   Measured wall ~10⁴ distinct surfaces (parent campaign); the SOTA corpus is 10⁵–10⁶
   (ohr_bench alone ≈250k mentions). The glinker repo's embed-first advice (precomputed
   biencoder embeddings) only cheapens PAIR SCORING — it does not touch the pair count.
   Consequence: S3 ships `disambiguation.exact_surface` (exact-match merges only), leaving
   the pattern-6 duplicates (ticket 07) unmerged at scale.
2. **Greedy pairwise decisions have no global objective.** The cascade is Union-Find over
   independently-decided pairs: transitive chains (A≈B, B≈C ⟹ A~C even when A,C are
   dissimilar) are exactly the over-merges a global objective repairs, and each pair is
   decided once with no revision.

## Task (Dennis's framing, 2026-08-04: disambiguation IS combinatorial optimization)

Cast corpus-internal entity resolution as **correlation clustering over an ANN-built sparse
similarity graph**, solved by seeded-greedy + tabu-refined local search:

1. **Move space (fixes limit 1)**: one batched embedding pass over all N distinct surfaces
   (reuse the pipeline's embedder seam; glinker biencoder as the quality candidate), FAISS
   top-k index (the `faiss` extra exists, unwired) → sparse k-NN similarity graph, O(N·k).
   All candidate moves live on this graph; a move's Δ-objective evaluates in O(k).
2. **Objective (two-sided — the one-sided "minimize duplicate entities" degenerates to
   merge-everything)**: correlation-clustering disagreement cost — penalize similar pairs
   split across clusters AND dissimilar pairs merged inside one, with the precision term
   weighted ≥ the recall term (over-merge is worse; ticket 07's ruling carries over).
3. **Constraints as first-class**: distinguishing-token guard, type filters, and cross-script
   homoglyph separation become HARD cannot-link constraints (never violated by any move);
   ticket 07's deterministic rules (name-subsequence, date canonicalization,
   truncation-prefix) become must-link priors / the greedy seed.
4. **Solver**: deterministic greedy seed (sorted move enumeration) → local search with
   merge / split / relocate moves and a tabu list (recently reversed moves barred, prevents
   oscillation); fixed seed, anytime — stop at wall-clock budget, keep best-found. Identical
   input + config ⟹ byte-identical clustering (the Baseline bar).
5. **Audit preserved**: the accepted-move log IS the merge audit (strategy=solver, Δ-cost,
   constraints checked), same `MergeDecision` shape; below-policy candidates absent exactly
   as ADR-0046 discloses.

Ship as a new provider behind the existing audited-cascade seam
(`disambiguation.correlation_clustering`), default OFF — `exact_surface` stays the shipped
default until the gate below says otherwise (standing rule: sacrifices/upgrades only with
data).

## Gate (pilot before any big-set run)

Pilot on vidoseek (smallest surface set, ~2.4k chunks), then ohr_bench:

- **Merge-precision sample**: deterministic-stride sample of accepted merges judged for
  correctness; precision floor ≥ the exact_surface baseline (which is ~1.0 by construction —
  every added merge must survive judgment).
- **Duplicate-count delta** on the ticket-07 pattern (title/body forms, date formats,
  truncated titles) — the recall side the objective exists to buy.
- **Retrieval non-regression**: the frozen M6 harness gates stay green (no point >2% worse);
  wall-clock and peak-RAM/VRAM recorded vs exact_surface.
- Solver diagnostics recorded: objective trajectory, moves accepted/tabu-rejected, time to
  best-found — the anytime curve that justifies (or kills) the metaheuristic layer.

## Outcome

Provider + tests (constraint-violation traps, determinism byte-tests, counter-example
non-merges) + pilot report with the three gate measurements; a ruling whether
`correlation_clustering` becomes the recommended large-corpus configuration.
