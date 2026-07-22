# T5 — Bound per-chunk induction cost at enterprise scale

**Stage:** `induce` / perf · **Depends on:** T1 · **Severity:** major (feasibility of the redesign on
real corpora)

## The problem
Per-chunk induction (T1) is **one LLM call per chunk**. On a real enterprise corpus (say 10⁵–10⁷
chunks) that is a large cost and latency surface. The redesign is correct for quality but must be
affordable, or it won't ship.

## Proposed change — three levers, layered
1. **Chunk-text dedup before inducing.** Many chunks are near-identical (boilerplate headers/footers,
   repeated clauses, templated sections). Induce once per *distinct* chunk text (content hash /
   near-dup cluster) and fan the result to its duplicates. This alone typically removes a large
   fraction of calls on real corpora. Deterministic, no quality loss.
2. **A pluggable, cheaper inducer for the per-chunk pass.** Per-chunk induction is a *narrow*
   task (list the entity/relation types present in one short chunk) — it does not need a frontier
   LLM. Add an inducer Provider seam so an operator can point the per-chunk pass at:
   - a small **local** model (served/in-process), or
   - a distilled **classifier / zero-shot tagger** over a candidate type space,
   while keeping the LLM inducer available for the doc/corpus passes and for high-value corpora.
   (This is a natural Provider-seam extension — `label_inducer.*` — consistent with ADR-0016/0004.)
3. **Concurrency + budget controls.** The inducer already fans out concurrently with retry/backoff;
   expose a max-in-flight + a per-run token/cost budget with a Quality-Report line reporting induction
   cost (calls, tokens, cache-hit rate). No silent runaway.

## Acceptance criteria
- [ ] Chunk-text dedup: a corpus with heavy boilerplate induces on distinct texts only; the
      call-count reduction is measured and reported; duplicates receive the identical induced set.
- [ ] The inducer is a swappable Provider; at least one non-frontier option (local model OR classifier)
      is available and documented for the per-chunk pass.
- [ ] The Quality Report carries induction cost (calls, tokens, cache-hit %, and — for a budgeted run —
      whether the budget bound.)
- [ ] Determinism preserved (dedup keyed on content; temp-0 induction); `verify-local` green.

## Affected files
`packages/latence-schema-inducer/.../provider.py` (dedup + budget), a new inducer Provider package for
the cheap/local option, the Quality-Report schema, `setup/wizard.py` (inducer selection), the docs.

## Honest note
This ticket is what makes T1 defensible for a customer at scale. Without it, "per-chunk induction" is
correct but reads as "one gpt-4.1-mini call per chunk across your whole corpus" — a number an
enterprise evaluator will (rightly) balk at. The dedup + cheap-inducer path is the answer; ship it
with T1, not after.
