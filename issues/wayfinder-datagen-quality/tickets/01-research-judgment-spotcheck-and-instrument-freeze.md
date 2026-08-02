# 01 — Research: judgment spot-check + instrument freeze

Type: research
Status: open
Blocked by: —

## Question

Every number this map optimizes against comes from 250 LLM-generated judgments
(`benchmark/official-data/musique-full/calibration-judgments.jsonl`, Claude Opus 2026-08-02,
pending human spot-check per `PROVENANCE.md`). Before any fix ticket claims a measured
improvement:

1. **Spot-check**: Dennis (or a delegated re-judge with a different model + disagreement
   review) labels a random ~50-judgment subset via the `latence-label` UI; report agreement
   (exact + partial-credit). If agreement is below a bar Dennis sets here, the sample is
   re-judged before ticket 02 proceeds.
2. **Freeze the instruments**: pin (a) the sampling protocol (sort by `(gamma, hyperedge_id)`,
   stride `N // 250`), (b) the scoring rules from `M6-CALIBRATION-OFFICIAL.md` (arity-split
   siblings not charged; redundant-but-true labels not charged; span-topic kind-terms not
   charged), and (c) the frozen retrieval harness invocation (`benchmark/official/`) as the
   regression gate every fix ticket must re-run.
3. **Set per-pattern baselines** so each fix ticket has a number to beat: fabrication rate
   58% (46/80 relation-bearing), common-noun promotions ~30/250, type errors ~29/250,
   cross-cell welds (pattern 5 count), surface-dupes (pattern 6 count), unextracted subjects
   8/250, sample mean 0.700.

## Outcome

An `INSTRUMENT.md` in this map's directory: agreement number, frozen protocol, per-pattern
baseline table. Unblocks every other ticket.
