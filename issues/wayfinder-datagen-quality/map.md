# Wayfinder map — Data-Generation Quality (the post-M6 spin-off)

Label: `wayfinder:map`

## Destination

The pipeline's extraction layer measurably closes the quality gaps the M6 official eval
exposed — proven by the **two standing instruments** the parent campaign left behind:

1. the **judged-sample protocol** (deterministic-stride sample, per-regime scoring;
   baseline = sample mean **0.700**, relation-fabrication rate **58%** of relation-bearing
   Hyperedges, `M6-CALIBRATION-OFFICIAL.md`), and
2. the **frozen M6 retrieval harness** (`benchmark/official/`), whose gates must stay green —
   the retrieval gates passed DESPITE extraction noise, so every extraction gain here
   compounds on top of a passing system, and no fix may regress them.

Reached when the tickets' fixes land with a re-judged sample showing measured improvement on
the pattern each ticket targets, AND the official retrieval gates (AUC + endpoints, no point
>2% worse) still pass. Zero LLM calls in indexing or the online path — unchanged.

## Provenance of this map

Charted 2026-08-02 per [parent ticket 11](../wayfinder-hypergraph-coe/tickets/11-grilling-datagen-sota-sweep-scope.md):
Hyperedges re-project the same extracted records, so upstream data-gen improvements flow into
the hypergraph by construction — generic pipeline quality is THIS map, prioritized by the
eval's **measured** weak spots, not guesses. Opening evidence file:
[M6-CALIBRATION-OFFICIAL.md](../wayfinder-hypergraph-coe/M6-CALIBRATION-OFFICIAL.md)
(250 judged Hyperedges, MuSiQue-full export, seven error patterns ranked by graph damage).

## Notes

- **This map carries execution** (parent-map precedent). Tracker: tickets in
  `tickets/NN-slug.md`, headers `Type:` / `Status:` (open/claimed/resolved) / `Blocked by:`.
  Frontier = open + unblocked + unclaimed, lowest number first.
- **STANDING RULE (Dennis, 2026-07-31, carries over): a capability is only dropped or
  weakened when DATA says so.** Every sacrifice cites a measurement; the shipped default is
  the strong configuration.
- **Default-unchanged rule carries over**: the default pipeline stays byte-identical unless a
  ticket's fix is a bug-class correction (fabricated relations ARE wrong output — fixing them
  changes extraction output *by design*, gated behind the measured evidence, and the change is
  disclosed per ticket; the *opt-in surfaces* stay opt-in).
- Standing prefs: no fake green (real models/corpora for any claim); diagnose tight (max 2
  live validation runs per fix cycle); licence discipline (HotpotQA-derived artifacts stay
  local/gitignored — CC BY-SA; MuSiQue-derived committable with the SOURCE.md attribution).
- **Measurement caveat carried until ticket 01 closes it**: the 250 judgments are
  LLM-generated (Claude Opus, 2026-08-02, `benchmark/official-data/PROVENANCE.md`), pending
  human spot-check. Numbers derived from them inherit the judge's errors — ticket 01 puts a
  bound on that before the map optimizes against them.

## Decisions so far

*(none yet — the map is freshly charted; the parent campaign's rulings that bind here:
ADR-0057 tau_text redaction discipline, ADR-0044 PII masking, the pydantic-only rule for
`latence-retrieval`, ADR-0012 no default model checkpoints.)*

## Tickets

| # | Type | Title | Blocked by |
|---|---|---|---|
| 01 | research | Judgment spot-check + instrument freeze | — |
| 02 | grilling | Relation-fabrication policy (the dominant error) | 01 |
| 03 | task | Title-only / no-predicate span gating | 02 |
| 04 | task | Common-noun + anaphora promotion filter | 01 |
| 05 | grilling | Entity type-assignment errors | 01 |
| 06 | task | Table/infobox cell-aware segmentation | 01 |
| 07 | task | Surface-form duplicate merge (coref-lite) | 01 |
| 08 | task | Unextracted-subject probe in the Quality Report | 01 |
| 09 | task | Per-regime γ calibration as a shipped workflow | 01 |
| 10 | grilling | Chunk-size ↔ hyperedge-yield coupling | 01 |
| 11 | task | Alias + description enrichment at export | 01 |
| 12 | task | ANN-blocked disambiguation as correlation clustering (tabu local search) | 01 |

## Fog (not yet specified)

- **Per-label extraction thresholds** — GLiNER2 emits one confidence per prediction; the
  label-wise precision table (`located in` reliable, `capital of` 1/11) suggests per-label
  γ floors, but whether GLiNER2's confidences are comparable across labels is unmeasured.
  Graduates inside ticket 02.
- **Learned relation-predication verifier** (zero-LLM: small classifier over span+triple) —
  only if ticket 02's cheap structural gates leave a measured residue worth a model.
- **DE-language error profile** — the 250-sample audit is English wiki text; the enterprise
  corpus is DE/EN but template-planted. A messy German corpus judgment sample is fog until a
  real one exists.
- **Windowing across chunk boundaries in the builder** — the fix ticket 10 reaches for ONLY
  if the coupling is confirmed harmful on the official corpora (parent-map ruling: a
  measured-need change, not built on suspicion).

## Out of scope

- Retrieval-side follow-ups (weighted incidence B, heat-kernel Propagator A/B, learned
  utility classifier, Retrieval MCP chains surface) — they stay on the
  [parent map](../wayfinder-hypergraph-coe/map.md) "Not yet specified" list.
- Productization of the eval-harness pieces (exact-surface disambiguator as a shipped
  provider, tier-2 query-NER packaging) — production-direction work, tracked outside this
  map's data-gen destination. (Ticket 12 is NOT this: it is a new provider with a measured
  gate, charted 2026-08-04 from the e2e-sota-run scale wall + Dennis's
  disambiguation-as-combinatorial-optimization framing.)
- Groundedness / contradiction handling; any LLM call in indexing or online retrieval —
  excluded exactly as on the parent map.
