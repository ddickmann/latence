# Grilling: evaluation and the acceptance bars

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 06, 10

## Question

Decide how quality is proven:

- gold sets for warehouse mapping (column ↔ concept, joins, synonyms) and for the text path
  (entities, relations, merges), in English for v1 (German follows once English passes);
- whether Metis's synthetic estate and golden set are reused;
- metrics: pairwise and B-cubed merge P/R, mapping accuracy, SHACL conformance, ULTRA MRR, and
  downstream question-answering accuracy through Metis;
- baselines to beat: current Latence, and where possible the open-source auto-ontology project
  and Genie;
- the acceptance bar per Done criterion;
- how the Quality Report grows to carry them.

## Answer

Decided by the owner on 2026-11-10, one question at a time.

1. **Three layers of gold, each with its own job.**
   - *Our own seeded synthetic data* (the estate and text generators from the spike). The
     answers are exact, it scales to any size and it runs in CI. **The hard bars are measured
     here.**
   - *A small expert-labelled slice of realistic data:* a few hundred items each for record
     matching, text linking and column mapping, with part of it labelled twice to measure how
     often experts agree. It checks that the synthetic results carry over.
   - *Public benchmarks with permissive licences,* reported for comparability only.
   - Metis's synthetic estate and golden questions are not copied. They stay in Metis.
   - English only for v1; German follows once English passes.
2. **Baselines.**
   - Current Latence is the only baseline that gates. Its numbers are recorded once
     (*Record the baseline*), and the upgraded text path must beat them.
   - The open-source auto-ontology project is run on the synthetic estate for column mapping
     and reported side by side. It does no record matching and no text.
   - Genie Ontology is not compared element by element in v1: it has no export.
   - The downstream effect is measured inside Metis: its own golden questions, answered with
     and without the engine's proposals applied. Reported, not gated.
3. **Hard bars.** Three, and no others during development:
   - auto-accept precision of at least 0.95 under Learn-then-Test, with its confidence stated,
     on the warehouse side and on the text side;
   - the text path strictly better than the recorded baseline on extraction F1, linking F1 and
     documents per second;
   - **a guard against an empty pass:** on the synthetic estate the machine alone finds at
     least 60% of the true matches (pairwise recall) while holding the precision bar. For
     reference, the spike's run at about 100k records had machine-only recall of about 0.78.
4. **Reported in every run, with no bar yet:** review rate and expert minutes; how often the
   right Concept is among the top three proposals for a column, and the calibration of those
   scores; validity of the standard exports (OWL 2 QL, SHACL, SKOS); graph-completion MRR and
   Hits@k with leak detection; the share of elements with an Assertion and an Actor; replay
   determinism. Bars for these are set from measured values when v1 becomes a release
   candidate.

**The Quality Report** gains one versioned section per item above: resolution (bands, realised
precision with its interval, recall, review rate, expert minutes), mapping, export validity,
graph completion and evidence coverage. Each is a `QUALITY_SCHEMA_VERSION` change and lands
with the slice that produces the numbers. This follows from the answers and was not asked
separately.

## From the map (moved 12 Nov)

[Evaluation and the acceptance bars](issues/18-evaluation-and-acceptance.md) — three layers of gold: our own seeded synthetic data (where the hard bars are measured), a small expert-labelled realistic slice, and public benchmarks for comparability. Current Latence is the only gating baseline; the Metis effect is measured inside Metis and reported. Three hard bars: precision at least 0.95, the text path better than the baseline, and machine-only recall of at least 0.60 on the synthetic estate. Everything else is reported until the release candidate.
