# Close-out audit — 2026-08-02 (paper/spec fidelity + original-briefing compliance)

Two independent adversarial audit passes over the finished campaign (main = the official-gate
state). Verdict up front: **the core holds** — the paper-verified pipeline order is genuinely
in the shipped code (`hypergraph_signals` structurally identical to the ticket-20 verified
prototype), the official gate arithmetic reproduces to the fourth decimal from the stored
result files, the big sacrifices (no global A_E, no CSR, KV-per-call, dropped
knapsack_refine) are ADR-recorded and measured, provenance labeling is exemplary, and the
pydantic-only/engine-agnostic rule held. The genuine findings, their dispositions, and what
was fixed:

## Fixed in code (this commit)

- **F-COE (spec 3.2 hard skip mis-declared as unrepresentable).** Dedup only collapses
  IDENTICAL (entity set, signature) pairs; strict-subset signatures and same-signature/
  different-entity-set records coexist (the id is addressed over both jointly). The KV solver
  fields carry no signatures (ADR-0056's deliberate minimum), so the signature-subset
  condition is unimplementable at the selector. Disposition: **declared deviation** — the
  entity-overlap discount is the shipped approximation (subset candidate → red→1.0 → marginal
  collapses to `w_conn`), and no entity-level hard skip is added (full entity containment ≠
  fact redundancy; skipping without data violates the standing rule). `coe.py` docstring
  corrected; the coexistence + discount residual pinned by a new test. Measured escalation
  path if judged data shows subset redundancy spending budget: signatures into the solver
  fields (KV format rev).
- **F-MERGE (undersized-merge could re-breach the token ceiling).** `_apply_size_targets`
  merged an under-floor span into its neighbour AFTER the oversized split without re-checking
  `max_tokens`. Fixed: the merge declines when the combined range would breach the ceiling
  (the ceiling is the declared invariant; an undersized standalone span is the lesser
  deviation). Regression test added.
- **F-HARNESS (exact-arm verification triple mislabeled).** The exact-cost A/B passed
  `estimate_tokens` labeled `tiktoken:cl100k_base`. Inert in every run that happened
  (matching identities skip the optional pass) but wrong under `verify_on_match`. Fixed:
  the exact arm now verifies with a real tiktoken counter.
- **Map/ticket drift**: subset MuSiQue AUC 0.508 → **0.5378** (transcription error vs
  `final-results-piecewise.json`; conservative direction); ticket 11 status open → resolved;
  official-gate headline now carries the tier-2/default-install caveat; "byte-identical"
  restated precisely (structural absence + verified fixture diff = schema stamp + null field).
- **recall@5 delivered** (`benchmark/official/run_recall5.py` → `recall5-results.json`) —
  ticket 08's remaining item, scored once on the pre-registered test files, no tuning.
  The numbers (paragraph-level, granularity-neutral):

  | test set | baseline BM25 | CoE tier1 (pure ρ) | CoE tier2 (pure ρ) |
  |---|---|---|---|
  | MuSiQue-full (1,208 q) | **0.4053** | 0.1297 | 0.2094 |
  | HotpotQA-full (3,702 q) | **0.6960** | 0.2191 | 0.3506 |

  Honest reading: as a STANDALONE paragraph ranker, pure-ρ is far below BM25 — ρ is an
  entry+propagation confidence signal designed to pick τ spans for budgeted composition,
  and the deployed configuration is the hybrid, which is what the recall@budget gates
  measured at parity. This also puts a number on F-VEC/F-BASE: without any dense/textual
  similarity signal (`eu_sims` never produced), the structural ranking alone is weak, and
  HippoRAG-2-class recall@5 (dense-retriever systems) is NOT matched by the zero-model
  structural path by itself. Comparability claim closed as: measured and reported, not
  competitive standalone — the compounding path is the eu_vecs producer + dense baseline.

## Declared, queued as measurements (not fixed here — each needs data or a decision)

- **F-VEC (spec 1.6 `entity_vecs`/`eu_vecs` have no producer).** Ticket 13 silently dropped
  its own scope item; consequence: `eu_sims` never exists, so the `s_hyp` prune ran at
  **β=1 (entry-overlap only)** in every official configuration — Eq 16's textual-similarity
  half was never exercised, and tier 0 was improvised in the harness (`Tier0Index`) rather
  than shipped. This is the same gap as F-BASE below and is **the single highest-value next
  measurement**: an opt-in embedding producer (adopter's model, per ADR-0012 no default
  checkpoint) feeding both eu_sims and a real dense baseline.
- **F-BASE (eval baseline deviated from its own pre-registration).** Ticket 08 pre-registered
  "dense first stage + Packer — the strongest thing we currently ship"; the delivered
  baseline is BM25→Packer, undeclared until now. BM25 is plausibly the softer bar on MuSiQue,
  i.e. the one place the harness leaned the CoE arm's way. The gates as run are honest
  against the baseline they state; the **dense-stage re-run is the production-direction
  measurement** that closes it.
- **F-METRIC (granularity asymmetry in recall@budget).** A gold paragraph counts as recalled
  via any one ≥40-char sentence; sub-paragraph τ windows earn full unit credit cheaper than
  whole paragraphs — a leniency structurally kinder to the finer-grained arm. Controls:
  paragraph-level recall@5 (now delivered) is granularity-neutral; canonical supporting-fact
  EM/F1 remains **undispositioned** — open item.
- **F-SUBSET-SCORING (adaptive second scoring on the subset gate).** Flat share failed the
  MuSiQue 1k test endpoint; the piecewise policy was then scored on the same test set.
  Disclosed at the time ("two scorings, both reported"), and the OFFICIAL run pre-registered
  piecewise in code before test contact — headline clean, subset discipline violated once,
  on the record.
- **F-6.1/6.2 (weighted incidence + heat-kernel arbitration lapsed).** Ticket 20 measured
  the cost (~12% latency, no benefit post-reorder) but the promised eval arbitration never
  ran (not among the four official A/Bs). Standing-rule residual: a declared deferral whose
  data half is incomplete.
- **F-6.3 (typed-traversal pre-propagation pass): no disposition existed.** Disposition now:
  **descoped to fog** — optional in the spec, zero demand signal from any measured eval
  finding; revisit only if a real corpus shows type-confusable expansion noise the hub cap
  doesn't absorb.
- **F-TIER2-LAT (tier-2 query-NER latency).** "Tens of ms CPU" is still an assertion; tier 2
  is the gate-winning config, so this is a required production-direction measurement.
  Cold-cache KV behavior likewise unmeasured (ticket 19's admitted residual).
- **F-1K-TIE.** The tuned policy is baseline@1k, so 1k "endpoint dominance" is a tie by
  construction — real dominance is 2k–16k. Stated here so nobody reads the endpoint line as
  a 1k win.
- **F-M1-MERGE-NEIGHBOUR (spec 1.2.5).** The undersized merge targets the *previous
  same-chunk* span, not "the neighbour sharing the most entities" — single-pass determinism
  choice, previously rationalized only in a code comment, now declared here. Unmeasured;
  folds into the datagen map's segmentation work if the judged data implicates it.

## Where the remaining risk concentrates

Extraction quality (the datagen spin-off map's whole subject) and the two unmeasured legs —
dense-stage baseline + eu_sims (F-VEC/F-BASE) and tier-2 latency (F-TIER2-LAT). Everything
else is bookkeeping now on the record.
