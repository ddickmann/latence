# 12 — Task: M1 — Hyperedge segmentation + builder

Type: task
Status: resolved
Blocked by: 03, 04

**CHECK IN with Dennis before starting — first long-running implementation on this map.**

## Question

Implement spec 1.2–1.5 as a **Graph Assembly** extension per
**[ADR-0057](../../../docs/adr/0057-hyperedge-is-a-third-graphscope-emitted-by-graph-assembly-and-tau-text-follows-corpus-redaction.md)**
(domain model) and **[ADR-0056](../../../docs/adr/0056-hypergraph-evidence-sidecar-is-read-per-call-over-a-point-lookup-kv-no-global-edge-adjacency.md)**
(what the store does and does not hold). Terminology: **Hyperedge**, never "Evidence Unit"/"EU".

Scope:

1. **Persist per-mention link confidence** through Disambiguation (glinker computes it, it is
   currently dropped — [ticket 01](01-research-spec-vs-repo-audit.md)). Prerequisite: without it
   `gamma` has no data source. Do this first; it is a contract change.
2. **Sentence splitting** — abbreviation-safe DE/EN. Today's is a naive `[.!?]\s` regex
   (`chunking.py:226`); pysbd (MIT) or a spaCy senter (MIT for de/en pipelines — **not** the
   CC BY-SA multilingual one, [ticket 02](02-research-external-landscape-and-licences.md)).
3. **Hyperedge segmentation** — window extension driven by relation head/tail argument offsets
   (recoverable by joining `RelationMention.head_mention_id`/`tail_mention_id` to `EntityMention`
   spans; no schema change needed), minimal-cover spans, merge/split to the 40–120 token target,
   weak Hyperedges behind the config toggle, `m_max=12` with relation-group splitting.
4. **`GraphHyperedge` payload + `HYPEREDGE` GraphScope** on `GraphRecord`; `SCHEMA_VERSION` bump;
   scope/payload consistency validation.
5. **`tau_text` follows the corpus redaction variant**, enforced at the writer, span kept in
   original coordinates (ADR-0057). Offset alignment via the ADR-0031 `offset_map` pattern.
6. **`gamma`** from persisted link confidence + relation confidence + weakest-mention penalty;
   calibration itself is [ticket 09](09-grilling-calibration-ownership.md)'s call — uncalibrated
   weighted raw scores are acceptable for M1.
7. **Dedup/merge** on identical entity-set AND relation signature, `support` counter, highest-γ
   instance's `tau_text` kept primary.
8. **Quality Report**: hyperedge counts, weak fraction, arity distribution, **token-cost
   distribution** (the one that predicts whether budgeted selection will behave).

Tests (spec M1 + the ADR-0057 additions): every relation's arguments fall inside its Hyperedge span;
no Hyperedge exceeds 180 tokens; merge is idempotent; **`tau_text` never contains a span the exported
corpus masked** (the PII-bypass regression test); deterministic ids are stable across runs.
Repo DoD bar applies.

## Answer

**BUILT + green on branch `hypergraph-m1`.** Full-suite regression proof: baseline before the
change was **4644 passed / 1 failed** (the failure was ADR-count drift on front-door pages from the
wayfinder ADR commits — repaired); after: **4661 passed / 36 skipped / 0 failed**, `ruff` clean,
`mypy --strict` clean over all 147 typed sources. 16 new tests.

What landed (all additive; a stack that never configures hyperedges is byte-identical to v19 apart
from the schema_version number — proven by fixture diff and the runner e2e off-test):

1. **Contracts v20**: `GraphHyperedge` + `HyperedgeRelation` + `HyperedgeSourceRef`,
   `GraphScope.HYPEREDGE`, three-way scope/payload validator,
   `CanonicalEntity.member_link_confidences`. Token-counter identity stored beside `token_cost`
   (ADR-0058).
2. **Link confidence persisted** end-to-end: glinker `MentionRecord.link_confidence` ←
   `LinkedEntity.score` (previously dropped at `_apply_linking_results`) → `CanonicalEntityRecord`
   → core `ResolvedCluster` → `CanonicalEntity`. Missing = unknown, never 0.0.
3. **`latence_core.sentences`**: zero-dep abbreviation-safe DE/EN `RuleSentenceSplitter` behind a
   `SentenceSplitter` protocol (12 trap cases pinned: `z. B. am 3. Mai`, `ca. 7 Mrd. Euro`,
   `Acme Inc. acquired Beta Ltd.`, decimals, initials, ellipsis, markdown breaks).
4. **`latence_core.hyperedge`**: segmentation (relation-seeded windows ≤3 sentences, overlap
   merge, weak units behind toggle), size targets (merge <25 / split >180 at min-cut-relations
   boundary), arity cap 12 via relation-group union-find split, corpus-wide dedup on entity-set +
   relation-signature with `support` + `merged_sources`, γ renormalised over available signals
   with the 0.5 weak discount, **`tau_text` in the corpus redaction variant enforced at the
   writer** (PIISpan substitution; TAG intact), content-addressed ids over entity set + signature
   (NOT span position).
5. **Wiring**: `CanonicalGraphAssembler` config `hyperedges:` (true or policy overrides) + the
   `wants_document_streams` opt-in gate; the Runner feeds chunk/mention/relation streams only to
   an opted-in Provider — no `GraphAssembler` protocol change, no fake edits.
6. **Export**: `graph-hyperedges.parquet` (solver columns first-class) only when the projection
   ran. **Quality Report**: `graph.hyperedges` section (counts, weak fraction, arity histogram,
   token-cost distribution) — `None` otherwise.
7. **Tests** ([test_hyperedge.py](../../../packages/latence-core/tests/test_hyperedge.py)):
   relation-args-in-span, token ceiling, dedup idempotency + support, **PII-bypass regression**
   (masked span never in `tau_text`; raw corpus ⇒ raw `tau_text`), determinism, weak toggle,
   arity split, splitter traps, byte-identity off-path, and TWO e2e runs through the REAL
   LocalRunner (on: parquet + report section; off: no file, no section, NODE/EDGE-only stream).

Deviations from ticket text, stated: sentence splitting is the in-core rule splitter behind a seam
(pysbd/spaCy senter = a Provider-package adapter later — thin-core ADR-0016 keeps the dep out of
core); relation argument offsets resolved via the head/tail mention join as planned (no schema
change needed). Follow-ups already tracked: exact operator-named token counter injection lands with
M2 (ticket 13); calibration stays ticket 09.

## Close-out audit addendum (2026-08-02)

Two findings against this milestone, see [CLOSEOUT-AUDIT.md](../CLOSEOUT-AUDIT.md): the
undersized-merge could re-breach the token ceiling (FIXED — merge now declines over-ceiling
absorbs; regression test added), and the merge targets the previous same-chunk span rather
than spec 1.2.5's most-shared-entities neighbour (now DECLARED; unmeasured, folds into the
datagen map's segmentation work if implicated).
