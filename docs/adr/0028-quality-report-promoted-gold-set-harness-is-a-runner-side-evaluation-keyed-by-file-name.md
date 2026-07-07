# The Quality Report is promoted to the full first-class artifact; the gold-set harness is a Runner-side evaluation keyed by file name, not a DAG Stage

S10 promotes the **Quality Report** (CONTEXT) from the skeleton that grew slice-by-slice
(S1–S9) to the full first-class artifact ADR-0008 called for: contract-completeness now
covers *offsets aligned* (not just Provenance/Classification present), a corpus-level drift
roll-up surfaces Provenance-integrity across every derived offset, the report renders a
human-readable summary alongside its machine-readable JSON, and — the headline — a
**gold-set harness** reports per-Stage precision/recall/F1 on the adopter's OWN corpus. Two
design questions had non-obvious answers.

## The gold-set harness is a Runner-side evaluation over the report inputs, not a Stage/Capability

Every prior slice added a Stage with a Capability protocol, a reference Provider, and a
scoped inter-Stage `Record` that flows through the DAG (S4 EntityMention, …, S9 GraphRecord).
The gold-set harness is deliberately **not** one of those. It does not consume or produce a
carrier the DAG threads; it does not run as a node in topological order; there is no
`Scorer` Capability and no `latence.providers` entry point for it. It is a pure, deterministic
evaluation the Runner performs *while building the report* — it reads the same extracted
mentions/relations the report already aggregates, matches them against the adopter's
annotations, and writes the scores into the Quality Report.

**Why.** A gold set is *reference data about the run*, not *data the pipeline transforms*.
Modelling it as a Stage would force artificial questions — what does the "scoring Stage"
emit downstream? where in the DAG does it sit relative to Export? — that have no honest
answer, because scoring is orthogonal to the pipeline's data flow. ADR-0008 already frames
the harness as part of the Quality Report artifact ("an *optional* gold-set harness lets an
adopter drop in 20–50 annotated documents and get precision/recall per Stage"), not as a new
pipeline capability. Keeping it Runner-side also means it recomputes identically on a resumed
run (from the reloaded checkpoint records) with no extra checkpoint of its own, and a
pipeline with no gold set pays exactly nothing. The scoring itself lives in a pure
`latence_core.goldset` engine module (like `graph.py` / `disambiguation.py`), so it is unit-
testable in isolation and reusable, while the Runner owns only the wiring.

**Rejected**: a `Scorer` Capability + Provider (invents a Stage with no carrier and no DAG
position); writing scores to a separate side-car file (splits the auditable "AI-ready"
substantiation away from the report that ADR-0008 says *is* that substantiation).

## Annotations bind by source file name; matching is offset-aware, label-sensitive, one-to-one

The gold set is a versioned Pydantic contract (`GoldSet` → `GoldDocument` → `GoldEntity` /
`GoldRelation`), authored as JSON or YAML (one-schema/dual-serialization, ADR-0006) and
loaded from Storage via an optional `gold_set_uri` on the Pipeline (`PIPELINE_SCHEMA_VERSION`
→ 2; additive, backward-compatible). Each `GoldDocument` is keyed by **`file_name`**, not the
content-addressed `document_id` the run computes — the file name is the stable key the adopter
knows when authoring annotations, and the harness resolves the run's records back to it through
their Provenance. Annotations a document the Source never read simply go unscored, and the gap
(`documents_matched` < `documents_annotated`) is visible in the report.

Matching is **deterministic** (Baseline bar): a prediction matches a gold annotation when
their labels are equal AND their character spans overlap by at least a configurable Jaccard
threshold (`match_min_overlap`; 0.0 = any overlap). Entity assignment is a **maximum-cardinality**
one-to-one matching (augmenting paths / Kuhn's algorithm, with deterministic tie-breaking) —
never a single-pass greedy, which would under-count precision and recall by letting one prediction
steal a gold span a later prediction was its only match for (#76). Relation assignment is likewise
one-to-one and maximum-cardinality over the pairs whose *both* endpoints resolve. Each gold
annotation and each prediction is consumed at most once, so precision and recall are well-defined
and never double-count, and the matcher never leaves a matchable pair unmatched. A relation matches only when its
label matches AND *both* endpoints resolve (by span + label) to the gold relation's head and
tail entities — a label-only match is not credited. Scores are computed overall and per label;
every derivation guards its empty denominator (no predictions ⇒ precision 0.0; no gold ⇒
recall 0.0), so the report never divides by zero. The result is byte-identical across seeded
runs.

## Completeness gains offset alignment; drift is a cross-Stage roll-up

`CompletenessCheck` now audits that every offset-bearing record's char span is well-ordered
and, when it carries page coordinates, that those are well-ordered too — the "offsets aligned"
half of the S10 contract-completeness AC. Page *presence* is not re-required here: the
sub-document carriers whose contracts already require a page span (EntityMention /
RelationMention) guarantee it at construction, while a whole-document carrier legitimately
holds a `[0, len)` char span with no page span. Separately, a corpus-level `DriftDiagnostics`
rolls up the ported `PageOffsetIndex` drift across every page-map-bearing chunk (a finer,
cross-Stage view than the Parse-only page-boundary audit): `clean` means every derived offset
resolves exactly to its source page. Both are counts only (no raw text), safe to persist.
