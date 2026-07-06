# Profiling is a corpus-level Stage emitting scoped FeatureRecords over a fixed feature set, streaming and with the enricher registry recast away

S7 adds **Profiling** (CONTEXT): the corpus-level Stage that computes statistical and
quality features (density, readability, Zipf, cross-document entity frequency,
co-occurrence). Four design questions had non-obvious answers; they extend ADR-0004/0007
(the Capability + CPU-first-reference pattern) and mirror how ADR-0022/0023/0024 handled
S4/S5/S6, with the concrete Profiling contract, the corpus-level multi-input seam, the
fixed feature set, and the streaming discipline.

## A FeatureRecord is a per-scope inter-Stage Record, not a side-car report

Profiling could write its features directly into the Quality Report and emit nothing on
the DAG. It does neither exclusively. `FeatureRecord` is its own versioned core contract
(`SCHEMA_VERSION` → 7), a subclass of `Record` — so it carries Provenance and
Classification like every other inter-Stage carrier, threads through the DAG as a Stage
output, is checkpointed/resumed uniformly, and exports to the corpus directly (the
feature set is an AI-ready artifact a downstream consumer reads, not merely a run log).

**Why scoped, not two contracts.** Profiling is corpus-level, so it produces features at
two granularities: **per-document** (density, readability, Zipf α, compression ratio,
structure — one record per parsed document) and **cross-document** (entity frequency,
co-occurrence, type consensus, source coverage — one record for the whole run). Rather
than two record types, a single `FeatureRecord` carries a `FeatureScope` tag
(`DOCUMENT` / `CORPUS`) and exactly one populated payload (`DocumentFeatures` /
`CorpusFeatures`). A consumer reads one exported stream and filters by `scope`; the DAG,
the checkpoint, and Export all treat one carrier. The scope/payload consistency is
contract-enforced (a `DOCUMENT` record must carry document features and a
`document_record_id` and no corpus payload; a `CORPUS` record the inverse), so a Provider
cannot emit a mis-scoped record. The single `CORPUS` record's Provenance is a synthetic
corpus-scope lineage (`latence:corpus`, never a real file) — it satisfies the required
`Record` Provenance while making the corpus scope unambiguous, and the Runner excludes it
from the run's source-document count.

## Profiling is corpus-level and multi-input, off Parse (+ optionally Entity Extraction)

Profiling depends on **Parse** for the `DocumentRecord`s — the per-document features and
the corpus text aggregates are computed over the document's *assembled markdown* at its
original-document offsets, the same coordinate system every other Stage uses, so it scans
the parsed document, not the markup-stripped chunk stream (the identical argument ADR-0024
made for Redaction). It **optionally** depends on **Entity Extraction** for the run's
`EntityMention`s — the cross-document entity frequency, type consensus, and co-occurrence
features. Unlike Relation Extraction (ADR-0023), the mention side is *not* required: a
Parse-only Profiling pipeline is valid and yields the corpus text/source-coverage features
with empty entity aggregates. The Runner partitions the gathered inputs by carrier and
errors only if a non-empty input carries no `DocumentRecord` (a mis-wired Parse edge).

## One fixed, closed feature set — the 9-enricher registry is recast away

The private `dataset-intelligence` stack computed features through a dynamic **registry**
of nine enrichers (PORTING-MAP.md: "9 enrichers as a registry → consolidate; the registry
is heavier than a fixed set needs"; ASSET-INVENTORY.md S7: "consolidate into one
`document_features` + `corpus_features` module … RECAST (simplify registry)"). We port the
*algorithms* faithfully — density, a dependency-free Flesch-style readability normalised to
`[0, 1]`, a least-squares Zipfian-α fit, a zlib compression ratio as a redundancy proxy,
plus the cross-document entity frequency / co-occurrence / type-consensus / source-coverage
aggregates — but drop the registry indirection. The feature set is a **fixed, closed**
Pydantic schema (`DocumentFeatures` / `CorpusFeatures`), not an open bag keyed by
plugin-registered names. This keeps the schema stable and auditable, keeps a seeded run
byte-identical (the Baseline bar), and removes a layer of configuration surface that bought
nothing for a fixed v1 feature set. A future feature is a schema addition (a version bump),
not a registry entry — a deliberate trade of extensibility for a stable, verifiable
contract, the same "framework owns the seam" stance as the rest of the spine.

## The feature pass is streaming / spill-to-disk friendly, never whole-corpus-in-RAM

The S7 acceptance criterion requires the feature pass to make no whole-corpus-in-RAM
assumption (PORTING-MAP.md keeps the private stack's "feature disk-spilling under memory
pressure" specifically for this Stage). So the engine's cross-document aggregation is a
streaming `CorpusAggregator`: documents are folded in one at a time and only bounded
aggregate state is held (frequency counters, coverage tallies, a running corpus
vocabulary, per-pair document sets) — never the corpus text. The reference Provider drives
it from a generator over the document stream, so corpus text is processed and released
document by document. The aggregation is commutative and every finalised map is sorted, so
the result is order-independent and byte-identical for a seeded run regardless of document
arrival order.

## Providers shipped (CPU-first reference, ADR-0007; no new model, no new license)

* **`profiling.statistical`** ships in `latence-core`: a pure-Python, dependency-free,
  deterministic corpus profiler wrapping the consolidated enricher engine
  (`latence_core.profiling`). It computes the full fixed feature set on CPU with no model
  download (no tokenizer, no embeddings — the Zipf fit and readability are closed-form),
  so first `git clone` → working profiling is hash-stable for the Baseline bar. Profiling
  needs no learned model in v1, so — unlike S4/S5/S6 — there is **no heavy Provider package
  and no new weights/code license to verify** (ADR-0012); a future embedding-based feature
  (semantic density, topic coverage) would be a post-v1 Provider behind this same
  `Profiler` seam.

The feature summary lands in the Quality Report as `ProfilingQuality`: per-document feature
averages (readability / density / diversity / Zipf / compression) plus the corpus
aggregates (word / vocabulary counts, corpus Zipf α, source coverage, entity-type
frequency, top co-occurrences) — counts and averages only, no raw document text — satisfying
the S7 "feature summary lands in the Quality Report" criterion.
