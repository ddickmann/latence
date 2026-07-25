# Type-vocabulary canonicalization is its own Capability in the resolution phase

**Status: accepted** — extraction redesign, T3. Adds the `TypeConsolidator` Capability + an OPTIONAL
corpus-level `type_consolidation` Stage (two in-core reference Providers, `type_consolidation.cascade`
and `type_consolidation.exact_surface` — pure-Python, deterministic, CPU, zero-dep), the
`TypeVocabulary` / `CanonicalType` contracts, the additive `EntityMention.raw_label` /
`RelationMention.raw_label` provenance fields (SCHEMA_VERSION 17), a persisted per-run vocabulary
sidecar, and a `type_consolidation` Quality-Report section (QUALITY_SCHEMA_VERSION 20). Amends
ADR-0038 (schema induction) and ADR-0046 (the GLinker resolver). Builds on ADR-0004 (Capability
protocols + Provider plugins), ADR-0007 (CPU-first), ADR-0012/0016 (no model, no extra deps in core),
ADR-0017 (files are the source of truth), ADR-0026 (the audited resolver cascade this mirrors),
ADR-0034 (typed errors, no silent failure), ADR-0035 (a second CPU reference Provider per seam),
ADR-0036 (profile + conformance + bake-off).

## Context

The extraction pipeline is being **inverted**. It used to induce ONE corpus-wide label schema, cap it
at ~40 types per slot by document-frequency salience, and apply that whole schema to EVERY chunk at
extract and redact time. That design bought consistency, and paid three prices for it:

1. the cap **silently dropped rare-but-critical labels** — precisely the ones an enterprise corpus
   cares about;
2. scoring every chunk against ~35 irrelevant labels **degraded zero-shot GLiNER2 precision**;
3. the label dimension was the root cause of a **CUDA OOM** (batch × ~768 tokens × ~40 labels → a
   ~17 GiB span-scoring tensor).

The redesign makes the whole front half of the pipeline chunk-local: labels are induced **per chunk**,
uncapped, so each chunk is scored against a small set of labels that are actually relevant to it.

That fixes all three, and creates one new problem: **type drift**. With nothing forcing a single
schema upstream, chunk A induces `org`, chunk B `organization`, chunk C `company` — three names for
one type. A knowledge graph built from those mentions has three node types where it should have one.

Nothing in the pipeline canonicalizes types today. The `disambiguation.glinker` Provider resolves
entity **mentions** into canonical **entities** (a self-referencing KB with a dominant-label vote, then
an audited exact/alias/acronym/substring/fuzzy/embedding fusion cascade). A grep across
`packages/*/src` for any corpus-wide type consolidation returns nothing. So the "cleaning phase" cleans
entities and leaves their types drifted — which means per-chunk induction, shipped as-is, would trade
an extraction bug for a KG-consistency regression.

The architectural principle the redesign rests on is: **consistency is EARNED DOWNSTREAM, never
IMPOSED UPSTREAM by capping.** Entities are resolved corpus-wide, after extraction has captured
everything. Types must be resolved the same way, in the same phase, for the same reason.

## Decisions

### 1. Type canonicalization is its OWN Capability, not a second job bolted onto the disambiguator

`TypeConsolidator` is a new Capability with its own `type_consolidation` Stage kind, sitting beside
`Disambiguator` in the resolution phase rather than inside it.

The alternative — extend `disambiguation.glinker` — was rejected. Resolving *which entity a mention
refers to* and resolving *what a type label should be called* are two different decisions over two
different vocabularies. Fusing them would make an already-large Provider larger without making it
deeper; it would tie a stack's ability to canonicalize types to its choice of entity resolver (an
adopter on `disambiguation.exact_surface` would silently lose type consolidation); and neither half
could then be swapped alone. One Capability per decision that genuinely varies is the ADR-0004/0016
discipline, and this decision genuinely varies — the two shipped Providers implement materially
different merge postures.

The Capability is a **corpus-level mention→mention transform**: it consumes the run's
`EntityMention`s (required — their labels *are* the vocabulary) and optionally its
`RelationMention`s, and returns the SAME records with `label` rewritten to canonical form, plus the
`TypeVocabulary` that defines the mapping. Because the output carrier is the input carrier, it
composes into the DAG with no new plumbing: a Disambiguation Stage simply depends on it instead of
on the extractor.

### 2. It runs BEFORE entity resolution, not after

The entity resolver votes a cluster's type from its members' labels, and gates its low-precision
rungs (acronym, substring) on **type compatibility**. Feeding it a drifted vocabulary therefore does
double damage: `org` vs `organization` reads as a type mismatch, so clusters that belong together are
split, and the clusters that do survive are typed by a vote over inconsistent labels.

Canonical types in ⇒ consistent entities and a consistent graph out. The reverse order would require
the resolver to be re-run after consolidation to benefit from it.

### 3. The mechanism is the resolver's audited cascade, applied to type STRINGS

`type_consolidation.cascade` mirrors `latence_core.disambiguation`'s design over a different
vocabulary — same discipline, so an operator reads one mental model:

| rung | folds | why it is gated |
| --- | --- | --- |
| `normalized` | equal after case/punctuation/plural normalization (`ORG`, `org.`, `Orgs`) | always on; pure surface drift is never a real distinction |
| `alias` | an operator-supplied `{label: label}` map | always on; the escape hatch, and the correction loop (see §5) |
| `acronym` | a single token equals the initials of a multi-word label (`poc` ↔ `point of contact`) | needs ≥2 words and ≥2 characters |
| `prefix` | a SHORT single token abbreviates a MUCH longer one (`org` ↔ `organization`) | both `max_abbreviation_len` (5) and `min_expansion_ratio` (2.0) must hold — an ungated prefix rule merges `product`/`production` |
| `embedding` | cosine over the type strings themselves (`org` ↔ `company` — a synonym no lexical rule can see) | OFF unless a stack wires an Embedder; threshold 0.86, deliberately strict |

The embedding rung **consumes the framework's existing `Embedder` seam** rather than inventing a
model dependency: the enterprise-SOTA stack already loads Granite r2 for the resolver, and embedding
a few dozen type strings against it is a negligible marginal cost. Core stays torch-free at import
(ADR-0016) — the rung is off by default and resolves its Provider lazily.

Canonical election is a **salience vote**: the cluster's most-observed label wins, with total,
deterministic tiebreaks (more tokens, then more characters, then lexicographic). This is the same
dominant-label vote the entity resolver uses for an entity's type.

Two Providers ship, per ADR-0035, and they differ in behaviour rather than in configuration framing:
`type_consolidation.cascade` runs all five rungs; `type_consolidation.exact_surface` runs only
`normalized` + `alias`, and cannot be configured out of that. The latter is the high-precision
baseline a regulated adopter chooses when the cost of a wrong merge (every node of a type silently
retyped) outweighs the cost of residual drift.

### 4. No LLM canonical-naming pass by default

T3 lists an optional LLM pass to pick the human-facing canonical label per cluster. It is
deliberately **not** wired as a default, and the reasoning is recorded in the Provider docstring so
it is not silently re-litigated:

* The corpus's own dominant usage already answers "what should this cluster be called" — that is what
  the salience vote reads. If a corpus overwhelmingly writes `organization`, that is the corpus's
  name for the type.
* An LLM call here adds a network dependency, a second model and a determinism risk to a decision
  that is currently deterministic and offline.
* Crucially, the decision is already **reviewable**: the vocabulary is emitted as data on the Quality
  Report, and an operator who prefers `organization` over `org` sets `aliases` and gets it —
  permanently, reproducibly, and visibly. A review loop beats a hidden model.

The seam is not foreclosed: a Provider that names clusters with an LLM is a drop-in behind the same
Capability, and would inherit the same records/vocabulary contract.

### 5. Auditability: the raw label survives on the record, and the alias map lands on the report

`EntityMention` and `RelationMention` gain an additive, defaulted `raw_label: str | None`. When the
Stage rewrites `label`, the extractor's original induced type is kept here — so every remap is
inspectable on the record itself, and a KG node can always be traced back to the surface the model
actually produced. `raw_label is None` means "never remapped", so it carries information rather than
noise, and a stack with no Type-Consolidation Stage is unchanged apart from an explicit `null`.

The Quality Report gains a `type_consolidation` section: **"N raw induced types → M canonical types"**
plus the full `alias → canonical` map, for entities and relations separately. It is computed **from
the emitted records** (`raw_label` next to `label`), not from the Provider's self-report — so the
report measures what happened to the data and cannot claim a consolidation the records do not show. A
Stage that silently failed to remap would honestly report `N raw → N canonical` and an empty map.

### 6. The vocabulary is persistable and MERGEABLE, for incremental delta runs (T4)

`TypeVocabulary` is a plain serializable model (not a `Record` — it describes the corpus's schema,
not a span of a document, and has no Provenance to carry). The Runner writes it to a per-run sidecar
`…/type-vocabulary/<stage>.json` and exposes `read_type_vocabulary(...)`, which is what makes it
survive the run (ADR-0017).

`TypeVocabulary.merge` is deliberately **append-don't-churn**: when a later run's cluster overlaps an
existing one, the EXISTING canonical label is kept and the new labels join as aliases; genuinely new
types are appended. Renaming an existing canonical label would silently retype every node already in
the KG, which is exactly the churn a delta run must not cause. The result is ordered by
`(kind, canonical)` with sorted alias lists, so merging is byte-stable.

### 7. Entity and relation vocabularies are canonicalized together but never mixed

An entity type `organization` and a relation type `organization of` are different vocabularies;
folding one into the other would corrupt the graph. Every lookup is keyed by `(kind, label)`, the two
vocabularies cluster independently, and they take separate alias maps.

### 8. Genuinely optional; off-by-default output is byte-identical

A stack with no `type_consolidation` Stage is unchanged: `raw_label` defaults `None`, no label is
rewritten, and the report section is absent. Even *with* the Stage, a record whose label is already
canonical is passed through by identity rather than copied — so a run whose vocabulary consolidated
nothing produces byte-identical records to a run without the Stage.

## Consequences

* Per-chunk induction (T1/T2) becomes safe to default: the recall/precision/OOM wins are kept, and the
  KG-consistency regression they would otherwise cause is resolved downstream.
* The enterprise-SOTA wizard profile gains a `consolidate_types` Stage between extraction and
  `disambiguate`, with the semantic rung on, riding the embedder block the resolver already declares.
  It becomes the disambiguator's sole mention/relation source, so drifted labels cannot reach the KG
  by a side path.
* Determinism is preserved end to end: sorted label processing, Union-Find with a lexicographic root,
  a total-order canonical election, and records re-emitted in input order.
* T4 (incremental delta reconciliation) has the durable, mergeable artifact it needs, with the
  no-rename guarantee already enforced by the contract rather than by convention.
* A future LLM-naming or learned type-clustering Provider is a drop-in behind the same seam.
