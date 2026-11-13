# Latence is re-scoped into a semantic-resolution and ontology engine that runs standalone and as a sidecar to Metis

**Status:** Accepted (2026-11-08)

Latence stops being a "messy documents → AI-ready data" framework. It becomes the engine that
builds a **traceable semantic and ontology layer**: canonical concepts, their names and aliases,
the relationships between them, and the bindings that tie them to physical data. Every element
carries Evidence and is reproducible. The engine runs **standalone** and as a **sidecar to Metis**,
whose semantic catalog consumes its output through a declared seam. **Warehouse metadata —
Databricks Unity Catalog and metric views first — is a new input modality** next to
unstructured text. The text path stays, with an extraction stack to be re-selected. Graph
Completion (ADR-0037, the ULTRA scorer) stays. Query-time retrieval tooling is removed (ADR-0068).

The detailed design is charted as a wayfinder map at `.scratch/semantic-sidecar/map.md`. This ADR
records only the direction, which the owner decided on 2026-11-08. Everything the map still has
to decide is named under *Open* below and is not decided here.

## Context

Enterprises building a semantic layer on Databricks have Unity Catalog and metric views as the
place where definitions and rights live. Databricks' own context layer (Genie Ontology, in
preview) builds a layer from modelled and inferred sources, but how it reaches a definition
cannot be traced. Projects that exist to build the semantic and ontology layer need an asset
where every concept, alias, merge and relationship can be traced to its sources, reviewed,
and replayed.

Latence already owns the hard half of that, behind stable Capability seams:

- the audited resolver cascade with Union-Find and its no-silent-over-merge log (ADR-0026,
  ADR-0046);
- type-vocabulary canonicalisation (ADR-0054) and relation normalisation;
- the Evidence contract, kept separate from Provenance;
- content-addressed graph ids and the asserted-vs-inferred edge split (ADR-0027, ADR-0037);
- incremental corpus deltas with exact recompute, retraction and purge (ADR-0018, ADR-0029,
  ADR-0061);
- the Quality Report and gold-set harness (ADR-0008, ADR-0028);
- schema induction (ADR-0038).

What Latence lacks for this job is just as clear. Its contracts assume files and character
offsets (Provenance requires file fields, mentions require spans). It has no warehouse input,
no ontology schema level (`TypeVocabulary` is a flat alias map), no persisted human decisions,
and no standard ontology export. These are the map's work.

Metis, the agent harness this engine serves, already drafts part of a semantic layer from
tables: profiling, header folding, code lists, key and join discovery, and owner approval. It
deliberately leaves out meaning-level folding, synonym proposals and documents. The owner's
standing rule for Metis on Databricks also binds this engine when it feeds Metis: Unity
Catalog is the semantic source of truth, and anything else is a minimal overlay keyed on UC
object ids, never a parallel definition.

## Decision

1. **Purpose.** The engine builds a semantic and ontology layer from warehouse metadata and
   from unstructured text. Every element is traceable to Evidence. "AI-ready data" remains a
   by-product, not the product.
2. **Two deployment shapes, one engine.** It runs standalone, with its own Runner, Storage and
   outputs, and as a sidecar to Metis. The sidecar shape adds an interface; it does not fork
   the engine. That interface is open (map ticket *Boundary with Metis*).
3. **Warehouse ingestion is a new first-class input modality.** Databricks Unity Catalog and
   metric views come first. Other warehouses follow behind the same seam.
4. **The text path stays and is upgraded.** Parse → Chunk → Entity/Relation Extraction →
   Disambiguation → Type Consolidation → Graph Assembly is kept. The extraction Providers are
   re-selected empirically against a recorded baseline (ADR-0013's rule, applied again).
5. **Graph Completion stays**, with the inferred-edge contract of ADR-0037 unchanged.
6. **The existing seams stay the architecture.** Capabilities and Providers (ADR-0004), a thin
   core (ADR-0016), Evidence on every inference, determinism, and no fake green. New behaviour
   enters as new Capabilities and Providers, not as special cases in core.
7. **Retrieval tooling is removed** (ADR-0068).

## Alternatives rejected

- **Build a new engine and treat Latence as a quarry only.** This was rejected because it
  throws away tested seams, gates and the audited resolver, for no gain in cleanliness that the
  map's carve-out cannot also deliver.
- **Build the semantic layer inside Metis.** This was rejected because the asset must also
  serve projects that do not run Metis. Metis also deliberately keeps meaning-level resolution
  and documents out of its layer job.
- **Adopt Databricks Genie Ontology or NVIDIA Auto Ontology.** Genie Ontology was rejected
  because it is opaque. Auto Ontology was rejected because its own issues record no
  per-element provenance and curation that is lost on rebuild, and because it reads no
  documents. Their ideas stay inputs to the map's research.

## Consequences

- ADR-0005's v1 scope and ADR-0017's "RAG corpus plus KG" deliverable no longer describe the
  product. They stay as the record of what v1 was. The new deliverable is fixed by the map.
- Contract generalisation (a structured locator next to character offsets, and an actor on
  every decision) is coming. It will bump `SCHEMA_VERSION`.
- Front-door documents (README, docs index, ROADMAP) describe the old product until the map's
  docs slice rewrites them. Doc-truth tests keep any numeric claim honest in the meantime.

## Open (decided on the map, not here)

The domain model and done criteria; the boundary with Metis and the sidecar interface; the
warehouse pipeline shape; the resolution engine upgrade; ontology induction and output
standards; the text-path model choice; the review and versioning lifecycle; runtime, storage
and deployment; evaluation and acceptance bars.
