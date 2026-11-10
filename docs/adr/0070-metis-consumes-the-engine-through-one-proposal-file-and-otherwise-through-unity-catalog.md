# Metis consumes the engine through one proposal file, and otherwise through Unity Catalog

**Status:** Accepted (2026-11-09)

The re-scoped engine (ADR-0067) runs standalone and as a sidecar to Metis. This record fixes the
boundary between the two: one Metis-specific interface, a reviewable proposal file, and nothing
else. Everything else the engine produces reaches Metis the way it reaches any other consumer,
through Unity Catalog. The owner decided this on 2026-11-09 (map ticket *Boundary with Metis*).
Nothing here is built yet.

## Context

Metis answers questions over governed data. It owns a semantic catalog, a layer job that
profiles sources and discovers and verifies keys and joins, and a per-use-case library of
synonyms, usage notes and declared readings that its data team reviews. It imports a platform's
own semantic definitions through an importer port, which keeps only definitions that rest on a
dataset kind one of its backends compiles (today a metric view) and never certifies a file
import.

The engine produces different things: resolved Entities and their crosswalk, an ontology of
Concepts, cross-language synonyms, and evidence for all of it (ADR-0069). Some of that changes
what Metis's library should say; most of it is data and vocabulary that belongs in the catalog
for every consumer. ADR-0069 also put drafting new measures in scope, which overlaps with the
measure candidates Metis proposes itself.

## Decision

1. **One Metis-specific interface: a proposal file.** The engine writes glossary content
   (synonyms, usage notes, declared readings, each with its evidence) in the proposal shape
   Metis's own proposal tooling defines: one reviewable file per use case, diffed against that
   use case's library. Metis's data team reviews it and Metis applies it. Nothing applies
   without review, and Metis's core does not change.
2. **Everything else goes through Unity Catalog.** The Entity crosswalk and the ontology are
   published as Delta tables keyed on Unity Catalog object ids. Metis reads them like any other
   source. They are catalog outputs, not a Metis interface.
3. **No shared code.** The engine never imports Metis. Standalone, it builds only the profiling
   that resolution needs. When Metis is present, the engine reads Metis's verified keys and
   joins as Assertions, with the Metis layer job as the Actor, instead of deriving them again.
4. **One route per source for new measures.** Metis keeps proposing measures from questions
   and SQL usage. The engine proposes measures only from what Metis does not read: documents and
   the ontology. With Metis present, the engine's drafts go into Metis's existing candidate
   route, so a reviewer never sees two proposals for one measure. Standalone, the engine writes
   the Unity Catalog proposal itself. Either way a human applies it upstream.
5. **The Unity Catalog rule binds the interface.** Everything Metis consumes is keyed on Unity
   Catalog object ids. A change the catalog should own goes upstream as a proposal a human
   applies, and Metis sees it only once it is in the catalog.

**Ownership.** The engine owns resolution, the ontology, the text path, cross-language
synonyms, meaning-level folding and the ledger. Metis owns answering, compiling and serving,
its verified keys and joins, the review of its own library, and certification.

## Alternatives considered

- **An importer plugin for Metis.** It is the port built for external definitions. Rejected for
  v1: the port keeps only definitions resting on a metric view and never certifies a file
  import, so Entities, the crosswalk and the class hierarchy have no place in it, and widening
  it is a change to Metis's core.
- **Live tool calls at question time.** Metis would ask the engine while answering. Rejected
  for v1: it puts the engine on Metis's answer path, adding latency and a runtime dependency to
  every question, for content that changes on review, not per question.
- **An open interchange file as the interface.** Vendor-neutral, and Metis already imports one
  such format. Rejected as the interface: that format's ontology part is an unreleased draft,
  and the existing importer reports descriptions and labels as unsupported. It stays a possible
  additional export.
- **Sharing Metis's profiling and join logic as a library.** Less duplicated work. Rejected:
  it couples two release cycles and makes the standalone engine depend on Metis.
- **Letting both sides propose any measure.** Rejected: one measure could reach a reviewer
  twice, from two tools, with different evidence.

## Consequences

- The engine needs a writer for Metis's proposal shape. It lives in an optional adapter
  package, so the standalone engine carries no Metis-specific code.
- The Delta tables are a public contract: their names, keys and columns are versioned.
- The proposal shape belongs to Metis. If it changes, the adapter follows.
- v1's *one seam* check is met by the proposal file alone.
