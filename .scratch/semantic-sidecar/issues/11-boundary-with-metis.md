# Grilling: the boundary with Metis and the sidecar interface

Type: grilling
Mode: HITL
Status: resolved
Blocked by: 01, 10

## Question

What does the engine own, what does Metis own, and through which seam does Metis consume the
engine's output?

- **Overlap:** Metis's layer job already profiles, folds headers, decodes code lists,
  discovers keys and joins, and verifies them. Standalone, this engine must do the same. Does
  it share that logic as a library, re-implement it, or delegate to Metis when Metis is
  present?
- **What Metis consumes.** Candidates:
  - glossary and library proposals in Metis's `layer.glossary` shape (ticket 207's proposal
    file);
  - an `Importer` plugin, which today is limited to metric views and never certified;
  - MCP tools served by an isolated plugin host (Metis ticket 33);
  - Delta tables in UC that Metis reads like any other source.
- **What Metis deliberately leaves out** and this engine supplies: meaning-level folding,
  cross-language synonyms, name-level value equivalence, the text path, the ontology.
- **The UC source-of-truth rule:** how it binds outputs that Metis consumes, and what happens
  when the engine proposes something UC should own (an upstream pull request, like Metis's
  metric candidates).

Any change to `metis-core` needs the owner's explicit OK, as recorded on Metis ticket 211.

## Answer

Decided by the owner on 2026-11-09 as one package (the recommendation, adopted unchanged).

1. **One Metis-specific seam: a reviewable proposal file.** The engine writes glossary content
   (synonyms, usage notes, declared readings, each with its evidence) in the proposal shape
   Metis's proposal tooling defines (Metis ticket 207: one file per use case, diffed against its
   `library.yaml`). Metis's data team reviews it and Metis applies it. Nothing applies without
   review, and `metis-core` does not change.
2. **Everything else reaches Metis through Unity Catalog,** the same way it reaches Genie. The
   Entity crosswalk and the ontology are published as Delta tables keyed on UC object ids. Metis
   reads them like any other source. These tables are UC outputs, not a Metis interface.
3. **No shared code.** The engine never imports Metis. Standalone, it builds only the profiling
   that resolution needs (profiles, value sketches, declared keys, lineage). When Metis is
   present, the engine reads Metis's verified keys and joins as Assertions (Actor: the Metis
   layer job) instead of re-deriving them. Header folding, code-list decoding and join
   verification stay Metis's.
4. **New measures have one route per source.** Metis keeps proposing measures from questions and
   SQL usage. The engine proposes measures only from what Metis does not read: documents and the
   ontology (a Concept with no measure bound). With Metis present, the engine's drafts go into
   Metis's existing candidate route, so a reviewer never sees two proposals for one measure.
   Standalone, the engine writes the UC proposal itself. Either way a human applies it upstream.
5. **Not in v1:**
   - the `Importer` plugin: it only keeps definitions that rest on a metric view (`COMPILED`),
     never certifies a file import, and widening it changes `metis-core`;
   - live MCP calls at question time: they would put the engine on Metis's answer path;
   - Ossie as the seam: its ontology part is an unreleased draft, and Metis's Ossie importer
     reports descriptions and labels as unsupported. An Ossie export stays a possible extra later.

**What each side owns.** The engine: resolution (Entities and the crosswalk), the ontology
(Concepts, hierarchy, relations), the text path, cross-language synonyms, meaning-level folding
and the ledger of Assertions and Decisions. Metis: answering questions, compiling and serving,
its layer job's verified keys and joins, the review of its own library, and certification.

**The UC rule.** Everything Metis consumes is keyed on UC object ids. A change UC should own
goes upstream as a proposal a human applies; Metis sees it only once it is in UC.

This satisfies Done check 4 (exactly one declared interface to Metis). Recorded as
[ADR-0070](../../../docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md).

## From the map (moved 12 Nov)

[ADR-0070 — Metis consumes the engine through one proposal file](../../docs/adr/0070-metis-consumes-the-engine-through-one-proposal-file-and-otherwise-through-unity-catalog.md) and [Boundary with Metis](issues/11-boundary-with-metis.md) — one Metis-specific seam: a reviewable proposal file in Metis's own proposal shape. The crosswalk and the ontology reach Metis through UC Delta tables. No shared code; with Metis present the engine reads its verified keys and joins as Assertions. One route per source for new measures. No importer plugin, live tool calls or interchange-file seam in v1.
