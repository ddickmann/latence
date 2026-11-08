# Grilling: the boundary with Metis and the sidecar interface

Type: grilling
Mode: HITL
Status: open
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
