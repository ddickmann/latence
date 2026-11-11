# A host shows the engine's export in a read-only screen of its own, off the answer path

**Status:** Accepted (2026-11-11)

ADR-0070 gave Metis exactly one interface to the re-scoped engine: a reviewable proposal file.
Everything else reaches Metis through Unity Catalog tables. This record adds how a person
*sees* the engine's ontology and knowledge graph inside a host application, which ADR-0070 did
not cover. The owner decided each point on 2026-11-11 (map ticket *The Ontology page over the
estate*). The first cut is built as a prototype; nothing here is a shipped package.

## Context

The owner demonstrates TalkToData and TalkToDashboards on an estate in Unity Catalog and then
wants to open an Ontology entry in the same application and show the ontology and graph the
engine built over the same data. The host's rules shape what is possible:

- its user interface may only talk to its own backend, cannot embed another origin, and has
  no plugin mechanism;
- every dependency it takes must come from a package registry at a pinned version;
- code from other work is rewritten in its own style, never copied in.

The engine, in turn, must not import the host (ADR-0070), and its review workbench is a
prototype that is typed for one synthetic estate.

## Decision

1. **The host owns the screen.** The Ontology page is written inside the host's own user
   interface, in its components, tokens and languages. The host and the engine share one thing:
   the format of the export file (`graph-export/2`), which has a JSON Schema in the engine's
   repository. No code crosses in either direction.
2. **The screen is read-only and off the answer path.** It reads the engine's export and
   changes nothing in the catalog, in the host's semantic layer or in the host's service. No
   question a person asks the host is answered through it. ADR-0070's rule stands: what the
   host *consumes* still arrives as a proposal file or through Unity Catalog.
3. **Files first.** For the first cut the engine writes the export as a file the host serves
   from its own address. Such a file is served without sign-in, so it holds synthetic and
   public data only, no personal column is read, and a test enforces that no value of one
   reaches the export.
4. **Decisions travel back as a file.** A reviewer's approvals and rejections are kept in the
   browser and downloaded as a decisions file that the engine appends to its ledger on the
   next run. The engine keeps them across rebuilds (ADR-0069).
5. **Later, in this order:** a read-only route in the host's service over the engine's tables
   in Unity Catalog, under the asker's rights, replaces the file; and once the engine's review
   workbench is generic, it is published as a registry package the host may take instead of
   its own screen.

## Alternatives considered

- **The host mounts the engine's workbench component now.** One codebase, but the workbench is
  not on a registry, and a release file pinned by checksum fails the host's dependency gate as
  written. Loosening that gate for one file was offered and declined. Kept as the later step
  in point 5.
- **A route in the host's service from the start.** The right end state, and the only way to
  put the data behind sign-in. It touches the host's service, its API description and its
  tests before anything can be shown, so it comes second.
- **Embedding the engine's own page in a frame, or calling the engine from the browser.** The
  host's content policy forbids both, deliberately.
- **Leaving the view out of the host and showing the engine's standalone page.** That breaks
  the demonstration the owner asked for: one application, one sign-in, the same data.

## Consequences

- There are two screens for one export for a while: the host's page and the engine's
  standalone workbench. The export's JSON Schema is the contract between them.
- ADR-0070's Done check (exactly one declared interface the host consumes) is unchanged: the
  page displays, it does not feed the host.
- Until the service route exists, an estate with data that may not be public cannot be shown
  this way.
